"""
GeneticTestingEnv — Gym-style MDP environment for sequential family genetic testing.

Maps directly onto the paper's MDP:
  State   : frozenset of (person, observed_genotype) pairs (the evidence so far)
  Obs     : float32 vector of shape (N * G,) — carrier probability per individual
             per gene (genes in canonical order, individuals in topological order)
  Actions : Discrete(N + 1)
             0 … N-1  → test individual i
             N        → stop (terminate episode)
  Reward  :
    Test action j  → r_test(j, current_beliefs) = expected benefit of resolving
                      uncertainty for j, minus testing costs
    Stop action    → sum_i r_no_test(i, current_beliefs) for all untested i
    All individuals tested (auto-terminal) → 0 additional reward
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from ..models.pedigree import Pedigree
from ..config import Config, get_config
from ..models.genetics_cpd import (
    GENOTYPE_STATES,
    founder_prior_distribution,
    make_inheritance_genotype_cpd_with_table,
)
from ..models.reward import r_reward, r_reward_test
from .numpy_inference import NumpyInference

GEN_STATES = tuple(GENOTYPE_STATES)  # (0, 1, 2)


class GeneticTestingEnv:
    """Sequential genetic testing environment.

    Parameters
    ----------
    pedigree:
        Pedigree object (topological order matters — founders first).
    config:
        Config with reward coefficients a, b, c, delta and test costs.
    genes:
        Tuple of gene names. Single-gene = ('BRCA1',). Each gene has its own
        allele frequency via config.allele_freqs.
    seed:
        Random seed for reproducible test-result sampling.
    """

    def __init__(
        self,
        pedigree: Pedigree,
        config: Config,
        *,
        genes: Optional[Tuple[str, ...]] = None,
        seed: Optional[int] = None,
    ):
        self.pedigree = pedigree
        self.config = config
        self.genes: Tuple[str, ...] = genes if genes else ("gene",)
        self.individuals: List[str] = pedigree.to_list()
        self.N = len(self.individuals)
        self._idx: Dict[str, int] = {ind: i for i, ind in enumerate(self.individuals)}
        self.rng = np.random.default_rng(seed)

        # Build inference engine
        self._infer = NumpyInference(pedigree, self.genes, config)

        # Compute prior beliefs once (no evidence)
        _prior_result = self._infer.posterior({})
        self._prior_beliefs = {ind: dict(d) for ind, d in _prior_result.marginals.items()}
        self._prior_per_gene_beliefs = {
            gene: {ind: dict(_prior_result._per_gene_gene_first[gene][ind])
                   for ind in self.individuals}
            for gene in self.genes
        } if _prior_result._per_gene_gene_first else {
            gene: {ind: dict(self._prior_beliefs[ind]) for ind in self.individuals}
            for gene in self.genes
        }

        # Episode state (reset initialises these)
        self._evidence: Dict[str, int] = {}
        self._per_gene_evidence: Dict[str, Dict[str, int]] = {g: {} for g in self.genes}
        self._beliefs: Dict[str, Dict[int, float]] = {}
        self._per_gene_beliefs: Dict[str, Dict[str, Dict[int, float]]] = {}
        self._tested: Set[str] = set()
        self._done: bool = False
        self._cumulative_reward: float = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        """Total number of actions including stop."""
        return self.N + 1

    @property
    def obs_dim(self) -> int:
        """Length of the observation vector.

        Layout: [carrier_probs (N*G), tested_flags (N)]
        tested_flags[i] = 1.0 if individual i has been tested, else 0.0.
        """
        return self.N * len(self.genes) + self.N

    def reset(self) -> np.ndarray:
        """Reset to the empty-evidence state. Returns initial observation."""
        self._evidence = {}
        self._per_gene_evidence = {g: {} for g in self.genes}
        self._beliefs = {ind: dict(d) for ind, d in self._prior_beliefs.items()}
        self._per_gene_beliefs = {
            gene: {ind: dict(self._prior_per_gene_beliefs[gene][ind])
                   for ind in self.individuals}
            for gene in self.genes
        }
        self._tested = set()
        self._done = False
        self._cumulative_reward = 0.0
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one decision.

        Parameters
        ----------
        action:
            Integer in [0, N]. Values 0…N-1 test the corresponding individual;
            N stops the episode.

        Returns
        -------
        obs:       New observation vector.
        reward:    Immediate reward for this step.
        done:      True when the episode has ended.
        info:      Diagnostic dictionary.
        """
        assert not self._done, "Episode ended — call reset() first."

        # ---- STOP -------------------------------------------------------
        if action == self.N:
            reward = self._terminal_reward()
            self._done = True
            self._cumulative_reward += reward
            return self._observation(), float(reward), True, {
                "action": "stop",
                "cumulative_reward": self._cumulative_reward,
                "n_tested": len(self._tested),
            }

        # ---- TEST -------------------------------------------------------
        assert 0 <= action < self.N, f"Invalid action {action}"
        individual = self.individuals[action]

        if individual in self._tested:
            # Invalid: already tested — return zero reward with a flag
            return self._observation(), 0.0, False, {
                "action": "already_tested",
                "individual": individual,
            }

        # Compute expected testing reward (pre-observation)
        reward = self._test_reward(individual)

        # Sample observed genotype from current belief
        observed = self._sample_result(individual)

        # Update evidence and propagate posteriors
        self._tested.add(individual)
        if len(self.genes) == 1:
            self._evidence[individual] = observed
        else:
            # For multi-gene: evidence value is a tuple (one genotype per gene)
            # Here we observe the same genotype across genes; real data would
            # provide per-gene observations. Sampled from gene-averaged marginal.
            self._evidence[individual] = observed
            for gene in self.genes:
                self._per_gene_evidence[gene][individual] = observed

        self._propagate()
        self._cumulative_reward += reward

        done = len(self._tested) == self.N
        if done:
            self._done = True

        obs = self._observation()
        return obs, float(reward), done, {
            "action": "test",
            "individual": individual,
            "observed_genotype": int(observed),
            "carrier_prob_before": self._carrier_prob_from_beliefs(
                self._prior_beliefs, individual
            ),
            "carrier_prob_after": self._carrier_prob(individual),
            "cumulative_reward": self._cumulative_reward,
        }

    def action_mask(self) -> np.ndarray:
        """Boolean mask of valid actions (untested individuals + stop)."""
        mask = np.ones(self.N + 1, dtype=bool)
        for i, ind in enumerate(self.individuals):
            if ind in self._tested:
                mask[i] = False
        return mask

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        """Observation vector: [carrier_probs (N*G), tested_flags (N)]."""
        probs = []
        for gene in self.genes:
            gene_beliefs = self._per_gene_beliefs.get(gene, self._beliefs)
            for ind in self.individuals:
                dist = gene_beliefs.get(ind, self._beliefs.get(ind, {}))
                probs.append(dist.get(1, 0.0) + dist.get(2, 0.0))
        tested_flags = [1.0 if ind in self._tested else 0.0 for ind in self.individuals]
        return np.array(probs + tested_flags, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _test_reward(self, individual: str) -> float:
        """Expected reward of testing `individual` (pre-observation)."""
        if self.config.c_gene and self.config.a_gene:
            per_gene_p12 = {
                gene: (
                    self._per_gene_beliefs[gene][individual].get(1, 0.0)
                    + self._per_gene_beliefs[gene][individual].get(2, 0.0)
                )
                for gene in self.genes
            }
            from ..models.reward import r_reward_testp
            return r_reward_testp(
                individual,
                self._carrier_prob(individual),
                self.config.a,
                self.config.b,
                self.config.c,
                self.config.delta,
                self.config.fixed_cost,
                self.config.variable_cost,
                per_gene_p12=per_gene_p12,
                a_gene=self.config.a_gene or None,
                c_gene=self.config.c_gene or None,
                delta_gene=self.config.delta_gene or None,
            )
        return r_reward_test(
            individual,
            self._beliefs,
            self.config.a,
            self.config.b,
            self.config.c,
            self.config.delta,
            self.config.fixed_cost,
            self.config.variable_cost,
        )

    def _no_test_reward(self, individual: str) -> float:
        """No-test welfare for `individual` under current beliefs."""
        if self.config.c_gene and self.config.a_gene:
            per_gene_p12 = {
                gene: (
                    self._per_gene_beliefs[gene][individual].get(1, 0.0)
                    + self._per_gene_beliefs[gene][individual].get(2, 0.0)
                )
                for gene in self.genes
            }
            from ..models.reward import r_reward_p
            return r_reward_p(
                individual,
                self._carrier_prob(individual),
                self.config.a,
                self.config.b,
                self.config.c,
                self.config.delta,
                per_gene_p12=per_gene_p12,
                a_gene=self.config.a_gene or None,
                b_gene=self.config.b_gene or None,
                c_gene=self.config.c_gene or None,
                delta_gene=self.config.delta_gene or None,
            )
        return r_reward(
            individual,
            self._beliefs,
            self.config.a,
            self.config.b,
            self.config.c,
            self.config.delta,
        )

    def _terminal_reward(self) -> float:
        """Sum of no-test rewards for all untested individuals."""
        return sum(
            self._no_test_reward(ind)
            for ind in self.individuals
            if ind not in self._tested
        )

    # ------------------------------------------------------------------
    # Belief helpers
    # ------------------------------------------------------------------

    def _carrier_prob(self, individual: str) -> float:
        dist = self._beliefs.get(individual, {})
        return dist.get(1, 0.0) + dist.get(2, 0.0)

    @staticmethod
    def _carrier_prob_from_beliefs(
        beliefs: Dict[str, Dict[int, float]], individual: str
    ) -> float:
        dist = beliefs.get(individual, {})
        return dist.get(1, 0.0) + dist.get(2, 0.0)

    def _sample_result(self, individual: str) -> int:
        """Sample observed genotype from current marginal belief."""
        dist = self._beliefs[individual]
        states = list(GEN_STATES)
        probs = np.array([dist.get(g, 0.0) for g in states], dtype=np.float64)
        probs /= probs.sum()
        return int(self.rng.choice(states, p=probs))

    def _propagate(self) -> None:
        """Recompute posterior beliefs given current evidence."""
        result = self._infer.posterior(self._evidence)
        self._beliefs = {ind: dict(d) for ind, d in result.marginals.items()}
        if result._per_gene_gene_first:
            for gene in self.genes:
                gene_marg = result._per_gene_gene_first.get(gene, {})
                self._per_gene_beliefs[gene] = {
                    ind: dict(gene_marg.get(ind, self._beliefs[ind]))
                    for ind in self.individuals
                }
        else:
            for gene in self.genes:
                self._per_gene_beliefs[gene] = {
                    ind: dict(self._beliefs[ind]) for ind in self.individuals
                }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def current_evidence(self) -> Dict[str, int]:
        return dict(self._evidence)

    @property
    def current_beliefs(self) -> Dict[str, Dict[int, float]]:
        return {ind: dict(d) for ind, d in self._beliefs.items()}

    @property
    def tested(self) -> Set[str]:
        return set(self._tested)

    @property
    def untested(self) -> List[str]:
        return [ind for ind in self.individuals if ind not in self._tested]

    @property
    def done(self) -> bool:
        return self._done

    def carrier_probs(self) -> Dict[str, float]:
        """Current carrier probability per individual."""
        return {ind: self._carrier_prob(ind) for ind in self.individuals}

    def __repr__(self) -> str:
        tested_str = ", ".join(f"{p}={g}" for p, g in self._evidence.items()) or "∅"
        return (
            f"GeneticTestingEnv(N={self.N}, genes={self.genes}, "
            f"tested=[{tested_str}], done={self._done})"
        )
