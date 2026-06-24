"""
Vectorized exact inference on a pedigree using numpy enumeration.

No pgmpy required. Works for any polytree pedigree by enumerating all 3^N
joint genotype configurations. Practical for N ≤ 13 per gene (~1.6M configs).

For multi-gene models the joint distribution factorizes across genes (loci are
independent), so inference is run independently per gene and results are merged.
"""

from __future__ import annotations

import itertools
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..models.genetics_cpd import (
    GENOTYPE_STATES,
    founder_prior_distribution,
    make_inheritance_genotype_cpd_with_table,
)
from ..models.belief import InferenceResult
from ..models.pedigree import Pedigree
from ..config import Config

GEN_STATES = tuple(GENOTYPE_STATES)  # (0, 1, 2)


class NumpyInference:
    """Exact posterior inference via vectorized joint enumeration.

    Builds a (3^N,) log-joint array at init time. Each posterior query
    masks out configs inconsistent with observed evidence and marginalizes.

    For multi-gene models (len(genes) > 1), each gene's network is built
    independently and inference results are merged into a single InferenceResult.
    """

    def __init__(self, pedigree: Pedigree, genes: Sequence[str], config: Config):
        self.pedigree = pedigree
        self.genes = tuple(genes)
        self.config = config
        self.individuals = pedigree.to_list()
        self.N = len(self.individuals)
        self._idx: Dict[str, int] = {ind: i for i, ind in enumerate(self.individuals)}

        # Precompute all 3^N index tuples — shape (3^N, N)
        # For N>=13, list(itertools.product) creates a ~3 GB Python list; use
        # a direct numpy construction instead (stays under 300 MB for N=15).
        n_configs = 3 ** self.N
        if self.N <= 12:
            self._configs = np.array(
                list(itertools.product(range(3), repeat=self.N)), dtype=np.int8
            )
        else:
            self._configs = np.empty((n_configs, self.N), dtype=np.int8)
            for col in range(self.N):
                repeat_each = 3 ** (self.N - 1 - col)
                n_tiles = 3 ** col
                self._configs[:, col] = np.tile(
                    np.repeat(np.arange(3, dtype=np.int8), repeat_each), n_tiles
                )

        # Per-gene log-joint arrays — shape (3^N,) each
        self._log_joint_per_gene: Dict[str, np.ndarray] = {
            gene: self._build_log_joint(gene) for gene in self.genes
        }

        # Cache: frozenset(evidence.items()) → InferenceResult
        # Capped at 10k entries — avoids OOM when exact DP explores 4^N states.
        # RL training visits far fewer unique states so this is always a full hit.
        self._posterior_cache: Dict[frozenset, "InferenceResult"] = {}
        self._cache_max = 10_000

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_log_joint(self, gene: str) -> np.ndarray:
        """Compute log P(g_1,...,g_N) for all 3^N configurations for one gene."""
        freq = self.config.get_allele_freq(gene)

        founder_log_priors: Dict[str, np.ndarray] = {}
        child_tables: Dict[str, Tuple[np.ndarray, int, int]] = {}

        for ind in self.individuals:
            parents = self.pedigree.get_parents(ind)
            if not parents:
                prior = founder_prior_distribution(freq)
                founder_log_priors[ind] = np.log(
                    np.array([prior[g] for g in range(3)], dtype=np.float64) + 1e-300
                )
            else:
                p1, p2 = parents[0], parents[1]
                _, table = make_inheritance_genotype_cpd_with_table(ind, p1, p2)
                child_tables[ind] = (table, self._idx[p1], self._idx[p2])

        configs = self._configs  # (3^N, N)
        log_joint = np.zeros(len(configs), dtype=np.float64)

        for ind in self.individuals:
            i = self._idx[ind]
            g_ind = configs[:, i].astype(np.int64)  # (3^N,)
            parents = self.pedigree.get_parents(ind)
            if not parents:
                log_joint += founder_log_priors[ind][g_ind]
            else:
                table, p1_idx, p2_idx = child_tables[ind]
                g_p1 = configs[:, p1_idx].astype(np.int64)
                g_p2 = configs[:, p2_idx].astype(np.int64)
                col = g_p1 * 3 + g_p2
                prob = table[g_ind, col]
                log_joint += np.log(prob + 1e-300)

        return log_joint

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def posterior(self, evidence: Mapping) -> "InferenceResult":
        """Return posterior marginals for all individuals given evidence.

        Results are cached by evidence set — repeated calls with the same
        evidence (common during RL rollouts) are O(1) after the first call.

        Parameters
        ----------
        evidence:
            Dict mapping person name → observed genotype (int 0/1/2 for
            single-gene, or tuple of ints for multi-gene).
        """
        cache_key = frozenset(evidence.items())
        if cache_key in self._posterior_cache:
            return self._posterior_cache[cache_key]
        if len(self.genes) == 1:
            result = self._posterior_single_gene(evidence, self.genes[0])
        else:
            result = self._posterior_multi_gene(evidence)
        if len(self._posterior_cache) < self._cache_max:
            self._posterior_cache[cache_key] = result
        return result

    def _scalar_evidence(
        self, evidence: Mapping, gene_idx: int
    ) -> Dict[str, int]:
        """Extract scalar genotype evidence for a single gene."""
        scalar: Dict[str, int] = {}
        for person, val in evidence.items():
            if isinstance(val, (tuple, list)):
                scalar[person] = int(val[gene_idx])
            else:
                scalar[person] = int(val)
        return scalar

    def _posterior_single_gene(
        self, evidence: Mapping, gene: str
    ) -> InferenceResult:
        gene_idx = self.genes.index(gene)
        scalar_ev = self._scalar_evidence(evidence, gene_idx)
        marginals = self._compute_marginals(
            self._log_joint_per_gene[gene], scalar_ev
        )
        return InferenceResult(
            marginals=marginals,
            per_gene={gene: {p: dict(d) for p, d in marginals.items()}},
            gene_order=(gene,),
            gen_states=GEN_STATES,
        )

    def _posterior_multi_gene(self, evidence: Mapping) -> InferenceResult:
        per_gene_marginals: Dict[str, Dict[str, Dict[int, float]]] = {}
        for gene_idx, gene in enumerate(self.genes):
            scalar_ev = self._scalar_evidence(evidence, gene_idx)
            per_gene_marginals[gene] = self._compute_marginals(
                self._log_joint_per_gene[gene], scalar_ev
            )

        # Aggregate marginals: average carrier prob across genes
        # (caller can use per_gene for gene-specific views)
        agg: Dict[str, Dict[int, float]] = {}
        for ind in self.individuals:
            agg[ind] = {g: 0.0 for g in GEN_STATES}
            for gene in self.genes:
                for g in GEN_STATES:
                    agg[ind][g] += per_gene_marginals[gene][ind][g]
            n = len(self.genes)
            agg[ind] = {g: v / n for g, v in agg[ind].items()}

        return InferenceResult(
            marginals=agg,
            per_gene=per_gene_marginals,
            gene_order=tuple(self.genes),
            gen_states=GEN_STATES,
        )

    def _compute_marginals(
        self,
        log_joint: np.ndarray,
        scalar_ev: Dict[str, int],
    ) -> Dict[str, Dict[int, float]]:
        """Condition on scalar evidence and return per-individual marginals."""
        configs = self._configs  # (3^N, N)
        mask = np.ones(len(configs), dtype=bool)

        for person, g_obs in scalar_ev.items():
            idx = self._idx[person]
            mask &= configs[:, idx] == g_obs

        if not mask.any():
            raise RuntimeError(
                f"Evidence {scalar_ev!r} has zero probability under the model."
            )

        lj = log_joint[mask]
        lj = lj - lj.max()
        joint = np.exp(lj)
        joint /= joint.sum()

        masked_configs = configs[mask]  # (K, N)

        marginals: Dict[str, Dict[int, float]] = {}
        for i, ind in enumerate(self.individuals):
            col = masked_configs[:, i]
            marginals[ind] = {
                g: float(joint[col == g].sum()) for g in GEN_STATES
            }
        return marginals

    # ------------------------------------------------------------------
    # Prior marginals (no evidence)
    # ------------------------------------------------------------------

    def prior_marginals(self) -> Dict[str, Dict[int, float]]:
        """Compute prior carrier probability for each individual (no evidence)."""
        return self._compute_marginals(
            sum(self._log_joint_per_gene[g] for g in self.genes) / len(self.genes),
            {},
        )
