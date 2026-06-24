"""
Exact DP via backward induction (no LP, no Gurobi).

Uses memoized recursion over evidence frozensets. State space ≤ 4^N.
Practical for N ≤ 9 with single gene (262 K states, < 10 s).

Bellman equation:
  V*(s) = max(
    stop:   Σ_{i ∉ tested} r_no_test(i, beliefs(s)),
    test j: r_test(j, beliefs(s)) + Σ_g P(G_j=g | s) · V*(s ∪ {(j,g)})
  )
"""

from __future__ import annotations

import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..models.pedigree import Pedigree
from ..config import Config
from ..models.reward import r_reward, r_reward_test
from ..envs.numpy_inference import NumpyInference

GEN_STATES = (0, 1, 2)


class BackwardInductionSolver:
    """Exact DP solver for sequential genetic testing.

    Parameters
    ----------
    pedigree : Pedigree object (topological order, founders first).
    config   : Reward coefficients + test costs.
    genes    : Gene names (single-gene default).
    verbose  : Print progress every N states (0 = silent).
    """

    def __init__(
        self,
        pedigree: Pedigree,
        config: Config,
        genes: Tuple[str, ...] = ("gene",),
        verbose: int = 0,
    ):
        self.pedigree = pedigree
        self.config = config
        self.genes = genes
        self.individuals: List[str] = pedigree.to_list()
        self.N = len(self.individuals)
        self.verbose = verbose

        self._infer = NumpyInference(pedigree, genes, config)
        self._V: Dict[FrozenSet, float] = {}
        self._policy: Dict[FrozenSet, Tuple[str, Optional[str]]] = {}
        self._n_states: int = 0
        self.solve_time: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self) -> float:
        """Run full backward induction. Returns V*(empty state)."""
        t0 = time.time()
        self._V.clear()
        self._policy.clear()
        self._n_states = 0
        v0 = self._v(frozenset())
        self.solve_time = time.time() - t0
        if self.verbose:
            print(
                f"Exact DP done: N={self.N}, states={self._n_states}, "
                f"time={self.solve_time:.2f}s, V*(∅)={v0:.6f}"
            )
        return v0

    @property
    def V0(self) -> float:
        return self._V[frozenset()]

    @property
    def n_states(self) -> int:
        return self._n_states

    def optimal_action(self, state: FrozenSet = None) -> Tuple[str, Optional[str]]:
        """Return ('stop', None) or ('test', individual_name). Call solve() first."""
        return self._policy[state if state is not None else frozenset()]

    def rollout(self, seed: int = 0) -> Tuple[float, List[dict]]:
        """One episode following the optimal policy (belief-MDP rollout)."""
        import numpy as np
        rng = np.random.default_rng(seed)
        state: FrozenSet = frozenset()
        evidence: Dict[str, int] = {}
        total = 0.0
        traj: List[dict] = []

        while True:
            kind, target = self._policy[state]
            beliefs = self._get_beliefs(state)
            if kind == "stop":
                r = sum(
                    self._no_test_r(ind, beliefs)
                    for ind in self.individuals
                    if ind not in evidence
                )
                total += r
                traj.append({"action": "stop", "reward": r})
                break
            r = self._test_r(target, beliefs)
            total += r
            dist = beliefs[target]
            probs = np.array([dist.get(g, 0.0) for g in GEN_STATES])
            probs /= probs.sum()
            obs = int(rng.choice(GEN_STATES, p=probs))
            traj.append({"action": "test", "individual": target, "observed": obs, "reward": r})
            evidence[target] = obs
            state = frozenset(evidence.items())

        return total, traj

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _v(self, state: FrozenSet) -> float:
        if state in self._V:
            return self._V[state]

        evidence = dict(state)
        tested = set(evidence.keys())
        untested = [ind for ind in self.individuals if ind not in tested]

        if not untested:
            self._V[state] = 0.0
            self._policy[state] = ("stop", None)
            self._n_states += 1
            return 0.0

        beliefs = self._get_beliefs(state)

        # Stop
        v_stop = sum(self._no_test_r(ind, beliefs) for ind in untested)

        # Best test
        best_v, best_ind = -1e18, None
        for ind in untested:
            r_t = self._test_r(ind, beliefs)
            ev_succ = sum(
                beliefs[ind].get(g, 0.0) * self._v(state | frozenset([(ind, g)]))
                for g in GEN_STATES
            )
            v_t = r_t + ev_succ
            if v_t > best_v:
                best_v, best_ind = v_t, ind

        if v_stop >= best_v:
            self._V[state] = v_stop
            self._policy[state] = ("stop", None)
        else:
            self._V[state] = best_v
            self._policy[state] = ("test", best_ind)

        self._n_states += 1
        if self.verbose and self._n_states % self.verbose == 0:
            print(f"  {self._n_states} states solved...")
        return self._V[state]

    def _get_beliefs(self, state: FrozenSet) -> Dict[str, Dict[int, float]]:
        result = self._infer.posterior(dict(state))
        return {ind: dict(d) for ind, d in result.marginals.items()}

    def _test_r(self, ind: str, beliefs: Dict[str, Dict[int, float]]) -> float:
        return r_reward_test(
            ind, beliefs,
            self.config.a, self.config.b, self.config.c, self.config.delta,
            self.config.fixed_cost, self.config.variable_cost,
        )

    def _no_test_r(self, ind: str, beliefs: Dict[str, Dict[int, float]]) -> float:
        return r_reward(
            ind, beliefs,
            self.config.a, self.config.b, self.config.c, self.config.delta,
        )
