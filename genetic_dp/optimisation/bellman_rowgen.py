"""
Bellman-consistent row generation implementation.

This module implements the Bellman-consistent approach described in BELLMAN_CONSISTENT_ROWGEN.md:
- Uses exact BN posteriors p_s[i,g] instead of MILP-optimized P_out
- Computes violations without MILP subproblems
- Supports multi-cut addition per iteration
- Includes state frontier management and posterior caching
"""

from typing import Dict, List, Tuple, FrozenSet, Optional, Hashable
import logging
from ..models.belief import (
    propagate_all_marginals,
    lift_single_gene_posteriors_to_genes,
    propagate_multigene_marginals,
)
from ..models.outcomes import (
    project_state_by_gene,
    project_successor_by_gene,
)
from ..models.reward import r_reward_testp
from ..optimisation.utils import (
    canonicalize_state_prob,
    build_canonical_evidence,
    remap_posteriors_from_canonical,
)

def select_successors(p_post_i: Dict[Hashable, float],
                      K: int | None = None,
                      pmin: float = 1e-9) -> List[Tuple[Hashable, float]]:
    """
    Return a pruned list of (g, p) pairs sorted by p desc, keeping at most K
    and dropping tiny probs (< pmin). Dropping successors shrinks the RHS, so
    the Bellman cut stays valid (conservative) but weaker.
    """
    items = sorted(p_post_i.items(), key=lambda kv: kv[1], reverse=True)
    kept = [(g, p) for g, p in items if p >= pmin]
    if K:
        kept = kept[:K]
    return kept


def select_tuple_successors(p_post_tuple: Dict[Tuple[int, ...], float],
                            K: int | None = None,
                            pmin: float = 1e-9) -> List[Tuple[Tuple[int, ...], float]]:
    items = sorted(p_post_tuple.items(), key=lambda kv: kv[1], reverse=True)
    kept = [(tuple(g), p) for g, p in items if p >= pmin]
    if K:
        kept = kept[:K]
    return kept

def bellman_violation(phi_S: float,
                      phi_succ: Dict[Hashable, float],
                      probs: Dict[Hashable, float],
                      r_immediate: float) -> Tuple[float, float]:
    """
    Compute violation and RHS for the canonical inequality:
        Φ(S) ≥ r_i(S) + Σ_g p(g) Φ(S∪{(i,g)})
    Returns (violation, rhs_value) where violation = RHS - LHS.
    Positive violation means the cut is violated.
    """
    rhs = r_immediate + sum(probs[g] * phi_succ[g] for g in probs)
    return rhs - phi_S, rhs



class BellmanRowGenerator:
    """Bellman-consistent row generation with exact posteriors."""
    
    def __init__(self, I: List, gen_states: List, infer, pedigree,
                 tolerance: float = 1e-12, verbose: bool = False,
                 role_groups: Optional[Dict]=None,
                 genes: Optional[List[str]] = None,
                 tuple_mode: bool = False,
                 outcomes: Optional[Tuple[Hashable, ...]] = None):
        """
        Initialize Bellman row generator.
        
        Args:
            I: List of all individuals
            gen_states: List of genotypes {0, 1, 2}
            infer: pgmpy inference object
            pedigree: Pedigree structure
            tolerance: Minimum violation threshold
            verbose: Enable diagnostic logging
            role_groups: Optional role groups for probability-only canonicalization
        """
        self.I = I
        self.gen_states = gen_states
        self.infer = infer
        self.pedigree = pedigree
        self.tolerance = tolerance
        self.verbose = verbose
        self.genes = tuple(genes) if genes else None
        self.tuple_mode = tuple_mode
        self.outcomes = tuple(outcomes) if outcomes is not None else tuple(gen_states)
        self.gene_order = tuple(genes) if genes else None
        
        # Probability-only posterior cache: prob_key -> p_s(canonical)
        self.posterior_cache_prob = {}
        self.role_groups = role_groups
        # --- instrumentation (used by tests; negligible overhead) ---
        self.cache_hits = 0
        self.cache_misses = 0
        self.bn_time = 0.0
        
        # Violation tracking for frontier expansion
        self.last_violations = {}
        # Projection caches for per-gene Φ_j lookup
        self._state_projection_cache: Dict[FrozenSet, Dict[str, FrozenSet[Tuple[str, int]]]] = {}
        self._successor_projection_cache: Dict[
            Tuple[FrozenSet, Hashable, Hashable],
            Tuple[FrozenSet, Dict[str, FrozenSet[Tuple[str, int]]]]
        ] = {}

    def _normalize_outcome(self, outcome: Hashable) -> Hashable:
        if self.tuple_mode and self.genes:
            if isinstance(outcome, tuple):
                if len(outcome) != len(self.genes):
                    raise AssertionError(f"Outcome {outcome!r} length mismatch for genes {self.genes}")
                return tuple(outcome)
            if outcome in self.gen_states:
                return tuple(outcome for _ in self.genes)
            raise AssertionError(f"Invalid scalar outcome {outcome!r} for tuple mode")
        if not self.tuple_mode:
            if isinstance(outcome, tuple):
                if len(outcome) == 1 and outcome[0] in self.gen_states:
                    return outcome[0]
                raise AssertionError(f"Tuple outcome {outcome!r} unexpected in scalar mode")
            if outcome not in self.gen_states:
                raise AssertionError(f"Invalid outcome {outcome!r}")
        return outcome

    def _build_successor_state(self, state: FrozenSet, person: Hashable, outcome: Hashable) -> FrozenSet:
        successor = dict(state)
        successor[person] = self._normalize_outcome(outcome)
        return frozenset(successor.items())

    def _project_state(self, state: FrozenSet) -> Optional[Dict[str, FrozenSet[Tuple[str, int]]]]:
        if not self.gene_order:
            return None
        cached = self._state_projection_cache.get(state)
        if cached is not None:
            return cached
        projected = project_state_by_gene(state, self.gene_order)
        self._state_projection_cache[state] = projected
        return projected

    def get_state_projection(self, state: FrozenSet) -> Optional[Dict[str, FrozenSet[Tuple[str, int]]]]:
        """
        Public accessor for per-gene projections of a state; computes and caches on demand.
        """
        return self._project_state(state)

    def _project_successor_state(
        self,
        state: FrozenSet,
        person: Hashable,
        outcome: Hashable,
    ) -> Tuple[FrozenSet, Optional[Dict[str, FrozenSet[Tuple[str, int]]]]]:
        normalized_outcome = self._normalize_outcome(outcome)
        succ_state = self._build_successor_state(state, person, normalized_outcome)
        if not self.gene_order:
            return succ_state, None
        cache_key = (state, person, normalized_outcome)
        cached = self._successor_projection_cache.get(cache_key)
        if cached is not None:
            return cached
        projection = project_successor_by_gene(state, person, normalized_outcome, self.gene_order)
        self._successor_projection_cache[cache_key] = (succ_state, projection)
        return succ_state, projection

    def get_successor_projection(
        self,
        state: FrozenSet,
        person: Hashable,
        outcome: Hashable,
    ) -> Tuple[FrozenSet, Optional[Dict[str, FrozenSet[Tuple[str, int]]]]]:
        """
        Public accessor that normalizes outcome, builds successor state, and returns per-gene projection.
        """
        return self._project_successor_state(state, person, outcome)
        
    def get_cached_posterior(self, state: FrozenSet) -> Tuple[Dict, Dict, Optional[Dict], Dict[str, Dict[Tuple[int, ...], float]]]:
        """Reuse symmetry for probabilities only (counts-based cache)."""
        import time
        t0 = time.perf_counter()
        propagate_fn = (
            (lambda evid: propagate_multigene_marginals(self.infer, self.I, self.gen_states, evid, self.genes))
            if self.genes else
            (lambda evid: propagate_all_marginals(self.infer, self.I, self.gen_states, evid))
        )

        tuple_pmfs: Dict[str, Dict[Tuple[int, ...], float]] = {}

        if not self.role_groups:
            # Fall back to identity evidence if no roles provided
            evidence = {i: g for (i, g) in state}
            result = propagate_fn(evidence)
            if hasattr(result, 'marginals'):
                p_s = result.marginals
                if self.tuple_mode:
                    tuple_pmfs = result.get_tuple_pmfs()
            else:
                p_s = result
            self.cache_misses += 1
        else:
            prob_key = canonicalize_state_prob(state, self.role_groups, self.gen_states)
            if prob_key not in self.posterior_cache_prob:
                evidence = build_canonical_evidence(state, self.role_groups, self.gen_states)
                result_canon = propagate_fn(evidence)
                p_s_canon = result_canon.marginals if hasattr(result_canon, 'marginals') else result_canon
                self.posterior_cache_prob[prob_key] = p_s_canon
                self.cache_misses += 1
            else:
                self.cache_hits += 1
            p_s_canon = self.posterior_cache_prob[prob_key]
            p_s = remap_posteriors_from_canonical(p_s_canon, state, self.role_groups, self.gen_states)
            tuple_pmfs = {}
        self.bn_time += (time.perf_counter() - t0)

        # one-hot z_s from state
        state_dict = {person: self._normalize_outcome(outcome) for person, outcome in state}
        z_s = {
            person: {outcome: (1.0 if state_dict.get(person) == outcome else 0.0) for outcome in self.outcomes}
            for person in self.I
        }
        gene_probs = lift_single_gene_posteriors_to_genes(p_s, self.genes) if self.genes else None
        return p_s, z_s, gene_probs, tuple_pmfs
    
    def evaluate_bellman_violations(
            self, 
            state: FrozenSet,
            current_phi: float,
            phi_values: Dict[FrozenSet, float],
            a: Dict, b: Dict, c: Dict, delta: Dict,
            fixed_cost: float, variable_cost: float,
            create_successors: bool = True,
            a_gene: Optional[Dict[str, Dict]] = None,
            c_gene: Optional[Dict[str, Dict]] = None,
            delta_gene: Optional[Dict[str, Dict]] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Compute Bellman violations for all testable individuals at given state.
        
        Args:
            state: Current state as frozenset of (individual, genotype) pairs
            current_phi: Current Φ(state) value
            phi_values: Dictionary of Φ(s') values for successor states
            a, b, c, delta: Reward parameters
            fixed_cost, variable_cost: Testing costs
            create_successors: Whether to create missing successor states
            
        Returns:
            List of (individual, violation, successor_info) tuples
        """
        # Get cached posterior for this state
        p_s, z_s, gene_probs, tuple_pmfs = self.get_cached_posterior(state)
        state_projection = self._project_state(state)
        
        tested_individuals = {i for (i, g) in state}
        violations = []
        
        if self.verbose:
            logging.info(f"[BELLMAN] Evaluating state with tested: {tested_individuals}")
        
        # For each testable individual i ∉ state
        for i in self.I:
            if i in tested_individuals:
                continue

            # Skip deterministic posteriors (testing yields no new information)
            scalar_max = max(p_s[i].values())
            tuple_max = max(tuple_pmfs.get(i, {}).values()) if tuple_pmfs.get(i) else 0.0
            effective_max = max(scalar_max, tuple_max)
            if self.verbose or os.getenv("BELL_LOG_DETERMINISTIC"):
                logging.info(
                    "[BELLMAN] posterior stats for %s — scalar_max=%.6f tuple_max=%.6f",
                    i,
                    scalar_max,
                    tuple_max,
                )
            if effective_max >= 1.0 - self.tolerance:
                if self.verbose or os.getenv("BELL_LOG_DETERMINISTIC"):
                    logging.info(
                        "[BELLMAN] Skipping %s (deterministic posterior >= %.3f)",
                        i,
                        effective_max,
                    )
                continue

            # Compute immediate reward r(s,i)
            p12 = p_s[i][1] + p_s[i][2]
            per_gene_p12 = None
            if gene_probs:
                per_gene_p12 = {}
                for gene, probs in gene_probs.items():
                    if i not in probs:
                        continue
                    per_gene_p12[gene] = probs[i].get(1, 0.0) + probs[i].get(2, 0.0)
                if not per_gene_p12:
                    per_gene_p12 = None
            r_i = r_reward_testp(
                i,
                p12,
                a,
                b,
                c,
                delta,
                fixed_cost,
                variable_cost,
                per_gene_p12=per_gene_p12,
                a_gene=a_gene,
                c_gene=c_gene,
                delta_gene=delta_gene,
            )
            
            # Compute expected successor value: Σ_g p_s[i,g] · Φ(s')
            expected_successor_value = 0.0
            successor_info = {}
            missing_successors = []

            if self.tuple_mode and tuple_pmfs.get(i):
                successor_candidates = select_tuple_successors(tuple_pmfs[i], K=K, pmin=self.tolerance)
                for outcome, prob in successor_candidates:
                    if prob < self.tolerance:
                        continue
                    normalized_outcome = self._normalize_outcome(outcome)
                    successor_state, projection = self._project_successor_state(state, i, normalized_outcome)
                    phi_successor = phi_values.get(successor_state, 0.0)
                    expected_successor_value += prob * phi_successor
                    successor_info[normalized_outcome] = {
                        'probability': prob,
                        'successor_state': successor_state,
                        'phi_value': phi_successor,
                        'per_gene_successor': projection,
                    }
                    if successor_state not in phi_values:
                        missing_successors.append((successor_state, prob))
            else:
                for g in self.gen_states:
                    prob_g = p_s[i][g]
                    if prob_g < self.tolerance:
                        continue

                    successor_state, projection = self._project_successor_state(state, i, g)

                    if successor_state in phi_values:
                        phi_successor = phi_values[successor_state]
                        expected_successor_value += prob_g * phi_successor
                        successor_info[self._normalize_outcome(g)] = {
                            'probability': prob_g,
                            'successor_state': successor_state,
                            'phi_value': phi_successor,
                            'per_gene_successor': projection,
                        }
                    else:
                        phi_successor = 0.0
                        expected_successor_value += prob_g * phi_successor
                        successor_info[self._normalize_outcome(g)] = {
                            'probability': prob_g,
                            'successor_state': successor_state,
                            'phi_value': phi_successor,
                            'per_gene_successor': projection,
                        }
                        missing_successors.append((successor_state, prob_g))
            
            # Compute Bellman violation: RHS_i - Φ(s)
            # RHS_i = r(s,i) + Σ_g p_s[i,g] · Φ(s')
            rhs_i = r_i + expected_successor_value
            violation = rhs_i - current_phi
            
            if violation > self.tolerance:
                violations.append((i, violation, {
                    'immediate_reward': r_i,
                    'expected_successor': expected_successor_value,
                    'rhs': rhs_i,
                    'current_phi': current_phi,
                    'successors': successor_info,
                    'state_projection': state_projection,
                    'prob_sum': (
                        sum(tuple_pmfs[i].values()) if (self.tuple_mode and tuple_pmfs.get(i))
                        else sum(p_s[i][g] for g in self.gen_states)
                    )
                }))
                
                if self.verbose:
                    logging.info(f"[BELLMAN] Individual {i}: violation={violation:.6f}, "
                               f"r_i={r_i:.6f}, E[Φ(s')]={expected_successor_value:.6f}")
        
        # Sort by violation magnitude (highest first)
        violations.sort(key=lambda x: x[1], reverse=True)
        
        # Store for frontier expansion
        self.last_violations[state] = violations
        
        if self.verbose:
            logging.info(f"[BELLMAN] Found {len(violations)} violations > {self.tolerance}")
            
        return violations
    
    def build_state_frontier(
            self, 
            current_phi_states: List[FrozenSet],
            top_k: int = 5
    ) -> List[FrozenSet]:
        """
        Build frontier of states to evaluate in next iteration.
        
        Args:
            current_phi_states: Currently known states with Φ variables
            top_k: Number of top violating states to expand
            
        Returns:
            List of states to evaluate
        """
        frontier = set()
        
        # Always include root
        root = frozenset()
        frontier.add(root)
        
        # Always include all single-test states
        for i in self.I:
            for outcome in self.outcomes:
                single_state = frozenset({(i, self._normalize_outcome(outcome))})
                frontier.add(single_state)
        
        # Add expansion from high-violation states
        if self.last_violations:
            # Sort states by their maximum violation
            state_max_violations = []
            for state, violations in self.last_violations.items():
                if violations:
                    max_violation = max(v[1] for v in violations)
                    state_max_violations.append((state, max_violation))
            
            # Take top K states with highest violations
            state_max_violations.sort(key=lambda x: x[1], reverse=True)
            for state, _ in state_max_violations[:top_k]:
                # Add single-step expansions of this state
                tested_individuals = {i for (i, g) in state}
                for i in self.I:
                    if i not in tested_individuals:
                        for outcome in self.outcomes:
                            expanded_state = self._build_successor_state(state, i, outcome)
                            frontier.add(expanded_state)
        
        return list(frontier)
    
    def get_required_successors(
            self,
            state: FrozenSet,
            violations: List[Tuple[str, float, Dict]]
    ) -> List[FrozenSet]:
        """Get list of successor states that need to be created for violations."""
        successors = set()
        
        # Get posterior for this state
        p_s, _, _, tuple_pmfs = self.get_cached_posterior(state)
        
        for i, violation, info in violations:
            if self.tuple_mode and tuple_pmfs.get(i):
                for outcome, prob in tuple_pmfs[i].items():
                    if prob < self.tolerance:
                        continue
                    successor_state = self._build_successor_state(state, i, outcome)
                    successors.add(successor_state)
            else:
                for g in self.gen_states:
                    prob_g = p_s[i][g]
                    if prob_g < self.tolerance:
                        continue
                    successor_state = self._build_successor_state(state, i, g)
                    successors.add(successor_state)
        
        return list(successors)
    
    def clear_cache(self):
        """Clear posterior cache to free memory."""
        self.posterior_cache.clear()
        self.last_violations.clear()
        self._state_projection_cache.clear()
        self._successor_projection_cache.clear()


def log_bellman_cut(state: FrozenSet, individual: str, violation: float, 
                   info: Dict, iteration: int):
    """
    Log minimal diagnostics for Bellman cut as specified in plan.
    
    Args:
        state: Current state
        individual: Chosen individual i*
        violation: Violation amount
        info: Successor info dictionary
        iteration: Current iteration number
    """
    tested_inds = {i for (i, g) in state}
    prob_sum = info.get('prob_sum', 1.0)
    rhs = info.get('rhs', 0.0)
    
    logging.info(f"[BELLMAN_CUT] iter={iteration} state={len(tested_inds)} "
               f"i*={individual} prob_sum={prob_sum:.6f} RHS={rhs:.6f} "
               f"violation={violation:.6f}")
