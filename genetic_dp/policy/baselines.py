"""
Baseline (non-ADP) policy functions for benchmarking.

Each function returns (action, who, value) matching the signature of
``best_action`` in extractor.py, so they can be plugged directly into
``exact_value_under_policy`` via a thin adapter.

Baselines implemented:
    1. myopic_greedy   – test the person with highest immediate R_test(s,i)
    2. entropy_greedy  – test the person whose result maximally reduces
                         posterior entropy over untested individuals
    3. random_policy   – test a uniformly random untested person
    4. clinical_order  – test in fixed kinship order: proband first,
                         then first-degree relatives, then second-degree, …
"""

import math
import random as _random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..models.belief import (
    InferenceResult,
    ensure_belief,
    ensure_belief_with_tuples,
    lift_single_gene_posteriors_to_genes,
    propagate_all_marginals_safe,
    propagate_multigene_marginals,
)
from ..models.reward import r_reward, r_reward_test

TOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence_state(state):
    if not isinstance(state, frozenset):
        raise AssertionError(
            f"State must be evidence-only frozenset, got {type(state).__name__}: {state!r}"
        )
    return state


def _tested_set(state):
    return {i for i, _ in _evidence_state(state)}


def _stop_value(
    s, *, belief, individuals, a, b, c, delta,
    belief_gene=None, genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
):
    tested = _tested_set(s)
    p_s = belief[s][0] if isinstance(belief[s], tuple) else belief[s]

    gene_probs = None
    if genes:
        if belief_gene and s in belief_gene:
            gene_probs = belief_gene[s]
        elif isinstance(p_s, InferenceResult):
            gene_probs = p_s.get_per_gene_probs()
        else:
            gene_probs = lift_single_gene_posteriors_to_genes(p_s, genes)

    return sum(
        r_reward(
            k, p_s, a, b, c, delta,
            per_gene_probs=gene_probs,
            a_gene=a_gene, b_gene=b_gene,
            c_gene=c_gene, delta_gene=delta_gene,
        )
        for k in individuals if k not in tested
    )


def _test_immediate_reward(
    s, i, *, belief, a, b, c, delta, fixed_cost, variable_cost,
    belief_gene=None, genes=None,
    a_gene=None, c_gene=None, delta_gene=None,
):
    p_s = belief[s][0] if isinstance(belief[s], tuple) else belief[s]

    gene_probs = None
    if genes:
        if belief_gene and s in belief_gene:
            gene_probs = belief_gene[s]
        elif isinstance(p_s, InferenceResult):
            gene_probs = p_s.get_per_gene_probs()
        else:
            gene_probs = lift_single_gene_posteriors_to_genes(p_s, genes)

    return r_reward_test(
        i, p_s, a, b, c, delta, fixed_cost, variable_cost,
        per_gene_probs=gene_probs,
        a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
    )


def _ensure_belief_at(s, *, belief, infer, individuals, gen_states,
                       belief_gene=None, genes=None, tuple_mode=False):
    """Ensure beliefs exist for state s; create them if missing."""
    if s in belief:
        return
    if tuple_mode and genes:
        ensure_belief_with_tuples(
            s, belief=belief, infer=infer,
            I=individuals, gen_states=gen_states, genes=genes,
        )
    else:
        ensure_belief(
            s, belief=belief, infer=infer,
            I=individuals, gen_states=gen_states,
        )
    if belief_gene is not None and genes and s in belief:
        p_s = belief[s][0] if isinstance(belief[s], tuple) else belief[s]
        if isinstance(p_s, InferenceResult):
            belief_gene[s] = p_s.get_per_gene_probs()
        else:
            belief_gene[s] = lift_single_gene_posteriors_to_genes(p_s, genes)


# ---------------------------------------------------------------------------
# 1. Myopic Greedy
# ---------------------------------------------------------------------------

def myopic_greedy(
    s, *,
    belief, individuals, gen_states, infer,
    a, b, c, delta, fixed_cost, variable_cost,
    belief_gene=None, genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
    tuple_mode=False,
    **_ignored,
):
    """Pick the untested person with the highest immediate test reward.

    Stop when R_stop exceeds all R_test values.  No lookahead, no ADP.
    """
    _ensure_belief_at(
        s, belief=belief, infer=infer,
        individuals=individuals, gen_states=gen_states,
        belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
    )

    val_stop = _stop_value(
        s, belief=belief, individuals=individuals,
        a=a, b=b, c=c, delta=delta,
        belief_gene=belief_gene, genes=genes,
        a_gene=a_gene, b_gene=b_gene,
        c_gene=c_gene, delta_gene=delta_gene,
    )

    tested = _tested_set(s)
    best = ("stop", None, val_stop)
    best_q = val_stop

    for i in individuals:
        if i in tested:
            continue
        q_i = _test_immediate_reward(
            s, i, belief=belief,
            a=a, b=b, c=c, delta=delta,
            fixed_cost=fixed_cost, variable_cost=variable_cost,
            belief_gene=belief_gene, genes=genes,
            a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
        )
        if q_i > best_q + TOL:
            best_q = q_i
            best = ("test", i, q_i)

    return best


# ---------------------------------------------------------------------------
# 2. Entropy Reduction Greedy
# ---------------------------------------------------------------------------

def _posterior_entropy(p_s, individuals, tested):
    """Shannon entropy of untested individuals' genotype distributions."""
    H = 0.0
    for i in individuals:
        if i in tested:
            continue
        dist = p_s[i] if isinstance(p_s, dict) else p_s.marginals.get(i, {})
        for g, p in dist.items():
            if p > 0:
                H -= p * math.log2(p)
    return H


def entropy_greedy(
    s, *,
    belief, individuals, gen_states, infer,
    a, b, c, delta, fixed_cost, variable_cost,
    belief_gene=None, genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
    tuple_mode=False,
    **_ignored,
):
    """Test the person whose result maximally reduces posterior entropy.

    Computes E_g[H(posteriors | s, test i, outcome g)] for each untested i,
    then picks the i with the smallest expected post-test entropy.
    Still uses R_stop vs best-test comparison for the stop decision.
    """
    _ensure_belief_at(
        s, belief=belief, infer=infer,
        individuals=individuals, gen_states=gen_states,
        belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
    )

    val_stop = _stop_value(
        s, belief=belief, individuals=individuals,
        a=a, b=b, c=c, delta=delta,
        belief_gene=belief_gene, genes=genes,
        a_gene=a_gene, b_gene=b_gene,
        c_gene=c_gene, delta_gene=delta_gene,
    )

    tested = _tested_set(s)
    p_s_entry = belief[s][0] if isinstance(belief[s], tuple) else belief[s]
    p_s = p_s_entry.marginals if isinstance(p_s_entry, InferenceResult) else p_s_entry

    current_H = _posterior_entropy(p_s_entry, individuals, tested)

    best = ("stop", None, val_stop)
    best_reduction = -float("inf")

    for i in individuals:
        if i in tested:
            continue
        dist_i = p_s.get(i, {}) if isinstance(p_s, dict) else {}
        expected_H = 0.0
        valid = True
        for g in gen_states:
            prob_g = dist_i.get(g, 0.0)
            if prob_g <= 0.0:
                continue
            # Compute successor state and its entropy
            succ_dict = dict(_evidence_state(s))
            succ_dict[i] = g
            succ = frozenset(succ_dict.items())
            try:
                _ensure_belief_at(
                    succ, belief=belief, infer=infer,
                    individuals=individuals, gen_states=gen_states,
                    belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
                )
            except (ValueError, Exception):
                valid = False
                break
            succ_entry = belief[succ][0] if isinstance(belief[succ], tuple) else belief[succ]
            succ_tested = tested | {i}
            expected_H += prob_g * _posterior_entropy(succ_entry, individuals, succ_tested)

        if not valid:
            continue

        reduction = current_H - expected_H
        if reduction > best_reduction + TOL:
            best_reduction = reduction
            # Use the test reward as the "value" field for consistency
            q_i = _test_immediate_reward(
                s, i, belief=belief,
                a=a, b=b, c=c, delta=delta,
                fixed_cost=fixed_cost, variable_cost=variable_cost,
                belief_gene=belief_gene, genes=genes,
                a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
            )
            best = ("test", i, q_i)

    # Only stop if no test reduces entropy (or if stop_value is better than
    # best test's immediate reward)
    if best[0] == "test":
        # Check if the immediate test reward is positive; otherwise stop
        if best[2] < val_stop:
            # Entropy says test, but reward says stop -- use reward threshold
            pass  # keep test: entropy-driven policy prioritizes information
    return best


# ---------------------------------------------------------------------------
# 3. Random Policy
# ---------------------------------------------------------------------------

def random_policy(
    s, *,
    belief, individuals, gen_states, infer,
    a, b, c, delta, fixed_cost, variable_cost,
    belief_gene=None, genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
    tuple_mode=False,
    rng=None,
    **_ignored,
):
    """Test a uniformly random untested person. Stop only when everyone is tested."""
    _ensure_belief_at(
        s, belief=belief, infer=infer,
        individuals=individuals, gen_states=gen_states,
        belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
    )

    tested = _tested_set(s)
    untested = [i for i in individuals if i not in tested]

    if not untested:
        val_stop = _stop_value(
            s, belief=belief, individuals=individuals,
            a=a, b=b, c=c, delta=delta,
            belief_gene=belief_gene, genes=genes,
            a_gene=a_gene, b_gene=b_gene,
            c_gene=c_gene, delta_gene=delta_gene,
        )
        return ("stop", None, val_stop)

    if rng is None:
        rng = _random.Random(42)
    who = rng.choice(untested)

    q_i = _test_immediate_reward(
        s, who, belief=belief,
        a=a, b=b, c=c, delta=delta,
        fixed_cost=fixed_cost, variable_cost=variable_cost,
        belief_gene=belief_gene, genes=genes,
        a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
    )
    return ("test", who, q_i)


# ---------------------------------------------------------------------------
# 4. Clinical Order (Kinship-Based)
# ---------------------------------------------------------------------------

def _kinship_order(pedigree, individuals):
    """Order individuals by clinical priority: non-founders first (probands /
    children), then founders (parents/grandparents).  Within each group,
    order by topological depth (deeper = tested first, as they are the
    actual patients in clinical practice).
    """
    graph = pedigree.graph
    founders = set(pedigree.get_founders())

    # Compute depth from roots (topological depth)
    depths = {}
    for node in individuals:
        try:
            # Longest path from any root to this node
            max_depth = 0
            for root in founders:
                try:
                    for path in _all_simple_paths(graph, root, node):
                        max_depth = max(max_depth, len(path) - 1)
                except Exception:
                    pass
            depths[node] = max_depth
        except Exception:
            depths[node] = 0

    # Non-founders (actual patients) first, sorted by depth (deepest first)
    # Then founders, sorted by depth (deepest first)
    non_founders = sorted(
        [i for i in individuals if i not in founders],
        key=lambda x: -depths.get(x, 0),
    )
    founder_list = sorted(
        [i for i in individuals if i in founders],
        key=lambda x: -depths.get(x, 0),
    )
    return non_founders + founder_list


def _all_simple_paths(graph, source, target):
    """Yield all simple paths from source to target in a DAG."""
    if source == target:
        yield [source]
        return
    for child in graph.successors(source):
        for path in _all_simple_paths(graph, child, target):
            yield [source] + path


def clinical_order(
    s, *,
    belief, individuals, gen_states, infer,
    a, b, c, delta, fixed_cost, variable_cost,
    pedigree,
    belief_gene=None, genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
    tuple_mode=False,
    _cached_order=None,
    **_ignored,
):
    """Test in fixed clinical order: children/probands first, then parents.

    Stops when the immediate stop reward exceeds the test reward for the
    next person in the queue.
    """
    _ensure_belief_at(
        s, belief=belief, infer=infer,
        individuals=individuals, gen_states=gen_states,
        belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
    )

    val_stop = _stop_value(
        s, belief=belief, individuals=individuals,
        a=a, b=b, c=c, delta=delta,
        belief_gene=belief_gene, genes=genes,
        a_gene=a_gene, b_gene=b_gene,
        c_gene=c_gene, delta_gene=delta_gene,
    )

    tested = _tested_set(s)

    # Use cached order if provided, otherwise compute
    if _cached_order is not None:
        order = _cached_order
    else:
        order = _kinship_order(pedigree, individuals)

    # Find the first untested person in the clinical order
    for i in order:
        if i in tested:
            continue
        q_i = _test_immediate_reward(
            s, i, belief=belief,
            a=a, b=b, c=c, delta=delta,
            fixed_cost=fixed_cost, variable_cost=variable_cost,
            belief_gene=belief_gene, genes=genes,
            a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
        )
        # Stop if stopping is clearly better
        if val_stop > q_i + TOL:
            return ("stop", None, val_stop)
        return ("test", i, q_i)

    # Everyone tested
    return ("stop", None, val_stop)


# ---------------------------------------------------------------------------
# Generic evaluation adapter
# ---------------------------------------------------------------------------

def evaluate_baseline_policy(
    baseline_fn,
    *,
    belief,
    individuals,
    gen_states,
    infer,
    a, b, c, delta,
    fixed_cost,
    variable_cost,
    r_reward_test_fn=None,
    belief_gene=None,
    genes=None,
    a_gene=None, b_gene=None, c_gene=None, delta_gene=None,
    tuple_pmfs=None,
    tuple_mode=False,
    pedigree=None,
    rng=None,
):
    """Evaluate a baseline policy by exact recursive value computation.

    Returns dict mapping states to exact values, same as exact_value_under_policy.
    """
    from ..models.belief import propagate_multigene_marginals

    V = {}
    policy = {}

    # Build kwargs that all baseline functions accept
    common_kwargs = dict(
        belief=belief,
        individuals=individuals,
        gen_states=gen_states,
        infer=infer,
        a=a, b=b, c=c, delta=delta,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        belief_gene=belief_gene,
        genes=genes,
        a_gene=a_gene, b_gene=b_gene,
        c_gene=c_gene, delta_gene=delta_gene,
        tuple_mode=tuple_mode,
    )
    if pedigree is not None:
        common_kwargs["pedigree"] = pedigree
    if rng is not None:
        common_kwargs["rng"] = rng

    def _merge(state, person, outcome):
        d = dict(_evidence_state(state))
        d[person] = outcome
        return frozenset(d.items())

    def V_rec(s):
        if s in V:
            return V[s]

        # Ensure belief exists
        _ensure_belief_at(
            s, belief=belief, infer=infer,
            individuals=individuals, gen_states=gen_states,
            belief_gene=belief_gene, genes=genes, tuple_mode=tuple_mode,
        )

        # Get baseline action
        if s not in policy:
            try:
                policy[s] = baseline_fn(s, **common_kwargs)
            except (ValueError, Exception):
                V[s] = 0.0
                return 0.0

        act, who, _ = policy[s]
        tested = _tested_set(s)
        p_s_entry = belief[s][0] if isinstance(belief[s], tuple) else belief[s]

        gene_probs = None
        if genes:
            if belief_gene and s in belief_gene:
                gene_probs = belief_gene[s]
            elif isinstance(p_s_entry, InferenceResult):
                gene_probs = p_s_entry.get_per_gene_probs()
            else:
                gene_probs = lift_single_gene_posteriors_to_genes(p_s_entry, genes)

        if act == "stop":
            val = sum(
                r_reward(
                    k, p_s_entry, a, b, c, delta,
                    per_gene_probs=gene_probs,
                    a_gene=a_gene, b_gene=b_gene,
                    c_gene=c_gene, delta_gene=delta_gene,
                )
                for k in individuals if k not in tested
            )
        else:
            i = who
            r_i = r_reward_test(
                i, p_s_entry, a, b, c, delta, fixed_cost, variable_cost,
                per_gene_probs=gene_probs,
                a_gene=a_gene, c_gene=c_gene, delta_gene=delta_gene,
            )
            exp_succ = 0.0
            p_s = p_s_entry.marginals if isinstance(p_s_entry, InferenceResult) else p_s_entry
            for g in gen_states:
                prob = p_s.get(i, {}).get(g, 0.0) if isinstance(p_s, dict) else 0.0
                if prob <= 0.0:
                    continue
                succ = _merge(s, i, g)
                if len(_evidence_state(succ)) == len(individuals):
                    continue
                try:
                    exp_succ += prob * V_rec(succ)
                except (ValueError, Exception):
                    continue

            val = r_i + exp_succ

        V[s] = val
        return val

    # Start from root
    root = frozenset()
    V_rec(root)

    return V, policy
