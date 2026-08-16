"""Shared eval logic: rollout with person-selection logging + ratio2.

Used by both mlp/run.py and gnn/run.py.
The only model-specific part is precompute_vhat — caller passes that in.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT        = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "ground-up-experiments"

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward, r_reward_test

GENES = ("GeneA", "GeneB", "GeneC")


def _get_entry(belief, state, genes):
    entry = belief[state]
    if isinstance(entry, InferenceResult):
        return entry.get_per_gene_probs(), entry.get_tuple_pmfs()
    return lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES), None


def _stop_val(per_gene, individuals, tested, config):
    return float(sum(
        r_reward(k, None, config.a, config.b, config.c, config.delta,
                 per_gene_probs=per_gene, a_gene=config.a_gene,
                 b_gene=config.b_gene, c_gene=config.c_gene,
                 delta_gene=config.delta_gene)
        for k in individuals if k not in tested
    ))


def _test_r(person, per_gene, config):
    return float(r_reward_test(
        person, None, config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
        per_gene_probs=per_gene, a_gene=config.a_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
    ))


def _q_hat(state, person, per_gene, tuple_pmfs, v_hat, belief):
    """Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')."""
    if tuple_pmfs is None:
        raise RuntimeError("tuple_pmfs is None — need InferenceResult in belief map")
    pmf   = tuple_pmfs.get(person, {})
    r_i   = _test_r(person, per_gene, belief["config"])
    exp_v = sum(
        prob * v_hat.get(frozenset(state | {(person, g)}), 0.0)
        for g, prob in pmf.items()
        if prob > 1e-12 and frozenset(state | {(person, g)}) in belief
    )
    return r_i + exp_v


def rollout(v_hat: dict, ds: dict, log: Callable = print, trace: bool = True) -> tuple[float, float]:
    """
    Greedy rollout using V_hat-derived action selection.
    Logs which person is selected at each step if trace=True.

    Returns (ratio2, accumulated_value).
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    genes       = ds.get("genes", GENES)

    # patch config into belief dict so _q_hat can reach it
    belief["config"] = config

    memo: dict = {}

    def value_at(state):
        if state in memo:
            return memo[state]

        per_gene, tuple_pmfs = _get_entry(belief, state, genes)
        tested   = {p for p, _ in state}
        untested = [p for p in individuals if p not in tested]

        if not untested:
            memo[state] = 0.0
            return 0.0

        v_stop = _stop_val(per_gene, individuals, tested, config)
        q_hats = {p: _q_hat(state, p, per_gene, tuple_pmfs, v_hat, belief)
                  for p in untested}

        best_person = max(untested, key=lambda p: q_hats[p])
        best_q      = q_hats[best_person]

        if trace:
            stage = len(tested)
            tested_str   = "{" + ", ".join(sorted(tested)) + "}" if tested else "root"
            q_str        = "  ".join(f"{p}={q_hats[p]:.3f}" for p in individuals if p not in tested)
            log(f"    step {stage}  state={tested_str}")
            log(f"           Q_hat: {q_str}")
            if best_q <= v_stop:
                log(f"           → STOP  (best_q={best_q:.3f} <= v_stop={v_stop:.3f})")
            else:
                log(f"           → TEST {best_person}  (Q={best_q:.3f}, v_stop={v_stop:.3f})")

        if best_q <= v_stop:
            memo[state] = v_stop
            return v_stop

        r_best   = _test_r(best_person, per_gene, config)
        pmf_best = tuple_pmfs.get(best_person, {})
        exp_true = sum(
            prob * value_at(frozenset(state | {(best_person, g)}))
            for g, prob in pmf_best.items()
            if prob > 1e-12 and frozenset(state | {(best_person, g)}) in belief
        )
        result = r_best + exp_true
        memo[state] = result
        return result

    L      = value_at(frozenset())
    V_root = ds["V_root"]
    V_stop = ds["V_stop_root"]
    denom  = V_root - V_stop
    ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

    # clean up patch
    del belief["config"]
    return float(ratio2), float(L)


def q_rollout(q_hat: dict, ds: dict, log: Callable = print, trace: bool = True) -> tuple[float, float]:
    """
    Greedy rollout using a directly-trained Q(s,a) model — the Q(s,a) analog of
    rollout() above. Same recursive structure: at each state, pick the action
    the MODEL itself prefers (not DP's), then branch over every real outcome
    probability (not just the most likely one), memoized over the reachable
    state space, all the way to termination.

    Unlike rollout(), this does not derive Q from a V_hat dict via the Bellman
    formula (_q_hat) — q_hat already contains the model's own Q(s,a) predictions,
    precomputed in one batched forward pass by the caller.

    q_hat: {state: {person: predicted_Q(s, person)}} — must cover every
           (state, untested person) pair reachable from the root under the
           greedy policy this induces.

    Returns (ratio2, accumulated_value). Same semantics as rollout(): ratio2=0
    means the induced policy matches exact DP; ratio2=1 means it's no better
    than stopping immediately.
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    config      = ds["config"]
    genes       = ds.get("genes", GENES)

    memo: dict = {}

    def value_at(state):
        if state in memo:
            return memo[state]

        per_gene, tuple_pmfs = _get_entry(belief, state, genes)
        tested   = {p for p, _ in state}
        untested = [p for p in individuals if p not in tested]

        if not untested:
            memo[state] = 0.0
            return 0.0

        v_stop = _stop_val(per_gene, individuals, tested, config)
        q_vals = q_hat[state]

        best_person = max(untested, key=lambda p: q_vals[p])
        best_q      = q_vals[best_person]

        if trace:
            stage = len(tested)
            tested_str = "{" + ", ".join(sorted(tested)) + "}" if tested else "root"
            q_str = "  ".join(f"{p}={q_vals[p]:.3f}" for p in individuals if p not in tested)
            log(f"    step {stage}  state={tested_str}")
            log(f"           Q_hat: {q_str}")
            if best_q <= v_stop:
                log(f"           → STOP  (best_q={best_q:.3f} <= v_stop={v_stop:.3f})")
            else:
                log(f"           → TEST {best_person}  (Q={best_q:.3f}, v_stop={v_stop:.3f})")

        if best_q <= v_stop:
            memo[state] = v_stop
            return v_stop

        r_best   = _test_r(best_person, per_gene, config)
        pmf_best = tuple_pmfs.get(best_person, {})
        exp_true = sum(
            prob * value_at(frozenset(state | {(best_person, g)}))
            for g, prob in pmf_best.items()
            if prob > 1e-12 and frozenset(state | {(best_person, g)}) in belief
        )
        result = r_best + exp_true
        memo[state] = result
        return result

    L      = value_at(frozenset())
    V_root = ds["V_root"]
    V_stop = ds["V_stop_root"]
    denom  = V_root - V_stop
    ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

    return float(ratio2), float(L)
