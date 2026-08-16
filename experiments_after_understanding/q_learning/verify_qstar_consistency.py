"""Self-consistency check on Q* labels, BEFORE trusting anything trained on them.

Bellman identity that must hold, by definition of V* and policy_dp:
    V*(s) = max( v_stop(s), max_a Q*(s, a) )
    and if policy_dp[s] says "test person p", then Q*(s, p) must equal V*(s) exactly
    (p is the argmax that produced V*(s) in the first place).

This script recomputes Q*(s, a) for every action at every state in a few configs,
using the exact formula in qsa_data.build_qsa_index, and checks both identities
against the cached V_star / policy_dp / v_stop — no training, no learned model
involved. If this doesn't hold, the (s,a,Q*) training data is wrong regardless
of anything downstream.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

HERE        = Path(__file__).resolve().parent
EXP_ROOT    = HERE.parent
ROOT        = EXP_ROOT.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))

from exputils.eval import _get_entry, _stop_val, _test_r  # noqa: E402

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

CHECK_CONFIGS = [
    "Trio_LowHigh_Base_3gene",              # train family, smallest
    "ThreeGeneration_LowHigh_Base_3gene",   # test family
]


def qstar(state, person, per_gene, tuple_pmfs, config, belief, V_star):
    r_i = _test_r(person, per_gene, config)
    pmf = tuple_pmfs.get(person, {})
    exp_v = sum(
        prob * V_star.get(frozenset(state | {(person, g)}), 0.0)
        for g, prob in pmf.items()
        if prob > 1e-12 and frozenset(state | {(person, g)}) in belief
    )
    return r_i + exp_v


def main():
    for key in CHECK_CONFIGS:
        print(f"\n{'='*70}\n{key}\n{'='*70}")
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        states      = ds["states"]
        individuals = ds["individuals"]
        belief      = ds["belief"]
        config      = ds["config"]
        genes       = ds.get("genes", ("GeneA", "GeneB", "GeneC"))
        V_star      = ds["V_star"]
        policy_dp   = ds["policy_dp"]

        n_checked = 0
        n_max_mismatch = 0
        n_argmax_mismatch = 0
        worst_max_err = 0.0
        worst_argmax_err = 0.0

        # Check every state (these configs are small enough) for the max-identity,
        # and separately confirm the policy_dp-chosen action's Q* matches V*.
        for state in states:
            tested = {p for p, _ in state}
            untested = [p for p in individuals if p not in tested]
            if not untested:
                continue

            per_gene, tuple_pmfs = _get_entry(belief, state, genes)
            v_stop = _stop_val(per_gene, individuals, tested, config)

            q_vals = {p: qstar(state, p, per_gene, tuple_pmfs, config, belief, V_star) for p in untested}
            recomputed_v = max(v_stop, max(q_vals.values()))
            cached_v = V_star[state]

            n_checked += 1
            err = abs(recomputed_v - cached_v)
            if err > 1e-6:
                n_max_mismatch += 1
                worst_max_err = max(worst_max_err, err)

            dp_action, dp_person, dp_value = policy_dp[state]
            if dp_action != "stop":
                argmax_err = abs(q_vals[dp_person] - cached_v)
                if argmax_err > 1e-6:
                    n_argmax_mismatch += 1
                    worst_argmax_err = max(worst_argmax_err, argmax_err)

        print(f"states checked: {n_checked:,}")
        print(f"[identity 1] V*(s) == max(v_stop, max_a Q*(s,a)):  "
              f"{n_checked - n_max_mismatch}/{n_checked} match, worst err = {worst_max_err:.2e}")
        print(f"[identity 2] Q*(s, policy_dp's action) == V*(s):   "
              f"mismatches = {n_argmax_mismatch}, worst err = {worst_argmax_err:.2e}")


if __name__ == "__main__":
    main()
