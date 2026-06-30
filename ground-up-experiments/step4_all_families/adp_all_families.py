"""
adp_all_families.py — compute ADP ratio2 FRESH for ALL families using HiGHS (via scipy).

Same ABCD-16 ALP formulation as the paper.
Gurobi's restricted license caps at ~2000 LP constraints, but the smallest
family has 40,720 constraints. HiGHS solves the same LP to the same optimal
solution — the solver choice is irrelevant to the result.

ALP formulation:
  min  Σ_s ρ(s) · θ·φ(s)
  s.t. θ·φ(s) ≥ V_stop(s)                              for all s
       θ·φ(s) ≥ r(i,s) + Σ_o P(o|i,s)·θ·φ(s')         for all s, i∉tested

Run:
    python ground-up-experiments/step4_all_families/adp_all_families.py
Output:
    ground-up-experiments/step4_all_families/results/adp_all_families.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from genetic_dp.exact_dp.utils  import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief   import InferenceResult
from genetic_dp.models.reward   import r_reward, r_reward_test
from genetic_dp.optimisation.myopic_adp import (
    build_state_features, ABCD16_DIRECT_FEATURES, FEATURE_SEMANTICS_ABCD_HAND,
)

CACHE_DIR = HERE / "results" / "cache"

ALL_FAMILIES = [
    "ThreeGeneration_LowHigh_Base",
    "ThreeGeneration_LowHigh_Aggressive",
    "ThreeGeneration_MediumEven_Base",
    "ThreeGeneration_MediumEven_Aggressive",
    "Extended_LowHigh_Base",
    "Extended_LowHigh_Aggressive",
    "Extended_MediumEven_Base",
    "Extended_MediumEven_Aggressive",
]

# Numbers from the ABCD-16 paper (for validation — we replace these with fresh runs)
PAPER_HARDCODED = {
    "Extended_LowHigh_Base":              0.10068959165,
    "ThreeGeneration_LowHigh_Aggressive": 0.05132042772,
    "Extended_MediumEven_Base":           0.15896943584,
}


def _marg(entry):
    return entry.marginals if isinstance(entry, InferenceResult) else entry

def _per_gene(entry, genes):
    if isinstance(entry, InferenceResult):
        return entry.get_per_gene_probs()
    return lift_tuple_posteriors_to_genes(entry, genes, GENOTYPE_STATES)

def _tuple_pmf(entry, person):
    if isinstance(entry, InferenceResult):
        return entry.get_tuple_pmfs().get(person, {})
    return entry[person]


def precompute_myopic(ds) -> tuple[dict, dict, dict]:
    """
    Compute one-step myopic policy for every state.

    myopic_policy[s]   = ("test", person) or ("stop",)
    myopic_values[s]   = V_myopic(s) = max(V_stop(s), max_i Q_myopic(i,s))
    myopic_residuals[s] = V_myopic(s) - V_stop(s)  (how much better than stopping)
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    policy, values, residuals = {}, {}, {}

    for s in belief:
        tested = {i for i, _ in s}
        vs     = compute_v_stop(s, ds)
        entry  = belief[s]
        best_q = vs
        best_p = None

        for i in individuals:
            if i in tested:
                continue
            ri    = compute_r_test(i, s, ds)
            pmf_i = _tuple_pmf(entry, i)
            exp_vs_next = sum(
                p * compute_v_stop(frozenset(s | {(i, g)}), ds)
                for g, p in pmf_i.items()
                if p > 1e-12 and frozenset(s | {(i, g)}) in belief
            )
            q_i = ri + exp_vs_next
            if q_i > best_q:
                best_q = q_i
                best_p = i

        if best_p is None:
            policy[s]    = ("stop",)
            values[s]    = vs
            residuals[s] = 0.0
        else:
            policy[s]    = ("test", best_p)
            values[s]    = best_q
            residuals[s] = best_q - vs

    return policy, values, residuals


def compute_features(state, ds, myopic_policy, myopic_values, myopic_residuals) -> np.ndarray:
    feat_dict = build_state_features(
        state,
        belief=ds["belief"],
        individuals=ds["individuals"],
        pedigree=ds["pedigree"],
        genes=ds.get("genes"),
        myopic_policy=myopic_policy,
        myopic_values=myopic_values,
        myopic_residuals=myopic_residuals,
        feature_semantics=FEATURE_SEMANTICS_ABCD_HAND,
    )
    return np.array([feat_dict.get(f, 0.0) for f in ABCD16_DIRECT_FEATURES], dtype=np.float64)


def compute_v_stop(state, ds) -> float:
    entry       = ds["belief"][state]
    marg        = _marg(entry)
    config      = ds["config"]
    individuals = ds["individuals"]
    tested      = {i for i, _ in state}
    genes       = ds.get("genes")
    if genes:
        pg = _per_gene(entry, genes)
        return float(sum(
            r_reward(k, marg, config.a, config.b, config.c, config.delta,
                     per_gene_probs=pg,
                     a_gene=config.a_gene, b_gene=config.b_gene,
                     c_gene=config.c_gene, delta_gene=config.delta_gene)
            for k in individuals if k not in tested
        ))
    return float(sum(
        r_reward(k, marg, config.a, config.b, config.c, config.delta)
        for k in individuals if k not in tested
    ))


def compute_r_test(i, state, ds) -> float:
    entry  = ds["belief"][state]
    marg   = _marg(entry)
    config = ds["config"]
    genes  = ds.get("genes")
    if genes:
        pg = _per_gene(entry, genes)
        return float(r_reward_test(
            i, marg, config.a, config.b, config.c, config.delta,
            config.fixed_cost, config.variable_cost,
            per_gene_probs=pg,
            a_gene=config.a_gene, c_gene=config.c_gene, delta_gene=config.delta_gene,
        ))
    return float(r_reward_test(
        i, marg, config.a, config.b, config.c, config.delta,
        config.fixed_cost, config.variable_cost,
    ))


def solve_adp(ds) -> tuple[np.ndarray, dict]:
    """
    Solve the ABCD-16 ALP (same as paper) using scipy/HiGHS.

    HiGHS produces the same optimal θ as Gurobi would for this LP.
    Gurobi's restricted dev license cannot handle >2000 constraints;
    our smallest family has 40,720 constraints so HiGHS is the right tool.
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    V_star      = ds["V_star"]
    genes       = ds.get("genes")

    states = [s for s in V_star if s in belief]
    n      = len(states)
    k      = len(ABCD16_DIRECT_FEATURES)

    print(f"    Pre-computing myopic policy for {n} states...")
    myopic_policy, myopic_values, myopic_residuals = precompute_myopic(ds)

    print(f"    Pre-computing features for {n} states...")
    phi = {s: compute_features(s, ds, myopic_policy, myopic_values, myopic_residuals) for s in states}
    phi_set = set(phi.keys())

    print(f"    Building ALP: {n} states × {k} features...")
    A_rows, b_rows = [], []

    for s in states:
        tested = {i for i, _ in s}

        # Stop constraint: φ(s)·θ ≥ V_stop(s)  →  -φ(s)·θ ≤ -V_stop(s)
        v_stop = compute_v_stop(s, ds)
        A_rows.append(-phi[s])
        b_rows.append(-v_stop)

        # Test constraints for each untested person
        entry = belief[s]
        for i in individuals:
            if i in tested:
                continue
            r_i   = compute_r_test(i, s, ds)
            pmf_i = _tuple_pmf(entry, i)

            exp_phi_next = np.zeros(k)
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12:
                    continue
                next_s = frozenset(s | {(i, g)})
                if next_s in phi_set:
                    exp_phi_next += prob_g * phi[next_s]

            # [φ(s) - E[φ(s')]] · θ ≥ r_i  →  -[φ(s) - E[φ(s')]] · θ ≤ -r_i
            A_rows.append(-(phi[s] - exp_phi_next))
            b_rows.append(-r_i)

    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_rows, dtype=np.float64)

    # Objective: minimise Σ_s φ(s)·θ
    c = np.sum([phi[s] for s in states], axis=0)

    print(f"    Solving LP: {A_ub.shape[0]} constraints × {k} vars (HiGHS)...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs",
                     bounds=[(None, None)] * k,  # theta unbounded — scipy default is x>=0 which is wrong
                     options={"disp": False, "time_limit": 600})

    if result.status != 0:
        print(f"    WARNING: LP solver status={result.status} ({result.message})")

    theta = result.x
    print(f"    Solved: obj={result.fun:.6f}")
    return theta, phi


def greedy_policy_value(theta: np.ndarray, phi: dict, ds) -> float:
    """True value of the ADP greedy policy (exact rollout)."""
    individuals = ds["individuals"]
    belief      = ds["belief"]

    memo = {}

    def value_at(state) -> float:
        if state in memo:
            return memo[state]
        tested = {i for i, _ in state}
        v_stop = compute_v_stop(state, ds)

        if len(tested) == len(individuals):
            memo[state] = 0.0
            return 0.0

        entry       = belief[state]
        best_q      = v_stop
        best_person = None

        for i in individuals:
            if i in tested:
                continue
            r_i   = compute_r_test(i, state, ds)
            pmf_i = _tuple_pmf(entry, i)
            exp_v = sum(
                prob_g * float(phi[frozenset(state | {(i, g)})] @ theta)
                for g, prob_g in pmf_i.items()
                if prob_g > 1e-12 and frozenset(state | {(i, g)}) in phi
            )
            q_i = r_i + exp_v
            if q_i > best_q:
                best_q      = q_i
                best_person = i

        if best_person is None:
            memo[state] = v_stop
            return v_stop

        r_best   = compute_r_test(best_person, state, ds)
        pmf_best = _tuple_pmf(entry, best_person)
        exp_true = sum(
            prob_g * value_at(frozenset(state | {(best_person, g)}))
            for g, prob_g in pmf_best.items()
            if prob_g > 1e-12 and frozenset(state | {(best_person, g)}) in belief
        )
        result      = r_best + exp_true
        memo[state] = result
        return result

    return value_at(frozenset())


def main():
    results = {}

    for key in ALL_FAMILIES:
        pkl_path = CACHE_DIR / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"\n[SKIP] {key} — cache not found")
            continue

        print(f"\n{'='*65}")
        print(f"ADP (fresh): {key}")
        print(f"{'='*65}")

        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)

        V_root      = ds["V_root"]
        V_stop_root = ds["V_stop_root"]

        theta, phi = solve_adp(ds)
        L          = greedy_policy_value(theta, phi, ds)
        denom      = V_root - V_stop_root
        ratio2     = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

        print(f"\n    V*(root)     = {V_root:.6f}")
        print(f"    V_stop(root) = {V_stop_root:.6f}")
        print(f"    L (ADP)      = {L:.6f}")
        print(f"    ratio2 (ADP) = {ratio2:.6f}")

        paper_val = PAPER_HARDCODED.get(key)
        if paper_val is not None:
            print(f"    paper ratio2 = {paper_val:.6f}  (diff = {ratio2-paper_val:+.6f})")

        results[key] = {
            "V_root":      float(V_root),
            "V_stop_root": float(V_stop_root),
            "L_adp":       float(L),
            "ratio2_adp":  float(ratio2),
            "theta":       theta.tolist(),
            "solver":      "HiGHS (scipy)",
        }

    print(f"\n{'='*65}")
    print("SUMMARY — ratio2_adp (lower = better, 0 = optimal)")
    print(f"{'='*65}")
    print(f"  {'Family':<42} {'Fresh':>9} {'Paper':>9} {'Delta':>8}")
    print(f"  {'-'*72}")
    for key, r in results.items():
        r2    = r["ratio2_adp"]
        paper = PAPER_HARDCODED.get(key)
        p_str = f"{paper:.6f}" if paper else "      --"
        d_str = f"{r2 - paper:+.4f}" if paper else "      --"
        print(f"  {key:<42} {r2:>9.6f} {p_str:>9} {d_str:>8}")

    out = HERE / "results" / "adp_all_families.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
