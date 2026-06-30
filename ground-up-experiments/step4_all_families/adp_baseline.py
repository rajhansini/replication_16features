"""
adp_baseline.py — compute ADP ratio2 for any family using scipy linprog.

Replicates the ABCD-16 ADP approach from the paper WITHOUT Gurobi:
  1. Compute 16 ABCD features φ(s) for every state s
  2. Solve the ALP (Approximate Linear Program):
       min  Σ_s ρ(s) · θ·φ(s)
       s.t. θ·φ(s) ≥ r_stop(s)                           for all s
            θ·φ(s) ≥ r_test(i,s) + Σ_o P(o|i,s)·θ·φ(s') for all s, i∉tested
  3. Run greedy policy under θ·φ to get L
  4. ratio2 = (V* - L) / (V* - V_stop)

Run:
    python step4_all_families/adp_baseline.py
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
from genetic_dp.optimisation.myopic_adp import build_state_features, ABCD16_DIRECT_FEATURES

# Families we want ADP baselines for
TARGET_FAMILIES = [
    "ThreeGeneration_MediumEven_Base",
    "ThreeGeneration_MediumEven_Aggressive",
]

CACHE_DIR = HERE / "results" / "cache"


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


def compute_features(state, ds) -> np.ndarray:
    """Compute 16 ABCD feature vector for a state."""
    belief      = ds["belief"]
    individuals = ds["individuals"]
    pedigree    = ds["pedigree"]
    genes       = ds.get("genes")
    feat_dict   = build_state_features(
        state,
        belief=belief,
        individuals=individuals,
        pedigree=pedigree,
        genes=genes,
    )
    return np.array([feat_dict.get(f, 0.0) for f in ABCD16_DIRECT_FEATURES], dtype=np.float64)


def compute_v_stop(state, ds) -> float:
    entry       = ds["belief"][state]
    marg        = _marg(entry)
    config      = ds["config"]
    individuals = ds["individuals"]
    genes       = ds.get("genes")
    tested      = {i for i, _ in state}
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


def solve_adp(ds) -> np.ndarray:
    """
    Solve the ALP to get feature coefficients θ.

    ALP (minimise over θ):
      min  Σ_s φ(s)·θ          (uniform state weights)
      s.t. φ(s)·θ ≥ V_stop(s)  for all s           (stop constraint)
           φ(s)·θ ≥ r(i,s) + Σ_o P(o|i,s) φ(s')·θ  for all s, i∉tested (test constraint)

    Rearranged test constraint (move RHS θ terms to left):
      [φ(s) - Σ_o P(o|i,s) φ(s')] · θ ≥ r(i,s)
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    V_star      = ds["V_star"]
    genes       = ds.get("genes")
    gen_states  = ds.get("two_gene_states", GENOTYPE_STATES)

    # Only reachable states
    states = [s for s in V_star if s in belief]
    n      = len(states)
    k      = len(ABCD16_DIRECT_FEATURES)

    print(f"    Building ALP: {n} states × {k} features...")

    # Pre-compute features
    phi = {}
    for s in states:
        phi[s] = compute_features(s, ds)

    # Build constraint rows: A_ub @ θ ≤ -b_ub  (scipy uses ≤)
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

            # Σ_o P(o|i,s) φ(s') where s' = s ∪ {(i,o)}
            exp_phi_next = np.zeros(k)
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12:
                    continue
                next_s = frozenset(s | {(i, g)})
                if next_s in phi:
                    exp_phi_next += prob_g * phi[next_s]

            # φ(s)·θ - Σ_o P(o)·φ(s')·θ ≥ r_i
            # → -(φ(s) - exp_phi_next)·θ ≤ -r_i
            A_rows.append(-(phi[s] - exp_phi_next))
            b_rows.append(-r_i)

    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_rows, dtype=np.float64)

    # Objective: minimize Σ_s φ(s)·θ = (Σ_s φ(s)) · θ
    c = np.sum([phi[s] for s in states], axis=0)

    print(f"    LP size: {A_ub.shape[0]} constraints × {k} variables — solving...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs",
                     options={"disp": False, "time_limit": 300})

    if result.status != 0:
        print(f"    WARNING: LP solver status={result.status} ({result.message})")

    theta = result.x
    print(f"    θ = {theta}")
    return theta, phi


def greedy_policy_value(theta, phi, ds) -> float:
    """Simulate ADP greedy policy and return its true expected value L."""
    belief      = ds["belief"]
    individuals = ds["individuals"]
    V_star      = ds["V_star"]
    genes       = ds.get("genes")

    memo = {}

    def value_at(state) -> float:
        if state in memo:
            return memo[state]
        tested  = {i for i, _ in state}
        v_stop  = compute_v_stop(state, ds)

        if len(tested) == len(individuals):
            memo[state] = 0.0
            return 0.0

        entry = belief[state]

        # ADP picks action with highest θ·φ(next) expected value
        best_q      = v_stop
        best_person = None

        for i in individuals:
            if i in tested:
                continue
            r_i   = compute_r_test(i, state, ds)
            pmf_i = _tuple_pmf(entry, i)
            exp_v = 0.0
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12: continue
                next_s = frozenset(state | {(i, g)})
                if next_s in phi:
                    exp_v += prob_g * float(phi[next_s] @ theta)
            q_i = r_i + exp_v
            if q_i > best_q:
                best_q      = q_i
                best_person = i

        if best_person is None:
            memo[state] = v_stop
            return v_stop

        # Compute TRUE value of following this choice
        r_best   = compute_r_test(best_person, state, ds)
        pmf_best = _tuple_pmf(entry, best_person)
        exp_true = 0.0
        for g, prob_g in pmf_best.items():
            if prob_g <= 1e-12: continue
            next_s = frozenset(state | {(best_person, g)})
            if next_s not in belief: continue
            exp_true += prob_g * value_at(next_s)

        result         = r_best + exp_true
        memo[state]    = result
        return result

    return value_at(frozenset())


def main():
    results = {}

    for key in TARGET_FAMILIES:
        pkl_path = CACHE_DIR / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"\n[SKIP] {key} — cache not found at {pkl_path}")
            continue

        print(f"\n{'='*60}")
        print(f"ADP baseline: {key}")
        print(f"{'='*60}")

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

        results[key] = {
            "V_root":      V_root,
            "V_stop_root": V_stop_root,
            "L_adp":       float(L),
            "ratio2_adp":  float(ratio2),
            "theta":       theta.tolist(),
        }

    # Print comparison with net results
    net_results = json.loads((HERE / "results" / "results.json").read_text())
    print(f"\n{'='*60}")
    print("COMPARISON (TEST families):")
    print(f"{'='*60}")
    print(f"  {'Family':<40} {'ratio2(ADP)':>12} {'ratio2(Net)':>12} {'Better':>8}")
    print(f"  {'-'*76}")
    for key in TARGET_FAMILIES:
        if key not in results: continue
        r2_adp = results[key]["ratio2_adp"]
        r2_net = net_results["families"][key]["ratio2_net"]
        better = f"{r2_adp/r2_net:.1f}x" if r2_net > 0 else "N/A"
        print(f"  {key:<40} {r2_adp:>12.6f} {r2_net:>12.6f} {better:>8}")

    # Save
    out = HERE / "results" / "adp_baseline_test.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
