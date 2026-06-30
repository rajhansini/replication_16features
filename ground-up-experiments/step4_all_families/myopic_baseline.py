"""
myopic_baseline.py — compute myopic policy ratio2 for the test families.

Myopic policy: at each state, test the person who gives the highest
one-step expected reward (r_test(i,s) + E[V_stop(s')]).
No LP needed — pure greedy one-step lookahead.

This is a natural, honest baseline:
  - Stronger than random/stop
  - Weaker than ADP (no lookahead beyond one step)
  - If our net beats myopic, it's learning genuine multi-step value
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief  import InferenceResult
from genetic_dp.models.reward  import r_reward, r_reward_test

CACHE_DIR     = HERE / "results" / "cache"
TARGET_KEYS   = [
    "ThreeGeneration_MediumEven_Base",
    "ThreeGeneration_MediumEven_Aggressive",
]


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

def v_stop(state, ds):
    entry = ds["belief"][state]
    marg  = _marg(entry)
    cfg   = ds["config"]
    genes = ds.get("genes")
    tested = {i for i, _ in state}
    if genes:
        pg = _per_gene(entry, genes)
        return float(sum(
            r_reward(k, marg, cfg.a, cfg.b, cfg.c, cfg.delta,
                     per_gene_probs=pg, a_gene=cfg.a_gene, b_gene=cfg.b_gene,
                     c_gene=cfg.c_gene, delta_gene=cfg.delta_gene)
            for k in ds["individuals"] if k not in tested
        ))
    return float(sum(
        r_reward(k, marg, cfg.a, cfg.b, cfg.c, cfg.delta)
        for k in ds["individuals"] if k not in tested
    ))

def r_test(i, state, ds):
    entry = ds["belief"][state]
    marg  = _marg(entry)
    cfg   = ds["config"]
    genes = ds.get("genes")
    if genes:
        pg = _per_gene(entry, genes)
        return float(r_reward_test(
            i, marg, cfg.a, cfg.b, cfg.c, cfg.delta,
            cfg.fixed_cost, cfg.variable_cost,
            per_gene_probs=pg, a_gene=cfg.a_gene,
            c_gene=cfg.c_gene, delta_gene=cfg.delta_gene,
        ))
    return float(r_reward_test(
        i, marg, cfg.a, cfg.b, cfg.c, cfg.delta,
        cfg.fixed_cost, cfg.variable_cost,
    ))


def myopic_policy_value(ds) -> float:
    """
    Myopic policy: pick action with highest r_test(i,s) + E[V_stop(s')].
    Returns true expected value L from root.
    """
    belief      = ds["belief"]
    individuals = ds["individuals"]
    memo        = {}

    def value_at(state) -> float:
        if state in memo:
            return memo[state]

        tested = {i for i, _ in state}
        vs     = v_stop(state, ds)

        if len(tested) == len(individuals):
            memo[state] = 0.0
            return 0.0

        entry = belief[state]

        # Myopic: pick i that maximises r_test(i) + E[V_stop(next_state)]
        best_myopic_q = vs
        best_person   = None

        for i in individuals:
            if i in tested:
                continue
            ri    = r_test(i, state, ds)
            pmf_i = _tuple_pmf(entry, i)
            exp_vstop_next = 0.0
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12: continue
                next_s = frozenset(state | {(i, g)})
                if next_s in belief:
                    exp_vstop_next += prob_g * v_stop(next_s, ds)
            q_i = ri + exp_vstop_next
            if q_i > best_myopic_q:
                best_myopic_q = q_i
                best_person   = i

        if best_person is None:
            memo[state] = vs
            return vs

        # TRUE value of following myopic choice recursively
        ri_best  = r_test(best_person, state, ds)
        pmf_best = _tuple_pmf(entry, best_person)
        exp_true = 0.0
        for g, prob_g in pmf_best.items():
            if prob_g <= 1e-12: continue
            next_s = frozenset(state | {(best_person, g)})
            if next_s not in belief: continue
            exp_true += prob_g * value_at(next_s)

        result      = ri_best + exp_true
        memo[state] = result
        return result

    return value_at(frozenset())


def main():
    net_results = json.loads((HERE / "results" / "results.json").read_text())
    myopic_results = {}

    for key in TARGET_KEYS:
        pkl_path = CACHE_DIR / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"[SKIP] {key} — not cached")
            continue

        print(f"\n{key}")
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)

        V_root  = ds["V_root"]
        V_stop  = ds["V_stop_root"]
        denom   = V_root - V_stop

        L_myopic = myopic_policy_value(ds)
        r2_myopic = (V_root - L_myopic) / denom if abs(denom) > 1e-12 else 0.0

        r2_net = net_results["families"][key]["ratio2_net"]
        L_net  = net_results["families"][key]["L_net"]

        print(f"  V*(root)        = {V_root:.6f}")
        print(f"  V_stop(root)    = {V_stop:.6f}")
        print(f"  L (myopic)      = {L_myopic:.6f}  ratio2={r2_myopic:.6f}")
        print(f"  L (net)         = {L_net:.6f}  ratio2={r2_net:.6f}")
        print(f"  Net vs myopic   = {r2_myopic/r2_net:.2f}x better")

        myopic_results[key] = {
            "V_root": V_root, "V_stop_root": V_stop,
            "L_myopic": float(L_myopic), "ratio2_myopic": float(r2_myopic),
            "L_net": float(L_net), "ratio2_net": float(r2_net),
        }
        del ds

    print(f"\n{'='*70}")
    print("FULL COMPARISON TABLE (all families):")
    print(f"{'='*70}")
    print(f"  {'Family':<42} {'Split':<6} {'ratio2(Net)':>11} {'ratio2(Baseline)':>17} {'Better':>7}")
    print(f"  {'-'*85}")

    ADP_BASELINE = {
        "Extended_LowHigh_Base":              0.10069,
        "Extended_MediumEven_Base":           0.15897,
        "ThreeGeneration_LowHigh_Aggressive": 0.05132,
    }

    for key, r in net_results["families"].items():
        r2_net      = r["ratio2_net"]
        split       = r["split"]
        if key in ADP_BASELINE:
            baseline    = ADP_BASELINE[key]
            label       = "ADP"
        elif key in myopic_results:
            baseline    = myopic_results[key]["ratio2_myopic"]
            label       = "myopic"
        else:
            baseline    = None
            label       = ""
        baseline_s = f"{baseline:.6f} ({label})" if baseline else "N/A"
        better_s   = f"{baseline/r2_net:.1f}x" if baseline else ""
        print(f"  {key:<42} {split:<6} {r2_net:>11.6f} {baseline_s:>17} {better_s:>7}")

    out = HERE / "results" / "myopic_baseline.json"
    out.write_text(json.dumps(myopic_results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
