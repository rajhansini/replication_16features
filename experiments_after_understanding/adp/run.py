"""ADP baseline on ThreeGeneration (3-gene) — 16 ABCD features + LP.

Adapted from ground-up-experiments/step4_all_families/adp_baseline.py.
Runs on all 12 ThreeGeneration test configs from the step9 3-gene cache.
No V* supervision — ADP derives theta via Bellman LP constraints only.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE         = Path(__file__).resolve().parent
ROOT         = HERE.parent.parent
EXPERIMENTS  = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from genetic_dp.exact_dp.utils  import GENOTYPE_STATES, lift_tuple_posteriors_to_genes
from genetic_dp.models.belief   import InferenceResult
from genetic_dp.models.reward   import r_reward, r_reward_test
from genetic_dp.optimisation.myopic_adp import build_state_features, ABCD16_DIRECT_FEATURES

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

ALLELE_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]
TEST_KEYS = [
    f"ThreeGeneration_{reg}_{pre}_3gene"
    for reg in ALLELE_REGIMES
    for pre in ["Base", "Aggressive"]
]


# ── helpers ────────────────────────────────────────────────────────────────────

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
    feat_dict = build_state_features(
        state,
        belief=ds["belief"],
        individuals=ds["individuals"],
        pedigree=ds["pedigree"],
        genes=ds.get("genes"),
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


# ── ADP LP ────────────────────────────────────────────────────────────────────

def solve_adp(ds, log=print):
    """Solve Bellman LP for theta. Never sees V* labels."""
    belief      = ds["belief"]
    individuals = ds["individuals"]
    V_star      = ds["V_star"]
    k           = len(ABCD16_DIRECT_FEATURES)

    states = [s for s in V_star if s in belief]
    n      = len(states)
    log(f"    Building features: {n} states × {k} features...")
    t0 = time.time()

    phi = {}
    for idx, s in enumerate(states):
        phi[s] = compute_features(s, ds)
        if (idx + 1) % 100000 == 0:
            log(f"      {idx+1}/{n} states  ({time.time()-t0:.0f}s)")

    log(f"    Features done in {time.time()-t0:.1f}s. Building LP constraints...")

    A_rows, b_rows = [], []
    for s in states:
        tested = {i for i, _ in s}

        # stop constraint: phi(s)·θ >= v_stop(s)  →  -phi(s)·θ <= -v_stop
        v_stop = compute_v_stop(s, ds)
        A_rows.append(-phi[s])
        b_rows.append(-v_stop)

        # test constraints
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
                if next_s in phi:
                    exp_phi_next += prob_g * phi[next_s]

            A_rows.append(-(phi[s] - exp_phi_next))
            b_rows.append(-r_i)

    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_rows, dtype=np.float64)
    c    = np.sum([phi[s] for s in states], axis=0)

    log(f"    LP: {A_ub.shape[0]} constraints × {k} vars — solving...")
    t1 = time.time()
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs",
                     options={"disp": False, "time_limit": 600})
    log(f"    LP done in {time.time()-t1:.1f}s  status={result.status}")

    if result.status != 0:
        log(f"    WARNING: LP solver status={result.status} ({result.message})")

    return result.x, phi


# ── greedy rollout ─────────────────────────────────────────────────────────────

def greedy_policy_value(theta, phi, ds, log=print) -> float:
    """Simulate ADP greedy policy, return its true expected value L."""
    belief      = ds["belief"]
    individuals = ds["individuals"]

    memo = {}

    def value_at(state) -> float:
        if state in memo:
            return memo[state]
        tested  = {i for i, _ in state}
        v_stop  = compute_v_stop(state, ds)

        if len(tested) == len(individuals):
            memo[state] = 0.0
            return 0.0

        entry   = belief[state]
        best_q  = v_stop
        best_i  = None

        for i in individuals:
            if i in tested:
                continue
            r_i   = compute_r_test(i, state, ds)
            pmf_i = _tuple_pmf(entry, i)
            exp_v = 0.0
            for g, prob_g in pmf_i.items():
                if prob_g <= 1e-12:
                    continue
                next_s = frozenset(state | {(i, g)})
                if next_s in phi:
                    exp_v += prob_g * float(phi[next_s] @ theta)
            q_i = r_i + exp_v
            if q_i > best_q:
                best_q = q_i
                best_i = i

        if best_i is None:
            memo[state] = v_stop
            return v_stop

        r_best   = compute_r_test(best_i, state, ds)
        pmf_best = _tuple_pmf(entry, best_i)
        exp_true = 0.0
        for g, prob_g in pmf_best.items():
            if prob_g <= 1e-12:
                continue
            next_s = frozenset(state | {(best_i, g)})
            if next_s not in belief:
                continue
            exp_true += prob_g * value_at(next_s)

        result      = r_best + exp_true
        memo[state] = result
        return result

    return value_at(frozenset())


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "run.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[ADP] {datetime.now().isoformat()}")
    log(f"features: {ABCD16_DIRECT_FEATURES}")
    log(f"n_features: {len(ABCD16_DIRECT_FEATURES)}")
    log(f"configs: {len(TEST_KEYS)}")

    results = {}

    # skip configs already in results (allows resuming a partial run)
    out = results_dir / "results.json"
    if out.exists():
        existing = json.loads(out.read_text())
        if existing:
            log(f"Resuming — already have {len(existing)} configs: {list(existing.keys())}")
            results = existing

    for key in TEST_KEYS:
        if key in results:
            log(f"  SKIP {key} — already done")
            continue
        log(f"\n{'─'*50}")
        log(f"[CONFIG] {key}")
        t0 = time.time()

        pkl_path = CACHE_DIR / f"{key}.pkl"
        if not pkl_path.exists():
            log(f"  SKIP — pkl not found: {pkl_path}")
            continue

        log(f"  Loading pkl...")
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)

        V_root      = ds["V_root"]
        V_stop_root = ds["V_stop_root"]
        log(f"  V*(root)={V_root:.6f}  V_stop(root)={V_stop_root:.6f}")
        log(f"  states: {len(ds['V_star']):,}")

        theta, phi = solve_adp(ds, log=log)
        log(f"  theta = {np.round(theta, 4).tolist()}")

        L      = greedy_policy_value(theta, phi, ds, log=log)
        denom  = V_root - V_stop_root
        ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

        log(f"  L(ADP)={L:.6f}  ratio2={ratio2:.6f}  ({time.time()-t0:.0f}s total)")
        results[key] = {
            "ratio2": float(ratio2),
            "L":      float(L),
            "V_root": float(V_root),
            "theta":  theta.tolist(),
        }

        # save incrementally so we have partial results if it crashes
        out = results_dir / "results.json"
        out.write_text(json.dumps(results, indent=2))

    avg = float(np.mean([r["ratio2"] for r in results.values()])) if results else float("nan")
    log(f"\n{'='*60}")
    log(f"ADP avg ratio2 = {avg:.4f}  ({len(results)}/{len(TEST_KEYS)} configs)")
    log_f.close()


if __name__ == "__main__":
    main()
