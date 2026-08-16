"""True myopic baseline on ThreeGeneration (3-gene).

Uses Kanix's canonical myopic policy directly — genetic_dp.policy.myopic
.evaluate_myopic_policy, built on genetic_dp.policy.baselines.myopic_greedy
(zero-lookahead: argmax_i immediate r_test(i,s) vs stop). No local
reimplementation of the policy or its rollout value — see
incremental_experiments/verify_myopic_one.py for the from-scratch
cross-validated version of this same call against Kanix's own belief-map
builder; this file reuses the cached belief map from step9_gnn_3gene's
build_multigene_dataset (same family as two_gene/run.py's belief-map path,
already confirmed bit-identical to Kanix's for V_root/V_stop_root).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from genetic_dp.exact_dp.utils import GENOTYPE_STATES
from genetic_dp.policy.myopic import evaluate_myopic_policy

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

ALLELE_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]
TEST_KEYS = [
    f"ThreeGeneration_{reg}_{pre}_3gene"
    for reg in ALLELE_REGIMES
    for pre in ["Base", "Aggressive"]
]

GENES_3 = ("GeneA", "GeneB", "GeneC")


def myopic_policy_value(ds, log=print):
    """Kanix's canonical myopic policy, evaluated to its true rollout value.
    `infer=None` and `state_pool=ds["belief"].keys()` are safe because the
    cached belief map already covers every reachable state (full exact-DP
    solve), so evaluate_myopic_policy never needs to expand a new state.

    The cached belief stores belief[state] as a bare InferenceResult;
    genetic_dp.policy.evaluator.exact_value_under_policy unpacks belief[state]
    as a (posterior, z_post) tuple (Kanix's own belief-map convention). The
    z_post half is discarded unused in that unpacking, so wrapping as
    (entry, None) is a pure format adapter, not a policy reimplementation."""
    config = ds["config"]
    genes = ds.get("genes", GENES_3)
    belief_wrapped = {s: e if isinstance(e, tuple) else (e, None) for s, e in ds["belief"].items()}
    result = evaluate_myopic_policy(
        belief=belief_wrapped,
        individuals=ds["individuals"],
        gen_states=GENOTYPE_STATES,
        infer=None,
        a=config.a, b=config.b, c=config.c, delta=config.delta,
        fixed_cost=config.fixed_cost, variable_cost=config.variable_cost,
        genes=genes,
        a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
        state_pool=ds["belief"].keys(),
    )
    return result.root_value


def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "run.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[MYOPIC, Kanix's canonical myopic_greedy] {datetime.now().isoformat()}")
    log(f"configs: {len(TEST_KEYS)}")

    results = {}
    out = results_dir / "results.json"

    for key in TEST_KEYS:
        log(f"\n{'─'*50}")
        log(f"[CONFIG] {key}")
        t0 = time.time()

        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        V_root      = ds["V_root"]
        V_stop_root = ds["V_stop_root"]
        log(f"  V*(root)={V_root:.6f}  V_stop(root)={V_stop_root:.6f}")

        L = myopic_policy_value(ds, log=log)
        denom = V_root - V_stop_root
        ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

        log(f"  L(myopic)={L:.6f}  ratio2={ratio2:.6f}  ({time.time()-t0:.0f}s)")
        results[key] = {"ratio2": float(ratio2), "L": float(L), "V_root": float(V_root)}
        out.write_text(json.dumps(results, indent=2))

    avg = float(np.mean([r["ratio2"] for r in results.values()])) if results else float("nan")
    log(f"\n{'='*60}")
    log(f"MYOPIC avg ratio2 = {avg:.4f}  ({len(results)}/{len(TEST_KEYS)} configs)")
    log_f.close()


if __name__ == "__main__":
    main()
