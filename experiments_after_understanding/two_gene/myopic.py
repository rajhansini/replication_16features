"""True myopic baseline for 2-gene ThreeGeneration.

Uses Kanix's canonical myopic policy directly — genetic_dp.policy.myopic
.evaluate_myopic_policy, built on genetic_dp.policy.baselines.myopic_greedy
(zero-lookahead: argmax_i immediate r_test(i,s) vs stop). No local
reimplementation of the policy or its rollout value — see
incremental_experiments/verify_myopic_one.py for the from-scratch
cross-validated version of this same call against Kanix's own belief-map
builder; this file reuses two_gene/run.py's belief map (build_two_gene_dataset),
already confirmed bit-identical to Kanix's for V_root/V_stop_root.
"""
from __future__ import annotations

import json
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

from shared.data_gen import build_two_gene_dataset
from genetic_dp.policy.myopic import evaluate_myopic_policy
from genetic_dp.exact_dp.utils import GENOTYPE_STATES

GENES = ("GeneA", "GeneB")

ALLELE_FREQ_REGIMES = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12},
}

TEST_CONFIGS = [
    ("ThreeGeneration", reg, pre)
    for reg in ALLELE_FREQ_REGIMES
    for pre in ["Base", "Aggressive"]
]


def myopic_policy_value(ds, log=print):
    """Kanix's canonical myopic policy, evaluated to its true rollout value.
    `infer=None` and `state_pool=ds["belief"].keys()` are safe because
    build_two_gene_dataset's belief map already covers every reachable state
    (full exact-DP solve), so evaluate_myopic_policy never needs to expand
    a new state on the fly.

    build_two_gene_dataset stores belief[state] as a bare InferenceResult;
    genetic_dp.policy.evaluator.exact_value_under_policy unpacks belief[state]
    as a (posterior, z_post) tuple (Kanix's own belief-map convention). The
    z_post half is discarded unused in that unpacking, so wrapping as
    (entry, None) is a pure format adapter, not a policy reimplementation."""
    config = ds["config"]
    belief_wrapped = {s: e if isinstance(e, tuple) else (e, None) for s, e in ds["belief"].items()}
    result = evaluate_myopic_policy(
        belief=belief_wrapped,
        individuals=ds["individuals"],
        gen_states=GENOTYPE_STATES,
        infer=None,
        a=config.a, b=config.b, c=config.c, delta=config.delta,
        fixed_cost=config.fixed_cost, variable_cost=config.variable_cost,
        genes=GENES,
        a_gene=config.a_gene, b_gene=config.b_gene,
        c_gene=config.c_gene, delta_gene=config.delta_gene,
        state_pool=ds["belief"].keys(),
    )
    L = result.root_value
    V_root = ds["V_root"]
    V_stop = ds["V_stop_root"]
    denom = V_root - V_stop
    ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0
    return float(ratio2), float(L)


def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "myopic.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[MYOPIC 2-GENE, Kanix's canonical myopic_greedy] {datetime.now().isoformat()}")

    results = {}
    out = results_dir / "myopic_results.json"

    for fam, reg, pre in TEST_CONFIGS:
        key = f"{fam}_{reg}_{pre}_2gene"

        log(f"\n{'─'*50}")
        log(f"[CONFIG] {key}")
        t0 = time.time()

        ds = build_two_gene_dataset(
            family_label=fam,
            allele_freqs=ALLELE_FREQ_REGIMES[reg],
            preset_label=pre,
            genes=GENES,
        )
        log(f"  states={len(ds['states'])}  V*(root)={ds['V_root']:.4f}  V_stop(root)={ds['V_stop_root']:.4f}")

        ratio2, L = myopic_policy_value(ds, log=log)
        log(f"  ratio2={ratio2:.4f}  L={L:.4f}  ({time.time()-t0:.0f}s)")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}
        out.write_text(json.dumps(results, indent=2))

    avg = np.mean([r["ratio2"] for r in results.values()]) if results else float("nan")
    log(f"\n{'='*60}")
    log(f"MYOPIC 2-gene avg ratio2 = {avg:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
