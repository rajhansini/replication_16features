"""Consolidated 3-gene ground truth: exact DP (V_root, V_stop_root, root
action) and Kanix's canonical myopic policy (genetic_dp.policy.myopic
.evaluate_myopic_policy, built on genetic_dp.policy.baselines.myopic_greedy
-- zero-lookahead), for all 48 configs in the consolidated train/test table:

    TRAIN: Trio, Nuclear             x 6 regimes x {Base, Aggressive} = 24
    TEST:  ThreeGeneration, Extended x 6 regimes x {Base, Aggressive} = 24

Mirrors build_2gene_ground_truth.py's logic exactly (same evaluate_myopic_policy
call, same belief-wrapping adapter -- verified correct and cross-checked
against Kanix's own belief-map builder to 0.00e+00 diff earlier this session).

Cost is NOT uniform: Trio/Nuclear/ThreeGeneration read from the already-built
cache (ground-up-experiments/step9_gnn_3gene/results/cache/, cheap, seconds
each). Extended has no cache -- building it from scratch costs ~27min/~100GB
PER CONFIG (already proven twice this session for E4 and E6 evaluation), so
this script runs one config at a time via --family/--regime/--preset for the
SLURM per-config split, same pattern as log_extended_fourway.py. No seed
loop needed here (unlike the fourway logs) -- DP and myopic don't depend on
any trained model checkpoint, so it's 12 Extended tasks, not 36.

Usage:
    python build_3gene_ground_truth.py                          # all 36 cheap configs
    python build_3gene_ground_truth.py --family Extended --regime LowHigh --preset Base
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_multigene_dataset
from genetic_dp.policy.myopic import evaluate_myopic_policy
from genetic_dp.exact_dp.utils import GENOTYPE_STATES

sys.path.insert(0, str(EXPERIMENTS / "step9_gnn_3gene"))
from build_datasets import ALLELE_FREQS as ALLELE_FREQ_REGIMES  # noqa: E402 -- single source of truth, not a hand copy

GENES = ("GeneA", "GeneB", "GeneC")

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration", "Extended"]
PRESETS        = ["Base", "Aggressive"]
CACHED_FAMILIES = {"Trio", "Nuclear", "ThreeGeneration"}  # have step9_gnn_3gene pickles
CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

OUT_DIR = ROOT / "fresh_dataset" / "3gene"


def load_ds(fam, reg, pre):
    if fam in CACHED_FAMILIES:
        key = f"{fam}_{reg}_{pre}_3gene"
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            return pickle.load(f)
    # Extended -- no cache, build fresh (expensive)
    return build_multigene_dataset(
        family_label=fam,
        allele_freqs=ALLELE_FREQ_REGIMES[reg],
        preset_label=pre,
        genes=GENES,
    )


def ground_truth_for_config(ds, log=print):
    """Kanix's canonical myopic policy, same call as build_2gene_ground_truth.py's
    ground_truth_for_config -- identical logic, 3-gene ds instead of 2-gene."""
    config = ds["config"]
    belief_wrapped = {s: e if isinstance(e, tuple) else (e, None) for s, e in ds["belief"].items()}

    myopic_result = evaluate_myopic_policy(
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
    L = myopic_result.root_value
    V_root = ds["V_root"]
    V_stop = ds["V_stop_root"]
    denom = V_root - V_stop
    ratio2 = (V_root - L) / denom if abs(denom) > 1e-12 else 0.0

    myopic_root_action = myopic_result.policy.get(frozenset())
    dp_root_action = ds["policy_dp"].get(frozenset())

    return {
        "V_root": float(V_root),
        "V_stop_root": float(V_stop),
        "myopic_L": float(L),
        "myopic_ratio2": float(ratio2),
        "myopic_root_action": [myopic_root_action[0], myopic_root_action[1]] if myopic_root_action else None,
        "dp_root_action": [dp_root_action[0], dp_root_action[1]] if dp_root_action else None,
        "root_action_match": (
            myopic_root_action[0] == dp_root_action[0] and myopic_root_action[1] == dp_root_action[1]
        ) if myopic_root_action and dp_root_action else None,
    }


def run_one(fam, reg, pre, log):
    split = "train" if fam in TRAIN_FAMILIES else "test"
    key = f"{fam}_{reg}_{pre}_3gene"
    log(f"\n{'─'*50}")
    log(f"[CONFIG] {key}  ({split}, {'cached' if fam in CACHED_FAMILIES else 'fresh-build'})")
    t0 = time.time()

    ds = load_ds(fam, reg, pre)
    log(f"  states={len(ds['states']):,}  V*(root)={ds['V_root']:.4f}  V_stop(root)={ds['V_stop_root']:.4f}")

    gt = ground_truth_for_config(ds, log=log)
    gt["split"] = split
    log(f"  myopic: L={gt['myopic_L']:.4f}  ratio2={gt['myopic_ratio2']:.4f}  "
        f"root_action={gt['myopic_root_action']}  dp_root_action={gt['dp_root_action']}  "
        f"match={gt['root_action_match']}  ({time.time()-t0:.0f}s)")
    return key, gt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default=None, choices=[None, "Trio", "Nuclear", "ThreeGeneration", "Extended"])
    p.add_argument("--regime", default=None, choices=[None] + list(ALLELE_FREQ_REGIMES))
    p.add_argument("--preset", default=None, choices=[None, "Base", "Aggressive"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

    if args.family is not None:
        assert args.regime is not None and args.preset is not None, \
            "--family requires --regime and --preset (single-config mode)"
        log_path = OUT_DIR / "configs" / f"{args.family}_{args.regime}_{args.preset}.log"
        log_f = open(log_path, "w")
        def log(msg=""):
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()
        log(f"[3gene ground truth, single config] {datetime.now().isoformat()}")

        key, gt = run_one(args.family, args.regime, args.preset, log)
        (OUT_DIR / "configs" / f"{key}.json").write_text(json.dumps(gt, indent=2))
        log(f"saved -> {OUT_DIR / 'configs' / f'{key}.json'}")
        log_f.close()
        return

    # Loop mode: all configs NOT requiring a fresh build (Trio, Nuclear, ThreeGeneration)
    log_f = open(OUT_DIR / "ground_truth_3gene_cheap.log", "w")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[3gene ground truth, cheap configs] {datetime.now().isoformat()}")
    configs = [
        (fam, reg, pre)
        for fam in TRAIN_FAMILIES + ["ThreeGeneration"]
        for reg in ALLELE_FREQ_REGIMES
        for pre in PRESETS
    ]
    log(f"configs: {len(configs)}")

    results = {}
    for fam, reg, pre in configs:
        key, gt = run_one(fam, reg, pre, log)
        results[key] = gt
        (OUT_DIR / "ground_truth_3gene_cheap.json").write_text(json.dumps(results, indent=2))

    train_ratio2 = np.mean([r["myopic_ratio2"] for r in results.values() if r["split"] == "train"])
    log(f"\nTRAIN (n={sum(1 for r in results.values() if r['split']=='train')}): avg myopic ratio2 = {train_ratio2:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
