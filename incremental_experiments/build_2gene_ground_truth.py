"""Consolidated 2-gene ground truth: exact DP (V_root, V_stop_root, root
action) and Kanix's canonical myopic policy (genetic_dp.policy.myopic
.evaluate_myopic_policy, built on genetic_dp.policy.baselines.myopic_greedy
-- zero-lookahead), for ALL 48 configs in the consolidated train/test table:

    TRAIN: Trio, Nuclear       x 6 regimes x {Base, Aggressive} = 24
    TEST:  ThreeGeneration, Extended x 6 regimes x {Base, Aggressive} = 24

One script, one code path, so train-side and test-side ground truth are
never at risk of subtly diverging the way two separately-written scripts
could. Extends two_gene/myopic.py's already-validated
myopic_policy_value() (which itself calls Kanix's real functions, no local
reimplementation) to all 4 families instead of just ThreeGeneration.

Usage:
    python build_2gene_ground_truth.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
TWO_GENE    = ROOT / "experiments_after_understanding" / "two_gene"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(TWO_GENE))

from shared.data_gen import build_two_gene_dataset
from genetic_dp.policy.myopic import evaluate_myopic_policy
from genetic_dp.policy.baselines import myopic_greedy
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

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration", "Extended"]
PRESETS        = ["Base", "Aggressive"]

ALL_CONFIGS = [
    (fam, reg, pre, "train" if fam in TRAIN_FAMILIES else "test")
    for fam in TRAIN_FAMILIES + TEST_FAMILIES
    for reg in ALLELE_FREQ_REGIMES
    for pre in PRESETS
]


def ground_truth_for_config(ds, log=print):
    """Kanix's canonical myopic policy (myopic_policy_value, unchanged logic
    from two_gene/myopic.py) plus DP's own root action for direct comparison."""
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


def main():
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "ground_truth_2gene.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[2-GENE GROUND TRUTH, Kanix's exact DP + canonical myopic_greedy] {datetime.now().isoformat()}")
    log(f"configs: {len(ALL_CONFIGS)} ({len(TRAIN_FAMILIES)} train families + {len(TEST_FAMILIES)} test families x "
        f"{len(ALLELE_FREQ_REGIMES)} regimes x {len(PRESETS)} presets)")

    results = {}
    out = results_dir / "ground_truth_2gene.json"

    for fam, reg, pre, split in ALL_CONFIGS:
        key = f"{fam}_{reg}_{pre}_2gene"
        log(f"\n{'─'*50}")
        log(f"[CONFIG] {key}  ({split})")
        t0 = time.time()

        ds = build_two_gene_dataset(
            family_label=fam,
            allele_freqs=ALLELE_FREQ_REGIMES[reg],
            preset_label=pre,
            genes=GENES,
        )
        log(f"  states={len(ds['states'])}  V*(root)={ds['V_root']:.4f}  V_stop(root)={ds['V_stop_root']:.4f}")

        gt = ground_truth_for_config(ds, log=log)
        gt["split"] = split
        log(f"  myopic: L={gt['myopic_L']:.4f}  ratio2={gt['myopic_ratio2']:.4f}  "
            f"root_action={gt['myopic_root_action']}  dp_root_action={gt['dp_root_action']}  "
            f"match={gt['root_action_match']}  ({time.time()-t0:.0f}s)")
        results[key] = gt
        out.write_text(json.dumps(results, indent=2))

    train_ratio2 = np.mean([r["myopic_ratio2"] for r in results.values() if r["split"] == "train"])
    test_ratio2  = np.mean([r["myopic_ratio2"] for r in results.values() if r["split"] == "test"])
    train_root_match = np.mean([r["root_action_match"] for r in results.values() if r["split"] == "train"])
    test_root_match  = np.mean([r["root_action_match"] for r in results.values() if r["split"] == "test"])
    log(f"\n{'='*60}")
    log(f"TRAIN (n={sum(1 for r in results.values() if r['split']=='train')}): "
        f"avg myopic ratio2 = {train_ratio2:.4f}  root-match vs DP = {train_root_match:.3f}")
    log(f"TEST  (n={sum(1 for r in results.values() if r['split']=='test')}): "
        f"avg myopic ratio2 = {test_ratio2:.4f}  root-match vs DP = {test_root_match:.3f}")
    log_f.close()


if __name__ == "__main__":
    main()
