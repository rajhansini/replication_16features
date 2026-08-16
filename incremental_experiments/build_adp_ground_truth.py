"""ADP ground truth: Kanix's canonical dual-DP approximate solver
(genetic_dp.optimisation.dual_dp.solve_dual_dp_with_domain, invoked through
his own orchestrator genetic_dp.experiments.core.run_and_compare_solvers --
exactly the call replicate.py makes, same direct16_env() locked environment)
for every one of THIS PROJECT's 96 configs (4 families x 6 regimes x 2
presets x 2 gene counts).

Reuses the exact same pedigree/config objects already used for the myopic/
exact-DP ground truth (build_two_gene_dataset / build_multigene_dataset) --
not a reimplementation, the SAME dataset builder, just also handed to
Kanix's ADP solver in addition to what it already does (exact DP + myopic).

One config per process (SLURM array task) -- ADP is expensive (~5-10min per
config based on this session's earlier checks against Kanix's own benchmark
rows) and needs its own Gurobi license (GRB_LICENSE_FILE set below to the
real WLS license, not the size-limited one bundled with gurobipy).

Usage:
    python build_adp_ground_truth.py --genes 2 --family Trio --regime LowHigh --preset Base
    python build_adp_ground_truth.py --genes 3 --family Extended --regime MixedB --preset Aggressive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRB_LICENSE_FILE", str(Path.home() / ".gurobi" / "gurobi.lic"))
os.environ.setdefault("EXACT_DP_CACHE_IN_MEMORY_ONLY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_two_gene_dataset, build_multigene_dataset  # noqa: E402
from genetic_dp.experiments.core import run_and_compare_solvers  # noqa: E402
from replicate import direct16_env, BENCHMARK_ENV  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from run_multigene_ratio45_new_settings import _extract_metrics  # noqa: E402 -- Kanix's own metric extraction, not a reimplementation

GENES_2 = ("GeneA", "GeneB")
GENES_3 = ("GeneA", "GeneB", "GeneC")

ALLELE_FREQ_REGIMES_2GENE = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12},
}
sys.path.insert(0, str(EXPERIMENTS / "step9_gnn_3gene"))
from build_datasets import ALLELE_FREQS as ALLELE_FREQ_REGIMES_3GENE  # noqa: E402 -- single source of truth

FAMILIES = ["Trio", "Nuclear", "ThreeGeneration", "Extended"]
TRAIN_FAMILIES = ["Trio", "Nuclear"]
PRESETS = ["Base", "Aggressive"]

OUT_DIR = ROOT / "fresh_dataset" / "adp"


def run_one(genes_n, fam, reg, pre, log):
    key = f"{fam}_{reg}_{pre}_{genes_n}gene"
    split = "train" if fam in TRAIN_FAMILIES else "test"
    log(f"[CONFIG] {key} ({split})")

    if genes_n == 2:
        ds = build_two_gene_dataset(
            family_label=fam, allele_freqs=ALLELE_FREQ_REGIMES_2GENE[reg],
            preset_label=pre, genes=GENES_2,
        )
    else:
        ds = build_multigene_dataset(
            family_label=fam, allele_freqs=ALLELE_FREQ_REGIMES_3GENE[reg],
            preset_label=pre, genes=GENES_3,
        )

    pedigree = ds["pedigree"]
    config = ds["config"]
    log(f"  states={len(ds['states']):,}  V*(root)={ds['V_root']:.6f}  V_stop(root)={ds['V_stop_root']:.6f}")

    env = dict(BENCHMARK_ENV)
    env.update(direct16_env())
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    t0 = time.time()
    try:
        raw = run_and_compare_solvers(
            pedigree, config,
            verbose=False,
            lookahead_depths=(0, 1),
            print_policies=False,
            progress_label=f"adp_gt::{key}",
            belief_parallelism=1,
            dfvr_bound=False,
            return_infer=True,
        )
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    dt = time.time() - t0

    metrics = _extract_metrics(raw)  # Kanix's own extraction -- exact_root_value, adp_phi, ratio2, ratio3, etc.

    result = {
        "genes": genes_n, "family": fam, "regime": reg, "preset": pre, "split": split,
        "V_star": metrics["exact_root_value"],
        "V_stop": metrics["stop_value"],
        "ADP_phi": metrics["adp_phi"],
        "production_policy_value": metrics["production_policy_value"],
        "production_policy_source": metrics["production_policy_source"],
        "adp_ratio2_deployed_policy": metrics["ratio2"],
        "adp_ratio3_certificate": metrics["ratio3"],
        "seconds": round(dt, 1),
    }
    log(f"  ADP: ratio2={metrics['ratio2']}  ratio3={metrics['ratio3']}  "
        f"policy_source={metrics['production_policy_source']}  ({dt:.0f}s)")
    return key, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, required=True, choices=[2, 3])
    p.add_argument("--family", required=True, choices=FAMILIES)
    p.add_argument("--regime", required=True)
    p.add_argument("--preset", required=True, choices=PRESETS)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "configs" / f"{args.genes}gene_{args.family}_{args.regime}_{args.preset}.log"
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"[adp_ground_truth] {datetime.now().isoformat()}  genes={args.genes}  "
        f"family={args.family}  regime={args.regime}  preset={args.preset}")

    key = f"{args.family}_{args.regime}_{args.preset}_{args.genes}gene"
    out_path = OUT_DIR / "configs" / f"{key}_adp.json"
    if out_path.exists():
        log(f"[SKIP] {key} already done -> {out_path} (resume: not re-solving)")
        log_f.close()
        return

    key, result = run_one(args.genes, args.family, args.regime, args.preset, log)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(out_path)  # atomic -- a killed task never leaves a half-written result file
    log(f"saved -> {out_path}")
    log_f.close()


if __name__ == "__main__":
    main()
