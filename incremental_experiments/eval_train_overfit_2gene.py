"""Overfitting check: evaluate existing 2-gene checkpoints (E0/E1/E2/E4) on
their own TRAIN families (Trio, Nuclear) instead of TEST (ThreeGeneration) --
same evaluate_rollout function each rung already uses for TEST_CONFIGS, just
pointed at TRAIN_CONFIGS instead. No new eval logic, just a different config
list, so the resulting ratio2 is directly comparable to each rung's already-
published TEST ratio2. Large train-vs-test gap = overfitting signature.

2-gene only -- 2-gene data was never affected by the 3-gene allele-frequency
bug, so this is valid to run against the current checkpoints without waiting
for retraining.

Usage:
    python eval_train_overfit_2gene.py --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR = EXP_ROOT / "q_learning"
FIXING = ROOT / "fixing_gnn_q"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(FIXING))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    seed = args.seed
    dev = torch.device(args.device)

    tg = _load_module("two_gene_run_overfit", EXP_ROOT / "two_gene" / "run.py")

    struct_cache, edge_cache = {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam] = tg.build_edge_index(pedigree, individuals)

    def log(msg=""):
        print(msg, flush=True)

    e0_gnn_mod = _load_module("e0_gnn_overfit", Q_DIR / "two_gene_gnn_q.py")
    e0_mlp_mod = _load_module("e0_mlp_overfit", Q_DIR / "two_gene_mlp_q.py")
    e1_gnn_mod = _load_module("e1_gnn_overfit", HERE / "e1_train_two_gene_gnn_q.py")
    e1_mlp_mod = _load_module("e1_mlp_overfit", HERE / "e1_train_two_gene_mlp_q.py")
    e2_gnn_mod = _load_module("e2_gnn_overfit", FIXING / "train_two_gene_gnn_q_ce.py")
    e2_mlp_mod = _load_module("e2_mlp_overfit", FIXING / "train_two_gene_mlp_q_ce.py")
    e4_gnn_mod = _load_module("e4_gnn_overfit", HERE / "e4_train_two_gene_gnn_q.py")

    # (rung, kind, ModelClass, checkpoint_path, module_whose_evaluate_rollout_to_use)
    rungs = [
        ("E0", "gnn", e0_gnn_mod.GNNQ, Q_DIR / "results_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_2gene.pt", e0_gnn_mod),
        ("E0", "mlp", e0_mlp_mod.MLPQ, Q_DIR / "results_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_2gene.pt", e0_mlp_mod),
        ("E1", "gnn", e0_gnn_mod.GNNQ, HERE / "results" / "e1_gnn_mse_grouped_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e1_2gene.pt", e1_gnn_mod),
        ("E1", "mlp", e0_mlp_mod.MLPQ, HERE / "results" / "e1_mlp_mse_grouped_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_e1_2gene.pt", e1_mlp_mod),
        ("E2", "gnn", e0_gnn_mod.GNNQ, FIXING / "results" / "gnn_q_ce_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_ce_2gene.pt", e2_gnn_mod),
        ("E2", "mlp", e0_mlp_mod.MLPQ, FIXING / "results" / "mlp_q_ce_2gene" / "seed_runs" / f"seed{seed}" / "mlp_q_ce_2gene.pt", e2_mlp_mod),
        ("E4", "gnn", e4_gnn_mod.GNNQBidir, HERE / "results" / "e4_gnn_bidir_2gene" / "seed_runs" / f"seed{seed}" / "gnn_q_e4_2gene.pt", e4_gnn_mod),
    ]

    results = {}
    for rung, kind, ModelClass, ckpt_path, mod in rungs:
        if not ckpt_path.exists():
            log(f"[SKIP] {rung}-{kind}: missing {ckpt_path}")
            continue
        model = ModelClass().to(dev)
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
        model.eval()
        log(f"\n=== {rung} {kind} (seed{seed}) train-family eval ===")
        if kind == "gnn":
            _, avg_train = mod.evaluate_rollout(model, tg.TRAIN_CONFIGS, struct_cache, edge_cache, dev, log)
        else:
            _, avg_train = mod.evaluate_rollout(model, tg.TRAIN_CONFIGS, struct_cache, dev, log)
        results[f"{rung}_{kind}"] = {"ratio2_train": avg_train}
        log(f"  TRAIN avg ratio2 = {avg_train:.4f}")

    out = HERE / "results" / f"overfit_2gene_seed{seed}.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
