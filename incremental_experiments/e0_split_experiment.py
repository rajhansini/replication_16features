"""E0 train/test split experiments -- root-cause test for the GNN-Q 2-gene
overfitting finding (GNN-Q shows a consistent TEST-TRAIN ratio2 gap across 3
seeds; MLP-Q does not -- see incremental_experiments/results/overfit_2gene_*.json).

Reuses E0's exact model classes and training/eval code UNCHANGED --
GNNQ/MLPQ, build_config, train_one_epoch, precompute_qhat, evaluate_rollout,
build_qsa_index -- imported directly from two_gene_gnn_q.py / two_gene_mlp_q.py
(same files E0's real 2-gene run uses), not reimplemented. The only thing
this script changes is WHICH families are TRAIN vs TEST -- the standard E0
run hardcodes TRAIN_FAMILIES=[Trio, Nuclear], TEST_FAMILIES=[ThreeGeneration]
via two_gene/run.py; this script takes those as arguments instead, to
isolate whether the overfitting is about topology diversity in general or
something specific to ThreeGeneration.

Experiment #3 (rotate): TRAIN=[Trio, ThreeGeneration]  TEST=[Nuclear]
    -- swaps which small family is held out. If GNN-Q still overfits here,
    it's not "something ThreeGeneration-specific", it's topology diversity.

Experiment #4 (add): TRAIN=[Trio, Nuclear, ThreeGeneration]  TEST=[Extended]
    -- 3 training topologies instead of 2, tests a 4th (Extended, 2-gene
    build is cheap -- no cache needed, same build_two_gene_dataset path).
    If the TEST-TRAIN gap shrinks, topology count is a real lever.

Each run trains a fresh model from scratch (own checkpoint dir, never
touches E0's real results_2gene/ checkpoints) and reports the same
TRAIN-config vs TEST-config ratio2 gap used in the 3-seed overfit check.

Usage:
    python e0_split_experiment.py --exp rotate --kind gnn --seed 0
    python e0_split_experiment.py --exp add    --kind mlp --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
Q_DIR    = ROOT / "experiments_after_understanding" / "q_learning"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground-up-experiments"))
sys.path.insert(0, str(Q_DIR))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gnn_mod = _load_module("split_gnn_mod", Q_DIR / "two_gene_gnn_q.py")
mlp_mod = _load_module("split_mlp_mod", Q_DIR / "two_gene_mlp_q.py")
tg = gnn_mod.tg  # shared two_gene/run.py module (ALLELE_FREQ_REGIMES, FAMILY_CASES, etc.)

RESULTS_ROOT = HERE / "results"

EXPERIMENTS = {
    "rotate": {"train": ["Trio", "ThreeGeneration"], "test": ["Nuclear"]},
    "add":    {"train": ["Trio", "Nuclear", "ThreeGeneration"], "test": ["Extended"]},
}


def build_configs(families):
    return [
        (fam, reg, pre)
        for fam in families
        for reg in tg.ALLELE_FREQ_REGIMES
        for pre in tg.PRESETS_LIST
    ]


def run_gnn(train_families, test_families, results_dir, epochs, batch_size, mode, seed, device, log):
    dev = torch.device(device)
    train_configs = build_configs(train_families)
    test_configs = build_configs(test_families)

    struct_cache, edge_cache = {}, {}
    for fam in train_families + test_families:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam] = tg.build_edge_index(pedigree, individuals)

    model = gnn_mod.GNNQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    ckpt_path = results_dir / "checkpoint_gnn.pt"
    model_path = results_dir / "gnn_q_split.pt"

    if mode in ("train", "both"):
        log(f"\n[1] Generating {len(train_configs)} train configs ({train_families}) + building (s,a,Q*) tensors...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}
        per_config = []
        for fam, reg, pre in train_configs:
            ds = gnn_mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            s_idx, a_idx, y = gnn_mod.build_qsa_index(ds, device=dev, cache_key=key)
            per_config.append((fam, base["nf"], base["gf"], s_idx, a_idx, y))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows")
        total_rows = sum(len(y) for *_, y in per_config)
        log(f"    total (s,a) rows: {total_rows:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = 1
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=dev)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            log(f"\n[RESUME] found checkpoint at epoch {ckpt['epoch']}, resuming from epoch {start_epoch}")

        log(f"\n[2] Training... (epochs {start_epoch}-{epochs})")
        t0 = time.time()
        for ep in range(start_epoch, epochs + 1):
            ep_loss, ep_count = 0.0, 0
            for fam, nf_c, gf_c, s_idx_c, a_idx_c, y_c in per_config:
                ei_c = edge_index_t[fam]
                loss = gnn_mod.train_one_epoch(model, nf_c, ei_c, gf_c, s_idx_c, a_idx_c, y_c, opt, batch_size, dev)
                ep_loss += loss * len(y_c)
                ep_count += len(y_c)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  mse={ep_loss/ep_count:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), model_path)
        log(f"    saved -> {model_path}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(model_path, map_location=dev))
        log(f"\n[3] Eval on TRAIN configs ({train_families}) -- overfit check...")
        train_results, train_avg = gnn_mod.evaluate_rollout(model, train_configs, struct_cache, edge_cache, dev, log)
        log(f"\n[4] Eval on TEST configs ({test_families})...")
        test_results, test_avg = gnn_mod.evaluate_rollout(model, test_configs, struct_cache, edge_cache, dev, log)
        (results_dir / "results_gnn.json").write_text(json.dumps(
            {"train": train_results, "test": test_results,
             "train_avg_ratio2": train_avg, "test_avg_ratio2": test_avg,
             "gap": test_avg - train_avg}, indent=2))
        log(f"\n  TRAIN avg ratio2 = {train_avg:.4f}   TEST avg ratio2 = {test_avg:.4f}   gap = {test_avg - train_avg:+.4f}")
        return train_avg, test_avg


def run_mlp(train_families, test_families, results_dir, epochs, batch_size, mode, seed, device, log):
    dev = torch.device(device)
    train_configs = build_configs(train_families)
    test_configs = build_configs(test_families)

    struct_cache = {}
    for fam in train_families + test_families:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)

    model = mlp_mod.MLPQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    ckpt_path = results_dir / "checkpoint_mlp.pt"
    model_path = results_dir / "mlp_q_split.pt"

    if mode in ("train", "both"):
        log(f"\n[1] Generating {len(train_configs)} train configs ({train_families}) + building (s,a,Q*) tensors...")
        per_config = []
        for fam, reg, pre in train_configs:
            ds = mlp_mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            s_idx, a_idx, y = mlp_mod.build_qsa_index(ds, device=dev, cache_key=key)
            per_config.append((base["nf"], base["gf"], s_idx, a_idx, y))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows")
        total_rows = sum(len(y) for *_, y in per_config)
        log(f"    total (s,a) rows: {total_rows:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = 1
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=dev)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            log(f"\n[RESUME] found checkpoint at epoch {ckpt['epoch']}, resuming from epoch {start_epoch}")

        log(f"\n[2] Training... (epochs {start_epoch}-{epochs})")
        t0 = time.time()
        for ep in range(start_epoch, epochs + 1):
            ep_loss, ep_count = 0.0, 0
            for nf_c, gf_c, s_idx_c, a_idx_c, y_c in per_config:
                loss = mlp_mod.train_one_epoch(model, nf_c, gf_c, s_idx_c, a_idx_c, y_c, opt, batch_size, dev)
                ep_loss += loss * len(y_c)
                ep_count += len(y_c)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  mse={ep_loss/ep_count:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), model_path)
        log(f"    saved -> {model_path}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(model_path, map_location=dev))
        log(f"\n[3] Eval on TRAIN configs ({train_families}) -- overfit check...")
        train_results, train_avg = mlp_mod.evaluate_rollout(model, train_configs, struct_cache, dev, log)
        log(f"\n[4] Eval on TEST configs ({test_families})...")
        test_results, test_avg = mlp_mod.evaluate_rollout(model, test_configs, struct_cache, dev, log)
        (results_dir / "results.json").write_text(json.dumps(
            {"train": train_results, "test": test_results,
             "train_avg_ratio2": train_avg, "test_avg_ratio2": test_avg,
             "gap": test_avg - train_avg}, indent=2))
        log(f"\n  TRAIN avg ratio2 = {train_avg:.4f}   TEST avg ratio2 = {test_avg:.4f}   gap = {test_avg - train_avg:+.4f}")
        return train_avg, test_avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    p.add_argument("--kind", required=True, choices=["gnn", "mlp"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = EXPERIMENTS[args.exp]
    train_families, test_families = spec["train"], spec["test"]

    results_dir = RESULTS_ROOT / f"e0_split_{args.exp}" / args.kind / "seed_runs" / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(results_dir / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}\n[E0-SPLIT exp={args.exp} kind={args.kind}] {datetime.now().isoformat()}  "
        f"mode={args.mode}  epochs={args.epochs}  seed={args.seed}")
    log(f"TRAIN={train_families}  TEST={test_families}")

    if args.kind == "gnn":
        run_gnn(train_families, test_families, results_dir, args.epochs, args.batch_size, args.mode, args.seed, args.device, log)
    else:
        run_mlp(train_families, test_families, results_dir, args.epochs, args.batch_size, args.mode, args.seed, args.device, log)

    log_f.close()


if __name__ == "__main__":
    main()
