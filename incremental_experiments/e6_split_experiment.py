"""E6 train/test split experiments -- same rotate/add root-cause test as
e0_split_experiment.py, but on E6 (sum-pooling GNN-Q / MLP-Q, best-performing
3-gene rung) instead of E0, since E6 is the rung that actually matters.

Reuses E6's exact model classes and training/eval code UNCHANGED --
GNNQBidirSumPool, MLPQSumPool, build_config, train_one_epoch, precompute_qhat,
evaluate_rollout, build_qsa_index, build_state_groups/combined_loss -- all
imported directly from e6_train_two_gene_gnn_q.py / e6_train_two_gene_mlp_q.py
(the same files E6's real 2-gene run uses), not reimplemented. The only
change is WHICH families are TRAIN vs TEST.

Experiment #3 (rotate): TRAIN=[Trio, ThreeGeneration]  TEST=[Nuclear]
Experiment #4 (add):    TRAIN=[Trio, Nuclear, ThreeGeneration]  TEST=[Extended]

Plus full leave-one-family-out (3 training families each, "add"'s pattern,
one fold per possible held-out family -- "add" above IS the
loo_extended fold, kept under its original name for continuity):
    loo_trio:        TRAIN=[Nuclear, ThreeGeneration, Extended]  TEST=[Trio]
    loo_nuclear:      TRAIN=[Trio, ThreeGeneration, Extended]     TEST=[Nuclear]
    loo_threegen:     TRAIN=[Trio, Nuclear, Extended]             TEST=[ThreeGeneration]

Each run trains a fresh model from scratch (own checkpoint dir, never
touches E6's real results/e6_*sumpool_2gene/ checkpoints) and reports the
TRAIN-config vs TEST-config ratio2 gap, same metric as the E0 overfit check.

Usage:
    python e6_split_experiment.py --exp rotate --kind gnn --seed 0
    python e6_split_experiment.py --exp add    --kind mlp --seed 0
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

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
Q_DIR    = ROOT / "experiments_after_understanding" / "q_learning"
FIXING   = ROOT / "fixing_gnn_q"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ground-up-experiments"))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(FIXING))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gnn_mod = _load_module("split_e6_gnn_mod", HERE / "e6_train_two_gene_gnn_q.py")
mlp_mod = _load_module("split_e6_mlp_mod", HERE / "e6_train_two_gene_mlp_q.py")
tg = gnn_mod.tg  # shared two_gene/run.py module (ALLELE_FREQ_REGIMES, FAMILY_CASES, etc.)

RESULTS_ROOT = HERE / "results"

EXPERIMENTS = {
    "rotate":       {"train": ["Trio", "ThreeGeneration"], "test": ["Nuclear"]},
    "add":          {"train": ["Trio", "Nuclear", "ThreeGeneration"], "test": ["Extended"]},
    "loo_trio":     {"train": ["Nuclear", "ThreeGeneration", "Extended"], "test": ["Trio"]},
    "loo_nuclear":  {"train": ["Trio", "ThreeGeneration", "Extended"], "test": ["Nuclear"]},
    "loo_threegen": {"train": ["Trio", "Nuclear", "Extended"], "test": ["ThreeGeneration"]},
}


def build_configs(families):
    return [
        (fam, reg, pre)
        for fam in families
        for reg in tg.ALLELE_FREQ_REGIMES
        for pre in tg.PRESETS_LIST
    ]


def evaluate_rollout_cached(mod, model, configs, ds_cache, struct_cache, extra_cache, device, log,
                            partial_path=None):
    """Same as mod.evaluate_rollout, but reuses an already-solved ds from
    ds_cache (populated during training-tensor building) instead of
    re-running build_two_gene_dataset's exact-DP solve a second time for
    configs that are BOTH train configs and being evaluated here (the TRAIN
    side of the overfit check). Each ThreeGeneration/Extended exact-DP solve
    is ~50s -- with 12 configs re-solved on every one of [1]/[3]/[4], this
    duplication alone was the dominant cost. Not used for TEST configs
    (never pre-built, no savings there). mod is gnn_mod or mlp_mod (same
    q_rollout/precompute_qhat either way, just different arg lists).

    If partial_path is given, per-config results are flushed to it as they
    land and reloaded on startup, so a run killed by the SLURM wall clock
    mid-eval resumes at the next unevaluated config instead of redoing the
    whole 36-config TRAIN sweep.
    """
    import numpy as _np
    results = {}
    if partial_path is not None and partial_path.exists():
        try:
            results = json.loads(partial_path.read_text())
        except json.JSONDecodeError:  # killed mid-write -- just start this stage over
            results = {}
        if results:
            log(f"  [RESUME] {len(results)}/{len(configs)} configs already evaluated -- skipping those")
    for fam, reg, pre in configs:
        key = f"{fam}_{reg}_{pre}_2gene"
        if key in results:
            continue
        if key in ds_cache:
            ds = ds_cache[key]
        else:
            ds = mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], device)
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

        if extra_cache is not None:  # GNN: edge_cache
            edge_index_t = torch.tensor(extra_cache[fam], device=device)
            q_hat = mod.precompute_qhat(model, ds, key, edge_index_t, device)
        else:  # MLP: no edges
            q_hat = mod.precompute_qhat(model, ds, key, device)
        ratio2, L = mod.q_rollout(q_hat, ds, log=log, trace=False)

        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}
        if partial_path is not None:
            partial_path.write_text(json.dumps(results, indent=2))

    avg = float(_np.mean([r["ratio2"] for r in results.values()]))
    log(f"  avg ratio2 = {avg:.4f}")
    return results, avg


def run_gnn(train_families, test_families, results_dir, epochs, groups_per_batch, lambda_ce, mode, seed, device, log):
    dev = torch.device(device)
    train_configs = build_configs(train_families)
    test_configs = build_configs(test_families)

    struct_cache, edge_cache = {}, {}
    for fam in train_families + test_families:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam] = tg.build_edge_index(pedigree, individuals)

    model = gnn_mod.GNNQBidirSumPool().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    ckpt_path = results_dir / "checkpoint.pt"
    model_path = results_dir / "gnn_q_e6_split.pt"
    ds_cache = {}

    # --- resume ------------------------------------------------------------
    # checkpoint.pt is written every epoch, so a task killed by the SLURM wall
    # clock (or requeued) restarts mid-training rather than from scratch. If it
    # already reached the final epoch there is nothing left to train: skip
    # block [1] entirely -- 36 exact-DP solves plus state-group tensors, the
    # dominant cost -- and go straight to eval, which rebuilds each dataset on
    # demand as it needs it.
    resume_epoch = 0
    if ckpt_path.exists():
        resume_epoch = int(torch.load(ckpt_path, map_location=dev)["epoch"])
    training_done = resume_epoch >= epochs

    if mode in ("train", "both") and training_done:
        log(f"\n[RESUME] checkpoint already at epoch {resume_epoch}/{epochs} -- "
            f"training complete, skipping [1] and [2]")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev)["model_state"])
        if not model_path.exists():
            torch.save(model.state_dict(), model_path)
            log(f"    saved -> {model_path}")

    if mode in ("train", "both") and not training_done:
        log(f"\n[1] Generating {len(train_configs)} train configs ({train_families}) + building state-grouped tensors...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}
        per_config = []
        for fam, reg, pre in train_configs:
            ds = gnn_mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
            ds_cache[key] = ds  # reused by the TRAIN-side overfit eval below -- skips a 2nd exact-DP solve
            s_idx, a_idx, y = gnn_mod.build_qsa_index(ds, device=dev, cache_key=key)
            k_max = len(ds["individuals"])
            group_rows, group_mask, group_target = gnn_mod.build_state_groups(s_idx, y, k_max)
            per_config.append((fam, {
                "nf": base["nf"], "gf": base["gf"],
                "state_idx": s_idx, "action_idx": a_idx, "y": y,
                "group_rows": group_rows, "group_mask": group_mask, "group_target": group_target,
            }))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows "
                f"-> {group_rows.shape[0]:,} state-groups (k_max={k_max})")
        total_groups = sum(d["group_rows"].shape[0] for _, d in per_config)
        log(f"    total state-groups: {total_groups:,}")

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
            ep_loss = ep_mse = ep_ce = 0.0
            for fam, data in per_config:
                ei_c = edge_index_t[fam]
                loss, mse_v, ce_v = gnn_mod.train_one_epoch(model, ei_c, data, opt, groups_per_batch, lambda_ce, dev)
                ep_loss += loss
                ep_mse += mse_v
                ep_ce += ce_v
            n = len(per_config)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  total={ep_loss/n:.5f}  mse={ep_mse/n:.5f}  ce={ep_ce/n:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), model_path)
        log(f"    saved -> {model_path}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(model_path, map_location=dev))
        log(f"\n[3] Eval on TRAIN configs ({train_families}) -- overfit check...")
        train_results, train_avg = evaluate_rollout_cached(gnn_mod, model, train_configs, ds_cache, struct_cache, edge_cache, dev, log,
                                                           partial_path=results_dir / "eval_train_partial.json")
        log(f"\n[4] Eval on TEST configs ({test_families})...")
        test_results, test_avg = gnn_mod.evaluate_rollout(model, test_configs, struct_cache, edge_cache, dev, log)
        (results_dir / "results_e6.json").write_text(json.dumps(
            {"train": train_results, "test": test_results,
             "train_avg_ratio2": train_avg, "test_avg_ratio2": test_avg,
             "gap": test_avg - train_avg}, indent=2))
        log(f"\n  TRAIN avg ratio2 = {train_avg:.4f}   TEST avg ratio2 = {test_avg:.4f}   gap = {test_avg - train_avg:+.4f}")
        return train_avg, test_avg


def run_mlp(train_families, test_families, results_dir, epochs, groups_per_batch, lambda_ce, mode, seed, device, log):
    dev = torch.device(device)
    train_configs = build_configs(train_families)
    test_configs = build_configs(test_families)

    struct_cache = {}
    for fam in train_families + test_families:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)

    model = mlp_mod.MLPQSumPool().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    ckpt_path = results_dir / "checkpoint.pt"
    model_path = results_dir / "mlp_q_e6_split.pt"
    ds_cache = {}

    # --- resume ------------------------------------------------------------
    # checkpoint.pt is written every epoch, so a task killed by the SLURM wall
    # clock (or requeued) restarts mid-training rather than from scratch. If it
    # already reached the final epoch there is nothing left to train: skip
    # block [1] entirely -- 36 exact-DP solves plus state-group tensors, the
    # dominant cost -- and go straight to eval, which rebuilds each dataset on
    # demand as it needs it.
    resume_epoch = 0
    if ckpt_path.exists():
        resume_epoch = int(torch.load(ckpt_path, map_location=dev)["epoch"])
    training_done = resume_epoch >= epochs

    if mode in ("train", "both") and training_done:
        log(f"\n[RESUME] checkpoint already at epoch {resume_epoch}/{epochs} -- "
            f"training complete, skipping [1] and [2]")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev)["model_state"])
        if not model_path.exists():
            torch.save(model.state_dict(), model_path)
            log(f"    saved -> {model_path}")

    if mode in ("train", "both") and not training_done:
        log(f"\n[1] Generating {len(train_configs)} train configs ({train_families}) + building state-grouped tensors...")
        per_config = []
        for fam, reg, pre in train_configs:
            ds = mlp_mod.build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
            ds_cache[key] = ds  # reused by the TRAIN-side overfit eval below -- skips a 2nd exact-DP solve
            s_idx, a_idx, y = mlp_mod.build_qsa_index(ds, device=dev, cache_key=key)
            k_max = len(ds["individuals"])
            group_rows, group_mask, group_target = mlp_mod.build_state_groups(s_idx, y, k_max)
            per_config.append({
                "nf": base["nf"], "gf": base["gf"],
                "state_idx": s_idx, "action_idx": a_idx, "y": y,
                "group_rows": group_rows, "group_mask": group_mask, "group_target": group_target,
            })
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows "
                f"-> {group_rows.shape[0]:,} state-groups (k_max={k_max})")
        total_groups = sum(d["group_rows"].shape[0] for d in per_config)
        log(f"    total state-groups: {total_groups:,}")

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
            ep_loss = ep_mse = ep_ce = 0.0
            for data in per_config:
                loss, mse_v, ce_v = mlp_mod.train_one_epoch(model, data, opt, groups_per_batch, lambda_ce, dev)
                ep_loss += loss
                ep_mse += mse_v
                ep_ce += ce_v
            n = len(per_config)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  total={ep_loss/n:.5f}  mse={ep_mse/n:.5f}  ce={ep_ce/n:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), model_path)
        log(f"    saved -> {model_path}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(model_path, map_location=dev))
        log(f"\n[3] Eval on TRAIN configs ({train_families}) -- overfit check...")
        train_results, train_avg = evaluate_rollout_cached(mlp_mod, model, train_configs, ds_cache, struct_cache, None, dev, log,
                                                           partial_path=results_dir / "eval_train_partial.json")
        log(f"\n[4] Eval on TEST configs ({test_families})...")
        test_results, test_avg = mlp_mod.evaluate_rollout(model, test_configs, struct_cache, dev, log)
        (results_dir / "results_e6.json").write_text(json.dumps(
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
    p.add_argument("--groups_per_batch", type=int, default=512)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="redo this (exp,kind,seed) even if results_e6.json already exists")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = EXPERIMENTS[args.exp]
    train_families, test_families = spec["train"], spec["test"]

    results_dir = RESULTS_ROOT / f"e6_split_{args.exp}" / args.kind / "seed_runs" / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent resubmit: the whole 0-29 array can be re-submitted after a
    # timeout and every already-finished task exits here in under a second,
    # so only the unfinished ones actually run.
    done_marker = results_dir / "results_e6.json"
    if done_marker.exists() and not args.force:
        print(f"[SKIP] exp={args.exp} kind={args.kind} seed={args.seed} already complete "
              f"({done_marker}). Use --force to redo.", flush=True)
        return
    if args.force:
        for stale in (results_dir / "eval_train_partial.json", results_dir / "checkpoint.pt"):
            stale.unlink(missing_ok=True)

    log_f = open(results_dir / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}\n[E6-SPLIT exp={args.exp} kind={args.kind}] {datetime.now().isoformat()}  "
        f"mode={args.mode}  epochs={args.epochs}  seed={args.seed}")
    log(f"TRAIN={train_families}  TEST={test_families}")

    if args.kind == "gnn":
        run_gnn(train_families, test_families, results_dir, args.epochs, args.groups_per_batch, args.lambda_ce, args.mode, args.seed, args.device, log)
    else:
        run_mlp(train_families, test_families, results_dir, args.epochs, args.groups_per_batch, args.lambda_ce, args.mode, args.seed, args.device, log)

    log_f.close()


if __name__ == "__main__":
    main()
