"""MLP-Q retrained with combined MSE + cross-entropy loss, PLUS depth-based
reweighting (3-gene). See train_gnn_q_ce_reweighted.py for the full
rationale — this is the same Fix 1 + Fix 2 combination applied to MLP-Q
instead of GNN-Q.

Usage:
    python train_mlp_q_ce_reweighted.py --seed 0 --epochs 500 --lambda_ce 1.0
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
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR    = EXP_ROOT / "q_learning"
EXPERIMENTS = ROOT / "ground-up-experiments"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(HERE))

from losses import build_state_groups, combined_loss, state_depths, depth_weight  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlp_mod = _load_module("mlp_run_cew", EXP_ROOT / "mlp" / "run.py")
mlp_q   = _load_module("mlp_q_cew_base", Q_DIR / "mlp_q.py")
from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

CACHE_DIR = mlp_mod.CACHE_DIR
RESULTS_BASE = HERE / "results" / "mlp_q_ce_reweighted"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)


def load_config_groups(key, struct_feats, device, log):
    with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = __import__("pickle").load(f)
    base = mlp_mod.load_dataset(key, struct_feats, device)
    state_idx, action_idx, y = build_qsa_index(ds, device=device, cache_key=key)
    k_max = len(ds["individuals"])
    group_rows, group_mask, group_target = build_state_groups(state_idx, y, k_max)

    row_depths = state_depths(ds["states"], state_idx)
    group_depth = row_depths[group_rows[:, 0]]
    group_weight = depth_weight(group_depth)

    log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows "
        f"-> {group_rows.shape[0]:,} state-groups (k_max={k_max})")
    return {
        "nf": base["nf"], "gf": base["gf"],
        "state_idx": state_idx, "action_idx": action_idx, "y": y,
        "group_rows": group_rows, "group_mask": group_mask, "group_target": group_target,
        "group_weight": group_weight,
    }


def train_one_epoch(model, data, opt, groups_per_batch, lambda_ce, device):
    model.train()
    nf, gf = data["nf"], data["gf"]
    state_idx, action_idx, y = data["state_idx"], data["action_idx"], data["y"]
    group_rows, group_mask, group_target = data["group_rows"], data["group_mask"], data["group_target"]
    group_weight = data["group_weight"]

    n_groups = group_rows.shape[0]
    perm = torch.randperm(n_groups, device=device)

    total_loss = total_mse = total_ce = 0.0
    n_batches = 0
    for start in range(0, n_groups, groups_per_batch):
        g_idx = perm[start:start + groups_per_batch]
        rows = group_rows[g_idx]
        mask = group_mask[g_idx]
        target = group_target[g_idx]
        weight = group_weight[g_idx]

        flat_rows = rows.reshape(-1)
        flat_mask = mask.reshape(-1)
        valid_rows = flat_rows[flat_mask]
        s_b = state_idx[valid_rows]
        a_b = action_idx[valid_rows]
        pred_valid = model(nf[s_b], gf[s_b], a_b)

        pred = torch.zeros(flat_rows.shape[0], device=device)
        pred[flat_mask] = pred_valid
        pred = pred.reshape(rows.shape)

        y_groups = torch.zeros(flat_rows.shape[0], device=device)
        y_groups[flat_mask] = y[valid_rows]
        y_groups = y_groups.reshape(rows.shape)

        loss, mse_val, ce_val = combined_loss(pred, mask, target, y_groups, lambda_ce, group_weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_mse += mse_val
        total_ce += ce_val
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_ce / n_batches


def evaluate_rollout(model, keys, struct_cache, device, log):
    results = {}
    for key in keys:
        fam = key.split("_")[0]
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = __import__("pickle").load(f)
        base = mlp_mod.load_dataset(key, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

        q_hat = mlp_q.precompute_qhat(model, ds, key, device)
        ratio2, L = q_rollout(q_hat, ds, log=log, trace=False)

        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"  TEST avg ratio2 = {avg:.4f}")
    return results, avg


def main(device="cpu", epochs=500, groups_per_batch=512, lambda_ce=1.0,
         mode="both", config_limit=None, test_config_limit=None, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    RESULTS_DIR = RESULTS_BASE / "seed_runs" / f"seed{seed}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_f = open(RESULTS_DIR / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}\n[MLP-Q-CE-REWEIGHTED] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}"
        f"  seed={seed}  lambda_ce={lambda_ce}  groups_per_batch={groups_per_batch}")

    dev = torch.device(device)
    train_keys = mlp_mod.TRAIN_KEYS[:config_limit] if config_limit else mlp_mod.TRAIN_KEYS
    test_keys  = mlp_mod.TEST_KEYS[:test_config_limit] if test_config_limit else mlp_mod.TEST_KEYS

    struct_cache = {}
    for fam in mlp_mod.TRAIN_FAMILIES + mlp_mod.TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = __import__("pickle").load(f)
        pedigree = generate_deterministic_pedigree(mlp_mod.FAMILY_CASES[fam])
        struct_cache[fam] = mlp_mod.compute_structural_features(pedigree, sample_ds["individuals"])

    model = mlp_q.MLPQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Building state-grouped (s,a,Q*) tensors for {len(train_keys)} train configs...")
        per_config = []
        for key in train_keys:
            fam = key.split("_")[0]
            data = load_config_groups(key, struct_cache[fam], dev, log)
            per_config.append(data)
        total_groups = sum(d["group_rows"].shape[0] for d in per_config)
        log(f"    total state-groups: {total_groups:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = RESULTS_DIR / "checkpoint_mlp_ce_rw.pt"
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
                loss, mse_v, ce_v = train_one_epoch(model, data, opt, groups_per_batch, lambda_ce, dev)
                ep_loss += loss
                ep_mse += mse_v
                ep_ce += ce_v
            n = len(per_config)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  total={ep_loss/n:.5f}  mse={ep_mse/n:.5f}  ce={ep_ce/n:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), RESULTS_DIR / "mlp_q_ce_rw.pt")
        log(f"    saved -> {RESULTS_DIR/'mlp_q_ce_rw.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "mlp_q_ce_rw.pt", map_location=dev))
        log(f"\n[3] Eval on TEST_KEYS ({len(test_keys)} configs), "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, test_keys, struct_cache, dev, log)
        (RESULTS_DIR / "results_mlp_ce_rw.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results_mlp_ce_rw.json'}")

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--groups_per_batch", type=int, default=512)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--configs", type=int, default=None)
    p.add_argument("--test_configs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, groups_per_batch=args.groups_per_batch,
         lambda_ce=args.lambda_ce, mode=args.mode, config_limit=args.configs,
         test_config_limit=args.test_configs, seed=args.seed)
