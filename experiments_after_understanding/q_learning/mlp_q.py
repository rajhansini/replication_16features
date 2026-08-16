"""MLP-Q — same embedding as mlp/run.py (mean-pool over people), new head.

Old:  V_hat(s)     <- mean-pool(node_feats) || cost_vec
New:  Q_hat(s, a)  <- mean-pool(node_feats) || node_feats[a] || cost_vec

Trained on (state, action, Q*) triples instead of (state, V*) pairs. Q* comes
from qsa_data.build_qsa_index — exact, computed from the cached DP solution,
no new DP solve.

Does not modify mlp/run.py. Imports it as a module purely for its data-loading
and structural-feature helpers, which are unchanged.

Usage:
    python mlp_q.py --mode both --epochs 200 --configs 4   # quick smoke test
    python mlp_q.py --mode both --epochs 200               # full train pool
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE        = Path(__file__).resolve().parent
EXP_ROOT    = HERE.parent
ROOT        = EXP_ROOT.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlp_mod = _load_module("mlp_run_base", EXP_ROOT / "mlp" / "run.py")

from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

NODE_FEAT = mlp_mod.NODE_FEAT
COST_DIM  = mlp_mod.COST_DIM
CACHE_DIR = mlp_mod.CACHE_DIR
GENES     = mlp_mod.GENES

BASE_RESULTS_DIR = Path(os.environ.get("Q_RESULTS_DIR", str(HERE / "results")))
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MLPQ(nn.Module):
    """mean-pool(node_feats) || node_feats[action] || cost_vec -> Q(s,a)."""

    def __init__(self, node_feat=NODE_FEAT, cost_dim=COST_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat * 2 + cost_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats, cost_vec, action_idx):
        pooled = node_feats.mean(dim=1)                                   # (B, 13) — same embedding as V-model
        B = node_feats.shape[0]
        candidate = node_feats[torch.arange(B, device=node_feats.device), action_idx]  # (B, 13)
        x = torch.cat([pooled, candidate, cost_vec], dim=-1)              # (B, 37)
        return self.net(x).squeeze(-1)


def load_config_tensors(key, struct_feats, device, log):
    ds = None
    with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)
    base = mlp_mod.load_dataset(key, struct_feats, device)   # nf, gf, y(unused here)
    t0 = time.time()
    state_idx, action_idx, q_star = build_qsa_index(ds, device=device, cache_key=key)
    log(f"    [{key}] {len(ds['states']):,} states -> {len(q_star):,} (s,a) rows "
        f"({time.time()-t0:.1f}s, cached after first run)")
    return ds, base["nf"], base["gf"], state_idx, action_idx, q_star


def train_one_epoch(model, nf, gf, state_idx, action_idx, y, opt, batch_size, device):
    model.train()
    perm = torch.randperm(len(y), device=device)
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        idx = perm[start:start + batch_size]
        s_idx, a_idx, target = state_idx[idx], action_idx[idx], y[idx]
        pred = model(nf[s_idx], gf[s_idx], a_idx)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    return total_loss / len(y)


def precompute_qhat(model, ds, key, device, batch_size=4096):
    """Batched Q_hat(s, person) for every (state, untested person) pair in ds,
    i.e. every row q_rollout's greedy pick could ever need. Reuses the same
    (state_idx, action_idx) enumeration as training (build_qsa_index), cached
    to disk under `key` — cheap to rebuild/reload even for test configs.

    Returns {state: {person: predicted_Q(s, person)}}.
    """
    state_idx, action_idx, _ = build_qsa_index(ds, device=device, cache_key=key)
    individuals = ds["individuals"]
    states = ds["states"]
    nf, gf = ds["_nf"], ds["_gf"]

    model.eval()
    q_hat: dict = {}
    with torch.no_grad():
        for start in range(0, len(state_idx), batch_size):
            end  = min(start + batch_size, len(state_idx))
            s_b  = state_idx[start:end]
            a_b  = action_idx[start:end]
            pred = model(nf[s_b], gf[s_b], a_b)
            s_b_cpu, a_b_cpu, pred_cpu = s_b.cpu(), a_b.cpu(), pred.cpu()
            for i in range(end - start):
                s = states[s_b_cpu[i].item()]
                p = individuals[a_b_cpu[i].item()]
                q_hat.setdefault(s, {})[p] = pred_cpu[i].item()
    return q_hat


def evaluate_rollout(model, keys, struct_cache, device, log):
    """Real policy evaluation: the model's own greedy Q(s,a) choices drive the
    rollout, branching over every true outcome probability (q_rollout), exactly
    mirroring the V(s) framework's rollout(). Replaces the old DP-forced-
    trajectory action-agreement metric, which never let the model's own
    predictions affect which states got evaluated.
    """
    results = {}
    for key in keys:
        fam = key.split("_")[0]
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        base = mlp_mod.load_dataset(key, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

        q_hat = precompute_qhat(model, ds, key, device)
        ratio2, L = q_rollout(q_hat, ds, log=log, trace=False)

        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"  TEST avg ratio2 = {avg:.4f}")
    return results, avg


def main(device="cpu", epochs=500, batch_size=2048, mode="both", config_limit=None,
         test_config_limit=None, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    RESULTS_DIR = BASE_RESULTS_DIR / "seed_runs" / f"seed{seed}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_f = open(RESULTS_DIR / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}\n[MLP-Q] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}  seed={seed}")

    dev = torch.device(device)
    train_keys = mlp_mod.TRAIN_KEYS[:config_limit] if config_limit else mlp_mod.TRAIN_KEYS
    test_keys  = mlp_mod.TEST_KEYS[:test_config_limit] if test_config_limit else mlp_mod.TEST_KEYS

    struct_cache = {}
    for fam in mlp_mod.TRAIN_FAMILIES + mlp_mod.TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree = generate_deterministic_pedigree(mlp_mod.FAMILY_CASES[fam])
        struct_cache[fam] = mlp_mod.compute_structural_features(pedigree, sample_ds["individuals"])

    model = MLPQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Building (s,a,Q*) tensors for {len(train_keys)} train configs...")
        # Kept as a per-config list, NOT concatenated: Trio has 3 people, Nuclear has 4 —
        # their nf tensors differ in dim 1, so torch.cat across configs is invalid.
        per_config = []
        for key in train_keys:
            fam = key.split("_")[0]
            ds, nf, gf, s_idx, a_idx, y = load_config_tensors(key, struct_cache[fam], dev, log)
            per_config.append((nf, gf, s_idx, a_idx, y))
        total_rows = sum(len(y) for *_, y in per_config)
        log(f"    total (s,a) rows: {total_rows:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = RESULTS_DIR / "checkpoint_mlp.pt"
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
                loss = train_one_epoch(model, nf_c, gf_c, s_idx_c, a_idx_c, y_c, opt, batch_size, dev)
                ep_loss += loss * len(y_c)
                ep_count += len(y_c)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  mse={ep_loss/ep_count:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), RESULTS_DIR / "mlp_q.pt")
        log(f"    saved -> {RESULTS_DIR/'mlp_q.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "mlp_q.pt", map_location=dev))
        log(f"\n[3] Eval on TEST_KEYS ({len(test_keys)} configs), "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, test_keys, struct_cache, dev, log)
        (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results.json'}")

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--configs", type=int, default=None, help="limit train configs, for quick smoke tests")
    p.add_argument("--test_configs", type=int, default=None, help="limit eval configs, for quick smoke tests")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
         mode=args.mode, config_limit=args.configs, test_config_limit=args.test_configs, seed=args.seed)
