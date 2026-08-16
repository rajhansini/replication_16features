"""GNN-Q — same embedding as gnn/run.py (2 rounds message passing), new head.

Old:  V_hat(s)     <- mean-pool(h) || cost_vec,             h = 2 rounds of message passing
New:  Q_hat(s, a)  <- mean-pool(h) || h[a] || cost_vec,      same h, unchanged

The candidate's post-message-passing embedding h[a] is richer than MLP's raw
node_feats[a] — it already encodes that person's relatives after 2 rounds.

Trained on (state, action, Q*) triples (see qsa_data.py), same as mlp_q.py.
Does not modify gnn/run.py — imported as a module purely for data/structural
helpers, which are unchanged.

Usage:
    python gnn_q.py --mode both --epochs 200 --configs 4   # quick smoke test
    python gnn_q.py --mode both --epochs 200               # full train pool
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


gnn_mod = _load_module("gnn_run_base", EXP_ROOT / "gnn" / "run.py")

from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

NODE_FEAT = gnn_mod.NODE_FEAT
COST_DIM  = gnn_mod.COST_DIM
HIDDEN    = gnn_mod.HIDDEN
N_ROUNDS  = gnn_mod.N_ROUNDS
CACHE_DIR = gnn_mod.CACHE_DIR
GENES     = gnn_mod.GENES

BASE_RESULTS_DIR = Path(os.environ.get("Q_RESULTS_DIR", str(HERE / "results")))
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class GNNQ(nn.Module):
    """Identical message-passing to gnn/run.py's GNN. Only the head differs."""

    def __init__(self, node_feat=NODE_FEAT, hidden=HIDDEN, cost_dim=COST_DIM, n_rounds=N_ROUNDS):
        super().__init__()
        self.n_rounds = n_rounds
        self.msg_layers = nn.ModuleList()
        self.upd_layers = nn.ModuleList()
        in_dim = node_feat
        for _ in range(n_rounds):
            self.msg_layers.append(nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.ReLU()))
            self.upd_layers.append(nn.Sequential(nn.Linear(in_dim + hidden, hidden), nn.ReLU()))
            in_dim = hidden

        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + cost_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _message_pass(self, h, src, dst, msg_fn, upd_fn):
        B, N, _ = h.shape
        msg_in = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        msgs   = msg_fn(msg_in)
        H      = msgs.shape[-1]
        agg = torch.zeros(B, N, H, device=h.device)
        idx = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx, msgs)
        return upd_fn(torch.cat([h, agg], dim=-1))

    def embed(self, node_feats, edge_index):
        h = node_feats
        src, dst = edge_index[0], edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._message_pass(h, src, dst, msg_fn, upd_fn)
        return h  # (B, N, hidden) — per-person embeddings, unchanged from gnn/run.py

    def forward(self, node_feats, edge_index, cost_vec, action_idx):
        h = self.embed(node_feats, edge_index)            # (B, N, 32) — same embedding as V-model
        pooled = h.mean(dim=1)                             # (B, 32)
        B = h.shape[0]
        candidate = h[torch.arange(B, device=h.device), action_idx]  # (B, 32)
        x = torch.cat([pooled, candidate, cost_vec], dim=-1)          # (B, 75)
        return self.head(x).squeeze(-1)


def load_config_tensors(key, struct_feats, edge_index, device, log):
    with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)
    base = gnn_mod.load_dataset(key, struct_feats, edge_index, device)  # nf, gf, ei
    t0 = time.time()
    state_idx, action_idx, q_star = build_qsa_index(ds, device=device, cache_key=key)
    log(f"    [{key}] {len(ds['states']):,} states -> {len(q_star):,} (s,a) rows "
        f"({time.time()-t0:.1f}s, cached after first run)")
    return ds, base["nf"], base["gf"], base["ei"], state_idx, action_idx, q_star


def train_one_epoch(model, nf, gf, ei, state_idx, action_idx, y, opt, batch_size, device):
    model.train()
    perm = torch.randperm(len(y), device=device)
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        idx = perm[start:start + batch_size]
        s_idx, a_idx, target = state_idx[idx], action_idx[idx], y[idx]
        pred = model(nf[s_idx], ei, gf[s_idx], a_idx)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    return total_loss / len(y)


def precompute_qhat(model, ds, key, edge_index_t, device, batch_size=4096):
    """Batched Q_hat(s, person) for every (state, untested person) pair — GNN-Q
    version. Same enumeration/caching as mlp_q.py's precompute_qhat, but the
    forward pass also needs the family's edge_index for message passing.

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
            pred = model(nf[s_b], edge_index_t, gf[s_b], a_b)
            s_b_cpu, a_b_cpu, pred_cpu = s_b.cpu(), a_b.cpu(), pred.cpu()
            for i in range(end - start):
                s = states[s_b_cpu[i].item()]
                p = individuals[a_b_cpu[i].item()]
                q_hat.setdefault(s, {})[p] = pred_cpu[i].item()
    return q_hat


def evaluate_rollout(model, keys, struct_cache, edge_cache, device, log):
    """Real policy evaluation via q_rollout — see mlp_q.py's evaluate_rollout
    for the full rationale. Replaces the old DP-forced-trajectory metric.
    """
    results = {}
    for key in keys:
        fam = key.split("_")[0]
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        base = gnn_mod.load_dataset(key, struct_cache[fam], edge_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        edge_index_t = torch.tensor(edge_cache[fam], device=device)

        q_hat = precompute_qhat(model, ds, key, edge_index_t, device)
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

    log(f"\n{'='*60}\n[GNN-Q] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}  seed={seed}")

    dev = torch.device(device)
    train_keys = gnn_mod.TRAIN_KEYS[:config_limit] if config_limit else gnn_mod.TRAIN_KEYS
    test_keys  = gnn_mod.TEST_KEYS[:test_config_limit] if test_config_limit else gnn_mod.TEST_KEYS

    struct_cache, edge_cache = {}, {}
    for fam in gnn_mod.TRAIN_FAMILIES + gnn_mod.TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
        struct_cache[fam] = gnn_mod.compute_structural_features(pedigree, sample_ds["individuals"])
        edge_cache[fam]   = gnn_mod.build_edge_index(pedigree, sample_ds["individuals"])

    model = GNNQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Building (s,a,Q*) tensors for {len(train_keys)} train configs...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}

        per_config = []  # (fam, nf, gf, state_idx, action_idx, y) — kept separate, edge_index differs by family
        for key in train_keys:
            fam = key.split("_")[0]
            ds, nf, gf, ei, s_idx, a_idx, y = load_config_tensors(
                key, struct_cache[fam], edge_cache[fam], dev, log)
            per_config.append((fam, nf, gf, s_idx, a_idx, y))
        total_rows = sum(len(y) for *_, y in per_config)
        log(f"    total (s,a) rows: {total_rows:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = RESULTS_DIR / "checkpoint_gnn.pt"
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
                perm = torch.randperm(len(y_c), device=dev)
                for bstart in range(0, len(y_c), batch_size):
                    bidx = perm[bstart:bstart + batch_size]
                    s_b, a_b, t_b = s_idx_c[bidx], a_idx_c[bidx], y_c[bidx]
                    pred = model(nf_c[s_b], ei_c, gf_c[s_b], a_b)
                    loss = nn.functional.mse_loss(pred, t_b)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item() * len(bidx)
                    ep_count += len(bidx)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  mse={ep_loss/ep_count:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), RESULTS_DIR / "gnn_q.pt")
        log(f"    saved -> {RESULTS_DIR/'gnn_q.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "gnn_q.pt", map_location=dev))
        log(f"\n[3] Eval on TEST_KEYS ({len(test_keys)} configs), "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, test_keys, struct_cache, edge_cache, dev, log)
        (RESULTS_DIR / "results_gnn.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results_gnn.json'}")

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
