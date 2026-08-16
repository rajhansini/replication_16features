"""Message-passing DEPTH ABLATION for GNN-Q (Q-track, 3-gene).

Runs the exact same Q(s,a) training/eval pipeline as
experiments_after_understanding/q_learning/gnn_q.py, but with the number of
message-passing rounds as a swept hyperparameter:

    --n_rounds 0   per-person linear lift (13 -> 32), NO message passing.
                   Isolates "message passing" from "just a per-node transform".
                   Head dims are kept identical to the >=1 round models so the
                   only thing that changes is whether neighbors talk.
    --n_rounds 1   one hop: each person sees direct parents (enough for the
                   Trio/Nuclear training topology, depth <= 1).
    --n_rounds 2   current production setting (matches ThreeGeneration depth 2).
    --n_rounds 3   over-smoothing probe: does depth-2 test topology want more,
                   or are embeddings already mixing too much?

NOTHING in the existing pipeline is modified. This script imports
experiments_after_understanding/gnn/run.py and q_learning/qsa_data.py purely as
libraries (data loading, Q* targets, structural features), all unchanged.

Results layout (self-contained, never collides with q_learning/results):
    <BASE>/rounds{N}/seed_runs/seed{S}/
        run.log            training MSE every 20 epochs + eval ratio2 per config
        gnn_q_rounds.pt    final weights (epoch `epochs`)
        checkpoint.pt      last-epoch checkpoint (resume support)
        results_gnn.json   per-config ratio2 / L / V*

BASE defaults to this file's ../results, override with Q_ABLATION_RESULTS_DIR.

Usage:
    python gnn_q_rounds.py --n_rounds 1 --epochs 500 --seed 0
    python gnn_q_rounds.py --n_rounds 0 --epochs 20 --configs 2 --test_configs 1  # smoke
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
ABLATION    = HERE.parent                       # things_to_improve_Q_star_experiments
ROOT        = ABLATION.parent                   # repo root
EXP_ROOT    = ROOT / "experiments_after_understanding"
EXPERIMENTS = ROOT / "ground-up-experiments"
Q_LEARNING  = EXP_ROOT / "q_learning"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_LEARNING))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gnn_mod = _load_module("gnn_run_base_ablation", EXP_ROOT / "gnn" / "run.py")

from qsa_data import build_qsa_index          # noqa: E402
from exputils.eval import q_rollout           # noqa: E402
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

NODE_FEAT = gnn_mod.NODE_FEAT
COST_DIM  = gnn_mod.COST_DIM
HIDDEN    = gnn_mod.HIDDEN
CACHE_DIR = gnn_mod.CACHE_DIR
GENES     = gnn_mod.GENES

BASE_RESULTS_DIR = Path(os.environ.get("Q_ABLATION_RESULTS_DIR", str(HERE / "results")))


# ── model ───────────────────────────────────────────────────────────────────

class GNNQRounds(nn.Module):
    """GNN-Q with a configurable number of message-passing rounds.

    n_rounds >= 1 : identical message passing to q_learning/gnn_q.py's GNNQ.
    n_rounds == 0 : no message passing; a single per-node Linear(13->32)+ReLU
                    "lift" replaces the graph rounds. Pooling + head are the
                    same, so this is a topology-blind control with matched
                    capacity in the readout.

    Head is always Linear(hidden*2 + cost -> 16) -> ReLU -> Linear(16 -> 1),
    consuming mean-pooled embedding || candidate embedding || cost, exactly as
    in production GNN-Q.
    """

    def __init__(self, node_feat=NODE_FEAT, hidden=HIDDEN, cost_dim=COST_DIM, n_rounds=2):
        super().__init__()
        self.n_rounds = n_rounds
        self.msg_layers = nn.ModuleList()
        self.upd_layers = nn.ModuleList()

        if n_rounds == 0:
            self.lift = nn.Sequential(nn.Linear(node_feat, hidden), nn.ReLU())
        else:
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
        if self.n_rounds == 0:
            return self.lift(node_feats)                  # (B, N, hidden), no neighbor info
        h = node_feats
        src, dst = edge_index[0], edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._message_pass(h, src, dst, msg_fn, upd_fn)
        return h                                          # (B, N, hidden)

    def forward(self, node_feats, edge_index, cost_vec, action_idx):
        h = self.embed(node_feats, edge_index)
        pooled = h.mean(dim=1)
        B = h.shape[0]
        candidate = h[torch.arange(B, device=h.device), action_idx]
        x = torch.cat([pooled, candidate, cost_vec], dim=-1)
        return self.head(x).squeeze(-1)


# ── data helpers (mirror gnn_q.py) ───────────────────────────────────────────

def load_config_tensors(key, struct_feats, edge_index, device, log):
    with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)
    base = gnn_mod.load_dataset(key, struct_feats, edge_index, device)
    t0 = time.time()
    state_idx, action_idx, q_star = build_qsa_index(ds, device=device, cache_key=key)
    log(f"    [{key}] {len(ds['states']):,} states -> {len(q_star):,} (s,a) rows "
        f"({time.time()-t0:.1f}s, cached after first run)")
    return ds, base["nf"], base["gf"], base["ei"], state_idx, action_idx, q_star


def precompute_qhat(model, ds, key, edge_index_t, device, batch_size=4096):
    state_idx, action_idx, _ = build_qsa_index(ds, device=device, cache_key=key)
    individuals = ds["individuals"]
    states = ds["states"]
    nf, gf = ds["_nf"], ds["_gf"]
    model.eval()
    q_hat: dict = {}
    with torch.no_grad():
        for start in range(0, len(state_idx), batch_size):
            end = min(start + batch_size, len(state_idx))
            s_b = state_idx[start:end]
            a_b = action_idx[start:end]
            pred = model(nf[s_b], edge_index_t, gf[s_b], a_b)
            s_b_cpu, a_b_cpu, pred_cpu = s_b.cpu(), a_b.cpu(), pred.cpu()
            for i in range(end - start):
                s = states[s_b_cpu[i].item()]
                p = individuals[a_b_cpu[i].item()]
                q_hat.setdefault(s, {})[p] = pred_cpu[i].item()
    return q_hat


def evaluate_rollout(model, keys, struct_cache, edge_cache, device, log):
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
         test_config_limit=None, seed=0, n_rounds=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    RESULTS_DIR = BASE_RESULTS_DIR / f"rounds{n_rounds}" / "seed_runs" / f"seed{seed}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_f = open(RESULTS_DIR / "run.log", "a")

    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}\n[GNN-Q ablation] {datetime.now().isoformat()}  "
        f"mode={mode}  epochs={epochs}  seed={seed}  n_rounds={n_rounds}")

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

    model = GNNQRounds(n_rounds=n_rounds).to(dev)
    arch = ("linear-lift (no MP)" if n_rounds == 0 else f"MP x{n_rounds}")
    log(f"architecture: {arch} -> mean pool || candidate || cost -> Linear({HIDDEN*2+COST_DIM}->16) -> ReLU -> Linear(16->1)")
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Building (s,a,Q*) tensors for {len(train_keys)} train configs...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}

        per_config = []
        for key in train_keys:
            fam = key.split("_")[0]
            ds, nf, gf, ei, s_idx, a_idx, y = load_config_tensors(
                key, struct_cache[fam], edge_cache[fam], dev, log)
            per_config.append((fam, nf, gf, s_idx, a_idx, y))
        total_rows = sum(len(y) for *_, y in per_config)
        log(f"    total (s,a) rows: {total_rows:,}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt_path = RESULTS_DIR / "checkpoint.pt"
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
        torch.save(model.state_dict(), RESULTS_DIR / "gnn_q_rounds.pt")
        log(f"    saved -> {RESULTS_DIR/'gnn_q_rounds.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "gnn_q_rounds.pt", map_location=dev))
        log(f"\n[3] Eval on TEST_KEYS ({len(test_keys)} configs), "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, test_keys, struct_cache, edge_cache, dev, log)
        payload = {"n_rounds": n_rounds, "seed": seed, "avg_ratio2": avg, "per_config": results}
        (RESULTS_DIR / "results_gnn.json").write_text(json.dumps(payload, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results_gnn.json'}")

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--configs", type=int, default=None, help="limit train configs (smoke test)")
    p.add_argument("--test_configs", type=int, default=None, help="limit eval configs (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_rounds", type=int, required=True, choices=[0, 1, 2, 3],
                   help="number of message-passing rounds (0 = per-node linear lift, no MP)")
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
         mode=args.mode, config_limit=args.configs, test_config_limit=args.test_configs,
         seed=args.seed, n_rounds=args.n_rounds)
