"""E9 -- combined pooling: concat E6's global-sum readout AND E7's local
neighbor-mean readout (2-gene GNN-Q). See e9_train_gnn_q.py (3-gene) for the
full rationale. Same GNNQBidirComboPool architecture as the 3-gene version --
only NODE_FEAT/COST_DIM/HIDDEN dims differ.

Usage:
    python e9_train_two_gene_gnn_q.py --seed 0 --epochs 500
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
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR    = EXP_ROOT / "q_learning"
EXPERIMENTS = ROOT / "ground-up-experiments"
FIXING   = ROOT / "fixing_gnn_q"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))
sys.path.insert(0, str(FIXING))

from losses import build_state_groups, combined_loss  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load_module("two_gene_run_e9", EXP_ROOT / "two_gene" / "run.py")
from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402

NODE_FEAT = tg.NODE_FEAT
COST_DIM  = tg.COST_DIM
HIDDEN    = tg.HIDDEN
GENES     = tg.GENES

RESULTS_BASE = HERE / "results" / "e9_gnn_combopool_2gene"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)


def build_neighbor_mask(edge_index_np, n_people):
    """(N, N) float mask: mask[i, j] = 1 if j is i's direct parent or child,
    0 otherwise (self excluded). edge_index is (2, E) parent(src) -> child(dst)."""
    mask = np.zeros((n_people, n_people), dtype=np.float32)
    src, dst = edge_index_np[0], edge_index_np[1]
    for s, d in zip(src, dst):
        mask[d, s] = 1.0
        mask[s, d] = 1.0
    return mask


class GNNQBidirComboPool(nn.Module):
    """Identical to E7's GNNQBidirNeighborPool except the readout concatenates
    BOTH E6's global-sum pooled vector and E7's local-neighbor-mean pooled
    vector, instead of only the local one."""

    def __init__(self, node_feat=NODE_FEAT, hidden=HIDDEN, cost_dim=COST_DIM, n_rounds=2):
        super().__init__()
        self.n_rounds = n_rounds
        self.msg_fwd = nn.ModuleList()
        self.msg_bwd = nn.ModuleList()
        self.upd_layers = nn.ModuleList()
        in_dim = node_feat
        for _ in range(n_rounds):
            self.msg_fwd.append(nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.ReLU()))
            self.msg_bwd.append(nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.ReLU()))
            self.upd_layers.append(nn.Sequential(nn.Linear(in_dim + hidden, hidden), nn.ReLU()))
            in_dim = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden * 3 + cost_dim, 16),  # <-- 3x hidden: global + local + candidate
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _message_pass(self, h, src, dst, msg_fwd, msg_bwd, upd_fn):
        B, N, _ = h.shape
        msg_in_f = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        msgs_f = msg_fwd(msg_in_f)
        msg_in_b = torch.cat([h[:, dst, :], h[:, src, :]], dim=-1)
        msgs_b = msg_bwd(msg_in_b)

        H = msgs_f.shape[-1]
        agg = torch.zeros(B, N, H, device=h.device)
        idx_f = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx_f, msgs_f)
        idx_b = src.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx_b, msgs_b)

        return upd_fn(torch.cat([h, agg], dim=-1))

    def embed(self, node_feats, edge_index):
        h = node_feats
        src, dst = edge_index[0], edge_index[1]
        for mf, mb, upd_fn in zip(self.msg_fwd, self.msg_bwd, self.upd_layers):
            h = self._message_pass(h, src, dst, mf, mb, upd_fn)
        return h

    def forward(self, node_feats, edge_index, cost_vec, action_idx, neighbor_mask):
        h = self.embed(node_feats, edge_index)
        B = h.shape[0]

        pooled_global = h.sum(dim=1)

        degree = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_mask_norm = neighbor_mask / degree
        pooled_all = torch.einsum('ij,bjd->bid', neighbor_mask_norm, h)
        pooled_local = pooled_all[torch.arange(B, device=h.device), action_idx]

        candidate = h[torch.arange(B, device=h.device), action_idx]
        x = torch.cat([pooled_global, pooled_local, candidate, cost_vec], dim=-1)
        return self.head(x).squeeze(-1)


def build_config(fam, reg, pre):
    return tg.build_two_gene_dataset(
        family_label=fam,
        allele_freqs=tg.ALLELE_FREQ_REGIMES[reg],
        preset_label=pre,
        genes=GENES,
    )


def train_one_epoch(model, ei, nmask, data, opt, groups_per_batch, lambda_ce, device):
    model.train()
    nf, gf = data["nf"], data["gf"]
    state_idx, action_idx, y = data["state_idx"], data["action_idx"], data["y"]
    group_rows, group_mask, group_target = data["group_rows"], data["group_mask"], data["group_target"]

    n_groups = group_rows.shape[0]
    perm = torch.randperm(n_groups, device=device)

    total_loss = total_mse = total_ce = 0.0
    n_batches = 0
    for start in range(0, n_groups, groups_per_batch):
        g_idx = perm[start:start + groups_per_batch]
        rows = group_rows[g_idx]
        mask = group_mask[g_idx]
        target = group_target[g_idx]

        flat_rows = rows.reshape(-1)
        flat_mask = mask.reshape(-1)
        valid_rows = flat_rows[flat_mask]
        s_b = state_idx[valid_rows]
        a_b = action_idx[valid_rows]
        pred_valid = model(nf[s_b], ei, gf[s_b], a_b, nmask)

        pred = torch.zeros(flat_rows.shape[0], device=device)
        pred[flat_mask] = pred_valid
        pred = pred.reshape(rows.shape)

        y_groups = torch.zeros(flat_rows.shape[0], device=device)
        y_groups[flat_mask] = y[valid_rows]
        y_groups = y_groups.reshape(rows.shape)

        loss, mse_val, ce_val = combined_loss(pred, mask, target, y_groups, lambda_ce)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_mse += mse_val
        total_ce += ce_val
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches, total_ce / n_batches


def precompute_qhat(model, ds, key, edge_index_t, nmask, device, batch_size=4096):
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
            pred = model(nf[s_b], edge_index_t, gf[s_b], a_b, nmask)
            s_b_cpu, a_b_cpu, pred_cpu = s_b.cpu(), a_b.cpu(), pred.cpu()
            for i in range(end - start):
                s = states[s_b_cpu[i].item()]
                p = individuals[a_b_cpu[i].item()]
                q_hat.setdefault(s, {})[p] = pred_cpu[i].item()
    return q_hat


def evaluate_rollout(model, configs, struct_cache, edge_cache, nmask_cache, device, log):
    results = {}
    for fam, reg, pre in configs:
        key = f"{fam}_{reg}_{pre}_2gene"
        ds = build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        edge_index_t = torch.tensor(edge_cache[fam], device=device)
        nmask = nmask_cache[fam]

        q_hat = precompute_qhat(model, ds, key, edge_index_t, nmask, device)
        ratio2, L = q_rollout(q_hat, ds, log=log, trace=False)

        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"  TEST avg ratio2 = {avg:.4f}")
    return results, avg


def main(device="cpu", epochs=500, groups_per_batch=512, lambda_ce=1.0, mode="both", seed=0):
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

    log(f"\n{'='*60}\n[E9-2GENE-GNN-Q-COMBOPOOL] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}"
        f"  seed={seed}  lambda_ce={lambda_ce}  groups_per_batch={groups_per_batch}")

    dev = torch.device(device)

    struct_cache, edge_cache, nmask_cache = {}, {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam] = tg.build_edge_index(pedigree, individuals)
        nmask_np = build_neighbor_mask(edge_cache[fam], len(individuals))
        nmask_cache[fam] = torch.tensor(nmask_np, device=dev)

    model = GNNQBidirComboPool().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Generating {len(tg.TRAIN_CONFIGS)} train configs + building state-grouped tensors...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}
        per_config = []
        for fam, reg, pre in tg.TRAIN_CONFIGS:
            ds = build_config(fam, reg, pre)
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            s_idx, a_idx, y = build_qsa_index(ds, device=dev, cache_key=key)
            k_max = len(ds["individuals"])
            group_rows, group_mask, group_target = build_state_groups(s_idx, y, k_max)
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
            ep_loss = ep_mse = ep_ce = 0.0
            for fam, data in per_config:
                ei_c = edge_index_t[fam]
                nmask = nmask_cache[fam]
                loss, mse_v, ce_v = train_one_epoch(model, ei_c, nmask, data, opt, groups_per_batch, lambda_ce, dev)
                ep_loss += loss
                ep_mse += mse_v
                ep_ce += ce_v
            n = len(per_config)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  total={ep_loss/n:.5f}  mse={ep_mse/n:.5f}  ce={ep_ce/n:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), RESULTS_DIR / "gnn_q_e9_2gene.pt")
        log(f"    saved -> {RESULTS_DIR/'gnn_q_e9_2gene.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "gnn_q_e9_2gene.pt", map_location=dev))
        log(f"\n[3] Eval on {len(tg.TEST_CONFIGS)} test configs, "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, tg.TEST_CONFIGS, struct_cache, edge_cache, nmask_cache, dev, log)
        (RESULTS_DIR / "results_e9.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results_e9.json'}")

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--groups_per_batch", type=int, default=512)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, groups_per_batch=args.groups_per_batch,
         lambda_ce=args.lambda_ce, mode=args.mode, seed=args.seed)
