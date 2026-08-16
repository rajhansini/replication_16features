"""2-gene GNN-Q — same embedding as two_gene/run.py's GNN (2 rounds message
passing), new head. Mirrors two_gene_mlp_q.py exactly except for the embedding.

Old:  V_hat(s)     <- mean-pool(h) || cost_vec,             h = 2 rounds of message passing
New:  Q_hat(s, a)  <- mean-pool(h) || h[a] || cost_vec,      same h, unchanged

Usage:
    python two_gene_gnn_q.py --mode both --epochs 200 --seed 0
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


tg = _load_module("two_gene_run_base2", EXP_ROOT / "two_gene" / "run.py")

from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402

NODE_FEAT = tg.NODE_FEAT   # 10
COST_DIM  = tg.COST_DIM    # 8
HIDDEN    = tg.HIDDEN      # 32
GENES     = tg.GENES

BASE_RESULTS_DIR = HERE / "results_2gene"
BASE_RESULTS_DIR.mkdir(exist_ok=True)


class GNNQ(nn.Module):
    """Identical message-passing to two_gene/run.py's GNN. Only the head differs."""

    def __init__(self, node_feat=NODE_FEAT, hidden=HIDDEN, cost_dim=COST_DIM, n_rounds=2):
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
        msgs = msg_fn(torch.cat([h[:, src, :], h[:, dst, :]], dim=-1))
        H = msgs.shape[-1]
        agg = torch.zeros(B, N, H, device=h.device)
        agg.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H), msgs)
        return upd_fn(torch.cat([h, agg], dim=-1))

    def embed(self, node_feats, edge_index):
        h = node_feats
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._message_pass(h, edge_index[0], edge_index[1], msg_fn, upd_fn)
        return h

    def forward(self, node_feats, edge_index, cost_vec, action_idx):
        h = self.embed(node_feats, edge_index)
        pooled = h.mean(dim=1)
        B = h.shape[0]
        candidate = h[torch.arange(B, device=h.device), action_idx]
        x = torch.cat([pooled, candidate, cost_vec], dim=-1)
        return self.head(x).squeeze(-1)


def build_config(fam, reg, pre):
    return tg.build_two_gene_dataset(
        family_label=fam,
        allele_freqs=tg.ALLELE_FREQ_REGIMES[reg],
        preset_label=pre,
        genes=GENES,
    )


def train_one_epoch(model, nf, ei, gf, state_idx, action_idx, y, opt, batch_size, device):
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
    version. See q_learning/gnn_q.py's precompute_qhat for the same approach.

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


def evaluate_rollout(model, configs, struct_cache, edge_cache, device, log):
    """Real policy evaluation via q_rollout — see q_learning/mlp_q.py's
    evaluate_rollout for the full rationale.
    """
    results = {}
    for fam, reg, pre in configs:
        key = f"{fam}_{reg}_{pre}_2gene"
        ds = build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]
        edge_index_t = torch.tensor(edge_cache[fam], device=device)

        q_hat = precompute_qhat(model, ds, key, edge_index_t, device)
        ratio2, L = q_rollout(q_hat, ds, log=log, trace=False)

        log(f"  [{key}]  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"  TEST avg ratio2 = {avg:.4f}")
    return results, avg


def main(device="cpu", epochs=500, batch_size=2048, mode="both", seed=0):
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

    log(f"\n{'='*60}\n[2GENE-GNN-Q] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}  seed={seed}")

    dev = torch.device(device)

    struct_cache, edge_cache = {}, {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)
        edge_cache[fam] = tg.build_edge_index(pedigree, individuals)

    model = GNNQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Generating {len(tg.TRAIN_CONFIGS)} train configs + building (s,a,Q*) tensors...")
        edge_index_t = {fam: torch.tensor(edge_cache[fam], device=dev) for fam in edge_cache}
        per_config = []
        for fam, reg, pre in tg.TRAIN_CONFIGS:
            ds = build_config(fam, reg, pre)
            ds["_struct_feats"] = struct_cache[fam]
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            s_idx, a_idx, y = build_qsa_index(ds, device=dev, cache_key=key)
            per_config.append((fam, base["nf"], base["gf"], s_idx, a_idx, y))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows")
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
                loss = train_one_epoch(model, nf_c, ei_c, gf_c, s_idx_c, a_idx_c, y_c, opt, batch_size, dev)
                ep_loss += loss * len(y_c)
                ep_count += len(y_c)
            if ep % 20 == 0 or ep == 1:
                log(f"    epoch {ep:4d}  mse={ep_loss/ep_count:.5f}")
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict()}, ckpt_path)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), RESULTS_DIR / "gnn_q_2gene.pt")
        log(f"    saved -> {RESULTS_DIR/'gnn_q_2gene.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "gnn_q_2gene.pt", map_location=dev))
        log(f"\n[3] Eval on {len(tg.TEST_CONFIGS)} test configs, "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, tg.TEST_CONFIGS, struct_cache, edge_cache, dev, log)
        (RESULTS_DIR / "results_gnn.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results_gnn.json'}")

    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, batch_size=args.batch_size,
         mode=args.mode, seed=args.seed)
