"""2-gene MLP-Q — same embedding as two_gene/run.py's MLP (mean-pool), new head.

Old:  V_hat(s)     <- mean-pool(node_feats) || cost_vec
New:  Q_hat(s, a)  <- mean-pool(node_feats) || node_feats[a] || cost_vec

Same design as q_learning/mlp_q.py (3-gene), adapted for two_gene/run.py's
on-the-fly dataset generation (no CACHE_DIR pickles — build_two_gene_dataset
runs exact DP fresh for each config, states/configs are much smaller than
3-gene so this is fast) and smaller dims (NODE_FEAT=10, COST_DIM=8).

Q* comes from qsa_data.build_qsa_index — exact, same formula as 3-gene, reused
unchanged (it only depends on ds["states"/"belief"/"config"/"V_star"], which
build_two_gene_dataset provides in the same shape as the 3-gene cache).

Eval: real greedy-policy rollout via q_rollout (exputils/eval.py) — the model's
own argmax_a Q_hat(s,a) choice drives the trajectory, branching over every true
outcome probability, memoized to termination. This replaces both the original
3-gene DP-forced-trajectory metric (model predictions never affected which
states got evaluated) and this script's earlier per-state-sample accuracy
metric (still single-step, no rollout, couldn't see compounding error either).
q_rollout produces the same ratio2/L used by the V(s) framework, so 2-gene and
3-gene Q(s,a) numbers are now directly comparable to each other and to V(s).

Usage:
    python two_gene_mlp_q.py --mode both --epochs 200 --seed 0
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


tg = _load_module("two_gene_run_base", EXP_ROOT / "two_gene" / "run.py")

from qsa_data import build_qsa_index  # noqa: E402
from exputils.eval import q_rollout  # noqa: E402

NODE_FEAT = tg.NODE_FEAT   # 10
COST_DIM  = tg.COST_DIM    # 8
GENES     = tg.GENES       # ("GeneA", "GeneB")

BASE_RESULTS_DIR = HERE / "results_2gene"
BASE_RESULTS_DIR.mkdir(exist_ok=True)


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
        pooled = node_feats.mean(dim=1)
        B = node_feats.shape[0]
        candidate = node_feats[torch.arange(B, device=node_feats.device), action_idx]
        x = torch.cat([pooled, candidate, cost_vec], dim=-1)
        return self.net(x).squeeze(-1)


def build_config(fam, reg, pre):
    """Generate one 2-gene config's ds on the fly (fast — small state spaces)."""
    return tg.build_two_gene_dataset(
        family_label=fam,
        allele_freqs=tg.ALLELE_FREQ_REGIMES[reg],
        preset_label=pre,
        genes=GENES,
    )


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
    """Batched Q_hat(s, person) for every (state, untested person) pair in ds.
    Same approach as q_learning/mlp_q.py's precompute_qhat, reused unchanged
    here since the model signature (nf, gf, action_idx) is identical.

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


def evaluate_rollout(model, configs, struct_cache, device, log):
    """Real policy evaluation via q_rollout — see q_learning/mlp_q.py's
    evaluate_rollout for the full rationale.
    """
    results = {}
    for fam, reg, pre in configs:
        key = f"{fam}_{reg}_{pre}_2gene"
        ds = build_config(fam, reg, pre)
        base = tg.ds_to_tensors(ds, struct_cache[fam], device)
        ds["_nf"], ds["_gf"] = base["nf"], base["gf"]

        q_hat = precompute_qhat(model, ds, key, device)
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

    log(f"\n{'='*60}\n[2GENE-MLP-Q] {datetime.now().isoformat()}  mode={mode}  epochs={epochs}  seed={seed}")

    dev = torch.device(device)

    struct_cache = {}
    for fam in tg.TRAIN_FAMILIES + tg.TEST_FAMILIES:
        pedigree = tg.generate_deterministic_pedigree(tg.FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = tg.compute_structural_features(pedigree, individuals)

    model = MLPQ().to(dev)
    log(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    if mode in ("train", "both"):
        log(f"\n[1] Generating {len(tg.TRAIN_CONFIGS)} train configs + building (s,a,Q*) tensors...")
        per_config = []
        for fam, reg, pre in tg.TRAIN_CONFIGS:
            ds = build_config(fam, reg, pre)
            ds["_struct_feats"] = struct_cache[fam]
            base = tg.ds_to_tensors(ds, struct_cache[fam], dev)
            key = f"{fam}_{reg}_{pre}_2gene"
            s_idx, a_idx, y = build_qsa_index(ds, device=dev, cache_key=key)
            per_config.append((base["nf"], base["gf"], s_idx, a_idx, y))
            log(f"    [{key}] {len(ds['states']):,} states -> {len(y):,} (s,a) rows")
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
        torch.save(model.state_dict(), RESULTS_DIR / "mlp_q_2gene.pt")
        log(f"    saved -> {RESULTS_DIR/'mlp_q_2gene.pt'}")

    if mode in ("eval", "both"):
        if mode == "eval":
            model.load_state_dict(torch.load(RESULTS_DIR / "mlp_q_2gene.pt", map_location=dev))
        log(f"\n[3] Eval on {len(tg.TEST_CONFIGS)} test configs, "
            f"real greedy-Q(s,a) rollout vs exact DP (q_rollout)...")
        results, avg = evaluate_rollout(model, tg.TEST_CONFIGS, struct_cache, dev, log)
        (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
        log(f"    saved -> {RESULTS_DIR/'results.json'}")

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
