"""GNN with proper train/val/test split.

Split:
    TRAIN : Trio + Nuclear  × [LowHigh, MediumEven, LowLow, HighHigh, MixedA] × [Base, Aggressive]
            = 5 regimes × 2 families × 2 presets = 20 configs
    VAL   : Trio + Nuclear  × MixedB × [Base, Aggressive]
            = 1 regime  × 2 families × 2 presets = 4 configs
            Val loss now measures regime-generalization on unseen cost/freq setting.
    TEST  : ThreeGeneration × all 6 regimes × [Base, Aggressive]
            = 12 configs (unseen topology)

Diff from gnn/run.py:
    - MixedB held out of training → used as proper val set
    - Early stopping on val loss (patience=50 epochs)
    - Val loss printed every epoch (not just every 50)
"""
from __future__ import annotations

import pickle
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE.parent))

from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from exputils.eval import rollout

GENES      = ("GeneA", "GeneB", "GeneC")
NODE_FEAT  = 3 * len(GENES) + 1 + 3   # 13
COST_DIM   = 3 * len(GENES) + 2        # 11
HIDDEN     = 32
N_ROUNDS   = 2

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

TRAIN_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA"]
VAL_REGIMES   = ["MixedB"]
ALL_REGIMES   = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]
PEDIGREE_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES     = ["ThreeGeneration"]

TRAIN_KEYS = [
    f"{fam}_{reg}_{pre}_3gene"
    for fam in PEDIGREE_FAMILIES
    for reg in TRAIN_REGIMES
    for pre in ["Base", "Aggressive"]
]
VAL_KEYS = [
    f"{fam}_{reg}_{pre}_3gene"
    for fam in PEDIGREE_FAMILIES
    for reg in VAL_REGIMES
    for pre in ["Base", "Aggressive"]
]
TEST_KEYS = [
    f"ThreeGeneration_{reg}_{pre}_3gene"
    for reg in ALL_REGIMES
    for pre in ["Base", "Aggressive"]
]


# ── model ─────────────────────────────────────────────────────────────────────

class GNN(nn.Module):
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
            nn.Linear(hidden + cost_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _message_pass(self, h, src, dst, msg_fn, upd_fn):
        B, N, _ = h.shape
        msg_in  = torch.cat([h[:, src, :], h[:, dst, :]], dim=-1)
        msgs    = msg_fn(msg_in)
        H       = msgs.shape[-1]
        agg     = torch.zeros(B, N, H, device=h.device)
        idx     = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)
        agg.scatter_add_(1, idx, msgs)
        return upd_fn(torch.cat([h, agg], dim=-1))

    def forward(self, node_feats, edge_index, cost_vec):
        h   = node_feats
        src = edge_index[0]
        dst = edge_index[1]
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._message_pass(h, src, dst, msg_fn, upd_fn)
        pooled = h.sum(dim=1)
        x      = torch.cat([pooled, cost_vec], dim=-1)
        return self.head(x).squeeze(-1)


# ── structural / edge features ─────────────────────────────────────────────────

def compute_structural_features(pedigree, individuals):
    idx            = {p: i for i, p in enumerate(individuals)}
    n              = len(individuals)
    n_parents_arr  = np.zeros(n, dtype=np.float32)
    n_children_arr = np.zeros(n, dtype=np.float32)
    depth_arr      = np.zeros(n, dtype=np.float32)
    children_map   = {p: [] for p in individuals}

    for child in pedigree.get_offspring():
        if child not in idx:
            continue
        parents = [p for p in pedigree.get_parents(child) if p in idx]
        n_parents_arr[idx[child]] = float(len(parents))
        for p in parents:
            n_children_arr[idx[p]] += 1.0
            children_map[p].append(child)

    in_deg = {p: int(n_parents_arr[idx[p]]) for p in individuals}
    queue  = deque([p for p in individuals if in_deg[p] == 0])
    while queue:
        person = queue.popleft()
        for child in children_map[person]:
            depth_arr[idx[child]] = max(depth_arr[idx[child]], depth_arr[idx[person]] + 1.0)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    return np.stack([n_parents_arr, n_children_arr, depth_arr], axis=-1)


def build_edge_index(pedigree, individuals):
    idx = {p: i for i, p in enumerate(individuals)}
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    if srcs:
        return np.array([srcs, dsts], dtype=np.int64)
    return np.zeros((2, 0), dtype=np.int64)


# ── data loading ───────────────────────────────────────────────────────────────

def config_to_cost_vec(config):
    vec = []
    for g in GENES:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def load_dataset(key, struct_feats, edge_index, device):
    with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
        ds = pickle.load(f)
    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    X      = ds["X"].reshape(N, n_people, 3 * len(GENES))
    tested = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested[i, person_idx[person], 0] = 1.0
    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))
    nf     = np.concatenate([X, tested, struct], axis=-1).astype(np.float32)

    cv = config_to_cost_vec(ds["config"])
    gf = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)
    y  = ds["Y"].astype(np.float32)

    return {
        "nf": torch.tensor(nf, device=device),
        "gf": torch.tensor(gf, device=device),
        "y":  torch.tensor(y,  device=device),
        "ei": torch.tensor(edge_index, device=device),
    }


# ── training with early stopping ───────────────────────────────────────────────

def train(model, train_groups, val_groups, epochs=500, lr=1e-3, batch_size=512,
          patience=50, device="cpu", log=print, results_dir=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse       = nn.MSELoss()

    n_train = sum(g["nf"].shape[0] for g in train_groups)
    n_val   = sum(g["nf"].shape[0] for g in val_groups)

    best_val    = float("inf")
    best_state  = None
    no_improve  = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_groups:
            M    = g["nf"].shape[0]
            perm = torch.randperm(M, device=device)
            for start in range(0, M, batch_size):
                sl = perm[start: start + batch_size]
                optimizer.zero_grad()
                pred = model(g["nf"][sl], g["ei"], g["gf"][sl])
                loss = mse(pred, g["y"][sl])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(sl)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in val_groups:
                M = g["nf"].shape[0]
                for start in range(0, M, batch_size):
                    end  = min(start + batch_size, M)
                    pred = model(g["nf"][start:end], g["ei"], g["gf"][start:end])
                    val_loss += mse(pred, g["y"][start:end]).item() * (end - start)
        val_loss /= n_val

        if epoch % 50 == 0:
            log(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
                + ("  *best*" if val_loss < best_val else f"  (no improve {no_improve})"))

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if results_dir:
                torch.save(best_state, results_dir / "gnn_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  Early stopping at epoch {epoch} (best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── eval ───────────────────────────────────────────────────────────────────────

def precompute_vhat(model, ds, struct_feats, edge_index, device, batch_size=4096):
    model.eval()
    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    X      = ds["X"].reshape(N, n_people, 3 * len(GENES))
    tested = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested[i, person_idx[person], 0] = 1.0
    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))
    nf_all = np.concatenate([X, tested, struct], axis=-1).astype(np.float32)

    cv     = config_to_cost_vec(ds["config"])
    gf_all = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)
    ei_t   = torch.tensor(edge_index, device=device)

    v_all = np.empty(N, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            nf_b = torch.tensor(nf_all[start:end], device=device)
            gf_b = torch.tensor(gf_all[start:end], device=device)
            v_all[start:end] = model(nf_b, ei_t, gf_b).cpu().numpy()

    return {state: float(v_all[i]) for i, state in enumerate(states)}


def run_eval(model, struct_cache, edge_cache, device, log):
    import json
    model.eval()
    results = {}

    for key in TEST_KEYS:
        fam = key.split("_")[0]
        log(f"\n{'─'*50}")
        log(f"[EVAL] {key}")
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        v_hat = precompute_vhat(model, ds, struct_cache[fam], edge_cache[fam], device)
        log(f"  V_hat range: [{min(v_hat.values()):.3f}, {max(v_hat.values()):.3f}]")
        log(f"  --- rollout trace ---")

        ratio2, L = rollout(v_hat, ds, log=log, trace=True)

        log(f"  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"\n{'='*50}")
    log(f"TEST avg ratio2 = {avg:.4f}")

    out = HERE / "results" / "results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"Saved -> {out}")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main(device="cpu", epochs=500, mode="both"):
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "run.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[GNN_proper_val] {datetime.now().isoformat()}  mode={mode}")
    log(f"TRAIN: Trio+Nuclear × {TRAIN_REGIMES} = {len(TRAIN_KEYS)} configs")
    log(f"VAL  : Trio+Nuclear × {VAL_REGIMES}   = {len(VAL_KEYS)} configs  (held-out regime)")
    log(f"TEST : ThreeGeneration × all 6 regimes = {len(TEST_KEYS)} configs")
    log(f"architecture: MP x{N_ROUNDS} -> sum pool -> Linear({HIDDEN+COST_DIM}->16) -> ReLU -> Linear(16->1)")
    log(f"device={device}  epochs={epochs}")

    dev = torch.device(device)

    # pedigree features for all families we need
    all_families = list(set(PEDIGREE_FAMILIES + TEST_FAMILIES))
    struct_cache = {}
    edge_cache   = {}
    for fam in all_families:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree    = generate_deterministic_pedigree(FAMILY_CASES[fam])
        individuals = sample_ds["individuals"]
        struct_cache[fam] = compute_structural_features(pedigree, individuals)
        edge_cache[fam]   = build_edge_index(pedigree, individuals)
        log(f"  [{fam}] {len(individuals)} people, {edge_cache[fam].shape[1]} edges")

    model    = GNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params}")

    if mode in ("train", "both"):
        log(f"\n[1] Loading {len(TRAIN_KEYS)} train configs...")
        train_groups = []
        for key in TRAIN_KEYS:
            fam = key.split("_")[0]
            train_groups.append(load_dataset(key, struct_cache[fam], edge_cache[fam], dev))
        log(f"    train states: {sum(g['nf'].shape[0] for g in train_groups):,}")

        log(f"\n[2] Loading {len(VAL_KEYS)} val configs (MixedB — unseen regime)...")
        val_groups = []
        for key in VAL_KEYS:
            fam = key.split("_")[0]
            val_groups.append(load_dataset(key, struct_cache[fam], edge_cache[fam], dev))
        log(f"    val states  : {sum(g['nf'].shape[0] for g in val_groups):,}")

        log(f"\n[3] Training (early stop patience=50)...")
        t0 = time.time()
        model = train(model, train_groups, val_groups, epochs=epochs,
                      device=device, log=log, results_dir=results_dir)
        log(f"    done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), results_dir / "gnn.pt")
        log(f"    saved -> results/gnn.pt")

    if mode in ("eval", "both"):
        ckpt = results_dir / "gnn_best.pt"
        if not ckpt.exists():
            ckpt = results_dir / "gnn.pt"
        if mode == "eval":
            model.load_state_dict(torch.load(ckpt, map_location=device))
            log(f"Loaded weights from {ckpt}")
        log(f"\n[4] Eval on ThreeGeneration (unseen topology)...")
        run_eval(model, struct_cache, edge_cache, device, log)

    log_f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--mode", default="both", choices=["train", "eval", "both"])
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, mode=args.mode)
