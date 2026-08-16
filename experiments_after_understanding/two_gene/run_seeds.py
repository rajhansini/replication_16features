"""Multi-seed variant of two_gene/run.py — 2-gene MLP + GNN.

Separate script, does not modify or import two_gene/run.py, so the original
script and its results/ outputs are untouched (backward compatible).
Only addition versus two_gene/run.py: an explicit --seed argument
(torch.manual_seed before model init and training), and outputs written to
results/seed_runs/seed{N}/ instead of results/ directly.

Everything else (architecture, data, split, epochs, batch size, optimizer,
loss, eval/rollout) is identical to two_gene/run.py (which already matches
mlp/run.py and gnn/run.py's val_frac=0.2 split, fixed this session).
"""
from __future__ import annotations

import json
import random
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

from shared.data_gen import FAMILY_CASES, build_two_gene_dataset
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from exputils.eval import rollout

GENES     = ("GeneA", "GeneB")
NODE_FEAT = 3 * len(GENES) + 1 + 3   # 10
COST_DIM  = 3 * len(GENES) + 2        # 8
HIDDEN    = 32

ALLELE_FREQ_REGIMES = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12},
}

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration"]
PRESETS_LIST   = ["Base", "Aggressive"]

TRAIN_CONFIGS = [
    (fam, reg, pre)
    for fam in TRAIN_FAMILIES
    for reg in ALLELE_FREQ_REGIMES
    for pre in PRESETS_LIST
]
TEST_CONFIGS = [
    (fam, reg, pre)
    for fam in TEST_FAMILIES
    for reg in ALLELE_FREQ_REGIMES
    for pre in PRESETS_LIST
]


# ── structural / edge helpers (identical to two_gene/run.py) ──────────────────

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


# ── data → tensors (identical to two_gene/run.py) ──────────────────────────────

def config_to_cost_vec(config):
    vec = []
    for g in GENES:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def ds_to_tensors(ds, struct_feats, device):
    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    X = ds["X"].reshape(N, n_people, 3 * len(GENES)).astype(np.float32)

    tested_arr = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_arr[i, person_idx[person], 0] = 1.0

    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))
    nf = np.concatenate([X, tested_arr, struct], axis=-1).astype(np.float32)

    cv = config_to_cost_vec(ds["config"])
    gf = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)

    y = ds["Y"].astype(np.float32)

    return {
        "nf": torch.tensor(nf, device=device),
        "gf": torch.tensor(gf, device=device),
        "y":  torch.tensor(y,  device=device),
    }


# ── models (identical to two_gene/run.py) ───────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NODE_FEAT + COST_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, node_feats, cost_vec):
        pooled = node_feats.mean(dim=1)
        return self.net(torch.cat([pooled, cost_vec], dim=-1)).squeeze(-1)


class GNN(nn.Module):
    def __init__(self, n_rounds=2):
        super().__init__()
        self.n_rounds   = n_rounds
        self.msg_layers = nn.ModuleList()
        self.upd_layers = nn.ModuleList()
        in_dim = NODE_FEAT
        for _ in range(n_rounds):
            self.msg_layers.append(nn.Sequential(nn.Linear(in_dim * 2, HIDDEN), nn.ReLU()))
            self.upd_layers.append(nn.Sequential(nn.Linear(in_dim + HIDDEN, HIDDEN), nn.ReLU()))
            in_dim = HIDDEN
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + COST_DIM, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def _message_pass(self, h, src, dst, msg_fn, upd_fn):
        B, N, _ = h.shape
        msgs = msg_fn(torch.cat([h[:, src, :], h[:, dst, :]], dim=-1))
        H    = msgs.shape[-1]
        agg  = torch.zeros(B, N, H, device=h.device)
        agg.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H), msgs)
        return upd_fn(torch.cat([h, agg], dim=-1))

    def forward(self, node_feats, edge_index, cost_vec):
        h = node_feats
        for msg_fn, upd_fn in zip(self.msg_layers, self.upd_layers):
            h = self._message_pass(h, edge_index[0], edge_index[1], msg_fn, upd_fn)
        pooled = h.mean(dim=1)
        return self.head(torch.cat([pooled, cost_vec], dim=-1)).squeeze(-1)


# ── training (identical to two_gene/run.py, post val-split fix) ────────────────

def train_model(model, groups, is_gnn, epochs=300, lr=1e-3,
                batch_size=512, val_frac=0.2, device="cpu", log=print, print_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse       = nn.MSELoss()

    train_groups, val_groups = [], []
    for g in groups:
        N   = g["nf"].shape[0]
        n_v = max(1, int(N * val_frac))
        n_t = N - n_v
        tg = {"nf": g["nf"][:n_t], "gf": g["gf"][:n_t], "y": g["y"][:n_t]}
        vg = {"nf": g["nf"][n_t:], "gf": g["gf"][n_t:], "y": g["y"][n_t:]}
        if is_gnn:
            tg["ei"] = g["ei"]
            vg["ei"] = g["ei"]
        train_groups.append(tg)
        val_groups.append(vg)

    n_train = sum(g["nf"].shape[0] for g in train_groups)
    n_val   = sum(g["nf"].shape[0] for g in val_groups)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_groups:
            M    = g["nf"].shape[0]
            perm = torch.randperm(M, device=device)
            for start in range(0, M, batch_size):
                sl = perm[start: start + batch_size]
                optimizer.zero_grad()
                pred = model(g["nf"][sl], g["ei"], g["gf"][sl]) if is_gnn \
                       else model(g["nf"][sl], g["gf"][sl])
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
                    pred = model(g["nf"][start:end], g["ei"], g["gf"][start:end]) if is_gnn \
                           else model(g["nf"][start:end], g["gf"][start:end])
                    val_loss += mse(pred, g["y"][start:end]).item() * (end - start)
        val_loss /= n_val

        if epoch % print_every == 0:
            log(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

    return model


# ── eval (identical to two_gene/run.py) ─────────────────────────────────────────

def precompute_vhat(model, ds, struct_feats, device, edge_index=None, batch_size=4096):
    model.eval()
    tensors = ds_to_tensors(ds, struct_feats, device)
    N       = tensors["nf"].shape[0]
    v_all   = np.empty(N, dtype=np.float64)

    ei_t = torch.tensor(edge_index, device=device) if edge_index is not None else None

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            nf_b = tensors["nf"][start:end]
            gf_b = tensors["gf"][start:end]
            if ei_t is not None:
                v_all[start:end] = model(nf_b, ei_t, gf_b).cpu().numpy()
            else:
                v_all[start:end] = model(nf_b, gf_b).cpu().numpy()

    return {state: float(v_all[i]) for i, state in enumerate(ds["states"])}


# ── main ───────────────────────────────────────────────────────────────────────

def main(device="cpu", epochs=500, seed=0):
    results_dir = HERE / "results" / "seed_runs" / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_f = open(results_dir / "run.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[2-GENE seed={seed}] {datetime.now().isoformat()}")
    log(f"genes     : {GENES}")
    log(f"device={device}  epochs={epochs}  seed={seed}  val_frac=0.2 (per-config, positional)")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = torch.device(device)

    struct_cache = {}
    edge_cache   = {}
    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        pedigree    = generate_deterministic_pedigree(FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = compute_structural_features(pedigree, individuals)
        edge_cache[fam]   = build_edge_index(pedigree, individuals)

    log(f"\n[1] Generating train/test datasets (build_two_gene_dataset)...")
    train_datasets = []
    for fam, reg, pre in TRAIN_CONFIGS:
        ds = build_two_gene_dataset(family_label=fam, allele_freqs=ALLELE_FREQ_REGIMES[reg],
                                     preset_label=pre, genes=GENES)
        train_datasets.append((fam, reg, pre, ds))
    log(f"  total train states: {sum(len(d[3]['states']) for d in train_datasets):,}")

    test_datasets = []
    for fam, reg, pre in TEST_CONFIGS:
        ds = build_two_gene_dataset(family_label=fam, allele_freqs=ALLELE_FREQ_REGIMES[reg],
                                     preset_label=pre, genes=GENES)
        test_datasets.append((fam, reg, pre, ds))
    log(f"  total test states: {sum(len(d[3]['states']) for d in test_datasets):,}")

    results = {"mlp": {}, "gnn": {}}

    for model_name, is_gnn in [("mlp", False), ("gnn", True)]:
        log(f"\n{'='*60}")
        log(f"[{model_name.upper()} seed={seed}]")

        model = (GNN() if is_gnn else MLP()).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  parameters: {n_params}")

        groups = []
        for fam, reg, pre, ds in train_datasets:
            g = ds_to_tensors(ds, struct_cache[fam], dev)
            if is_gnn:
                g["ei"] = torch.tensor(edge_cache[fam], device=dev)
            groups.append(g)

        log(f"\n  Training...")
        t0 = time.time()
        train_model(model, groups, is_gnn=is_gnn, epochs=epochs, device=device, log=log)
        log(f"  done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), results_dir / f"{model_name}.pt")

        log(f"\n  Evaluating on ThreeGeneration (unseen topology)...")
        model.eval()
        for fam, reg, pre, ds in test_datasets:
            key = f"{fam}_{reg}_{pre}_2gene"
            ei = edge_cache[fam] if is_gnn else None
            v_hat = precompute_vhat(model, ds, struct_cache[fam], dev, edge_index=ei)
            ratio2, L = rollout(v_hat, ds, log=log, trace=False)
            log(f"  [EVAL] {key}  ratio2={ratio2:.4f}  L={L:.4f}")
            results[model_name][key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

        avg = np.mean([r["ratio2"] for r in results[model_name].values()])
        log(f"\n  {model_name.upper()} avg ratio2 = {avg:.4f}")

    out = results_dir / "results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nSaved -> {out}")
    log_f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, seed=args.seed)
