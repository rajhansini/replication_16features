"""Multi-seed variant of mlp/run.py — 3-gene MLP.

Separate script, does not modify or import mlp/run.py, so the original
script and its results/ outputs are untouched (backward compatible).
Only addition versus mlp/run.py: an explicit --seed argument
(torch.manual_seed before model init and training), and outputs written to
results/seed_runs/seed{N}/ instead of results/ directly, so repeated seeds
never collide with each other or with the original single run.

Everything else (architecture, data, split, epochs, batch size, optimizer,
loss, eval/rollout) is identical to mlp/run.py.
"""
from __future__ import annotations

import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE         = Path(__file__).resolve().parent
ROOT         = HERE.parent.parent
EXPERIMENTS  = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE.parent))

from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
from exputils.eval import rollout

GENES      = ("GeneA", "GeneB", "GeneC")
NODE_FEAT  = 3 * len(GENES) + 1 + 3   # 13
COST_DIM   = 3 * len(GENES) + 2        # 11

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"

ALLELE_REGIMES  = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]
TRAIN_FAMILIES  = ["Trio", "Nuclear"]
TEST_FAMILIES   = ["ThreeGeneration"]

TRAIN_KEYS = [
    f"{fam}_{reg}_{pre}_3gene"
    for fam in TRAIN_FAMILIES
    for reg in ALLELE_REGIMES
    for pre in ["Base", "Aggressive"]
]
TEST_KEYS = [
    f"{fam}_{reg}_{pre}_3gene"
    for fam in TEST_FAMILIES
    for reg in ALLELE_REGIMES
    for pre in ["Base", "Aggressive"]
]


# ── model (identical to mlp/run.py) ────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, node_feat=NODE_FEAT, cost_dim=COST_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat + cost_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats: torch.Tensor, cost_vec: torch.Tensor) -> torch.Tensor:
        pooled = node_feats.mean(dim=1)
        x      = torch.cat([pooled, cost_vec], dim=-1)
        return self.net(x).squeeze(-1)


# ── structural features (identical to mlp/run.py) ──────────────────────────────

def compute_structural_features(pedigree, individuals: list) -> np.ndarray:
    from collections import deque
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


# ── data loading (identical to mlp/run.py) ──────────────────────────────────────

def config_to_cost_vec(config) -> np.ndarray:
    vec = []
    for g in GENES:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def load_dataset(key: str, struct_feats: np.ndarray, device: torch.device) -> dict:
    pkl = CACHE_DIR / f"{key}.pkl"
    with open(pkl, "rb") as f:
        ds = pickle.load(f)

    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    X    = ds["X"].reshape(N, n_people, 3 * len(GENES))

    tested = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested[i, person_idx[person], 0] = 1.0

    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))

    nf = np.concatenate([X, tested, struct], axis=-1).astype(np.float32)

    cv = config_to_cost_vec(ds["config"])
    gf = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)

    y = ds["Y"].astype(np.float32)

    return {
        "nf": torch.tensor(nf, device=device),
        "gf": torch.tensor(gf, device=device),
        "y":  torch.tensor(y,  device=device),
    }


# ── training (identical to mlp/run.py) ──────────────────────────────────────────

def train(model, groups, epochs=500, lr=1e-3, batch_size=512,
          val_frac=0.2, device="cpu", log=print, print_every=50):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse       = nn.MSELoss()

    train_groups, val_groups = [], []
    for g in groups:
        N   = g["nf"].shape[0]
        n_v = max(1, int(N * val_frac))
        n_t = N - n_v
        train_groups.append({"nf": g["nf"][:n_t], "gf": g["gf"][:n_t], "y": g["y"][:n_t]})
        val_groups.append(  {"nf": g["nf"][n_t:], "gf": g["gf"][n_t:], "y": g["y"][n_t:]})

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
                pred = model(g["nf"][sl], g["gf"][sl])
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
                    pred = model(g["nf"][start:end], g["gf"][start:end])
                    val_loss += mse(pred, g["y"][start:end]).item() * (end - start)
        val_loss /= n_val

        if epoch % print_every == 0:
            log(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

    return model


# ── eval (identical to mlp/run.py) ──────────────────────────────────────────────

def precompute_vhat(model, ds, struct_feats, device, batch_size=4096):
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

    v_all = np.empty(N, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            nf_b = torch.tensor(nf_all[start:end], device=device)
            gf_b = torch.tensor(gf_all[start:end], device=device)
            v_all[start:end] = model(nf_b, gf_b).cpu().numpy()

    return {state: float(v_all[i]) for i, state in enumerate(states)}


def run_eval(model, struct_cache, device, log, results_dir):
    import json
    model.eval()
    results = {}

    for key in TEST_KEYS:
        fam = key.split("_")[0]
        log(f"\n{'─'*50}")
        log(f"[EVAL] {key}")
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        v_hat  = precompute_vhat(model, ds, struct_cache[fam], device)
        log(f"  V_hat range: [{min(v_hat.values()):.3f}, {max(v_hat.values()):.3f}]")

        ratio2, L = rollout(v_hat, ds, log=log, trace=False)

        log(f"  ratio2={ratio2:.4f}  L={L:.4f}  V*={ds['V_root']:.4f}")
        results[key] = {"ratio2": ratio2, "L": L, "V_root": ds["V_root"]}

    avg = float(np.mean([r["ratio2"] for r in results.values()]))
    log(f"\n{'='*50}")
    log(f"TEST avg ratio2 = {avg:.4f}")

    out = results_dir / "results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"Saved -> {out}")
    return results


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
    log(f"[MLP seed={seed}] {datetime.now().isoformat()}")
    log(f"architecture: mean-pool -> Linear({NODE_FEAT+COST_DIM}->32) -> ReLU -> Linear(32->1)")
    log(f"device={device}  epochs={epochs}  seed={seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = torch.device(device)

    struct_cache = {}
    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        sample_key = f"{fam}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        pedigree = generate_deterministic_pedigree(FAMILY_CASES[fam])
        struct_cache[fam] = compute_structural_features(pedigree, sample_ds["individuals"])

    model = MLP().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params}")

    log(f"\n[1] Loading {len(TRAIN_KEYS)} training configs...")
    train_groups = []
    for key in TRAIN_KEYS:
        fam = key.split("_")[0]
        train_groups.append(load_dataset(key, struct_cache[fam], dev))
    log(f"    states: {sum(g['nf'].shape[0] for g in train_groups):,}")

    log(f"\n[2] Training...")
    t0 = time.time()
    model = train(model, train_groups, epochs=epochs, device=device, log=log)
    log(f"    done in {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), results_dir / "mlp.pt")
    log(f"    saved -> {results_dir}/mlp.pt")

    log(f"\n[3] Eval on ThreeGeneration (unseen)...")
    run_eval(model, struct_cache, device, log, results_dir)

    log_f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs, seed=args.seed)
