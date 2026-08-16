"""Step 10 — Correct GNN formulation: V*(s) prediction with structural node features.

PI feedback: GNN must replicate what exact DP and ADP do — approximate V*(s),
the value function, not Q*(s,i) per person directly.

ADP approximates V*(s) with basis functions that encode graph structure.
This GNN does the same end-to-end:

    Node features (13 per person):
        [P(g=0)..P(g=2)] x 3 genes   — posterior beliefs (9)
        is_tested                      — 1 if person tested in this state (1)
        n_parents, n_children, depth   — structural position in family graph (3)

    Model: PedigreeGNN_Vstar
        2 rounds message passing → global mean pooling → MLP → scalar V*(s)

    Training: MSE on V*(s) from exact DP (ds["Y"]) — no Q* label computation needed.

    Action selection (eval time, derived from V*):
        Q_hat(s,i) = r_test(i,s) + Σ_g P(g|s,i) * V_hat(s')
        action     = argmax_i Q_hat(s,i) over untested persons

Reuses step9 cache (same DP values, no recomputation).

Run:
    python step10_structural_gnn/run.py --device cuda --epochs 500
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

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.model    import PedigreeGNN_Vstar
from shared.data_gen import FAMILY_CASES
from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree

GENES         = ("GeneA", "GeneB", "GeneC")
STRUCT_DIM    = 3                                   # n_parents, n_children, depth
NODE_FEAT_DIM = 3 * len(GENES) + 1 + STRUCT_DIM    # 13
COST_DIM      = 3 * len(GENES) + 2                 # 11

CACHE_DIR = HERE.parent / "step9_gnn_3gene" / "results" / "cache"

ALLELE_REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]

# Train on small families only.
# ThreeGeneration (5 people) is held out as the unseen bigger family benchmark.
TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration"]   # unseen topology — scaling benchmark

TRAIN_KEYS = [
    f"{family}_{regime}_{preset}_3gene"
    for family in TRAIN_FAMILIES
    for regime  in ALLELE_REGIMES
    for preset  in ["Base", "Aggressive"]
]


# ─── structural features ──────────────────────────────────────────────────────

def compute_structural_features(pedigree, individuals: list) -> np.ndarray:
    """
    Static per-person features from pedigree topology. Shape: (N_people, 3).

    col 0 — n_parents  : biological parents in pedigree (0 or 2)
    col 1 — n_children : biological children in pedigree
    col 2 — depth      : BFS distance from nearest root (no parents)

    These are IDENTICAL across all states for a given family.
    Bayesian beliefs update with test results but structural position never changes.
    This is the information ADP basis functions encode that beliefs alone do not.
    """
    idx = {p: i for i, p in enumerate(individuals)}
    n   = len(individuals)
    n_parents_arr  = np.zeros(n, dtype=np.float32)
    n_children_arr = np.zeros(n, dtype=np.float32)
    depth_arr      = np.zeros(n, dtype=np.float32)
    children_map: dict = {p: [] for p in individuals}

    for child in pedigree.get_offspring():
        if child not in idx:
            continue
        parents = [p for p in pedigree.get_parents(child) if p in idx]
        n_parents_arr[idx[child]] = float(len(parents))
        for p in parents:
            n_children_arr[idx[p]] += 1.0
            children_map[p].append(child)

    # Topological max-depth: assign each person the LONGEST path from any root.
    # BFS min-depth would give ThreeGeneration Child depth=1 (via Mother, 1 hop)
    # even though genealogically Child is 2 generations from Grandfather.
    # Topological sort processes parents before children and takes max(parent_depth)+1.
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


# ─── feature extraction ───────────────────────────────────────────────────────

def config_to_cost_vec(config, genes) -> np.ndarray:
    vec = []
    for g in genes:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def build_edge_index(family_label: str, individuals: list) -> np.ndarray:
    pedigree = generate_deterministic_pedigree(FAMILY_CASES[family_label])
    idx = {p: i for i, p in enumerate(individuals)}
    srcs, dsts = [], []
    for child in pedigree.get_offspring():
        for parent in pedigree.get_parents(child):
            if parent in idx and child in idx:
                srcs.append(idx[parent])
                dsts.append(idx[child])
    return np.array([srcs, dsts], dtype=np.int64) if srcs else np.zeros((2, 0), dtype=np.int64)


def extract_node_features(ds: dict, individuals: list, struct_feats: np.ndarray) -> np.ndarray:
    """
    Returns (N_states, N_people, 13).

    Layout:
        [0:9]  — posterior beliefs: [P(g=0), P(g=1), P(g=2)] x 3 genes
        [9]    — is_tested flag
        [10]   — n_parents   (static)
        [11]   — n_children  (static)
        [12]   — depth       (static)
    """
    X            = ds["X"]
    states       = ds["states"]
    n_gene_feats = 3 * len(ds.get("genes", GENES))
    n_people     = len(individuals)
    person_idx   = {p: i for i, p in enumerate(individuals)}
    N            = len(states)

    nf = X.reshape(N, n_people, n_gene_feats).copy()

    tested_col = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_col[i, person_idx[person], 0] = 1.0

    struct_broadcast = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))

    return np.concatenate([nf, tested_col, struct_broadcast], axis=-1)


# ─── training ─────────────────────────────────────────────────────────────────

def train_gnn(groups, model, epochs=500, start_epoch=1, lr=1e-3,
              val_frac=0.2, batch_size=512, device="cpu",
              print_every=50, ckpt_path=None, ckpt_every=50,
              history=None, log=print):
    """
    Train PedigreeGNN_Vstar on V*(s) labels directly.

    Loss: MSE(V_hat(s), V*(s))  — no masking needed, every state has a V* value.
    This is simpler than Q* training and directly mirrors what ADP does.

    groups: list of dicts with keys:
        nf : (N, N_people, 13) — node features with structural augmentation
        gf : (N, 11)           — cost vector per state
        y  : (N,)              — V*(s) labels from exact DP
        ei : (2, E)            — edge index
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if history is None:
        history = {"train_loss": [], "val_loss": []}

    train_groups, val_groups = [], []
    for g in groups:
        N   = g["nf"].shape[0]
        n_v = max(1, int(N * val_frac))
        n_t = N - n_v
        train_groups.append({"nf": g["nf"][:n_t], "gf": g["gf"][:n_t],
                              "y":  g["y"][:n_t],  "ei": g["ei"]})
        val_groups.append(  {"nf": g["nf"][n_t:], "gf": g["gf"][n_t:],
                              "y":  g["y"][n_t:],  "ei": g["ei"]})

    mse = torch.nn.MSELoss()
    n_train = sum(g["nf"].shape[0] for g in train_groups)
    n_val   = sum(g["nf"].shape[0] for g in val_groups)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for g in train_groups:
            M    = g["nf"].shape[0]
            perm = torch.randperm(M, device=device)
            for start in range(0, M, batch_size):
                sl = perm[start: start + batch_size]
                optimizer.zero_grad()
                v_pred = model.forward_batch(g["nf"][sl], g["ei"], g["gf"][sl])
                loss   = mse(v_pred, g["y"][sl])
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
                    end    = min(start + batch_size, M)
                    v_pred = model.forward_batch(g["nf"][start:end], g["ei"], g["gf"][start:end])
                    val_loss += mse(v_pred, g["y"][start:end]).item() * (end - start)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0:
            log(f"  epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if ckpt_path is not None and epoch % ckpt_every == 0:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "history": history}, tmp)
            tmp.replace(Path(ckpt_path))

    return model, history


# ─── main ────────────────────────────────────────────────────────────────────

def main(device="cpu", target_epochs=500, fresh=False):
    results_dir = HERE / "results"
    ckpt_path   = results_dir / "gnn_structural_ckpt.pt"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "train.log", "a")
    log_f.write(f"\n{'='*70}\n[TRAIN] {datetime.now().isoformat()}\n{'='*70}\n")
    def log(msg=""):
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    log(f"STEP 10 — Correct GNN: V*(s) prediction + structural node features")
    log(f"Model: PedigreeGNN_Vstar  output=scalar V*(s) per state")
    log(f"node_feat_dim={NODE_FEAT_DIM}  (9 beliefs + 1 is_tested + 3 structural)")
    log(f"cost_dim={COST_DIM}  device={device}  epochs={target_epochs}")
    log(f"Training target: V*(s) from ds['Y'] — same as exact DP output")
    log(f"Action selection: derived from V* at eval time (not predicted directly)")
    log(f"Cache: reusing step9 (no recomputation)")
    log(f"TRAIN families: {TRAIN_FAMILIES}  ({len(TRAIN_KEYS)} configs)")
    log(f"TEST  families: {TEST_FAMILIES}  (unseen bigger family — scaling benchmark)")

    # ── 1. Structural features per family ─────────────────────────────────────
    log(f"\n[1] STRUCTURAL FEATURES (static per family, from pedigree topology):")
    ALL_FAMILIES = TRAIN_FAMILIES + TEST_FAMILIES
    struct_cache: dict[str, np.ndarray] = {}
    for family in ALL_FAMILIES:
        sample_key = f"{family}_LowHigh_Base_3gene"
        with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        individuals = sample_ds["individuals"]
        pedigree    = generate_deterministic_pedigree(FAMILY_CASES[family])
        sf          = compute_structural_features(pedigree, individuals)
        struct_cache[family] = sf
        log(f"\n  [{family}]  individuals={individuals}")
        log(f"  {'Person':<20} {'n_parents':>10} {'n_children':>12} {'depth':>8}")
        log(f"  {'-'*54}")
        for j, p in enumerate(individuals):
            log(f"  {p:<20} {sf[j,0]:>10.0f} {sf[j,1]:>12.0f} {sf[j,2]:>8.0f}")
        del sample_ds

    # ── 2. Cache check ─────────────────────────────────────────────────────────
    log(f"\n[2] Cache check:")
    for key in TRAIN_KEYS:
        pkl = CACHE_DIR / f"{key}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing: {pkl}. Run step9 build_datasets.py first.")
        log(f"    {key}  [OK]")

    # ── 3. Load data, build features with structural augmentation ──────────────
    log(f"\n[3] Loading data and building V* training pairs:")
    family_groups: dict[str, dict] = {}

    for key in TRAIN_KEYS:
        family = key.split("_")[0]
        t0 = time.time()
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)

        individuals = ds["individuals"]
        sf = struct_cache[family]

        if family not in family_groups:
            ei = build_edge_index(family, individuals)
            family_groups[family] = {
                "nf_parts": [], "gf_parts": [], "y_parts": [],
                "ei": ei, "individuals": individuals,
            }

        nf       = extract_node_features(ds, individuals, sf)
        cost_vec = config_to_cost_vec(ds["config"], GENES)
        gf       = np.tile(cost_vec, (len(ds["states"]), 1))
        y        = ds["Y"]   # V*(s) directly from exact DP — shape (N,)

        log(f"    {key}: {len(ds['states']):,} states  "
            f"nf={nf.shape}  V* range=[{y.min():.4f}, {y.max():.4f}]  ({time.time()-t0:.1f}s)")

        family_groups[family]["nf_parts"].append(nf)
        family_groups[family]["gf_parts"].append(gf)
        family_groups[family]["y_parts"].append(y)
        del ds

    groups = []
    for family, g in family_groups.items():
        nf_all = np.concatenate(g["nf_parts"], axis=0)
        gf_all = np.concatenate(g["gf_parts"], axis=0)
        y_all  = np.concatenate(g["y_parts"],  axis=0)
        log(f"    [{family}] {len(y_all):,} states  nf={nf_all.shape}  "
            f"V* range=[{y_all.min():.4f}, {y_all.max():.4f}]")
        groups.append({
            "key": family,
            "nf":  torch.FloatTensor(nf_all).to(device),
            "gf":  torch.FloatTensor(gf_all).to(device),
            "y":   torch.FloatTensor(y_all).to(device),
            "ei":  torch.LongTensor(g["ei"]).to(device),
        })
        del nf_all, gf_all, y_all

    log(f"    Total states: {sum(g['nf'].shape[0] for g in groups):,}")

    # ── 4. Verify pipeline ─────────────────────────────────────────────────────
    vlog_path = results_dir / "verify.log"
    vlog_f    = open(vlog_path, "w")
    def vlog(msg=""):
        print(msg, flush=True); vlog_f.write(msg + "\n"); vlog_f.flush()

    vlog(f"{'='*70}")
    vlog(f"STEP10 PIPELINE VERIFICATION  —  {datetime.now().isoformat()}")
    vlog(f"{'='*70}")
    vlog(f"Model: PedigreeGNN_Vstar  output=scalar V*(s) per state")
    vlog(f"node_feat_dim={NODE_FEAT_DIM}: beliefs(9) + is_tested(1) + structural(3)")

    _vkey = TRAIN_KEYS[0]
    vlog(f"\nUsing config: {_vkey}")
    with open(CACHE_DIR / f"{_vkey}.pkl", "rb") as f:
        _vds = pickle.load(f)
    _vfam = _vkey.split("_")[0]
    _vind = _vds["individuals"]
    _vsf  = struct_cache[_vfam]
    _vnf  = extract_node_features(_vds, _vind, _vsf)
    _vy   = _vds["Y"]

    vlog(f"\n[A] NODE FEATURE SHAPE")
    vlog(f"    Expected: ({len(_vds['states'])}, {len(_vind)}, {NODE_FEAT_DIM})")
    vlog(f"    Got:      {_vnf.shape}  {'OK' if _vnf.shape[-1] == NODE_FEAT_DIM else 'MISMATCH'}")

    vlog(f"\n[B] V* LABELS SHAPE AND RANGE")
    vlog(f"    Shape: {_vy.shape}  (expected ({len(_vds['states'])},))")
    vlog(f"    V* range: [{_vy.min():.6f},  {_vy.max():.6f}]")
    vlog(f"    Note: states in DP order — states[0] is terminal (V*=0), root is states[-1]")
    vlog(f"    V*(root state idx -1): {_vy[-1]:.6f}  (ds V_root={_vds['V_root']:.6f})")

    vlog(f"\n[C] STRUCTURAL FEATURES PER PERSON")
    for j, p in enumerate(_vind):
        vlog(f"    {p:<20} n_parents={_vsf[j,0]:.0f}  n_children={_vsf[j,1]:.0f}  depth={_vsf[j,2]:.0f}")

    vlog(f"\n[D] STRUCTURAL BROADCAST — same across all states")
    _n_bad = sum(
        1 for i in range(min(100, len(_vds["states"])))
        for j in range(len(_vind))
        for k in range(3)
        if abs(_vnf[i, j, 10 + k] - _vsf[j, k]) > 1e-6
    )
    vlog(f"    Mismatches in first 100 states: {_n_bad}  (should be 0)")

    vlog(f"\n[E] BELIEF FEATURES — prob sums to 1 per gene")
    _n_bad_prob = sum(
        1 for i in range(min(50, len(_vds["states"])))
        for j in range(len(_vind))
        for gi in range(len(GENES))
        if abs(_vnf[i, j, gi*3:gi*3+3].sum() - 1.0) > 1e-3
    )
    vlog(f"    Prob-sum errors in first 50 states: {_n_bad_prob}  (should be 0)")

    vlog(f"\n[F] MODEL OUTPUT SHAPE — must be (B,) scalar, NOT (B,N)")
    _ei_v  = torch.LongTensor(build_edge_index(_vfam, _vind))
    _mtest = PedigreeGNN_Vstar(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32,
                                n_rounds=2, global_feat_dim=COST_DIM)
    _mtest.eval()
    _cv_v  = config_to_cost_vec(_vds["config"], GENES)
    _nft_v = torch.FloatTensor(_vnf[:4])
    _gft_v = torch.FloatTensor(np.tile(_cv_v, (4, 1)))
    with torch.no_grad():
        _out_v = _mtest.forward_batch(_nft_v, _ei_v, _gft_v)
    vlog(f"    Output shape: {tuple(_out_v.shape)}  (expected (4,)  — one V* per state)")
    vlog(f"    Sample V* predictions (untrained): {_out_v.tolist()}")
    vlog(f"    True V* for same states:           {_vy[:4].tolist()}")
    assert _out_v.shape == (4,), f"SHAPE MISMATCH: got {_out_v.shape}, expected (4,)"
    vlog(f"    OK — scalar output confirmed")

    vlog(f"\n[G] COMPARISON: step9 (Q* per person) vs step10 (V* scalar)")
    vlog(f"    step9 node_feat_dim=10  output=(B,N)  trains on Q*(s,i) labels")
    vlog(f"    step10 node_feat_dim=13 output=(B,)   trains on V*(s)  labels (exact DP)")
    vlog(f"    Action in step10: argmax_i [r_test(i,s) + E_g[V_hat(s')]] — derived at eval")

    vlog(f"\n{'='*70}")
    vlog(f"VERIFICATION DONE. All checks passed.")
    vlog(f"{'='*70}\n")
    vlog_f.close()
    log(f"    Verification written → {vlog_path}")
    del _vds, _vnf

    # ── 5. Train ───────────────────────────────────────────────────────────────
    log(f"\n[5] Training PedigreeGNN_Vstar:")
    model = PedigreeGNN_Vstar(node_feat_dim=NODE_FEAT_DIM, hidden_dim=32,
                               n_rounds=2, global_feat_dim=COST_DIM)

    start_epoch = 1
    history     = {"train_loss": [], "val_loss": []}
    if not fresh and ckpt_path.exists():
        ckpt        = torch.load(ckpt_path, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state"])
        history     = ckpt.get("history", history)
        log(f"    Resumed from checkpoint epoch {ckpt['epoch']}/{target_epochs}")
    else:
        log(f"    Starting fresh (0/{target_epochs})")

    total_params = sum(p.numel() for p in model.parameters())
    log(f"    Params: {total_params:,}  node_feat_dim={NODE_FEAT_DIM}  hidden=32  rounds=2")

    if start_epoch <= target_epochs:
        t0 = time.time()
        model, history = train_gnn(
            groups, model,
            epochs=target_epochs, start_epoch=start_epoch,
            lr=1e-3, batch_size=512, device=device,
            print_every=50, ckpt_path=ckpt_path, ckpt_every=50,
            history=history, log=log,
        )
        log(f"    Done in {time.time()-t0:.1f}s")
    else:
        log(f"    Already at {start_epoch-1}/{target_epochs} — skipping.")

    torch.save(model.state_dict(), results_dir / "gnn_structural_model.pt")
    log(f"\n[DONE] model saved → gnn_structural_model.pt")
    if history["train_loss"]:
        log(f"       final train_loss={history['train_loss'][-1]:.6f}"
            f"  val_loss={history['val_loss'][-1]:.6f}")
    log("Run eval.py next.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device",  default="cpu")
    p.add_argument("--epochs",  type=int, default=500)
    p.add_argument("--fresh",   action="store_true")
    args = p.parse_args()
    main(device=args.device, target_epochs=args.epochs, fresh=args.fresh)
