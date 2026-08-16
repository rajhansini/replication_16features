"""2-gene MLP and GNN experiment.

Data: generated on the fly via build_two_gene_dataset (same function as 3-gene pipeline,
      identical belief map construction, identical exact DP labels).

Split:
    TRAIN : Trio + Nuclear  x 6 regimes x 2 presets = 24 configs
    TEST  : ThreeGeneration x 6 regimes x 2 presets = 12 configs  (unseen topology)

Models:
    MLP: node_feats (N,10) -> sum pool -> (10,) -> concat cost (8,) -> Linear(18->32) -> ReLU -> Linear(32->1)
    GNN: node_feats (N,10) -> MP x2 -> (N,32) -> sum pool -> (32,) -> concat cost (8,) -> Linear(40->16) -> ReLU -> Linear(16->1)

Node features (10 per person):
    6 belief dims  : P(g=0,1,2) x 2 genes
    1 is_tested    : 1 if person has been tested in this state
    3 structural   : n_parents, n_children, depth (fixed per family)

Cost vector (8 dims): [a, b, delta per gene x2, fixed_cost, variable_cost]

Loss: MSE on V*(s) labels from exact DP (same labels as 3-gene pipeline).
Eval: greedy rollout using Q_hat(s,i) = r_test + sum_g P(g|s,i)*V_hat[s'].
      ratio2 = (V* - L) / (V* - V_stop) via shared exputils/eval.py.
"""
from __future__ import annotations

import json
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
from exputils.eval import rollout   # same rollout used for 3-gene eval

GENES     = ("GeneA", "GeneB")
NODE_FEAT = 3 * len(GENES) + 1 + 3   # 10: 6 beliefs + is_tested + n_parents + n_children + depth
COST_DIM  = 3 * len(GENES) + 2        # 8:  a,b,delta per gene x2 + fixed + variable
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


# ── structural / edge helpers ──────────────────────────────────────────────────

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


# ── data → tensors ─────────────────────────────────────────────────────────────

def config_to_cost_vec(config):
    """8-dim cost vector: [a, b, delta per gene x2, fixed_cost, variable_cost]."""
    vec = []
    for g in GENES:
        vec.append(min(config.a_gene[g].values(), key=abs))
        vec.append(min(config.b_gene[g].values(), key=abs))
        vec.append(list(config.delta_gene[g].values())[0])
    vec.append(config.fixed_cost)
    vec.append(config.variable_cost)
    return np.array(vec, dtype=np.float32)


def ds_to_tensors(ds, struct_feats, device):
    """
    ds["X"] : (N, n_people * 3 * n_genes)  — belief probs, computed by build_two_gene_dataset
    struct_feats: (n_people, 3)             — n_parents, n_children, depth
    Returns dict with nf (N, n_people, NODE_FEAT), gf (N, COST_DIM), y (N,)
    """
    individuals = ds["individuals"]
    n_people    = len(individuals)
    person_idx  = {p: i for i, p in enumerate(individuals)}
    states      = ds["states"]
    N           = len(states)

    # belief features: (N, n_people, 6) — P(g=0,1,2) per gene per person
    X = ds["X"].reshape(N, n_people, 3 * len(GENES)).astype(np.float32)

    # is_tested flag: 1 if person has been tested in this state
    tested_arr = np.zeros((N, n_people, 1), dtype=np.float32)
    for i, state in enumerate(states):
        for person, _ in state:
            tested_arr[i, person_idx[person], 0] = 1.0

    # structural features: same for every state in this family
    struct = np.tile(struct_feats[np.newaxis, :, :], (N, 1, 1))  # (N, n_people, 3)

    # node features: (N, n_people, 10)
    nf = np.concatenate([X, tested_arr, struct], axis=-1).astype(np.float32)

    # cost vector: same for all states in this config, broadcast to (N, 8)
    cv = config_to_cost_vec(ds["config"])
    gf = np.tile(cv[np.newaxis, :], (N, 1)).astype(np.float32)

    # V* labels
    y = ds["Y"].astype(np.float32)

    return {
        "nf": torch.tensor(nf, device=device),
        "gf": torch.tensor(gf, device=device),
        "y":  torch.tensor(y,  device=device),
    }


# ── models ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Mean-pool node features (no message passing), concat cost, predict V*(s).
    Identical design to experiments_after_understanding/mlp/run.py but for 2 genes.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NODE_FEAT + COST_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, node_feats, cost_vec):
        pooled = node_feats.mean(dim=1)                        # (B, 10) — mean pool: size-invariant
        return self.net(torch.cat([pooled, cost_vec], dim=-1)).squeeze(-1)  # (B,)


class GNN(nn.Module):
    """
    2-round message passing (parent->child), sum pool, MLP head.
    Identical design to experiments_after_understanding/gnn/run.py but for 2 genes.
    """
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
        pooled = h.mean(dim=1)                                     # mean pool: size-invariant
        return self.head(torch.cat([pooled, cost_vec], dim=-1)).squeeze(-1)


# ── training ───────────────────────────────────────────────────────────────────

def train_model(model, groups, is_gnn, epochs=300, lr=1e-3,
                batch_size=512, val_frac=0.2, device="cpu", log=print, print_every=50):
    """Matches mlp/run.py and gnn/run.py's train(): 80/20 per-config positional
    split (val_frac=0.2, no shuffle before the cut), val loss logged every
    print_every epochs. Kept identical across MLP/GNN and across 2-gene/3-gene
    so training conditions are directly comparable."""
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


# ── eval ───────────────────────────────────────────────────────────────────────

def precompute_vhat(model, ds, struct_feats, device, edge_index=None, batch_size=4096):
    """Run model on all states. Returns {state: V_hat(s)} lookup table."""
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

def main(device="cpu", epochs=500):
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    log_f = open(results_dir / "run.log", "a")
    def log(msg=""):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*60}")
    log(f"[2-GENE] {datetime.now().isoformat()}")
    log(f"genes     : {GENES}")
    log(f"NODE_FEAT : {NODE_FEAT}  (6 beliefs + 1 is_tested + 3 structural)")
    log(f"COST_DIM  : {COST_DIM}   (a,b,delta x2 genes + fixed + variable)")
    log(f"HIDDEN    : {HIDDEN}")
    log(f"train     : {TRAIN_FAMILIES} x {list(ALLELE_FREQ_REGIMES)} x {PRESETS_LIST} = {len(TRAIN_CONFIGS)} configs")
    log(f"test      : {TEST_FAMILIES} x all regimes x {PRESETS_LIST} = {len(TEST_CONFIGS)} configs")
    log(f"device={device}  epochs={epochs}  val_frac=0.2 (per-config, positional — matches mlp/run.py, gnn/run.py)")

    dev = torch.device(device)

    # pedigree structural features and edge indices per family
    log(f"\n[0] Building pedigree structures...")
    struct_cache = {}
    edge_cache   = {}
    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        pedigree    = generate_deterministic_pedigree(FAMILY_CASES[fam])
        individuals = pedigree.to_list()
        struct_cache[fam] = compute_structural_features(pedigree, individuals)
        edge_cache[fam]   = build_edge_index(pedigree, individuals)
        log(f"  [{fam}] individuals={individuals}  edges={edge_cache[fam].shape[1]}")

    # generate datasets
    log(f"\n[1] Generating train datasets (build_two_gene_dataset)...")
    train_datasets = []
    for fam, reg, pre in TRAIN_CONFIGS:
        ds = build_two_gene_dataset(
            family_label=fam,
            allele_freqs=ALLELE_FREQ_REGIMES[reg],
            preset_label=pre,
            genes=GENES,
        )
        train_datasets.append((fam, reg, pre, ds))
        log(f"  {fam}_{reg}_{pre}: {len(ds['states'])} states  V*(root)={ds['V_root']:.4f}")
    log(f"  total train states: {sum(len(d[3]['states']) for d in train_datasets):,}")

    log(f"\n[2] Generating test datasets...")
    test_datasets = []
    for fam, reg, pre in TEST_CONFIGS:
        ds = build_two_gene_dataset(
            family_label=fam,
            allele_freqs=ALLELE_FREQ_REGIMES[reg],
            preset_label=pre,
            genes=GENES,
        )
        test_datasets.append((fam, reg, pre, ds))
        log(f"  {fam}_{reg}_{pre}: {len(ds['states'])} states  V*(root)={ds['V_root']:.4f}")
    log(f"  total test states: {sum(len(d[3]['states']) for d in test_datasets):,}")

    results = {"mlp": {}, "gnn": {}}

    for model_name, is_gnn in [("mlp", False), ("gnn", True)]:
        log(f"\n{'='*60}")
        if is_gnn:
            log(f"[GNN] MP x2 -> sum pool -> Linear({HIDDEN+COST_DIM}->16) -> ReLU -> Linear(16->1)")
        else:
            log(f"[MLP] sum pool -> Linear({NODE_FEAT+COST_DIM}->{HIDDEN}) -> ReLU -> Linear({HIDDEN}->1)")

        model = (GNN() if is_gnn else MLP()).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  parameters: {n_params}")

        # build tensor groups for training
        groups = []
        for fam, reg, pre, ds in train_datasets:
            g = ds_to_tensors(ds, struct_cache[fam], dev)
            if is_gnn:
                g["ei"] = torch.tensor(edge_cache[fam], device=dev)
            groups.append(g)

        log(f"\n  Training...")
        t0 = time.time()
        train_model(model, groups, is_gnn=is_gnn, epochs=epochs,
                    device=device, log=log)
        log(f"  done in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), results_dir / f"{model_name}.pt")
        log(f"  saved -> results/{model_name}.pt")

        log(f"\n  Evaluating on ThreeGeneration (unseen topology)...")
        model.eval()
        for fam, reg, pre, ds in test_datasets:
            key = f"{fam}_{reg}_{pre}_2gene"
            log(f"\n  {'─'*40}")
            log(f"  [EVAL] {key}")

            ei = edge_cache[fam] if is_gnn else None
            v_hat = precompute_vhat(model, ds, struct_cache[fam], dev,
                                    edge_index=ei)
            log(f"  V_hat range: [{min(v_hat.values()):.4f}, {max(v_hat.values()):.4f}]")
            log(f"  V*(root) = {ds['V_root']:.4f}  V_stop(root) = {ds['V_stop_root']:.4f}")
            log(f"  --- rollout trace ---")

            # uses the same rollout from exputils/eval.py — identical to 3-gene eval
            ratio2, L = rollout(v_hat, ds, log=log, trace=True)

            log(f"  ratio2={ratio2:.4f}  L={L:.4f}")
            results[model_name][key] = {
                "ratio2":  ratio2,
                "L":       L,
                "V_root":  ds["V_root"],
            }

        avg = np.mean([r["ratio2"] for r in results[model_name].values()])
        log(f"\n  {model_name.upper()} avg ratio2 = {avg:.4f}")

    # save and print summary
    out = results_dir / "results.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"\nSaved -> {out}")

    log(f"\n{'='*60}")
    log(f"FINAL SUMMARY — 2-gene, ThreeGeneration (unseen topology)")
    log(f"{'Config':<30} {'V*(root)':>10} {'MLP ratio2':>12} {'GNN ratio2':>12}")
    log(f"{'─'*66}")
    for key in results["mlp"]:
        name    = key.replace("ThreeGeneration_","").replace("_2gene","")
        m       = results["mlp"][key]["ratio2"]
        g       = results["gnn"][key]["ratio2"]
        v_root  = results["mlp"][key]["V_root"]
        log(f"{name:<30} {v_root:>10.4f} {m:>12.4f} {g:>12.4f}")
    mlp_avg = np.mean([r["ratio2"] for r in results["mlp"].values()])
    gnn_avg = np.mean([r["ratio2"] for r in results["gnn"].values()])
    log(f"{'─'*66}")
    log(f"{'AVERAGE':<30} {'':>10} {mlp_avg:>12.4f} {gnn_avg:>12.4f}")

    log_f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()
    main(device=args.device, epochs=args.epochs)
