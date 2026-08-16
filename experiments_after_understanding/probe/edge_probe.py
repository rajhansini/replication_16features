"""Structural probe: can the family tree (parent->child edges) be recovered
from the pooled/per-person embeddings the trained GNN/MLP already produce?

Does NOT retrain GNN/MLP. Loads the already-trained, frozen V(s) checkpoints
(gnn/results/seed_runs/seed{N}/gnn.pt) and asks a much smaller auxiliary
classifier -- the "probe" -- to predict, for every ordered pair of people
(a, b) within a state, whether a is a parent of b, using only concat(h_a, h_b)
as input.

Levels probed (GNN only has all three; MLP only ever has level 0, since
MLP.forward never builds a per-person representation -- it goes straight from
raw node_feats to a pooled mean):
    raw     -- node_feats itself, before any learned transform (= what MLP
               would see if it kept per-person vectors instead of pooling)
    round1  -- after 1 round of GNN message passing
    round2  -- after 2 rounds (what GNN actually pools for V_hat)

Train/test split for the PROBE mirrors the main experiment exactly: probe
trained on pairs from Trio + Nuclear (depth <=1, 3-4 people), evaluated on
pairs from ThreeGeneration (depth 2, 5 people) -- a topology, and a
grandparent->parent->child relation, the probe never saw a single edge
example for during its own training.

Usage:
    python edge_probe.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
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


gnn_mod = _load_module("gnn_run_base_probe", EXP_ROOT / "gnn" / "run.py")

from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

CACHE_DIR = gnn_mod.CACHE_DIR
NODE_FEAT = gnn_mod.NODE_FEAT   # 13
HIDDEN    = gnn_mod.HIDDEN      # 32

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FAMILIES = ["Trio", "Nuclear"]
TEST_FAMILIES  = ["ThreeGeneration"]
PROBE_REGIMES  = ["LowHigh", "MediumEven", "LowLow"]   # 3 of the 6 -- enough state diversity, keeps this fast
PROBE_PRESET   = "Base"
STATES_PER_CONFIG = 300
SEEDS = [0, 1, 2]


def family_setup(fam):
    sample_key = f"{fam}_LowHigh_Base_3gene"
    with open(CACHE_DIR / f"{sample_key}.pkl", "rb") as f:
        sample_ds = pickle.load(f)
    individuals = sample_ds["individuals"]
    pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
    struct_feats = gnn_mod.compute_structural_features(pedigree, individuals)
    edge_index = gnn_mod.build_edge_index(pedigree, individuals)  # (2, E) true parent->child edges
    true_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    n = len(individuals)
    all_pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    labels = np.array([1.0 if (a, b) in true_edges else 0.0 for a, b in all_pairs], dtype=np.float32)
    return individuals, struct_feats, edge_index, all_pairs, labels


def sample_node_feats(fam, struct_feats, device, rng):
    """(n_states_sampled, n_people, NODE_FEAT) raw node_feats across a few regimes of this family."""
    chunks = []
    for reg in PROBE_REGIMES:
        key = f"{fam}_{reg}_{PROBE_PRESET}_3gene"
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            sample_ds = pickle.load(f)
        individuals = sample_ds["individuals"]
        edge_index = gnn_mod.build_edge_index(
            generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam]), individuals)
        ds = gnn_mod.load_dataset(key, struct_feats, edge_index, device)
        nf = ds["nf"]  # (N_states, n_people, NODE_FEAT)
        n_take = min(STATES_PER_CONFIG, nf.shape[0])
        idx = rng.choice(nf.shape[0], size=n_take, replace=False)
        chunks.append(nf[idx].cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def embed_all_levels(model, node_feats, edge_index_t):
    """Returns {level: (n_states, n_people, F)} for level in raw/round1/round2."""
    h0 = node_feats
    h1 = model._message_pass(h0, edge_index_t[0], edge_index_t[1], model.msg_layers[0], model.upd_layers[0])
    h2 = model._message_pass(h1, edge_index_t[0], edge_index_t[1], model.msg_layers[1], model.upd_layers[1])
    return {"raw": h0, "round1": h1, "round2": h2}


def build_pair_dataset(embeds, all_pairs, labels, n_states):
    """embeds: (n_states, n_people, F). Returns X:(n_states*n_pairs, 2F), y:(n_states*n_pairs,)."""
    F = embeds.shape[-1]
    n_pairs = len(all_pairs)
    labels_t = torch.from_numpy(labels).float()  # (n_pairs,)
    X = torch.zeros(n_states * n_pairs, 2 * F)
    y = labels_t.repeat(n_states)
    for si in range(n_states):
        for pi, (a, b) in enumerate(all_pairs):
            row = si * n_pairs + pi
            X[row, :F] = embeds[si, a]
            X[row, F:] = embeds[si, b]
    return X, y


class Probe(nn.Module):
    """Logistic regression on concat(h_a, h_b) -- deliberately tiny: this is
    asking whether the SIGNAL is linearly present, not whether a big enough
    decoder can brute-force it."""
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def standardize(X_train, X_test):
    """z-score using TRAIN stats only. Levels have very different raw scales
    (raw features are ~0-2; round1/round2 are post-ReLU, unbounded above) --
    without this, a single fixed lr/epoch budget is not a fair comparison
    across levels."""
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_train - mean) / std, (X_test - mean) / std


def train_probe(X_train, y_train, epochs=500, lr=0.05):
    torch.manual_seed(0)
    probe = Probe(X_train.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    # class-balanced BCE: edges are the rare class (20-33% positive rate depending on family)
    pos_weight = torch.tensor([(y_train == 0).sum().item() / max((y_train == 1).sum().item(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loss_history = []
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        train_loss_history.append(loss.item())
    return probe, train_loss_history


@torch.no_grad()
def evaluate_probe(probe, X_test, y_test):
    logits = probe(X_test)
    pred = (torch.sigmoid(logits) > 0.5).float()
    tp = ((pred == 1) & (y_test == 1)).sum().item()
    fp = ((pred == 1) & (y_test == 0)).sum().item()
    fn = ((pred == 0) & (y_test == 1)).sum().item()
    tn = ((pred == 0) & (y_test == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc       = (tp + tn) / len(y_test)
    majority_baseline_acc = max(y_test.mean().item(), 1 - y_test.mean().item())
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "n_test": len(y_test), "n_positive": int(y_test.sum().item()),
            "majority_baseline_acc": majority_baseline_acc}


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(0)

    print("[1] Building true-edge labels + pair index per family...")
    fam_setup = {}
    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        individuals, struct_feats, edge_index, all_pairs, labels = family_setup(fam)
        fam_setup[fam] = dict(individuals=individuals, struct_feats=struct_feats,
                               edge_index=edge_index, all_pairs=all_pairs, labels=labels)
        n_edges = int(labels.sum())
        print(f"    {fam}: {len(individuals)} people, {len(all_pairs)} ordered pairs, "
              f"{n_edges} true edges ({100*n_edges/len(all_pairs):.0f}% positive)")

    print("\n[2] Sampling node_feats per family (raw, pre-model)...")
    nf_by_fam = {}
    for fam in TRAIN_FAMILIES + TEST_FAMILIES:
        nf = sample_node_feats(fam, fam_setup[fam]["struct_feats"], device, rng)
        nf_by_fam[fam] = nf
        print(f"    {fam}: sampled {nf.shape[0]} states")

    all_results = {}  # level -> seed -> metrics
    for level in ["raw", "round1", "round2"]:
        all_results[level] = []

    for seed in SEEDS:
        print(f"\n[3] seed {seed}: loading frozen GNN checkpoint, embedding, training probes...")
        model = gnn_mod.GNN().to(device)
        ckpt_path = EXP_ROOT / "gnn" / "results" / "seed_runs" / f"seed{seed}" / "gnn.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        embeds_by_fam = {}
        for fam in TRAIN_FAMILIES + TEST_FAMILIES:
            ei = torch.tensor(fam_setup[fam]["edge_index"], device=device)
            embeds_by_fam[fam] = embed_all_levels(model, nf_by_fam[fam], ei)

        for level in ["raw", "round1", "round2"]:
            X_train_parts, y_train_parts = [], []
            for fam in TRAIN_FAMILIES:
                n_states = embeds_by_fam[fam][level].shape[0]
                X, y = build_pair_dataset(embeds_by_fam[fam][level], fam_setup[fam]["all_pairs"],
                                           fam_setup[fam]["labels"], n_states)
                X_train_parts.append(X)
                y_train_parts.append(y)
            X_train = torch.cat(X_train_parts, dim=0)
            y_train = torch.cat(y_train_parts, dim=0)

            X_test_parts, y_test_parts = [], []
            for fam in TEST_FAMILIES:
                n_states = embeds_by_fam[fam][level].shape[0]
                X, y = build_pair_dataset(embeds_by_fam[fam][level], fam_setup[fam]["all_pairs"],
                                           fam_setup[fam]["labels"], n_states)
                X_test_parts.append(X)
                y_test_parts.append(y)
            X_test = torch.cat(X_test_parts, dim=0)
            y_test = torch.cat(y_test_parts, dim=0)

            X_train_std, X_test_std = standardize(X_train, X_test)
            probe, loss_hist = train_probe(X_train_std, y_train)
            train_metrics = evaluate_probe(probe, X_train_std, y_train)
            metrics = evaluate_probe(probe, X_test_std, y_test)
            metrics["seed"] = seed
            metrics["train_f1"] = train_metrics["f1"]
            metrics["train_acc"] = train_metrics["accuracy"]
            metrics["final_train_loss"] = loss_hist[-1]
            all_results[level].append(metrics)
            print(f"    [{level:7s}] train_pairs={len(y_train):6d}  test_pairs={len(y_test):6d}  "
                  f"train_acc={train_metrics['accuracy']:.3f} train_f1={train_metrics['f1']:.3f}  |  "
                  f"test_acc={metrics['accuracy']:.3f}  test_f1={metrics['f1']:.3f}  "
                  f"(majority-class baseline acc={metrics['majority_baseline_acc']:.3f})  "
                  f"final_loss={loss_hist[-1]:.4f}")

    print("\n[4] Summary (mean +/- std across 3 seeds, held out on ThreeGeneration):")
    summary = {}
    for level in ["raw", "round1", "round2"]:
        accs = [m["accuracy"] for m in all_results[level]]
        f1s  = [m["f1"] for m in all_results[level]]
        precs = [m["precision"] for m in all_results[level]]
        recs  = [m["recall"] for m in all_results[level]]
        train_f1s = [m["train_f1"] for m in all_results[level]]
        summary[level] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "precision_mean": float(np.mean(precs)), "recall_mean": float(np.mean(recs)),
            "train_f1_mean": float(np.mean(train_f1s)),
            "per_seed": all_results[level],
        }
        print(f"    {level:7s}  train_f1={summary[level]['train_f1_mean']:.3f}  |  "
              f"test_acc={summary[level]['accuracy_mean']:.3f}+/-{summary[level]['accuracy_std']:.3f}  "
              f"test_f1={summary[level]['f1_mean']:.3f}+/-{summary[level]['f1_std']:.3f}")

    (RESULTS_DIR / "edge_probe_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {RESULTS_DIR/'edge_probe_results.json'}")


if __name__ == "__main__":
    main()
