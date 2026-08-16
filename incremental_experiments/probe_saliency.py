"""Saliency: at root, take the model's own top-choice candidate, compute
d(Q_hat)/d(each input feature) via backprop, and see which people's
information the model actually reacts to -- not just what it was fed, but
what it's sensitive to.

Complements probe_embedding_shift.py: that showed Grandfather's embedding
barely reflects anyone else's info under E0, and doesn't move at all under
E4 when a 2-hop-away relative gets tested. This asks the same question from
the output side: when the model scores "test Grandfather," does that score
actually respond to other people's genotype info, or mostly just his own?

Usage:
    python probe_saliency.py
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import torch

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
EXP_ROOT = ROOT / "experiments_after_understanding"
Q_DIR    = EXP_ROOT / "q_learning"
EXPERIMENTS = ROOT / "ground-up-experiments"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(Q_DIR))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DEVICE = torch.device("cpu")
KEY = "ThreeGeneration_LowHigh_Base_3gene"
SEED = 0
N_GENES_FEATS = 9   # 3 genes x 3 genotype probs
TESTED_IDX = 9
STRUCT_IDX = (10, 11, 12)  # n_parents, n_children, depth


def saliency_for_model(label, model, nf, edge_index_t, gf, individuals, is_gnn, root_idx):
    nf_root = nf[root_idx:root_idx+1].clone().detach().requires_grad_(True)
    gf_root = gf[root_idx:root_idx+1]

    # find the model's own top-choice candidate at root by evaluating all untested people
    with torch.no_grad():
        scores = {}
        for i, person in enumerate(individuals):
            a_idx = torch.tensor([i], device=DEVICE)
            if is_gnn:
                q = model(nf[root_idx:root_idx+1], edge_index_t, gf_root, a_idx)
            else:
                q = model(nf[root_idx:root_idx+1], gf_root, a_idx)
            scores[person] = q.item()
    top_person = max(scores, key=scores.get)
    top_idx = individuals.index(top_person)
    print(f"{label}: root Q_hat per candidate = " + "  ".join(f"{p}={s:.3f}" for p, s in scores.items()))
    print(f"{label}: top choice = {top_person}")

    a_idx = torch.tensor([top_idx], device=DEVICE)
    if is_gnn:
        q_top = model(nf_root, edge_index_t, gf_root, a_idx)
    else:
        q_top = model(nf_root, gf_root, a_idx)
    q_top.backward()

    grad = nf_root.grad[0]  # (N_people, NODE_FEAT)
    print(f"{label}: saliency for Q_hat(root, TEST {top_person}) w.r.t. each person's features")
    rows = []
    for i, person in enumerate(individuals):
        g = grad[i]
        gene_mag = torch.norm(g[:N_GENES_FEATS]).item()
        struct_mag = torch.norm(g[list(STRUCT_IDX)]).item()
        total_mag = torch.norm(g).item()
        rows.append((person, total_mag, gene_mag, struct_mag))
    rows.sort(key=lambda r: -r[1])
    for person, total_mag, gene_mag, struct_mag in rows:
        marker = " <- candidate" if person == top_person else ""
        print(f"    {person:12s}  total={total_mag:.4f}  genotype-info={gene_mag:.4f}  structural={struct_mag:.4f}{marker}")
    print()
    return top_person, rows


def main():
    gnn_mod = _load_module("gnn_run_saliency", EXP_ROOT / "gnn" / "run.py")
    gnn_q_e0 = _load_module("gnn_q_e0_saliency", Q_DIR / "gnn_q.py")
    gnn_q_e4 = _load_module("gnn_q_e4_saliency", HERE / "e4_train_gnn_q.py")

    e0_wt = Q_DIR / "results" / "seed_runs" / f"seed{SEED}" / "gnn_q.pt"
    e4_wt = HERE / "results" / "e4_gnn_bidir" / "seed_runs" / f"seed{SEED}" / "gnn_q_e4.pt"

    model_e0 = gnn_q_e0.GNNQ().to(DEVICE)
    model_e0.load_state_dict(torch.load(e0_wt, map_location=DEVICE))
    model_e0.eval()

    model_e4 = gnn_q_e4.GNNQBidir().to(DEVICE)
    model_e4.load_state_dict(torch.load(e4_wt, map_location=DEVICE))
    model_e4.eval()

    with open(gnn_mod.CACHE_DIR / f"{KEY}.pkl", "rb") as f:
        ds = pickle.load(f)

    from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree
    fam = "ThreeGeneration"
    pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
    struct_feats = gnn_mod.compute_structural_features(pedigree, ds["individuals"])
    edge_index = gnn_mod.build_edge_index(pedigree, ds["individuals"])
    edge_index_t = torch.tensor(edge_index, device=DEVICE)

    base = gnn_mod.load_dataset(KEY, struct_feats, edge_index, DEVICE)
    nf, gf = base["nf"], base["gf"]

    individuals = ds["individuals"]
    states = ds["states"]
    root_idx = states.index(frozenset())

    print(f"config: {KEY}  seed={SEED}  state: root\n")
    saliency_for_model("E0 (one-directional)", model_e0, nf, edge_index_t, gf, individuals, True, root_idx)
    saliency_for_model("E4 (bidirectional)", model_e4, nf, edge_index_t, gf, individuals, True, root_idx)


if __name__ == "__main__":
    main()
