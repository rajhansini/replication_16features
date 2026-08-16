"""Mechanistic check: does bidirectional message passing (E4) actually move
an ancestor's embedding more than one-directional message passing (E0) does,
when a descendant gets tested?

State A = root (nobody tested). State B = same config, Child tested (most
likely outcome, genotype (0,0,0)). Grandfather is untested in both -- any
change in his embedding must come from message passing (his own input
features barely move -- see note below), not from "he got tested."

Control: Father, who already has both a parent-edge and a child-edge, so
his embedding should react similarly under E0 and E4 -- if it doesn't,
the whole measurement is suspect.

Usage:
    python probe_embedding_shift.py
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
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


from genetic_dp.utils.pedigree_generator import generate_deterministic_pedigree  # noqa: E402

DEVICE = torch.device("cpu")
KEY = "ThreeGeneration_LowHigh_Base_3gene"
SEED = 0


def main():
    gnn_mod = _load_module("gnn_run_probe", EXP_ROOT / "gnn" / "run.py")
    gnn_q_e0 = _load_module("gnn_q_e0_probe", Q_DIR / "gnn_q.py")
    gnn_q_e4 = _load_module("gnn_q_e4_probe", HERE / "e4_train_gnn_q.py")

    e0_wt = Q_DIR / "results" / "seed_runs" / f"seed{SEED}" / "gnn_q.pt"
    e4_wt = HERE / "results" / "e4_gnn_bidir" / "seed_runs" / f"seed{SEED}" / "gnn_q_e4.pt"
    assert e0_wt.exists(), f"missing {e0_wt}"
    assert e4_wt.exists(), f"missing {e4_wt}"

    model_e0 = gnn_q_e0.GNNQ().to(DEVICE)
    model_e0.load_state_dict(torch.load(e0_wt, map_location=DEVICE))
    model_e0.eval()

    model_e4 = gnn_q_e4.GNNQBidir().to(DEVICE)
    model_e4.load_state_dict(torch.load(e4_wt, map_location=DEVICE))
    model_e4.eval()

    with open(gnn_mod.CACHE_DIR / f"{KEY}.pkl", "rb") as f:
        ds = pickle.load(f)

    fam = "ThreeGeneration"
    pedigree = generate_deterministic_pedigree(gnn_mod.FAMILY_CASES[fam])
    struct_feats = gnn_mod.compute_structural_features(pedigree, ds["individuals"])
    edge_index = gnn_mod.build_edge_index(pedigree, ds["individuals"])
    edge_index_t = torch.tensor(edge_index, device=DEVICE)

    base = gnn_mod.load_dataset(KEY, struct_feats, edge_index, DEVICE)
    nf = base["nf"]

    individuals = ds["individuals"]
    person_idx = {p: i for i, p in enumerate(individuals)}
    states = ds["states"]
    state_row = {s: i for i, s in enumerate(states)}

    state_A = frozenset()
    state_B = frozenset({("Child", (0, 0, 0))})

    assert state_A in state_row, "root state missing"
    assert state_B in state_row, f"state B {state_B} missing from belief -- check most-likely outcome encoding"

    idx_A = state_row[state_A]
    idx_B = state_row[state_B]

    print(f"config: {KEY}  seed={SEED}")
    print(f"state A (root): {sorted(state_A)}")
    print(f"state B (Child tested): {sorted(state_B)}")
    print(f"individuals: {individuals}")
    print()

    results = {}
    for label, model in (("E0 (one-directional)", model_e0), ("E4 (bidirectional)", model_e4)):
        with torch.no_grad():
            h_A = model.embed(nf[idx_A:idx_A+1], edge_index_t)[0]  # (N, hidden)
            h_B = model.embed(nf[idx_B:idx_B+1], edge_index_t)[0]

        row = {}
        for person in ("Grandfather", "Father"):
            p_idx = person_idx[person]
            shift = torch.norm(h_B[p_idx] - h_A[p_idx]).item()
            emb_norm_A = torch.norm(h_A[p_idx]).item()
            row[person] = {"shift": shift, "relative_shift": shift / emb_norm_A if emb_norm_A > 1e-9 else float("nan")}
        results[label] = row
        print(f"{label}:")
        for person, r in row.items():
            print(f"    {person:12s}  ||shift|| = {r['shift']:.5f}   (relative to ||emb(A)||: {r['relative_shift']:.3f})")
        print()

    print("="*60)
    gf_e0 = results["E0 (one-directional)"]["Grandfather"]["shift"]
    gf_e4 = results["E4 (bidirectional)"]["Grandfather"]["shift"]
    fa_e0 = results["E0 (one-directional)"]["Father"]["shift"]
    fa_e4 = results["E4 (bidirectional)"]["Father"]["shift"]
    print(f"Grandfather (ancestor, no incoming edges under E0): E0={gf_e0:.5f}  E4={gf_e4:.5f}  ratio E4/E0={gf_e4/gf_e0 if gf_e0>1e-9 else float('nan'):.2f}x")
    print(f"Father (control, has both edge types already):      E0={fa_e0:.5f}  E4={fa_e4:.5f}  ratio E4/E0={fa_e4/fa_e0 if fa_e0>1e-9 else float('nan'):.2f}x")


if __name__ == "__main__":
    main()
