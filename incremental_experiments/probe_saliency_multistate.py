"""Extends probe_saliency.py: is Grandfather's dead self-gradient a root-only
quirk, or does it show up at other states too, and in a second seed?

At each state (Grandfather still untested), computes self-saliency: the
gradient of Q_hat(state, TEST Grandfather) with respect to Grandfather's OWN
input features. Near-zero self-saliency at every state, both seeds, would
mean the model's assessment of testing Grandfather never actually responds
to his own data -- a structural pathology, not a one-off.

States checked: root, after Child tested, after Child+Mother tested (Grandfather
untested throughout).

Usage:
    python probe_saliency_multistate.py
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
N_GENES_FEATS = 9

STATES_TO_CHECK = [
    ("root", frozenset()),
    ("Child tested", frozenset({("Child", (0, 0, 0))})),
    ("Child+Mother tested", frozenset({("Child", (0, 0, 0)), ("Mother", (0, 0, 0))})),
]


def self_saliency(model, nf, edge_index_t, gf, individuals, state_idx, target_person):
    nf_s = nf[state_idx:state_idx+1].clone().detach().requires_grad_(True)
    gf_s = gf[state_idx:state_idx+1]
    p_idx = individuals.index(target_person)
    a_idx = torch.tensor([p_idx], device=DEVICE)
    q = model(nf_s, edge_index_t, gf_s, a_idx)
    q.backward()
    grad = nf_s.grad[0]  # (N_people, NODE_FEAT)
    self_grad = torch.norm(grad[p_idx]).item()
    other_grad = torch.norm(grad).item() - 0.0  # total norm across all people incl self
    total_excl_self = (torch.norm(grad) ** 2 - torch.norm(grad[p_idx]) ** 2).clamp(min=0).sqrt().item()
    return q.item(), self_grad, total_excl_self


def main():
    gnn_mod = _load_module("gnn_run_ms", EXP_ROOT / "gnn" / "run.py")
    gnn_q_e0 = _load_module("gnn_q_e0_ms", Q_DIR / "gnn_q.py")
    gnn_q_e4 = _load_module("gnn_q_e4_ms", HERE / "e4_train_gnn_q.py")

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
    state_row = {s: i for i, s in enumerate(states)}

    print(f"config: {KEY}\nself-saliency for Q_hat(state, TEST Grandfather) w.r.t. Grandfather's own features\n")

    for seed in (0, 1):
        e0_wt = Q_DIR / "results" / "seed_runs" / f"seed{seed}" / "gnn_q.pt"
        e4_wt = HERE / "results" / "e4_gnn_bidir" / "seed_runs" / f"seed{seed}" / "gnn_q_e4.pt"
        if not e0_wt.exists() or not e4_wt.exists():
            print(f"seed {seed}: missing checkpoint(s), skipping")
            continue

        model_e0 = gnn_q_e0.GNNQ().to(DEVICE)
        model_e0.load_state_dict(torch.load(e0_wt, map_location=DEVICE))
        model_e0.eval()

        model_e4 = gnn_q_e4.GNNQBidir().to(DEVICE)
        model_e4.load_state_dict(torch.load(e4_wt, map_location=DEVICE))
        model_e4.eval()

        print(f"=== seed {seed} ===")
        for state_label, state in STATES_TO_CHECK:
            if state not in state_row:
                print(f"  [{state_label}] state missing from belief, skipping")
                continue
            if any(p == "Grandfather" for p, _ in state):
                print(f"  [{state_label}] Grandfather already tested, skipping")
                continue
            idx = state_row[state]
            q0, self0, other0 = self_saliency(model_e0, nf, edge_index_t, gf, individuals, idx, "Grandfather")
            q4, self4, other4 = self_saliency(model_e4, nf, edge_index_t, gf, individuals, idx, "Grandfather")
            print(f"  [{state_label}]")
            print(f"      E0: Q_hat={q0:.4f}  self-grad={self0:.5f}  other-grad={other0:.5f}")
            print(f"      E4: Q_hat={q4:.4f}  self-grad={self4:.5f}  other-grad={other4:.5f}")
        print()


if __name__ == "__main__":
    main()
