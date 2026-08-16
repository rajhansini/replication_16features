"""Confirm the fixed greedy_pick feature reconstruction exactly matches ds['X'],
the real training-time features, for real sample states. No training involved.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HERE        = Path(__file__).resolve().parent
EXP_ROOT    = HERE.parent
ROOT        = EXP_ROOT.parent
EXPERIMENTS = ROOT / "ground-up-experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXP_ROOT))

from exputils.eval import _get_entry  # noqa: E402

CACHE_DIR = EXPERIMENTS / "step9_gnn_3gene" / "results" / "cache"
GENES = ("GeneA", "GeneB", "GeneC")


def reconstruct_row(state, belief, individuals, genes):
    per_gene, _ = _get_entry(belief, state, genes)
    n = len(individuals)
    X = np.zeros((n, 3 * len(genes)), dtype=np.float32)
    for i, p in enumerate(individuals):
        vec = [per_gene.get(g, {}).get(p, {}).get(k, 0.0) for g in genes for k in (0, 1, 2)]
        X[i, :] = vec
    return X


def main():
    for key in ["Trio_LowHigh_Base_3gene", "ThreeGeneration_HighHigh_Base_3gene"]:
        with open(CACHE_DIR / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        states, belief, individuals = ds["states"], ds["belief"], ds["individuals"]
        n_people = len(individuals)
        X_true_all = ds["X"].reshape(len(states), n_people, 3 * len(GENES))

        n_check = min(500, len(states))
        max_err = 0.0
        n_nonzero_true = 0
        for idx in np.linspace(0, len(states) - 1, n_check, dtype=int):
            state = states[idx]
            recon = reconstruct_row(state, belief, individuals, GENES)
            true = X_true_all[idx]
            max_err = max(max_err, float(np.abs(recon - true).max()))
            n_nonzero_true += int(np.abs(true).sum() > 0)

        print(f"{key}: checked {n_check} states, max abs error = {max_err:.2e}, "
              f"true features nonzero in {n_nonzero_true}/{n_check} (sanity: should be ~all)")


if __name__ == "__main__":
    main()
