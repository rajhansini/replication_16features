"""Step 1: Single gene, ThreeGeneration family, flat MLP.

Run:
    python step1_single_gene/run.py                  # both GeneA and GeneB
    python step1_single_gene/run.py --gene GeneA     # GeneA only
    python step1_single_gene/run.py --gene GeneB     # GeneB only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_single_gene_dataset
from shared.evaluate import compute_ratio2, sanity_checks
from shared.model   import MLPValueNet
from shared.train   import train_model

GENE_CONFIGS = {
    "GeneA": {"allele_freq": 0.02},
    "GeneB": {"allele_freq": 0.15},
}


def run_one(gene_label: str, preset: str = "Base", device: str = "cpu") -> dict:
    allele_freq = GENE_CONFIGS[gene_label]["allele_freq"]
    results_dir = HERE / "results" / gene_label
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Step 1 | Gene={gene_label}  allele_freq={allele_freq}  preset={preset}")
    print(f"{'='*60}")

    # ── 1. Generate dataset ──────────────────────────────────────────────────
    print("\n[1] Generating exact-DP dataset...")
    ds = build_single_gene_dataset(
        family_label="ThreeGeneration",
        allele_freq=allele_freq,
        preset_label=preset,
    )
    X, Y = ds["X"], ds["Y"]
    print(f"    Reachable states : {len(X)}")
    print(f"    Input dim        : {X.shape[1]}")
    print(f"    V*(root)         : {ds['V_root']:.6f}")
    print(f"    V_stop(root)     : {ds['V_stop_root']:.6f}")
    print(f"    Gap (V*-V_stop)  : {ds['V_root'] - ds['V_stop_root']:.6f}")

    # ── 2. Sanity checks on exact-DP values ─────────────────────────────────
    print("\n[2] Sanity checks on V*(s)...")
    errors = sanity_checks(ds, verbose=True)

    # ── 3. Train MLP ─────────────────────────────────────────────────────────
    print("\n[3] Training MLP (500 epochs)...")
    model   = MLPValueNet(input_dim=X.shape[1], hidden_dims=(64, 32))
    model, history = train_model(
        X, Y, model, epochs=500, lr=1e-3, device=device, print_every=100,
    )

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print("\n[4] Evaluating...")
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()

    mse = float(np.mean((Y_pred - Y) ** 2))
    mae = float(np.mean(np.abs(Y_pred - Y)))
    print(f"    MSE  : {mse:.8f}")
    print(f"    MAE  : {mae:.8f}")

    ratio2, L = compute_ratio2(model, ds, device=device)
    print(f"\n    V*(root)          = {ds['V_root']:.6f}")
    print(f"    V_stop(root)      = {ds['V_stop_root']:.6f}")
    print(f"    L (net policy)    = {L:.6f}")
    print(f"    ratio2            = {ratio2:.6f}   (lower=better, 0=perfect)")

    # ── 5. Save ──────────────────────────────────────────────────────────────
    results = {
        "gene":            gene_label,
        "allele_freq":     allele_freq,
        "preset":          preset,
        "family":          "ThreeGeneration",
        "n_states":        int(len(X)),
        "input_dim":       int(X.shape[1]),
        "V_root":          float(ds["V_root"]),
        "V_stop_root":     float(ds["V_stop_root"]),
        "L_net":           float(L),
        "ratio2":          float(ratio2),
        "mse":             float(mse),
        "mae":             float(mae),
        "sanity_errors":   int(len(errors)),
        "train_loss_final": float(history["train_loss"][-1]),
        "val_loss_final":   float(history["val_loss"][-1]),
    }
    (results_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), results_dir / "model.pt")
    np.save(results_dir / "train_history.npy", history)
    print(f"\n    Saved → {results_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene",   choices=["GeneA", "GeneB", "both"], default="both")
    parser.add_argument("--preset", choices=["Base", "Aggressive"],      default="Base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    genes = ["GeneA", "GeneB"] if args.gene == "both" else [args.gene]
    for g in genes:
        run_one(g, preset=args.preset, device=args.device)


if __name__ == "__main__":
    main()
