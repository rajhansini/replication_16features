"""Step 3: Two genes, ThreeGeneration family, flat MLP.

X grows from 15 → 30 numbers (3 probs × 2 genes × 5 people).
Compare ratio2 against step 1 single-gene results.

Run:
    python step3_two_genes/run.py
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

from shared.data_gen import ALLELE_FREQ_REGIMES, build_two_gene_dataset
from shared.evaluate import compute_ratio2, sanity_checks
from shared.model    import MLPValueNet
from shared.train    import train_model


def run_one(regime: str = "LowHigh", preset: str = "Base", device: str = "cpu") -> dict:
    allele_freqs = ALLELE_FREQ_REGIMES[regime]
    results_dir  = HERE / "results" / f"{regime}_{preset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Step 3 | Two genes | regime={regime}  preset={preset}")
    print(f"        GeneA={allele_freqs['GeneA']}  GeneB={allele_freqs['GeneB']}")
    print(f"{'='*60}")

    # ── 1. Generate ───────────────────────────────────────────────────────────
    print("\n[1] Generating two-gene exact-DP dataset...")
    ds = build_two_gene_dataset(
        family_label="ThreeGeneration",
        allele_freqs=allele_freqs,
        preset_label=preset,
    )
    X, Y = ds["X"], ds["Y"]
    print(f"    Reachable states : {len(X)}")
    print(f"    Input dim        : {X.shape[1]}  (= 5 people × 2 genes × 3 probs)")
    print(f"    V*(root)         : {ds['V_root']:.6f}")
    print(f"    V_stop(root)     : {ds['V_stop_root']:.6f}")
    print(f"    Gap              : {ds['V_root'] - ds['V_stop_root']:.6f}")

    # ── 2. Sanity ─────────────────────────────────────────────────────────────
    print("\n[2] Sanity checks...")
    sanity_checks(ds, verbose=True)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[3] Training MLP (500 epochs)...")
    model   = MLPValueNet(input_dim=X.shape[1], hidden_dims=(128, 64, 32))
    model, history = train_model(
        X, Y, model, epochs=500, lr=1e-3, device=device, print_every=100,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4] Evaluating...")
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()

    mse    = float(np.mean((Y_pred - Y) ** 2))
    mae    = float(np.mean(np.abs(Y_pred - Y)))
    ratio2, L = compute_ratio2(model, ds, device=device)

    print(f"    MSE              : {mse:.8f}")
    print(f"    MAE              : {mae:.8f}")
    print(f"    V*(root)         : {ds['V_root']:.6f}")
    print(f"    V_stop(root)     : {ds['V_stop_root']:.6f}")
    print(f"    L (net policy)   : {L:.6f}")
    print(f"    ratio2           : {ratio2:.6f}   (lower=better)")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    results = {
        "regime":          regime,
        "allele_freqs":    allele_freqs,
        "preset":          preset,
        "family":          "ThreeGeneration",
        "n_genes":         2,
        "n_states":        int(len(X)),
        "input_dim":       int(X.shape[1]),
        "V_root":          float(ds["V_root"]),
        "V_stop_root":     float(ds["V_stop_root"]),
        "L_net":           float(L),
        "ratio2":          float(ratio2),
        "mse":             float(mse),
        "mae":             float(mae),
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
    parser.add_argument("--regime", choices=["LowHigh", "MediumEven", "both"], default="LowHigh")
    parser.add_argument("--preset", choices=["Base", "Aggressive"], default="Base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    regimes = ["LowHigh", "MediumEven"] if args.regime == "both" else [args.regime]
    for r in regimes:
        run_one(r, preset=args.preset, device=args.device)


if __name__ == "__main__":
    main()
