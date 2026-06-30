"""Step 4: All 8 canonical families, two genes, train/test split.

Trains ONE MLP on all training families combined.
Evaluates ratio2 on held-out test families.
Compares against the ADP baseline ratio2 values from the report.

Resume logic:
  - Datasets are cached to results/cache/<key>.pkl — skipped on re-run.
  - Model checkpoint saved to results/model_ckpt.pt — training skipped if it exists.
  - Evaluation results saved incrementally to results/partial_eval.json.
  - Full results written to results/results.json only when everything is done.

Run:
    python step4_all_families/run.py [--device cpu] [--fresh]
    --fresh  : ignore all caches and start over
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_two_gene_dataset
from shared.evaluate import compute_ratio2
from shared.model    import MLPValueNet
from shared.train    import train_model

# ── Family configs ────────────────────────────────────────────────────────────
ORIGINAL_8 = [
    {"family": "Extended",        "regime": "LowHigh",    "preset": "Base"},
    {"family": "Extended",        "regime": "LowHigh",    "preset": "Aggressive"},
    {"family": "Extended",        "regime": "MediumEven", "preset": "Base"},
    {"family": "Extended",        "regime": "MediumEven", "preset": "Aggressive"},
    {"family": "ThreeGeneration", "regime": "LowHigh",    "preset": "Base"},
    {"family": "ThreeGeneration", "regime": "LowHigh",    "preset": "Aggressive"},
    {"family": "ThreeGeneration", "regime": "MediumEven", "preset": "Base"},
    {"family": "ThreeGeneration", "regime": "MediumEven", "preset": "Aggressive"},
]

ALLELE_FREQS = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08},
}

ADP_BASELINE_RATIO2 = {
    "Extended_LowHigh_Base":              0.10068959165,
    "ThreeGeneration_LowHigh_Aggressive": 0.05132042772,
    "Extended_MediumEven_Base":           0.15896943584,
}


def family_key(cfg: dict) -> str:
    return f"{cfg['family']}_{cfg['regime']}_{cfg['preset']}"


def _pad_X(X: np.ndarray, target_dim: int) -> np.ndarray:
    if X.shape[1] == target_dim:
        return X
    pad = np.zeros((len(X), target_dim - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


# ── Logging: write every print to both stdout and a log file ──────────────────
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def _setup_log(results_dir: Path) -> None:
    log_path = results_dir / "run.log"
    log_f    = open(log_path, "a")
    log_f.write(f"\n{'='*60}\n[RUN START] {datetime.now().isoformat()}\n{'='*60}\n")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    print(f"Logging to {log_path}")


def main(device: str = "cpu", fresh: bool = False):
    results_dir = HERE / "results"
    cache_dir   = results_dir / "cache"
    results_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    _setup_log(results_dir)

    max_dim    = 36   # Extended=36 dims, ThreeGeneration=30 → padded to 36
    keys       = [family_key(cfg) for cfg in ORIGINAL_8]
    train_keys = keys[:6]
    test_keys  = keys[6:]

    # ── 1. Build/cache datasets; collect X,Y for training WITHOUT keeping all
    #       belief dicts in RAM at once (each Extended pkl is ~100 MB) ─────────
    print("\n[1] Building / loading datasets (one at a time to save RAM)...")
    X_parts, Y_parts = [], []

    for cfg in ORIGINAL_8:
        key      = family_key(cfg)
        pkl_path = cache_dir / f"{key}.pkl"

        ds = None
        if not fresh and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    ds = pickle.load(f)
                print(f"    {key}  [CACHED, {len(ds['X'])} states]")
            except Exception:
                print(f"    {key}  [CACHE CORRUPT — rebuilding]")
                pkl_path.unlink(missing_ok=True)

        if ds is None:
            t0 = time.time()
            print(f"    {key}  [building...]", end=" ", flush=True)
            ds = build_two_gene_dataset(
                family_label=cfg["family"],
                allele_freqs=ALLELE_FREQS[cfg["regime"]],
                preset_label=cfg["preset"],
            )
            tmp = pkl_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(ds, f)
            tmp.replace(pkl_path)
            print(f"done in {time.time()-t0:.1f}s  ({len(ds['X'])} states)")

        if key in train_keys:
            X_parts.append(_pad_X(ds["X"], max_dim))
            Y_parts.append(ds["Y"])
        del ds   # free RAM immediately — reload per-family during eval

    # ── 2. Combine training data ──────────────────────────────────────────────
    print(f"\n[2] Train ({len(train_keys)}): {train_keys}")
    print(f"    Test  ({len(test_keys)}) : {test_keys}")
    X_train = np.concatenate(X_parts, axis=0)
    Y_train = np.concatenate(Y_parts, axis=0)
    del X_parts, Y_parts
    print(f"    Combined: {len(X_train)} states, dim={max_dim}")

    # ── 3. Train with mid-run checkpointing ──────────────────────────────────
    ckpt_path = results_dir / "model_ckpt.pt"
    model     = MLPValueNet(input_dim=max_dim, hidden_dims=(128, 64, 32))
    if fresh and ckpt_path.exists():
        ckpt_path.unlink()

    print("\n[3] Training MLP (800 epochs, batch=512, checkpoint every 50 epochs)...")
    model, history = train_model(
        X_train, Y_train, model,
        epochs=800, lr=1e-3, batch_size=512,
        device=device, print_every=100,
        checkpoint_path=ckpt_path,
        checkpoint_every=50,
    )
    del X_train, Y_train
    print(f"    Training done.")

    # ── 4. Evaluate ratio2 — load each family fresh, one at a time ───────────
    partial_path = results_dir / "partial_eval.json"
    if not fresh and partial_path.exists():
        all_results = json.loads(partial_path.read_text())
        print(f"\n[4] Evaluating ratio2 (resuming — {len(all_results)} already done)...")
    else:
        all_results = {}
        print("\n[4] Evaluating ratio2...")

    for cfg in ORIGINAL_8:
        key   = family_key(cfg)
        split = "TRAIN" if key in train_keys else "TEST"

        if key in all_results:
            r = all_results[key]
            print(f"    [SKIP] {key}  ratio2_net={r['ratio2_net']:.6f}")
            continue

        # Load one family at a time
        pkl_path = cache_dir / f"{key}.pkl"
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)
        ds["X"]         = _pad_X(ds["X"], max_dim)
        ds["input_dim"] = max_dim

        t0 = time.time()
        ratio2, L = compute_ratio2(model, ds, device=device)
        adp   = ADP_BASELINE_RATIO2.get(key)
        adp_s = f"{adp:.6f}" if adp is not None else "N/A"
        print(f"    [{split}] {key}  ({time.time()-t0:.1f}s)")
        print(f"           ratio2 (net) = {ratio2:.6f}  |  ratio2 (ADP) = {adp_s}")

        all_results[key] = {
            "split":       split,
            "ratio2_net":  float(ratio2),
            "ratio2_adp":  adp,
            "L_net":       float(L),
            "V_root":      float(ds["V_root"]),
            "V_stop_root": float(ds["V_stop_root"]),
        }
        partial_path.write_text(json.dumps(all_results, indent=2))
        del ds   # free RAM before loading next family

    # ── 5. Save final summary ─────────────────────────────────────────────────
    summary = {
        "train_keys":       train_keys,
        "test_keys":        test_keys,
        "n_train_states":   int(len(X_train)),
        "input_dim":        int(max_dim),
        "families":         all_results,
        "train_loss_final": float(history["train_loss"][-1]),
        "val_loss_final":   float(history["val_loss"][-1]),
    }
    (results_dir / "results.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), results_dir / "model.pt")
    print(f"\n    Done. Results → {results_dir / 'results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fresh",  action="store_true",
                        help="Ignore all caches and restart from scratch")
    args = parser.parse_args()
    main(device=args.device, fresh=args.fresh)
