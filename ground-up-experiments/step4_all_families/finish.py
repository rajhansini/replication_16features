"""
finish.py — resume step4 from wherever it stopped.

Checks what's already done and only runs what's pending:
  - Datasets     : loads from cache (never rebuilds)
  - Training     : resumes from model_ckpt.pt (skips if already at target epochs)
  - Evaluation   : skips families already in partial_eval.json

Run:
    python step4_all_families/finish.py [--device cpu] [--epochs 800]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_two_gene_dataset
from shared.evaluate import compute_ratio2
from shared.model    import MLPValueNet
from shared.train    import train_model

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

def family_key(cfg): return f"{cfg['family']}_{cfg['regime']}_{cfg['preset']}"
def pad(X, d):
    if X.shape[1] == d: return X
    return np.concatenate([X, np.zeros((len(X), d - X.shape[1]), dtype=np.float32)], axis=1)


def main(device="cpu", target_epochs=800):
    results_dir  = HERE / "results"
    cache_dir    = results_dir / "cache"
    ckpt_path    = results_dir / "model_ckpt.pt"
    partial_path = results_dir / "partial_eval.json"
    max_dim      = 36

    keys       = [family_key(cfg) for cfg in ORIGINAL_8]
    train_keys = keys[:6]
    test_keys  = keys[6:]

    log_f = open(results_dir / "run.log", "a")
    log_f.write(f"\n{'='*60}\n[FINISH RUN] {datetime.now().isoformat()}\n{'='*60}\n")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    # ── Step 1: check datasets ────────────────────────────────────────────────
    log("\n[1] Dataset status:")
    missing = []
    for cfg in ORIGINAL_8:
        key = family_key(cfg)
        pkl = cache_dir / f"{key}.pkl"
        if pkl.exists():
            log(f"    {key}  [OK]")
        else:
            log(f"    {key}  [MISSING — will build]")
            missing.append(cfg)

    if missing:
        log(f"\n    Building {len(missing)} missing dataset(s)...")
        for cfg in missing:
            key = family_key(cfg)
            t0  = time.time()
            log(f"    {key} building...", )
            ds  = build_two_gene_dataset(
                family_label=cfg["family"],
                allele_freqs=ALLELE_FREQS[cfg["regime"]],
                preset_label=cfg["preset"],
            )
            pkl = cache_dir / f"{key}.pkl"
            tmp = pkl.with_suffix(".tmp")
            with open(tmp, "wb") as f: pickle.dump(ds, f)
            tmp.replace(pkl)
            log(f"    done in {time.time()-t0:.1f}s")

    # ── Step 2: check training ────────────────────────────────────────────────
    log(f"\n[2] Training status:")
    model = MLPValueNet(input_dim=max_dim, hidden_dims=(128, 64, 32))

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        done_epochs = ckpt["epoch"]
        log(f"    Checkpoint found: epoch {done_epochs}/{target_epochs}")
    else:
        done_epochs = 0
        log("    No checkpoint found.")

    if done_epochs < target_epochs:
        log(f"    Need {target_epochs - done_epochs} more epochs — loading training data...")
        X_parts, Y_parts = [], []
        for cfg in ORIGINAL_8:
            key = family_key(cfg)
            if key not in train_keys: continue
            with open(cache_dir / f"{key}.pkl", "rb") as f:
                ds = pickle.load(f)
            X_parts.append(pad(ds["X"], max_dim))
            Y_parts.append(ds["Y"])
            del ds
        X_train = np.concatenate(X_parts); Y_train = np.concatenate(Y_parts)
        del X_parts, Y_parts
        log(f"    Training data: {len(X_train)} states")

        model, history = train_model(
            X_train, Y_train, model,
            epochs=target_epochs, lr=1e-3, batch_size=512,
            device=device, print_every=50,
            checkpoint_path=ckpt_path, checkpoint_every=50,
        )
        del X_train, Y_train
        log("    Training complete.")
    else:
        log(f"    Training already done ({done_epochs} epochs). Loading weights...")
        model.load_state_dict(ckpt["model_state"])
        history = ckpt.get("history", {})

    model.to(device).eval()

    # ── Step 3: evaluate pending families ────────────────────────────────────
    all_results = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    pending = [cfg for cfg in ORIGINAL_8 if family_key(cfg) not in all_results]

    log(f"\n[3] Evaluation status:")
    log(f"    Done    : {len(all_results)}/8  {list(all_results.keys())}")
    log(f"    Pending : {len(pending)}/8  {[family_key(c) for c in pending]}")

    for cfg in pending:
        key   = family_key(cfg)
        split = "TRAIN" if key in train_keys else "TEST"

        log(f"\n    [{split}] {key} — loading...")
        with open(cache_dir / f"{key}.pkl", "rb") as f:
            ds = pickle.load(f)
        ds["X"] = pad(ds["X"], max_dim)
        ds["input_dim"] = max_dim

        t0 = time.time()
        ratio2, L = compute_ratio2(model, ds, device=device)
        adp   = ADP_BASELINE_RATIO2.get(key)
        adp_s = f"{adp:.6f}" if adp else "N/A"
        log(f"    ratio2 (net) = {ratio2:.6f}  |  ratio2 (ADP) = {adp_s}  ({time.time()-t0:.1f}s)")

        all_results[key] = {
            "split": split, "ratio2_net": float(ratio2), "ratio2_adp": adp,
            "L_net": float(L), "V_root": float(ds["V_root"]),
            "V_stop_root": float(ds["V_stop_root"]),
        }
        partial_path.write_text(json.dumps(all_results, indent=2))
        del ds

    # ── Step 4: write final results ───────────────────────────────────────────
    summary = {
        "train_keys": train_keys, "test_keys": test_keys,
        "families": all_results,
        "train_loss_final": float(history["train_loss"][-1]) if history.get("train_loss") else None,
    }
    (results_dir / "results.json").write_text(json.dumps(summary, indent=2))
    torch.save(model.state_dict(), results_dir / "model.pt")

    log(f"\n[DONE] Results saved to {results_dir / 'results.json'}")
    log("\nSUMMARY:")
    log(f"  {'Family':<45} {'Split':<6} {'ratio2_net':>10}  {'ratio2_ADP':>10}")
    log(f"  {'-'*75}")
    for key, r in all_results.items():
        adp_s = f"{r['ratio2_adp']:.6f}" if r["ratio2_adp"] else "   N/A   "
        log(f"  {key:<45} {r['split']:<6} {r['ratio2_net']:>10.6f}  {adp_s:>10}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=800)
    args = p.parse_args()
    main(device=args.device, target_epochs=args.epochs)
