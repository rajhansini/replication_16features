"""
Step 9 — Phase 1: Build and save 3-gene datasets for GNN training.

Families (Option B — structural diversity):
    Trio           N=3   Parent1, Parent2, Child
    Nuclear        N=4   Parent1, Parent2, Child1, Child2
    ThreeGeneration N=5  Grandparents, Father, Mother, Child

Allele frequency regimes (Option A — parameter diversity). GeneA/GeneB are
locked to the 2-gene regime values (experiments_after_understanding/two_gene/
run.py::ALLELE_FREQ_REGIMES) so 3-gene is a genuine extension of 2-gene, not a
different problem under the same regime name — only GeneC is new:
    LowHigh    GeneA=0.02, GeneB=0.15, GeneC=0.10
    MediumEven GeneA=0.08, GeneB=0.08, GeneC=0.08
    LowLow     GeneA=0.02, GeneB=0.02, GeneC=0.02
    HighHigh   GeneA=0.15, GeneB=0.15, GeneC=0.20
    MixedA     GeneA=0.02, GeneB=0.10, GeneC=0.15
    MixedB     GeneA=0.05, GeneB=0.12, GeneC=0.02

3 families × 6 regimes × 2 presets = 36 configs total.
Already-cached files are skipped.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

HERE         = Path(__file__).resolve().parent
EXPERIMENTS  = HERE.parent
PROJECT_ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from shared.data_gen import build_multigene_dataset

GENES = ("GeneA", "GeneB", "GeneC")

FAMILIES = ["Trio", "Nuclear", "ThreeGeneration"]

ALLELE_FREQS = {
    "LowHigh":    {"GeneA": 0.02, "GeneB": 0.15, "GeneC": 0.10},
    "MediumEven": {"GeneA": 0.08, "GeneB": 0.08, "GeneC": 0.08},
    "LowLow":     {"GeneA": 0.02, "GeneB": 0.02, "GeneC": 0.02},
    "HighHigh":   {"GeneA": 0.15, "GeneB": 0.15, "GeneC": 0.20},
    "MixedA":     {"GeneA": 0.02, "GeneB": 0.10, "GeneC": 0.15},
    "MixedB":     {"GeneA": 0.05, "GeneB": 0.12, "GeneC": 0.02},
}

PRESETS = ["Base", "Aggressive"]

CACHE_DIR = HERE / "results" / "cache"


def cfg_key(family, regime, preset):
    return f"{family}_{regime}_{preset}_3gene"


def main(fresh: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (family, regime, preset)
        for family in FAMILIES
        for regime  in ALLELE_FREQS
        for preset  in PRESETS
    ]

    print(f"3-gene datasets: {len(FAMILIES)} families × {len(ALLELE_FREQS)} regimes × {len(PRESETS)} presets = {len(configs)} configs")
    print(f"Cache dir: {CACHE_DIR}")
    print("=" * 70)

    built, skipped = 0, 0

    for family, regime, preset in configs:
        key = cfg_key(family, regime, preset)
        pkl = CACHE_DIR / f"{key}.pkl"

        if not fresh and pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    ds = pickle.load(f)
                print(f"[CACHED] {key}  —  {len(ds['states']):,} states  V_root={ds['V_root']:.6f}")
                skipped += 1
                continue
            except Exception as e:
                print(f"[CORRUPT] {key}  ({e}) — rebuilding")
                pkl.unlink()

        allele_freqs = {g: ALLELE_FREQS[regime][g] for g in GENES}

        print(f"\n[BUILD] {key}")
        print(f"  family={family}  regime={regime}  preset={preset}")
        print(f"  allele_freqs={allele_freqs}")
        t0 = time.time()

        ds = build_multigene_dataset(
            family_label=family,
            allele_freqs=allele_freqs,
            preset_label=preset,
            genes=GENES,
        )

        elapsed = time.time() - t0
        n_states = len(ds["states"])
        print(f"  states={n_states:,}  X={ds['X'].shape}  V_root={ds['V_root']:.6f}  V_stop={ds['V_stop_root']:.6f}  ({elapsed:.1f}s)")

        tmp = pkl.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ds, f)
        tmp.replace(pkl)
        print(f"  saved → {pkl.name}")
        built += 1
        del ds

    print(f"\n{'='*70}")
    print(f"Done. Built {built} new, skipped {skipped} cached.")
    print("\nAll files:")
    for family, regime, preset in configs:
        key = cfg_key(family, regime, preset)
        pkl = CACHE_DIR / f"{key}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                ds = pickle.load(f)
            print(f"  {key}: {len(ds['states']):,} states")
            del ds


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    main(fresh=args.fresh)
