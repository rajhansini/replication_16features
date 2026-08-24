"""Rebuild results/myopic_TRUE_summary.json from the per-config myopic files.

verify_myopic_one.py writes one JSON per (genes, regime, preset) into
results/myopic_TRUE/ so each SLURM array task releases its memory. This
collects them back into the single summary the write-ups quote, and reports
the aggregate myopic ratio2 per gene count -- the reference value that model
ratio2 numbers are compared against.

It also refuses to build a summary from a partial set: if any of the 12
configs per gene count is missing, that gene count is reported as incomplete
rather than averaged over whatever happens to be on disk. A myopic reference
averaged over 9 of 12 configs silently misstates the baseline.

Usage:
    python build_myopic_summary.py            # rebuild + print
    python build_myopic_summary.py --check    # print only, don't write
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "myopic_TRUE"
OUT = HERE / "results" / "myopic_TRUE_summary.json"

REGIMES = ["LowHigh", "MediumEven", "LowLow", "HighHigh", "MixedA", "MixedB"]
PRESETS = ["Base", "Aggressive"]
FAMILY = "ThreeGeneration"   # the TEST family these are evaluated on


def collect(genes):
    found, missing = {}, []
    for reg in REGIMES:
        for pre in PRESETS:
            key = f"{FAMILY}_{reg}_{pre}_{genes}gene"
            f = SRC / f"{key}.json"
            if f.exists():
                found[key] = json.loads(f.read_text())
            else:
                missing.append(key)
    return found, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print only, do not write the summary")
    args = ap.parse_args()

    summary, report = {}, []
    for genes in (2, 3):
        found, missing = collect(genes)
        summary[f"{genes}gene"] = found
        if missing:
            report.append(f"{genes}-gene: INCOMPLETE -- {len(found)}/12 configs "
                          f"(missing: {', '.join(m.split('_', 1)[1] for m in missing)})")
            continue
        r2 = statistics.mean(v["ratio2_TRUE"] for v in found.values())
        agree = sum(1 for v in found.values() if v["root_action_myopic"] == v["root_action_dp"])
        oldest = min((SRC / f"{k}.json").stat().st_mtime for k in found)
        newest = max((SRC / f"{k}.json").stat().st_mtime for k in found)
        stamp = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
        report.append(f"{genes}-gene: 12/12 configs | myopic avg ratio2 = {r2:.4f} | "
                      f"root action matches DP in {agree}/12 | files {stamp(oldest)} .. {stamp(newest)}")

    print("\n".join(report))
    if not args.check:
        OUT.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
