#!/usr/bin/env python3
"""Standalone replication + audit of the ABCD16 results.

Reproduces ratio2/ratio3 for a sample of benchmark families using the ABCD-16
model — the fixed 16-feature ADP value function — and asserts they match the
published report to 1e-6.

Everything this script needs lives inside this directory. It reads nothing and
writes nothing outside this folder. Run it from inside this directory:

    python replicate.py

The run is pinned to the ABCD-16 model by `direct16_env()` + the `ABCD16_DIRECT`
feature bank. See README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent

# --- self-containment: run entirely inside this directory ---------------------
# The copied scripts resolve their manifest paths relative to the cwd, so we cd in.
os.chdir(PKG)
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
# Force a genuine from-scratch solve on every invocation: the exact-DP cache is
# memory-only (never read from / written to disk), so no result is ever replayed
# across runs — every row recomputes and logs "Exact DP cache miss". (Within a
# single row the ADP solver still legitimately reuses that row's own exact-DP
# beliefs; that is intra-row, not a cross-run cache.)
os.environ.setdefault("EXACT_DP_CACHE_IN_MEMORY_ONLY", "1")
# Any incidental solver scratch (belief snapshots, debug LP) stays inside output/.
os.environ.setdefault("EXACT_DP_CACHE_ROOT", str(PKG / "output" / ".cache"))
os.environ.setdefault("TQDM_DISABLE", "1")  # quiet the solver's progress bars

from scripts.load_suite_cases import load_cases  # noqa: E402
from scripts.run_multigene_ratio45_new_settings import RunnerSpec, _run_setting  # noqa: E402

FEATURE_BANK_PATH = PKG / "documentation" / "abcd16_direct_feature_bank_20260528.json"
DIRECT16_SELECTION = "fixed_all_16"  # use the whole fixed 16-feature bank on every family

# Generic solver configuration: it pins the dual/Gurobi LP solver, the
# exhaustive-Bellman row generation, and the stage theta mode.
BENCHMARK_ENV = {
    "EXACT_DP_SOLVER": "dual",
    "ENABLE_TUPLE_ROWGEN": "1",
    "ENABLE_PER_GENE_PHI": "1",
    "EXHAUSTIVE_BELLMAN": "1",
    "EXHAUSTIVE_STRICT": "1",
    "MAX_STATES_PER_ITER": "110000",
    "MAX_CUTS_PER_ITER": "1500000",
    "EXHAUSTIVE_WALLTIME_LIMIT_SEC": "7200",
    "EXHAUSTIVE_NO_PROGRESS_LIMIT_SEC": "600",
    "EXHAUSTIVE_HEARTBEAT_EVERY_SEC": "30",
    "EXHAUSTIVE_HEARTBEAT_EVERY_ITERS": "1",
    "GUROBI_SEED": "0",
    "PULP_SOLVER": "gurobi",
    "DISABLE_TRUNCATED_TUPLE_STRENGTHENING": "1",
    "THETA_MODE": "stage",
    "ENABLE_EDGE_FEATURES": "0",
    "ENABLE_TRIO_FEATURES": "0",
    "THETA_MODEL": "",
    "THETA_MODEL_SPEC_PATH": "",
}


def direct16_env() -> dict[str, str]:
    """The locked environment that pins the fixed ABCD-16 model.

    These are the settings the run actually depends on; the solver's other
    optional paths default to off, so they are left at their defaults. The two
    feature-bank keys below are required — the solver reads those exact names to
    select the ABCD-16 bank and its semantics (without them it falls back to a
    different bank and produces different numbers).
    """
    env = dict(BENCHMARK_ENV)
    env.update(
        {
            "ABCD16_DIRECT_ENABLED": "1",
            "ABCD16_DIRECT_SELECTION": DIRECT16_SELECTION,  # "fixed_all_16"
            "ABCD16_REQUIRE_MYOPIC_PRECOMPUTE": "1",
            "ABCD16_FORBID_ADP_SEED_PRESOLVE": "1",
            "EXACT_BELIEF_BUILD_MODE": "dense",
            "GAUGED_REGIME_FEATURE_BANK": "ABCD_HAND",       # selects the ABCD-16 bank
            "GAUGED_REGIME_FEATURE_SEMANTICS": "abcd_hand_v1",
            "MYOPIC_SAFE_GUARDRAIL_ENABLED": "1",
            "MYOPIC_SAFE_EPSILON": "0.0",
            "ADP_TOL": "5e-7",
            "ADP_MAX_ITERS": "300",
        }
    )
    return env


def _clear_scratch() -> None:
    """Drop the large per-family solver scratch.

    The engine writes belief snapshots (hundreds of MB each) and a debug LP dump
    to the cwd / cache; they are intra-family and regenerable. Clearing them
    after every family keeps peak disk to a single family's scratch instead of
    letting a 102-family run accumulate tens of GB.
    """
    shutil.rmtree(PKG / "output" / ".cache", ignore_errors=True)
    for stray in (*PKG.glob("*.lp"), *PKG.glob("*.ilp")):
        stray.unlink(missing_ok=True)


def _find_case(suite: str, row_id: str):
    for case in load_cases(suite):
        if case.row_id == row_id or case.setting.name == row_id:
            return case
    raise ValueError(f"row {row_id!r} not found in suite {suite!r}")


def _check_feature_bank() -> None:
    bank = json.loads(FEATURE_BANK_PATH.read_text(encoding="utf-8"))
    if bank.get("feature_bank_name") != "ABCD16_DIRECT" or len(bank.get("features") or []) != 16:
        raise ValueError("feature bank is not the 16-feature ABCD16_DIRECT bank")
    if bank.get("selection_mode") != "fixed_all_features":
        raise ValueError("feature bank does not fix all 16 features")


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _metric_table(rows: list[dict], metric: str) -> str:
    """Markdown table: reproduced `metric` summary per suite + overall."""
    header = (
        f"| suite | n | pass | mean | std | median | min | max | max\\|diff\\| |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    suites = sorted({r["suite"] for r in rows})
    lines = []
    for suite in [*suites, "ALL"]:
        grp = rows if suite == "ALL" else [r for r in rows if r["suite"] == suite]
        vals = [r["reproduced"][metric] for r in grp]
        s = _stats(vals)
        npass = sum(1 for r in grp if r["pass"])
        maxdiff = max(abs(r["diff"][metric]) for r in grp)
        lines.append(
            f"| {suite} | {len(grp)} | {npass}/{len(grp)} | {s['mean']:.6g} | {s['std']:.4g} | "
            f"{s['median']:.6g} | {s['min']:.6g} | {s['max']:.6g} | {maxdiff:.2e} |"
        )
    return header + "\n".join(lines) + "\n"


def write_report(rows: list[dict], *, tol: float, verdict: str, out_dir: Path, full102: bool) -> Path:
    n = len(rows)
    npass = sum(1 for r in rows if r["pass"])
    md2 = max((abs(r["diff"]["ratio2"]) for r in rows), default=0.0)
    md3 = max((abs(r["diff"]["ratio3"]) for r in rows), default=0.0)
    scope = "all 102 families" if full102 else f"{n}-family sample"
    detail = "\n".join(
        f"| {r['suite']} | {r['row_id']} | {r['reproduced']['ratio2']:.10f} | "
        f"{r['reproduced']['ratio3']:.10f} | {r['diff']['ratio2']:+.1e} | {r['diff']['ratio3']:+.1e} | "
        f"{'PASS' if r['pass'] else 'FAIL'} | {r['seconds']:.0f} |"
        for r in rows
    )
    body = f"""# ABCD16 Replication Report

- Scope: **{scope}**
- Verdict: **{verdict}** (absolute tolerance {tol:g})
- Families reproduced within tolerance: **{npass} / {n}**
- Largest deviation from the published values: `ratio2` {md2:.2e}, `ratio3` {md3:.2e}

Each family was re-solved from scratch; `ratio2`/`ratio3` below are the reproduced
values, compared against the published report.

## Reproduced `ratio3` (ADP certificate) by suite

{_metric_table(rows, "ratio3")}
## Reproduced `ratio2` (deployed policy) by suite

{_metric_table(rows, "ratio2")}
## Per-family detail

| suite | family | ratio2 | ratio3 | diff(ratio2) | diff(ratio3) | audit | sec |
|---|---|---:|---:|---:|---:|:--:|---:|
{detail}
"""
    path = out_dir / ("full102_report.md" if full102 else "replication_report.md")
    path.write_text(body, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run only the row_id matching this substring")
    parser.add_argument(
        "--full102",
        action="store_true",
        help="reproduce ALL 102 report rows (original8+local40+phase6_54) instead of the 4-row sample. "
        "WARNING: re-solves 102 instances; takes hours.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="do NOT solve; just load every targeted row and confirm the package can build it + has its "
        "expected value. Use with --full102 to prove full-102 capability without running the solve.",
    )
    args = parser.parse_args()

    _check_feature_bank()
    spec_path = PKG / ("full102_expected.json" if args.full102 else "expected.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    tol = float(spec["tolerance_abs"])
    samples = spec["samples"]
    if args.only:
        samples = [s for s in samples if args.only in s["row_id"]]

    scope = "full102 (all 102 rows)" if args.full102 else f"{len(samples)}-row sample"
    if args.plan:
        # Capability proof: build every targeted row's Setting (no solve) and
        # confirm an expected value exists for it. Loads each suite once.
        from collections import Counter

        suites = sorted({s["suite"] for s in samples})
        index = {(c.suite, c.row_id): c for su in suites for c in load_cases(su)}
        missing = [s for s in samples if (s["suite"], s["row_id"]) not in index]
        print(f"PLAN — {scope}: {len(samples)} rows targeted, spec {spec_path.name}")
        for suite, n in sorted(Counter(s["suite"] for s in samples).items()):
            print(f"  {suite:9s} {n:3d} rows loadable from manifest")
        if missing:
            print(f"  MISSING from package manifests: {len(missing)} -> {[m['row_id'] for m in missing][:5]}")
            print("PLAN: FAIL")
            return 1
        print(f"PLAN: OK — all {len(samples)} rows build from the bundled manifests and have embedded "
              f"expected values. Run without --plan (add --full102 for all 102) to solve + audit them.")
        return 0

    results = []
    all_pass = True
    print(f"ABCD-16 replication — {scope}, tolerance abs {tol:g}\n")
    for s in samples:
        case = _find_case(s["suite"], s["row_id"])
        t0 = time.time()
        metrics, _raw, _meta = _run_setting(
            case.setting,
            RunnerSpec(label="abcd16", env=direct16_env()),
            benchmark_tier="abcd16",
            progress_prefix=f"abcd16-direct::{case.suite}",
        )
        dt = time.time() - t0
        got = {"ratio2": float(metrics["ratio2"]), "ratio3": float(metrics["ratio3"])}
        d2 = got["ratio2"] - s["ratio2"]
        d3 = got["ratio3"] - s["ratio3"]
        ok = abs(d2) <= tol and abs(d3) <= tol
        all_pass &= ok
        row = {
            "suite": s["suite"],
            "row_id": s["row_id"],
            "expected": {"ratio2": s["ratio2"], "ratio3": s["ratio3"]},
            "reproduced": got,
            "diff": {"ratio2": d2, "ratio3": d3},
            "intermediate": {
                "V_star": metrics.get("exact_root_value"),
                "V_stop": metrics.get("stop_value"),
                "U_adp_phi": metrics.get("adp_phi"),
                "L_policy": metrics.get("production_policy_value"),
            },
            "pass": ok,
            "seconds": round(dt, 1),
        }
        results.append(row)
        print(f"[{'PASS' if ok else 'FAIL'}] {s['suite']:9s} {s['row_id']}")
        print(f"    ratio2  expected {s['ratio2']:.12f}  reproduced {got['ratio2']:.12f}  diff {d2:+.2e}")
        print(f"    ratio3  expected {s['ratio3']:.12f}  reproduced {got['ratio3']:.12f}  diff {d3:+.2e}")
        print(f"    ({dt:.0f}s)\n")
        _clear_scratch()  # keep peak disk to one family's scratch

    verdict = "PASS" if all_pass else "FAIL"
    out = PKG / "output" / ("full102_results.json" if args.full102 else "replication_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tolerance_abs": tol, "verdict": verdict, "rows": results}, indent=2))
    report = write_report(results, tol=tol, verdict=verdict, out_dir=out.parent, full102=args.full102)

    _clear_scratch()
    print(f"VERDICT: {verdict}")
    print(f"  results: {out.relative_to(PKG)}")
    print(f"  report:  {report.relative_to(PKG)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
