"""Verify this project's "myopic ratio2" against Kanix's OWN reproducibility
harness (replicate.py / expected.json), end to end -- not just tracing that
our code imports his function, but literally running his top-level
_run_setting() + _compute_myopic_value() on his own published benchmark rows.

Kanix's harness reports "ratio2" for the DEPLOYED ADP policy (production
policy), not pure myopic -- this script additionally computes the pure-myopic
ratio2 ((V* - V_myopic) / (V* - V_stop)) using his own _compute_myopic_value,
so it's directly comparable to what this project has been calling "myopic
ratio2" throughout E0-E9 and fresh_dataset.

Usage:
    python verify_myopic_against_kanix_harness.py --only Extended_LowHigh_Base
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
os.chdir(PKG)
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
os.environ.setdefault("EXACT_DP_CACHE_IN_MEMORY_ONLY", "1")
os.environ.setdefault("EXACT_DP_CACHE_ROOT", str(PKG / "output" / ".cache"))
os.environ.setdefault("TQDM_DISABLE", "1")

from scripts.load_suite_cases import load_cases  # noqa: E402
from scripts.run_multigene_ratio45_new_settings import (  # noqa: E402
    RunnerSpec, _run_setting, _compute_myopic_value,
)
from replicate import direct16_env, _check_feature_bank, _clear_scratch  # noqa: E402


def find_case(suite, row_id):
    for case in load_cases(suite):
        if case.row_id == row_id or case.setting.name == row_id:
            return case
    raise ValueError(f"row {row_id!r} not found in suite {suite!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="original8")
    p.add_argument("--only", required=True, help="row_id to run")
    args = p.parse_args()

    _check_feature_bank()
    spec = json.loads((PKG / "expected.json").read_text())
    expected = {s["row_id"]: s for s in spec["samples"]}
    if args.only not in expected:
        raise SystemExit(f"{args.only} not in expected.json samples: {list(expected)}")
    exp = expected[args.only]

    case = find_case(exp["suite"], args.only)
    print(f"[running] suite={exp['suite']} row_id={args.only}")
    t0 = time.time()
    metrics, raw_results, solver_meta = _run_setting(
        case.setting,
        RunnerSpec(label="abcd16", env=direct16_env()),
        benchmark_tier="abcd16",
        progress_prefix=f"verify::{exp['suite']}",
    )
    dt_solve = time.time() - t0
    print(f"[solve done] {dt_solve:.0f}s")

    t1 = time.time()
    myopic = _compute_myopic_value(case.setting, raw_results)
    dt_myopic = time.time() - t1
    print(f"[myopic eval done] {dt_myopic:.0f}s")

    exact = float(metrics["exact_root_value"])
    stop = float(metrics["stop_value"])
    myopic_value = myopic["myopic_value"]
    denom = exact - stop
    myopic_ratio2_own = (exact - myopic_value) / denom if denom else None

    result = {
        "row_id": args.only,
        "suite": exp["suite"],
        "kanix_expected_ratio2_deployed_policy": exp["ratio2"],
        "kanix_expected_ratio3": exp["ratio3"],
        "reproduced_ratio2_deployed_policy": float(metrics["ratio2"]),
        "reproduced_ratio3": float(metrics["ratio3"]),
        "deployed_policy_matches_published": abs(float(metrics["ratio2"]) - exp["ratio2"]) <= 1e-6,
        "V_star": exact,
        "V_stop": stop,
        "myopic_root_action": myopic["myopic_root_action_label"],
        "V_myopic": myopic_value,
        "myopic_ratio2_pure_standalone": myopic_ratio2_own,
        "production_policy_source": raw_results.get("production_policy_source"),
        "solve_seconds": round(dt_solve, 1),
        "myopic_eval_seconds": round(dt_myopic, 1),
    }
    print(json.dumps(result, indent=2))
    out_dir = PKG / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"myopic_verify_{args.only}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")
    _clear_scratch()


if __name__ == "__main__":
    main()
