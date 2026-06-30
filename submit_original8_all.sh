#!/bin/bash
#SBATCH --job-name=rep_original8
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/original8_all_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/original8_all_%j.err

export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic
cd /net/projects/ranalab/rajhansini/replication_16features
PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
mkdir -p output/original8

$PYTHON - <<'PYEOF'
import os, sys, json, time, shutil
from pathlib import Path

os.chdir("/net/projects/ranalab/rajhansini/replication_16features")
sys.path.insert(0, "/net/projects/ranalab/rajhansini/replication_16features")
os.environ.setdefault("EXACT_DP_CACHE_IN_MEMORY_ONLY", "1")
os.environ.setdefault("EXACT_DP_CACHE_ROOT", "/net/projects/ranalab/rajhansini/replication_16features/output/.cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from scripts.load_suite_cases import load_cases
from scripts.run_multigene_ratio45_new_settings import RunnerSpec, _run_setting

BENCHMARK_ENV = {
    "EXACT_DP_SOLVER": "dual", "ENABLE_TUPLE_ROWGEN": "1", "ENABLE_PER_GENE_PHI": "1",
    "EXHAUSTIVE_BELLMAN": "1", "EXHAUSTIVE_STRICT": "1", "MAX_STATES_PER_ITER": "110000",
    "MAX_CUTS_PER_ITER": "1500000", "EXHAUSTIVE_WALLTIME_LIMIT_SEC": "7200",
    "EXHAUSTIVE_NO_PROGRESS_LIMIT_SEC": "600", "EXHAUSTIVE_HEARTBEAT_EVERY_SEC": "30",
    "EXHAUSTIVE_HEARTBEAT_EVERY_ITERS": "1", "GUROBI_SEED": "0", "PULP_SOLVER": "gurobi",
    "DISABLE_TRUNCATED_TUPLE_STRENGTHENING": "1", "THETA_MODE": "stage",
    "ENABLE_EDGE_FEATURES": "0", "ENABLE_TRIO_FEATURES": "0",
    "THETA_MODEL": "", "THETA_MODEL_SPEC_PATH": "",
}

def direct16_env():
    env = dict(BENCHMARK_ENV)
    env.update({
        "ABCD16_DIRECT_ENABLED": "1", "ABCD16_DIRECT_SELECTION": "fixed_all_16",
        "ABCD16_REQUIRE_MYOPIC_PRECOMPUTE": "1", "ABCD16_FORBID_ADP_SEED_PRESOLVE": "1",
        "EXACT_BELIEF_BUILD_MODE": "dense", "GAUGED_REGIME_FEATURE_BANK": "ABCD_HAND",
        "GAUGED_REGIME_FEATURE_SEMANTICS": "abcd_hand_v1", "MYOPIC_SAFE_GUARDRAIL_ENABLED": "1",
        "MYOPIC_SAFE_EPSILON": "0.0", "ADP_TOL": "5e-7", "ADP_MAX_ITERS": "300",
    })
    return env

ORIGINAL8_FAMILIES = [
    "Extended_LowHigh_Aggressive",
    "Extended_LowHigh_Base",
    "Extended_MediumEven_Aggressive",
    "Extended_MediumEven_Base",
    "ThreeGeneration_LowHigh_Aggressive",
    "ThreeGeneration_LowHigh_Base",
    "ThreeGeneration_MediumEven_Aggressive",
    "ThreeGeneration_MediumEven_Base",
]

# Build exact index: suite + exact row_id
case_index = {case.setting.name: case for case in load_cases("original8")}

outdir = Path("output/original8")
results = []

for family in ORIGINAL8_FAMILIES:
    case = case_index.get(family)
    if case is None:
        print(f"[ERROR] {family} not found in original8 suite")
        continue

    print(f"\n{'='*60}")
    print(f"Running: {family}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    metrics, _raw, _meta = _run_setting(
        case.setting,
        RunnerSpec(label="abcd16", env=direct16_env()),
        benchmark_tier="abcd16",
        progress_prefix="abcd16-direct::original8",
    )
    dt = time.time() - t0

    row = {
        "suite": "original8",
        "row_id": family,
        "reproduced": {
            "ratio2": float(metrics["ratio2"]),
            "ratio3": float(metrics["ratio3"]),
        },
        "intermediate": {
            "V_star": metrics.get("exact_root_value"),
            "V_stop": metrics.get("stop_value"),
            "U_adp_phi": metrics.get("adp_phi"),
            "L_policy": metrics.get("production_policy_value"),
        },
        "seconds": round(dt, 1),
    }
    results.append(row)

    (outdir / f"{family}.json").write_text(json.dumps(row, indent=2))
    shutil.rmtree("output/.cache", ignore_errors=True)

    print(f"[DONE] {family}")
    print(f"  ratio2 = {row['reproduced']['ratio2']:.12f}")
    print(f"  ratio3 = {row['reproduced']['ratio3']:.12f}")
    print(f"  ({dt:.0f}s)")
    sys.stdout.flush()

# Write combined
combined = {"suite": "original8", "n_families": len(results), "rows": results}
Path("output/original8_combined.json").write_text(json.dumps(combined, indent=2))

print("\n" + "="*60)
print("ALL ORIGINAL8 FAMILIES DONE")
print("="*60)
for row in results:
    r = row["reproduced"]
    print(f"{row['row_id']:45s}  ratio2={r['ratio2']:.10f}  ratio3={r['ratio3']:.10f}  ({row['seconds']:.0f}s)")
PYEOF
