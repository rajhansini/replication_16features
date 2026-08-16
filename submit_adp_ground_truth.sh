#!/bin/bash
#SBATCH --job-name=adp_ground_truth
#SBATCH --output=fresh_dataset/adp/slurm_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=350G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-95
#SBATCH --requeue

# ADP ground truth: Kanix's real run_and_compare_solvers (dual-DP LP solve,
# same direct16_env() locked config replicate.py uses) on all 96 of this
# project's configs (4 families x 6 regimes x 2 presets x 2 gene counts).
# One config per task -- each is a real Gurobi solve, ~5-10min based on the
# smoke test / earlier replicate.py timings.
cd /net/projects/ranalab/rajhansini/replication_16features

export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic

FAMILIES=(Trio Nuclear ThreeGeneration Extended)
REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

# 96 = 2 (genes) x 4 (families) x 6 (regimes) x 2 (presets)
GENES_IDX=$(( SLURM_ARRAY_TASK_ID / 48 ))
REM=$(( SLURM_ARRAY_TASK_ID % 48 ))
FAM_IDX=$(( REM / 12 ))
REM2=$(( REM % 12 ))
REG_IDX=$(( REM2 / 2 ))
PRE_IDX=$(( REM2 % 2 ))

GENES=$(( GENES_IDX == 0 ? 2 : 3 ))
FAMILY=${FAMILIES[$FAM_IDX]}
REGIME=${REGIMES[$REG_IDX]}
PRESET=${PRESETS[$PRE_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES family=$FAMILY regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_adp_ground_truth.py --genes "$GENES" --family "$FAMILY" --regime "$REGIME" --preset "$PRESET"
