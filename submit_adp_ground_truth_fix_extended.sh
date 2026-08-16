#!/bin/bash
#SBATCH --job-name=adp_gt_fix_ext
#SBATCH --output=fresh_dataset/adp/slurm_fix_ext_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=900G
#SBATCH --cpus-per-task=8
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-11%2
#SBATCH --requeue

# ADP ground truth: the 12 3-gene Extended configs that OOM'd at 350G / one
# timed out at 4hr under 18-way concurrency. %2 respects the Gurobi WLS
# 2-session cap (same fix as the cheap tier) AND removes the CPU/memory
# contention from 18 simultaneous solves, which may itself help these finish
# faster even without more time budget. Bumped to 900G (only p003-p010 have
# enough memory; p001/p002 do not, SLURM will place accordingly) since exact
# DP alone took ~100GB for 3-gene Extended earlier this session and ADP adds
# substantially more. If any of these still don't finish in 4hr, that's a
# real infrastructure ceiling to report, not a bug to silently paper over.
cd /net/projects/ranalab/rajhansini/replication_16features
export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic

REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

REG_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
PRE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
REGIME=${REGIMES[$REG_IDX]}
PRESET=${PRESETS[$PRE_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> genes=3 family=Extended regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_adp_ground_truth.py --genes 3 --family Extended --regime "$REGIME" --preset "$PRESET"
