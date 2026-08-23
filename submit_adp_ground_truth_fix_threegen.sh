#!/bin/bash
#SBATCH --job-name=adp_gt_fix_3g
#SBATCH --output=fresh_dataset/adp/slurm_fix_3g_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-11%2
#SBATCH --requeue

# ADP ground truth: the 12 3-gene ThreeGeneration configs that OOM'd at 64G.
# 1,054,528 states -- cheap for exact-DP/myopic, but ADP's LP row-generation
# needs far more memory. Bumped to 400G. %2 keeps respecting the Gurobi WLS
# 2-concurrent-session cap. build_adp_ground_truth.py skips already-done
# configs, so safe to run even though some in this family may have partially
# succeeded elsewhere.
cd /net/projects/ranalab/rajhansini/replication_16features
export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic

REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

REG_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
PRE_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
REGIME=${REGIMES[$REG_IDX]}
PRESET=${PRESETS[$PRE_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> genes=3 family=ThreeGeneration regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_adp_ground_truth.py --genes 3 --family ThreeGeneration --regime "$REGIME" --preset "$PRESET"
