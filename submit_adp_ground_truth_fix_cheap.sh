#!/bin/bash
#SBATCH --job-name=adp_gt_fix_cheap
#SBATCH --output=fresh_dataset/adp/slurm_fix_cheap_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-17%2
#SBATCH --requeue

# ADP ground truth: the 18 configs that FAILED with "Too many sessions, 5
# active sessions for a baseline of 2" -- our academic Gurobi WLS license
# only allows 2 CONCURRENT sessions, but the first array run let 18 tasks
# run in parallel. %2 throttles to at most 2 concurrent tasks, matching the
# license. build_adp_ground_truth.py already skips any config that already
# has a saved result, so this is safe even if some finished already.
cd /net/projects/ranalab/rajhansini/replication_16features
export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic

CONFIGS=(
  "2 Trio LowHigh Aggressive"
  "2 Trio MediumEven Base"
  "2 Trio MediumEven Aggressive"
  "2 Trio MixedA Base"
  "3 Trio MediumEven Base"
  "3 Trio MediumEven Aggressive"
  "3 ThreeGeneration LowHigh Base"
  "3 ThreeGeneration LowHigh Aggressive"
  "3 ThreeGeneration MediumEven Base"
  "3 ThreeGeneration MediumEven Aggressive"
  "3 ThreeGeneration LowLow Base"
  "3 ThreeGeneration LowLow Aggressive"
  "3 ThreeGeneration HighHigh Base"
  "3 ThreeGeneration HighHigh Aggressive"
  "3 ThreeGeneration MixedA Base"
  "3 ThreeGeneration MixedA Aggressive"
  "3 ThreeGeneration MixedB Base"
  "3 ThreeGeneration MixedB Aggressive"
)

read -r GENES FAMILY REGIME PRESET <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES family=$FAMILY regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_adp_ground_truth.py --genes "$GENES" --family "$FAMILY" --regime "$REGIME" --preset "$PRESET"
