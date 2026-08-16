#!/bin/bash
#SBATCH --job-name=verify_myopic_TRUE
#SBATCH --output=incremental_experiments/results/slurm_verify_myopic_%A_task%a.log
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-23
#SBATCH --requeue

# Re-verify the myopic baseline using Kanix's ORIGINAL genetic_dp/policy/
# {myopic.py,baselines.py} (myopic_greedy = zero-lookahead), one config per
# array task so memory is released between configs instead of accumulating
# in one long-lived process.
# incremental_experiments/verify_myopic_one.py
# 24 tasks: 0-11 -> 2-gene (6 regimes x 2 presets), 12-23 -> 3-gene (same)

cd /net/projects/ranalab/rajhansini/replication_16features

REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

if [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
  GENES=2
  IDX=$SLURM_ARRAY_TASK_ID
else
  GENES=3
  IDX=$(( SLURM_ARRAY_TASK_ID - 12 ))
fi

REGIME=${REGIMES[$(( IDX / 2 ))]}
PRESET=${PRESETS[$(( IDX % 2 ))]}

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/verify_myopic_one.py --genes "$GENES" --regime "$REGIME" --preset "$PRESET"
