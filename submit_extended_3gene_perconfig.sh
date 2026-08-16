#!/bin/bash
#SBATCH --job-name=extended_3gene_cfg
#SBATCH --output=incremental_experiments/results/slurm_extended3g_%A_task%a.log
#SBATCH --time=03:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-35
#SBATCH --requeue

# Extended family, 3-gene, one config per task (the all-12-configs-per-seed
# version timed out at 4hr with no partial output -- this splits so each
# task only solves ONE config, safely inside the time limit, and writes its
# own checkpoint file immediately, so nothing is lost if anything fails.
# incremental_experiments/log_extended_fourway.py --regime X --preset Y
# 36 tasks: 12 configs (6 regimes x 2 presets) x 3 seeds

cd /net/projects/ranalab/rajhansini/replication_16features

REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % 12 ))
SEED=$(( SLURM_ARRAY_TASK_ID / 12 ))
REGIME=${REGIMES[$(( CONFIG_IDX / 2 ))]}
PRESET=${PRESETS[$(( CONFIG_IDX % 2 ))]}

echo "task=$SLURM_ARRAY_TASK_ID -> regime=$REGIME preset=$PRESET seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes 3 --seed "$SEED" \
  --regime "$REGIME" --preset "$PRESET" --device cpu
