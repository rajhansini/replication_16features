#!/bin/bash
#SBATCH --job-name=extended_3gene_cfg_e6
#SBATCH --output=incremental_experiments/results/slurm_extended3g_e6_%A_task%a.log
#SBATCH --time=03:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-35
#SBATCH --requeue

# Extended family, 3-gene, E6 (sum pooling) checkpoints, one config per task
# -- same proven split as submit_extended_3gene_perconfig.sh, --variant e6.
# 36 tasks: 12 configs (6 regimes x 2 presets) x 3 seeds

cd /net/projects/ranalab/rajhansini/replication_16features

REGIMES=(LowHigh MediumEven LowLow HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % 12 ))
SEED=$(( SLURM_ARRAY_TASK_ID / 12 ))
REGIME=${REGIMES[$(( CONFIG_IDX / 2 ))]}
PRESET=${PRESETS[$(( CONFIG_IDX % 2 ))]}

echo "task=$SLURM_ARRAY_TASK_ID -> regime=$REGIME preset=$PRESET seed=$SEED variant=e6"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes 3 --seed "$SEED" \
  --regime "$REGIME" --preset "$PRESET" --variant e6 --device cpu
