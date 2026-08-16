#!/bin/bash
#SBATCH --job-name=threeway_actions
#SBATCH --output=things_to_improve_Q_star_experiments/action_compare/results/slurm_threeway_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# Three-way action logs: GNN-Q vs MLP-Q vs MYOPIC (no training).
# 6 tasks:
#   0,1,2 -> 3-gene, seeds 0,1,2
#   3,4,5 -> 2-gene, seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

mkdir -p things_to_improve_Q_star_experiments/action_compare/results

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  GENES=3
  SEED=$SLURM_ARRAY_TASK_ID
else
  GENES=2
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  things_to_improve_Q_star_experiments/action_compare/log_threeway.py \
  --genes "$GENES" --seed "$SEED" --device cpu
