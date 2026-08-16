#!/bin/bash
#SBATCH --job-name=extended_fourway_e7
#SBATCH --output=incremental_experiments/results/slurm_extended_e7_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --requeue

# Extended family (6 people, branching Uncle sibling), 2-gene, E7 (neighbor-
# mean pooling) checkpoints -- same as submit_extended_fourway_e6.sh but --variant e7.
# 3 tasks: 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

SEED=$SLURM_ARRAY_TASK_ID

echo "task=$SLURM_ARRAY_TASK_ID -> genes=2 seed=$SEED variant=e7"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes 2 --seed "$SEED" --variant e7 --device cpu
