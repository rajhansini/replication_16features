#!/bin/bash
#SBATCH --job-name=e1_fourway_log
#SBATCH --output=incremental_experiments/results/slurm_e1_log_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E1 four-way (DP/myopic/GNN/MLP) action log.
# Submit with --dependency=afterok:<gnn_train_jobid>,<mlp_train_jobid>
# incremental_experiments/log_e1_fourway.py
# 6 tasks: 0,1,2 -> 3-gene seeds 0,1,2 ; 3,4,5 -> 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  GENES=3
  SEED=$SLURM_ARRAY_TASK_ID
else
  GENES=2
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_e1_fourway.py --genes "$GENES" --seed "$SEED" --device cpu
