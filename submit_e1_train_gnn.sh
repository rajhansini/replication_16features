#!/bin/bash
#SBATCH --job-name=e1_gnn_train
#SBATCH --output=incremental_experiments/results/slurm_e1_train_gnn_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E1: state-grouped batching, loss still plain MSE (GNN-Q).
# incremental_experiments/e1_train_gnn_q.py / e1_train_two_gene_gnn_q.py
# 6 tasks: 0,1,2 -> 3-gene seeds 0,1,2 ; 3,4,5 -> 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p incremental_experiments/results

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=incremental_experiments/e1_train_gnn_q.py
  SEED=$SLURM_ARRAY_TASK_ID
else
  SCRIPT=incremental_experiments/e1_train_two_gene_gnn_q.py
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
