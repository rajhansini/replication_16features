#!/bin/bash
#SBATCH --job-name=e7_neighborpool_train
#SBATCH --output=incremental_experiments/results/slurm_e7_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E7: neighbor-mean pooling (candidate's direct 1-hop relatives only, mean
# aggregator held fixed vs E4) instead of E4's global-mean pooling. GNN-Q
# only for now (MLP-Q needs edge_index added to its pipeline -- follow-up).
# Smoke tests showed 2-gene training notably slower per state-group than
# 3-gene (2.75x) in an ad-hoc, possibly node-contended Bash run -- bumped
# time budget to 6hr as a safety margin; per-epoch checkpointing + --requeue
# already in place either way, so a timeout loses nothing, just resumes.
# incremental_experiments/e7_train_gnn_q.py / e7_train_two_gene_gnn_q.py
# 6 tasks: 0,1,2 -> 3-gene seeds 0,1,2 ; 3,4,5 -> 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=incremental_experiments/e7_train_gnn_q.py
  SEED=$SLURM_ARRAY_TASK_ID
else
  SCRIPT=incremental_experiments/e7_train_two_gene_gnn_q.py
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
