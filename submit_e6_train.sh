#!/bin/bash
#SBATCH --job-name=e6_sumpool_train
#SBATCH --output=incremental_experiments/results/slurm_e6_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-11
#SBATCH --requeue

# E6: sum pooling instead of mean pooling, GNN-Q and MLP-Q, both gene counts.
# Everything else held fixed at E4 (GNN) / E2 (MLP) settings.
# incremental_experiments/e6_train_{gnn_q,two_gene_gnn_q,mlp_q,two_gene_mlp_q}.py
# 12 tasks: 0-2 3gene-GNN seeds 0-2, 3-5 2gene-GNN seeds 0-2,
#           6-8 3gene-MLP seeds 0-2, 9-11 2gene-MLP seeds 0-2

cd /net/projects/ranalab/rajhansini/replication_16features

SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
GROUP=$(( SLURM_ARRAY_TASK_ID / 3 ))

case $GROUP in
  0) SCRIPT=incremental_experiments/e6_train_gnn_q.py ;;
  1) SCRIPT=incremental_experiments/e6_train_two_gene_gnn_q.py ;;
  2) SCRIPT=incremental_experiments/e6_train_mlp_q.py ;;
  3) SCRIPT=incremental_experiments/e6_train_two_gene_mlp_q.py ;;
esac

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
