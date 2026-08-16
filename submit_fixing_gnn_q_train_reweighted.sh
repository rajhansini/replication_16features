#!/bin/bash
#SBATCH --job-name=gnn_q_ce_rw_train
#SBATCH --output=fixing_gnn_q/results/slurm_train_rw_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# Retrain GNN-Q with combined MSE + cross-entropy loss + depth reweighting
# (fixing_gnn_q/losses.py: combined_loss + depth_weight). Fix 1 (ranking loss)
# + Fix 2 (upweight shallow/root states) together.
# 6 tasks:
#   0,1,2 -> 3-gene, seeds 0,1,2
#   3,4,5 -> 2-gene, seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p fixing_gnn_q/results

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=fixing_gnn_q/train_gnn_q_ce_reweighted.py
  SEED=$SLURM_ARRAY_TASK_ID
else
  SCRIPT=fixing_gnn_q/train_two_gene_gnn_q_ce_reweighted.py
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
