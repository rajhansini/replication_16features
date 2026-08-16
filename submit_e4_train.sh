#!/bin/bash
#SBATCH --job-name=e4_gnn_bidir_train
#SBATCH --output=incremental_experiments/results/slurm_e4_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E4: bidirectional message passing (GNN-Q only — R1 fix). MLP-Q has no
# message passing, so it's unaffected and reuses E2's checkpoint.
# incremental_experiments/e4_train_gnn_q.py / e4_train_two_gene_gnn_q.py
# 6 tasks: 0,1,2 -> 3-gene seeds 0,1,2 ; 3,4,5 -> 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p incremental_experiments/results

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=incremental_experiments/e4_train_gnn_q.py
  SEED=$SLURM_ARRAY_TASK_ID
else
  SCRIPT=incremental_experiments/e4_train_two_gene_gnn_q.py
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
