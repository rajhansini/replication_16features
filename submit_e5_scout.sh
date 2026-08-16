#!/bin/bash
#SBATCH --job-name=e5_gnn_rounds_scout
#SBATCH --output=incremental_experiments/results/slurm_e5_scout_%A_task%a.log
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E5 scout: message-passing round count sweep, seed 0 only, n_rounds in {1,3,4}.
# n_rounds=2 is E4 itself (already trained) -- not rerun here.
# incremental_experiments/e5_train_gnn_q.py / e5_train_two_gene_gnn_q.py
# 6 tasks: 0,1,2 -> 3-gene rounds 1,3,4 ; 3,4,5 -> 2-gene rounds 1,3,4

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p incremental_experiments/results

ROUNDS_LIST=(1 3 4)

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=incremental_experiments/e5_train_gnn_q.py
  ROUNDS=${ROUNDS_LIST[$SLURM_ARRAY_TASK_ID]}
else
  SCRIPT=incremental_experiments/e5_train_two_gene_gnn_q.py
  ROUNDS=${ROUNDS_LIST[$(( SLURM_ARRAY_TASK_ID - 3 ))]}
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT n_rounds=$ROUNDS seed=0"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed 0 --n_rounds "$ROUNDS"
