#!/bin/bash
#SBATCH --job-name=e9_combopool_train
#SBATCH --output=incremental_experiments/results/slurm_e9_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E9: combined pooling -- head sees BOTH E6's global-sum readout AND E7's
# local neighbor-mean readout concatenated, instead of either alone.
# embed() (message passing) unchanged from E4/E6/E7. GNN-Q only.
# incremental_experiments/e9_train_gnn_q.py / e9_train_two_gene_gnn_q.py
# 6 tasks: 0,1,2 -> 3-gene seeds 0,1,2 ; 3,4,5 -> 2-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  SCRIPT=incremental_experiments/e9_train_gnn_q.py
  SEED=$SLURM_ARRAY_TASK_ID
else
  SCRIPT=incremental_experiments/e9_train_two_gene_gnn_q.py
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED"
