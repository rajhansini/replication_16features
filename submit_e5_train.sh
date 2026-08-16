#!/bin/bash
#SBATCH --job-name=e5_gnn_rounds3_train
#SBATCH --output=incremental_experiments/results/slurm_e5_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-3
#SBATCH --requeue

# E5 (committed): n_rounds=3 won the scout sweep for both gene counts
# (3-gene 0.0497, 2-gene 0.1038, both beating n_rounds=1/2/4). Filling in
# seeds 1,2 to match E0-E4's 3-seed rigor -- seed 0 already trained by the
# scout (incremental_experiments/results/e5_gnn_rounds3{,_2gene}/seed_runs/seed0/).
# 4 tasks: 0,1 -> 3-gene seeds 1,2 ; 2,3 -> 2-gene seeds 1,2

cd /net/projects/ranalab/rajhansini/replication_16features

SEEDS=(1 2)

if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
  SCRIPT=incremental_experiments/e5_train_gnn_q.py
  SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
  SCRIPT=incremental_experiments/e5_train_two_gene_gnn_q.py
  SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID - 2 ))]}
fi

echo "task=$SLURM_ARRAY_TASK_ID -> script=$SCRIPT seed=$SEED n_rounds=3"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  "$SCRIPT" --device cpu --epochs 500 --mode both --seed "$SEED" --n_rounds 3
