#!/bin/bash
#SBATCH --job-name=gnn_q_depth_ablation
#SBATCH --output=things_to_improve_Q_star_experiments/depth_ablation/results/slurm_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-11
#SBATCH --requeue

# Message-passing DEPTH ABLATION for GNN-Q (Q-track, 3-gene).
# 12 tasks = 4 round-counts {0,1,2,3} x 3 seeds {0,1,2}.
#   task_id // 3 -> n_rounds   (0,0,0, 1,1,1, 2,2,2, 3,3,3)
#   task_id % 3  -> seed       (0,1,2, 0,1,2, 0,1,2, 0,1,2)

cd /net/projects/ranalab/rajhansini/replication_16features

export Q_ABLATION_RESULTS_DIR=/net/projects/ranalab/rajhansini/replication_16features/things_to_improve_Q_star_experiments/depth_ablation/results
mkdir -p "$Q_ABLATION_RESULTS_DIR"

N_ROUNDS=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "task=$SLURM_ARRAY_TASK_ID -> n_rounds=$N_ROUNDS seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  things_to_improve_Q_star_experiments/depth_ablation/gnn_q_rounds.py \
  --device cpu --epochs 500 --mode both --seed "$SEED" --n_rounds "$N_ROUNDS"
