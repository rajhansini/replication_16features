#!/bin/bash
#SBATCH --job-name=e2_fourway_log
#SBATCH --output=incremental_experiments/results/slurm_e2_log_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E2 four-way (DP/myopic/GNN/MLP) action log — uses the ALREADY-TRAINED
# fixing_gnn_q ce-variant checkpoints (state-grouped batching + CE,
# lambda_ce=1.0). No new training needed for this rung.
# incremental_experiments/log_e2_fourway.py
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
  incremental_experiments/log_e2_fourway.py --genes "$GENES" --seed "$SEED" --device cpu
