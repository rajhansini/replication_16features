#!/bin/bash
#SBATCH --job-name=e5_fourway_log
#SBATCH --output=incremental_experiments/results/slurm_e5_log_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# E5 four-way (DP/myopic/GNN/MLP) action log — GNN uses the E5 (n_rounds=3)
# checkpoints, MLP reuses E2's (unchanged, rounds only affects GNN's MP).
# incremental_experiments/log_e5_fourway.py
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
  incremental_experiments/log_e5_fourway.py --genes "$GENES" --seed "$SEED" --device cpu
