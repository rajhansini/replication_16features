#!/bin/bash
#SBATCH --job-name=e5_gnn_rounds3_train
#SBATCH --output=incremental_experiments/results/slurm_e5_train_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-2
#SBATCH --requeue

# E5, n_rounds=3, 3-gene only, ALL 3 seeds retrained from scratch -- the
# previous version of this script assumed seed0 was already correctly
# trained by an earlier scout run and only filled in seeds 1,2. That seed0
# checkpoint was trained on the pre-fix (wrong allele-frequency) 3-gene
# data, so it must be redone too, not reused.

cd /net/projects/ranalab/rajhansini/replication_16features
SEED=$SLURM_ARRAY_TASK_ID

echo "task=$SLURM_ARRAY_TASK_ID -> script=incremental_experiments/e5_train_gnn_q.py seed=$SEED n_rounds=3"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/e5_train_gnn_q.py --device cpu --epochs 500 --mode both --seed "$SEED" --n_rounds 3
