#!/bin/bash
#SBATCH --job-name=overfit2g_eval
#SBATCH --output=incremental_experiments/results/slurm_overfit2g_%A_seed%a.log
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-2

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/eval_train_overfit_2gene.py --seed "$SLURM_ARRAY_TASK_ID"
