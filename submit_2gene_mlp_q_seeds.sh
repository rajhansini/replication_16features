#!/bin/bash
#SBATCH --job-name=2g_mlp_q
#SBATCH --output=experiments_after_understanding/q_learning/results_2gene/seed_runs/slurm_mlp_q_%A_seed%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --requeue

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p experiments_after_understanding/q_learning/results_2gene/seed_runs
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/two_gene_mlp_q.py --device cpu --epochs 500 --mode both --seed $SLURM_ARRAY_TASK_ID
