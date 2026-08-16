#!/bin/bash
#SBATCH --job-name=mlp_q_eval
#SBATCH --output=experiments_after_understanding/q_learning/results/slurm_mlp_q_eval_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/mlp_q.py --device cpu --mode eval
