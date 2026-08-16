#!/bin/bash
#SBATCH --job-name=gnn_q_eval
#SBATCH --output=experiments_after_understanding/q_learning/results/slurm_gnn_q_eval_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/gnn_q.py --device cpu --mode eval
