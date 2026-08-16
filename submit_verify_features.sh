#!/bin/bash
#SBATCH --job-name=verify_feat
#SBATCH --output=experiments_after_understanding/q_learning/results/slurm_verify_feat_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/verify_greedy_pick_features.py
