#!/bin/bash
#SBATCH --job-name=3gene_gt_cheap
#SBATCH --output=fresh_dataset/3gene/slurm_cheap_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# 3-gene ground truth: the 36 cheap configs (Trio, Nuclear, ThreeGeneration),
# read from the already-built cache -- fast.
cd /net/projects/ranalab/rajhansini/replication_16features

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_3gene_ground_truth.py
