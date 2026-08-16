#!/bin/bash
#SBATCH --job-name=2gene_ground_truth
#SBATCH --output=incremental_experiments/results/slurm_2gene_ground_truth_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Consolidated 2-gene ground truth (exact DP + Kanix's canonical myopic_greedy)
# for all 48 configs: TRAIN (Trio, Nuclear) + TEST (ThreeGeneration, Extended)
# x 6 regimes x {Base, Aggressive}.
# incremental_experiments/build_2gene_ground_truth.py

cd /net/projects/ranalab/rajhansini/replication_16features

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_2gene_ground_truth.py
