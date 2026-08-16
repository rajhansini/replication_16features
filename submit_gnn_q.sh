#!/bin/bash
#SBATCH --job-name=gnn_q
#SBATCH --output=experiments_after_understanding/q_learning/results/slurm_gnn_q_%j.log
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p experiments_after_understanding/q_learning/results
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/gnn_q.py --device cpu --epochs 200 --mode both
