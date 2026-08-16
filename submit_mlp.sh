#!/bin/bash
#SBATCH --job-name=mlp_mean_pool
#SBATCH --output=experiments_after_understanding/mlp/results/slurm_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/mlp/run.py --device cpu --epochs 500 --mode both
