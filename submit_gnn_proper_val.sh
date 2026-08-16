#!/bin/bash
#SBATCH --job-name=gnn_proper_val
#SBATCH --output=experiments_after_understanding/gnn_proper_val/results/slurm_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/gnn_proper_val/run.py --device cpu --epochs 500 --mode both
