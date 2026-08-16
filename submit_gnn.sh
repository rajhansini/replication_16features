#!/bin/bash
#SBATCH --job-name=gnn_eval
#SBATCH --output=experiments_after_understanding/gnn/results/slurm_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/gnn/run.py --device cpu --epochs 500 --mode eval
