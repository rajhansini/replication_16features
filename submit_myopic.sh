#!/bin/bash
#SBATCH --job-name=myopic_eval
#SBATCH --output=experiments_after_understanding/myopic/results/slurm_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/myopic/run.py
