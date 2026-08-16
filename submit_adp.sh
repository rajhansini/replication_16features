#!/bin/bash
#SBATCH --job-name=adp_eval
#SBATCH --output=experiments_after_understanding/adp/results/slurm_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/adp/run.py
