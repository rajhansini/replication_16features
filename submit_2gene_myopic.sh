#!/bin/bash
#SBATCH --job-name=myopic_2gene
#SBATCH --output=experiments_after_understanding/two_gene/results/slurm_myopic_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/two_gene/myopic.py
