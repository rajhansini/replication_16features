#!/bin/bash
#SBATCH --job-name=two_gene_mean_pool
#SBATCH --output=experiments_after_understanding/two_gene/results/slurm_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/two_gene/run.py --device cpu --epochs 500
