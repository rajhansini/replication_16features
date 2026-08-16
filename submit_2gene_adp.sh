#!/bin/bash
#SBATCH --job-name=adp_2gene_eval
#SBATCH --output=experiments_after_understanding/two_gene/results/slurm_adp_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

cd /net/projects/ranalab/rajhansini/replication_16features
python experiments_after_understanding/two_gene/adp_run.py
