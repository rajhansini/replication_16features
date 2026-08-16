#!/bin/bash
#SBATCH --job-name=2gene_seeds
#SBATCH --output=experiments_after_understanding/two_gene/results/seed_runs/slurm_%A_seed%a.log
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-2

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p experiments_after_understanding/two_gene/results/seed_runs
python experiments_after_understanding/two_gene/run_seeds.py --device cpu --epochs 500 --seed $SLURM_ARRAY_TASK_ID
