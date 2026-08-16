#!/bin/bash
#SBATCH --job-name=gnn_seeds
#SBATCH --output=experiments_after_understanding/gnn/results/seed_runs/slurm_%A_seed%a.log
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-2

cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p experiments_after_understanding/gnn/results/seed_runs
python experiments_after_understanding/gnn/run_seeds.py --device cpu --epochs 500 --seed $SLURM_ARRAY_TASK_ID
