#!/bin/bash
#SBATCH --job-name=save_table
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/save_table_%j.out

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

$PYTHON scripts/save_final_table.py
