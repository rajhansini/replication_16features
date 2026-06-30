#!/bin/bash
#SBATCH --job-name=adp_all
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step4_all_families/results/adp_all_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step4_all_families/results/adp_all_%j.err

cd /net/projects/ranalab/rajhansini/replication_16features
/opt/conda/bin/python ground-up-experiments/step4_all_families/adp_all_families.py
