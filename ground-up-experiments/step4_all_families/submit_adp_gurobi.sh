#!/bin/bash
#SBATCH --job-name=adp_gurobi
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step4_all_families/results/adp_gurobi_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step4_all_families/results/adp_gurobi_%j.err

export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic
cd /net/projects/ranalab/rajhansini/replication_16features
/opt/conda/bin/python ground-up-experiments/step4_all_families/adp_gurobi.py
