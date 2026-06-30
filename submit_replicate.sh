#!/bin/bash
#SBATCH --job-name=replicate_adp
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/replicate_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/replicate_%j.err

export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic
cd /net/projects/ranalab/rajhansini/replication_16features
mkdir -p output
/opt/conda/bin/python replicate.py
