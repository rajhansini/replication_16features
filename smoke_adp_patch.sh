#!/bin/bash
#SBATCH --job-name=smoke_adp_patch
#SBATCH --output=fresh_dataset/adp/smoke_patch_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=900G
#SBATCH --cpus-per-task=8
#SBATCH --partition=peanut-cpu

cd /net/projects/ranalab/rajhansini/replication_16features
export GRB_LICENSE_FILE=/home/rajhansini/.gurobi/gurobi.lic

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_adp_ground_truth.py --genes 3 --family Extended --regime LowHigh --preset Base
