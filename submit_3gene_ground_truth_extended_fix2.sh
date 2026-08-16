#!/bin/bash
#SBATCH --job-name=3gene_gt_ext_fix2
#SBATCH --output=fresh_dataset/3gene/slurm_extended_fix2_%A.log
#SBATCH --time=03:00:00
#SBATCH --mem=350G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu

# The one remaining myopic/DP ground-truth config from the earlier fix batch
# that timed out at 1hr on the general partition -- Extended_HighHigh_Base,
# 3-gene. Bumped time and moved to peanut-cpu (less contention, bigger nodes).
cd /net/projects/ranalab/rajhansini/replication_16features

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_3gene_ground_truth.py --family Extended --regime HighHigh --preset Base
