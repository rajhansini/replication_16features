#!/bin/bash
#SBATCH --job-name=step8_scaling
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/step8_scaling_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/step8_scaling_%j.err

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

echo "=== Step 8: Exact DP scaling experiment (k genes) ==="
echo "Job ID: $SLURM_JOB_ID"
date

$PYTHON -u ground-up-experiments/step8_scaling/run.py

echo ""
echo "=== DONE ==="
date
