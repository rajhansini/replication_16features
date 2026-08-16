#!/bin/bash
#SBATCH --job-name=step9_build
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/step9_build_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/step9_build_%j.err

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

echo "=== Step 9: Build ThreeGeneration 3-gene datasets ==="
echo "Job ID: $SLURM_JOB_ID"
date

$PYTHON -u ground-up-experiments/step9_gnn_3gene/build_datasets.py

echo ""
echo "=== DONE ==="
date
