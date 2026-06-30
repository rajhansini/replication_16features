#!/bin/bash
#SBATCH --job-name=step5_retrain
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/step5_retrain_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/step5_retrain_%j.err

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

echo "=== Step 5 GNN retrain with corrected GeneB parameters ==="
echo "Job ID: $SLURM_JOB_ID"
date

$PYTHON ground-up-experiments/step5_gnn/run.py --fresh

echo ""
echo "=== DONE ==="
date
