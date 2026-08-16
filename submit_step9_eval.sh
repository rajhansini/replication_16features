#!/bin/bash
#SBATCH --job-name=step9_eval
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/step9_eval_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/step9_eval_%j.err

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

echo "=== Step 9: Eval 3-gene GNN (ratio2) ==="
echo "Job ID: $SLURM_JOB_ID"
date

$PYTHON -u ground-up-experiments/step9_gnn_3gene/eval.py \
    --device cuda

echo ""
echo "=== DONE ==="
date
