#!/bin/bash
#SBATCH --job-name=step10_train
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/output/step10_train_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/output/step10_train_%j.err

PYTHON=/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python
cd /net/projects/ranalab/rajhansini/replication_16features

echo "=== Step 10: Structural GNN (node_feat_dim=13, +n_parents/n_children/depth) ==="
echo "Job ID: $SLURM_JOB_ID"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "Using device: $DEVICE"

$PYTHON -u ground-up-experiments/step10_structural_gnn/run.py \
    --device $DEVICE \
    --epochs 500

echo ""
echo "=== DONE ==="
date
