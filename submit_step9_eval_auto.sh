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

CKPT=ground-up-experiments/step9_gnn_3gene/results/gnn3_ckpt.pt
MODEL=ground-up-experiments/step9_gnn_3gene/results/gnn3_model.pt

# Extract model weights from checkpoint (works even if training was killed mid-run)
if [ -f "$CKPT" ]; then
    echo "Extracting model from checkpoint..."
    $PYTHON -c "
import torch, sys
ckpt = torch.load('$CKPT', map_location='cpu')
print(f'  checkpoint epoch: {ckpt[\"epoch\"]}')
print(f'  train_loss: {ckpt[\"history\"][\"train_loss\"][-1]:.6f}')
print(f'  val_loss:   {ckpt[\"history\"][\"val_loss\"][-1]:.6f}')
torch.save(ckpt['model_state'], '$MODEL')
print(f'  saved → $MODEL')
"
else
    echo "ERROR: no checkpoint found at $CKPT"
    exit 1
fi

echo ""
$PYTHON -u ground-up-experiments/step9_gnn_3gene/eval.py --device cuda --fresh

echo ""
echo "=== DONE ==="
date
