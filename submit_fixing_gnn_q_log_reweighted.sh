#!/bin/bash
#SBATCH --job-name=gnn_q_ce_rw_log
#SBATCH --output=fixing_gnn_q/results/slurm_log_rw_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# Myopic-vs-GNN(CE+reweighted-trained) trajectory logs
# (fixing_gnn_q/log_myopic_vs_gnn.py --variant ce_reweighted).
# Submit with --dependency=afterok:<train_job_id>
# (submit_fixing_gnn_q_train_reweighted.sh) so it only runs once those
# checkpoints exist.
# 6 tasks:
#   0,1,2 -> 3-gene, seeds 0,1,2
#   3,4,5 -> 2-gene, seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  GENES=3
  SEED=$SLURM_ARRAY_TASK_ID
else
  GENES=2
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  fixing_gnn_q/log_myopic_vs_gnn.py --genes "$GENES" --seed "$SEED" --device cpu --variant ce_reweighted
