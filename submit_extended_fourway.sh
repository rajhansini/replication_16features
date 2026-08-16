#!/bin/bash
#SBATCH --job-name=extended_fourway
#SBATCH --output=incremental_experiments/results/slurm_extended_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# Extended family (6 people, branching Uncle sibling) OOD four-way action log
# -- DP/myopic (Kanix's canonical myopic_greedy)/GNN-Q(E4)/MLP-Q(E2), no
# retraining. 2-gene builds on the fly (fast); 3-gene solves exact DP from
# scratch (no cache exists for Extended at 3 genes) -- may be slow/large,
# hence the 64G/4hr budget instead of the usual 32G/2hr for a fourway log.
# incremental_experiments/log_extended_fourway.py
# 6 tasks: 0,1,2 -> 2-gene seeds 0,1,2 ; 3,4,5 -> 3-gene seeds 0,1,2

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  GENES=2
  SEED=$SLURM_ARRAY_TASK_ID
else
  GENES=3
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes "$GENES" --seed "$SEED" --device cpu
