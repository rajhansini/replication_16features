#!/bin/bash
#SBATCH --job-name=mlp_q_eval_rerun
#SBATCH --output=experiments_after_understanding/q_learning/results/seed_runs/slurm_mlp_q_evalrerun_%A_seed%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --requeue

# Reruns eval only, reusing the already-trained checkpoints from
# submit_mlp_q_seeds.sh (job 2110163). Training (Q* from exact DP) was sound;
# only the eval metric changed (DP-forced-trajectory action agreement ->
# q_rollout, a real greedy-Q(s,a) policy rollout). See mlp_q.py's
# evaluate_rollout for details.

cd /net/projects/ranalab/rajhansini/replication_16features
/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  experiments_after_understanding/q_learning/mlp_q.py --device cpu --mode eval --seed $SLURM_ARRAY_TASK_ID
