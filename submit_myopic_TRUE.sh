#!/bin/bash
#SBATCH --job-name=myopic_TRUE_rerun
#SBATCH --output=incremental_experiments/results/slurm_myopic_TRUE_%A_task%a.log
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --requeue

# Regenerate experiments_after_understanding/{myopic,two_gene}/results/*.json
# using Kanix's canonical myopic_greedy (genetic_dp.policy.myopic) instead of
# the wrong one-step-lookahead reimplementation. task 0 = 3-gene, task 1 = 2-gene.

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  rm -f experiments_after_understanding/myopic/results/results.json
  /net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python experiments_after_understanding/myopic/run.py
else
  rm -f experiments_after_understanding/two_gene/results/myopic_results.json
  /net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python experiments_after_understanding/two_gene/myopic.py
fi
