#!/bin/bash
#SBATCH --job-name=extended_fourway
#SBATCH --output=incremental_experiments/results/slurm_extended_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Extended family (6 people, branching Uncle sibling) OOD four-way action log
# -- DP/myopic (Kanix's canonical myopic_greedy)/GNN-Q(E4)/MLP-Q(E2), no
# retraining. 2-gene builds on the fly (fast); 3-gene solves exact DP from
# scratch (no cache exists for Extended at 3 genes) -- may be slow/large,
# hence the 64G/4hr budget instead of the usual 32G/2hr for a fourway log.
# incremental_experiments/log_extended_fourway.py
# 6 tasks: 0,1,2 -> 2-gene seeds 0,1,2 ; 3,4,5 -> 3-gene seeds 0,1,2
#
# RESUME (added 2026-08-24): 3-gene Extended solves exact DP over 9,190,992
# states per config with no cache, and 12 configs do not fit in one 4h slot --
# the first attempt (jobs 2206213 / 2206261) died at the wall having written
# NOTHING, because the JSON was only produced after all 12 finished. Now every
# config is checkpointed to seed*/partial*/ the moment it is solved, and a
# requeued task skips what it already has. Resubmitting this array is safe:
# partials are keyed to the SLURM job id, so a requeue resumes but a fresh
# sbatch wipes them rather than inheriting numbers from a superseded run.
# Use --force on the python call to re-solve a task from scratch.

cd /net/projects/ranalab/rajhansini/replication_16features

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  GENES=2
  SEED=$SLURM_ARRAY_TASK_ID
else
  GENES=3
  SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi

echo "task=$SLURM_ARRAY_TASK_ID -> genes=$GENES seed=$SEED"


# Requeue on the pre-wall-clock signal instead of being killed outright.
# 300s before the 4h MaxTime SLURM sends USR1 here; the task requeues itself
# and log_extended_fourway.py picks back up from its per-config partials.
# The child keeps running until SLURM's own SIGTERM arrives, so the last
# partial write lands as late as possible.
requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 received (300s to wall clock) -- requeueing task ${SLURM_ARRAY_TASK_ID}; it will resume from its solved configs"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes "$GENES" --seed "$SEED" --device cpu &
CHILD=$!
wait "$CHILD"
