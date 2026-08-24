#!/bin/bash
#SBATCH --job-name=ext_e6e7_rerun
#SBATCH --output=incremental_experiments/results/slurm_ext_e6e7_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-11
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Extended-family (OOD) four-way logs for the E6 and E7 variants, BOTH gene
# counts, 3 seeds -- 12 tasks.
#
# Why this exists: submit_extended_fourway.sh covers only --variant e4, and the
# pre-existing submit_extended_fourway_e6.sh / _e7.sh are 2-gene only (3 tasks
# each). The briefing's Extended section quotes GNN-Q(E6) and GNN-Q(E7) rows for
# 3-gene as well, and those came from results/extended/3gene/*/configs_e6 and
# configs_e7, last written 07-26 and 08-09 -- i.e. before the 3-gene
# allele-frequency fix (08-15/16) and the E0-E9 retrain (08-22). Without this
# job the Extended section would still be half stale after job 2206213.
#
# 12 tasks = 2 variants x {3-gene, 2-gene} x 3 seeds.
# 256G matches submit_extended_fourway.sh: 3-gene Extended solves exact DP from
# scratch, there is no cache for Extended at 3 genes.
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

VARIANTS=(e6 e7)

VAR_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
if [ "$REM" -lt 3 ]; then GENES=3; SEED=$REM; else GENES=2; SEED=$(( REM - 3 )); fi
VARIANT=${VARIANTS[$VAR_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> variant=$VARIANT genes=$GENES seed=$SEED"


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
  incremental_experiments/log_extended_fourway.py --genes "$GENES" --seed "$SEED" \
  --variant "$VARIANT" --device cpu &
CHILD=$!
wait "$CHILD"
