#!/bin/bash
#SBATCH --job-name=e6_split
#SBATCH --output=incremental_experiments/results/slurm_e6split_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-29
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# E6 (sum-pooling, best 3-gene rung) train/test split root-cause experiments
# -- rotate, add, and full leave-one-family-out (loo_trio/loo_nuclear/
# loo_threegen) -- GNN + MLP, 3 seeds each. See
# incremental_experiments/e6_split_experiment.py for the rationale.
# 30 tasks = 5 experiments x 2 kinds x 3 seeds.
#
# RESUME: this array is safe to resubmit as-is at any time.
#   * tasks that already wrote results_e6.json exit in <1s (skip guard in
#     e6_split_experiment.py's main)
#   * tasks killed mid-training restart from checkpoint.pt (written every epoch)
#   * tasks killed mid-TRAIN-eval restart from eval_train_partial.json
#   * 300s before the wall clock SLURM sends USR1 to this script, which
#     requeues the task so it picks itself back up instead of dying.
#
# The requeue is REQUIRED, not a safety net: every partition on this cluster
# (general, peanut-cpu, threedle-*) has a hard 4h MaxTime, and the
# leave-one-out GNN folds -- which train on Extended plus two other families --
# do not fit in 4h. They only finish by checkpointing and requeueing across
# several 4h slots. That is what silently killed them on the first attempt.
# Use --force on the python call to redo a finished (exp,kind,seed) from scratch.
cd /net/projects/ranalab/rajhansini/replication_16features

EXPS=(rotate add loo_trio loo_nuclear loo_threegen)
KINDS=(gnn mlp)

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
KIND_IDX=$(( REM / 3 ))
SEED=$(( REM % 3 ))

EXP=${EXPS[$EXP_IDX]}
KIND=${KINDS[$KIND_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> exp=$EXP kind=$KIND seed=$SEED"

# Requeue on the pre-wall-clock signal instead of being killed outright.
# The python child keeps running until SLURM's own SIGTERM arrives, so the
# last per-epoch checkpoint write is as late as possible.
requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 received (300s to wall clock) -- requeueing task $SLURM_ARRAY_TASK_ID; it will resume from its checkpoint"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/e6_split_experiment.py --exp "$EXP" --kind "$KIND" \
  --device cpu --epochs 500 --mode both --seed "$SEED" &
CHILD=$!
wait "$CHILD"
