#!/bin/bash
#SBATCH --job-name=ovfit_probe
#SBATCH --output=incremental_experiments/results/slurm_ovfitprobe_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-17
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300

# Three overfitting probes on 2-gene, all on the FAIR (rotate) family split:
#   curve         -- train vs held-out validation loss every epoch
#   randlabel     -- shuffled-target control (can the model memorize at all?)
#   configholdout -- unseen regimes of SEEN families vs a fully unseen family
# See incremental_experiments/overfit_probes_2gene.py for the rationale.
# 18 tasks = 3 probes x 2 kinds x 3 seeds.
#
# RESUME: safe to resubmit as-is -- finished (probe,kind,seed) exit in <1s via
# the results.json skip guard, unfinished ones restart from checkpoint.pt
# (written every epoch) and partial eval files. USR1 300s before the wall
# clock requeues the task, which is required because every partition here has
# a hard 4h MaxTime.
cd /net/projects/ranalab/rajhansini/replication_16features

PROBES=(curve randlabel configholdout)
KINDS=(gnn mlp)

PROBE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
KIND_IDX=$(( REM / 3 ))
SEED=$(( REM % 3 ))

PROBE=${PROBES[$PROBE_IDX]}
KIND=${KINDS[$KIND_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> probe=$PROBE kind=$KIND seed=$SEED"

requeue_handler() {
    echo "[SIGNAL] $(date -Is) USR1 (300s to wall clock) -- requeueing task $SLURM_ARRAY_TASK_ID; resumes from checkpoint"
    scontrol requeue "${SLURM_JOB_ID}"
}
trap requeue_handler USR1

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/overfit_probes_2gene.py --probe "$PROBE" --kind "$KIND" \
  --device cpu --epochs 500 --seed "$SEED" &
CHILD=$!
wait "$CHILD"
