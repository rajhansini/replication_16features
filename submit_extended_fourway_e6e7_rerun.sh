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

cd /net/projects/ranalab/rajhansini/replication_16features

VARIANTS=(e6 e7)

VAR_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
if [ "$REM" -lt 3 ]; then GENES=3; SEED=$REM; else GENES=2; SEED=$(( REM - 3 )); fi
VARIANT=${VARIANTS[$VAR_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> variant=$VARIANT genes=$GENES seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/log_extended_fourway.py --genes "$GENES" --seed "$SEED" \
  --variant "$VARIANT" --device cpu
