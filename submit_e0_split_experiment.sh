#!/bin/bash
#SBATCH --job-name=e0_split
#SBATCH --output=incremental_experiments/results/slurm_e0split_%A_task%a.log
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=peanut-cpu
#SBATCH --array=0-11
#SBATCH --requeue

# E0 train/test split root-cause experiments (rotate + add), GNN + MLP, 3
# seeds each -- see incremental_experiments/e0_split_experiment.py for the
# rationale. 12 tasks = 2 experiments x 2 kinds x 3 seeds.
cd /net/projects/ranalab/rajhansini/replication_16features

EXPS=(rotate add)
KINDS=(gnn mlp)

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
KIND_IDX=$(( REM / 3 ))
SEED=$(( REM % 3 ))

EXP=${EXPS[$EXP_IDX]}
KIND=${KINDS[$KIND_IDX]}

echo "task=$SLURM_ARRAY_TASK_ID -> exp=$EXP kind=$KIND seed=$SEED"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/e0_split_experiment.py --exp "$EXP" --kind "$KIND" \
  --device cpu --epochs 500 --mode both --seed "$SEED"
