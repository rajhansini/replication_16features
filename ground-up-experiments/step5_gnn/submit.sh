#!/bin/bash
#SBATCH --job-name=gnn_step5
#SBATCH --partition=threedle-contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step5_gnn/results/slurm_%j.out
#SBATCH --error=/net/projects/ranalab/rajhansini/replication_16features/ground-up-experiments/step5_gnn/results/slurm_%j.err

cd /net/projects/ranalab/rajhansini/replication_16features
conda activate /net/projects/ranalab/rajhansini/conda_envs/genetic-rl
python ground-up-experiments/step5_gnn/run.py --epochs 300
