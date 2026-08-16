#!/bin/bash
#SBATCH --job-name=3gene_gt_ext_fix
#SBATCH --output=fresh_dataset/3gene/slurm_extended_fix_%A_task%a.log
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --requeue

# 3-gene ground truth rebuild: Extended family, ONLY the 3 regimes whose
# GeneA/GeneB allele frequencies were wrong (HighHigh/MixedA/MixedB -- see
# ALLELE_FREQS fix in step9_gnn_3gene/build_datasets.py). LowHigh/MediumEven/
# LowLow are unaffected and NOT rebuilt here.
cd /net/projects/ranalab/rajhansini/replication_16features

REGIMES=(HighHigh MixedA MixedB)
PRESETS=(Base Aggressive)

REGIME=${REGIMES[$(( SLURM_ARRAY_TASK_ID / 2 ))]}
PRESET=${PRESETS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}

echo "task=$SLURM_ARRAY_TASK_ID -> regime=$REGIME preset=$PRESET"

/net/projects/ranalab/rajhansini/conda_envs/genetic-rl/bin/python \
  incremental_experiments/build_3gene_ground_truth.py --family Extended --regime "$REGIME" --preset "$PRESET"
