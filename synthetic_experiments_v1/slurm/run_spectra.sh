#!/bin/bash
# Spectra baseline runner: one (cond, seed) per array task.
# Spectra has its own conda env because of torch/numpy pins incompatible
# with our `jax_gpu` env. Output schema mirrors src/experiment.py but only
# contains the `spectra_lr` method; the C1 analyzer is patched to merge
# this directory with the main C1 raw dir.
#
# Usage:
#   sbatch --array=0-49%25 slurm/run_spectra.sh configs/C1_method_ranking.yaml

#SBATCH --job-name=spectra_c1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --exclude=mantis-034
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.out
#SBATCH --error=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.err

set -eo pipefail

CONFIG=${1:?missing config path}

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate spectra

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1:${PYTHONPATH:-}

# C1 is single-condition (cond_idx=0) over 50 seeds.
# For multi-condition configs, override SEED_OFFSET / N_SEEDS upstream.
N_SEEDS=${N_SEEDS:-50}
SEED_OFFSET=${SEED_OFFSET:-0}
COND_IDX=${COND_IDX:-0}

IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((SEED_OFFSET + IDX))

CFG_STEM=$(basename "$CONFIG" .yaml)
OUT_DIR=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/results/raw/${CFG_STEM}_spectra

echo "[$(date)] SPECTRA host=$(hostname) JOB=$SLURM_JOB_ID ARR=$IDX cond=$COND_IDX seed=$SEED"

python -u -m src.experiment_spectra \
    --config "$CONFIG" \
    --condition-idx "$COND_IDX" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --num-epochs 1000

echo "[$(date)] done IDX=$IDX seed=$SEED"
