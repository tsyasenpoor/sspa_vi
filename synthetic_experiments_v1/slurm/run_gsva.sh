#!/bin/bash
# GSVA baseline runner: one (cond, seed) per array task.
# Uses the `gsva_env` conda env (gseapy + sspa + pyyaml).
# Output schema mirrors src/experiment.py but only contains gsva_lr; the
# C1 analyzer merges this dir via --extra-raw-dirs.
#
# Usage:
#   sbatch --array=0-49%25 slurm/run_gsva.sh configs/C1_method_ranking.yaml

#SBATCH --job-name=gsva_c1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --exclude=mantis-034,mantis-064
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.out
#SBATCH --error=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.err

set -eo pipefail

CONFIG=${1:?missing config path}

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate gsva_env

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1:${PYTHONPATH:-}

N_SEEDS=${N_SEEDS:-50}
SEED_OFFSET=${SEED_OFFSET:-0}
COND_IDX=${COND_IDX:-0}

IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((SEED_OFFSET + IDX))

CFG_STEM=$(basename "$CONFIG" .yaml)
OUT_DIR=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/results/raw/${CFG_STEM}_gsva

echo "[$(date)] GSVA host=$(hostname) JOB=$SLURM_JOB_ID ARR=$IDX cond=$COND_IDX seed=$SEED"

python -u -m src.experiment_gsva \
    --config "$CONFIG" \
    --condition-idx "$COND_IDX" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --threads 4

echo "[$(date)] done IDX=$IDX seed=$SEED"
