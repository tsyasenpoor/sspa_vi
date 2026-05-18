#!/bin/bash
# Rescue script: re-run specific (cond, seed) pairs listed in a text file.
# Usage:
#   sbatch --array=0-28%30 slurm/run_rescue.sh configs/D1_nb_likelihood.yaml slurm/rescue_d1.txt
#   sbatch --array=0-52%50 slurm/run_rescue.sh configs/D3_mask_corruption.yaml slurm/rescue_d3.txt

#SBATCH --job-name=drgp_rescue
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --exclude=mantis-034
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.out
#SBATCH --error=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/%x_%A_%a.err

set -eo pipefail

CONFIG=${1:?missing config path}
LIST=${2:?missing list file with "cond seed" lines}

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay:${PYTHONPATH:-}

JAX_SCRATCH=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/.scratch/jax_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$JAX_SCRATCH"
export TMPDIR="$JAX_SCRATCH"
export XLA_FLAGS="--xla_dump_to=${JAX_SCRATCH}/xla_dump"
trap 'rm -rf "$JAX_SCRATCH"' EXIT

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1

IDX=${SLURM_ARRAY_TASK_ID:-0}
LINE=$(sed -n "$((IDX+1))p" "$LIST")
COND=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')

if [[ -z "$COND" || -z "$SEED" ]]; then
    echo "No entry at line $((IDX+1)) of $LIST -- exiting"
    exit 0
fi

CFG_STEM=$(basename "$CONFIG" .yaml)
OUT_DIR=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/results/raw/$CFG_STEM

echo "[$(date)] RESCUE host=$(hostname) JOB=$SLURM_JOB_ID ARR=$IDX  cond=$COND seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "(no GPU)"

python -u -m src.experiment \
    --config "$CONFIG" \
    --condition-idx "$COND" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"

echo "[$(date)] done rescue IDX=$IDX cond=$COND seed=$SEED"
