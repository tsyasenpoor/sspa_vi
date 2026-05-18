#!/bin/bash
# Submit a full sweep: SLURM array index = condition_idx * N_SEEDS + seed_idx
# Usage:
#   N_SEEDS=50 N_CONDITIONS=21 sbatch --array=0-1049%100 slurm/run_sweep.sh configs/A1_factor_recovery.yaml
#
# Tunables:
#   N_SEEDS        seeds per condition (default 50)
#   N_CONDITIONS   conditions in the config (default 21)
#   ARRAY_THROTTLE max simultaneously running tasks (use %N in sbatch --array)
#
# The script reads --array index from $SLURM_ARRAY_TASK_ID and dispatches one
# (condition, seed) experiment unit.

#SBATCH --job-name=drgp_synth
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
N_SEEDS=${N_SEEDS:-50}

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay:${PYTHONPATH:-}

# Redirect XLA/JAX scratch off the node's /tmp (which can fill up under
# concurrent jobs and trigger RESOURCE_EXHAUSTED during cuInit / PTX compile).
JAX_SCRATCH=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/.scratch/jax_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$JAX_SCRATCH"
export TMPDIR="$JAX_SCRATCH"
export XLA_FLAGS="--xla_dump_to=${JAX_SCRATCH}/xla_dump"
trap 'rm -rf "$JAX_SCRATCH"' EXIT

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1

IDX=${SLURM_ARRAY_TASK_ID:-0}
COND=$(( IDX / N_SEEDS ))
SEED=$(( IDX % N_SEEDS ))

CFG_STEM=$(basename "$CONFIG" .yaml)
OUT_DIR=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/results/raw/$CFG_STEM

echo "[$(date)] host=$(hostname) JOB=$SLURM_JOB_ID ARR=$IDX  cond=$COND seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "(no GPU)"

python -u -m src.experiment \
    --config "$CONFIG" \
    --condition-idx "$COND" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR"

echo "[$(date)] done IDX=$IDX"
