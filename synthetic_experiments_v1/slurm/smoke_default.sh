#!/bin/bash
#SBATCH --job-name=drgp_synth_smoke
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --exclude=mantis-034
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/smoke_default_%j.out
#SBATCH --error=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/smoke_default_%j.err

set -eo pipefail
source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay:${PYTHONPATH:-}

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1

echo "=== Smoke (default scale n=500, p=5000, K_true=K_fit=10) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "(no GPU detected)"

python -u -m src.smoke_test \
    --scale default \
    --seed 0 \
    --drgp-max-iter 600

echo "=== smoke_default.sh done ==="
