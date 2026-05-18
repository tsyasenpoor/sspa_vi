#!/bin/bash
#SBATCH --job-name=drgp_ood_diag
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --exclude=mantis-034
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/ood_diag_%j.out
#SBATCH --error=/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1/logs/ood_diag_%j.err

set -eo pipefail
source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay:${PYTHONPATH:-}

cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
python -u -m src.diagnose_ood_auroc
