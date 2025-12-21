#!/bin/bash
#SBATCH --job-name=optuna_test
#SBATCH --cpus-per-task=4 
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mail-type=END,FAIL 
#SBATCH --mem=256G   
#SBATCH --mail-user=tsyasenpoor@uconn.edu
#SBATCH -o optuna_test_%j.out
#SBATCH -e optuna_test_%j.err

# Remove the exec lines that were causing duplication
# exec > >(tee -a opt_vi_${SLURM_JOB_ID}.out)
# exec 2> >(tee -a opt_vi_${SLURM_JOB_ID}.err >&2)

echo "Job started on: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate bray_cpu

echo "Python version: $(python --version)"

python -u /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/hyperparameter_optimization.py \
    --n_trials 10 \
    --max_iter 100 \
    --n_jobs 1 \
    --output_dir ./optuna_test

echo "Job finished at: $(date)"
echo "Job exit status: $?"