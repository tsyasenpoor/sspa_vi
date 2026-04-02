#!/bin/bash
#SBATCH --job-name=ibd_bayes_opt
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=mantis-034
#SBATCH --mail-user=tsyasenpoor@uconn.edu
#SBATCH -o /labs/Aguiar/SSPA_BRAY/results/ibd_bayes_opt/logs/bayes_opt_%j.out
#SBATCH -e /labs/Aguiar/SSPA_BRAY/results/ibd_bayes_opt/logs/bayes_opt_%j.err

set -eo pipefail
echo "Job started on: $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
echo "Python: $(python --version)"

DATA_DIR="/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed_ibd"
GENE_ANNOT="${DATA_DIR}/gene_annotation.csv"
VI_DIR="/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference"
OUTPUT_DIR="/labs/Aguiar/SSPA_BRAY/results/ibd_bayes_opt_spike"

mkdir -p "$OUTPUT_DIR/logs"

python -u ${VI_DIR}/bayes_opt.py \
    --data "$DATA_DIR" \
    --mode unmasked \
    --label-column disease \
    --aux-columns sex_female age \
    --gene-annotation "$GENE_ANNOT" \
    --dataset-preset emtab \
    --v-prior laplace \
    --n-trials 100 \
    --max-iter 500 \
    --seed 42 \
    --output-dir "$OUTPUT_DIR" \
    --fixed-params use_spike_slab_beta=True regression_weight=1.0 \
    --evaluate-best \
    --verbose

echo "Job finished at: $(date)"
