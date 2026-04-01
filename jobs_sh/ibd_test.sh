#!/bin/bash
# IBD test — Coordinate Ascent VI
# Dataset: EMTAB11349 (590 samples × 14183 genes)
# Labels:  "disease" = 1 if Crohn's disease OR ulcerative colitis
# Aux:     age, sex_female
#SBATCH --job-name=ibd_test
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --exclude=mantis-034
#SBATCH --mail-user=tsyasenpoor@uconn.edu
#SBATCH -o ibd_test_%j.out
#SBATCH -e ibd_test_%j.err

echo "Job started on: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
echo "Python version: $(python --version)"

# ---------- Preprocessing ----------
# 1. The EMTAB loader expects gene_expression_raw_processed.csv.gz
#    but the file is gene_expression.csv.gz → symlink it.
# 2. Create a combined binary "disease" label from Crohn's/UC columns.
DATA_SRC="/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed"
DATA_DIR="/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed_ibd"

python -u -c "
import os, shutil, pandas as pd, numpy as np
from pathlib import Path

src = Path('${DATA_SRC}')
dst = Path('${DATA_DIR}')
dst.mkdir(parents=True, exist_ok=True)

# Symlink gene expression with expected name
ge_link = dst / 'gene_expression_raw_processed.csv.gz'
if not ge_link.exists():
    os.symlink(src / 'gene_expression.csv.gz', ge_link)

# Copy aux_data as-is
aux_link = dst / 'aux_data.csv.gz'
if not aux_link.exists():
    os.symlink(src / 'aux_data.csv.gz', aux_link)

# Build combined disease label
resp = pd.read_csv(src / 'responses.csv.gz', compression='gzip')
resp['disease'] = ((resp[\"Crohn's disease\"] == 1) | (resp['ulcerative colitis'] == 1)).astype(int)
resp.to_csv(dst / 'responses.csv.gz', index=False, compression='gzip')

print(f'Preprocessed directory: {dst}')
print(f'disease label distribution:\n{resp[\"disease\"].value_counts()}')

# Build gene annotation compatible with data_loader (Genename, GeneID, Genetype)
annot_src = '/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_biomaRt_humanGRCh38_genebiotypes_Jan102024.csv'
annot = pd.read_csv(annot_src)
annot = annot.drop_duplicates('ensembl_gene_id')
annot_out = annot[['ensembl_gene_id', 'hgnc_symbol', 'gene_biotype']].copy()
annot_out.columns = ['GeneID', 'Genename', 'Genetype']
annot_out.to_csv(dst / 'gene_annotation.csv', index=False)
print(f'Gene annotation: {len(annot_out)} genes ({(annot_out[\"Genetype\"]==\"protein_coding\").sum()} protein-coding)')
"

# ---------- Run VI ----------
python -u /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/quick_reference.py \
    --data "${DATA_DIR}" \
    --gene-annotation "${DATA_DIR}/gene_annotation.csv" \
    --label-column disease \
    --aux-columns age sex_female \
    --mode unmasked \
    --output-dir ./results/ibd_vi/unmaskedlaplacespike\
    --n-factors 100 \
    --a 0.3 \
    --c 0.3 \
    --sigma-v 2.0 \
    --sigma-gamma 0.5 \
    --regression-weight 1.0 \
    --max-iter 10000 \
    --tol 0.001 \
    --v-warmup 50 \
    --v-prior laplace \
    --check-freq 5 \
    --early-stopping heldout_ll \
    --spike-slab-beta   \
    --verbose

echo "Job finished at: $(date)"
