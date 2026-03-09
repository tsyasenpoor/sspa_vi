#!/bin/bash
# PBMC experiments — Coordinate Ascent VI (full-batch)
# Dataset: scdesign3_PBMC_10kcells_2kgenes
# Labels:  severity + outcome (joint multi-label, kappa=2)
# Aux:     sex comorbidity
for exp in exp0_easy exp1_medium exp2_hard exp3_intersectional; do
    for mode in masked unmasked; do
        cat > pbmc_vi_${mode}_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=vi_${mode}_${exp}
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tsyasenpoor@uconn.edu
#SBATCH -o pbmc_vi_${mode}_${exp}_%j.out
#SBATCH -e pbmc_vi_${mode}_${exp}_%j.err
echo "Job started on: \$(hostname)"
echo "Job started at: \$(date)"
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
echo "Python version: \$(python --version)"
python -u /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/quick_reference.py \\
    --method vi \\
    --data /labs/Aguiar/SSPA_BRAY/scdesign3_PBMC_10kcells_2kgenes/${exp}/${exp}.csv.gz \\
    --label-column severity outcome \\
    --aux-columns sex comorbidity \\
    --mode ${mode} \\
    --output-dir ./results/pbmc_vi/${mode}/${exp} \\
    --n-factors 50 \\
    --a 0.3 \\
    --c 0.3 \\
    --sigma-v 2.0 \\
    --sigma-gamma 0.5 \\
    --regression-weight 1.0 \\
    --max-iter 600 \\
    --tol 0.001 \\
    --v-warmup 50 \\
    --check-freq 5 \\
    --verbose
echo "Job finished at: \$(date)"
EOF
        sbatch pbmc_vi_${mode}_${exp}.sh
    done
done
