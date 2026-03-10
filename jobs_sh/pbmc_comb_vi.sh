#!/bin/bash
# PBMC experiments — Coordinate Ascent VI (full-batch)
# Dataset: scdesign3_PBMC_10kcells_2kgenes
# Labels:  severity + outcome (joint multi-label, kappa=2)
# Aux:     sex comorbidity

for exp in exp0_easy exp1_medium exp2_hard exp3_intersectional; do
    for mode in masked unmasked; do
        cat > pbmc_svi_${mode}_${exp}.sh << EOF
#!/bin/bash
#SBATCH --job-name=svi_${mode}_${exp}
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tsyasenpoor@uconn.edu
#SBATCH -o pbmc_svi_${mode}_${exp}_%j.out
#SBATCH -e pbmc_svi_${mode}_${exp}_%j.err

echo "Job started on: \$(hostname)"
echo "Job started at: \$(date)"
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"

source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu

echo "Python version: \$(python --version)"

python -u /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/quick_reference.py \\
    --method svi \\
    --data /labs/Aguiar/SSPA_BRAY/scdesign3_PBMC_10kcells_2kgenes/${exp}/${exp}.csv.gz \\
    --label-column severity \\
    --aux-columns sex comorbidity \\
    --mode ${mode} \\
    --output-dir ./results/pbmc_svi/${mode}/${exp} \\
    --n-factors 50 \\
    --use-spike-slab true \\
    --alpha-theta 3.214354 \\
    --alpha-beta 9.526157 \\
    --alpha-xi 5.378054 \\
    --alpha-eta 1.276108 \\
    --lambda-xi 9.819736 \\
    --lambda-eta 2.541186 \\
    --sigma-v 2.0 \\
    --sigma-gamma 0.5 \\
    --pi-v 0.6 \\
    --regression-weight 1.0 \\
    --max-iter 600 \\
    --min-iter 120 \\
    --rel-tol 2e-4 \\
    --patience 5 \\
    --theta-damping 0.8 \\
    --beta-damping 0.8 \\
    --v-damping 0.1 \\
    --gamma-damping 0.1 \\
    --xi-damping 0.9 \\
    --eta-damping 0.9 \\
    --adaptive-damping true \\
    --v-warmup 50 \\
    --v-anneal 50 \\
    --elbo-freq 1 \\
    --verbose

echo "Job finished at: \$(date)"
EOF

        sbatch pbmc_svi_${mode}_${exp}.sh
    done
done
