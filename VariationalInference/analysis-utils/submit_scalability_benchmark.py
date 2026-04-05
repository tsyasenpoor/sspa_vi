#!/usr/bin/env python
"""
Submit scalability benchmark SLURM jobs.
=========================================

All methods subsample on-the-fly from the full h5ad — no Phase 1 needed.
Each job loads the full dataset, subsamples to the requested number of
patients in memory, and runs the method directly.

Patient counts:
    - DRGP + baselines: 15, 30, 50, 148 (full)
    - Spectra + scHPF:  15, 30, 50 only

Usage:
    python submit_scalability_benchmark.py [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Patient counts for subsampling (148 = full dataset, no subsampling)
PATIENT_COUNTS_SUBSAMPLE = [15, 30, 50]       # subsampled sizes
PATIENT_COUNTS_FULL = PATIENT_COUNTS_SUBSAMPLE + [148]  # includes full

SEEDS = [42, 123, 456, 789, 1024]

H5AD = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/covid19_filtered_fullgenes_clean.h5ad"
GMT_FILE = "/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt"
RESULTS_ROOT = "/labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_patient_level"
METHODS_ROOT = f"{RESULTS_ROOT}/methods"

VI_DIR = "/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference"
CONDA_SH = "/home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh"
MAIL_USER = "tsyasenpoor@uconn.edu"
SUBSAMPLE_SEED = 0  # deterministic patient-level subsampling seed

# Common DRGP parameters
DRGP_COMMON = (
    '--n-factors 50 '
    '--a 0.3 --c 0.3 '
    '--sigma-v 2.0 --v-prior laplace '
    '--sigma-gamma 0.5 '
    '--max-iter 2000 --tol 0.001 '
    '--v-warmup 50 --check-freq 10 '
    '--early-stopping heldout_ll '
    '--label-column "CoVID-19 severity" Outcome '
    '--aux-columns Sex cm_asthma_copd cm_cardio cm_diabetes '
    '--patient-column sampleID '
    f'--pathway-file {GMT_FILE}'
)

# Resource profiles keyed by n_patients: (mem_gb, time_hours, partition, qos, cpus, gpu)
# Updated based on scalability_benchmark log analysis (2026-03-30):
#   - baselines 15/30/50p: all OOM during SVM → doubled memory
#   - baselines 148p: 2/5 cancelled still running at 11h → bumped to 48h
#   - scHPF 15p: 3/5 timed out at 12h (2/5 finished in ~11h) → bumped to 24h
#   - scHPF 30p: all OOM at 250GB → bumped to 400GB on himem
#   - drgp 30p: pathway_init timed out at 670 iters/12h (~64s/iter) → 48h for full 2000 iters
#   - drgp 50p: pathway_init only at iter 270 → estimated ~72h for 2000 iters → 96h
#   - drgp 148p: masked ~654s/check-freq=10 (~65s/iter) → ~36h for 2000 iters → 48h w/ buffer
#   - drgp_combined 30p: still running at 12h → bumped to 48h (same as drgp 30p)
#   - spectra_sup 50p: ~108s/iter, needs ~300h at 10k epochs → bumped to 96h
RESOURCE_PROFILES = {
    15: {
        "drgp":       (64,  12, "general", "general", 4, 1),
        "spectra":    (160, 12, "general", "general", 4, 1),
        "schpf":      (160, 24, "general", "general", 8, 0),
        "baselines":  (128, 4,  "general", "general", 8, 0),
    },
    30: {
        "drgp":       (100, 48, "general", "general", 4, 1),
        "spectra":    (250, 24, "general", "general", 4, 1),
        "schpf":      (400, 48, "himem",   "himem",   8, 0),
        "baselines":  (200, 12, "general", "general", 8, 0),
    },
    50: {
        "drgp":       (200, 96, "general", "general", 4, 1),
        "spectra":    (500, 96, "himem",   "himem",   8, 1),
        "schpf":      (600, 48, "himem",   "himem",   8, 0),
        "baselines":  (300, 16, "himem",   "himem",   8, 0),
    },
    148: {
        "drgp":       (300, 48, "general", "general", 8, 1),
        "baselines":  (300, 48, "general", "general", 8, 0),
    },
}

# Which methods run on which patient counts
METHODS_SUBSAMPLE_ONLY = {"spectra_sup", "schpf"}  # 15, 30, 50 only
METHODS_ALL = {"drgp_unmasked", "drgp_masked", "drgp_pathway_init",
               "drgp_combined", "baselines"}         # 15, 30, 50, 148


def _subsample_arg(n_patients: int) -> str:
    """Return the CLI flag for on-the-fly subsampling, or empty string for full."""
    if n_patients >= 148:
        return ""
    return f"--subsample-n-patients {n_patients} --subsample-seed {SUBSAMPLE_SEED}"


def _seeds_bash_array() -> str:
    return "(" + " ".join(str(s) for s in SEEDS) + ")"


def _slurm_header(
    job_name: str, mem_gb: int, time_hours: int, partition: str, qos: str,
    cpus: int, gpu: int, array_size: int = 5
) -> str:
    time_str = f"{time_hours}:00:00"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{array_size - 1}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_str}",
        "#SBATCH --mail-type=END,FAIL",
        "#SBATCH --exclude=mantis-034",
        f"#SBATCH --mail-user={MAIL_USER}",
        f"#SBATCH -o {RESULTS_ROOT}/logs/{job_name}_%A_%a.out",
        f"#SBATCH -e {RESULTS_ROOT}/logs/{job_name}_%A_%a.err",
    ]
    if gpu > 0:
        lines.insert(7, f"#SBATCH --gres=gpu:{gpu}")
    return "\n".join(lines)


def _common_preamble(conda_env: str = "jax_gpu") -> str:
    return textwrap.dedent(f"""\
        set -eo pipefail
        echo "Job started on: $(hostname) at $(date)"
        echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        START_TIME=$(date +%s)

        SEEDS={_seeds_bash_array()}
        SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}
        echo "Using seed: $SEED"

        source {CONDA_SH}
        conda activate {conda_env}
        echo "Python: $(python --version)"
    """)


def _benchmark_footer(output_dir_var: str = "$OUTPUT") -> str:
    """Save wall time, peak memory, and SLURM resource usage to benchmark.json."""
    return textwrap.dedent(f"""\
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "Job finished at: $(date)"
        echo "Total elapsed: ${{ELAPSED}}s"

        # Peak RSS: check cgroup (SLURM), then fall back to /proc for this
        # process tree.  SLURM cgroup tracks ALL processes in the job step,
        # so it captures the Python child correctly.
        PEAK_RSS_MB=0
        # Try cgroup v1 (common on HPC)
        if [[ -f /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${{SLURM_JOB_ID}}/memory.max_usage_in_bytes ]]; then
            PEAK_RSS_BYTES=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${{SLURM_JOB_ID}}/memory.max_usage_in_bytes 2>/dev/null || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_BYTES / 1048576))
        # Try cgroup v2
        elif [[ -f /sys/fs/cgroup/memory.peak ]]; then
            PEAK_RSS_BYTES=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_BYTES / 1048576))
        # Try SLURM sstat (works for running job steps)
        elif command -v sstat &>/dev/null; then
            PEAK_RSS_MB=$(sstat -j ${{SLURM_JOB_ID}}.batch --format=MaxRSS -n 2>/dev/null | tr -d ' K' | head -1 || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_MB / 1024))  # KB -> MB
        fi
        # Fallback: shell process /proc (underestimates child memory)
        if [[ "$PEAK_RSS_MB" -eq 0 ]]; then
            PEAK_RSS_KB=$(grep "VmHWM" /proc/$$/status 2>/dev/null | awk '{{print $2}}' || echo "0")
            PEAK_RSS_MB=$((PEAK_RSS_KB / 1024))
        fi
        echo "Peak memory: ${{PEAK_RSS_MB}} MB"

        # Save benchmark metrics
        python3 -c "
import json, os
bm = {{
    'wall_time_seconds': $ELAPSED,
    'peak_rss_mb': $PEAK_RSS_MB,
    'slurm_job_id': '$SLURM_JOB_ID',
    'slurm_array_task_id': '$SLURM_ARRAY_TASK_ID',
    'hostname': '$(hostname)',
    'seed': $SEED,
}}
os.makedirs('{output_dir_var}'.replace('\\$SEED', str($SEED)).replace('\\${{SEED}}', str($SEED)), exist_ok=True)
path = os.path.join('{output_dir_var}'.replace('\\$SEED', str($SEED)).replace('\\${{SEED}}', str($SEED)), 'benchmark.json')
with open(path, 'w') as f:
    json.dump(bm, f, indent=2)
print(f'Benchmark saved to {{path}}')
" 2>/dev/null || echo "WARNING: Could not save benchmark.json"
    """)


# ============================================================================
# Job script generators
# ============================================================================

def gen_drgp_script(n_patients: int, mode: str, res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = f"scale_drgp_{mode}_{n_patients}p"
    out_tmpl = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/drgp_{mode}"
    subsample = _subsample_arg(n_patients)

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    cmd = textwrap.dedent(f"""\
        OUTPUT="{out_tmpl}"
        mkdir -p "$OUTPUT"

        python -u {VI_DIR}/quick_reference.py \\
            --data {H5AD} \\
            --mode {mode} \\
            {DRGP_COMMON} \\
            {subsample} \\
            --seed $SEED \\
            --output-dir "$OUTPUT" \\
            --verbose
    """)

    footer = _benchmark_footer(out_tmpl)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_spectra_sup_script(n_patients: int, res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = f"scale_spectra_sup_{n_patients}p"
    subsample = _subsample_arg(n_patients)

    spectra_out = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/spectra_sup"
    baseline_out = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/spectra_sup_baselines"

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    # Spectra saves cell_scores.npy; downstream baselines use those scores
    # plus metadata generated on-the-fly inside run_spectra_supervised.py.
    # The metadata CSV for run_spectra_baselines.py is written by a small
    # inline Python step after Spectra finishes (avoids needing pre-saved files).
    cmd = textwrap.dedent(f"""\
        SPECTRA_OUT="{spectra_out}"
        BASELINE_OUT="{baseline_out}"
        mkdir -p "$SPECTRA_OUT" "$BASELINE_OUT"

        # Fit Spectra (supervised with REACTOME pathways)
        FIT_START=$(date +%s)
        python -u {VI_DIR}/comp/run_spectra_supervised.py \\
            --h5ad {H5AD} \\
            --gmt-file {GMT_FILE} \\
            --output-dir "$SPECTRA_OUT" \\
            --require-prefix REACTOME \\
            --lam 0.01 \\
            --num-epochs 10000 \\
            {subsample} \\
            --seed $SEED \\
            --verbose
        FIT_END=$(date +%s)
        FIT_ELAPSED=$((FIT_END - FIT_START))
        echo "Spectra sup fit completed in ${{FIT_ELAPSED}}s"

        # Generate metadata CSV on-the-fly for downstream classifiers
        python3 -c "
import sys; sys.path.insert(0, '{VI_DIR}/..')
import anndata as ad, numpy as np
from VariationalInference.create_subsamples import subsample_adata, _build_metadata
src = ad.read_h5ad('{H5AD}')
src.var_names_make_unique()
{'src = subsample_adata(src, n_patients=' + str(n_patients) + ', subsample_seed=' + str(SUBSAMPLE_SEED) + ')' if n_patients < 148 else '# full dataset'}
meta = _build_metadata(src.obs)
meta.to_csv('$SPECTRA_OUT/metadata_covid.csv')
print(f'Wrote metadata: {{len(meta)}} cells')
"

        # Downstream classifiers
        python -u {VI_DIR}/comp/run_spectra_baselines.py \\
            --cell-scores "$SPECTRA_OUT/spectra_cell_scores.npy" \\
            --data "$SPECTRA_OUT/metadata_covid.csv" \\
            --labels severity outcome \\
            --output-dir "$BASELINE_OUT" \\
            --seed $SEED
    """)

    footer = _benchmark_footer(spectra_out)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_schpf_script(n_patients: int, res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = f"scale_schpf_{n_patients}p"

    schpf_out = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/schpf"
    baseline_out = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/schpf_baselines"
    model_file = f"${{SCHPF_OUT}}/covid_sub.scHPF_K50_b0_5trials.joblib"

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)

    preamble = textwrap.dedent(f"""\
        set -eo pipefail
        echo "Job started on: $(hostname) at $(date)"
        echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        START_TIME=$(date +%s)

        SEEDS={_seeds_bash_array()}
        SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}
        echo "Using seed: $SEED"

        source {CONDA_SH}
    """)

    # scHPF CLI needs mtx files on disk. Write them to /tmp and clean up after.
    cmd = textwrap.dedent(f"""\
        SCHPF_OUT="{schpf_out}"
        BASELINE_OUT="{baseline_out}"
        mkdir -p "$SCHPF_OUT" "$BASELINE_OUT"

        # Prepare temp mtx + metadata from subsampled data (written to /tmp)
        TMPDIR_SCHPF=$(mktemp -d /tmp/schpf_${{SLURM_JOB_ID}}_${{SEED}}_XXXXXX)
        echo "Temp dir for scHPF inputs: $TMPDIR_SCHPF"

        conda activate jax_gpu
        python3 -c "
import sys; sys.path.insert(0, '{VI_DIR}/..')
import anndata as ad, numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from VariationalInference.create_subsamples import subsample_adata, _build_metadata
from pathlib import Path

src = ad.read_h5ad('{H5AD}')
src.var_names_make_unique()
{'src = subsample_adata(src, n_patients=' + str(n_patients) + ', subsample_seed=' + str(SUBSAMPLE_SEED) + ')' if n_patients < 148 else '# full dataset'}

tmpdir = Path('$TMPDIR_SCHPF')

# Write mtx
X = src.layers['raw'] if 'raw' in src.layers else src.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)
else:
    X = X.tocsr()
X_int = X.copy()
X_int.data = np.rint(np.clip(X_int.data, 0, None)).astype(np.int32, copy=False)
mmwrite(tmpdir / 'filtered.mtx', X_int.tocoo(), field='integer')

# Write genes
genes = pd.DataFrame({{'gene_id': src.var_names.astype(str), 'gene_name': src.var_names.astype(str)}})
genes.to_csv(tmpdir / 'genes.txt', sep='\\t', header=False, index=False)

# Write metadata
meta = _build_metadata(src.obs)
meta.to_csv(tmpdir / 'metadata_covid.csv')
print(f'Wrote scHPF inputs to {{tmpdir}}: {{src.n_obs}} cells x {{src.n_vars}} genes')
"

        # Step 1: Train scHPF (requires schpf_p37 env)
        conda activate schpf_p37
        echo "Python (scHPF): $(python --version)"

        FIT_START=$(date +%s)
        scHPF train \\
            -i "$TMPDIR_SCHPF/filtered.mtx" \\
            -o "$SCHPF_OUT" \\
            -p covid_sub \\
            -k 50 \\
            -t 5 \\
            -M 1000 \\
            -m 30 \\
            -e 0.001 \\
            -f 10 \\
            --better-than-n-ago 5
        FIT_END=$(date +%s)
        FIT_ELAPSED=$((FIT_END - FIT_START))
        echo "scHPF train completed in ${{FIT_ELAPSED}}s"

        # Step 2: Downstream classifiers (requires jax_gpu env)
        conda activate jax_gpu
        echo "Python (baselines): $(python --version)"

        python -u {VI_DIR}/comp/run_schpf_baselines.py \\
            --model {model_file} \\
            --data "$TMPDIR_SCHPF/metadata_covid.csv" \\
            --labels severity outcome \\
            --output-dir "$BASELINE_OUT" \\
            --seed $SEED \\
            --verbose

        # Clean up temp files
        rm -rf "$TMPDIR_SCHPF"
        echo "Cleaned up temp dir"
    """)

    footer = _benchmark_footer(schpf_out)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_baselines_script(n_patients: int, res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = f"scale_baselines_{n_patients}p"
    out_root = f"{METHODS_ROOT}/{n_patients}p/seed_${{SEED}}/baselines"
    subsample = _subsample_arg(n_patients)

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    cmd = textwrap.dedent(f"""\
        OUT_ROOT="{out_root}"

        for LABEL in "CoVID-19 severity" "Outcome"; do
            if [[ "$LABEL" == "CoVID-19 severity" ]]; then
                LABEL_TAG="severity"
            else
                LABEL_TAG="outcome"
            fi

            RUN_OUT="$OUT_ROOT/$LABEL_TAG"
            mkdir -p "$RUN_OUT"

            echo "Running baselines for $LABEL_TAG (seed=$SEED)..."

            python -u {VI_DIR}/comp/run_baselines.py \\
                --data {H5AD} \\
                --label-column "$LABEL" \\
                --aux-columns Sex cm_asthma_copd cm_cardio cm_diabetes \\
                --patient-column sampleID \\
                {subsample} \\
                --output-dir "$RUN_OUT" \\
                --latent-dim 50 \\
                --seed $SEED \\
                --verbose

            echo "$LABEL_TAG finished at: $(date)"

            # Patient-level evaluation from saved pickles (no retraining)
            PAT_OUT="$OUT_ROOT/${{LABEL_TAG}}_patient"
            mkdir -p "$PAT_OUT"
            echo "Running patient-level eval for $LABEL_TAG (seed=$SEED)..."

            python -u {VI_DIR}/comp/run_baselines_patient_eval.py \
                --data {H5AD} \
                --model-dir "$RUN_OUT" \
                --label-column "$LABEL" \
                --aux-columns Sex cm_asthma_copd cm_cardio cm_diabetes \
                --patient-column sampleID \
                {subsample} \
                --output-dir "$PAT_OUT" \
                --latent-dim 50 \
                --seed $SEED \
                --verbose

            echo "$LABEL_TAG patient-level eval finished at: $(date)"
        done
    """)

    footer = _benchmark_footer(out_root)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


# ============================================================================
# Main submission logic
# ============================================================================

def submit_script(script_path: str, dry_run: bool = False) -> str | None:
    cmd = ["sbatch", script_path]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR submitting {script_path}: {result.stderr}")
        return None

    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted: {script_path} -> job {job_id}")
    return job_id


def parse_args():
    p = argparse.ArgumentParser(description="Submit scalability benchmark SLURM jobs")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate scripts but don't submit")
    p.add_argument("--n-patients", type=int, nargs="+", default=None,
                   help="Patient counts to run (default: 15 30 50 for spectra/schpf, "
                        "15 30 50 148 for drgp/baselines)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["drgp_unmasked", "drgp_masked", "drgp_pathway_init",
                            "drgp_combined", "spectra_sup",
                            "schpf", "baselines"],
                   help="Methods to run")
    return p.parse_args()


def main():
    args = parse_args()

    jobs_dir = Path(RESULTS_ROOT) / "jobs_generated"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(RESULTS_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCALABILITY BENCHMARK - JOB SUBMISSION (on-the-fly subsampling)")
    print("=" * 70)
    print(f"  Patient counts (subsample): {PATIENT_COUNTS_SUBSAMPLE}")
    print(f"  Patient counts (full):      {PATIENT_COUNTS_FULL}")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Methods: {args.methods}")
    print(f"  Results: {RESULTS_ROOT}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  NOTE: No Phase 1 needed — subsampling happens on-the-fly")

    method_generators = {
        "drgp_unmasked":     lambda np_, res: gen_drgp_script(np_, "unmasked", res),
        "drgp_masked":       lambda np_, res: gen_drgp_script(np_, "masked", res),
        "drgp_pathway_init": lambda np_, res: gen_drgp_script(np_, "pathway_init", res),
        "drgp_combined":     lambda np_, res: gen_drgp_script(np_, "combined", res),
        "spectra_sup":       gen_spectra_sup_script,
        "schpf":             gen_schpf_script,
        "baselines":         gen_baselines_script,
    }

    method_to_resource_class = {
        "drgp_unmasked": "drgp",
        "drgp_masked": "drgp",
        "drgp_pathway_init": "drgp",
        "drgp_combined": "drgp",
        "spectra_sup": "spectra",
        "schpf": "schpf",
        "baselines": "baselines",
    }

    total_jobs = 0
    for method in args.methods:
        if method not in method_generators:
            print(f"  WARNING: Unknown method '{method}', skipping")
            continue

        # Determine which patient counts this method runs on
        if args.n_patients:
            patient_counts = args.n_patients
        elif method in METHODS_SUBSAMPLE_ONLY:
            patient_counts = PATIENT_COUNTS_SUBSAMPLE
        else:
            patient_counts = PATIENT_COUNTS_FULL

        res_class = method_to_resource_class[method]
        print(f"\n  {method} (patients: {patient_counts}):")

        for np_ in sorted(patient_counts):
            if np_ not in RESOURCE_PROFILES:
                print(f"    WARNING: No resource profile for {np_} patients, skipping")
                continue
            if res_class not in RESOURCE_PROFILES[np_]:
                print(f"    WARNING: No {res_class} profile for {np_} patients, skipping")
                continue

            res = RESOURCE_PROFILES[np_][res_class]
            script_content = method_generators[method](np_, res)

            script_path = jobs_dir / f"{method}_{np_}p.sh"
            script_path.write_text(script_content)

            submit_script(str(script_path), dry_run=args.dry_run)
            total_jobs += len(SEEDS)

    print(f"\n{'='*70}")
    print(f"Total job array tasks: {total_jobs}")
    print(f"Generated scripts in: {jobs_dir}")
    print(f"Logs will be in: {logs_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
