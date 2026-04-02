#!/usr/bin/env python
"""
Submit IBD benchmark SLURM jobs.
=================================

Runs all methods on the full IBD dataset (EMTAB11349, 590 bulk RNA-seq samples).
No subsampling — each method runs on the complete dataset.

Methods:
  - DRGP variants: unmasked, masked, pathway_init, combined
  - Baselines: PCA/NMF + classifiers
  - Spectra supervised (pathway-informed factorization)
  - scHPF (hierarchical Poisson factorization)

Note: Spectra and scHPF are single-cell methods being tested on bulk data.
      IBD columns are mapped to COVID-expected column names so the existing
      scripts run without modification.  Results labelled "severity" and
      "outcome" are really predicting the IBD binary "disease" label.

Usage:
    python submit_ibd_benchmark.py [--dry-run] [--methods ...]
"""
from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed_ibd"
GENE_ANNOT = f"{DATA_DIR}/gene_annotation.csv"
GMT_FILE = "/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt"
RESULTS_ROOT = "/labs/Aguiar/SSPA_BRAY/results/ibd_benchmark"
METHODS_ROOT = f"{RESULTS_ROOT}/methods"

VI_DIR = "/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference"
CONDA_SH = "/home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh"
MAIL_USER = "tsyasenpoor@uconn.edu"

# H5AD derived from IBD CSVs (for spectra/schpf); created by this script
H5AD_IBD = f"{RESULTS_ROOT}/derived/ibd_bulk.h5ad"

SEEDS = [42, 123, 456, 789, 1024]

# Resource profiles: (mem_gb, time_hours, partition, qos, cpus, gpu)
# 590 bulk samples — much lighter than single-cell
RESOURCE_PROFILES = {
    "drgp":      (100, 48, "general", "general", 4, 1),
    "spectra":   (64,  24, "general", "general", 4, 1),
    "schpf":     (64,  24, "general", "general", 8, 0),
    "baselines": (128, 24, "general", "general", 8, 0),
}

METHODS_ALL = {
    "drgp_unmasked", "drgp_masked", "drgp_pathway_init",
    "drgp_combined", "baselines", "spectra_sup", "schpf",
}


# ============================================================================
# Helpers
# ============================================================================

def _seeds_bash_array() -> str:
    return "(" + " ".join(str(s) for s in SEEDS) + ")"


def _slurm_header(
    job_name: str, mem_gb: int, time_hours: int, partition: str, qos: str,
    cpus: int, gpu: int, array_size: int = 5,
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
    """Save wall time and peak memory to benchmark.json."""
    return textwrap.dedent(f"""\
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "Job finished at: $(date)"
        echo "Total elapsed: ${{ELAPSED}}s"

        PEAK_RSS_MB=0
        if [[ -f /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${{SLURM_JOB_ID}}/memory.max_usage_in_bytes ]]; then
            PEAK_RSS_BYTES=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${{SLURM_JOB_ID}}/memory.max_usage_in_bytes 2>/dev/null || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_BYTES / 1048576))
        elif [[ -f /sys/fs/cgroup/memory.peak ]]; then
            PEAK_RSS_BYTES=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_BYTES / 1048576))
        elif command -v sstat &>/dev/null; then
            PEAK_RSS_MB=$(sstat -j ${{SLURM_JOB_ID}}.batch --format=MaxRSS -n 2>/dev/null | tr -d ' K' | head -1 || echo 0)
            PEAK_RSS_MB=$((PEAK_RSS_MB / 1024))
        fi
        if [[ "$PEAK_RSS_MB" -eq 0 ]]; then
            PEAK_RSS_KB=$(grep "VmHWM" /proc/$$/status 2>/dev/null | awk '{{print $2}}' || echo "0")
            PEAK_RSS_MB=$((PEAK_RSS_KB / 1024))
        fi
        echo "Peak memory: ${{PEAK_RSS_MB}} MB"

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
# H5AD creation (for spectra / scHPF)
# ============================================================================

def create_ibd_h5ad(dry_run: bool = False) -> None:
    """Create h5ad from IBD CSVs with columns mapped for spectra/schpf scripts.

    Maps IBD columns to COVID-expected names so existing scripts work as-is:
      - disease=1 → "CoVID-19 severity"="severe/critical", disease=0 → "mild/moderate"
      - disease=1 → "Outcome"="deceased", disease=0 → "discharged"
      - sex_female=1 → "Sex"="F", sex_female=0 → "Sex"="M"
      - majorType = "bulk" (no cell types in bulk data)
      - comorbidity columns set to 0
    """
    h5ad_path = Path(H5AD_IBD)
    if h5ad_path.exists():
        print(f"  H5AD already exists: {H5AD_IBD}")
        return

    if dry_run:
        print(f"  [DRY RUN] Would create {H5AD_IBD}")
        return

    import anndata as ad
    import numpy as np
    import pandas as pd

    print(f"  Creating IBD h5ad for spectra/scHPF ...")
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    ge = pd.read_csv(f"{DATA_DIR}/gene_expression_raw_processed.csv.gz", compression="gzip")
    resp = pd.read_csv(f"{DATA_DIR}/responses.csv.gz", compression="gzip")
    aux = pd.read_csv(f"{DATA_DIR}/aux_data.csv.gz", compression="gzip")

    sample_ids = ge["Sample_ID"].astype(str)
    gene_cols = [c for c in ge.columns if c != "Sample_ID"]
    X = ge[gene_cols].values.astype(np.float32)

    disease = resp["disease"].values
    sex_female = aux["sex_female"].values

    obs = pd.DataFrame({
        "sampleID": sample_ids.values,
        "disease": disease,
        "age": aux["age"].values,
        "sex_female": sex_female,
        # Mapped columns for spectra/schpf compatibility
        "CoVID-19 severity": np.where(disease == 1, "severe/critical", "mild/moderate"),
        "Outcome": np.where(disease == 1, "deceased", "discharged"),
        "Sex": np.where(sex_female == 1, "F", "M"),
        "cm_asthma_copd": np.int32(0),
        "cm_cardio": np.int32(0),
        "cm_diabetes": np.int32(0),
        "majorType": "bulk",
    }, index=sample_ids.values)

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_cols),
    )
    # Store raw counts (rounded, clipped to 0) for scHPF
    adata.layers["raw"] = np.rint(np.clip(X, 0, None)).astype(np.float32)

    adata.write_h5ad(str(h5ad_path))
    print(f"  Created: {H5AD_IBD} ({adata.n_obs} samples x {adata.n_vars} genes)")


# ============================================================================
# Job script generators
# ============================================================================

def gen_drgp_script(mode: str, res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = f"ibd_drgp_{mode}"
    out_tmpl = f"{METHODS_ROOT}/seed_${{SEED}}/drgp_{mode}"

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    # pathway_file only needed for masked/pathway_init/combined
    pathway_flag = f"--pathway-file {GMT_FILE}" if mode != "unmasked" else ""

    tuned_unmasked_args = ""
    if mode == "unmasked":
        tuned_unmasked_args = textwrap.dedent("""\
            --alpha-theta 2.2346624826836234 \\
            --alpha-beta 2.923276954893561 \\
            --alpha-xi 2.894187744774477 \\
            --alpha-eta 2.0968778579403313 \\
            --b-v 0.10291201492104285 \\
            --sigma-gamma 0.1383067648727466 \\
            --regression-weight 1.7777516006642433 \\
            --alpha-pi 0.5098212762286258 \\
            --beta-pi-scale 2.2003828796521834 \\
        """)

    cmd = textwrap.dedent(f"""\
        OUTPUT="{out_tmpl}"
        mkdir -p "$OUTPUT"

        python -u {VI_DIR}/quick_reference.py \\
            --data {DATA_DIR} \\
            --gene-annotation {GENE_ANNOT} \\
            --mode {mode} \\
            --n-factors {130 if mode == 'unmasked' else 500} \\
            --a 0.3 --c 0.3 \\
            --sigma-v 2.0 \\
            --sigma-gamma 0.5 \\
            --regression-weight 1.0 \\
            {tuned_unmasked_args}            --max-iter 50000 --tol 0.001 \\
            --v-warmup 50 --check-freq 5 \\
            --early-stopping none \\
            --label-column disease \\
            --aux-columns age sex_female \\
            {pathway_flag} \\
            --seed $SEED \\
            --output-dir "$OUTPUT" \\
            --verbose
    """)

    footer = _benchmark_footer(out_tmpl)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_baselines_script(res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = "ibd_baselines"
    out_root = f"{METHODS_ROOT}/seed_${{SEED}}/baselines"

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    cmd = textwrap.dedent(f"""\
        OUT_ROOT="{out_root}"
        mkdir -p "$OUT_ROOT"

        echo "Running baselines for disease (seed=$SEED)..."

        python -u {VI_DIR}/run_baselines.py \\
            --data {DATA_DIR} \\
            --label-column disease \\
            --aux-columns age sex_female \\
            --output-dir "$OUT_ROOT" \\
            --latent-dim 50 \\
            --seed $SEED \\
            --verbose

        echo "Baselines finished at: $(date)"
    """)

    footer = _benchmark_footer(out_root)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_spectra_sup_script(res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = "ibd_spectra_sup"

    spectra_out = f"{METHODS_ROOT}/seed_${{SEED}}/spectra_sup"
    baseline_out = f"{METHODS_ROOT}/seed_${{SEED}}/spectra_sup_baselines"

    header = _slurm_header(job_name, mem, time_h, part, qos, cpus, gpu)
    preamble = _common_preamble()

    cmd = textwrap.dedent(f"""\
        SPECTRA_OUT="{spectra_out}"
        BASELINE_OUT="{baseline_out}"
        mkdir -p "$SPECTRA_OUT" "$BASELINE_OUT"

        # Fit Spectra (supervised with REACTOME pathways)
        FIT_START=$(date +%s)
        python -u {VI_DIR}/run_spectra_supervised.py \\
            --h5ad {H5AD_IBD} \\
            --gmt-file {GMT_FILE} \\
            --output-dir "$SPECTRA_OUT" \\
            --require-prefix REACTOME \\
            --lam 0.01 \\
            --num-epochs 10000 \\
            --seed $SEED \\
            --verbose
        FIT_END=$(date +%s)
        FIT_ELAPSED=$((FIT_END - FIT_START))
        echo "Spectra sup fit completed in ${{FIT_ELAPSED}}s"


        python3 -c "
import sys; sys.path.insert(0, '{VI_DIR}/..')
import anndata as ad
from VariationalInference.create_subsamples import _build_metadata
src = ad.read_h5ad('{H5AD_IBD}')
meta = _build_metadata(src.obs)
meta.to_csv('$SPECTRA_OUT/metadata_ibd.csv')
print(f'Wrote metadata: {{len(meta)}} samples')
"

        # Downstream classifiers
        python -u {VI_DIR}/run_spectra_baselines.py \\
            --cell-scores "$SPECTRA_OUT/spectra_cell_scores.npy" \\
            --data "$SPECTRA_OUT/metadata_ibd.csv" \\
            --labels disease \\
            --output-dir "$BASELINE_OUT" \\
            --seed $SEED
    """)

    footer = _benchmark_footer(spectra_out)
    return f"{header}\n\n{preamble}\n{cmd}\n{footer}"


def gen_schpf_script(res: tuple) -> str:
    mem, time_h, part, qos, cpus, gpu = res
    job_name = "ibd_schpf"

    schpf_out = f"{METHODS_ROOT}/seed_${{SEED}}/schpf"
    baseline_out = f"{METHODS_ROOT}/seed_${{SEED}}/schpf_baselines"
    model_file = f"${{SCHPF_OUT}}/ibd_bulk.scHPF_K50_b0_5trials.joblib"

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

    cmd = textwrap.dedent(f"""\
        SCHPF_OUT="{schpf_out}"
        BASELINE_OUT="{baseline_out}"
        mkdir -p "$SCHPF_OUT" "$BASELINE_OUT"

        # Prepare temp mtx + metadata from h5ad
        TMPDIR_SCHPF=$(mktemp -d /tmp/schpf_ibd_${{SLURM_JOB_ID}}_${{SEED}}_XXXXXX)
        echo "Temp dir for scHPF inputs: $TMPDIR_SCHPF"

        conda activate jax_gpu
        python3 -c "
import sys; sys.path.insert(0, '{VI_DIR}/..')
import anndata as ad, numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from VariationalInference.create_subsamples import _build_metadata
from pathlib import Path

src = ad.read_h5ad('{H5AD_IBD}')
tmpdir = Path('$TMPDIR_SCHPF')

# Write mtx (integer counts from raw layer)
X = src.layers['raw'] if 'raw' in src.layers else src.X
if sp.issparse(X):
    X = X.tocsr()
else:
    X = sp.csr_matrix(X)
X_int = X.copy()
X_int.data = np.rint(np.clip(X_int.data, 0, None)).astype(np.int32, copy=False)
mmwrite(tmpdir / 'filtered.mtx', X_int.tocoo(), field='integer')

# Write genes
genes = pd.DataFrame({{'gene_id': src.var_names.astype(str), 'gene_name': src.var_names.astype(str)}})
genes.to_csv(tmpdir / 'genes.txt', sep='\\t', header=False, index=False)

# Write metadata
meta = _build_metadata(src.obs)
meta.to_csv(tmpdir / 'metadata_ibd.csv')
print(f'Wrote scHPF inputs to {{tmpdir}}: {{src.n_obs}} samples x {{src.n_vars}} genes')
"

        # Train scHPF (requires schpf_p37 env)
        conda activate schpf_p37
        echo "Python (scHPF): $(python --version)"

        FIT_START=$(date +%s)
        scHPF train \\
            -i "$TMPDIR_SCHPF/filtered.mtx" \\
            -o "$SCHPF_OUT" \\
            -p ibd_bulk \\
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

        # Downstream classifiers (requires jax_gpu env)
        conda activate jax_gpu
        echo "Python (baselines): $(python --version)"

        python -u {VI_DIR}/run_schpf_baselines.py \\
            --model {model_file} \\
            --data "$TMPDIR_SCHPF/metadata_ibd.csv" \\
            --labels disease \\
            --output-dir "$BASELINE_OUT" \\
            --seed $SEED \\
            --verbose

        # Clean up temp files
        rm -rf "$TMPDIR_SCHPF"
        echo "Cleaned up temp dir"
    """)

    footer = _benchmark_footer(schpf_out)
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
    p = argparse.ArgumentParser(description="Submit IBD benchmark SLURM jobs")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate scripts but don't submit")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["drgp_unmasked", "drgp_masked", "drgp_pathway_init",
                            "drgp_combined", "spectra_sup", "schpf", "baselines"],
                   help="Methods to run")
    return p.parse_args()


def main():
    args = parse_args()

    jobs_dir = Path(RESULTS_ROOT) / "jobs_generated"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(RESULTS_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IBD BENCHMARK - JOB SUBMISSION (full dataset, no subsampling)")
    print("=" * 70)
    print(f"  Data:    {DATA_DIR} (590 samples x 14183 genes)")
    print(f"  Seeds:   {SEEDS}")
    print(f"  Methods: {args.methods}")
    print(f"  Results: {RESULTS_ROOT}")
    print(f"  Dry run: {args.dry_run}")

    # Create h5ad for spectra/schpf if needed
    needs_h5ad = any(m in args.methods for m in ["spectra_sup", "schpf"])
    if needs_h5ad:
        print("\n  Preparing h5ad for spectra/scHPF ...")
        create_ibd_h5ad(dry_run=args.dry_run)

    method_generators = {
        "drgp_unmasked":     lambda res: gen_drgp_script("unmasked", res),
        "drgp_masked":       lambda res: gen_drgp_script("masked", res),
        "drgp_pathway_init": lambda res: gen_drgp_script("pathway_init", res),
        "drgp_combined":     lambda res: gen_drgp_script("combined", res),
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

        res_class = method_to_resource_class[method]
        res = RESOURCE_PROFILES[res_class]

        script_content = method_generators[method](res)
        script_path = jobs_dir / f"{method}.sh"
        script_path.write_text(script_content)

        print(f"\n  {method}:")
        submit_script(str(script_path), dry_run=args.dry_run)
        total_jobs += len(SEEDS)

    print(f"\n{'='*70}")
    print(f"Total job array tasks: {total_jobs}")
    print(f"Generated scripts in: {jobs_dir}")
    print(f"Logs will be in: {logs_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
