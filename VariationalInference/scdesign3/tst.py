"""Run scDesign3, extract NB parameters (mu, sigma), and verify the
parameterization convention (sim_sex_only plan gotcha 7).

IMPORTANT: this needs ~30-50 GB RAM at peak (count matrix CSV write + the two
N x G parameter matrices). The previous interactive run on a sub node OOM'd
during the CSV write. Run under sbatch with --mem=64G (or larger), e.g.:

    sbatch --gres=gpu:0 --mem=64G --time=4:00:00 --partition=general \
           --wrap="bash -c 'source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh && \
                            conda activate jax_gpu && \
                            python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/scdesign3/tst.py'"
"""
import sys
from pathlib import Path

import h5py
import numpy as np

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from VariationalInference.scdesign3 import ScDesign3Simulator

INPUT_FILE = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/control_adata.h5ad"
OUTPUT_DIR = Path("./scdesign3_covid19_8kcells_10kgenes").resolve()

simulator = ScDesign3Simulator(
    r_executable="/home/FCAM/tyasenpoor/miniconda3/envs/bray_cpu/bin/Rscript",
)

# Skip the full sim if a prior run already produced marginal fits + simulated SCE.
# extract_only re-runs just construct_data + extract_para using saved RDS files —
# cheap (~30s) compared to the full marginal/copula refit.
prior_artifacts = ["marginal_models.rds", "simulated_sce.rds", "simulated_counts.csv"]
extract_only = all((OUTPUT_DIR / f).exists() for f in prior_artifacts)
if extract_only:
    print(f"\n[extract-only] Reusing prior fits in {OUTPUT_DIR}")

result = simulator.simulate(
    input_file=INPUT_FILE,
    output_dir=str(OUTPUT_DIR),
    n_cells=8000, n_genes=10000,
    celltype_column="majorType",
    family="nb", copula="gaussian",
    return_model=True,           # ensure marginal_models.rds is written when running fresh
    extract_only=extract_only,
)
print(f"\nSimulated counts (genes x cells): {result.counts.shape}")

# ---------------------------------------------------------------------------
# Load extracted NB parameters
# ---------------------------------------------------------------------------
params_file = OUTPUT_DIR / "nb_params.h5"
if not params_file.exists():
    raise FileNotFoundError(
        f"{params_file} missing — run_scdesign3.R didn't write params. "
        "Check the extract_para block executed (needs return_model=True)."
    )

with h5py.File(params_file, "r") as f:
    mu_mat    = f["mu_mat"][...]
    sigma_mat = f["sigma_mat"][...]
    fu = f["family_use"][()]
    if isinstance(fu, np.ndarray):
        fu = fu.item() if fu.ndim == 0 else fu.flat[0]
    family_use = fu.decode() if isinstance(fu, bytes) else fu

# rhdf5 writes R matrices with R's (row, col) dims, which h5py reads back transposed
# relative to C-order. R-side mean_mat is (cells x genes); detect orientation by matching
# to the count matrix (genes x cells) and transpose to (cells x genes) for the moments below.
counts_gc = result.counts                       # (genes x cells)
n_genes, n_cells = counts_gc.shape
if mu_mat.shape == (n_cells, n_genes):
    pass
elif mu_mat.shape == (n_genes, n_cells):
    mu_mat = mu_mat.T
    sigma_mat = sigma_mat.T
else:
    raise ValueError(
        f"mu_mat shape {mu_mat.shape} matches neither (cells,genes)=({n_cells},{n_genes}) "
        f"nor its transpose; check rhdf5 write."
    )

print(f"mu_mat:    {mu_mat.shape} (cells x genes), dtype={mu_mat.dtype}")
print(f"sigma_mat: {sigma_mat.shape}")
print(f"family_use (from HDF5): {family_use!r}")

# extract_para pads QC-failed genes with mu=0, sigma=NaN. Restrict the comparison
# to QC-passed genes so we don't propagate NaN into the moment stats.
qc_pass = ~np.isnan(sigma_mat).any(axis=0)
print(f"QC-passed genes (non-NaN sigma): {int(qc_pass.sum())} / {n_genes}")
mu_qc    = mu_mat[:, qc_pass]
sigma_qc = sigma_mat[:, qc_pass]
counts_qc = counts_gc.T[:, qc_pass].astype(np.float64)   # (cells x genes_qc)

# ---------------------------------------------------------------------------
# Gotcha 7: which NB parameterization does scDesign3's sigma_mat use?
#   NBI (GAMLSS default):   Var = mu + sigma * mu^2
#   size-theta convention:  Var = mu + mu^2 / sigma
# Per-gene moments are clean — copula leaves marginals intact.
# ---------------------------------------------------------------------------
obs_mean = counts_qc.mean(axis=0)
obs_var  = counts_qc.var(axis=0)
exp_mean = mu_qc.mean(axis=0)
exp_var_NBI       = (mu_qc + sigma_qc * mu_qc ** 2).mean(axis=0)
exp_var_sizeTheta = (mu_qc + mu_qc ** 2 / np.clip(sigma_qc, 1e-12, None)).mean(axis=0)

# Drop near-zero genes so log ratios stay sane.
keep = (obs_mean > 1.0) & (exp_mean > 1.0) & (obs_var > 1e-6)
print(f"\n--- Gotcha 7 sanity check ({int(keep.sum())} of {int(qc_pass.sum())} QC genes with mean > 1) ---")

mean_ratio = np.median(obs_mean[keep] / exp_mean[keep])
print(f"median (obs_mean / exp_mean) = {mean_ratio:.3f}    (expect ~1.0)")

def med_logr(a, b):
    return float(np.median(np.log(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None))))

r_nbi  = med_logr(obs_var[keep], exp_var_NBI[keep])
r_size = med_logr(obs_var[keep], exp_var_sizeTheta[keep])
print(f"median log(obs_var / NBI_var       ) = {r_nbi:+.3f}    (NBI:  Var = mu + sigma*mu^2)")
print(f"median log(obs_var / sizeTheta_var ) = {r_size:+.3f}    (size: Var = mu + mu^2/sigma)")

if abs(r_nbi) < abs(r_size):
    print("=> NBI parameterization. For downstream NB draws use   size_theta = 1/sigma_mat.")
else:
    print("=> size-theta parameterization. Use sigma_mat directly as size in NB draws.")

# ---------------------------------------------------------------------------
# With sigma_formula='1', sigma_qc should be ~constant across cells per gene.
# Confirm so the downstream plan can collapse to a per-gene phi vector.
# ---------------------------------------------------------------------------
sigma_range = sigma_qc.max(axis=0) - sigma_qc.min(axis=0)
print(f"\nsigma_mat per-gene (max - min) over QC-passed genes:")
print(f"  median = {np.median(sigma_range):.2e}    max = {sigma_range.max():.2e}")
print("  (both should be ~0 if sigma is truly intercept-only)")
