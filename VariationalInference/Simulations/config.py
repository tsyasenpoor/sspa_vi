# BRay/VariationalInference/Simulations/config.py
"""Locked defaults for the DRGP flat-model simulation v1.

All knobs documented in docs/superpowers/specs/2026-06-08-DRGP-flat-simulation-design.md §8.
Edit-once; runners read from here. Per-knob source of truth.
"""
from __future__ import annotations
from pathlib import Path

# ---- Paths --------------------------------------------------------------
REPO_ROOT = Path("/labs/Aguiar/SSPA_BRAY")
SCDESIGN3_DIR = REPO_ROOT / "scdesign3_covid19_cellTypeMarginal_8kcells_10kgenes"
NB_PARAMS_H5 = SCDESIGN3_DIR / "nb_params.h5"
BASELINE_COUNTS_CSV = SCDESIGN3_DIR / "simulated_counts.csv"
BASELINE_META_CSV = SCDESIGN3_DIR / "simulated_metadata.csv"
SIM_ROOT = REPO_ROOT / "data" / "Simulations" / "sim_flat_v1"

# ---- Cell-type vocabulary (order matches majorType labels) ---------------
CELL_TYPES = ["B", "CD4", "CD8", "Mono", "myeloid cells", "NK"]   # T = 6
T = len(CELL_TYPES)
TYPE_TO_INT = {t: i for i, t in enumerate(CELL_TYPES)}

# ---- Generator constants ------------------------------------------------
N_CELLS = 8000
N_GENES = 10000
N_PATIENTS = 80                  # G; 100 cells each. v2: patient-inherited labels (see
                                 # docs/.../2026-06-11-DRGP-sim-v2-patient-label-design.md)
CARRIER_RATE = 0.5               # P(K_{g,l}=1) per-program carrier; factorization-signal knob,
                                 # independent of h2 and of label prevalence.
# Within-type perturbation fraction rho: fraction of a carrier patient's responder cells that
# actually carry signal for a program. Swept; delta calibrated at the headline rho only, so
# lower rho genuinely dilutes total signal.
PERTURB_FRAC_VALUES = [0.1, 0.3, 0.6]
PERTURB_FRAC_HEADLINE = 0.3
LIABILITY_FORM = "probit"        # probit Gaussian-liability: exact variance partition, no chi calib
TARGET_PREVALENCE = 0.5          # P(D_g=1); enforced by tau = median(ell_g) at every h2
KAPPA_PATH = 2                   # iota_path
KAPPA_DENOVO = 2                 # iota_denovo
IOTA = KAPPA_PATH + KAPPA_DENOVO  # = 4 causal programs
N_DECOY = 1                      # decoy column at l=0
L_COLS = IOTA + N_DECOY          # = 5 columns of (A, U, v_star)
CARRIER_SIZE_LO, CARRIER_SIZE_HI = 50, 100
U_LO, U_HI = 0.5, 1.5            # unit pattern range
RESPONDER_SIZE_LO, RESPONDER_SIZE_HI = 2, 3  # |T_l| in {2,3}
ALPHA_B, LAMBDA_B = 2.0, 2.0     # Gamma(2,2) - mean 1
THETA_BASE = 0.01
DIRICHLET_A0 = 20.0              # composition-distractor stiffness

# v_star magnitudes - one strong / one weak per pathway and de-novo block
V_STAR_MAGNITUDES = [2.0, 0.7, 2.0, 0.7]   # signs drawn balanced per truth
V_STAR_DECOY = 0.0

# ---- Sweep grid ---------------------------------------------------------
G_TRUTH = 5
H2_VALUES = [0.1, 0.3, 0.5, 0.7]
R_VALUES = [0.05, 0.15, 0.30]
KAPPA_VALUES = [8, 10, 14]
INNER_SEED_GRID = 0              # single seed across grid; inner-seed only at stability
INNER_SEEDS_STABILITY = list(range(10))

HEADLINE_CELL = dict(h2=0.3, r=0.15, K=8, rho=PERTURB_FRAC_HEADLINE)
STABILITY_CELL = dict(h2=0.5, r=0.15, K=10, rho=PERTURB_FRAC_HEADLINE)
# rho 1-D sweep at the headline (h2,r,K) — the "fewer cells carry signal" axis.
RHO_SENSITIVITY_CELLS = [dict(h2=0.3, r=0.15, K=8, rho=rho) for rho in PERTURB_FRAC_VALUES]
RW_SENSITIVITY_CELLS = [
    dict(h2=0.3, r=0.15, K=8, rho=PERTURB_FRAC_HEADLINE),    # headline
    dict(h2=0.1, r=0.15, K=8, rho=PERTURB_FRAC_HEADLINE),    # low-h2 corner
]
RW_SENSITIVITY_VALUES = [5.0, 15.0, 50.0]

# ---- CAVI knobs (locked per design §8) ----------------------------------
# PG-VI fix (2026-06-10): the supervised update is taken at the derivation's
# natural weight 1 (DRGP_VI_full_derivation.md Eq 8.1-8.2), NOT the nnz/n
# tempering that floored b_theta and diverged theta at scale; and b_v is held
# fixed (no in-loop _calibrate_b_v, which on un-settled theta floors b_v and
# freezes v). In 'one' mode REGRESSION_WEIGHT below is inert. See
# docs/PLAN_A_normalized_design_FUTURE_WORK.md for the full arc.
SUPERVISED_UPDATE_WEIGHT = "one"   # 'one' = weight 1 (fixed); 'rw' = old broken nnz/n tempering
CALIBRATE_B_V = False              # keep b_v fixed at CAVI_B_V (no in-loop calibration)
CAVI_B_V = 1.0
# Plan A (2026-06-10): L1-normalized regression design s=θ/‖θ‖₁. The rw sweep on
# the raw design showed NO weight recovers programs (matched_cosine flat ~0.05 for
# rw∈{1,5,20,50}; high rw only memorizes train→1.0 / test→chance). Normalized design
# severs the magnitude channel so supervision shapes simplex DIRECTION = recovery.
REGRESSION_DESIGN = "normalized"   # 'normalized' | 'raw'
REGRESSION_WEIGHT = 15.0
CAVI_A = 0.3
CAVI_C = 0.3
CAVI_MAX_ITER = 3000
CAVI_CHECK_FREQ = 5
CAVI_TOL = 0.001
CAVI_V_WARMUP = 10   # Must be << regression-stop patience (~20 iters); otherwise Reg never moves before the stopper fires and best_reg checkpoint is the pre-supervision state. Found 2026-06-09 via truth-2 diagnostic.
EARLY_STOPPING = "elbo"   # 2026-06-10: the runner passes no validation set, so 'heldout_ll' fell back to monitoring TRAINING Reg (patience 15) and, at weight-1 where Reg is non-monotone during the 200-iter ramp, fired at iter ~30 and restored the pre-supervision best_reg checkpoint (theta-only AUC ~chance, cosine ~0). 'elbo' now: (a) ELBO is the monotone objective (Fix 3), label-leak-free; (b) restore is mode-aware so 'elbo' keeps the final converged state instead of best_reg; (c) convergence-stop is blocked until v_warmup+ramp_iters so supervision fully engages.

# ---- Classifier knobs ---------------------------------------------------
import numpy as np
LR_C_GRID = np.logspace(-4, 2, 11).tolist()
LR_CV_FOLDS = 3
LR_MAX_ITER = 2000
SPECTRA_EPOCHS = 2000   # Spectra full-batch training epochs (CPU torch); baseline fit budget

# ---- Split knobs --------------------------------------------------------
N_TEST_PATIENTS = N_PATIENTS // 5   # 16 of 80 held out; used by every runner
K_FOLD_GRID = 1                  # single 64/16 patient split on grid
K_FOLD_STABILITY = 5

# ---- Crossover gate thresholds (Phase B) --------------------------------
GATE_RLOW_MIN = 0.10
GATE_RMID_LO, GATE_RMID_HI = 0.03, 0.10
GATE_RHIGH_MAX = 0.03
GATE_N_TRUTHS = 3                # average over truths {0, 1, 2}

# ---- Phase A rw-engagement gate -----------------------------------------
RW_ENGAGEMENT_RW0_AUC_MAX = 0.55       # Theta-only train AUC at rw=0 (γ excluded); pure factor signal at chance
RW_ENGAGEMENT_GAP_MIN = 0.10           # Theta-only train AUC gap rw=15 vs rw=0; factor-level supervision

# ---- NB parameterization (filled by T3) ---------------------------------
# Either "size" (negative_binomial(n=sigma_mat_mean, p=...)) or "dispersion"
# (negative_binomial(n=1/sigma_mat_mean, p=...)). Resolved by Phase A.2.
NB_SIZE_FROM_SIGMA: str = "dispersion"   # resolved by verify_nb_param Phase A.2; see data/Simulations/sim_flat_v1/nb_param_gate.json
