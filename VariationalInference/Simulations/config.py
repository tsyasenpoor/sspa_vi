# BRay/VariationalInference/Simulations/config.py
"""Locked defaults for the DRGP flat-model simulation v1.

All knobs documented in docs/superpowers/specs/2026-06-08-DRGP-flat-simulation-design.md §8.
Edit-once; runners read from here. Per-knob source of truth.
"""
from __future__ import annotations
from pathlib import Path

# ---- Paths --------------------------------------------------------------
REPO_ROOT = Path("/labs/Aguiar/SSPA_BRAY")
SCDESIGN3_DIR = REPO_ROOT / "scdesign3_covid19_8kcells_10kgenes"
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
N_PATIENTS = 40                  # G; balanced 20/20
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

HEADLINE_CELL = dict(h2=0.3, r=0.15, K=8)
STABILITY_CELL = dict(h2=0.5, r=0.15, K=10)
RW_SENSITIVITY_CELLS = [
    dict(h2=0.3, r=0.15, K=8),    # headline
    dict(h2=0.1, r=0.15, K=8),    # low-h2 corner
]
RW_SENSITIVITY_VALUES = [5.0, 15.0, 50.0]

# ---- CAVI knobs (locked per design §8) ----------------------------------
REGRESSION_WEIGHT = 15.0
CAVI_A = 0.3
CAVI_C = 0.3
CAVI_MAX_ITER = 3000
CAVI_CHECK_FREQ = 5
CAVI_TOL = 0.001
CAVI_V_WARMUP = 50
EARLY_STOPPING = "heldout_ll"

# ---- Classifier knobs ---------------------------------------------------
import numpy as np
LR_C_GRID = np.logspace(-4, 2, 11).tolist()
LR_CV_FOLDS = 3
LR_MAX_ITER = 2000

# ---- Split knobs --------------------------------------------------------
K_FOLD_GRID = 1                  # single 32/8 patient split on grid
K_FOLD_STABILITY = 5

# ---- Crossover gate thresholds (Phase B) --------------------------------
GATE_RLOW_MIN = 0.10
GATE_RMID_LO, GATE_RMID_HI = 0.03, 0.10
GATE_RHIGH_MAX = 0.03
GATE_N_TRUTHS = 3                # average over truths {0, 1, 2}

# ---- Phase A rw-engagement gate -----------------------------------------
RW_ENGAGEMENT_RW0_AUC_MAX = 0.52       # cell_auc_integrated[rw=0] must be ~chance
RW_ENGAGEMENT_GAP_MIN = 0.05           # cell_auc[rw=15] - cell_auc[rw=0]

# ---- NB parameterization (filled by T3) ---------------------------------
# Either "size" (negative_binomial(n=sigma_mat_mean, p=...)) or "dispersion"
# (negative_binomial(n=1/sigma_mat_mean, p=...)). Resolved by Phase A.2.
NB_SIZE_FROM_SIGMA: str = "dispersion"   # resolved by verify_nb_param Phase A.2; see data/Simulations/sim_flat_v1/nb_param_gate.json
