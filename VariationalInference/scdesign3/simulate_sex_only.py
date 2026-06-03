#!/usr/bin/env python
"""Sex-only DRGP synthetic suite v1 (up-regulated only).

Implements `sim_sex_only_implementation_plan_v1.md`:
- Phase A baseline = scDesign3 NB params (nb_params.h5) at simulated cells.
- Phase B = patient-level genetic liability g_p drives carrier-cell expression
  shifts on D disease programs; sex_p enters the label only (channel separation).
- v1: up-only (pi_up=1.0), no noise programs, default 100 patients x 3 programs x 30 genes.

Outputs `perturbed.h5ad` + `ground_truth.npz` + `config.json` into <out_dir>.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import roc_auc_score


# ============================== Config ==================================

@dataclass
class SimConfig:
    n_patients: int = 100
    n_disease_programs: int = 3
    n_noise_programs: int = 0           # v1: off
    genes_per_program: int = 30
    delta_min: float = 0.1
    delta_max: float = 0.3
    pi_up: float = 1.0                  # v1: all up
    lambda_l: float = 1.0
    carrier_types: dict | None = None   # None => every program covers all cell types
    w_g: float = 2.0
    w_s: float = 0.5
    target_prevalence: float = 0.5
    seed: int = 42


# ============================ Step 1: Baseline ==========================

def load_baseline(scdesign_dir: Path, celltype_col: str = "majorType"):
    """Load (mu0, phi, cell_type, gene_names, cell_names) for QC-passed genes.

    mu0   : (N, G) NB means at simulated cells.
    phi   : (G,)   NB size_theta = 1/sigma under GAMLSS NBI (verified by tst.py).
    """
    with h5py.File(scdesign_dir / "nb_params.h5", "r") as f:
        mu_mat    = f["mu_mat"][...]
        sigma_mat = f["sigma_mat"][...]
        gene_names_all = [s.decode() if isinstance(s, bytes) else s
                          for s in f["gene_names"][...]]
        cell_names_all = [s.decode() if isinstance(s, bytes) else s
                          for s in f["cell_names"][...]]

    meta = pd.read_csv(scdesign_dir / "simulated_metadata.csv", index_col=0)

    # Orientation: we want (cells x genes).
    n_cells_h5, n_genes_h5 = len(cell_names_all), len(gene_names_all)
    if mu_mat.shape == (n_genes_h5, n_cells_h5):
        mu_mat = mu_mat.T
        sigma_mat = sigma_mat.T
    elif mu_mat.shape != (n_cells_h5, n_genes_h5):
        raise ValueError(f"mu_mat shape {mu_mat.shape} doesn't match h5 dims")

    meta = meta.reindex(cell_names_all)
    if meta.isna().any().any():
        raise ValueError("metadata reindex produced NaN — cell-name mismatch with nb_params.h5")
    if celltype_col not in meta.columns:
        raise ValueError(f"column {celltype_col!r} not in metadata: {list(meta.columns)}")

    # QC-pass: scDesign3 fills failed marginals with sigma=NaN (mu=0).
    qc_pass = ~np.isnan(sigma_mat).any(axis=0)
    mu0   = mu_mat[:, qc_pass].astype(np.float64)
    phi   = 1.0 / sigma_mat[0, qc_pass].astype(np.float64)         # sigma constant under sigma_formula='1'
    gene_names = [g for g, k in zip(gene_names_all, qc_pass) if k]
    cell_type  = meta[celltype_col].astype(str).to_numpy()
    return mu0, phi, cell_type, gene_names, cell_names_all


# ============================ Step 2: Patients ==========================

def assign_patients(cell_type: np.ndarray, n_patients: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Round-robin within each cell type so each patient spans cell types."""
    N = len(cell_type)
    patient_id = np.full(N, -1, dtype=np.int64)
    for ct in np.unique(cell_type):
        idx = np.where(cell_type == ct)[0]
        rng.shuffle(idx)
        patient_id[idx] = np.arange(len(idx)) % n_patients
    assert (patient_id >= 0).all()
    return patient_id


# ============================ Step 3: Patient vars ======================

def patient_vars(n_patients: int, rng: np.random.Generator):
    sex_p = rng.integers(0, 2, size=n_patients).astype(np.int8)
    g_p   = rng.normal(0.0, 1.0, size=n_patients).astype(np.float64)
    return sex_p, g_p


# ============================ Step 4: Programs ==========================

def define_programs(cfg: SimConfig, gene_names: list[str], cell_type: np.ndarray,
                    rng: np.random.Generator):
    """Returns beta (G,D), program_gene_idx (list of (k,) idx), carrier_masks (list of (N,) bool)."""
    G, D = len(gene_names), cfg.n_disease_programs
    needed = D * cfg.genes_per_program
    if needed > G:
        raise ValueError(f"Need {needed} disease genes, only {G} QC-passed available.")

    # Disjoint gene sets across programs (sample without replacement).
    pool = rng.choice(G, size=needed, replace=False)
    program_gene_idx = [pool[l * cfg.genes_per_program : (l + 1) * cfg.genes_per_program]
                        for l in range(D)]

    beta = np.zeros((G, D), dtype=np.float64)
    for l, g_idx in enumerate(program_gene_idx):
        delta = rng.uniform(cfg.delta_min, cfg.delta_max, size=g_idx.size)
        # v1 up-only: all signs +1, so beta entries are |delta|.
        beta[g_idx, l] = delta

    if cfg.carrier_types is None:
        carrier_masks = [np.ones(cell_type.size, dtype=bool) for _ in range(D)]
    else:
        carrier_masks = []
        for l in range(D):
            ct_set = cfg.carrier_types.get(l)
            carrier_masks.append(
                np.ones(cell_type.size, dtype=bool) if ct_set is None
                else np.isin(cell_type, ct_set)
            )
    return beta, program_gene_idx, carrier_masks


# ============================ Step 5: eta ===============================

def build_eta(mu0: np.ndarray, beta: np.ndarray, carrier_masks: list[np.ndarray],
              g_cell: np.ndarray, lambda_l: float) -> np.ndarray:
    """eta[i,j] = sum_l 1[c(i) in S_l] * beta[j,l] * lambda_l * g_cell[i].
    Vectorized via outer product, exploiting per-program gene sparsity.
    """
    N, G = mu0.shape
    D = beta.shape[1]
    eta = np.zeros((N, G), dtype=np.float64)
    for l in range(D):
        carrier = carrier_masks[l]
        nz = np.nonzero(beta[:, l])[0]
        if not carrier.any() or nz.size == 0:
            continue
        rows = np.where(carrier)[0]
        contrib = (g_cell[rows] * lambda_l)[:, None] * beta[nz, l][None, :]
        eta[np.ix_(rows, nz)] += contrib
    return eta


# ============================ Step 6: Counts ============================

def draw_counts(mu: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """NB(mean=mu, size=phi) draw under  Var = mu + mu^2/phi  (size-theta convention).

    numpy's negative_binomial uses (n, p) with  Var = mu * (1 + mu/n).
    Match by setting n = phi, p = phi / (phi + mu).
    """
    n = np.broadcast_to(phi[None, :], mu.shape)
    p = n / (n + mu)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return rng.negative_binomial(n, p).astype(np.int32)


# ============================ Step 7: Labels ============================

def draw_labels(sex_p: np.ndarray, g_p: np.ndarray, cfg: SimConfig,
                rng: np.random.Generator):
    """y_p ~ Bernoulli(sigmoid(b0 + w_g*g_p + w_s*sex_p)), b0 solved for target prevalence."""
    logit_no_b0 = cfg.w_g * g_p + cfg.w_s * sex_p
    gap = lambda b: float(expit(b + logit_no_b0).mean() - cfg.target_prevalence)
    b0  = float(brentq(gap, -20.0, 20.0))
    y_p = rng.binomial(1, expit(b0 + logit_no_b0)).astype(np.int8)
    return b0, y_p


# ============================ Step 8: Assemble ==========================

def simulate(scdesign_dir: Path, out_dir: Path, cfg: SimConfig,
             celltype_col: str = "majorType") -> ad.AnnData:
    rng = np.random.default_rng(cfg.seed)

    # Step 1
    mu0, phi, cell_type, gene_names, cell_names = load_baseline(scdesign_dir, celltype_col)
    N, G = mu0.shape
    print(f"[baseline] {N} cells x {G} QC-passed genes; "
          f"cell types: {sorted(set(cell_type))}")

    # Step 2
    patient_id = assign_patients(cell_type, cfg.n_patients, rng)

    # Step 3
    sex_p, g_p = patient_vars(cfg.n_patients, rng)
    g_cell, sex_cell = g_p[patient_id], sex_p[patient_id]

    # Step 4
    beta, program_gene_idx, carrier_masks = define_programs(cfg, gene_names, cell_type, rng)
    print(f"[programs] D={cfg.n_disease_programs}, |G_l|={cfg.genes_per_program}, "
          f"delta in [{cfg.delta_min},{cfg.delta_max}], all up-regulated")

    # Step 5
    eta = build_eta(mu0, beta, carrier_masks, g_cell, cfg.lambda_l)
    mu  = mu0 * np.exp(eta)

    # Step 6
    x = draw_counts(mu, phi, rng)
    print(f"[counts] x shape {x.shape}, total counts {x.sum():,}")

    # Step 7
    b0, y_p = draw_labels(sex_p, g_p, cfg, rng)
    y_cell = y_p[patient_id]
    print(f"[labels] b0={b0:.3f}, prevalence(patient)={y_p.mean():.3f}")

    # Step 8: AnnData
    obs = pd.DataFrame({
        "patient_id": patient_id,
        "cell_type":  cell_type,
        "sex":        sex_cell,
        "y":          y_cell,
        "g":          g_cell,
        "x_aux":      sex_cell,          # alias used by DRGP aux/gamma channel
    }, index=cell_names)
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=x, obs=obs, var=var)

    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_dir / "perturbed.h5ad", compression="gzip")
    np.savez_compressed(
        out_dir / "ground_truth.npz",
        beta=beta,
        program_gene_idx_concat=np.concatenate(program_gene_idx),
        program_gene_idx_sizes=np.array([p.size for p in program_gene_idx], dtype=np.int64),
        carrier_masks=np.stack(carrier_masks, axis=0),
        lambda_l=np.float64(cfg.lambda_l),
        g_patient=g_p,
        sex_patient=sex_p,
        y_patient=y_p,
        b0=np.float64(b0),
        gene_names=np.array(gene_names),
        patient_id=patient_id,
        cell_type=cell_type,
    )
    cfg_dict = asdict(cfg)
    cfg_dict["scdesign_dir"] = str(scdesign_dir)
    cfg_dict["celltype_col"] = celltype_col
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[write] {out_dir}/{{perturbed.h5ad,ground_truth.npz,config.json}}")

    run_sanity_checks(adata, beta, program_gene_idx, carrier_masks, y_p, patient_id, cfg)
    return adata


# ============================ Sanity checks =============================

def run_sanity_checks(adata: ad.AnnData, beta: np.ndarray,
                      program_gene_idx: list[np.ndarray],
                      carrier_masks: list[np.ndarray],
                      y_patient: np.ndarray, patient_id: np.ndarray,
                      cfg: SimConfig) -> None:
    print("\n--- sanity checks ---")
    x = np.asarray(adata.X)
    y_cell = adata.obs["y"].to_numpy()
    D, G = beta.shape[1], x.shape[1]

    # 1. Carrier specificity at the gene level (per program).
    for l in range(D):
        carrier = carrier_masks[l]
        pg = program_gene_idx[l]
        case_mask = carrier & (y_cell == 1)
        ctrl_mask = carrier & (y_cell == 0)
        if not (case_mask.any() and ctrl_mask.any()):
            continue
        case_mean = x[case_mask][:, pg].mean(axis=0)
        ctrl_mean = x[ctrl_mask][:, pg].mean(axis=0) + 1e-9
        print(f"  program {l}: median case/control count ratio on program genes "
              f"(carrier cells) = {float(np.median(case_mean / ctrl_mean)):.3f}  (>1 expected, up-only)")

    # 2. Null at background genes.
    all_pg = np.unique(np.concatenate(program_gene_idx))
    non_pg = np.setdiff1d(np.arange(G), all_pg, assume_unique=False)
    case_mean = x[y_cell == 1][:, non_pg].mean(axis=0)
    ctrl_mean = x[y_cell == 0][:, non_pg].mean(axis=0) + 1e-9
    print(f"  background: median case/control ratio = {float(np.median(case_mean / ctrl_mean)):.3f}  (~1.0 expected)")

    # 3. Patient-level program-score AUC: pseudo-bulk over carrier cells per patient.
    for l in range(D):
        carrier = carrier_masks[l]
        pg = program_gene_idx[l]
        sgn = np.sign(beta[pg, l])                                  # all +1 here
        s_cell = (sgn[None, :] * np.log1p(x[:, pg])).sum(axis=1)    # (N,)
        s_p = np.zeros(cfg.n_patients, dtype=np.float64)
        for pid in range(cfg.n_patients):
            sel = (patient_id == pid) & carrier
            if sel.any():
                s_p[pid] = s_cell[sel].mean()
        try:
            auc = roc_auc_score(y_patient, s_p)
        except ValueError:
            auc = float("nan")
        print(f"  program {l}: patient-level AUC(program score, y) = {auc:.3f}")

    # 4. Single-gene LR baseline vs program-score LR (cheap pre-check from the plan).
    #    Per-gene pseudobulk over all that patient's cells, log1p, then pick the best single-gene AUC.
    pseudo = np.zeros((cfg.n_patients, G), dtype=np.float64)
    for pid in range(cfg.n_patients):
        cells = (patient_id == pid)
        pseudo[pid] = np.log1p(x[cells]).mean(axis=0) if cells.any() else 0.0
    aucs = np.array([
        (roc_auc_score(y_patient, pseudo[:, j]) if pseudo[:, j].std() > 0 else 0.5)
        for j in range(G)
    ])
    print(f"  single-gene AUC: median={float(np.median(aucs)):.3f}, "
          f"max={float(aucs.max()):.3f}  (program-score AUC should clearly dominate)")


# ============================ __main__ ==================================

if __name__ == "__main__":
    cfg = SimConfig()
    scdesign_dir = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_covid19_8kcells_10kgenes")
    out_dir      = scdesign_dir / "perturbed_sex_only_v1"
    simulate(scdesign_dir, out_dir, cfg)
