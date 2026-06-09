"""Per-dataset draw: composition, activity, perturb+NB-sample, liability, label."""
from __future__ import annotations
import numpy as np
import pandas as pd
import h5py
import anndata as ad
from pathlib import Path
from . import config


def patient_composition(types: np.ndarray, rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Assign each baseline cell to a patient (returned as g_i) and draw D_g.
    Composition is independent of D by construction."""
    N = len(types)
    G = config.N_PATIENTS
    D = np.zeros(G, dtype=np.int8); D[: G // 2] = 1; rng.shuffle(D)
    pi_global = np.bincount(types, minlength=config.T) / N
    alpha = config.DIRICHLET_A0 * pi_global
    pi_g = rng.dirichlet(alpha, size=G)
    cells_per_patient = N // G
    quota = np.floor(pi_g * cells_per_patient).astype(int)
    deficit = cells_per_patient - quota.sum(axis=1)
    patient_order = rng.permutation(G)
    for g in patient_order:
        if deficit[g]:
            # Distribute leftover slots to the type whose floor-rounding most undershot
            residuals = pi_g[g] * cells_per_patient - quota[g]
            for _ in range(int(deficit[g])):
                t = int(np.argmax(residuals))
                quota[g, t] += 1
                residuals[t] -= 1
    g_i = np.full(N, -1, dtype=np.int32)
    by_type = {t: rng.permutation(np.flatnonzero(types == t)).tolist()
               for t in range(config.T)}
    for g in patient_order:
        for t in range(config.T):
            k = int(quota[g, t])
            if k <= 0:
                continue
            avail = min(k, len(by_type[t]))
            sel = by_type[t][:avail]; del by_type[t][:avail]
            g_i[sel] = g
    orphans = np.flatnonzero(g_i < 0)
    if orphans.size:
        for c in orphans:
            counts = np.bincount(g_i[g_i >= 0], minlength=G)
            g_i[c] = int(np.argmin(counts))
    return g_i, D


def activity(truth: dict, types: np.ndarray, g_i: np.ndarray, D: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """theta*_{i,l} (n x L_cols) per §3.4. Decoy is column 0."""
    N = len(types); L = config.L_COLS
    theta_star = np.full((N, L), config.THETA_BASE, dtype=np.float32)
    bbar = config.ALPHA_B / config.LAMBDA_B
    for l in range(L):
        T_l = set(truth["T_ell"][l].tolist())
        responder = np.isin(types, list(T_l))
        active = (D[g_i] == 1) & responder
        if active.any():
            b = rng.gamma(config.ALPHA_B, 1.0 / config.LAMBDA_B, size=active.sum())
            theta_star[active, l] = config.THETA_BASE + (b / bbar).astype(np.float32)
    return theta_star


def _per_gene_size(sigma_mat_gene_by_cell: np.ndarray) -> np.ndarray:
    """Collapse sigma_mat to per-gene NB size, applying the rule from verify_nb_param.
    Reads NB_SIZE_FROM_SIGMA from the gate json at runtime if available, else falls back
    to config (which T3 wrote in)."""
    import json
    gate_path = config.SIM_ROOT / "nb_param_gate.json"
    if gate_path.exists():
        rule = json.loads(gate_path.read_text())["winner"]
    else:
        rule = config.NB_SIZE_FROM_SIGMA
    mean_per_gene = np.nanmean(sigma_mat_gene_by_cell, axis=1)
    median_valid = float(np.nanmedian(mean_per_gene))
    mean_per_gene = np.where(np.isnan(mean_per_gene), median_valid, mean_per_gene)
    if rule == "size":
        return mean_per_gene
    if rule == "dispersion":
        return 1.0 / np.maximum(mean_per_gene, 1e-6)
    raise RuntimeError(f"Unknown NB rule: {rule}")


def perturb_and_sample(mu_cell_by_gene: np.ndarray, size_per_gene: np.ndarray,
                       truth: dict, theta_star: np.ndarray, delta: float,
                       rng: np.random.Generator) -> np.ndarray:
    """log lambda = log mu0 + sum_l delta * u_jl * (theta*_il - theta_base). NB-resample the FULL matrix."""
    N, G = mu_cell_by_gene.shape
    L = config.L_COLS
    carrier_idx = np.unique(np.concatenate([truth["S"][l] for l in range(L)])).astype(np.int64)
    u_c = truth["u"][carrier_idx]                                  # (n_carrier, L)
    A_dev = (theta_star - config.THETA_BASE).astype(np.float32)    # (N, L)
    log_pert_c = delta * (A_dev @ u_c.T)                           # (N, n_carrier)
    log_lam = np.log(np.maximum(mu_cell_by_gene, 1e-6)).astype(np.float32)  # (N, G)
    log_lam[:, carrier_idx] += log_pert_c
    lam = np.exp(log_lam, dtype=np.float32)
    n_arr = np.broadcast_to(size_per_gene[None, :].astype(np.float32), lam.shape)
    p = n_arr / (n_arr + np.maximum(lam, 1e-6))
    return rng.negative_binomial(n=n_arr, p=p).astype(np.int32)


from scipy.optimize import brentq


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def liability_label(theta_star: np.ndarray, v_star: np.ndarray, h2: float,
                    T_ell: np.ndarray, D: np.ndarray, g_i: np.ndarray,
                    types: np.ndarray, rng: np.random.Generator) -> dict:
    """§3.6 - standardize causal theta, build liability, solve alpha, sample y. Decoy excluded."""
    N, L = theta_star.shape
    causal = np.arange(1, L)             # exclude decoy at index 0
    th = theta_star[:, causal].astype(np.float64)
    sd = th.std(axis=0, ddof=0)
    sd[sd < 1e-8] = 1.0
    th_tilde = (th - th.mean(axis=0)) / sd
    lin = th_tilde @ v_star[causal].astype(np.float64)
    c = lin.std(ddof=0)
    if c < 1e-8:
        raise ValueError("Liability has zero variance - check v_star or activity draw")
    liability = (lin / c).astype(np.float32)
    chi = float(np.sqrt((np.pi ** 2 / 3.0) * h2 / (1.0 - h2)))

    causal_resp = np.zeros(N, dtype=bool)
    for l in causal:
        causal_resp |= (D[g_i] == 1) & np.isin(types, T_ell[l])
    target = float(causal_resp.mean())

    def f(alpha):
        return _sigmoid(alpha + chi * liability).mean() - target

    alpha = brentq(f, -20.0, 20.0, xtol=1e-4)
    A_star = (alpha + chi * liability).astype(np.float32)
    pi_true = _sigmoid(A_star).astype(np.float32)
    y = (rng.uniform(size=N) < pi_true).astype(np.int8)
    return dict(liability=liability, A_star=A_star, pi_true=pi_true,
                y=y, alpha=float(alpha), chi=chi)


from .truths import load_truth
from .calibrate import load_delta
from ._runner_utils import _hash_to_int32
import scipy.sparse as sp


_MU_CACHE: dict = {}


def _load_mu_sigma_types():
    if "mu" not in _MU_CACHE:
        with h5py.File(config.NB_PARAMS_H5, "r") as f:
            mu_gc = np.asarray(f["mu_mat"], dtype=np.float32)        # (G, N)
            sg_gc = np.asarray(f["sigma_mat"], dtype=np.float32)
            gene_names = [g.decode() for g in np.asarray(f["gene_names"])]
            cell_names = [c.decode() for c in np.asarray(f["cell_names"])]
        meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
        types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
        _MU_CACHE.update(dict(mu_cg=mu_gc.T, sg_gc=sg_gc, types=types,
                              gene_names=gene_names, cell_names=cell_names))
    return _MU_CACHE


def build_dataset(truth_idx: int, h2: float, r: float, inner_seed: int,
                  is_stability: bool = False) -> ad.AnnData:
    cache = _load_mu_sigma_types()
    mu_cg = cache["mu_cg"]            # (N, G) cells x genes
    types = cache["types"]
    truth = load_truth(truth_idx)
    delta = load_delta(truth_idx, r)
    rng = np.random.default_rng(_hash_to_int32("dataset", truth_idx, h2, r, inner_seed))

    g_i, D = patient_composition(types, rng)
    theta_star = activity(truth, types, g_i, D, rng)
    size_g = _per_gene_size(cache["sg_gc"])
    X = perturb_and_sample(mu_cg, size_g, truth, theta_star, delta, rng)
    lab = liability_label(theta_star, truth["v_star"], h2, truth["T_ell"], D, g_i, types, rng)

    obs = pd.DataFrame({
        "patient_id": [f"P{g:02d}" for g in g_i],
        "cell_type": [config.CELL_TYPES[t] for t in types],
        "y": lab["y"],
        "liability": lab["liability"],
        "pi_true": lab["pi_true"],
    })
    var = pd.DataFrame({"gene": cache["gene_names"]}).set_index("gene")
    A = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    A.uns["S_ell"]      = {str(l): truth["S"][l].astype(np.int64) for l in range(config.L_COLS)}
    A.uns["u"]          = truth["u"]
    A.uns["delta"]      = float(delta)
    A.uns["T_ell"]      = {str(l): truth["T_ell"][l].astype(np.int8) for l in range(config.L_COLS)}
    A.uns["v_star"]     = truth["v_star"]
    A.uns["theta_star"] = theta_star
    A.uns["mask_M"]     = truth["mask_M"]
    A.uns["alpha"]      = lab["alpha"]
    A.uns["chi"]        = lab["chi"]
    A.uns["truth_idx"]  = int(truth_idx)
    A.uns["h2"]         = float(h2)
    A.uns["r"]          = float(r)
    A.uns["inner_seed"] = int(inner_seed)
    A.uns["is_stability"] = bool(is_stability)
    return A


def dataset_path(truth_idx: int, h2: float, r: float, inner_seed: int,
                 is_stability: bool = False) -> Path:
    sub = "stability_datasets" if is_stability else "datasets"
    name = f"truth{truth_idx}_h{h2}_r{r}_seed{inner_seed}.h5ad"
    return config.SIM_ROOT / sub / name


def write_dataset(truth_idx: int, h2: float, r: float, inner_seed: int,
                  is_stability: bool = False) -> Path:
    A = build_dataset(truth_idx, h2, r, inner_seed, is_stability=is_stability)
    p = dataset_path(truth_idx, h2, r, inner_seed, is_stability=is_stability)
    p.parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(p, compression="gzip")
    return p
