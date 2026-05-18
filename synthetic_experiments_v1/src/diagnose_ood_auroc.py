"""
Diagnose the OOD AUROC gap between DRGP and NMF.

Fits DRGP and NMF on one synthetic dataset at default scale, then evaluates
multiple OOD score variants:

  DRGP scores:
    A. predict_proba           (probit-shrunk; current pipeline)
    B. raw E[logit]            (E_theta @ mu_v + X_aux @ mu_gamma, no probit)
    C. mean_proba(no aux)      (E_theta @ mu_v only, no aux covariates)
    D. mean_proba(aux only)    (X_aux @ mu_gamma only, no theta)

  NMF score:
    E. lr.predict_proba(Theta_new + X_aux)

Per-sample probit shrinkage Var[logit] is recorded; we plot raw vs probit
to confirm the rank-changing hypothesis.

Output:
  figures/diagnostic/ood_auroc_diagnostic.png
  results/diagnostic/ood_auroc_diagnostic.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", "/labs/Aguiar/SSPA_BRAY/BRay")
sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")

from src.generator import generate                                  # noqa: E402
from src.baselines import fit_drgp, fit_nmf_lr, _coo, _to_numpy     # noqa: E402
from src.metrics import held_out_auroc                              # noqa: E402


def drgp_score_variants(model, gt_ood):
    """Return dict of {score_name: (n,) array of float scores}."""
    n_new = gt_ood.Y.shape[0]
    X_aux_new = model._prepend_intercept(
        np.asarray(gt_ood.X, dtype=np.float32), n=n_new)
    from VariationalInference.jax_backend import to_device
    X_aux_dev = to_device(X_aux_new)

    a_th, b_th = model._infer_theta_sparse(
        _coo(gt_ood.Y), n_new=n_new, n_iter=20, X_aux_new=X_aux_dev,
    )
    E_theta = _to_numpy(a_th / b_th)                                # (n_new, K)
    Var_theta = _to_numpy(a_th / (b_th ** 2))                       # (n_new, K)
    mu_v = _to_numpy(model.mu_v)                                    # (kappa=1, K)
    sigma_v = _to_numpy(model.sigma_v_diag)                         # (kappa=1, K)
    mu_gamma = _to_numpy(model.mu_gamma)                            # (1, p_aux+intercept)
    X_aux_np = _to_numpy(X_aux_dev)                                 # includes intercept col

    logits = E_theta @ mu_v.T                                       # (n, 1)
    if mu_gamma.shape[1] > 0:
        logits = logits + X_aux_np @ mu_gamma.T

    E_v_sq = mu_v ** 2 + sigma_v
    var_logits = Var_theta @ E_v_sq.T + (E_theta ** 2) @ sigma_v.T
    if mu_gamma.shape[1] > 0:
        # gamma posterior variance, diagonal: model.Sigma_gamma[0] is (p_aux, p_aux)
        gamma_var = np.diag(_to_numpy(model.Sigma_gamma[0]))         # (p_aux+1,)
        var_logits = var_logits + (X_aux_np ** 2) @ gamma_var.reshape(-1, 1)

    scale = np.sqrt(1.0 + (np.pi / 3.0) * var_logits)
    probit = 1.0 / (1.0 + np.exp(-logits / scale))

    # Ablations: theta-only, aux-only
    logits_theta_only = (E_theta @ mu_v.T).ravel()
    if mu_gamma.shape[1] > 0:
        logits_aux_only = (X_aux_np @ mu_gamma.T).ravel()
    else:
        logits_aux_only = np.zeros(n_new)

    return {
        "A_probit": probit.ravel(),
        "B_raw_logit": logits.ravel(),
        "C_theta_only": logits_theta_only,
        "D_aux_only": logits_aux_only,
        "var_logits": var_logits.ravel(),
        "E_theta": E_theta,
    }


def main():
    out_root = Path("/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1")
    fig_dir = out_root / "figures" / "diagnostic"
    res_dir = out_root / "results" / "diagnostic"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Generating default-scale dataset (n=500, p=5000, K=10) ...")
    gen_kwargs = dict(n=500, p=5000, K_true=10, K_rel=3, q=5,
                      m_choices=[30, 50, 80], seed=0)
    gt = generate(**gen_kwargs)
    gt_ood = generate(**{**gen_kwargs, "asthma_rate": 0.6, "n": 1000,
                          "seed": 10_000}, freeze_params=gt)
    print(f"  train: n={gt.Y.shape[0]} p(asthma)={gt.X[:,2].mean():.3f} y={gt.y.mean():.3f}")
    print(f"  ood:   n={gt_ood.Y.shape[0]} p(asthma)={gt_ood.X[:,2].mean():.3f} y={gt_ood.y.mean():.3f}")

    runs = [
        ("DRGP_base",        dict(sigma_gamma=1.0, val_frac=0.1, early_stopping="heldout_ll")),
        ("DRGP_tightGamma",  dict(sigma_gamma=0.1, val_frac=0.1, early_stopping="heldout_ll")),
        ("DRGP_fullN",       dict(sigma_gamma=1.0, val_frac=0.0, early_stopping="none")),
        ("DRGP_both",        dict(sigma_gamma=0.1, val_frac=0.0, early_stopping="none")),
    ]
    fitted = {}
    for label, overrides in runs:
        print(f"\nFitting {label} ({overrides}) ...")
        fit = fit_drgp(gt, K_fit=10, mode="unmasked", max_iter=600,
                       random_state=0, verbose=False, **overrides)
        print(f"  done in {fit['elapsed_s']:.1f}s")
        fitted[label] = fit

    print("\nFitting NMF + L1-LR ...")
    nmf = fit_nmf_lr(gt, K_fit=10, random_state=0)
    print(f"  done in {nmf['elapsed_s']:.1f}s")

    print("\nComputing OOD scores ...")
    base_scores = drgp_score_variants(fitted["DRGP_base"]["extra"]["model"], gt_ood)
    drgp_scores = base_scores
    model = fitted["DRGP_base"]["extra"]["model"]
    nmf_proba = nmf["predict"](gt_ood)

    aurocs = {}
    aurocs["A_DRGP_predict_proba (probit-shrunk)"] = held_out_auroc(gt_ood.y, base_scores["A_probit"])
    aurocs["B_DRGP_raw_logit (no probit)"]        = held_out_auroc(gt_ood.y, base_scores["B_raw_logit"])
    aurocs["C_DRGP_theta_only"]                    = held_out_auroc(gt_ood.y, base_scores["C_theta_only"])
    aurocs["D_DRGP_aux_only"]                      = held_out_auroc(gt_ood.y, base_scores["D_aux_only"])
    aurocs["E_NMF_LR_predict_proba"]               = held_out_auroc(gt_ood.y, nmf_proba)
    # Probes
    for label in ("DRGP_tightGamma", "DRGP_fullN", "DRGP_both"):
        scores = drgp_score_variants(fitted[label]["extra"]["model"], gt_ood)
        aurocs[f"F_{label}_predict_proba"]  = held_out_auroc(gt_ood.y, scores["A_probit"])
        aurocs[f"G_{label}_theta_only"]     = held_out_auroc(gt_ood.y, scores["C_theta_only"])

    print("\nOOD AUROC comparison:")
    for k, v in aurocs.items():
        print(f"  {k:50s}  {v:.4f}")

    # Save text report
    with open(res_dir / "ood_auroc_diagnostic.txt", "w") as f:
        f.write("OOD AUROC diagnostic (default scale, seed=0)\n\n")
        for k, v in aurocs.items():
            f.write(f"  {k:60s}  {v:.4f}\n")
        for label, fit in fitted.items():
            f.write(f"\n{label} v_hat = "
                    f"{np.round(np.asarray(fit['v_hat']), 3).tolist()}\n")
            f.write(f"{label} gamma_hat = "
                    f"{np.round(np.asarray(fit['gamma_hat']), 3).tolist()}\n")
        f.write(f"\nNMF L1-LR factor coef = {np.round(np.asarray(nmf['v_hat']), 3).tolist()}\n")
        f.write(f"NMF L1-LR aux coef    = {np.round(np.asarray(nmf['gamma_hat']), 3).tolist()}\n")
        f.write(f"\nVar[logit] per-sample summary (DRGP_base, OOD):\n")
        vl = drgp_scores["var_logits"]
        f.write(f"  min={vl.min():.3f}  p25={np.quantile(vl,0.25):.3f}  "
                f"median={np.median(vl):.3f}  p75={np.quantile(vl,0.75):.3f}  "
                f"max={vl.max():.3f}\n")

    # Plots: raw logit vs probit-shrunk score, colored by y_ood
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    y = gt_ood.y
    pos = y == 1; neg = y == 0

    ax = axes[0]
    ax.scatter(drgp_scores["B_raw_logit"][neg], drgp_scores["A_probit"][neg],
               s=8, alpha=0.4, color="steelblue", label="y=0")
    ax.scatter(drgp_scores["B_raw_logit"][pos], drgp_scores["A_probit"][pos],
               s=8, alpha=0.4, color="salmon", label="y=1")
    ax.set_xlabel("DRGP raw E[logit] (B)"); ax.set_ylabel("DRGP probit-shrunk (A)")
    ax.set_title(f"raw logit vs probit\nAUROC: raw={aurocs['B_DRGP_raw_logit (no probit)']:.3f}  "
                  f"probit={aurocs['A_DRGP_predict_proba (probit-shrunk)']:.3f}")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.scatter(drgp_scores["var_logits"][neg], drgp_scores["B_raw_logit"][neg],
               s=8, alpha=0.4, color="steelblue", label="y=0")
    ax.scatter(drgp_scores["var_logits"][pos], drgp_scores["B_raw_logit"][pos],
               s=8, alpha=0.4, color="salmon", label="y=1")
    ax.set_xlabel("Var[logit] (probit denominator)"); ax.set_ylabel("DRGP raw E[logit]")
    ax.set_title("does Var[logit] vary?")
    ax.legend(loc="upper left")

    ax = axes[2]
    names = ["DRGP\nprobit", "DRGP\nraw logit", "DRGP\ntheta only", "DRGP\naux only", "NMF\nL1-LR"]
    vals = [aurocs[k] for k in [
        "A_DRGP_predict_proba (probit-shrunk)",
        "B_DRGP_raw_logit (no probit)",
        "C_DRGP_theta_only",
        "D_DRGP_aux_only",
        "E_NMF_LR_predict_proba",
    ]]
    colors = ["#9ab0c8", "#5577aa", "#caa", "#cca", "#a3c39d"]
    ax.bar(names, vals, color=colors)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("OOD AUROC")
    ax.set_title("OOD AUROC by score variant")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "ood_auroc_diagnostic.png", dpi=180)
    plt.close()

    print(f"\nFigure: {fig_dir}/ood_auroc_diagnostic.png")
    print(f"Report: {res_dir}/ood_auroc_diagnostic.txt")


if __name__ == "__main__":
    main()
