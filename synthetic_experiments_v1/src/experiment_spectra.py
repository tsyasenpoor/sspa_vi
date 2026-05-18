"""
Spectra baseline runner for C1 (and any other config that wants it).

Spectra (Kunes et al. 2023) lives in its own conda env (`spectra`) because
its torch/numpy pin clashes with our `jax_gpu` env. This script is the
standalone entrypoint: it imports the pure-numpy generator + metrics from
src/, fits a single Spectra+L1-LR model on one (cond, seed), and writes a
per-(cond, seed) .npz that mirrors the schema of src/experiment.py but
contains only the `spectra_lr` method.

Notes on the Spectra fit:
  - Spectra's `mimno_coherence_2011` divides by zero when gene-set dict is
    empty/trivial. For synthetic data we have no real pathways, so we feed
    three dummy gene sets of 20 disjoint genes; `clean_gs=False` and
    `filter_sets=False` prevent Spectra from culling them.
  - cell_scores come from a per-cell learned parameter (no encoder), so
    Spectra cannot project new cells. For OOD prediction we use a least-
    squares projection: Theta_new = log1p(Y_new) @ pinv(Beta.T), matching
    the role of `nmf.transform` for the NMF baseline.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate                                            # noqa: E402
from src.metrics import (                                                     # noqa: E402
    hungarian_match, matched_cosine, matched_jaccard_topm,
    v_spearman, v_kendall, precision_at_k, held_out_auroc,
    is_valid_permutation,
)


def fit_spectra_lr(gt, K_fit: int, random_state: int = 0,
                   num_epochs: int = 1000, **_) -> dict:
    """Spectra + L1-LR. Mirrors the fit_*_lr contract in src/baselines.py."""
    import anndata as ad
    from Spectra import Spectra as spc
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    np.random.seed(random_state)
    import torch
    torch.manual_seed(random_state)

    n, p = gt.Y.shape
    var_names = [f"g{i}" for i in range(p)]
    adata = ad.AnnData(X=np.log1p(gt.Y.astype(np.float32)))
    adata.var_names = var_names

    # Three dummy disjoint gene sets keep Spectra's coherence calc from
    # dividing by zero. Size 20 each follows the smoke-test that worked.
    n_dummy = 3
    size = 20
    gs = {
        f"dummy_set_{i}": var_names[i * size:(i + 1) * size]
        for i in range(n_dummy)
    }

    model = spc.est_spectra(
        adata,
        gene_set_dictionary=gs,
        L=K_fit,
        use_cell_types=False,
        use_highly_variable=False,
        label_factors=False,
        clean_gs=False,
        filter_sets=False,
        num_epochs=num_epochs,
        verbose=False,
    )

    Theta_hat = np.asarray(model.return_cell_scores())          # (n, K)
    Beta_hat = np.asarray(model.return_factors()).T             # (p, K)

    feats = np.hstack([Theta_hat, gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)
    full_coef = lr.coef_.ravel()
    v_hat = full_coef[:K_fit]
    gamma_hat = full_coef[K_fit:]

    # Spectra has no native project method. Use the LS projection that
    # corresponds to "given gene factors, what cell scores best reconstruct
    # log1p(Y_new)". Y_log ≈ Theta @ Beta.T  =>  Theta = Y_log @ pinv(Beta.T).
    Beta_pinv = np.linalg.pinv(Beta_hat.T)                      # (p, K)

    def predict(gt_new) -> np.ndarray:
        Y_log_new = np.log1p(gt_new.Y.astype(np.float32))
        Theta_new = Y_log_new @ Beta_pinv                       # (n_new, K)
        return lr.predict_proba(np.hstack([Theta_new, gt_new.X]))[:, 1]

    return {
        "Beta_hat": Beta_hat,
        "Theta_hat": Theta_hat,
        "v_hat": v_hat,
        "gamma_hat": gamma_hat,
        "R_beta": None,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {},
    }


def _eval_one(name: str, fit_out: dict, gt) -> dict:
    out: dict[str, Any] = {"elapsed_s": fit_out.get("elapsed_s", float("nan"))}
    if fit_out.get("Beta_hat") is None:
        out["note"] = "no factorization"
        return out
    Beta_hat = np.asarray(fit_out["Beta_hat"])
    if not np.isfinite(Beta_hat).all():
        out["error"] = "Beta_hat contains non-finite entries"
        return out

    pi = hungarian_match(gt.Beta, Beta_hat)
    out["pi"] = pi
    out["valid_perm"] = bool(is_valid_permutation(pi))
    out["max_abs_beta"] = float(np.max(np.abs(Beta_hat)))

    cos = matched_cosine(gt.Beta, Beta_hat, pi)
    out["cos_per_prog"] = cos.astype(np.float32)
    out["cos_mean"] = float(np.nanmean(cos))

    jac = matched_jaccard_topm(gt.S, Beta_hat, pi, m=50)
    out["jaccard_top50_per_prog"] = jac.astype(np.float32)
    out["jaccard_top50_mean"] = float(np.nanmean(jac))

    v_hat = fit_out.get("v_hat")
    if v_hat is not None and v_hat.size > 0:
        K_rel_true = int(np.sum(gt.v != 0))
        out["v_spearman"] = v_spearman(gt.v, np.asarray(v_hat), pi)
        out["v_kendall"]  = v_kendall(gt.v, np.asarray(v_hat), pi)
        out["precision_at_rel"] = precision_at_k(gt.v, np.asarray(v_hat), pi, K_rel=K_rel_true)
        out["v_hat"] = np.asarray(v_hat, dtype=np.float32)
    gamma_hat = fit_out.get("gamma_hat")
    if gamma_hat is not None:
        out["gamma_hat"] = np.asarray(gamma_hat, dtype=np.float32)

    return out


def run_one(config: dict, cond_idx: int, seed: int, out_dir: str,
            num_epochs: int = 1000) -> str:
    cond = config["conditions"][cond_idx]
    gen_defaults = config["generator_defaults"]

    gen_kwargs = dict(gen_defaults)
    gen_kwargs.update(cond.get("generator_overrides", {}))
    if isinstance(gen_kwargs.get("overlap_pair"), list):
        gen_kwargs["overlap_pair"] = tuple(gen_kwargs["overlap_pair"])
    gen_kwargs["seed"] = seed

    t0 = time.time()
    gt = generate(**gen_kwargs)

    gt_ood = None
    ood_cfg = config.get("ood_test", {})
    if ood_cfg.get("enabled", False):
        ood_kwargs = dict(gen_kwargs)
        ood_kwargs["asthma_rate"] = ood_cfg.get("asthma_rate", 0.6)
        ood_kwargs["n"] = ood_cfg.get("n", min(2 * gen_kwargs["n"], 2000))
        ood_kwargs["seed"] = seed + 10_000
        gt_ood = generate(**ood_kwargs, freeze_params=gt)

    record: dict[str, Any] = {
        "seed": seed,
        "condition_idx": cond_idx,
        "K_fit": int(cond["K_fit"]),
        "pi_label": float(cond.get("pi_label", float("nan"))),
        "v_true": gt.v.astype(np.float32),
        "rel_idx": gt.rel_idx,
        "library_mean": float(gt.Y.sum(axis=1).mean()),
        "phenotype_rate": float(gt.y.mean()),
        "pathway_program_indices": np.asarray([]),
    }

    try:
        fit_out = fit_spectra_lr(gt, K_fit=int(cond["K_fit"]),
                                  random_state=seed, num_epochs=num_epochs)
        metrics = _eval_one("spectra_lr", fit_out, gt)
        if gt_ood is not None and fit_out.get("predict") is not None:
            try:
                scores = np.asarray(fit_out["predict"](gt_ood))
                metrics["ood_auroc"] = held_out_auroc(gt_ood.y, scores)
            except Exception as e:
                metrics["ood_predict_error"] = str(e)
                metrics["ood_auroc"] = float("nan")
        record["spectra_lr"] = metrics
    except Exception:
        record["spectra_lr"] = {"error": traceback.format_exc(),
                                 "elapsed_s": float("nan")}

    record["wall_time_s"] = float(time.time() - t0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cond{cond_idx:03d}_seed{seed:04d}.npz")
    np.savez_compressed(
        out_path,
        **{k: (v if isinstance(v, np.ndarray)
               else np.asarray(v, dtype=object)
               if isinstance(v, dict)
               else np.asarray(v))
           for k, v in record.items()},
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--condition-idx", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-epochs", type=int, default=1000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = args.out_dir or str(
        ROOT / "results" / "raw" / f"{Path(args.config).stem}_spectra"
    )

    out = run_one(cfg, args.condition_idx, args.seed, out_dir,
                  num_epochs=args.num_epochs)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
