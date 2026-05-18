"""
GSVA baseline runner for the synthetic C1 (and other K_fit=K_true configs).

GSVA needs an explicit pathway dictionary, so on synthetic data we feed it
the ground-truth program supports `gt.S` (p × K_true binary) as the oracle
gene-set dictionary. This is the "perfect mask" upper bound for any
pathway-aggregation method that does NOT learn the loadings — directly
comparable to drgp_masked (which also gets the perfect mask but DOES learn
real-valued loadings on top).

Output schema mirrors src/experiment.py but contains only the gsva_lr
method. Output dir: results/raw/<config_stem>_gsva/.
Use the `gsva_env` conda env (gseapy is pinned there).

Notes:
  - gseapy.gsva expects a (gene × sample) matrix. We pass gt.Y.T after a
    light log1p transform so per-gene ranks are well-behaved on count data.
  - gene names are synthesised as g0..g{p-1}; gene_sets are built from gt.S.
  - For OOD prediction we re-run GSVA on gt_new with the same gene_sets,
    then apply the fitted L1-LR. This is the natural "frozen pathway
    library" projection — no LS shortcut needed.
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
import pandas as pd
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


def _gsva_enrichment(Y: np.ndarray, S: np.ndarray, threads: int = 4) -> np.ndarray:
    """Run GSVA on (n × p) counts with binary support S (p × K).
    Returns (n × K) enrichment scores, columns in original program order."""
    import gseapy
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items
    n, p = Y.shape
    K = S.shape[1]
    gene_names = [f"g{i}" for i in range(p)]
    cell_names = [f"c{j}" for j in range(n)]
    expr = pd.DataFrame(
        np.log1p(Y.astype(np.float32)).T,           # (p × n) gene × sample
        index=gene_names, columns=cell_names)
    # Build gene_sets dict ordered P0..P{K-1}; keep ordering deterministic
    # so the returned enrichment columns align with the original program idx.
    gs = {}
    for k in range(K):
        members = [gene_names[i] for i in np.flatnonzero(S[:, k])]
        # gseapy.gsva requires ≥2 genes per set
        if len(members) < 2:
            members = members + gene_names[:2]
        gs[f"P{k}"] = members
    res = gseapy.gsva(
        expr, gene_sets=gs, min_size=2, max_size=max(2000, p),
        threads=threads, outdir=None, verbose=False,
    )
    mat = res.res2d.pivot(index="Term", columns="Name", values="ES")
    # Re-order rows to P0..P{K-1} and columns to c0..c{n-1}
    mat = mat.reindex(index=[f"P{k}" for k in range(K)],
                       columns=cell_names)
    return mat.values.T.astype(np.float32)          # (n × K)


def fit_gsva_lr(gt, K_fit: int, random_state: int = 0,
                threads: int = 4, **_) -> dict:
    """GSVA on oracle gene sets + L1-LR. Returns standard fit dict."""
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    K_true = gt.S.shape[1]
    if K_fit != K_true:
        # Synthetic GSVA is only meaningful when K_fit == K_true (we feed
        # the K_true ground-truth gene sets). For other configs, pad/truncate
        # would be ambiguous. Bail loudly so the experiment runner records
        # an explicit error instead of silently producing wrong-shape output.
        raise ValueError(
            f"gsva_lr requires K_fit == K_true (got K_fit={K_fit}, "
            f"K_true={K_true}); use a different config or extend the wrapper.")

    Theta_hat = _gsva_enrichment(gt.Y, gt.S, threads=threads)    # (n × K_true)
    Beta_hat = gt.S.astype(np.float32)                            # (p × K_true), frozen

    feats = np.hstack([Theta_hat, gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)
    full_coef = lr.coef_.ravel()
    v_hat = full_coef[:K_true]
    gamma_hat = full_coef[K_true:]

    # Capture the SAME gene-set dictionary for the OOD projection.
    S_frozen = gt.S.copy()

    def predict(gt_new) -> np.ndarray:
        Theta_new = _gsva_enrichment(gt_new.Y, S_frozen, threads=threads)
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
            threads: int = 4) -> str:
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
        fit_out = fit_gsva_lr(gt, K_fit=int(cond["K_fit"]),
                              random_state=seed, threads=threads)
        metrics = _eval_one("gsva_lr", fit_out, gt)
        if gt_ood is not None and fit_out.get("predict") is not None:
            try:
                scores = np.asarray(fit_out["predict"](gt_ood))
                metrics["ood_auroc"] = held_out_auroc(gt_ood.y, scores)
            except Exception as e:
                metrics["ood_predict_error"] = str(e)
                metrics["ood_auroc"] = float("nan")
        record["gsva_lr"] = metrics
    except Exception:
        record["gsva_lr"] = {"error": traceback.format_exc(),
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
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = args.out_dir or str(
        ROOT / "results" / "raw" / f"{Path(args.config).stem}_gsva"
    )

    out = run_one(cfg, args.condition_idx, args.seed, out_dir,
                  threads=args.threads)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
