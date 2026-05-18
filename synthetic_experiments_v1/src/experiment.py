"""
One experimental unit: (config x condition_idx x seed).

Generates one synthetic dataset, fits each method listed in the config,
applies Hungarian matching, computes metrics, optionally evaluates on a
frozen-params OOD cohort, saves a single .npz per (cond, seed).

Used by slurm/run_sweep.sh as the SLURM array body:
    python -u -m src.experiment --config configs/A1_factor_recovery.yaml \
        --condition-idx 7 --seed 12

Output:
    <out_dir>/cond{cond:03d}_seed{seed:04d}.npz
with all per-method scalars, the per-program cosine vector, pi, and a
small subset of posteriors (R_beta for DRGP). Heavy quantities (Beta_hat,
Theta_hat) are saved only when --save-posteriors is passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", "/labs/Aguiar/SSPA_BRAY/BRay")

from src.generator import generate                                            # noqa: E402
from src.baselines import METHOD_FITTERS, fit_drgp, fit_nmf_lr, fit_pca_lr, fit_plain_lr  # noqa: E402
from src.metrics import (                                                     # noqa: E402
    hungarian_match, matched_cosine, matched_jaccard_topm,
    support_auprc, fdr_at_threshold, v_spearman, v_kendall,
    precision_at_k, held_out_auroc, is_valid_permutation,
)


def _build_drgp_kwargs(drgp_overrides: dict, seed: int) -> dict:
    kw = dict(drgp_overrides) if drgp_overrides else {}
    kw.setdefault("max_iter", 600)
    kw.setdefault("check_freq", 5)
    kw.setdefault("tol", 1e-3)
    kw.setdefault("early_stopping", "heldout_ll")
    kw.setdefault("b_v", 1.0)
    kw.setdefault("regression_weight", 1.0)
    kw.setdefault("use_intercept", True)
    kw.setdefault("val_frac", 0.1)
    kw["random_state"] = seed
    return kw


def _eval_one_method(name: str, fit_out: dict, gt) -> dict:
    """Compute the standard metric bundle for a fitted method."""
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

    R_beta = fit_out.get("R_beta")
    if R_beta is not None:
        R_beta = np.asarray(R_beta)
        out["support_auprc"] = support_auprc(gt.S, R_beta, pi)
        out["fdr_at_0p5"] = fdr_at_threshold(gt.S, R_beta, pi, thr=0.5)

    v_hat = fit_out.get("v_hat")
    if v_hat is not None and v_hat.size > 0:
        K_rel_true = int(np.sum(gt.v != 0))
        out["v_spearman"] = v_spearman(gt.v, np.asarray(v_hat), pi)
        out["v_kendall"]  = v_kendall(gt.v, np.asarray(v_hat), pi)
        out["precision_at_rel"] = precision_at_k(gt.v, np.asarray(v_hat), pi, K_rel=K_rel_true)
        # Always save the raw weights -- tiny (~K_fit + q floats) and
        # needed for downstream analyses (B2 indirect effect, etc.)
        out["v_hat"] = np.asarray(v_hat, dtype=np.float32)
    gamma_hat = fit_out.get("gamma_hat")
    if gamma_hat is not None:
        out["gamma_hat"] = np.asarray(gamma_hat, dtype=np.float32)

    return out


def run_one(config: dict, cond_idx: int, seed: int, out_dir: str,
            save_posteriors: bool = False, verbose: bool = False) -> str:
    cond = config["conditions"][cond_idx]
    methods = config["methods"]
    gen_defaults = config["generator_defaults"]
    # Config can force save_posteriors=true for experiments that need Theta/Beta
    # downstream (e.g. B2 mediation, where the analyzer regresses theta_hat on X)
    if config.get("save_posteriors"):
        save_posteriors = True

    gen_kwargs = dict(gen_defaults)
    gen_kwargs.update(cond.get("generator_overrides", {}))
    # YAML can't represent dict-with-tuple-key, so we accept delta_entries
    # (list of [k, ell, val] triples) and convert to the dict form
    # the generator expects.
    if "delta_entries" in gen_kwargs:
        entries = gen_kwargs.pop("delta_entries")
        gen_kwargs["delta_spec"] = {(int(k), int(ell)): float(v) for k, ell, v in entries}
    # overlap_pair must be a tuple for the generator (YAML gives a list)
    if isinstance(gen_kwargs.get("overlap_pair"), list):
        gen_kwargs["overlap_pair"] = tuple(gen_kwargs["overlap_pair"])
    gen_kwargs["seed"] = seed

    t0 = time.time()
    # If the config requests it, freeze the structural ground-truth (B, v, alpha,
    # Delta, S, xi_0) to a canonical seed so only the count noise + X covariates
    # vary across seeds. This is how N3 actually measures cross-seed stability.
    freeze_structural_seed = config.get("freeze_structural_seed", None)
    if freeze_structural_seed is not None and int(freeze_structural_seed) != seed:
        anchor_kwargs = dict(gen_kwargs)
        anchor_kwargs["seed"] = int(freeze_structural_seed)
        gt_anchor = generate(**anchor_kwargs)
        gt = generate(**gen_kwargs, freeze_params=gt_anchor)
    else:
        gt = generate(**gen_kwargs)

    # Optional OOD cohort
    gt_ood = None
    ood_cfg = config.get("ood_test", {})
    if ood_cfg.get("enabled", False):
        ood_kwargs = dict(gen_kwargs)
        ood_kwargs["asthma_rate"] = ood_cfg.get("asthma_rate", 0.6)
        ood_kwargs["n"] = ood_cfg.get("n", min(2 * gen_kwargs["n"], 2000))
        ood_kwargs["seed"] = seed + 10_000
        gt_ood = generate(**ood_kwargs, freeze_params=gt)

    # Per-condition method override (e.g. B1 conditions specify which drgp variant to run)
    cond_methods = cond.get("methods") or methods

    # Build pathway_mask if the condition specifies program indices (for masked/combined modes)
    # The pathway_mask is (p, K_path) binary; K_path = len(pathway_program_indices).
    # We construct it from the ground-truth S so each pathway column corresponds to
    # the support of one true program.
    pathway_indices = cond.get("pathway_program_indices")
    pathway_mask = None
    if pathway_indices is not None:
        pathway_indices = [int(i) for i in pathway_indices]
        # CAVI expects pathway_mask in (n_pathways, p) layout (it transposes internally).
        # gt.S is (p, K_true), so slicing columns then transposing gives (K_path, p).
        pathway_mask = gt.S[:, pathway_indices].T.astype(np.int32)
        # Optional corruption (D3): for each pathway row, swap a fraction `eta`
        # of in-mask genes with the same number of out-of-mask genes.
        # Preserves the row's column count so n_genes_per_pathway is unchanged.
        eta = float(cond.get("mask_corruption_eta", 0.0))
        if eta > 0.0:
            corruption_rng = np.random.default_rng(seed + 7919)
            corrupted = pathway_mask.copy()
            p_total = corrupted.shape[1]
            for k_row in range(corrupted.shape[0]):
                in_mask = np.flatnonzero(corrupted[k_row])
                out_mask = np.flatnonzero(corrupted[k_row] == 0)
                n_swap = int(round(eta * in_mask.size))
                if n_swap <= 0 or out_mask.size < n_swap or in_mask.size < n_swap:
                    continue
                drop = corruption_rng.choice(in_mask, size=n_swap, replace=False)
                add  = corruption_rng.choice(out_mask, size=n_swap, replace=False)
                corrupted[k_row, drop] = 0
                corrupted[k_row, add] = 1
            pathway_mask = corrupted
    n_pathway_factors = cond.get("n_pathway_factors")

    record: dict[str, Any] = {
        "seed": seed,
        "condition_idx": cond_idx,
        "K_fit": int(cond["K_fit"]),
        "pi_label": float(cond.get("pi_label", float("nan"))),
        "v_true": gt.v.astype(np.float32),
        "rel_idx": gt.rel_idx,
        "library_mean": float(gt.Y.sum(axis=1).mean()),
        "phenotype_rate": float(gt.y.mean()),
        "pathway_program_indices": np.asarray(pathway_indices) if pathway_indices is not None else np.asarray([]),
    }

    drgp_kwargs = _build_drgp_kwargs(config.get("drgp_overrides", {}), seed)

    for label in cond_methods:
        if label not in METHOD_FITTERS:
            record[label] = {"error": f"unknown method '{label}'"}
            continue
        try:
            if label.startswith("drgp"):
                kw = {**drgp_kwargs, "verbose": verbose}
                # For masked / combined modes, inject pathway_mask + n_pathway_factors
                if label in ("drgp_masked", "drgp_combined"):
                    if pathway_mask is None:
                        raise ValueError(f"{label} requires pathway_program_indices in condition")
                    kw["pathway_mask"] = pathway_mask
                if label == "drgp_combined":
                    if n_pathway_factors is None:
                        raise ValueError("drgp_combined requires n_pathway_factors in condition")
                    kw["n_pathway_factors"] = int(n_pathway_factors)
                fit_out = METHOD_FITTERS[label](gt, K_fit=int(cond["K_fit"]), **kw)
            else:
                fit_out = METHOD_FITTERS[label](gt, K_fit=int(cond["K_fit"]), random_state=seed)
            metrics = _eval_one_method(label, fit_out, gt)
            if gt_ood is not None and fit_out.get("predict") is not None:
                try:
                    scores = np.asarray(fit_out["predict"](gt_ood))
                    metrics["ood_auroc"] = held_out_auroc(gt_ood.y, scores)
                except Exception as e:
                    metrics["ood_predict_error"] = str(e)
                    metrics["ood_auroc"] = float("nan")
            if save_posteriors:
                metrics["Beta_hat"] = (np.asarray(fit_out["Beta_hat"]).astype(np.float32)
                                       if fit_out.get("Beta_hat") is not None else None)
                metrics["Theta_hat"] = (np.asarray(fit_out["Theta_hat"]).astype(np.float32)
                                        if fit_out.get("Theta_hat") is not None else None)
                if fit_out.get("R_beta") is not None:
                    metrics["R_beta"] = np.asarray(fit_out["R_beta"]).astype(np.float32)
            # Always save DRGP r_beta even without full posteriors — small, useful for A2
            if label.startswith("drgp") and not save_posteriors and fit_out.get("R_beta") is not None:
                metrics["R_beta"] = np.asarray(fit_out["R_beta"]).astype(np.float32)
            record[label] = metrics
        except Exception:
            record[label] = {"error": traceback.format_exc(),
                              "elapsed_s": float("nan")}

    record["wall_time_s"] = float(time.time() - t0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cond{cond_idx:03d}_seed{seed:04d}.npz")
    # numpy savez requires arrays; wrap nested dicts as object arrays
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
    ap.add_argument("--out-dir", default=None,
                    help="Default: <repo>/results/raw/<config_stem>/")
    ap.add_argument("--save-posteriors", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = args.out_dir or str(ROOT / "results" / "raw" / Path(args.config).stem)

    out = run_one(cfg, args.condition_idx, args.seed, out_dir,
                  save_posteriors=args.save_posteriors, verbose=args.verbose)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
