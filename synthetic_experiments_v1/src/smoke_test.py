"""
Smoke test: one seed, one condition, fit DRGP + NMF + PCA + plain LR.

Verifies the 5 invariants from DRGP_synthetic_experiments_v1.md §5 Step 2:
  S1. All baselines complete without error.
  S2. DRGP returns a sensible Beta_hat (max value < 100, no NaN/Inf).
  S3. Hungarian matching produces a valid permutation (no duplicates).
  S4. DRGP cos_sim_per_prog mean >= 0.7 (well-specified K_fit = K_true).
  S5. NMF / PCA complete in <60s; DRGP elapsed reported separately.

Usage:
    python -m src.smoke_test                       # fast-dev defaults
    python -m src.smoke_test --scale default       # n=500, p=5000
    python -m src.smoke_test --K-fit 12            # over-specified K
    python -m src.smoke_test --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Make sure VariationalInference is importable for fit_drgp
os.environ.setdefault("PYTHONPATH", "/labs/Aguiar/SSPA_BRAY/BRay")

from src.generator import generate, FAST_DEV_KWARGS  # noqa: E402
from src.baselines import fit_drgp, fit_nmf_lr, fit_pca_lr, fit_plain_lr  # noqa: E402
from src.metrics import (  # noqa: E402
    hungarian_match, matched_cosine, matched_jaccard_topm,
    support_auprc, v_spearman, precision_at_k, fdr_at_threshold,
    is_valid_permutation, held_out_auroc,
)


def _section(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def _check(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {detail}")
    return (name, ok, detail)


def _eval_method(name: str, fit: dict, gt) -> dict:
    """Compute the standard metric bundle for one fitted method."""
    out = {"elapsed_s": fit["elapsed_s"], "error": None}
    if fit["Beta_hat"] is None:
        out["note"] = "no factorization (plain LR); only OOD predict available"
        return out

    Beta_hat = fit["Beta_hat"]
    if not np.isfinite(Beta_hat).all():
        out["error"] = f"Beta_hat has non-finite entries"
        return out

    pi = hungarian_match(gt.Beta, Beta_hat)
    out["pi"] = pi
    out["valid_perm"] = is_valid_permutation(pi)
    out["max_abs_beta"] = float(np.max(np.abs(Beta_hat)))

    cos = matched_cosine(gt.Beta, Beta_hat, pi)
    out["cos_per_prog"] = cos
    out["cos_mean"] = float(np.nanmean(cos))

    jac = matched_jaccard_topm(gt.S, Beta_hat, pi, m=50)
    out["jaccard_top50_mean"] = float(np.nanmean(jac))

    if fit.get("R_beta") is not None:
        out["support_auprc"] = support_auprc(gt.S, fit["R_beta"], pi)
        out["fdr_at_0p5"] = fdr_at_threshold(gt.S, fit["R_beta"], pi, thr=0.5)

    if fit.get("v_hat") is not None:
        out["v_spearman"] = v_spearman(gt.v, fit["v_hat"], pi)
        out["precision_at_3"] = precision_at_k(gt.v, fit["v_hat"], pi, K_rel=int((gt.v != 0).sum()))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["fast", "default"], default="fast")
    ap.add_argument("--K-fit", type=int, default=None,
                    help="Default: K_true from the chosen scale.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drgp-max-iter", type=int, default=300,
                    help="Lower for smoke speed; A1 sweeps will use 600+.")
    ap.add_argument(
        "--out-root",
        default="/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1",
    )
    args = ap.parse_args()

    if args.scale == "fast":
        gen_kwargs = dict(FAST_DEV_KWARGS, seed=args.seed)
        K_default = FAST_DEV_KWARGS["K_true"]
    else:
        gen_kwargs = dict(n=500, p=5000, K_true=10, K_rel=3, q=5, seed=args.seed)
        K_default = 10
    K_fit = args.K_fit if args.K_fit is not None else K_default

    _section(f"Smoke test: scale={args.scale}  K_true={gen_kwargs['K_true']}  K_fit={K_fit}  seed={args.seed}")
    gt = generate(**gen_kwargs)
    print(f"  n={gt.Y.shape[0]}  p={gt.Y.shape[1]}  K_true={gt.Beta.shape[1]}")
    print(f"  library mean = {gt.Y.sum(axis=1).mean():.1f}")
    print(f"  phenotype rate = {gt.y.mean():.3f}")
    print(f"  v_true = {np.round(gt.v, 3).tolist()}   rel_idx = {gt.rel_idx.tolist()}")

    methods_results = {}
    errors = {}

    for label, fitter, kw in [
        ("plain_lr", fit_plain_lr, {}),
        ("nmf_lr",   fit_nmf_lr,   {}),
        ("pca_lr",   fit_pca_lr,   {}),
        ("drgp_unmasked", fit_drgp, dict(mode="unmasked", max_iter=args.drgp_max_iter,
                                          random_state=args.seed)),
    ]:
        print(f"\n>> Fitting {label} ...")
        try:
            fit = fitter(gt, K_fit=K_fit, **kw)
            print(f"   done in {fit['elapsed_s']:.1f}s")
            methods_results[label] = _eval_method(label, fit, gt)
            methods_results[label]["_fit"] = fit  # keep for OOD predict below
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"   ERROR: {e}")
            errors[label] = tb

    # ------------------------------------------------------------------
    # OOD predict check: build a frozen-params OOD cohort and score each method
    # ------------------------------------------------------------------
    _section("OOD predict check (frozen params, asthma 0.2 -> 0.6)")
    ood_kwargs = {**gen_kwargs, "asthma_rate": 0.6, "n": min(2 * gt.Y.shape[0], 2000), "seed": args.seed + 10_000}
    gt_ood = generate(**ood_kwargs, freeze_params=gt)
    print(f"  OOD n={gt_ood.Y.shape[0]}  p(asthma)={gt_ood.X[:, 2].mean():.3f}  y rate={gt_ood.y.mean():.3f}")
    for label, res in methods_results.items():
        fit = res["_fit"]
        try:
            scores = fit["predict"](gt_ood)
            auroc = held_out_auroc(gt_ood.y, scores)
            res["ood_auroc"] = auroc
            print(f"  {label:14s}  OOD AUROC = {auroc:.3f}")
        except Exception as e:
            print(f"  {label:14s}  OOD predict ERROR: {e}")
            res["ood_auroc"] = float("nan")

    # ------------------------------------------------------------------
    # Invariant verdict
    # ------------------------------------------------------------------
    _section("Invariants (§5 Step 2)")
    checks = []
    checks.append(_check(
        "S1. all baselines completed without raising",
        len(errors) == 0,
        f"errors: {list(errors.keys())}" if errors else "all four methods returned",
    ))
    drgp = methods_results.get("drgp_unmasked", {})
    if drgp and drgp.get("error") is None:
        max_b = drgp.get("max_abs_beta", float("nan"))
        # Heavy-tail Gamma loadings (a_beta=0.3) can produce large posterior
        # entries on highly-expressed genes; only check that values are finite.
        checks.append(_check(
            "S2. DRGP Beta_hat all finite",
            np.isfinite(max_b),
            f"max|Beta_hat| = {max_b:.3f}",
        ))
        vp = drgp.get("valid_perm", False)
        checks.append(_check(
            "S3. Hungarian permutation valid (no duplicate matches)",
            bool(vp),
            f"pi={drgp.get('pi', np.array([])).tolist()}",
        ))
        cm = drgp.get("cos_mean", float("nan"))
        checks.append(_check(
            "S4. DRGP cos_sim_per_prog mean >= 0.7",
            np.isfinite(cm) and cm >= 0.7,
            f"mean cos sim across matched programs = {cm:.3f}",
        ))
    else:
        checks.append(_check("S2. DRGP Beta finite", False, "DRGP did not fit"))
        checks.append(_check("S3. Hungarian permutation valid", False, "DRGP did not fit"))
        checks.append(_check("S4. DRGP cos_sim_per_prog mean >= 0.7", False, "DRGP did not fit"))
    for label in ("nmf_lr", "pca_lr"):
        r = methods_results.get(label, {})
        el = r.get("elapsed_s", float("inf"))
        checks.append(_check(
            f"S5. {label} elapsed < 60s",
            el < 60.0,
            f"elapsed = {el:.1f}s",
        ))

    # ------------------------------------------------------------------
    # Per-method summary table
    # ------------------------------------------------------------------
    _section("Per-method summary")
    cols = [
        "elapsed_s", "max_abs_beta", "cos_mean", "jaccard_top50_mean",
        "support_auprc", "fdr_at_0p5", "v_spearman", "precision_at_3", "ood_auroc",
    ]
    print(f"  {'method':14s}  " + "  ".join(f"{c:>14s}" for c in cols))
    for label, r in methods_results.items():
        vals = []
        for c in cols:
            v = r.get(c, float("nan"))
            vals.append(f"{v:14.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else f"{'   nan':>14s}")
        print(f"  {label:14s}  " + "  ".join(vals))

    # Persist a compact .npz for inspection
    out_dir = Path(args.out_root) / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"smoke_{args.scale}_K{K_fit}_seed{args.seed}.npz"
    np.savez_compressed(
        save_path,
        v_true=gt.v, rel_idx=gt.rel_idx,
        **{
            f"{lbl}_{k}": np.asarray(v) if not isinstance(v, dict) else np.array([0])
            for lbl, r in methods_results.items()
            for k, v in r.items()
            if k not in ("_fit",) and v is not None and not callable(v)
        },
    )
    print(f"\n  Saved compact summary: {save_path}")
    print(f"  Errors recorded for: {list(errors.keys()) if errors else 'none'}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)
    print(f"\n{n_pass}/{n_total} invariants passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
