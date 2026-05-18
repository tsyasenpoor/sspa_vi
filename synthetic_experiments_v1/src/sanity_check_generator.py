"""
Sanity checks for src/generator.generate().

Runs the 6 invariants from DRGP_synthetic_experiments_v1.md §5 Step 1
plus extras for:
  - frozen-params OOD generation (C2 dependency)
  - NB variance correctness
  - reproducibility (same seed -> identical Y, y)
  - mediation effect direction at multiple seeds (avoid lucky-seed claims)

Outputs:
  results/sanity/<scale>_report.txt   one-line-per-invariant pass/fail
  figures/sanity/<scale>_*.png        diagnostic plots

Usage:
  python -m src.sanity_check_generator                    # default n=500,p=5000
  python -m src.sanity_check_generator --scale fast       # n=100,p=500
  python -m src.sanity_check_generator --scale both
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `from src...` work when invoked from the project root or via -m
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate, FAST_DEV_KWARGS  # noqa: E402


# Covariate column order in X
SEX, AGE, ASTHMA, SMOKER, BMI = 0, 1, 2, 3, 4


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _check(name: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: {detail}")
    return (name, ok, detail)


def invariants(gt, scale_label: str) -> list[tuple[str, bool, str]]:
    """Run the 6 plan invariants + a few extras on a single GroundTruth."""
    results: list[tuple[str, bool, str]] = []

    # 1) Library size mean within a factor of 2 of the analytical target.
    #    E[lib] = sum_j sum_k E[theta_ik] * E[beta_jk] * S_{jk}.
    #    With S_{jk}=1 having coverage m_bar per program: E[lib] = K * m_bar * E[theta]_avg * E[beta].
    #    E[theta]_avg accounts for mediation: mediated programs have shape boost from X.
    cfg = gt.config
    E_beta = cfg["a_beta"] / cfg["b_beta"]
    # Per-program E[theta]: a_theta + E[max(X.Delta_k, 0)] divided by b_theta.
    # Use a quick numerical estimate over the empirical X.
    XD = gt.X @ gt.Delta.T                                     # (n, K)
    E_shape_per_k = cfg["a_theta"] + np.maximum(XD, 0).mean(axis=0)
    E_theta_per_k = E_shape_per_k / cfg["b_theta"]
    m_bar = float(np.mean(cfg["m_choices"]))
    target_lib = float(cfg["K_true"] * m_bar * E_theta_per_k.mean() * E_beta)
    lo, hi = target_lib / 2.0, target_lib * 2.0
    lib = gt.Y.sum(axis=1)
    mean_lib = float(lib.mean())
    results.append(_check(
        f"1. library size mean within 3x of analytical target ({target_lib:.0f})",
        lo <= mean_lib <= hi,
        f"mean={mean_lib:.1f}  min={int(lib.min())}  max={int(lib.max())}  "
        f"band=[{lo:.0f}, {hi:.0f}]",
    ))

    # 2) Per-gene density: each support gene appears in some non-trivial fraction
    #    of patients. Pool over support genes only; non-support genes are all zero
    #    by construction.
    support_any = gt.S.any(axis=1)
    if support_any.sum() == 0:
        results.append(_check(
            "2. gene density on support genes",
            False, "no support genes (empty mask)",
        ))
    else:
        nz_on_support = float((gt.Y[:, support_any] > 0).mean())
        # Each support gene draws Y ~ Poisson(theta_k * beta) for k where S_jk=1;
        # with E[theta * beta] ~ 2 (default) → P(Y>0) ~ 1 - exp(-2) ~ 0.86.
        results.append(_check(
            "2. per-gene density on support >= 10%",
            nz_on_support >= 0.10,
            f"P(Y>0 | gene in some support) = {nz_on_support:.3f}",
        ))

    # 3) Phenotype calibration y.mean() in [0.4, 0.6]
    yrate = float(gt.y.mean())
    results.append(_check(
        "3. phenotype rate in [0.4, 0.6]",
        0.4 <= yrate <= 0.6,
        f"y.mean()={yrate:.3f}  (target 0.5)",
    ))

    # 4) Loadings respect support: Beta[~S]==0; Beta[S]>0 on average
    mask = gt.S.astype(bool)
    nonsupport_sum = float(gt.Beta[~mask].sum())
    support_mean = float(gt.Beta[mask].mean()) if mask.any() else float("nan")
    results.append(_check(
        "4. Beta respects support mask",
        nonsupport_sum == 0.0 and support_mean > 0.0,
        f"sum(Beta[~S])={nonsupport_sum:.6f}  mean(Beta[S])={support_mean:.4f}",
    ))

    # 5) Mediation: asthma+ patients have higher Theta[:,0] (program 0 mediated by asthma)
    a_idx = (gt.X[:, ASTHMA] == 1)
    nA_idx = (gt.X[:, ASTHMA] == 0)
    if a_idx.sum() > 0 and nA_idx.sum() > 0:
        m1 = float(gt.Theta[a_idx, 0].mean())
        m0 = float(gt.Theta[nA_idx, 0].mean())
        ratio = m1 / max(m0, 1e-12)
        results.append(_check(
            "5. asthma+ raises Theta[:,0] (mediation)",
            m1 > m0,
            f"mean(asthma+)={m1:.3f}  mean(asthma-)={m0:.3f}  ratio={ratio:.2f}",
        ))
    else:
        results.append(_check(
            "5. mediation check",
            False,
            "not enough asthma+/- patients in cohort",
        ))

    # 6) Disease-relevant programs all have nonzero |v|; non-relevant have v==0
    rel = gt.rel_idx
    nonrel = np.array([k for k in range(gt.v.size) if k not in set(rel.tolist())])
    rel_ok = bool(np.all(np.abs(gt.v[rel]) > 0))
    nonrel_ok = bool(np.all(gt.v[nonrel] == 0)) if nonrel.size > 0 else True
    results.append(_check(
        "6. v=0 off rel_idx, |v|>0 on rel_idx",
        rel_ok and nonrel_ok,
        f"v={np.round(gt.v, 3).tolist()}  rel_idx={rel.tolist()}",
    ))

    return results


def reproducibility_check(default_kwargs: dict) -> tuple[str, bool, str]:
    gt_a = generate(**default_kwargs, seed=42)
    gt_b = generate(**default_kwargs, seed=42)
    ok = (
        np.array_equal(gt_a.Y, gt_b.Y)
        and np.array_equal(gt_a.y, gt_b.y)
        and np.allclose(gt_a.Beta, gt_b.Beta)
        and np.allclose(gt_a.Theta, gt_b.Theta)
    )
    return _check(
        "R1. same seed -> identical (Y, y, Beta, Theta)",
        ok,
        "deterministic" if ok else "DRIFT detected across reruns",
    )


def frozen_params_check(default_kwargs: dict) -> list[tuple[str, bool, str]]:
    """Generate a 'train' set, then an OOD test set via freeze_params + shift."""
    gt_train = generate(**default_kwargs, seed=0)
    # OOD: shifted asthma rate, larger n, frozen structural params
    ood_kwargs = {**default_kwargs, "asthma_rate": 0.6, "n": min(2 * default_kwargs["n"], 2000)}
    gt_ood = generate(**ood_kwargs, seed=10_000, freeze_params=gt_train)
    out = []
    out.append(_check(
        "F1. frozen Beta identical across train/OOD",
        np.array_equal(gt_train.Beta, gt_ood.Beta),
        f"|Beta|={gt_train.Beta.shape}",
    ))
    out.append(_check(
        "F2. frozen v identical",
        np.array_equal(gt_train.v, gt_ood.v),
        f"v={np.round(gt_train.v, 3).tolist()}",
    ))
    out.append(_check(
        "F3. frozen Delta identical",
        np.array_equal(gt_train.Delta, gt_ood.Delta),
        f"|Delta|={gt_train.Delta.shape}",
    ))
    out.append(_check(
        "F4. frozen xi_0 identical",
        float(gt_train.xi_0) == float(gt_ood.xi_0),
        f"xi_0={gt_train.xi_0:.4f}",
    ))
    out.append(_check(
        "F5. OOD covariate shift took effect (asthma rate up)",
        gt_ood.X[:, ASTHMA].mean() > gt_train.X[:, ASTHMA].mean() + 0.15,
        f"train_p(asthma)={gt_train.X[:, ASTHMA].mean():.3f}  "
        f"ood_p(asthma)={gt_ood.X[:, ASTHMA].mean():.3f}",
    ))
    return out


def nb_variance_check(default_kwargs: dict) -> list[tuple[str, bool, str]]:
    """Verify NB sampler: empirical var ≈ mu + phi*mu^2 for a fixed mu grid."""
    rng = np.random.default_rng(0)
    n_draws = 20000
    phi = 0.5
    out = []
    for mu in (1.0, 5.0, 20.0):
        n_param = 1.0 / phi
        p_param = 1.0 / (1.0 + phi * mu)
        samples = rng.negative_binomial(n_param, p_param, size=n_draws)
        emp_mean = float(samples.mean())
        emp_var = float(samples.var())
        expected_var = mu + phi * mu ** 2
        ok_mean = abs(emp_mean - mu) / mu < 0.05
        ok_var = abs(emp_var - expected_var) / expected_var < 0.10
        out.append(_check(
            f"NB1. mu={mu}: mean ok",
            ok_mean,
            f"emp_mean={emp_mean:.3f}  target={mu}",
        ))
        out.append(_check(
            f"NB2. mu={mu}: var ≈ mu + phi*mu^2",
            ok_var,
            f"emp_var={emp_var:.3f}  expected={expected_var:.3f}",
        ))
    return out


def mediation_robustness_check(default_kwargs: dict, n_seeds: int = 5) -> tuple[str, bool, str]:
    """Across n_seeds runs, asthma+ patients should consistently have higher Theta[:,0]."""
    wins = 0
    for s in range(n_seeds):
        gt = generate(**default_kwargs, seed=s)
        if gt.X[:, ASTHMA].sum() == 0 or (gt.X[:, ASTHMA] == 0).sum() == 0:
            continue
        m1 = gt.Theta[gt.X[:, ASTHMA] == 1, 0].mean()
        m0 = gt.Theta[gt.X[:, ASTHMA] == 0, 0].mean()
        if m1 > m0:
            wins += 1
    return _check(
        f"M1. mediation effect direction consistent across {n_seeds} seeds",
        wins == n_seeds,
        f"asthma+ > asthma- in Theta[:,0] for {wins}/{n_seeds} seeds",
    )


def plot_diagnostics(gt, fig_dir: Path, scale_label: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Panel 1: library sizes and gene density
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(gt.Y.sum(axis=1), bins=40, color="steelblue")
    axes[0].set_xlabel("library size"); axes[0].set_ylabel("# patients")
    axes[0].set_title("Library size distribution")

    gene_density = (gt.Y > 0).mean(axis=0)
    axes[1].hist(gene_density, bins=40, color="steelblue")
    axes[1].set_xlabel("frac patients with Y>0"); axes[1].set_ylabel("# genes")
    axes[1].set_title("Per-gene density")

    # Phenotype rate
    axes[2].bar([0, 1], [float((gt.y == 0).mean()), float((gt.y == 1).mean())],
                color=["lightgray", "salmon"])
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["y=0", "y=1"])
    axes[2].set_title(f"Phenotype rate = {gt.y.mean():.3f}")
    fig.suptitle(f"Generator diagnostics ({scale_label})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scale_label}_distributions.png", dpi=150)
    plt.close(fig)

    # Panel 2: mediation effect
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    a_mask = (gt.X[:, ASTHMA] == 1)
    axes[0].hist(gt.Theta[a_mask, 0], bins=30, alpha=0.6, label="asthma+", color="salmon")
    axes[0].hist(gt.Theta[~a_mask, 0], bins=30, alpha=0.6, label="asthma-", color="steelblue")
    axes[0].set_title(f"Theta[:,0]  (Delta[0,asthma]={gt.Delta[0, ASTHMA]})")
    axes[0].set_xlabel("Theta[:,0]"); axes[0].legend()

    axes[1].scatter(gt.X[:, AGE], gt.Theta[:, 1], s=6, alpha=0.5, color="darkgreen")
    axes[1].set_xlabel("age (standardized)"); axes[1].set_ylabel("Theta[:,1]")
    axes[1].set_title(f"Theta[:,1] vs age  (Delta[1,age]={gt.Delta[1, AGE]})")
    fig.suptitle(f"Mediation effects ({scale_label})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scale_label}_mediation.png", dpi=150)
    plt.close(fig)

    # Panel 3: Beta support and v
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].imshow(gt.S.T, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_xlabel("gene"); axes[0].set_ylabel("program")
    axes[0].set_title("Support mask S (programs x genes)")
    axes[1].bar(np.arange(gt.v.size), gt.v, color="navy")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("program k"); axes[1].set_ylabel("v_k (true)")
    axes[1].set_title("Regression weights v")
    fig.suptitle(f"Structure ({scale_label})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{scale_label}_structure.png", dpi=150)
    plt.close(fig)


def run(scale: str, out_root: Path) -> int:
    fig_dir = out_root / "figures" / "sanity"
    rep_dir = out_root / "results" / "sanity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    if scale == "fast":
        kwargs = dict(FAST_DEV_KWARGS)
        label = "fast_n100_p500_K3"
    elif scale == "default":
        kwargs = dict(n=500, p=5000, K_true=10, K_rel=3, q=5)
        label = "default_n500_p5000_K10"
    else:
        raise ValueError(f"unknown scale: {scale}")

    _section(f"SANITY CHECKS — {label}")
    gt = generate(**kwargs, seed=0, verbose=True)

    results: list[tuple[str, bool, str]] = []
    print("\n-- Plan invariants 1-6 --")
    results.extend(invariants(gt, label))

    print("\n-- Reproducibility --")
    results.append(reproducibility_check(kwargs))

    print("\n-- Frozen-params OOD --")
    results.extend(frozen_params_check(kwargs))

    print("\n-- NB variance --")
    results.extend(nb_variance_check(kwargs))

    print("\n-- Mediation robustness across seeds --")
    results.append(mediation_robustness_check(kwargs, n_seeds=5))

    # Plots from the default seed=0 cohort
    plot_diagnostics(gt, fig_dir, label)

    report_path = rep_dir / f"{label}_report.txt"
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    with open(report_path, "w") as f:
        f.write(f"Sanity report  --  {label}\n")
        f.write(f"Generator config: {gt.config}\n\n")
        f.write(f"{n_pass}/{n_total} checks passed\n\n")
        for name, ok, detail in results:
            f.write(f"  [{'PASS' if ok else 'FAIL'}] {name}\n      {detail}\n")

    print(f"\n{n_pass}/{n_total} checks passed.")
    print(f"  Report: {report_path}")
    print(f"  Figures: {fig_dir}/{label}_*.png")
    return 0 if n_pass == n_total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["fast", "default", "both"], default="both")
    ap.add_argument(
        "--out-root",
        default="/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1",
    )
    args = ap.parse_args()
    out_root = Path(args.out_root)
    rc = 0
    if args.scale in ("fast", "both"):
        rc |= run("fast", out_root)
    if args.scale in ("default", "both"):
        rc |= run("default", out_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
