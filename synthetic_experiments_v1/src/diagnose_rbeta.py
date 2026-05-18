"""
Diagnose the spike-and-slab posterior inclusion distribution (r_beta).

Loads a saved experiment .npz that contains a DRGP fit with R_beta and the
true support mask, and emits:
  - Histograms of r_beta separated by ground-truth support / non-support
  - r_beta CDF
  - Per-program scatter: r_beta vs |beta_hat|
  - Numerical summary: mean / median / 5-95th percentile per group

Output:
  figures/diagnostic/<stem>_rbeta_{hist,cdf,scatter}.png
  results/diagnostic/<stem>_rbeta_summary.txt

Usage:
  python -m src.diagnose_rbeta \
      --npz results/raw/_smoke_fast/cond000_seed0000.npz \
      --method drgp_unmasked
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import generate                                  # noqa: E402
from src.metrics import hungarian_match                             # noqa: E402


def load_drgp_record(npz_path: str, method: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    if method not in d.files:
        raise KeyError(f"method {method} not in {npz_path}; have {list(d.files)}")
    m = d[method].item()
    if "R_beta" not in m:
        raise KeyError(f"R_beta not saved in {npz_path}; resave with save_posteriors=True")
    return {
        "R_beta": np.asarray(m["R_beta"]),       # (p, K_fit)
        "pi": np.asarray(m["pi"]),                # (K_fit,)
        "cos_per_prog": np.asarray(m.get("cos_per_prog", [])),
        "support_auprc": float(m.get("support_auprc", float("nan"))),
        "fdr_at_0p5": float(m.get("fdr_at_0p5", float("nan"))),
        "raw": d,
    }


def reconstruct_ground_truth(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive the true support S and Beta by re-running the generator with
    the same seed and the same config. We pull seed, generator config from the
    npz; generator params live in the config YAML, but the generator is
    deterministic given seed + kwargs so we re-run with FAST_DEV_KWARGS or the
    matching default-scale kwargs.

    Simpler approach: the experiment.py npz already contains v_true and rel_idx
    but not the full S/Beta. We re-run the generator here using the stored seed
    plus assumptions about scale (inferred from R_beta shape).
    """
    raise NotImplementedError(
        "use --config to point at the YAML so we can re-generate"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="experiment.py output .npz")
    ap.add_argument("--config", required=True, help="config YAML used")
    ap.add_argument("--method", default="drgp_unmasked")
    ap.add_argument("--out-root",
                    default="/labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    drgp = load_drgp_record(args.npz, args.method)
    R_beta = drgp["R_beta"]
    pi = drgp["pi"]
    raw = drgp["raw"]
    cond_idx = int(raw["condition_idx"])
    seed = int(raw["seed"])

    # Re-generate the same dataset to get the ground-truth S/Beta
    cond = cfg["conditions"][cond_idx]
    gen_kwargs = dict(cfg["generator_defaults"])
    gen_kwargs.update(cond.get("generator_overrides", {}))
    gen_kwargs["seed"] = seed
    gt = generate(**gen_kwargs)
    S = gt.S
    p, K_true = S.shape

    print(f"Loaded:")
    print(f"  npz       = {args.npz}")
    print(f"  config    = {args.config}")
    print(f"  cond_idx  = {cond_idx}  seed = {seed}")
    print(f"  K_fit     = {R_beta.shape[1]}  K_true = {K_true}  p = {p}")
    print(f"  pi        = {pi.tolist()}")
    print(f"  support_auprc = {drgp['support_auprc']:.3f}")
    print(f"  fdr_at_0.5    = {drgp['fdr_at_0p5']:.3f}")

    # Collect r_beta separated by support / non-support on matched columns
    rbeta_supp, rbeta_off = [], []
    for k_fit in range(R_beta.shape[1]):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        in_supp = S[:, k_true] == 1
        rbeta_supp.append(R_beta[in_supp, k_fit])
        rbeta_off.append(R_beta[~in_supp, k_fit])
    rbeta_supp = np.concatenate(rbeta_supp) if rbeta_supp else np.array([])
    rbeta_off = np.concatenate(rbeta_off) if rbeta_off else np.array([])

    def _q(x, qs=(0.05, 0.50, 0.95)):
        if x.size == 0:
            return [float("nan")] * len(qs)
        return [float(np.quantile(x, q)) for q in qs]

    q_supp = _q(rbeta_supp)
    q_off = _q(rbeta_off)
    summary = (
        f"r_beta summary (matched columns)\n"
        f"  on support     n={rbeta_supp.size:6d}  "
        f"mean={rbeta_supp.mean():.3f}  median={np.median(rbeta_supp):.3f}  "
        f"P5={q_supp[0]:.3f}  P50={q_supp[1]:.3f}  P95={q_supp[2]:.3f}\n"
        f"  off support    n={rbeta_off.size:6d}  "
        f"mean={rbeta_off.mean():.3f}  median={np.median(rbeta_off):.3f}  "
        f"P5={q_off[0]:.3f}  P50={q_off[1]:.3f}  P95={q_off[2]:.3f}\n"
        f"  P(r_beta>0.5 | on support)  = {(rbeta_supp > 0.5).mean():.3f}\n"
        f"  P(r_beta>0.5 | off support) = {(rbeta_off > 0.5).mean():.3f}\n"
        f"  P(r_beta>0.1 | on support)  = {(rbeta_supp > 0.1).mean():.3f}\n"
        f"  P(r_beta>0.1 | off support) = {(rbeta_off > 0.1).mean():.3f}\n"
    )
    print(summary)

    stem = Path(args.npz).stem
    out_fig = Path(args.out_root) / "figures" / "diagnostic"
    out_res = Path(args.out_root) / "results" / "diagnostic"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)
    with open(out_res / f"{stem}_rbeta_summary.txt", "w") as f:
        f.write(summary)

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(rbeta_off, bins=bins, alpha=0.6, label=f"off support (n={rbeta_off.size})",
            color="lightgray", density=True)
    ax.hist(rbeta_supp, bins=bins, alpha=0.7, label=f"on support (n={rbeta_supp.size})",
            color="salmon", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$r_{\beta,jk}$ (posterior inclusion probability)")
    ax.set_ylabel("density")
    ax.set_title(f"r_beta posterior by ground-truth support\n"
                 f"cond={cond_idx} seed={seed} K_fit={R_beta.shape[1]} K_true={K_true}")
    ax.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(out_fig / f"{stem}_rbeta_hist.png", dpi=180)
    plt.close()

    # CDF
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, x, c in [("on support", rbeta_supp, "salmon"),
                       ("off support", rbeta_off, "lightgray")]:
        if x.size == 0:
            continue
        xs = np.sort(x)
        ys = np.linspace(0, 1, len(xs))
        ax.plot(xs, ys, label=name, color=c)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$r_{\beta,jk}$")
    ax.set_ylabel(r"CDF")
    ax.set_title(f"r_beta CDF by support  (AUPRC={drgp['support_auprc']:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_fig / f"{stem}_rbeta_cdf.png", dpi=180)
    plt.close()

    print(f"\nFigures: {out_fig}/{stem}_rbeta_{{hist,cdf}}.png")
    print(f"Summary: {out_res}/{stem}_rbeta_summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
