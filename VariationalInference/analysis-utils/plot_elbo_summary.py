#!/usr/bin/env python3
"""Extract ELBO traces from all drgp_full runs and create summary plots."""

import os
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "/labs/Aguiar/SSPA_BRAY/results/sim_pathway/drgp_full"
OUT_DIR = os.path.join(BASE, "elbo_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_elbo(filepath):
    """Extract (iteration, ELBO) pairs from a .out file."""
    iters, elbos = [], []
    pattern = re.compile(r'Iter\s+(\d+):\s+ELBO=([-\d.e+]+)')
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                elbos.append(float(m.group(2)))
    return np.array(iters), np.array(elbos)

def parse_label(relpath):
    """Parse relpath like laplace/combined/exp0_easy/...out into label parts."""
    parts = relpath.split('/')
    vprior = parts[0]       # laplace or normal
    mode = parts[1]         # combined, masked, unmasked
    config_full = parts[2]  # exp0_easy, exp1_medium, etc.
    config = config_full.split('_')[0]  # exp0, exp1, ...
    return mode, config, vprior, f"{mode}-{config}-{vprior}"

# Collect all runs
out_files = sorted(glob.glob(os.path.join(BASE, "*/*/*/*.out")))
runs = {}
for fp in out_files:
    relpath = os.path.relpath(fp, BASE)
    mode, config, vprior, label = parse_label(relpath)
    iters, elbos = extract_elbo(fp)
    if len(iters) > 0:
        runs[label] = {
            'iters': iters, 'elbos': elbos,
            'mode': mode, 'config': config, 'vprior': vprior,
            'file': fp
        }

print(f"Found {len(runs)} runs")
for label in sorted(runs):
    r = runs[label]
    print(f"  {label}: {len(r['iters'])} iterations, "
          f"ELBO range [{r['elbos'].min():.3e}, {r['elbos'].max():.3e}], "
          f"final={r['elbos'][-1]:.3e}")

# --- Color/style setup ---
modes = ['combined', 'masked', 'unmasked']
vpriors = ['laplace', 'normal']
configs = ['exp0', 'exp1', 'exp2', 'exp3']
config_colors = {'exp0': '#1f77b4', 'exp1': '#ff7f0e', 'exp2': '#2ca02c', 'exp3': '#d62728'}
mode_colors_hex = {'combined': '#1f77b4', 'masked': '#ff7f0e', 'unmasked': '#2ca02c'}

# ============================================================
# PLOT 1: All runs on one figure (full ELBO, may be hard to read)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
for label in sorted(runs):
    r = runs[label]
    ax.plot(r['iters'], r['elbos'], '.', markersize=2, label=label, alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('ELBO')
ax.set_title('All ELBO Traces (raw scale)')
ax.legend(fontsize=6, ncol=3, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "elbo_all_raw.png"), dpi=200)
plt.close(fig)
print("Saved elbo_all_raw.png")

# ============================================================
# PLOT 2: Grid by mode (rows) x vprior (cols), configs overlaid
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False, sharey=False)
for i, mode in enumerate(modes):
    for j, vp in enumerate(vpriors):
        ax = axes[i, j]
        found = False
        for cfg in configs:
            label = f"{mode}-{cfg}-{vp}"
            if label in runs:
                r = runs[label]
                ax.plot(r['iters'], r['elbos'], '.', markersize=3,
                        color=config_colors[cfg], label=cfg, alpha=0.8)
                found = True
        ax.set_title(f"{mode} / {vp}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No runs', transform=ax.transAxes, ha='center', va='center')
fig.suptitle('ELBO Traces by Mode and Prior', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "elbo_grid_mode_prior.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved elbo_grid_mode_prior.png")

# ============================================================
# PLOT 3: Grid by config (rows) x vprior (cols), modes overlaid
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=False, sharey=False)
for i, cfg in enumerate(configs):
    for j, vp in enumerate(vpriors):
        ax = axes[i, j]
        found = False
        for mode in modes:
            label = f"{mode}-{cfg}-{vp}"
            if label in runs:
                r = runs[label]
                ax.plot(r['iters'], r['elbos'], '.', markersize=3,
                        color=mode_colors_hex[mode], label=mode, alpha=0.8)
                found = True
        ax.set_title(f"{cfg} / {vp}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No runs', transform=ax.transAxes, ha='center', va='center')
fig.suptitle('ELBO Traces by Config and Prior', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "elbo_grid_config_prior.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved elbo_grid_config_prior.png")

# ============================================================
# PLOT 4: Log scale -ELBO grid by mode x prior
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False, sharey=False)
for i, mode in enumerate(modes):
    for j, vp in enumerate(vpriors):
        ax = axes[i, j]
        found = False
        for cfg in configs:
            label = f"{mode}-{cfg}-{vp}"
            if label in runs:
                r = runs[label]
                neg_elbo = -r['elbos']
                ax.plot(r['iters'], neg_elbo, '.', markersize=3,
                        color=config_colors[cfg], label=cfg, alpha=0.8)
                found = True
        ax.set_title(f"{mode} / {vp}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('-ELBO (log scale)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No runs', transform=ax.transAxes, ha='center', va='center')
fig.suptitle('Negative ELBO (log scale) by Mode and Prior', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "elbo_grid_mode_prior_logscale.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved elbo_grid_mode_prior_logscale.png")

# ============================================================
# PLOT 5: Zoomed early iterations (first 500) — linear scale
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False, sharey=False)
for i, mode in enumerate(modes):
    for j, vp in enumerate(vpriors):
        ax = axes[i, j]
        found = False
        for cfg in configs:
            label = f"{mode}-{cfg}-{vp}"
            if label in runs:
                r = runs[label]
                mask = r['iters'] <= 500
                if mask.any():
                    ax.plot(r['iters'][mask], r['elbos'][mask], '.', markersize=4,
                            color=config_colors[cfg], label=cfg, alpha=0.8)
                    found = True
        ax.set_title(f"{mode} / {vp} (iter 0-500)", fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No runs', transform=ax.transAxes, ha='center', va='center')
fig.suptitle('ELBO Traces — First 500 Iterations', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "elbo_grid_early_500.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved elbo_grid_early_500.png")

# ============================================================
# PLOT 6: Per-run individual plots
# ============================================================
indiv_dir = os.path.join(OUT_DIR, "individual")
os.makedirs(indiv_dir, exist_ok=True)
for label in sorted(runs):
    r = runs[label]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(r['iters'], r['elbos'], '.', markersize=3, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO')
    ax1.set_title(f'{label} — Full Trace')
    ax1.grid(True, alpha=0.3)

    neg = -r['elbos']
    pos_mask = neg > 0
    if pos_mask.any():
        ax2.plot(r['iters'][pos_mask], neg[pos_mask], '.', markersize=3, color='firebrick', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('-ELBO (log scale)')
    ax2.set_yscale('log')
    ax2.set_title(f'{label} — -ELBO Log Scale')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    safe_label = label.replace('-', '_')
    fig.savefig(os.path.join(indiv_dir, f"elbo_{safe_label}.png"), dpi=150)
    plt.close(fig)

print(f"Saved {len(runs)} individual plots to {indiv_dir}/")

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*90)
print(f"{'Label':<35} {'Iters':>6} {'Init ELBO':>14} {'Best ELBO':>14} {'Final ELBO':>14} {'Stable?':>8}")
print("="*90)
for label in sorted(runs):
    r = runs[label]
    init_e = r['elbos'][0]
    best_e = r['elbos'].max()
    final_e = r['elbos'][-1]
    n_iter = len(r['iters'])
    stable = abs(final_e) < abs(best_e) * 100
    print(f"{label:<35} {n_iter:>6} {init_e:>14.3e} {best_e:>14.3e} {final_e:>14.3e} {'Yes' if stable else 'NO':>8}")

print(f"\nAll plots saved to: {OUT_DIR}/")
