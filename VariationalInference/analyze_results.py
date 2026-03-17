"""
Analyze and visualize VI model results from saved output files.

Generates:
  1. Gene program loading bar charts (top-N most positive/negative/nearest-zero ν)
  2. ν weight distribution (density + histogram by label)
  3. ROC curves (train/val/test) per label
  4. KDE of predicted probabilities per label
  5. Convergence plots (ELBO and held-out LL)
  6. γ weight summary table

Usage:
    python -m VariationalInference.analyze_results --results-dir ./results
    python -m VariationalInference.analyze_results --results-dir ./results --prefix vi --n-top-programs 10 --n-top-genes 25
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV that may or may not be gzipped."""
    gz = path.with_suffix(path.suffix + '.gz') if not str(path).endswith('.gz') else path
    plain = Path(str(path).replace('.gz', '')) if str(path).endswith('.gz') else path
    if gz.exists():
        return pd.read_csv(gz, index_col=0)
    elif plain.exists():
        return pd.read_csv(plain, index_col=0)
    else:
        return None


def _load_csv_no_index(path: Path) -> pd.DataFrame:
    gz = path.with_suffix(path.suffix + '.gz') if not str(path).endswith('.gz') else path
    plain = Path(str(path).replace('.gz', '')) if str(path).endswith('.gz') else path
    if gz.exists():
        return pd.read_csv(gz)
    elif plain.exists():
        return pd.read_csv(plain)
    else:
        return None


def _load_summary(results_dir: Path, prefix: str) -> dict:
    for ext in ['.json.gz', '.json']:
        p = results_dir / f'{prefix}_summary{ext}'
        if p.exists():
            opener = gzip.open if ext.endswith('.gz') else open
            with opener(p, 'rt') as f:
                return json.load(f)
    return {}


def load_results(results_dir: Path, prefix: str = 'vi') -> Dict:
    """Load all result files from *results_dir* into a dict."""
    results_dir = Path(results_dir)
    data = {}

    # Gene programs (beta + v_weights)
    for ft in ['gene', 'pathway']:
        gp = _load_csv(results_dir / f'{prefix}_{ft}_programs.csv.gz')
        if gp is not None:
            data['programs'] = gp
            data['feature_type'] = ft
            break

    # Gamma weights
    data['gamma'] = _load_csv(results_dir / f'{prefix}_gamma_weights.csv.gz')
    data['gamma_var'] = _load_csv(results_dir / f'{prefix}_gamma_variance.csv.gz')

    # Theta
    data['theta_train'] = _load_csv(results_dir / f'{prefix}_theta_train.csv.gz')

    # Convergence histories
    data['elbo_hist'] = _load_csv_no_index(results_dir / f'{prefix}_elbo_history.csv.gz')
    data['holl_hist'] = _load_csv_no_index(results_dir / f'{prefix}_holl_history.csv.gz')

    # Summary
    data['summary'] = _load_summary(results_dir, prefix)

    # Model params (npz)
    npz_path = results_dir / f'{prefix}_model_params.npz'
    if npz_path.exists():
        data['params'] = dict(np.load(npz_path, allow_pickle=True))

    return data


# ── extraction ───────────────────────────────────────────────────────────────

def extract_v_weights(programs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract v_weight columns from the programs DataFrame."""
    v_cols = [c for c in programs_df.columns if c.startswith('v_weight')]
    return programs_df[v_cols].copy()


def extract_beta(programs_df: pd.DataFrame) -> pd.DataFrame:
    """Extract beta (gene loadings) from the programs DataFrame."""
    non_v = [c for c in programs_df.columns if not c.startswith('v_weight')]
    return programs_df[non_v].copy()


# ── 1. Gene program loading bar charts ───────────────────────────────────────

def plot_top_programs(
    programs_df: pd.DataFrame,
    label: str,
    n_top_programs: int = 10,
    n_top_genes: int = 25,
    figsize_per_panel: Tuple[float, float] = (3.2, 5.0),
    output_path: Optional[Path] = None,
):
    """
    For a given label's v_weight column, plot the top-N most positive,
    nearest-zero, and most negative gene programs as horizontal bar charts
    of their top gene loadings.
    """
    v_col = [c for c in programs_df.columns if c.startswith('v_weight') and label in c]
    if not v_col:
        # try generic matching
        v_cols = [c for c in programs_df.columns if c.startswith('v_weight')]
        if not v_cols:
            print(f"  No v_weight columns found; skipping program plots.")
            return
        v_col = v_cols
    v_col = v_col[0]

    v_weights = programs_df[v_col].sort_values()
    beta = extract_beta(programs_df)

    # Sort programs by v_weight
    sorted_programs = v_weights.sort_values(ascending=False)

    groups = {
        f'{label} — Top-{n_top_programs} Most Positive ν': sorted_programs.head(n_top_programs),
        f'{label} — {n_top_programs} Nearest-Zero ν': _nearest_zero(sorted_programs, n_top_programs),
        f'{label} — Top-{n_top_programs} Most Negative ν': sorted_programs.tail(n_top_programs).iloc[::-1],
    }
    colors = {'Positive': '#1f77b4', 'Nearest': '#2ca02c', 'Negative': '#d62728'}
    color_keys = ['Positive', 'Nearest', 'Negative']

    n_cols = min(5, n_top_programs)
    for idx, (title, gps) in enumerate(groups.items()):
        n_rows = int(np.ceil(len(gps) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
            squeeze=False,
        )
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        color = colors[color_keys[idx]]

        for i, (gp_name, v_val) in enumerate(gps.items()):
            ax = axes[i // n_cols, i % n_cols]
            loadings = beta.loc[gp_name].sort_values(ascending=False).head(n_top_genes)
            ax.barh(range(len(loadings)), loadings.values[::-1], color=color)
            ax.set_yticks(range(len(loadings)))
            ax.set_yticklabels(loadings.index[::-1], fontsize=7)
            ax.set_title(f'{gp_name}\nν = {v_val:.3e}', fontsize=9)
            ax.set_xlabel('β (gene loading)', fontsize=8)

        # hide unused axes
        for j in range(len(gps), n_rows * n_cols):
            axes[j // n_cols, j % n_cols].set_visible(False)

        plt.tight_layout()
        if output_path is not None:
            suffix = ['pos', 'zero', 'neg'][idx]
            fig.savefig(output_path / f'gp_loadings_{label}_{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved gene program loading plots for '{label}'")


def _nearest_zero(series: pd.Series, n: int) -> pd.Series:
    """Return the n entries closest to zero by absolute value."""
    return series.reindex(series.abs().sort_values().head(n).index)


# ── 2. ν weight distribution ────────────────────────────────────────────────

def plot_v_distribution(
    programs_df: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Density + histogram of ν weights, one panel per label."""
    v_df = extract_v_weights(programs_df)
    n_labels = v_df.shape[1]

    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 4), squeeze=False)
    fig.suptitle('Distribution of ν weights by label', fontsize=13, fontweight='bold')

    for i, col in enumerate(v_df.columns):
        ax = axes[0, i]
        vals = v_df[col].values
        label_name = col.replace('v_weight_', '')

        # histogram on secondary y-axis
        ax2 = ax.twinx()
        ax2.hist(vals, bins=30, alpha=0.25, color='slateblue', edgecolor='none')
        ax2.set_ylabel('Count', fontsize=9, color='slateblue')

        # KDE on primary
        from scipy.stats import gaussian_kde
        xs = np.linspace(vals.min() - 0.1 * np.ptp(vals), vals.max() + 0.1 * np.ptp(vals), 300)
        kde = gaussian_kde(vals)
        ax.plot(xs, kde(xs), color='midnightblue', lw=1.5)
        ax.fill_between(xs, kde(xs), alpha=0.15, color='slateblue')
        ax.set_xlabel('ν weight', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(label_name, fontsize=11)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path / 'v_weight_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved ν weight distribution plot")


# ── 3. ROC curves ───────────────────────────────────────────────────────────

def plot_roc(
    results_dir: Path,
    prefix: str = 'vi',
    summary: Optional[dict] = None,
    output_path: Optional[Path] = None,
):
    """
    Plot ROC curves from saved predictions.

    Expects files: {prefix}_predictions_{split}.csv.gz
    with columns: true label(s) and predicted probability columns.
    """
    # Try to load predictions for each split
    # quick_reference.py saves as: {prefix}_{split}_predictions.csv.gz
    # with columns: cell_id, true_{label}, prob_{label}, pred_{label}
    splits_data = {}
    for split in ['train', 'val', 'test']:
        for pattern in [
            f'{prefix}_{split}_predictions',
            f'{prefix}_predictions_{split}',
            f'{prefix}_pred_{split}',
        ]:
            p = results_dir / f'{pattern}.csv.gz'
            if p.exists():
                splits_data[split] = pd.read_csv(p)
                break
            p_plain = results_dir / f'{pattern}.csv'
            if p_plain.exists():
                splits_data[split] = pd.read_csv(p_plain)
                break

    if not splits_data:
        print("  No prediction files found; skipping ROC plot.")
        print(f"  (Looked for {prefix}_{{train,val,test}}_predictions.csv.gz)")
        return

    # Determine label columns: look for true_* columns in prediction files
    sample_df = next(iter(splits_data.values()))
    true_cols = [c for c in sample_df.columns if c.startswith('true_')]
    if true_cols:
        label_cols = true_cols
    elif summary and summary.get('label_columns'):
        label_cols = [f'true_{lc}' for lc in summary['label_columns']]
    else:
        print("  Cannot determine true/predicted columns; skipping ROC.")
        return

    n_labels = len(label_cols)
    fig, axes = plt.subplots(2, n_labels, figsize=(5 * n_labels, 9), squeeze=False)

    split_styles = {
        'train': ('solid', 'midnightblue', 'Train'),
        'val': ('dashed', 'teal', 'Validation'),
        'test': ('dotted', 'palevioletred', 'Test'),
    }

    for li, true_col in enumerate(label_cols):
        label_name = true_col.replace('true_', '')
        prob_col = f'prob_{label_name}'

        ax_roc = axes[0, li]
        ax_kde = axes[1, li]

        for split, (ls, color, split_label) in split_styles.items():
            if split not in splits_data:
                continue
            df = splits_data[split]

            if true_col not in df.columns or prob_col not in df.columns:
                continue

            y_true = df[true_col].values
            y_pred = df[prob_col].values

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, ls=ls, color=color, lw=1.5,
                        label=f'{split_label} (AUC={roc_auc:.3f})')

            # KDE of predicted probs
            from scipy.stats import gaussian_kde
            if len(np.unique(y_pred)) > 5:
                xs = np.linspace(0, 1, 300)
                try:
                    kde = gaussian_kde(y_pred, bw_method=0.05)
                    ax_kde.plot(xs, kde(xs), ls=ls, color=color, lw=1.5, label=split_label)
                except np.linalg.LinAlgError:
                    pass

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.set_title(f'vi — {label_name} | ROC', fontsize=11)
        ax_roc.legend(fontsize=8)

        ax_kde.axvline(0.5, color='grey', ls='--', lw=0.8)
        ax_kde.set_xlabel('Predicted Prob')
        ax_kde.set_ylabel('Density')
        ax_kde.set_title(f'vi — {label_name} | KDE', fontsize=11)
        ax_kde.legend(fontsize=8)

    # Common legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    for ax in axes[0]:
        ax.get_legend().remove()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path is not None:
        fig.savefig(output_path / 'roc_kde.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved ROC & KDE plot")


# ── 4. Convergence plots ────────────────────────────────────────────────────

def plot_convergence(
    data: Dict,
    output_path: Optional[Path] = None,
):
    """Plot ELBO and held-out LL convergence."""
    has_elbo = data.get('elbo_hist') is not None
    has_holl = data.get('holl_hist') is not None
    if not has_elbo and not has_holl:
        print("  No convergence history found; skipping.")
        return

    n_panels = int(has_elbo) + int(has_holl)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)
    idx = 0

    if has_elbo:
        ax = axes[0, idx]
        df = data['elbo_hist']
        ax.plot(df['iteration'], df['elbo'], color='midnightblue', lw=1.2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_title('ELBO Convergence')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        idx += 1

    if has_holl:
        ax = axes[0, idx]
        df = data['holl_hist']
        ax.plot(df['iteration'], df['heldout_ll'], color='teal', lw=1.2, label='Total')
        if 'heldout_pois_ll' in df.columns:
            ax.plot(df['iteration'], df['heldout_pois_ll'], color='steelblue',
                    lw=1, ls='--', label='Poisson')
            ax.plot(df['iteration'], df['heldout_reg_ll'], color='coral',
                    lw=1, ls='--', label='Regression')
            ax.legend(fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Held-out Log-Likelihood')
        ax.set_title('Held-out LL Convergence')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path / 'convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved convergence plot")


# ── 5. γ weight summary ─────────────────────────────────────────────────────

def plot_gamma_weights(
    gamma_df: pd.DataFrame,
    gamma_var_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
):
    """Heatmap-style table and bar chart of γ weights."""
    if gamma_df is None:
        print("  No gamma weights found; skipping.")
        return

    n_labels = gamma_df.shape[0]
    n_aux = gamma_df.shape[1]

    # Bar chart: one panel per label, bars = aux features sorted by |γ|
    fig, axes = plt.subplots(1, n_labels, figsize=(max(6, n_aux * 0.35), 4 * n_labels),
                             squeeze=False)
    fig.suptitle('γ weights (auxiliary covariate effects)', fontsize=13, fontweight='bold')

    for i, label in enumerate(gamma_df.index):
        ax = axes[0, i]
        vals = gamma_df.loc[label].sort_values()
        colors = ['#d62728' if v < 0 else '#1f77b4' for v in vals]
        ax.barh(range(len(vals)), vals.values, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=7)
        ax.set_xlabel('γ', fontsize=9)
        ax.set_title(str(label), fontsize=10)
        ax.axvline(0, color='k', lw=0.5)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path / 'gamma_weights.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also save gamma as a styled CSV summary
    if output_path is not None:
        summary_path = output_path / 'gamma_summary.csv'
        out = gamma_df.copy()
        if gamma_var_df is not None:
            # Add standard errors
            se = np.sqrt(gamma_var_df)
            se.index = [f'{idx}_SE' for idx in se.index]
            out = pd.concat([out, se])
        out.to_csv(summary_path)
        print(f"  Saved gamma summary to {summary_path}")

    print("  Saved γ weight plot")


# ── main ─────────────────────────────────────────────────────────────────────

def analyze(
    results_dir: str,
    prefix: str = 'vi',
    n_top_programs: int = 10,
    n_top_genes: int = 25,
    output_dir: Optional[str] = None,
):
    """Run the full analysis pipeline on saved VI results."""
    results_dir = Path(results_dir)
    if output_dir is None:
        output_path = results_dir / 'figures'
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    data = load_results(results_dir, prefix)

    summary = data.get('summary', {})
    label_columns = summary.get('label_columns') or []

    # 1. Gene program loading plots
    if 'programs' in data:
        programs = data['programs']
        v_df = extract_v_weights(programs)

        if not label_columns:
            label_columns = [c.replace('v_weight_', '') for c in v_df.columns]

        for label in label_columns:
            plot_top_programs(
                programs, label,
                n_top_programs=n_top_programs,
                n_top_genes=n_top_genes,
                output_path=output_path,
            )

        # 2. ν distribution
        plot_v_distribution(programs, output_path=output_path)
    else:
        print("  No gene programs file found; skipping program plots.")

    # 3. ROC + KDE
    plot_roc(results_dir, prefix, summary, output_path=output_path)

    # 4. Convergence
    plot_convergence(data, output_path=output_path)

    # 5. γ weights
    plot_gamma_weights(data.get('gamma'), data.get('gamma_var'), output_path=output_path)

    # 6. Print model summary
    if 'params' in data:
        params = data['params']
        print("\n=== Model Parameter Summary ===")
        for key in ['n_factors', 'n', 'p', 'p_aux']:
            if key in params:
                print(f"  {key}: {params[key]}")
        if 'E_beta' in params:
            eb = params['E_beta']
            print(f"  E[beta] shape: {eb.shape}  range: [{eb.min():.4f}, {eb.max():.4f}]  mean: {eb.mean():.4f}")
        if 'mu_v' in params:
            mv = params['mu_v']
            print(f"  mu_v shape: {mv.shape}  range: [{mv.min():.4f}, {mv.max():.4f}]")

    print(f"\nAll figures saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize VI model results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing saved VI result files')
    parser.add_argument('--prefix', default='vi',
                        help='File prefix used during save_results (default: vi)')
    parser.add_argument('--n-top-programs', type=int, default=10,
                        help='Number of top/bottom/nearest-zero programs to plot (default: 10)')
    parser.add_argument('--n-top-genes', type=int, default=25,
                        help='Number of top genes to show per program (default: 25)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for figures (default: <results-dir>/figures)')
    args = parser.parse_args()

    analyze(
        results_dir=args.results_dir,
        prefix=args.prefix,
        n_top_programs=args.n_top_programs,
        n_top_genes=args.n_top_genes,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
