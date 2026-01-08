#!/usr/bin/env python3
"""
Collect and summarize metrics from multiple SVI runs.
Aggregates train/val/test performance across all runs in a results directory.
"""

import json
import gzip
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    roc_auc_score, roc_curve, auc
)

# Paul Tol's colorblind-safe qualitative palette (up to 8 categories)
COLORS_QUALITATIVE = [
    '#332288', '#88CCEE', '#44AA99', '#117733',
    '#999933', '#DDCC77', '#CC6677', '#AA4499'
]

# Sizing for 2-column figure (183mm ≈ 7.2in width)
FIG_WIDTH_1COL = 3.5   # inches (~89mm)
FIG_WIDTH_2COL = 7.2   # inches (~183mm)
FIG_ASPECT = 0.75      # height/width

# Font sizes calibrated for NO scaling at print
FONT_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 8,
}

STYLE_CONFIG = {
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.frameon': False,
    'figure.dpi': 150,           # screen preview
    'savefig.dpi': 300,          # publication output
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,          # TrueType (editable in Illustrator)
    'ps.fonttype': 42,
}

def apply_publication_style():
    """Apply publication-ready matplotlib configuration."""
    mpl.rcParams.update(FONT_CONFIG)
    mpl.rcParams.update(STYLE_CONFIG)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS_QUALITATIVE)

def fig_size(width='2col', aspect=None):
    """Return (width, height) tuple for standard figure sizes."""
    w = FIG_WIDTH_2COL if width == '2col' else FIG_WIDTH_1COL
    a = aspect if aspect else FIG_ASPECT
    return (w, w * a)


def load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    """Load predictions CSV for a given split."""
    pred_file = run_dir / f"svi_{split}_predictions.csv.gz"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing {pred_file}")
    return pd.read_csv(pred_file, compression='gzip')


def compute_metrics(predictions: pd.DataFrame) -> dict:
    """Compute classification metrics from predictions DataFrame."""
    y_true = predictions['true_label'].values
    y_pred = predictions['pred_label'].values
    y_prob = predictions['pred_prob'].values
    
    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }


def load_summary_json(run_dir: Path) -> dict:
    """Load svi_summary.json.gz for training metadata."""
    summary_file = run_dir / "svi_summary.json.gz"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing {summary_file}")
    with gzip.open(summary_file, 'rt') as f:
        return json.load(f)


def collect_run_metrics(run_dir: Path, run_number: int) -> dict:
    """Extract all metrics for a single run."""
    train_pred = load_predictions(run_dir, 'train')
    val_pred = load_predictions(run_dir, 'val')
    test_pred = load_predictions(run_dir, 'test')
    
    summary = load_summary_json(run_dir)
    
    train_metrics = compute_metrics(train_pred)
    val_metrics = compute_metrics(val_pred)
    test_metrics = compute_metrics(test_pred)
    
    # Extract training metadata
    training_time_hrs = summary['training']['training_time'] / 3600
    n_iterations = summary['training']['n_iterations']
    final_elbo = summary['elbo_history'][-1][1] if summary['elbo_history'] else np.nan
    
    return {
        'run': run_number,
        'training_time_hrs': training_time_hrs,
        'n_iterations': n_iterations,
        'final_elbo': final_elbo,
        'train_accuracy': train_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'train_roc_auc': train_metrics['roc_auc'],
        'val_accuracy': val_metrics['accuracy'],
        'val_precision': val_metrics['precision'],
        'val_recall': val_metrics['recall'],
        'val_f1': val_metrics['f1'],
        'val_roc_auc': val_metrics['roc_auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_roc_auc': test_metrics['roc_auc'],
    }


def collect_all_runs(results_dir: Path) -> pd.DataFrame:
    """Collect metrics from all run_* subdirectories."""
    all_metrics = []
    
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    for run_dir in run_dirs:
        try:
            run_number = int(run_dir.name.split('_')[1])
            metrics = collect_run_metrics(run_dir, run_number)
            all_metrics.append(metrics)
            print(f"✓ Collected run {run_number}")
        except Exception as e:
            print(f"✗ Failed run {run_dir.name}: {e}")
    
    return pd.DataFrame(all_metrics)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'run']
    
    summary_stats = []
    for col in numeric_cols:
        summary_stats.append({
            'metric': col,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        })
    
    return pd.DataFrame(summary_stats)


def plot_all_elbos(results_dir: Path, output_path: Path):
    """Plot ELBO trajectories for all runs on a single figure."""
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=fig_size('2col', aspect=0.6))
    
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    # Use colormap for 20+ runs
    n_runs = len(run_dirs)
    cmap = plt.cm.viridis if n_runs > len(COLORS_QUALITATIVE) else None
    
    for idx, run_dir in enumerate(run_dirs):
        try:
            run_number = int(run_dir.name.split('_')[1])
            summary = load_summary_json(run_dir)
            
            iterations = np.array([item[0] for item in summary['elbo_history']])
            elbo_values = np.array([item[1] for item in summary['elbo_history']])
            
            # Assign color
            if cmap:
                color = cmap(idx / n_runs)
            else:
                color = COLORS_QUALITATIVE[idx % len(COLORS_QUALITATIVE)]
            
            # Plot line with low alpha for individual runs
            ax.plot(iterations, elbo_values, linewidth=0.6, 
                   color=color, alpha=0.4, label=f'Run {run_number}')
            
        except Exception as e:
            print(f"✗ Failed to plot run {run_dir.name}: {e}")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    ax.set_title(f'ELBO Trajectories Across {n_runs} Runs')
    
    # Only show legend if runs <= 10, otherwise too cluttered
    if n_runs <= 10:
        ax.legend(loc='lower right', frameon=False, fontsize=6, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved ELBO plot to {output_path}")
    plt.close()


def plot_elbo_summary_stats(results_dir: Path, output_path: Path):
    """Plot ELBO mean ± std across all runs."""
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=fig_size('2col', aspect=0.6))
    
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    # Collect all ELBO histories
    all_elbos = {}
    for run_dir in run_dirs:
        try:
            summary = load_summary_json(run_dir)
            for iteration, elbo in summary['elbo_history']:
                if iteration not in all_elbos:
                    all_elbos[iteration] = []
                all_elbos[iteration].append(elbo)
        except Exception as e:
            print(f"✗ Failed to load ELBO from {run_dir.name}: {e}")
    
    # Compute statistics at each iteration
    iterations = np.array(sorted(all_elbos.keys()))
    means = np.array([np.mean(all_elbos[it]) for it in iterations])
    stds = np.array([np.std(all_elbos[it]) for it in iterations])
    
    # Plot mean with shaded std
    ax.plot(iterations, means, linewidth=1.2, color=COLORS_QUALITATIVE[0], 
           label='Mean ELBO', zorder=10)
    ax.fill_between(iterations, means - stds, means + stds, 
                    color=COLORS_QUALITATIVE[0], alpha=0.2, label='± 1 SD')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    ax.set_title(f'ELBO Summary: Mean ± SD Across {len(run_dirs)} Runs')
    ax.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✓ Saved ELBO summary plot to {output_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect SVI run metrics')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Path to results directory containing run_* folders')
    parser.add_argument('--output', type=str, default='metrics_summary.csv',
                        help='Output CSV filename')
    parser.add_argument('--summary-output', type=str, default='metrics_summary_stats.csv',
                        help='Summary statistics CSV filename')
    parser.add_argument('--plot-elbo', action='store_true',
                        help='Generate ELBO plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Collecting metrics from {results_dir}")
    print("=" * 70)
    
    # Collect all runs
    df = collect_all_runs(results_dir)
    
    if df.empty:
        print("No runs collected. Exiting.")
        return
    
    # Save full results
    output_path = results_dir / args.output
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved full metrics to {output_path}")
    
    # Compute and save summary statistics
    summary_df = summarize_metrics(df)
    summary_path = results_dir / args.summary_output
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary statistics to {summary_path}")
    
    # Generate ELBO plots if requested
    if args.plot_elbo:
        print("\nGenerating ELBO plots...")
        plot_all_elbos(results_dir, results_dir / 'elbo_all_runs.pdf')
        plot_elbo_summary_stats(results_dir, results_dir / 'elbo_summary.pdf')
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (mean ± std)")
    print("=" * 70)
    
    # Format output nicely
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} SET:")
        split_metrics = summary_df[summary_df['metric'].str.startswith(split)]
        for _, row in split_metrics.iterrows():
            metric_name = row['metric'].replace(f'{split}_', '')
            print(f"  {metric_name:15s}: {row['mean']:.4f} ± {row['std']:.4f}  "
                  f"[{row['min']:.4f}, {row['max']:.4f}]")
    
    print(f"\nTRAINING:")
    train_metrics = summary_df[summary_df['metric'].isin(['training_time_hrs', 'n_iterations', 'final_elbo'])]
    for _, row in train_metrics.iterrows():
        print(f"  {row['metric']:20s}: {row['mean']:.2f} ± {row['std']:.2f}  "
              f"[{row['min']:.2f}, {row['max']:.2f}]")
    
    print("\n" + "=" * 70)
    print(f"Total runs analyzed: {len(df)}")


if __name__ == '__main__':
    main()
