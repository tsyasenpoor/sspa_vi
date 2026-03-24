#!/usr/bin/env python3
"""
Aggregate all simulation experiment metrics into a unified table.
Reads results from baselines, scHPF, Spectra, and DRGP Bayesian optimization.
Outputs CSV and LaTeX table for the paper.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_ROOT = "/labs/Aguiar/SSPA_BRAY/results/sim_pathway"
EXPERIMENTS = ["exp0_easy", "exp1_medium", "exp2_hard", "exp3_intersectional"]
EXP_LABELS = {"exp0_easy": "C0", "exp1_medium": "C1", "exp2_hard": "C2", "exp3_intersectional": "C3"}
LABELS = ["severity", "outcome"]


def load_baseline_metrics():
    """Load metrics from baselines/{exp}/{label}/summary.json"""
    rows = []
    for exp in EXPERIMENTS:
        for label in LABELS:
            path = os.path.join(RESULTS_ROOT, "baselines", exp, label, "summary.json")
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            for method, metrics in data["results"].items():
                if metrics.get("status") != "success":
                    continue
                rows.append({
                    "method": method.upper(),
                    "method_type": _baseline_type(method),
                    "experiment": exp,
                    "label": label,
                    "val_auc": metrics["val_auc"],
                    "test_auc": None,
                })
    return rows


def _baseline_type(method):
    if method == "svm":
        return "Black-box"
    elif method in ("lr", "lrl", "lrr"):
        return "Linear"
    elif method.startswith("mf"):
        return "NMF + LR"
    return "Unknown"


def load_schpf_metrics():
    """Load metrics from schpf_baselines/{exp}/schpf_baselines_summary.json"""
    rows = []
    for exp in EXPERIMENTS:
        path = os.path.join(RESULTS_ROOT, "schpf_baselines", exp, "schpf_baselines_summary.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for key, metrics in data["results"].items():
            # key format: schpf_{clf}_{label}
            parts = key.split("_")
            clf = parts[1]  # lr, lrl, lrr
            label = parts[2]  # severity, outcome
            method_name = f"scHPF-{clf.upper()}"
            if clf == "lrl":
                method_name = "scHPF-L1"
            elif clf == "lrr":
                method_name = "scHPF-Ridge"
            elif clf == "lr":
                method_name = "scHPF-LR"
            rows.append({
                "method": method_name,
                "method_type": "scHPF + LR",
                "experiment": exp,
                "label": label,
                "val_auc": metrics["val"]["roc_auc"],
                "test_auc": metrics["test"]["roc_auc"],
            })
    return rows


def load_spectra_metrics():
    """Load metrics from spectra_sup_baselines/{exp}/spectra_baselines_summary.json"""
    rows = []
    for exp in EXPERIMENTS:
        path = os.path.join(RESULTS_ROOT, "spectra_sup_baselines", exp, "spectra_baselines_summary.json")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for key, metrics in data["results"].items():
            parts = key.split("_")
            clf = parts[1]  # lr, lrl, lrr
            label = parts[2]  # severity, outcome
            method_name = f"Spectra-{clf.upper()}"
            if clf == "lrl":
                method_name = "Spectra-L1"
            elif clf == "lrr":
                method_name = "Spectra-Ridge"
            elif clf == "lr":
                method_name = "Spectra-LR"
            rows.append({
                "method": method_name,
                "method_type": "Spectra + LR",
                "experiment": exp,
                "label": label,
                "val_auc": metrics["val"]["roc_auc"],
                "test_auc": metrics["test"]["roc_auc"],
            })
    return rows


def load_drgp_bayes_opt_metrics():
    """Load best metrics from pbmc_bayes_opt/{prior}/{mode}/{exp}/best_params_vi_*.json"""
    rows = []
    priors = ["laplace"]
    modes = ["combined", "masked", "unmasked"]

    for prior in priors:
        for mode in modes:
            for exp in EXPERIMENTS:
                exp_dir = os.path.join(RESULTS_ROOT, "pbmc_bayes_opt", prior, mode, exp)
                if not os.path.isdir(exp_dir):
                    continue
                # Find latest best_params file
                param_files = sorted(glob.glob(os.path.join(exp_dir, "best_params_vi_*.json")))
                if not param_files:
                    continue
                latest = param_files[-1]  # sorted by timestamp in filename
                with open(latest) as f:
                    data = json.load(f)

                bvm = data["best_val_metrics"]
                prior_label = prior.capitalize()
                mode_label = mode.capitalize()
                method_name = f"DRGP {prior_label}/{mode_label}"

                for label in LABELS:
                    val_key = f"val_{label}_auc"
                    rows.append({
                        "method": method_name,
                        "method_type": f"DRGP ({prior}/{mode})",
                        "experiment": exp,
                        "label": label,
                        "val_auc": bvm.get(val_key),
                        "test_auc": None,
                        "source_file": os.path.basename(latest),
                        "best_trial": data.get("best_trial"),
                        "n_trials": data.get("n_trials"),
                    })
    return rows


def load_drgp_full_metrics():
    """Load metrics from drgp_full/{prior}/{mode}/{exp}/seed*/vi_metrics.csv"""
    rows = []
    full_dir = os.path.join(RESULTS_ROOT, "drgp_full")
    if not os.path.isdir(full_dir):
        return rows

    priors = ["laplace"]
    modes = ["combined", "masked", "unmasked"]

    for prior in priors:
        for mode in modes:
            for exp in EXPERIMENTS:
                seed_dirs = sorted(glob.glob(os.path.join(full_dir, prior, mode, exp, "seed*")))
                for seed_dir in seed_dirs:
                    metrics_path = os.path.join(seed_dir, "vi_metrics.csv")
                    if not os.path.exists(metrics_path):
                        continue
                    try:
                        df = pd.read_csv(metrics_path)
                        seed = os.path.basename(seed_dir)
                        prior_label = prior.capitalize()
                        mode_label = mode.capitalize()
                        method_name = f"DRGP-Full {prior_label}/{mode_label}"

                        for label in LABELS:
                            val_row = df[(df["split"] == "val") & (df["label"] == label)]
                            test_row = df[(df["split"] == "test") & (df["label"] == label)]
                            rows.append({
                                "method": method_name,
                                "method_type": f"DRGP-Full ({prior}/{mode})",
                                "experiment": exp,
                                "label": label,
                                "val_auc": val_row["auc"].iloc[0] if len(val_row) > 0 else None,
                                "test_auc": test_row["auc"].iloc[0] if len(test_row) > 0 else None,
                                "source_file": os.path.relpath(metrics_path, RESULTS_ROOT),
                                "seed": seed,
                            })
                    except Exception as e:
                        print(f"  WARNING: failed to read {metrics_path}: {e}")
    return rows


def aggregate_all():
    """Aggregate all metrics into a single DataFrame."""
    print("Loading baseline metrics...")
    all_rows = load_baseline_metrics()
    print(f"  {len(all_rows)} rows")

    print("Loading scHPF metrics...")
    schpf = load_schpf_metrics()
    all_rows.extend(schpf)
    print(f"  {len(schpf)} rows")

    print("Loading Spectra metrics...")
    spectra = load_spectra_metrics()
    all_rows.extend(spectra)
    print(f"  {len(spectra)} rows")

    print("Loading DRGP Bayes opt metrics...")
    drgp = load_drgp_bayes_opt_metrics()
    all_rows.extend(drgp)
    print(f"  {len(drgp)} rows")

    print("Loading DRGP full pipeline metrics...")
    drgp_full = load_drgp_full_metrics()
    all_rows.extend(drgp_full)
    print(f"  {len(drgp_full)} rows")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} rows")
    return df


def make_pivot_table(df, metric="val_auc"):
    """Create a pivot table: methods x (experiments, labels)."""
    pivot = df.pivot_table(
        index=["method", "method_type"],
        columns=["label", "experiment"],
        values=metric,
        aggfunc="first"
    )
    return pivot


def generate_latex_table(df):
    """Generate LaTeX table matching paper format."""
    # Define method display order and names
    method_order = [
        ("SVM", "Black-box"),
        ("LR", "Linear"),
        ("LRL", "Linear"),
        ("MFLR", "NMF + LR"),
        ("MFLRL", "NMF + LR"),
        ("MFLRR", "NMF + LR"),
        ("scHPF-LR", "scHPF + LR"),
        ("scHPF-L1", "scHPF + LR"),
        ("Spectra-LR", "Spectra + LR"),
        ("Spectra-L1", "Spectra + LR"),
        ("DRGP Normal/Combined", "DRGP (normal/combined)"),
        ("DRGP Normal/Masked", "DRGP (normal/masked)"),
        ("DRGP Normal/Unmasked", "DRGP (normal/unmasked)"),
        ("DRGP Laplace/Combined", "DRGP (laplace/combined)"),
        ("DRGP Laplace/Masked", "DRGP (laplace/masked)"),
        ("DRGP Laplace/Unmasked", "DRGP (laplace/unmasked)"),
        ("DRGP-Full Normal/Combined", "DRGP-Full (normal/combined)"),
        ("DRGP-Full Normal/Masked", "DRGP-Full (normal/masked)"),
        ("DRGP-Full Normal/Unmasked", "DRGP-Full (normal/unmasked)"),
        ("DRGP-Full Laplace/Combined", "DRGP-Full (laplace/combined)"),
        ("DRGP-Full Laplace/Masked", "DRGP-Full (laplace/masked)"),
        ("DRGP-Full Laplace/Unmasked", "DRGP-Full (laplace/unmasked)"),
    ]

    display_names = {
        "SVM": "SVM",
        "LR": r"LR$^\dagger$",
        "LRL": "LR-L1",
        "LRR": "LR-Ridge",
        "MFLR": r"MFLR$^\dagger$",
        "MFLRL": "MFLR-L1",
        "MFLRR": "MFLR-Ridge",
        "scHPF-LR": r"scHPF$^\dagger$",
        "scHPF-L1": "scHPF-L1",
        "Spectra-LR": r"Spectra$^\dagger$",
        "Spectra-L1": "Spectra-L1",
        "DRGP Normal/Combined": "DRGP",
        "DRGP Normal/Masked": "DRGP",
        "DRGP Normal/Unmasked": "DRGP",
        "DRGP Laplace/Combined": "DRGP",
        "DRGP Laplace/Masked": "DRGP",
        "DRGP Laplace/Unmasked": "DRGP",
        "DRGP-Full Normal/Combined": "DRGP",
        "DRGP-Full Normal/Masked": "DRGP",
        "DRGP-Full Normal/Unmasked": "DRGP",
        "DRGP-Full Laplace/Combined": "DRGP",
        "DRGP-Full Laplace/Masked": "DRGP",
        "DRGP-Full Laplace/Unmasked": "DRGP",
    }

    type_names = {
        "Black-box": "Black-box",
        "Linear": "Linear",
        "NMF + LR": "NMF + LR",
        "scHPF + LR": "scHPF + LR",
        "Spectra + LR": "Spectra + LR",
        "DRGP (normal/combined)": "Normal + Combined",
        "DRGP (normal/masked)": "Normal + Masked",
        "DRGP (normal/unmasked)": "Normal + Unmasked",
        "DRGP (laplace/combined)": "Laplace + Combined",
        "DRGP (laplace/masked)": "Laplace + Masked",
        "DRGP (laplace/unmasked)": "Laplace + Unmasked",
        "DRGP-Full (normal/combined)": "Normal + Combined*",
        "DRGP-Full (normal/masked)": "Normal + Masked*",
        "DRGP-Full (normal/unmasked)": "Normal + Unmasked*",
        "DRGP-Full (laplace/combined)": "Laplace + Combined*",
        "DRGP-Full (laplace/masked)": "Laplace + Masked*",
        "DRGP-Full (laplace/unmasked)": "Laplace + Unmasked*",
    }

    # Build data matrix
    data_rows = []
    for method, mtype in method_order:
        row_data = {"method": method, "display": display_names.get(method, method),
                     "type": type_names.get(mtype, mtype)}
        mask = df["method"] == method
        for label in LABELS:
            for exp in EXPERIMENTS:
                cell_mask = mask & (df["label"] == label) & (df["experiment"] == exp)
                vals = df.loc[cell_mask, "val_auc"]
                if len(vals) > 0:
                    row_data[f"{label}_{EXP_LABELS[exp]}"] = vals.iloc[0]
                else:
                    row_data[f"{label}_{EXP_LABELS[exp]}"] = None
        data_rows.append(row_data)

    # Find best per column
    best = {}
    for label in LABELS:
        for cl in ["C0", "C1", "C2", "C3"]:
            key = f"{label}_{cl}"
            vals = [(r[key], i) for i, r in enumerate(data_rows) if r[key] is not None]
            if vals:
                best_val = max(vals, key=lambda x: x[0])
                best[key] = best_val[1]

    # Generate LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Validation-set AUC for severity and outcome prediction on simulated data across four configurations of increasing difficulty.}")
    lines.append(r"\label{tab:prediction_results}")
    lines.append(r"\begin{tabular}{ll cccc cccc}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{4}{c}{\textbf{Severity (AUC)}} & \multicolumn{4}{c}{\textbf{Outcome (AUC)}} \\")
    lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    lines.append(r"\textbf{Method} & \textbf{Type} & C0 & C1 & C2 & C3 & C0 & C1 & C2 & C3 \\")
    lines.append(r"\midrule")

    # Define method groups for midrule placement
    group_boundaries = {"Black-box", "Linear", "NMF + LR", "scHPF + LR", "Spectra + LR"}
    prev_group = None
    for i, row in enumerate(data_rows):
        # Determine group
        rtype = row["type"]
        if any(rtype.startswith(g) for g in ["Normal", "Laplace"]):
            curr_group = "DRGP"
        else:
            curr_group = rtype
        if prev_group is not None and curr_group != prev_group:
            lines.append(r"\midrule")
        prev_group = curr_group

        cells = []
        for label in LABELS:
            for cl in ["C0", "C1", "C2", "C3"]:
                key = f"{label}_{cl}"
                val = row[key]
                if val is None:
                    cells.append("--")
                else:
                    formatted = f"{val:.3f}"  # e.g., "0.719"
                    if formatted.startswith("0"):
                        formatted = formatted[1:]  # ".719"
                    if best.get(key) == i:
                        formatted = r"\textbf{" + formatted + "}"
                    cells.append(formatted)

        line = f"{row['display']:20s} & {row['type']:22s} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    df = aggregate_all()

    # Save CSV
    csv_path = os.path.join(RESULTS_ROOT, "all_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Print summary pivot
    print("\n=== SEVERITY AUC (val) ===")
    for exp in EXPERIMENTS:
        exp_df = df[(df["experiment"] == exp) & (df["label"] == "severity")]
        print(f"\n{EXP_LABELS[exp]}:")
        for _, row in exp_df.iterrows():
            auc = row["val_auc"]
            if auc is not None:
                print(f"  {row['method']:30s} {auc:.4f}")

    print("\n=== OUTCOME AUC (val) ===")
    for exp in EXPERIMENTS:
        exp_df = df[(df["experiment"] == exp) & (df["label"] == "outcome")]
        print(f"\n{EXP_LABELS[exp]}:")
        for _, row in exp_df.iterrows():
            auc = row["val_auc"]
            if auc is not None:
                print(f"  {row['method']:30s} {auc:.4f}")

    # Generate LaTeX
    latex = generate_latex_table(df)
    latex_path = os.path.join(RESULTS_ROOT, "prediction_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nSaved LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
