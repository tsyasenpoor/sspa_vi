#!/usr/bin/env python3
"""
Gene Program Recovery Analysis for scHPF, Spectra, and (later) DRGP.

Compares learned factors from each method against ground truth gene programs
from the simulation, using analysis approaches from the Spectra (Kunes et al. 2024)
and scHPF (Levitin et al. 2019) papers.

Usage:
    python analyze_program_recovery.py --exp exp0_easy
    python analyze_program_recovery.py --exp exp0_easy --methods schpf spectra
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Add scHPF to path
sys.path.insert(0, str(Path("/labs/Aguiar/SSPA_BRAY/scHPF")))

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/labs/Aguiar/SSPA_BRAY")
SIM_DATA = BASE / "scdesign3_PBMC_10kcells_2kgenes" / "synthetic_programs_pathway"
RESULTS = BASE / "results" / "sim_pathway"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Load ground truth and method outputs
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(exp: str):
    """Load all ground truth files for an experiment."""
    gt_dir = SIM_DATA / exp / "ground_truth"
    meta_path = SIM_DATA / exp / "metadata.csv"

    gt = {
        "beta": np.load(gt_dir / "beta_true.npy"),           # (P, 7)
        "alpha": np.load(gt_dir / "alpha_true.npy"),          # (N, 7)
        "v_true": np.load(gt_dir / "v_true.npy"),             # (5,)
        "disease_mask": np.load(gt_dir / "disease_mask.npy"), # (7,) bool
    }

    with open(gt_dir / "program_membership.json") as f:
        gt["membership"] = json.load(f)

    with open(gt_dir / "pathway_program_map.json") as f:
        gt["pathway_map"] = json.load(f)

    # program_names.txt is tab-separated: name\ttype\talignment\tpathway
    raw_lines = (gt_dir / "program_names.txt").read_text().strip().split("\n")
    gt["program_names"] = [line.split("\t")[0] for line in raw_lines]

    if meta_path.exists():
        gt["metadata"] = pd.read_csv(meta_path)

    # Load severity labels
    y_sev_path = SIM_DATA / exp / "y_severity.npy"
    if y_sev_path.exists():
        gt["y_severity"] = np.load(y_sev_path)
    elif "metadata" in gt and "severity" in gt["metadata"].columns:
        gt["y_severity"] = gt["metadata"]["severity"].values

    return gt


def load_schpf(exp: str):
    """Load scHPF model and extract factors."""
    model_dir = RESULTS / "schpf" / exp
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        print(f"  [WARN] No scHPF model found in {model_dir}")
        return None

    model = joblib.load(model_files[0])
    cell_scores = model.cell_score()  # (N, K) = theta * xi

    # Gene scores: E[eta_g] * E[beta_gk]  (scHPF paper definition)
    beta_raw = model.beta.e_x         # (P, K)
    eta_raw = model.eta.e_x           # (P,)
    gene_scores = beta_raw * eta_raw[:, None]  # (P, K)

    return {
        "name": "scHPF",
        "gene_scores": gene_scores,       # (P, K) - gene program matrix
        "cell_scores": cell_scores,       # (N, K) - cell factor loadings
        "beta_raw": beta_raw,
        "K": cell_scores.shape[1],
    }


def load_nmf(exp: str, K=50, seed=42):
    """Fit NMF on the expression data and extract factors."""
    from sklearn.decomposition import NMF

    data_path = SIM_DATA / exp / "X.npy"
    if not data_path.exists():
        print(f"  [WARN] No X.npy found in {SIM_DATA / exp}")
        return None

    X = np.load(data_path)  # (N, P)
    print(f"  Fitting NMF (K={K}) on X shape {X.shape}...")
    nmf = NMF(n_components=K, random_state=seed, max_iter=500)
    cell_scores = nmf.fit_transform(X)     # (N, K)
    gene_loadings = nmf.components_.T      # (K, P) -> (P, K)

    return {
        "name": "NMF",
        "gene_scores": gene_loadings,      # (P, K)
        "cell_scores": cell_scores,        # (N, K)
        "K": K,
    }


def load_spectra(exp: str):
    """Load Spectra factors and cell scores."""
    spec_dir = RESULTS / "spectra_sup" / exp
    factors_path = spec_dir / "spectra_factors.npy"
    scores_path = spec_dir / "spectra_cell_scores.npy"
    config_path = spec_dir / "spectra_config.json"

    if not factors_path.exists():
        print(f"  [WARN] No Spectra factors found in {spec_dir}")
        return None

    factors = np.load(factors_path)        # (K, P)
    cell_scores = np.load(scores_path)     # (N, K)

    with open(config_path) as f:
        config = json.load(f)
    pathway_names = config.get("gene_set_names", [])

    return {
        "name": "Spectra",
        "gene_scores": factors.T,          # (P, K) - transpose to match scHPF convention
        "cell_scores": cell_scores,        # (N, K)
        "K": factors.shape[0],
        "pathway_names": pathway_names,
    }


def load_drgp(exp: str, prior: str = "laplace", mode: str = "combined", seed: str = "seed42"):
    """Load DRGP full pipeline factors and cell scores."""
    run_dir = RESULTS / "drgp_full" / prior / mode / exp / seed
    gp_path = run_dir / "vi_gene_programs.csv.gz"

    if not gp_path.exists():
        print(f"  [WARN] No DRGP results found in {run_dir}")
        return None

    gp = pd.read_csv(gp_path)
    factor_names = gp["Unnamed: 0"].tolist()
    v_sev = gp["v_weight_severity"].values
    v_out = gp["v_weight_outcome"].values
    gene_cols = [c for c in gp.columns if c not in ["Unnamed: 0", "v_weight_outcome", "v_weight_severity"]]
    gene_scores = gp[gene_cols].values.T  # (K, P) -> transpose to (P, K)

    # Masked mode stores only pathway genes (e.g., 874). Expand to full 2000-gene space
    # so similarity/recovery uses the same gene basis as ground truth beta_true.
    if gene_scores.shape[0] != 2000:
        global_gene_list_path = BASE / "scdesign3_PBMC_10kcells_2kgenes" / "gene_names.txt"
        if not global_gene_list_path.exists():
            raise FileNotFoundError(f"Missing global gene name reference: {global_gene_list_path}")

        full_genes = global_gene_list_path.read_text().strip().splitlines()
        full_gene_to_idx = {g: i for i, g in enumerate(full_genes)}

        expanded = np.zeros((len(full_genes), gene_scores.shape[1]), dtype=gene_scores.dtype)
        matched = 0
        for local_idx, gene_name in enumerate(gene_cols):
            full_idx = full_gene_to_idx.get(gene_name)
            if full_idx is not None:
                expanded[full_idx, :] = gene_scores[local_idx, :]
                matched += 1

        gene_scores = expanded
        print(
            f"  [INFO] Expanded DRGP gene_scores to full space: "
            f"matched {matched}/{len(gene_cols)} genes, shape={gene_scores.shape}"
        )

    # Load cell scores from theta files
    meta_cols = ["cell_id", "sex", "severity", "outcome", "comorbidity", "cell_type"]
    theta_parts = []
    for split in ["train", "val", "test"]:
        theta_path = run_dir / f"vi_theta_{split}.csv.gz"
        if theta_path.exists():
            try:
                t = pd.read_csv(theta_path)
                factor_cols = [c for c in t.columns if c not in meta_cols]
                theta_parts.append(t[factor_cols].values)
            except Exception as e:
                print(f"  [WARN] Skipping corrupted {theta_path.name}: {e}")
    cell_scores = np.vstack(theta_parts) if theta_parts else np.empty((0, len(factor_names)))

    K = len(factor_names)
    print(f"  DRGP ({prior}/{mode}/{seed}): K={K}, gene_scores={gene_scores.shape}, cell_scores={cell_scores.shape}")

    return {
        "name": f"DRGP ({prior}/{mode})",
        "gene_scores": gene_scores,       # (P, K)
        "cell_scores": cell_scores,        # (N, K)
        "K": K,
        "pathway_names": factor_names,
        "v_weights": {"severity": v_sev, "outcome": v_out},
        "factor_names": factor_names,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Factor-to-program matching (cosine similarity + Hungarian)
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return np.dot(a, b) / (na * nb)


def compute_similarity_matrix(method_genes, gt_beta):
    """
    Compute cosine similarity between each method factor and each GT program.

    method_genes: (P, K_method) gene scores
    gt_beta: (P, 7) ground truth programs
    Returns: (K_method, 7) similarity matrix
    """
    K_method = method_genes.shape[1]
    n_programs = gt_beta.shape[1]
    sim = np.zeros((K_method, n_programs))
    for k in range(K_method):
        for p in range(n_programs):
            sim[k, p] = cosine_similarity(method_genes[:, k], gt_beta[:, p])
    return sim


def match_factors_to_programs(sim_matrix, program_names):
    """
    Match method factors to GT programs using:
    1. Max cosine similarity per GT program (best match, allows sharing)
    2. Hungarian algorithm for optimal 1:1 matching
    """
    K_method, n_programs = sim_matrix.shape

    # Best match (greedy, allows multiple factors to match same program)
    best_match_idx = np.argmax(sim_matrix, axis=0)      # (n_programs,)
    best_match_sim = np.max(sim_matrix, axis=0)          # (n_programs,)

    # Hungarian (1:1 optimal matching) - use negative sim as cost
    # Only match n_programs factors to n_programs programs
    cost = -sim_matrix  # (K_method, n_programs)
    row_ind, col_ind = linear_sum_assignment(cost)
    hungarian_match = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    hungarian_sim = {p: sim_matrix[hungarian_match[p], p] for p in range(n_programs)}

    results = []
    for p in range(n_programs):
        results.append({
            "program": program_names[p],
            "program_idx": p,
            "best_factor": int(best_match_idx[p]),
            "best_cosine": float(best_match_sim[p]),
            "hungarian_factor": int(hungarian_match.get(p, -1)),
            "hungarian_cosine": float(hungarian_sim.get(p, 0.0)),
        })
    return results


def direct_pathway_match(spectra_data, gt):
    """For Spectra: directly match factors by pathway name."""
    pathway_names = spectra_data.get("pathway_names", [])
    matches = {}
    for prog_idx_str, info in gt["pathway_map"].items():
        prog_idx = int(prog_idx_str)
        target_pathway = info["pathway_name"]
        if target_pathway and target_pathway in pathway_names:
            factor_idx = pathway_names.index(target_pathway)
            matches[prog_idx] = factor_idx
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Gene-level precision/recall
# ═══════════════════════════════════════════════════════════════════════════════

def overlap_coefficient(set_a, set_b):
    """Overlap coefficient = |A∩B| / min(|A|, |B|) (Spectra paper)."""
    if min(len(set_a), len(set_b)) == 0:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def gene_recovery_metrics(gene_scores_col, gt_gene_indices, top_ks=(10, 20, 50, 100)):
    """
    Compute gene recovery metrics for a single factor vs a GT program.

    gene_scores_col: (P,) gene scores for one factor
    gt_gene_indices: set of gene indices in the GT program
    """
    P = len(gene_scores_col)
    ranked = np.argsort(-np.abs(gene_scores_col))  # descending by |score|
    gt_set = set(gt_gene_indices)
    n_gt = len(gt_set)

    metrics = {"n_gt_genes": n_gt}
    for K in top_ks:
        top_k_set = set(ranked[:K].tolist())
        tp = len(top_k_set & gt_set)
        metrics[f"precision@{K}"] = tp / K if K > 0 else 0
        metrics[f"recall@{K}"] = tp / n_gt if n_gt > 0 else 0
        metrics[f"overlap@{K}"] = overlap_coefficient(top_k_set, gt_set)

    # AUROC: binary labels for GT membership
    y_true = np.zeros(P)
    for idx in gt_gene_indices:
        y_true[idx] = 1
    y_score = np.abs(gene_scores_col)
    if y_true.sum() > 0 and y_true.sum() < P:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
        metrics["avg_precision"] = average_precision_score(y_true, y_score)
    else:
        metrics["auroc"] = np.nan
        metrics["avg_precision"] = np.nan

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Factor separation of correlated programs
# ═══════════════════════════════════════════════════════════════════════════════

def factor_correlation_analysis(method_data, match_results, gt):
    """
    Check if correlated GT programs are separated or merged.
    Compute Pearson correlation between matched factor cell scores.
    """
    cell_scores = method_data["cell_scores"]
    n_programs = len(match_results)

    # Get matched factor indices
    matched_factors = [m["best_factor"] for m in match_results]

    # Correlation matrix for matched factors
    matched_cell_scores = cell_scores[:, matched_factors]  # (N, n_programs)
    corr = np.corrcoef(matched_cell_scores.T)  # (n_programs, n_programs)

    # Known correlated pairs from Table 3
    corr_pairs = [
        (0, 2, 0.7, "IFN-gamma <-> IFN-alpha/beta"),
        (1, 3, 0.5, "Chemokine <-> IL-10"),
        (0, 1, 0.4, "IFN-gamma <-> Chemokine"),
        (4, 0, 0.3, "T-cell exhaustion <-> IFN-gamma"),
    ]

    pair_results = []
    for p1, p2, gt_rho, label in corr_pairs:
        if p1 < n_programs and p2 < n_programs:
            learned_rho = corr[p1, p2]
            pair_results.append({
                "pair": label,
                "gt_rho": gt_rho,
                "learned_rho": float(learned_rho),
                "factor1": matched_factors[p1],
                "factor2": matched_factors[p2],
            })

    return corr, pair_results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Disease program identification via LR coefficients
# ═══════════════════════════════════════════════════════════════════════════════

def disease_program_identification(method_data, match_results, gt):
    """
    Check if downstream LR assigns high coefficients to disease-relevant factors.
    Uses cell scores + severity labels to fit a simple LR and extract coefficients.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cell_scores = method_data["cell_scores"]
    y = gt.get("y_severity")
    if y is None:
        return None

    # Use all cell scores (not just matched factors)
    scaler = StandardScaler()
    X = scaler.fit_transform(cell_scores)

    lr = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=42)
    lr.fit(X, y)
    coefs = np.abs(lr.coef_[0])  # (K,)

    # For each GT program, check if its matched factor has a high coefficient
    matched_factors = [m["best_factor"] for m in match_results]
    disease_mask = gt["disease_mask"]

    results = []
    for i, m in enumerate(match_results):
        results.append({
            "program": m["program"],
            "factor_idx": m["best_factor"],
            "lr_coef_abs": float(coefs[m["best_factor"]]),
            "disease_relevant": bool(disease_mask[i]),
            "cosine_sim": m["best_cosine"],
        })

    # Rank factors by |coef| and check precision@5
    ranked_factors = np.argsort(-coefs)
    disease_factors = set(matched_factors[i] for i in range(len(disease_mask)) if disease_mask[i])
    top5 = set(ranked_factors[:5].tolist())
    precision_at_5 = len(top5 & disease_factors) / 5

    return {
        "per_program": results,
        "precision_at_5": precision_at_5,
        "all_coefs": coefs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Cell scores stratified by severity
# ═══════════════════════════════════════════════════════════════════════════════

def severity_stratification(method_data, match_results, gt):
    """Compare cell score distributions between severe and mild cells."""
    y = gt.get("y_severity")
    if y is None:
        return None

    cell_scores = method_data["cell_scores"]
    severe_mask = y == 1
    mild_mask = y == 0

    results = []
    for m in match_results:
        factor_idx = m["best_factor"]
        scores_severe = cell_scores[severe_mask, factor_idx]
        scores_mild = cell_scores[mild_mask, factor_idx]

        stat, pval = mannwhitneyu(scores_severe, scores_mild, alternative="two-sided")

        results.append({
            "program": m["program"],
            "factor_idx": factor_idx,
            "mean_severe": float(np.mean(scores_severe)),
            "mean_mild": float(np.mean(scores_mild)),
            "fold_change": float(np.mean(scores_severe) / max(np.mean(scores_mild), 1e-12)),
            "mannwhitney_stat": float(stat),
            "pvalue": float(pval),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 & 8: Free program recovery and noise rejection
# ═══════════════════════════════════════════════════════════════════════════════

def free_program_analysis(method_data, gt):
    """
    Check recovery of T-cell exhaustion (free, disease-relevant)
    and rejection of noise programs (ribosomal, stress).
    """
    gene_scores = method_data["gene_scores"]  # (P, K)
    membership = gt["membership"]

    results = {}
    for prog_idx, prog_name in enumerate(gt["program_names"]):
        prog_info = membership[prog_name]
        gene_indices = prog_info["gene_idx"]

        # Find best matching factor by overlap
        K = gene_scores.shape[1]
        best_overlap = 0.0
        best_factor = -1
        best_metrics = {}

        for k in range(K):
            ranked = np.argsort(-np.abs(gene_scores[:, k]))
            top50 = set(ranked[:50].tolist())
            gt_set = set(gene_indices)
            ov = overlap_coefficient(top50, gt_set)
            if ov > best_overlap:
                best_overlap = ov
                best_factor = k
                best_metrics = gene_recovery_metrics(gene_scores[:, k], gene_indices)

        results[prog_name] = {
            "prog_idx": prog_idx,
            "best_factor": best_factor,
            "best_overlap@50": best_overlap,
            "is_pathway_aligned": gt["pathway_map"][str(prog_idx)]["is_pathway_aligned"],
            "disease_relevant": gt["pathway_map"][str(prog_idx)]["disease_relevant"],
            "metrics": best_metrics,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cosine_heatmap(sim_matrix, match_results, method_name, program_names, out_dir):
    """Heatmap of cosine similarity between top-matched factors and GT programs."""
    matched_factors = [m["best_factor"] for m in match_results]
    sub_sim = sim_matrix[matched_factors, :]  # (7, 7)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sub_sim, cmap="RdBu_r", vmin=-0.1, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(program_names)))
    ax.set_xticklabels([p.replace("_", "\n") for p in program_names], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matched_factors)))
    ax.set_yticklabels([f"Factor {f}" for f in matched_factors], fontsize=8)
    ax.set_xlabel("Ground Truth Program")
    ax.set_ylabel(f"{method_name} Factor")
    ax.set_title(f"{method_name}: Cosine Similarity (matched factors vs GT programs)")

    for i in range(sub_sim.shape[0]):
        for j in range(sub_sim.shape[1]):
            ax.text(j, i, f"{sub_sim[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if sub_sim[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fig.savefig(out_dir / f"cosine_heatmap_{safe_name}.pdf", dpi=150)
    plt.close(fig)


def plot_factor_correlation(corr, program_names, method_name, out_dir):
    """Factor-factor correlation heatmap for matched factors (Spectra Fig 3d style)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(program_names)))
    ax.set_xticklabels([p.replace("_", "\n") for p in program_names], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(program_names)))
    ax.set_yticklabels([p.replace("_", "\n") for p in program_names], fontsize=7)
    ax.set_title(f"{method_name}: Factor-Factor Correlation (matched)")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fig.savefig(out_dir / f"factor_correlation_{safe_name}.pdf", dpi=150)
    plt.close(fig)


def plot_severity_boxplots(sev_results, method_data, gt, method_name, out_dir):
    """Box plots of cell scores by severity for each matched program (Spectra Fig 3f style)."""
    if sev_results is None:
        return

    y = gt["y_severity"]
    cell_scores = method_data["cell_scores"]
    n_progs = len(sev_results)

    fig, axes = plt.subplots(1, n_progs, figsize=(3 * n_progs, 4), sharey=False)
    if n_progs == 1:
        axes = [axes]

    for i, sr in enumerate(sev_results):
        ax = axes[i]
        fidx = sr["factor_idx"]
        severe_scores = cell_scores[y == 1, fidx]
        mild_scores = cell_scores[y == 0, fidx]

        bp = ax.boxplot([mild_scores, severe_scores], labels=["Mild", "Severe"],
                        patch_artist=True, showfliers=False, widths=0.6)
        bp["boxes"][0].set_facecolor("#4DBEEE")
        bp["boxes"][1].set_facecolor("#D95319")

        pval_str = f"p={sr['pvalue']:.2e}" if sr["pvalue"] < 0.001 else f"p={sr['pvalue']:.3f}"
        ax.set_title(f"{sr['program']}\n{pval_str}", fontsize=7)
        ax.set_ylabel("Cell Score" if i == 0 else "")
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{method_name}: Cell Scores by Severity", fontsize=10)
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    fig.savefig(out_dir / f"severity_boxplots_{safe_name}.pdf", dpi=150)
    plt.close(fig)


def plot_gene_recovery_summary(all_recovery, program_names, out_dir):
    """Bar plot comparing gene recovery across methods (Spectra Fig 2e style)."""
    methods = list(all_recovery.keys())
    n_progs = len(program_names)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for metric_idx, metric_name in enumerate(["recall@50", "auroc"]):
        ax = axes[metric_idx]
        x = np.arange(n_progs)
        width = 0.8 / len(methods)

        for m_idx, method in enumerate(methods):
            vals = []
            for p in range(n_progs):
                pname = program_names[p]
                r = all_recovery[method].get(pname, {}).get("metrics", {})
                vals.append(r.get(metric_name, 0))

            offset = (m_idx - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=method, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in program_names], fontsize=7)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Gene Recovery: {metric_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "gene_recovery_summary.pdf", dpi=150)
    plt.close(fig)


def plot_overlap_comparison(all_recovery, program_names, out_dir):
    """Box plot of max overlap coefficients per method (Spectra Fig 2c style)."""
    methods = list(all_recovery.keys())

    fig, ax = plt.subplots(figsize=(6, 4))
    data = []
    labels = []
    for method in methods:
        overlaps = []
        for pname in program_names:
            ov = all_recovery[method].get(pname, {}).get("best_overlap@50", 0)
            overlaps.append(ov)
        data.append(overlaps)
        labels.append(method)

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    colors = ["#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.7)

    ax.set_ylabel("Overlap Coefficient @50")
    ax.set_title("Factor Interpretability: Max Overlap with GT Programs")
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.3, label="Spectra threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "overlap_comparison.pdf", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_method(method_data, gt, out_dir):
    """Run full analysis pipeline for one method."""
    name = method_data["name"]
    print(f"\n{'='*60}")
    print(f"  Analyzing: {name} (K={method_data['K']})")
    print(f"{'='*60}")

    gene_scores = method_data["gene_scores"]  # (P, K)
    program_names = gt["program_names"]
    gt_beta = gt["beta"]  # (P, 7)

    # ── Step 2: Factor matching ──────────────────────────────────
    print("\n  [Step 2] Factor-to-program matching...")
    sim_matrix = compute_similarity_matrix(gene_scores, gt_beta)
    match_results = match_factors_to_programs(sim_matrix, program_names)

    print(f"  {'Program':<35s} {'Best Factor':>12s} {'Cosine':>8s}")
    print(f"  {'-'*55}")
    for m in match_results:
        marker = " *" if gt["disease_mask"][m["program_idx"]] else ""
        print(f"  {m['program']:<35s} {m['best_factor']:>12d} {m['best_cosine']:>8.4f}{marker}")

    # For Spectra: also do direct pathway name matching
    direct_matches = None
    if "pathway_names" in method_data:
        direct_matches = direct_pathway_match(method_data, gt)
        print(f"\n  Direct pathway matches: {direct_matches}")
        # Override match_results for pathway-aligned programs
        for prog_idx, factor_idx in direct_matches.items():
            match_results[prog_idx]["direct_factor"] = factor_idx
            match_results[prog_idx]["direct_cosine"] = float(
                cosine_similarity(gene_scores[:, factor_idx], gt_beta[:, prog_idx])
            )

    plot_cosine_heatmap(sim_matrix, match_results, name, program_names, out_dir)

    # ── Step 3: Gene recovery ────────────────────────────────────
    print("\n  [Step 3] Gene-level recovery...")
    recovery = {}
    for m in match_results:
        prog_idx = m["program_idx"]
        prog_name = m["program"]
        gene_indices = gt["membership"][prog_name]["gene_idx"]
        factor_idx = m["best_factor"]

        metrics = gene_recovery_metrics(gene_scores[:, factor_idx], gene_indices)
        recovery[prog_name] = {
            "metrics": metrics,
            "factor_idx": factor_idx,
        }
        print(f"    {prog_name:<35s} P@50={metrics['precision@50']:.3f}  "
              f"R@50={metrics['recall@50']:.3f}  AUROC={metrics['auroc']:.3f}")

    # ── Step 4: Factor separation ────────────────────────────────
    print("\n  [Step 4] Factor correlation analysis...")
    corr, pair_results = factor_correlation_analysis(method_data, match_results, gt)
    for pr in pair_results:
        print(f"    {pr['pair']:<40s} GT={pr['gt_rho']:.2f}  Learned={pr['learned_rho']:.3f}")

    plot_factor_correlation(corr, program_names, name, out_dir)

    # ── Step 5: Disease program identification ───────────────────
    print("\n  [Step 5] Disease program identification via LR...")
    disease_id = disease_program_identification(method_data, match_results, gt)
    if disease_id:
        print(f"    Precision@5 for disease programs: {disease_id['precision_at_5']:.2f}")
        for pp in disease_id["per_program"]:
            marker = " *" if pp["disease_relevant"] else ""
            print(f"    {pp['program']:<35s} |coef|={pp['lr_coef_abs']:.4f}{marker}")

    # ── Step 6: Severity stratification ──────────────────────────
    print("\n  [Step 6] Severity stratification...")
    sev_results = severity_stratification(method_data, match_results, gt)
    if sev_results:
        for sr in sev_results:
            sig = "***" if sr["pvalue"] < 0.001 else ("**" if sr["pvalue"] < 0.01 else ("*" if sr["pvalue"] < 0.05 else ""))
            print(f"    {sr['program']:<35s} FC={sr['fold_change']:.3f}  p={sr['pvalue']:.2e} {sig}")

    plot_severity_boxplots(sev_results, method_data, gt, name, out_dir)

    # ── Steps 7 & 8: Free program and noise analysis ─────────────
    print("\n  [Steps 7-8] Free program recovery & noise rejection...")
    free_results = free_program_analysis(method_data, gt)
    for prog_name, fr in free_results.items():
        aligned = "pathway" if fr["is_pathway_aligned"] else "FREE"
        disease = "DISEASE" if fr["disease_relevant"] else "noise"
        print(f"    {prog_name:<35s} [{aligned:>7s}] [{disease:>7s}]  "
              f"overlap@50={fr['best_overlap@50']:.3f}  "
              f"AUROC={fr['metrics'].get('auroc', np.nan):.3f}")

    # Merge free_results into recovery for unified output
    for prog_name, fr in free_results.items():
        if prog_name not in recovery:
            recovery[prog_name] = fr
        else:
            recovery[prog_name].update(fr)

    return {
        "match_results": match_results,
        "recovery": recovery,
        "correlation": {"matrix": corr, "pairs": pair_results},
        "disease_id": disease_id,
        "severity": sev_results,
        "free_programs": free_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Gene Program Recovery Analysis")
    parser.add_argument("--exp", default="exp0_easy", help="Experiment name")
    parser.add_argument("--methods", nargs="+", default=["schpf", "spectra"],
                        choices=["schpf", "spectra", "nmf", "drgp"], help="Methods to analyze")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--prior", default="laplace", choices=["normal", "laplace"],
                        help="DRGP prior (default: laplace)")
    parser.add_argument("--mode", default="combined", choices=["combined", "masked", "unmasked"],
                        help="DRGP mode (default: combined)")
    parser.add_argument("--seed", default="seed42", help="DRGP seed directory (default: seed42)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS / "program_recovery_figures" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.exp}")
    print(f"Methods:    {args.methods}")
    print(f"Output:     {out_dir}")

    # Load ground truth
    print("\n[Loading ground truth...]")
    gt = load_ground_truth(args.exp)
    print(f"  beta shape:     {gt['beta'].shape}")
    print(f"  alpha shape:    {gt['alpha'].shape}")
    print(f"  programs:       {gt['program_names']}")
    print(f"  disease mask:   {gt['disease_mask']}")
    print(f"  v_true:         {gt['v_true']}")

    # Load and analyze each method
    all_results = {}
    all_recovery = {}

    loaders = {
        "schpf": lambda exp: load_schpf(exp),
        "spectra": lambda exp: load_spectra(exp),
        "nmf": lambda exp: load_nmf(exp),
        "drgp": lambda exp: load_drgp(exp, args.prior, args.mode, args.seed),
    }
    for method_name in args.methods:
        print(f"\n[Loading {method_name}...]")
        method_data = loaders[method_name](args.exp)
        if method_data is None:
            continue
        print(f"  gene_scores shape: {method_data['gene_scores'].shape}")
        print(f"  cell_scores shape: {method_data['cell_scores'].shape}")

        results = analyze_method(method_data, gt, out_dir)
        all_results[method_data["name"]] = results
        all_recovery[method_data["name"]] = results["free_programs"]

    # Cross-method comparison plots
    print(f"\n{'='*60}")
    print("  Cross-method comparison plots")
    print(f"{'='*60}")
    plot_gene_recovery_summary(all_recovery, gt["program_names"], out_dir)
    plot_overlap_comparison(all_recovery, gt["program_names"], out_dir)

    # Save metrics CSV
    rows = []
    for method, results in all_results.items():
        for m in results["match_results"]:
            prog_name = m["program"]
            rec = results["recovery"].get(prog_name, {})
            metrics = rec.get("metrics", {})
            row = {
                "method": method,
                "program": prog_name,
                "best_factor": m["best_factor"],
                "cosine_sim": m["best_cosine"],
                "overlap@50": rec.get("best_overlap@50", np.nan),
                "precision@50": metrics.get("precision@50", np.nan),
                "recall@50": metrics.get("recall@50", np.nan),
                "auroc": metrics.get("auroc", np.nan),
                "avg_precision": metrics.get("avg_precision", np.nan),
                "disease_relevant": bool(gt["disease_mask"][m["program_idx"]]),
                "is_pathway_aligned": gt["pathway_map"][str(m["program_idx"])]["is_pathway_aligned"],
            }
            # Add severity info if available
            sev = results.get("severity")
            if sev:
                for sr in sev:
                    if sr["program"] == prog_name:
                        row["severity_pvalue"] = sr["pvalue"]
                        row["severity_fc"] = sr["fold_change"]
            rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "program_recovery_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False, float_format="%.3f"))

    print(f"\nFigures saved to {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
