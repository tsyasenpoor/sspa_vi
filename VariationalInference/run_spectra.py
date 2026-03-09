#!/usr/bin/env python
"""
Fit Spectra on scdesign3 simulation experiments.
=================================================

Loads CSV, builds AnnData with log1p-normalized counts, runs Spectra
without cell-type-specific gene sets (use_cell_types=False), and
saves the resulting AnnData (with SPECTRA_cell_scores in .obsm).

Usage:
    python run_spectra.py \
        --data /path/to/exp/exp.csv.gz \
        --output-dir /path/to/results/spectra/exp \
        --n-factors 50
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import json
from datetime import datetime

# Spectra
import Spectra


META_COLS = ["sex", "comorbidity", "severity", "outcome", "cell_type"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit Spectra on simulation CSV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to experiment CSV (.csv.gz)")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory")
    p.add_argument("--n-factors", "-L", type=int, default=50,
                   help="Number of latent factors")
    p.add_argument("--lam", type=float, default=0.01,
                   help="Lambda: weight of graph vs expression loss")
    p.add_argument("--num-epochs", type=int, default=10000,
                   help="Training epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 80)
    print("SPECTRA MODEL FITTING")
    print("=" * 80)
    print(f"  Data:       {args.data}")
    print(f"  Factors:    {args.n_factors}")
    print(f"  Lambda:     {args.lam}")
    print(f"  Epochs:     {args.num_epochs}")

    # ------------------------------------------------------------------
    # 1. Load CSV → AnnData
    # ------------------------------------------------------------------
    print("\n[1] Loading CSV...")
    df = pd.read_csv(args.data, index_col=0)

    # Split metadata vs genes
    meta_cols = [c for c in META_COLS if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    meta = df[meta_cols].copy()
    counts = df[gene_cols].values.astype(np.float32)
    gene_names = gene_cols

    print(f"    {counts.shape[0]} cells x {counts.shape[1]} genes")
    print(f"    Metadata: {meta_cols}")

    # Build AnnData
    adata = sc.AnnData(
        X=counts,
        obs=meta,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = df.index.astype(str)
    adata.var_names = gene_names

    # ------------------------------------------------------------------
    # 2. Normalize: library-size normalize + log1p (Spectra expects this)
    # ------------------------------------------------------------------
    print("[2] Normalizing (library size + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Mark all genes as highly variable (pre-filtered to 2000 genes)
    adata.var["highly_variable"] = True

    # ------------------------------------------------------------------
    # 3. Run Spectra (no cell-type-specific gene sets)
    # ------------------------------------------------------------------
    print("[3] Fitting Spectra...")

    # Empty gene set dict: Spectra learns factors purely from expression.
    # Must be flat {} (not {"global": {}}) so compute_init_scores_noct
    # iterates zero keys and avoids mimno_coherence on empty gene lists.
    gene_set_dict = {}

    model = Spectra.est_spectra(
        adata=adata,
        gene_set_dictionary=gene_set_dict,
        L=args.n_factors,
        use_highly_variable=True,
        cell_type_key=None,
        use_cell_types=False,
        use_weights=False,  # must be False with empty gs_dict to avoid NaN from amatrix_weighted
        lam=args.lam,
        delta=0.001,
        kappa=None,
        rho=0.001,
        n_top_vals=50,
        label_factors=False,
        filter_sets=False,
        clean_gs=False,
        num_epochs=args.num_epochs,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    print("[4] Saving outputs...")
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Save full AnnData (contains SPECTRA_cell_scores in .obsm)
    h5ad_path = os.path.join(out, "spectra_adata.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"    Saved AnnData: {h5ad_path}")

    # Save cell scores separately as .npy for downstream classifier
    cell_scores = adata.obsm["SPECTRA_cell_scores"]
    np.save(os.path.join(out, "spectra_cell_scores.npy"), cell_scores)
    print(f"    Cell scores shape: {cell_scores.shape}")

    # Save factors
    if "SPECTRA_factors" in adata.uns:
        np.save(os.path.join(out, "spectra_factors.npy"), adata.uns["SPECTRA_factors"])

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_factors": int(cell_scores.shape[1]),
    }
    with open(os.path.join(out, "spectra_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Cell scores: {cell_scores.shape}")


if __name__ == "__main__":
    main()
