#!/usr/bin/env python
"""
Fit Spectra in SUPERVISED mode using pathway gene sets.
========================================================

Loads an h5ad file, uses load_pathways() from utils.py with REACTOME filtering
(same as DRGP), and runs Spectra with pathway-informed factorization.

Usage:
    python run_spectra_supervised.py \
        --h5ad /path/to/covid_subsample.h5ad \
        --gmt-file /archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt \
        --output-dir /path/to/results/spectra_sup \
        --seed 42
"""
import os
import sys
from pathlib import Path

# Add BRay/ to path for VariationalInference imports (comp/ → VI/ → BRay/)
script_dir = Path(__file__).resolve().parent
bray_dir = script_dir.parent.parent
if str(bray_dir) not in sys.path:
    sys.path.insert(0, str(bray_dir))

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import json
from datetime import datetime
import anndata as ad
import scipy.sparse as sp

import Spectra


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit Spectra in supervised mode with pathway gene sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5ad", default=None, help="Path to h5ad file (real data)")
    p.add_argument("--data", default=None,
                   help="Path to CSV/CSV.gz (sim data: genes as columns, metadata merged)")
    p.add_argument("--gmt-file", default="/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt",
                   help="Path to GMT file with pathway gene sets")
    p.add_argument("--output-dir", "-o", required=True, help="Output directory")
    p.add_argument("--require-prefix", default="REACTOME",
                   help="Only keep pathways starting with this prefix (set to empty to keep all)")
    p.add_argument("--min-genes", type=int, default=0,
                   help="Minimum genes per pathway after filtering")
    p.add_argument("--max-genes", type=int, default=200000,
                   help="Maximum genes per pathway")
    p.add_argument("--n-factors", "-L", type=int, default=None,
                   help="Number of latent factors (default: auto from gene set count)")
    p.add_argument("--lam", type=float, default=0.01,
                   help="Lambda: weight of graph vs expression loss")
    p.add_argument("--num-epochs", type=int, default=10000,
                   help="Training epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    # On-the-fly subsampling (for scalability benchmarking without saving to disk)
    p.add_argument("--subsample-ratio", type=float, default=None,
                   help="Subsample dataset to this fraction of patients before processing")
    p.add_argument("--subsample-n-patients", type=int, default=None,
                   help="Subsample dataset to exactly this many patients")
    p.add_argument("--subsample-seed", type=int, default=0,
                   help="Seed for patient-level subsampling")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.h5ad is None and args.data is None:
        raise ValueError("Must provide either --h5ad or --data")

    input_path = args.h5ad or args.data
    print("=" * 80)
    print("SPECTRA MODEL FITTING (SUPERVISED / PATHWAY-INFORMED)")
    print("=" * 80)
    print(f"  Input:      {input_path}")
    print(f"  GMT:        {args.gmt_file}")
    print(f"  Prefix:     {args.require_prefix}")
    print(f"  Lambda:     {args.lam}")
    print(f"  Epochs:     {args.num_epochs}")

    # ------------------------------------------------------------------
    # 1. Load data (h5ad or CSV)
    # ------------------------------------------------------------------
    if args.data is not None:
        # --- CSV path (simulation data) ---
        print(f"\n[1] Loading CSV: {args.data}")
        df = pd.read_csv(args.data, index_col=0)
        meta_cols = ["severity", "outcome", "sex", "comorbidity", "cell_type"]
        present_meta = [c for c in meta_cols if c in df.columns]
        gene_cols = [c for c in df.columns if c not in meta_cols]

        X = df[gene_cols].values.astype(np.float32)
        meta = df[present_meta].copy() if present_meta else pd.DataFrame(index=df.index)
        for c in meta_cols:
            if c not in meta.columns:
                meta[c] = "unknown" if c == "cell_type" else np.int32(0)

        adata = sc.AnnData(
            X=X,
            obs=meta,
            var=pd.DataFrame(index=gene_cols),
        )
        adata.obs_names = df.index.astype(str)
        adata.var_names = pd.Index(gene_cols)
        print(f"    {adata.n_obs} cells x {adata.n_vars} genes")

    else:
        # --- H5AD path (real data) ---
        print(f"\n[1] Loading h5ad: {args.h5ad}")
        src = ad.read_h5ad(args.h5ad)
        src.var_names_make_unique()

        # On-the-fly subsampling (avoids saving large subsample h5ad files to disk)
        if args.subsample_ratio is not None or args.subsample_n_patients is not None:
            from VariationalInference.create_subsamples import subsample_adata
            print(f"[SUBSAMPLE] On-the-fly subsampling "
                  f"(ratio={args.subsample_ratio}, n_patients={args.subsample_n_patients}, "
                  f"seed={args.subsample_seed})")
            src = subsample_adata(
                src,
                ratio=args.subsample_ratio,
                n_patients=args.subsample_n_patients,
                subsample_seed=args.subsample_seed,
                verbose=True,
            )

        X = src.layers["raw"] if "raw" in src.layers else src.X
        if sp.issparse(X):
            X = X.tocsr().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)

        obs = src.obs.copy()
        meta = pd.DataFrame(index=obs.index.astype(str))

        # Sex: M=1, F=0
        if "Sex" in obs.columns:
            meta["sex"] = (obs["Sex"].astype(str).str.strip() == "M").astype(np.int32)
        else:
            meta["sex"] = np.int32(0)

        # Comorbidity
        cm_cols = [c for c in ["cm_asthma_copd", "cm_cardio", "cm_diabetes"] if c in obs.columns]
        if cm_cols:
            cm_vals = np.column_stack([
                (obs[c].astype(str).str.strip() == "1").astype(np.int32) for c in cm_cols
            ])
            meta["comorbidity"] = (cm_vals.sum(axis=1) > 0).astype(np.int32)
        else:
            meta["comorbidity"] = np.int32(0)

        # Severity: severe/critical=1
        if "CoVID-19 severity" in obs.columns:
            meta["severity"] = (obs["CoVID-19 severity"].astype(str).str.strip() == "severe/critical").astype(np.int32)
        else:
            meta["severity"] = np.int32(0)

        # Outcome: deceased=1
        if "Outcome" in obs.columns:
            meta["outcome"] = (obs["Outcome"].astype(str).str.strip() == "deceased").astype(np.int32)
        else:
            meta["outcome"] = np.int32(0)

        # Cell type
        if "majorType" in obs.columns:
            meta["cell_type"] = obs["majorType"].astype(str)
        elif "cell_type" in obs.columns:
            meta["cell_type"] = obs["cell_type"].astype(str)
        else:
            meta["cell_type"] = "unknown"

        adata = sc.AnnData(
            X=X,
            obs=meta,
            var=pd.DataFrame(index=src.var_names.astype(str)),
        )
        adata.obs_names = src.obs_names.astype(str)
        adata.var_names = src.var_names.astype(str)
        print(f"    {adata.n_obs} cells x {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # 2. Load pathways using same logic as DRGP (utils.load_pathways)
    #    but WITHOUT Ensembl conversion since Spectra uses gene symbols
    # ------------------------------------------------------------------
    print("\n[2] Loading pathways (same filtering as DRGP)...")
    from VariationalInference.utils import load_pathways

    require_prefix = args.require_prefix if args.require_prefix else None

    pathway_mat, pathway_names, pathway_genes = load_pathways(
        gmt_path=args.gmt_file,
        convert_to_ensembl=False,  # Spectra uses gene symbols, not Ensembl
        species='human',
        gene_filter=list(adata.var_names),
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        use_cache=True,
        require_prefix=require_prefix,
    )

    print(f"    Pathways: {len(pathway_names)}, Genes in pathways: {len(pathway_genes)}")

    # Convert binary matrix to gene_set_dictionary for Spectra
    # Format: {pathway_name: [gene_symbol_1, gene_symbol_2, ...]}
    gene_set_dict = {}
    for i, pw_name in enumerate(pathway_names):
        gene_indices = np.where(pathway_mat[i] > 0)[0]
        genes = [pathway_genes[j] for j in gene_indices]
        if genes:
            gene_set_dict[pw_name] = genes

    print(f"    Gene set dict: {len(gene_set_dict)} entries")

    if len(gene_set_dict) == 0:
        raise ValueError("No gene sets remain after filtering. Check GMT file and prefix filter.")

    sizes = [len(v) for v in gene_set_dict.values()]
    print(f"    Genes/set: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # Determine L (number of factors)
    L = args.n_factors if args.n_factors is not None else len(gene_set_dict)
    print(f"    L (n_factors): {L}")

    # ------------------------------------------------------------------
    # 3. Normalize: library-size normalize + log1p (Spectra expects this)
    # ------------------------------------------------------------------
    print("\n[3] Normalizing (library size + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var["highly_variable"] = True

    # ------------------------------------------------------------------
    # 4. Run Spectra with gene set dictionary (supervised mode)
    # ------------------------------------------------------------------
    print(f"\n[4] Fitting Spectra (supervised, {len(gene_set_dict)} gene sets, L={L})...")

    model = Spectra.est_spectra(
        adata=adata,
        gene_set_dictionary=gene_set_dict,
        L=L,
        use_highly_variable=False,
        cell_type_key=None,
        use_cell_types=False,
        use_weights=True,
        lam=args.lam,
        delta=0.001,
        kappa=None,
        rho=0.001,
        n_top_vals=50,
        label_factors=True,
        filter_sets=True,
        clean_gs=True,
        num_epochs=args.num_epochs,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    print("\n[5] Saving outputs...")
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Save full AnnData
    h5ad_path = os.path.join(out, "spectra_adata.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"    Saved AnnData: {h5ad_path}")

    # Save cell scores
    cell_scores = adata.obsm["SPECTRA_cell_scores"]
    np.save(os.path.join(out, "spectra_cell_scores.npy"), cell_scores)
    print(f"    Cell scores shape: {cell_scores.shape}")

    # Save factors
    if "SPECTRA_factors" in adata.uns:
        np.save(os.path.join(out, "spectra_factors.npy"), adata.uns["SPECTRA_factors"])

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "mode": "supervised",
        "args": vars(args),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_gene_sets_input": len(gene_set_dict),
        "n_factors": int(cell_scores.shape[1]),
        "gene_set_names": list(gene_set_dict.keys()),
    }
    with open(os.path.join(out, "spectra_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Cell scores: {cell_scores.shape}")


if __name__ == "__main__":
    main()
