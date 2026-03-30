#!/usr/bin/env python
"""
Prepare COVID experimental inputs for scHPF/Spectra downstream baselines.

Creates:
  - metadata_covid.csv      : index=cell_id, columns=[sex,comorbidity,severity,outcome,cell_type]
  - schpf_input/filtered.mtx: cell x gene sparse matrix (integer MatrixMarket)
  - schpf_input/genes.txt   : gene_id<TAB>gene_name (uses var names for both)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite


def _to_binary(series: pd.Series, pos_tokens: tuple[str, ...], neg_tokens: tuple[str, ...], default: int = 0) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    out = np.full(len(s), default, dtype=np.int32)
    out[s.isin(pos_tokens)] = 1
    out[s.isin(neg_tokens)] = 0
    return out


def _coerce_numeric_binary(series: pd.Series, default: int = 0) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce")
    out = np.full(len(series), default, dtype=np.int32)
    valid = ~x.isna()
    out[valid] = (x[valid].to_numpy() > 0).astype(np.int32)
    return out


def _build_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    idx = obs.index.astype(str)
    md = pd.DataFrame(index=idx)

    if "Sex" in obs.columns:
        sex_num = _coerce_numeric_binary(obs["Sex"], default=0)
        sex_txt = _to_binary(obs["Sex"], ("male", "m", "1", "true"), ("female", "f", "0", "false"), default=0)
        md["sex"] = np.where(pd.to_numeric(obs["Sex"], errors="coerce").notna(), sex_num, sex_txt).astype(np.int32)
    else:
        md["sex"] = 0

    cm_cols = [c for c in ["cm_asthma_copd", "cm_cardio", "cm_diabetes"] if c in obs.columns]
    if cm_cols:
        cm_vals = np.column_stack([_coerce_numeric_binary(obs[c], default=0) for c in cm_cols])
        md["comorbidity"] = (cm_vals.sum(axis=1) > 0).astype(np.int32)
    else:
        md["comorbidity"] = 0

    if "CoVID-19 severity" in obs.columns:
        sev_num = _coerce_numeric_binary(obs["CoVID-19 severity"], default=0)
        sev_txt = _to_binary(
            obs["CoVID-19 severity"],
            pos_tokens=("severe", "critical", "severe/critical", "1", "true"),
            neg_tokens=("mild", "moderate", "mild/moderate", "0", "false"),
            default=0,
        )
        md["severity"] = np.where(pd.to_numeric(obs["CoVID-19 severity"], errors="coerce").notna(), sev_num, sev_txt).astype(np.int32)
    else:
        md["severity"] = 0

    if "Outcome" in obs.columns:
        out_num = _coerce_numeric_binary(obs["Outcome"], default=0)
        out_txt = _to_binary(
            obs["Outcome"],
            pos_tokens=("deceased", "dead", "death", "1", "true"),
            neg_tokens=("discharged", "alive", "recovered", "0", "false"),
            default=0,
        )
        md["outcome"] = np.where(pd.to_numeric(obs["Outcome"], errors="coerce").notna(), out_num, out_txt).astype(np.int32)
    else:
        md["outcome"] = 0

    if "majorType" in obs.columns:
        md["cell_type"] = obs["majorType"].astype(str).fillna("unknown")
    elif "cell_type" in obs.columns:
        md["cell_type"] = obs["cell_type"].astype(str).fillna("unknown")
    else:
        md["cell_type"] = "unknown"

    # Patient/donor ID (for patient-grouped splitting)
    if "sampleID" in obs.columns:
        md["sampleID"] = obs["sampleID"].astype(str).values

    return md


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare COVID experimental inputs for scHPF/Spectra")
    p.add_argument("--h5ad", required=True, help="Path to COVID h5ad")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--write-mtx", action="store_true", help="Write MatrixMarket files for scHPF")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad)
    adata.var_names_make_unique()

    if "raw" in adata.layers:
        X = adata.layers["raw"]
    else:
        X = adata.X

    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    print(f"Counts matrix: {X.shape[0]} cells x {X.shape[1]} genes  (nnz={X.nnz:,})")

    metadata = _build_metadata(adata.obs.copy())
    metadata_path = out_root / "metadata_covid.csv"
    metadata.to_csv(metadata_path, index=True)
    print(f"Wrote metadata: {metadata_path}")

    if args.write_mtx:
        schpf_dir = out_root / "schpf_input"
        schpf_dir.mkdir(parents=True, exist_ok=True)

        X_int = X.copy()
        X_int.data = np.rint(np.clip(X_int.data, 0, None)).astype(np.int32, copy=False)
        mtx_path = schpf_dir / "filtered.mtx"
        mmwrite(mtx_path, X_int.tocoo(), field="integer")
        print(f"Wrote MatrixMarket: {mtx_path}")

        genes = pd.DataFrame({"gene_id": adata.var_names.astype(str), "gene_name": adata.var_names.astype(str)})
        genes_path = schpf_dir / "genes.txt"
        genes.to_csv(genes_path, sep="\t", header=False, index=False)
        print(f"Wrote genes file: {genes_path}")


if __name__ == "__main__":
    main()
