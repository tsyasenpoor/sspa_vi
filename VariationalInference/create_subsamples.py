#!/usr/bin/env python
"""
Create stratified subsamples of the COVID-19 h5ad for scalability benchmarking.
===============================================================================

Samples at the PATIENT level (sampleID) to preserve within-patient cell
composition. Stratification is on severity x outcome at the patient level.

Usage:
    python create_subsamples.py \
        --h5ad /path/to/covid19_filtered_fullgenes_clean.h5ad \
        --output-root /path/to/results/scalability_benchmark/subsamples \
        --ratios 0.1 0.25 0.5 0.75 \
        --subsample-seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
ad.settings.allow_write_nullable_strings = True
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from sklearn.model_selection import train_test_split


# Exact label values in the dataset
SEVERITY_POS = "severe/critical"   # -> 1
SEVERITY_NEG = "mild/moderate"     # -> 0
OUTCOME_POS = "deceased"           # -> 1
OUTCOME_NEG = "discharged"         # -> 0
SEX_MALE = "M"                    # -> 1
SEX_FEMALE = "F"                  # -> 0
CELL_TYPES = ["B", "CD4", "CD8", "Mono", "NK", "myeloid cells"]


def _build_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    """Extract binary metadata columns from h5ad obs."""
    idx = obs.index.astype(str)
    md = pd.DataFrame(index=idx)

    # Sex: M=1, F=0
    if "Sex" in obs.columns:
        md["sex"] = (obs["Sex"].astype(str).str.strip() == SEX_MALE).astype(np.int32)
    else:
        md["sex"] = np.int32(0)

    # Comorbidity: any of cm_asthma_copd, cm_cardio, cm_diabetes > 0
    cm_cols = [c for c in ["cm_asthma_copd", "cm_cardio", "cm_diabetes"] if c in obs.columns]
    if cm_cols:
        cm_vals = np.column_stack([
            (obs[c].astype(str).str.strip() == "1").astype(np.int32)
            for c in cm_cols
        ])
        md["comorbidity"] = (cm_vals.sum(axis=1) > 0).astype(np.int32)
    else:
        md["comorbidity"] = np.int32(0)

    # Severity: severe/critical=1, mild/moderate=0
    if "CoVID-19 severity" in obs.columns:
        md["severity"] = (obs["CoVID-19 severity"].astype(str).str.strip() == SEVERITY_POS).astype(np.int32)
    else:
        md["severity"] = np.int32(0)

    # Outcome: deceased=1, discharged=0
    if "Outcome" in obs.columns:
        md["outcome"] = (obs["Outcome"].astype(str).str.strip() == OUTCOME_POS).astype(np.int32)
    else:
        md["outcome"] = np.int32(0)

    # Cell type
    if "majorType" in obs.columns:
        md["cell_type"] = obs["majorType"].astype(str).fillna("unknown")
    elif "cell_type" in obs.columns:
        md["cell_type"] = obs["cell_type"].astype(str).fillna("unknown")
    else:
        md["cell_type"] = "unknown"

    return md


def _write_schpf_inputs(adata_sub: ad.AnnData, out_dir: Path) -> None:
    """Write MatrixMarket + genes.txt for scHPF."""
    schpf_dir = out_dir / "schpf_input"
    schpf_dir.mkdir(parents=True, exist_ok=True)

    X = adata_sub.layers["raw"] if "raw" in adata_sub.layers else adata_sub.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    X_int = X.copy()
    X_int.data = np.rint(np.clip(X_int.data, 0, None)).astype(np.int32, copy=False)
    mmwrite(schpf_dir / "filtered.mtx", X_int.tocoo(), field="integer")

    genes = pd.DataFrame({
        "gene_id": adata_sub.var_names.astype(str),
        "gene_name": adata_sub.var_names.astype(str),
    })
    genes.to_csv(schpf_dir / "genes.txt", sep="\t", header=False, index=False)


def _get_patient_labels(adata: ad.AnnData) -> pd.DataFrame:
    """Get per-patient labels for stratification.

    Returns DataFrame indexed by sampleID with severity, outcome, and composite stratum.
    """
    obs = adata.obs.copy()
    patient_df = obs.groupby("sampleID").first()[["CoVID-19 severity", "Outcome"]]
    patient_df["severity_bin"] = (patient_df["CoVID-19 severity"].astype(str).str.strip() == SEVERITY_POS).astype(int)
    patient_df["outcome_bin"] = (patient_df["Outcome"].astype(str).str.strip() == OUTCOME_POS).astype(int)
    patient_df["stratum"] = patient_df["severity_bin"].astype(str) + "_" + patient_df["outcome_bin"].astype(str)
    return patient_df


def subsample_adata(
    adata: ad.AnnData,
    ratio: float = None,
    n_patients: int = None,
    subsample_seed: int = 0,
    verbose: bool = True,
) -> ad.AnnData:
    """Subsample an AnnData object at the patient level (stratified).

    Parameters
    ----------
    adata : AnnData
        Full dataset with ``sampleID``, ``CoVID-19 severity``, and
        ``Outcome`` columns in ``.obs``.
    ratio : float, optional
        Fraction of patients to keep. Values >= 1.0 return the original data.
        Exactly one of ``ratio`` or ``n_patients`` must be provided.
    n_patients : int, optional
        Exact number of patients to keep. If >= total patients, returns
        the original data.
    subsample_seed : int
        Deterministic seed for the patient-level split.
    verbose : bool
        Print progress messages.

    Returns
    -------
    AnnData
        Subsampled AnnData (a copy; the original is not modified).
    """
    if ratio is None and n_patients is None:
        raise ValueError("Must provide either ratio or n_patients")

    patient_df = _get_patient_labels(adata)
    all_patients = patient_df.index.values
    total_patients = len(all_patients)

    # Convert n_patients to ratio
    if n_patients is not None:
        if n_patients >= total_patients:
            if verbose:
                print(f"[subsample] n_patients={n_patients} >= total "
                      f"({total_patients}) -> returning full dataset")
            return adata
        ratio = n_patients / total_patients
    elif ratio >= 1.0:
        if verbose:
            print(f"[subsample] ratio={ratio} -> returning full dataset "
                  f"({adata.n_obs} cells)")
        return adata

    # Map sampleID -> cell indices
    sample_ids = adata.obs["sampleID"].astype(str).values
    patient_to_cells: dict[str, list[int]] = {}
    for i, sid in enumerate(sample_ids):
        patient_to_cells.setdefault(sid, []).append(i)

    strata = patient_df.loc[all_patients, "stratum"].values
    try:
        _, sub_patients = train_test_split(
            all_patients,
            test_size=ratio,
            stratify=strata,
            random_state=subsample_seed,
        )
    except ValueError:
        strata_fallback = patient_df.loc[all_patients, "severity_bin"].values
        _, sub_patients = train_test_split(
            all_patients,
            test_size=ratio,
            stratify=strata_fallback,
            random_state=subsample_seed,
        )

    cell_indices = np.concatenate(
        [np.array(patient_to_cells[pid]) for pid in sub_patients]
    )
    cell_indices = np.sort(cell_indices)
    adata_sub = adata[cell_indices].copy()

    if verbose:
        print(f"[subsample] {len(sub_patients)}/{total_patients} "
              f"patients, {adata_sub.n_obs} cells")

    return adata_sub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create stratified patient-level subsamples for scalability benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5ad", required=True, help="Path to full COVID h5ad")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="Subsample ratios (fraction of patients to keep; 1.0 = full data)")
    p.add_argument("--subsample-seed", type=int, default=0,
                   help="Fixed seed for subsampling (deterministic per ratio)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)

    print(f"Loading {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad)
    adata.var_names_make_unique()
    n_total_cells = adata.n_obs
    print(f"  Total cells: {n_total_cells}")

    # Get per-patient labels for stratification
    patient_df = _get_patient_labels(adata)
    all_patients = patient_df.index.values
    n_patients = len(all_patients)
    print(f"  Total patients: {n_patients}")
    print(f"  Patient strata (severity_outcome): {patient_df['stratum'].value_counts().to_dict()}")

    # Map sampleID -> cell indices for fast subsetting
    sample_ids = adata.obs["sampleID"].astype(str).values
    patient_to_cells = {}
    for i, sid in enumerate(sample_ids):
        if sid not in patient_to_cells:
            patient_to_cells[sid] = []
        patient_to_cells[sid].append(i)
    patient_to_cells = {k: np.array(v) for k, v in patient_to_cells.items()}

    manifest = {
        "source": args.h5ad,
        "total_cells": n_total_cells,
        "total_patients": n_patients,
        "subsample_seed": args.subsample_seed,
        "patient_strata": patient_df["stratum"].value_counts().to_dict(),
        "subsamples": {},
    }

    for ratio in sorted(args.ratios):
        ratio_str = f"ratio_{ratio:.2f}"
        ratio_dir = out_root / ratio_str
        ratio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")

        if ratio >= 1.0:
            # Full dataset: just create metadata + scHPF inputs, no subsampling
            print(f"Ratio {ratio} -> FULL dataset ({n_patients} patients, {n_total_cells} cells)")
            print(f"{'='*60}")

            sub_patients = all_patients
            adata_sub = adata  # no copy needed, we're just writing outputs

        else:
            n_target_patients = max(1, int(n_patients * ratio))
            print(f"Ratio {ratio} -> target ~{n_target_patients} of {n_patients} patients")
            print(f"{'='*60}")

            # Stratified patient-level subsample
            strata = patient_df.loc[all_patients, "stratum"].values
            try:
                _, sub_patients = train_test_split(
                    all_patients,
                    test_size=ratio,
                    stratify=strata,
                    random_state=args.subsample_seed,
                )
            except ValueError as e:
                print(f"  WARNING: Stratified split failed ({e}), falling back to severity-only")
                strata_fallback = patient_df.loc[all_patients, "severity_bin"].values
                _, sub_patients = train_test_split(
                    all_patients,
                    test_size=ratio,
                    stratify=strata_fallback,
                    random_state=args.subsample_seed,
                )

            # Gather all cell indices for selected patients
            cell_indices = np.concatenate([patient_to_cells[pid] for pid in sub_patients])
            cell_indices = np.sort(cell_indices)
            adata_sub = adata[cell_indices].copy()

            # Save subsampled h5ad
            h5ad_path = ratio_dir / "covid_subsample.h5ad"
            adata_sub.write_h5ad(h5ad_path)
            print(f"  Saved: {h5ad_path}")

            # Save patient list and cell indices for reference
            np.save(ratio_dir / "subsample_cell_indices.npy", cell_indices)
            pd.Series(sub_patients).to_csv(ratio_dir / "subsample_patients.csv", index=False, header=["sampleID"])

        print(f"  Selected {len(sub_patients)} patients, {adata_sub.n_obs} cells")

        # Report strata
        sub_patient_df = patient_df.loc[sub_patients]
        sub_strata = sub_patient_df["stratum"].value_counts().to_dict()
        print(f"  Patient strata: {sub_strata}")

        # Cell type distribution
        meta_sub = _build_metadata(adata_sub.obs.copy())
        ct_dist = meta_sub["cell_type"].value_counts().to_dict()
        print(f"  Cell types: {ct_dist}")

        # Save metadata CSV (for Spectra/scHPF downstream classifiers)
        meta_sub.to_csv(ratio_dir / "metadata_covid.csv", index=True)
        print(f"  Saved: {ratio_dir / 'metadata_covid.csv'}")

        # Save scHPF inputs
        _write_schpf_inputs(adata_sub, ratio_dir)
        print(f"  Saved scHPF inputs: {ratio_dir / 'schpf_input'}")

        manifest["subsamples"][ratio_str] = {
            "ratio": ratio,
            "n_patients": int(len(sub_patients)),
            "n_cells": int(adata_sub.n_obs),
            "n_genes": int(adata_sub.n_vars),
            "patient_strata": sub_strata,
            "cell_type_dist": ct_dist,
        }

    # Save manifest
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
