"""
calibrate_severity.py
=====================
Computes fold-changes for severe vs mild COVID to calibrate
the new perturbation model.

Input: pre-filtered h5ad (PBMC, COVID+ only, target cell types).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import os

# --- PATHS ---
h5ad_path = '/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/covid19_filtered_fullgenes_clean.h5ad'
output_dir = '/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/calibration_severity'
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
adata = sc.read_h5ad(h5ad_path)
print(f"  Shape: {adata.shape}")
print(f"  Columns: {list(adata.obs.columns)}")

# Quick inventory
for col in ['CoVID-19 severity', 'Outcome', 'Sex', 'cm_asthma_copd',
            'cm_diabetes', 'cm_cardio', 'majorType']:
    if col in adata.obs.columns:
        print(f"\n  {col}:")
        print(f"  {adata.obs[col].value_counts().to_dict()}")

# ---- HVGs ----
# Use same 2000 HVGs as scDesign3 fitting if you have them:
# with open('hvg_2000_genes.txt') as f:
#     gene_list = [l.strip() for l in f if l.strip()]
# adata_sub = adata[:, gene_list].copy()

# Otherwise compute fresh:
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
gene_list = adata.var_names[adata.var['highly_variable']].tolist()
adata_sub = adata[:, gene_list].copy()
print(f"\n  Using {len(gene_list)} HVGs")

eps = 1e-2
gene_arr = np.array(gene_list)
keep_types = [ct for ct in adata_sub.obs['majorType'].unique() if ct != 'unknown']
print(f"  Cell types: {keep_types}")

# ============================================================
# 1. SEVERITY FC: severe vs mild, per cell type
# ============================================================
print("\n" + "=" * 70)
print("SEVERITY FOLD-CHANGES (severe vs mild)")
print("=" * 70)

severity_results = {}
for ct in keep_types:
    ct_data = adata_sub[adata_sub.obs['majorType'] == ct]
    mild = ct_data[ct_data.obs['CoVID-19 severity'] == 'mild/moderate']
    severe = ct_data[ct_data.obs['CoVID-19 severity'] == 'severe/critical']

    mu_mild = np.asarray(mild.X.mean(axis=0)).flatten()
    mu_severe = np.asarray(severe.X.mean(axis=0)).flatten()
    fc = (mu_severe + eps) / (mu_mild + eps)
    log2fc = np.log2(fc)

    severity_results[ct] = {
        'mu_mild': mu_mild, 'mu_severe': mu_severe,
        'fc': fc, 'log2fc': log2fc,
        'n_mild': mild.shape[0], 'n_severe': severe.shape[0],
    }

    up = fc[fc > 1]
    print(f"\n  {ct} (mild={mild.shape[0]}, severe={severe.shape[0]}):")
    print(f"    All genes:   median FC = {np.median(fc):.4f}")
    print(f"    Upregulated: median = {np.median(up):.4f}, "
          f"75th = {np.percentile(up, 75):.4f}, "
          f"90th = {np.percentile(up, 90):.4f}")
    print(f"    DE (|log2FC|>0.5): n = {(np.abs(log2fc) > 0.5).sum()}")
    print(f"    DE (|log2FC|>1.0): n = {(np.abs(log2fc) > 1.0).sum()}")
    if (np.abs(log2fc) > 0.5).sum() > 0:
        de = fc[np.abs(log2fc) > 0.5]
        print(f"    DE FC (>0.5): median = {np.median(np.abs(de)):.3f}, "
              f"90th = {np.percentile(np.abs(de), 90):.3f}")

# ============================================================
# 2. SEX FC: male vs female within COVID patients
# ============================================================
print("\n" + "=" * 70)
print("SEX FOLD-CHANGES (M vs F, COVID patients)")
print("=" * 70)

sex_results = {}
for ct in keep_types:
    ct_data = adata_sub[adata_sub.obs['majorType'] == ct]
    male = ct_data[ct_data.obs['Sex'] == 'M']
    female = ct_data[ct_data.obs['Sex'] == 'F']

    mu_m = np.asarray(male.X.mean(axis=0)).flatten()
    mu_f = np.asarray(female.X.mean(axis=0)).flatten()
    fc = (mu_m + eps) / (mu_f + eps)

    sex_results[ct] = {'fc': fc, 'log2fc': np.log2(fc),
                       'n_male': male.shape[0], 'n_female': female.shape[0]}
    print(f"  {ct} (M={male.shape[0]}, F={female.shape[0]}): "
          f"median |FC| = {np.median(np.abs(fc)):.4f}, "
          f"90th = {np.percentile(np.abs(fc), 90):.4f}, "
          f"95th = {np.percentile(np.abs(fc), 95):.4f}")

# ============================================================
# 3. COMORBIDITY FC: asthma+ vs asthma-
# ============================================================
print("\n" + "=" * 70)
print("COMORBIDITY FOLD-CHANGES (asthma/COPD)")
print("=" * 70)

adata_cm = adata_sub[~adata_sub.obs['cm_asthma_copd'].isin(['exclude', 'unknown'])].copy()

comorbidity_results = {}
for ct in keep_types:
    ct_data = adata_cm[adata_cm.obs['majorType'] == ct]
    pos = ct_data[ct_data.obs['cm_asthma_copd'].astype(str) == '1']
    neg = ct_data[ct_data.obs['cm_asthma_copd'].astype(str) == '0']

    if pos.shape[0] < 50:
        print(f"  {ct}: SKIPPED ({pos.shape[0]} asthma+ cells)")
        continue

    mu_pos = np.asarray(pos.X.mean(axis=0)).flatten()
    mu_neg = np.asarray(neg.X.mean(axis=0)).flatten()
    fc = (mu_pos + eps) / (mu_neg + eps)

    comorbidity_results[ct] = {'fc': fc, 'log2fc': np.log2(fc),
                                'n_pos': pos.shape[0], 'n_neg': neg.shape[0]}
    print(f"  {ct} (asthma+={pos.shape[0]}, asthma-={neg.shape[0]}): "
          f"median |FC| = {np.median(np.abs(fc)):.4f}, "
          f"90th = {np.percentile(np.abs(fc), 90):.4f}")

# ============================================================
# 4. OUTCOME FC: deceased vs discharged
# ============================================================
print("\n" + "=" * 70)
print("OUTCOME FOLD-CHANGES (deceased vs discharged)")
print("=" * 70)

adata_out = adata_sub[adata_sub.obs['Outcome'].isin(['discharged', 'deceased'])].copy()

outcome_results = {}
for ct in keep_types:
    ct_data = adata_out[adata_out.obs['majorType'] == ct]
    discharged = ct_data[ct_data.obs['Outcome'] == 'discharged']
    deceased = ct_data[ct_data.obs['Outcome'] == 'deceased']

    mu_dis = np.asarray(discharged.X.mean(axis=0)).flatten()
    mu_dec = np.asarray(deceased.X.mean(axis=0)).flatten()
    fc = (mu_dec + eps) / (mu_dis + eps)
    log2fc = np.log2(fc)

    outcome_results[ct] = {'fc': fc, 'log2fc': log2fc,
                           'n_discharged': discharged.shape[0],
                           'n_deceased': deceased.shape[0]}
    print(f"  {ct} (discharged={discharged.shape[0]}, deceased={deceased.shape[0]}): "
          f"DE(>0.5)={(np.abs(log2fc) > 0.5).sum()}, "
          f"DE(>1.0)={(np.abs(log2fc) > 1.0).sum()}")

# ============================================================
# 5. SEVERITY GENE SETS
# ============================================================
print("\n" + "=" * 70)
print("SEVERITY-RELEVANT GENE SETS")
print("=" * 70)

per_ct_genes = {}
for threshold in [0.3, 0.5, 1.0]:
    per_ct_genes[threshold] = {}
    for ct in keep_types:
        log2fc = severity_results[ct]['log2fc']
        per_ct_genes[threshold][ct] = set(gene_arr[np.abs(log2fc) > threshold])

    union = set().union(*per_ct_genes[threshold].values())
    print(f"\n  |log2FC| > {threshold}:")
    print(f"    Union: {len(union)} genes")
    for ct in keep_types:
        print(f"      {ct}: {len(per_ct_genes[threshold][ct])}")

# ============================================================
# 6. OVERLAP: severity vs outcome gene sets
# ============================================================
print("\n" + "=" * 70)
print("SEVERITY vs OUTCOME GENE SET OVERLAP")
print("=" * 70)

for threshold in [0.5, 1.0]:
    sev_union = set()
    out_union = set()
    for ct in keep_types:
        sev_union |= set(gene_arr[np.abs(severity_results[ct]['log2fc']) > threshold])
        if ct in outcome_results:
            out_union |= set(gene_arr[np.abs(outcome_results[ct]['log2fc']) > threshold])

    overlap = sev_union & out_union
    print(f"  |log2FC| > {threshold}: severity={len(sev_union)}, "
          f"outcome={len(out_union)}, overlap={len(overlap)}")

# ============================================================
# SAVE
# ============================================================
for threshold in [0.3, 0.5, 1.0]:
    union = set().union(*per_ct_genes[threshold].values())
    with open(f'{output_dir}/severity_genes_log2fc_{threshold}.txt', 'w') as f:
        for g in sorted(union):
            f.write(g + '\n')

fc_df = pd.DataFrame(index=gene_arr)
for ct in keep_types:
    fc_df[f'{ct}_severity_fc'] = severity_results[ct]['fc']
    fc_df[f'{ct}_severity_log2fc'] = severity_results[ct]['log2fc']
    fc_df[f'{ct}_sex_fc'] = sex_results[ct]['fc']
    if ct in comorbidity_results:
        fc_df[f'{ct}_comorbidity_fc'] = comorbidity_results[ct]['fc']
    if ct in outcome_results:
        fc_df[f'{ct}_outcome_fc'] = outcome_results[ct]['fc']

fc_df.to_csv(f'{output_dir}/fold_changes_severity.csv.gz', compression='gzip')

# Save gene list for reference
with open(f'{output_dir}/hvg_2000_genes.txt', 'w') as f:
    for g in gene_list:
        f.write(g + '\n')

print(f"\nAll outputs saved to {output_dir}/")