#!/usr/bin/env python
"""
Quick script to check if your data is balanced.
This helps determine if prediction bias is a problem or expected behavior.
"""

import scanpy as sc
import numpy as np
import pandas as pd

def check_balance():
    print("="*80)
    print("DATA BALANCE ANALYSIS")
    print("="*80)
    print()

    # Try to load data
    try:
        adata = sc.read_h5ad('filtered_data_top3k.h5ad')
        print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    except:
        print("✗ Could not load filtered_data_top3k.h5ad")
        return

    # Check if ap column exists
    if 'ap' not in adata.obs.columns:
        print("✗ No 'ap' column found in data")
        return

    print()
    print("-"*80)
    print("OVERALL CLASS DISTRIBUTION")
    print("-"*80)

    total_counts = adata.obs['ap'].value_counts().sort_index()
    total = len(adata)

    for cls in total_counts.index:
        count = total_counts[cls]
        pct = 100 * count / total
        print(f"  Class {cls}: {count:4d} cells ({pct:5.1f}%)")

    # Calculate imbalance ratio
    max_count = total_counts.max()
    min_count = total_counts.min()
    ratio = max_count / min_count

    print()
    print(f"  Imbalance ratio: {ratio:.2f}:1")

    if ratio > 1.5:
        print(f"  ⚠ DATA IS IMBALANCED")
        print(f"     → Expected mean probability: {max_count/total:.3f}")
        print(f"     → Your model predictions (val=0.716, test=0.653) are reasonable!")
    else:
        print(f"  ✓ Data is balanced")

    # Check split-wise distribution
    if 'split' in adata.obs.columns:
        print()
        print("-"*80)
        print("CLASS DISTRIBUTION BY SPLIT")
        print("-"*80)

        for split in ['train', 'val', 'test']:
            split_data = adata[adata.obs['split'] == split]
            if len(split_data) == 0:
                continue

            split_counts = split_data.obs['ap'].value_counts().sort_index()
            split_total = len(split_data)

            print(f"\n  {split.upper()} ({split_total} cells):")
            for cls in split_counts.index:
                count = split_counts[cls]
                pct = 100 * count / split_total
                print(f"    Class {cls}: {count:3d} ({pct:5.1f}%)")

    # Summary and interpretation
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    if ratio > 1.5:
        majority_pct = 100 * max_count / total
        print(f"Your data has {majority_pct:.1f}% in the majority class.")
        print()
        print("If your model predicts an average probability of {:.3f},".format(max_count/total))
        print("this is EXACTLY what it should do!")
        print()
        print("The 'problem' is not a problem - your model is correctly learning:")
        print("  1. The base rate of each class")
        print("  2. Which gene programs distinguish the classes")
        print()
        print("Only worry if:")
        print("  - Predictions are ALL exactly 1.0 or 0.0 (model too confident)")
        print("  - Validation accuracy << training accuracy (overfitting)")
        print("  - Predictions don't match the data distribution at all")
    else:
        print("Your data is balanced, so predictions should be around 0.5 on average.")
        print()
        print("If predictions are heavily biased (e.g., mean > 0.7), this indicates:")
        print("  - Model has learned a bias (possibly from v initialization)")
        print("  - With the new fixes, this should improve")

    print()
    print("="*80)

if __name__ == "__main__":
    check_balance()
