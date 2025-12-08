"""
Analysis script for VI model results.

This script demonstrates how to:
1. Load saved model and results
2. Analyze gene programs (beta matrix)
3. Analyze sample loadings (theta matrices)
4. Identify key biological signatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
import pickle

# ============================================================================
# Load Saved Results
# ============================================================================

print("="*80)
print("Loading VI Model Results")
print("="*80)

# Load full model (if needed)
with open('ajm_vi_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Loaded model")

# Load gene programs (beta matrix)
gene_programs = pd.read_csv('ajm_gene_programs.csv.gz', compression='gzip')
print(f"✓ Loaded gene programs: {gene_programs.shape}")

# Load theta matrices
theta_train = pd.read_csv('ajm_theta_train.csv.gz', compression='gzip', index_col=0)
theta_val = pd.read_csv('ajm_theta_val.csv.gz', compression='gzip', index_col=0)
theta_test = pd.read_csv('ajm_theta_test.csv.gz', compression='gzip', index_col=0)
print(f"✓ Loaded theta matrices:")
print(f"  - Train: {theta_train.shape}")
print(f"  - Val:   {theta_val.shape}")
print(f"  - Test:  {theta_test.shape}")

# Load predictions
val_preds = pd.read_csv('ajm_val_predictions.csv.gz', compression='gzip')
test_preds = pd.read_csv('ajm_test_predictions.csv.gz', compression='gzip')
print(f"✓ Loaded predictions")

# Load summary
with gzip.open('ajm_vi_summary.json.gz', 'rt') as f:
    summary = json.load(f)
print(f"✓ Loaded summary")

print(f"\nModel configuration:")
print(f"  - Latent factors: {summary['hyperparameters']['d']}")
print(f"  - Genes: {summary['data_shapes']['n_genes']}")
print(f"  - Validation AUC: {summary['metrics']['validation']['auc']:.4f}")
print(f"  - Test AUC: {summary['metrics']['test']['auc']:.4f}")

# ============================================================================
# Analysis 1: Identify Key Gene Programs
# ============================================================================

print("\n" + "="*80)
print("Analysis 1: Key Gene Programs for Classification")
print("="*80)

# Extract classification weights (v weights)
program_cols = [col for col in gene_programs.columns if col.startswith('GP')]
v_weights = gene_programs['v_weight'].values

# Sort programs by absolute classification weight
program_importance = pd.DataFrame({
    'program': gene_programs['program'],
    'v_weight': v_weights,
    'abs_v_weight': np.abs(v_weights)
}).sort_values('abs_v_weight', ascending=False)

print("\nTop 5 most important gene programs for classification:")
print(program_importance.head())

# ============================================================================
# Analysis 2: Top Genes per Program
# ============================================================================

print("\n" + "="*80)
print("Analysis 2: Top Genes in Each Program")
print("="*80)

def get_top_genes_in_program(gene_programs_df, program_name, top_n=10):
    """Extract top genes for a given program."""
    program_row = gene_programs_df[gene_programs_df['program'] == program_name]
    
    # Get gene columns (exclude metadata columns)
    gene_cols = [col for col in program_row.columns 
                 if col not in ['program', 'v_weight']]
    
    # Extract gene loadings
    gene_loadings = program_row[gene_cols].iloc[0]
    
    # Sort by absolute loading
    top_genes = gene_loadings.abs().sort_values(ascending=False).head(top_n)
    
    return pd.DataFrame({
        'gene': top_genes.index,
        'loading': gene_loadings[top_genes.index].values
    })

# Show top genes for the most important program
most_important_program = program_importance.iloc[0]['program']
print(f"\nTop 10 genes in {most_important_program} (most important for classification):")
top_genes = get_top_genes_in_program(gene_programs, most_important_program, top_n=10)
print(top_genes.to_string(index=False))

# Show top genes for each of top 3 programs
print(f"\nTop 5 genes in each of the 3 most important programs:")
for i in range(min(3, len(program_importance))):
    program = program_importance.iloc[i]['program']
    v_weight = program_importance.iloc[i]['v_weight']
    print(f"\n{program} (v_weight={v_weight:.4f}):")
    top_genes = get_top_genes_in_program(gene_programs, program, top_n=5)
    for _, row in top_genes.iterrows():
        print(f"  {row['gene']:20s} {row['loading']:8.4f}")

# ============================================================================
# Analysis 3: Sample-Level Program Expression
# ============================================================================

print("\n" + "="*80)
print("Analysis 3: Sample-Level Program Expression")
print("="*80)

# Compare program expression between high and low cytokine groups
# (assuming we have labels in the predictions)

# Combine theta with predictions for validation set
theta_val_with_labels = theta_val.copy()
theta_val_with_labels['true_label'] = val_preds.set_index('cell_id')['true_label']
theta_val_with_labels['pred_label'] = val_preds.set_index('cell_id')['pred_label']

# Calculate mean theta per group
mean_theta_by_label = theta_val_with_labels.groupby('true_label')[program_cols].mean()

print("\nMean program expression by cytokine group (validation set):")
print(mean_theta_by_label.T.to_string())

# Find programs with largest difference between groups
theta_diff = (mean_theta_by_label.loc[1] - mean_theta_by_label.loc[0]).abs().sort_values(ascending=False)
print("\nPrograms with largest expression difference between groups:")
print(theta_diff.head())

# ============================================================================
# Analysis 4: Correctly vs Incorrectly Classified Samples
# ============================================================================

print("\n" + "="*80)
print("Analysis 4: Correctly vs Incorrectly Classified Samples")
print("="*80)

# Identify correct and incorrect predictions
val_preds_with_theta = theta_val.copy()
val_preds_with_theta['correct'] = (val_preds['true_label'] == val_preds['pred_label']).values

# Compare program expression
mean_theta_by_correctness = val_preds_with_theta.groupby('correct')[program_cols].mean()

print("\nMean program expression by classification correctness:")
print(mean_theta_by_correctness.T.to_string())

# Programs that distinguish correct from incorrect
correctness_diff = (mean_theta_by_correctness.loc[True] - 
                   mean_theta_by_correctness.loc[False]).abs().sort_values(ascending=False)
print("\nPrograms most different between correct and incorrect predictions:")
print(correctness_diff.head())

# ============================================================================
# Analysis 5: Visualization Examples
# ============================================================================

print("\n" + "="*80)
print("Analysis 5: Creating Visualizations")
print("="*80)

# Set style
sns.set_style("whitegrid")

# 1. Heatmap of top gene programs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get top 3 programs
top_3_programs = program_importance.head(3)['program'].values

# Plot heatmap of mean expression by group
ax = axes[0]
sns.heatmap(mean_theta_by_label[top_3_programs].T, 
            annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=['Low Cyto', 'High Cyto'], ax=ax)
ax.set_title('Mean Program Expression by Cytokine Group')
ax.set_ylabel('Gene Program')

# 2. Distribution of most important program
most_important_gp_col = program_cols[0]  # Assuming GP1 is first
ax = axes[1]
for label in [0, 1]:
    subset = theta_val_with_labels[theta_val_with_labels['true_label'] == label]
    ax.hist(subset[most_important_gp_col], alpha=0.5, bins=30,
           label=f'{"High" if label == 1 else "Low"} Cytokine')
ax.set_xlabel(f'{most_important_gp_col} Expression')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of {most_important_gp_col} Expression')
ax.legend()

plt.tight_layout()
plt.savefig('gene_program_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved 'gene_program_analysis.png'")
plt.close()

# 3. Program correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
corr = theta_val[program_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
            vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Gene Program Correlation (Validation Set)')
plt.tight_layout()
plt.savefig('program_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved 'program_correlation.png'")
plt.close()

# 4. Classification weight importance
fig, ax = plt.subplots(figsize=(10, 6))
top_10 = program_importance.head(10)
colors = ['red' if x < 0 else 'blue' for x in top_10['v_weight']]
ax.barh(top_10['program'], top_10['v_weight'], color=colors, alpha=0.7)
ax.set_xlabel('Classification Weight (v)')
ax.set_title('Top 10 Gene Programs by Classification Importance')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('classification_weights.png', dpi=300, bbox_inches='tight')
print("✓ Saved 'classification_weights.png'")
plt.close()

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print("\nGenerated files:")
print("  - gene_program_analysis.png")
print("  - program_correlation.png")
print("  - classification_weights.png")
print("\nNext steps:")
print("  1. Run pathway enrichment on top genes from each program")
print("  2. Correlate theta with other phenotypes (timepoint, batch, etc.)")
print("  3. Investigate misclassified samples for biological insights")
print("  4. Compare program expression across timepoints")
print("="*80)
