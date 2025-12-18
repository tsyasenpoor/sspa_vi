import json
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score

base_dir = '/labs/Aguiar/SSPA_BRAY/BRay'
sys.path.append(base_dir)

from VariationalInference.vi import VI


# Load data
df = pd.read_pickle(os.path.join(base_dir, 'sspa_test/df.pkl'))
features = pd.read_pickle(os.path.join(base_dir, 'sspa_test/features.pkl'))
train_df = pd.read_csv(os.path.join(base_dir, 'sspa_test/train_data_full_genes.csv'), index_col=0)

# Load splits
with open(os.path.join(base_dir, 'sspa_test/data_split_cell_ids.json'), 'r') as f:
    splits = json.load(f)

# Load gene list
with open(os.path.join(base_dir, 'sspa_test/gene_list.txt'), 'r') as f:
    gene_list = [line.strip() for line in f]

print(f"Training cells: {len(splits['train'])}")
print(f"Validation cells: {len(splits['val'])}")
print(f"Test cells: {len(splits['test'])}")
print(f"Genes: {len(gene_list)}")

# ============================================================================
# STEP 3: Prepare Training Data
# ============================================================================

def prepare_matrices(df, features, cell_ids, gene_list):
    """Extract X, X_aux, y for given cell IDs."""
    df_subset = df.loc[df.index.isin(cell_ids)]
    features_subset = features.loc[features.index.isin(cell_ids)]
    
    # Align
    common_idx = df_subset.index.intersection(features_subset.index)
    df_subset = df_subset.loc[common_idx]
    features_subset = features_subset.loc[common_idx]
    
    # Extract matrices
    X = df_subset[gene_list].values
    X_aux = np.zeros((X.shape[0], 0))  # No auxiliary features in this example
    # Use 't2dm' column (or first column if column name varies)
    y_col = 't2dm' if 't2dm' in features_subset.columns else features_subset.columns[0]
    y = features_subset[y_col].values
    y = y.astype(int)
    
    return X, X_aux, y

# Prepare training data
X_train, X_aux_train, y_train = prepare_matrices(
    df, features, splits['train'], gene_list
)

print(f"\nX_train: {X_train.shape}")
print(f"X_aux_train: {X_aux_train.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")

# ============================================================================
# STEP 4: Train VI Model
# ============================================================================

print("\n" + "="*80)
print("Training VI Model")
print("="*80)

model = VI(
    n_factors=50,
    alpha_theta=0.5,   # Loose prior on theta (allow high variance)
    alpha_beta=2.0,    # TIGHT: Keep beta regularized to prevent explosion
    alpha_xi=2.0,
    lambda_xi=2.0,
    sigma_v=2.0,
    sigma_gamma=1.0
)

model.fit(
    X=X_train,
    y=y_train,
    X_aux=X_aux_train,
    max_iter=200,
    tol=10.0,
    rel_tol=2e-4,
    elbo_freq=10,
    min_iter=50,
    patience=5,
    verbose=True,
    # Strong damping to prevent runaway scaling
    theta_damping=0.8,   # was 0.7 - Even slower updates for theta
    beta_damping=0.8,    # was 0.7 - Slower beta updates too
    v_damping=0.7,       # was 0.6
    gamma_damping=0.7,   # was 0.6
    xi_damping=0.9,      # was 0.8
    eta_damping=0.9,     # was 0.8
    debug = True
)

print("\nTraining complete!")

# ============================================================================
# STEP 5: Evaluate on Validation Set
# ============================================================================

print("\n" + "="*80)
print("Validation Set Evaluation")
print("="*80)

# Prepare validation data
X_val, X_aux_val, y_val = prepare_matrices(
    df, features, splits['val'], gene_list
)

# Method 1: Direct prediction (internally calls infer_theta)
y_val_proba = model.predict_proba(X_val, X_aux_val, max_iter=100, verbose=True)

y_val_proba_pos = y_val_proba.ravel()  # Already 1D
y_val_pred = (y_val_proba_pos > 0.5).astype(int)

# # Metrics
# val_acc = accuracy_score(y_val, y_val_pred)
# val_auc = roc_auc_score(y_val, y_val_proba_pos)

# print(f"\nValidation Results:")
# print(f"  Accuracy: {val_acc:.4f}")
# print(f"  AUC:      {val_auc:.4f}")

# Method 2: Explicit theta inference (if you need the latent factors)
print("\nInferring theta for validation set (for analysis)...")
E_theta_val, a_theta_val, b_theta_val = model.infer_theta(
    X_val, max_iter=100, tol=1e-4, verbose=True
)
print(f"  E_theta_val shape: {E_theta_val.shape}")
print(f"  Theta mean: {E_theta_val.mean():.4f}")
print(f"  Theta std:  {E_theta_val.std():.4f}")

# ============================================================================
# STEP 6: Evaluate on Test Set
# ============================================================================

print("\n" + "="*80)
print("Test Set Evaluation")
print("="*80)

# Prepare test data
X_test, X_aux_test, y_test = prepare_matrices(
    df, features, splits['test'], gene_list
)

# Predict
y_test_proba = model.predict_proba(X_test, X_aux_test, max_iter=100, verbose=True)
y_test_proba_pos = y_test_proba.ravel()  # Already 1D

y_test_pred = (y_test_proba_pos > 0.5).astype(int)

# # Metrics
# test_acc = accuracy_score(y_test, y_test_pred)
# test_auc = roc_auc_score(y_test, y_test_proba_pos)

# print(f"\nTest Results:")
# print(f"  Accuracy: {test_acc:.4f}")
# print(f"  AUC:      {test_auc:.4f}")

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("\n" + "="*80)
print("Saving Results")
print("="*80)

import pickle
import gzip

# Save full model
with open('sspa_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved full model to 'sspa_model.pkl'")

# --- Save Beta Matrix (Gene Programs) ---
# Beta matrix shows which genes belong to which latent programs
# Shape: (n_genes, d) where d is the number of latent factors
print("\n  Saving Beta matrix (gene programs)...")

beta_df_data = []
for k in range(model.E_beta.shape[1]):  # Iterate over latent factors
    row = {
        "program": f"GP{k+1}",  # Gene Program 1, 2, 3, ...
        "v_weight": model.E_v[0, k] if model.kappa == 1 else None  # Classification weight
    }
    # Add gene loadings
    for g, gene in enumerate(gene_list):
        row[gene] = model.E_beta[g, k]
    beta_df_data.append(row)

beta_df = pd.DataFrame(beta_df_data)
beta_df.to_csv('sspa_gene_programs.csv.gz', index=False, compression='gzip')
print(f"  ✓ Saved gene programs to 'sspa_gene_programs.csv.gz'")
print(f"    - Shape: {model.E_beta.shape} (genes x programs)")
print(f"    - Programs: {model.E_beta.shape[1]}")

# --- Save Theta (Sample Loadings) for Train/Val/Test ---
# Theta shows how much each sample expresses each gene program
# Shape: (n_samples, d) where d is the number of latent factors
# print("\n  Saving Theta matrices (sample loadings)...")

# Training theta
E_theta_train = model.E_theta
theta_train_df = pd.DataFrame(
    E_theta_train,
    index=train_df.index,
    columns=[f"GP{k+1}" for k in range(E_theta_train.shape[1])]
)
theta_train_df.to_csv('sspa_theta_train.csv.gz', compression='gzip')
print(f"  ✓ Saved training theta to 'sspa_theta_train.csv.gz'")
print(f"    - Shape: {E_theta_train.shape} (samples x programs)")

# Validation theta (already inferred above)
theta_val_df = pd.DataFrame(
    E_theta_val,
    index=df.loc[df.index.isin(splits['val'])].index,
    columns=[f"GP{k+1}" for k in range(E_theta_val.shape[1])]
)
theta_val_df.to_csv('sspa_theta_val.csv.gz', compression='gzip')
print(f"  ✓ Saved validation theta to 'sspa_theta_val.csv.gz'")
print(f"    - Shape: {E_theta_val.shape} (samples x programs)")

# Infer and save test theta
print("\n  Inferring theta for test set...")
E_theta_test, _, _ = model.infer_theta(X_test, max_iter=500, tol=1e-4, verbose=False)
theta_test_df = pd.DataFrame(
    E_theta_test,
    index=df.loc[df.index.isin(splits['test'])].index,
    columns=[f"GP{k+1}" for k in range(E_theta_test.shape[1])]
)
theta_test_df.to_csv('sspa_theta_test.csv.gz', compression='gzip')
print(f"  ✓ Saved test theta to 'sspa_theta_test.csv.gz'")
print(f"    - Shape: {E_theta_test.shape} (samples x programs)")

# --- Save Prediction Results (similar to run_experiments.py) ---
print("\n  Saving prediction results...")

# Helper function to save split results
def save_split_predictions(probs, preds, true_labels, cell_ids, split_name, filename):
    """Save predictions for a data split."""
    data = []
    for i, cell_id in enumerate(cell_ids):
        data.append({
            'cell_id': cell_id,
            'true_label': true_labels[i],
            'pred_prob': probs[i, 0],  # Binary classification
            'pred_label': preds[i]
        })
    df_out = pd.DataFrame(data)
    df_out.to_csv(filename, index=False, compression='gzip')

val_cell_ids = df.loc[df.index.isin(splits['val'])].index.tolist()
save_split_predictions(y_val_proba_pos.reshape(-1, 1), y_val_pred, y_val, val_cell_ids, 
                       'val_cells', 'sspa_val_predictions.csv.gz')
print("  ✓ Saved validation predictions to 'sspa_val_predictions.csv.gz'")

# Save test predictions
test_cell_ids = df.loc[df.index.isin(splits['test'])].index.tolist()
save_split_predictions(y_test_proba_pos.reshape(-1, 1), y_test_pred, y_test, test_cell_ids,
                      'test_cells', 'sspa_test_predictions.csv.gz')

print("  ✓ Saved test predictions to 'sspa_test_predictions.csv.gz'")

# --- Save Summary with Metrics ---
print("\n  Saving summary...")

summary = {
    'hyperparameters': {
        'd': model.d,
        'alpha_theta': model.alpha_theta,
        'alpha_beta': model.alpha_beta,
        'alpha_xi': model.alpha_xi,
        'alpha_eta': model.alpha_eta,
        'lambda_xi': model.lambda_xi,
        'lambda_eta': model.lambda_eta,
        'sigma_v': model.sigma_v,
        'sigma_gamma': model.sigma_gamma
    },
    'data_shapes': {
        'n_genes': len(gene_list),
        'n_train': X_train.shape[0],
        'n_val': X_val.shape[0],
        'n_test': X_test.shape[0],
        'n_factors': model.d
    },
    # 'metrics': {
    #     'validation': {
    #         'accuracy': val_acc,
    #         'auc': val_auc
    #     },
    #     'test': {
    #         'accuracy': test_acc,
    #         'auc': test_auc
    #     }
    # },
    'elbo_history': model.elbo_history_
}

with gzip.open('sspa_vi_summary.json.gz', 'wt') as f:
    json.dump(summary, f, indent=2)
print("  ✓ Saved summary to 'sspa_vi_summary.json.gz'")

print("\n" + "="*80)
print("Complete!")
print("="*80)
print("\nSaved files:")
print("  - sspa_model.pkl (full model for loading later)")
print("  - sspa_gene_programs.csv.gz (beta matrix with gene loadings)")
print("  - sspa_theta_train.csv.gz (sample loadings for training set)")
print("  - sspa_theta_val.csv.gz (sample loadings for validation set)")
print("  - sspa_theta_test.csv.gz (sample loadings for test set)")
print("  - sspa_val_predictions.csv.gz (validation predictions)")
print("  - sspa_test_predictions.csv.gz (test predictions)")
print("  - sspa_vi_summary.json.gz (hyperparameters and metrics)")

# ============================================================================
# BONUS: Analyze Learned Parameters
# ============================================================================

print("\n" + "="*80)
print("Model Parameters Summary & Analysis")
print("="*80)

print(f"\nGlobal Parameters (learned from training):")
print(f"  Beta (gene loadings):  {model.E_beta.shape}")
print(f"    - Mean: {model.E_beta.mean():.4f}")
print(f"    - Std:  {model.E_beta.std():.4f}")
print(f"    - Top loaded genes per program can reveal biological pathways")
print(f"  V (classification weights): {model.E_v.shape}")
print(f"    - Mean: {model.E_v.mean():.4f}")
print(f"    - Std:  {model.E_v.std():.4f}")
print(f"    - Shows which gene programs drive classification")
print(f"  Gamma (batch effects):  {model.E_gamma.shape}")
print(f"    - Mean: {model.E_gamma.mean():.4f}")
print(f"    - Std:  {model.E_gamma.std():.4f}")

# Display spike-and-slab sparsity if available
if hasattr(model, 'rho_beta') and hasattr(model, 'rho_v'):
    threshold = 0.5
    beta_active = model.rho_beta > threshold
    v_active = model.rho_v > threshold
    
    print(f"\n[SPIKE-AND-SLAB SPARSITY]")
    print(f"  Beta (genes):")
    print(f"    - Active: {beta_active.sum()}/{model.rho_beta.size} ({(1-beta_active.mean())*100:.1f}%)")
    print(f"    - Sparsity: {(1-beta_active.mean())*100:.1f}%")
    print(f"    - Active per factor: {beta_active.sum(axis=0)}")
    print(f"  V (classification weights):")
    print(f"    - Active: {v_active.sum()}/{model.rho_v.size} ({(1-v_active.mean())*100:.1f}%)")
    print(f"    - Sparsity: {(1-v_active.mean())*100:.1f}%")
    print(f"    - Active per factor: {v_active.sum(axis=0)}")

print(f"\nLocal Parameters (sample-specific):")
print(f"  Theta (training):   {model.E_theta.shape}")
print(f"    - Mean: {model.E_theta.mean():.4f}")
print(f"    - Std:  {model.E_theta.std():.4f}")
print(f"  Theta (validation): {E_theta_val.shape}")
print(f"    - Mean: {E_theta_val.mean():.4f}")
print(f"    - Std:  {E_theta_val.std():.4f}")
print(f"  Theta (test):       {E_theta_test.shape}")
print(f"    - Mean: {E_theta_test.mean():.4f}")
print(f"    - Std:  {E_theta_test.std():.4f}")

# Example: Find top genes in most influential gene program
print(f"\n--- Gene Program Analysis Example ---")
most_influential_gp = np.argmax(np.abs(model.E_v[0]))  # Program with highest classification weight
top_genes_idx = np.argsort(model.E_beta[:, most_influential_gp])[-10:][::-1]
print(f"\nMost influential gene program: GP{most_influential_gp + 1}")
print(f"  Classification weight: {model.E_v[0, most_influential_gp]:.4f}")
print(f"  Top 10 genes in this program:")
for idx in top_genes_idx:
    print(f"    - {gene_list[idx]}: {model.E_beta[idx, most_influential_gp]:.4f}")

# Example: Sample with highest/lowest program expression
print(f"\n--- Sample-Level Analysis Example ---")
sample_with_max = np.argmax(E_theta_val[:, most_influential_gp])
sample_with_min = np.argmin(E_theta_val[:, most_influential_gp])
print(f"\nValidation samples with extreme GP{most_influential_gp + 1} expression:")
print(f"  Highest: {val_cell_ids[sample_with_max]} = {E_theta_val[sample_with_max, most_influential_gp]:.4f}")
print(f"  Lowest:  {val_cell_ids[sample_with_min]} = {E_theta_val[sample_with_min, most_influential_gp]:.4f}")

print("\n" + "="*80)
print("\nNext steps for analysis:")
print("  1. Load gene programs: pd.read_csv('ajm_gene_programs.csv.gz')")
print("  2. Identify top genes per program and run pathway enrichment")
print("  3. Load theta matrices to see which programs are active per sample")
print("  4. Correlate theta values with phenotypes/outcomes")
print("  5. Compare theta distributions between high/low ap groups")
print("="*80)