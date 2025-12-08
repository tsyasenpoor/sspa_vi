import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import jax
import jax.numpy as jnp
import jax.random as jax_random
import inspect
import traceback

# Enhanced GPU configuration - only set GPU if available
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Only set GPU platform if CUDA is available
try:
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        print("✓ GPU detected, setting JAX_PLATFORM_NAME=gpu")
    else:
        print("✗ No GPU devices detected, using CPU")
except Exception as e:
    print(f"✗ Error during GPU detection: {e}")
    print("Continuing with CPU fallback...")

# Display current JAX devices
try:
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        print(f"✓ Using GPU: {len(gpu_devices)} GPU device(s)")
        print(f"Primary device: {gpu_devices[0]}")
    else:
        print("✗ Using CPU (no GPU devices detected)")
        print("This will run slower but should complete successfully")
except Exception as e:
    print(f"✗ Error detecting devices: {e}")
    print("Continuing with CPU fallback...")

import gseapy
from gseapy import read_gmt
import mygene  
import argparse
import gc
import psutil  
import pickle  # Add pickle for caching results
import random  # Import for random sampling of pathways
import gzip # Import for gzipping files
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory

# Log initial memory
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

from svi_jax_cleaned import NaturalGradientSVI
from VariationalInference.old_data import *


def convert_numpy_to_lists(obj):
    """Convert numpy and JAX arrays to lists for JSON serialization - OPTIMIZED VERSION"""
    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    elif hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) if isinstance(v, (np.ndarray, jnp.ndarray, dict, list)) else v 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) if isinstance(item, (np.ndarray, jnp.ndarray, dict, list)) else item 
                for item in obj]
    else:
        return obj

def run_svi_experiment(X_train, Y_train, X_aux_train,
                       X_val, Y_val, X_aux_val,
                       X_test, Y_test, X_aux_test,
                       var_names, hyperparams, seed, max_iters, batch_size):
    """
    Handles the full SVI pipeline: training on train set, predicting on all sets, and returning results.
    """
    n_samples, n_genes = X_train.shape
    n_factors = hyperparams.get('d', 10)
    n_outcomes = Y_train.shape[1]

    # 1. Initialize and fit the model ON TRAINING DATA ONLY
    model = NaturalGradientSVI(
        n_samples=n_samples, n_genes=n_genes, n_factors=n_factors, n_outcomes=n_outcomes,
        key=jax_random.PRNGKey(seed if seed is not None else 42),
        **{k: v for k, v in hyperparams.items() if k != 'd'} # Pass other hypers
    )
    
    fit_results = model.fit(
        X=X_train, Y=Y_train, X_aux=X_aux_train,
        n_iter=max_iters, batch_size=batch_size, verbose=True, elbo_freq=100
    )
    trained_nat_params = fit_results['nat_params']

    # 2. Predict on Train, Validation, and Test sets by inferring their local thetas
    print("\n--- Inferring local parameters and predicting on all data splits ---")
    train_probs = model.predict(X_train, X_aux_train, Y_train, trained_nat_params)
    val_probs = model.predict(X_val, X_aux_val, Y_val, trained_nat_params)
    test_probs = model.predict(X_test, X_aux_test, Y_test, trained_nat_params)

    # 3. Calculate metrics for each split
    def calculate_metrics(y_true, y_prob):
        y_true = np.array(y_true).ravel()
        y_prob = np.array(y_prob).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }

    train_metrics = calculate_metrics(Y_train, train_probs)
    val_metrics = calculate_metrics(Y_val, val_probs)
    test_metrics = calculate_metrics(Y_test, test_probs)

    # 4. Package all results
    moment_params = model._natural_to_moment(trained_nat_params)
    expected_params = model._compute_expectations(moment_params)
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_probabilities': train_probs,
        'val_probabilities': val_probs,
        'test_probabilities': test_probs,
        'nat_params': trained_nat_params,
        'moment_params': moment_params,
        'expected': expected_params,
        'elbo_history': fit_results.get('elbo_history', []),
        'hyperparameters': hyperparams,
    }

# This function is left for compatibility but should be phased out.
def run_vi(*args, **kwargs):
    print("WARNING: `run_vi` is deprecated. Please update experiment script.")
    # Add a dummy implementation or raise an error
    return {"error": "VI function not implemented in this script."}

def save_split_results(results, y_data, sample_ids, split_indices, split_name, output_dir, prefix):
    print(f"DEBUG: Saving {split_name} results with prefix {prefix}")
    if f"{split_name}_probabilities" not in results:
        print(f"ERROR: {split_name}_probabilities not found in results!")
        return
    
    probs = np.array(results[f"{split_name}_probabilities"])
    y_true = y_data[split_indices]
    sample_id_list = [sample_ids[i] for i in split_indices]
    preds = (probs >= 0.5).astype(int)
    n_labels = y_true.shape[1]
    data = []
    for i, idx in enumerate(split_indices):
        row = {"sample_id": sample_id_list[i]}
        for k in range(n_labels):
            row[f"true_label_{k+1}"] = y_true[i, k]
            row[f"pred_prob_{k+1}"] = probs[i, k]
            row[f"pred_label_{k+1}"] = preds[i, k]
        data.append(row)
    df = pd.DataFrame(data)
    out_path = os.path.join(output_dir, f"{prefix}_{split_name}_results.csv.gz")
    df.to_csv(out_path, index=False, compression='gzip')

def save_beta_matrix_csv(expected_params, gene_names, row_names, out_path):
    E_beta = np.array(expected_params['E_beta'])
    mu_v = np.array(expected_params['E_v'])
    n_programs = E_beta.shape[1]
    n_labels = mu_v.shape[0]
    data = []
    for k in range(n_programs):
        row = {"name": row_names[k]}
        for l in range(n_labels):
            row[f"v_{l+1}"] = mu_v[l, k]
        for g, gene in enumerate(gene_names):
            row[gene] = E_beta[g, k]
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False, compression="gzip")

def run_all_experiments(datasets, hyperparams_map, output_dir, seed=None, mask=None, max_iter=100, pathway_names=None, run_fn_name="svi", args=None):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for dataset_name, (adata, label_col) in datasets.items():
        print(f"\nRunning experiment on dataset {dataset_name}, label={label_col}")
        
        # --- Data Preparation ---
        label_names = [label_col]
        Y = adata.obs[label_col].values.astype(float).reshape(-1, 1)
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
        var_names = list(adata.var_names)
        
        # Handle different auxiliary variables for different datasets
        if dataset_name == "emtab":
            x_aux = adata.obs[['age', 'sex_female']].values.astype(float)
        elif dataset_name == "thyroid":
            x_aux = adata.obs[['Age', 'sex_female']].values.astype(float)
        elif dataset_name == "sim":
            aux_cols = [col for col in adata.obs.columns if col != label_col and adata.obs[col].dtype in ['int64', 'float64']]
            x_aux = adata.obs[aux_cols].values.astype(float) if aux_cols else np.ones((X.shape[0], 1))
        else: # ajm_cyto, ajm_ap, signal_noise
            x_aux = np.ones((X.shape[0], 1))

        sample_ids = adata.obs.index.tolist()

        hyperparams = hyperparams_map[dataset_name].copy()
        d_values = hyperparams.pop("d", [50]) # Default to d=50 if not specified
        if not isinstance(d_values, list):
            d_values = [d_values]

        # --- Train/Val/Test Split ---
        indices = np.arange(X.shape[0])
        X_train, X_temp, Y_train, Y_temp, X_aux_train, X_aux_temp, train_idx, temp_idx = train_test_split(
            X, Y, x_aux, indices, test_size=0.3, random_state=seed if seed is not None else 42, stratify=Y)
        X_val, X_test, Y_val, Y_test, X_aux_val, X_aux_test, val_idx, test_idx = train_test_split(
            X_temp, Y_temp, X_aux_temp, temp_idx, test_size=0.5, random_state=seed if seed is not None else 42, stratify=Y_temp)

        for d in d_values:
            print(f"Running with d={d}")
            hyperparams["d"] = d
            
            try:
                if run_fn_name == "svi":
                    results = run_svi_experiment(
                        X_train, Y_train, X_aux_train,
                        X_val, Y_val, X_aux_val,
                        X_test, Y_test, X_aux_test,
                        var_names, hyperparams, seed, max_iter, args.batch_size
                    )
                else:
                    results = run_vi( # Assuming run_vi is defined elsewhere and follows a similar API
                         x_data=X_train, x_aux=X_aux_train, y_data=Y_train, var_names=var_names,
                         hyperparams=hyperparams, seed=seed, max_iters=max_iter, mask=mask
                    )

                results["label_names"] = label_names
                
                # --- Save results ---
                prefix = f"{dataset_name}_{label_col}_d_{d}"
                save_split_results(results, Y, sample_ids, train_idx, "train", output_dir, prefix)
                save_split_results(results, Y, sample_ids, val_idx, "val", output_dir, prefix)
                save_split_results(results, Y, sample_ids, test_idx, "test", output_dir, prefix)
                
                # Save lightweight JSON summary
                main_results = {k: v for k, v in results.items() if k not in ['nat_params', 'moment_params', 'expected']}
                main_results = convert_numpy_to_lists(main_results)
                with gzip.open(os.path.join(output_dir, f"{prefix}_summary.json.gz"), "wt") as f:
                    json.dump(main_results, f, indent=2)

                all_results[prefix] = results

                # Save Beta matrix
                row_names = [f"DRGP{i+1}" for i in range(d)]
                save_beta_matrix_csv(results['expected'], var_names, row_names, os.path.join(output_dir, "gene_programs.csv.gz"))

            except Exception as e:
                print(f"--- UNHANDLED EXCEPTION for d={d} ---")
                traceback.print_exc()
                all_results[f"{dataset_name}_{label_col}_d_{d}"] = {"error": str(e), "status": "crashed"}
            clear_memory()

    return all_results
# This is the main experiment running logic that was restored.
def main():
    parser = argparse.ArgumentParser(description='Run various experiments')
    parser.add_argument("--mask", action="store_true", help="Use mask derived from pathways matrix")
    parser.add_argument("--d", type=int, help="Value of d when mask is not provided")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations for variational inference")
    parser.add_argument("--dataset", type=str, default="cyto", 
                       choices=["cyto", "ap", "emtab", "thyroid", "sim"], 
                       help="Dataset to use.")
    parser.add_argument("--method", choices=["svi"], default="svi", help="Inference method to use")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for SVI")
    args = parser.parse_args()

    # Create a timestamp-based subdirectory for this specific run
    run_dir = os.path.join("/labs/Aguiar/SSPA_BRAY/BRay/SVIResults/unmasked", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # --- DATASET LOADING LOGIC ---
    print(f"Loading data for experiment: {args.dataset}")
    if args.dataset == "emtab":
        adata = prepare_and_load_emtab()
        label_col = "disease"
    elif args.dataset == "thyroid":
        adata = prepare_and_load_thyroid()
        label_col = "Clinical_History"
    elif args.dataset == "sim":
        adata = load_data_simulation()
        label_col = "disease"
    else: # cyto or ap
        ajm_ap_samples, ajm_cyto_samples = prepare_ajm_dataset()
        if args.dataset == "cyto":
            adata = filter_protein_coding_genes(ajm_cyto_samples, gene_annotation)
            label_col = "cyto"
        else: # ap
            adata = filter_protein_coding_genes(ajm_ap_samples, gene_annotation)
            label_col = "ap"
    
    datasets = {f"ajm_{args.dataset}" if args.dataset in ['cyto', 'ap'] else args.dataset: (adata, label_col)}

    # --- HYPERPARAMETERS ---
    # Using a single set of general-purpose hyperparameters for simplicity
    hyperparams = {
        "alpha_eta": 1.0, "lambda_eta": 1.0, "alpha_beta": 1.0,
        "alpha_xi": 1.0, "lambda_xi": 1.0, "alpha_theta": 1.0,
        "sigma2_v": 1.0, "sigma2_gamma": 1.0,
        "d": args.d if args.d is not None else 50
    }
    hyperparams_map = {list(datasets.keys())[0]: hyperparams}

    # --- RUN EXPERIMENT ---
    all_results = run_all_experiments(
        datasets, hyperparams_map, output_dir=run_dir,
        max_iter=args.max_iter, run_fn_name=args.method, args=args
    )

    print("\nAll experiments completed!")
    print(f"Results saved to: {run_dir}")

    # --- SUMMARIZE RESULTS ---
    print("\nSummary of results:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 80)
    for exp_name, res in all_results.items():
        if "error" not in res:
            train_m, val_m, test_m = res['train_metrics'], res['val_metrics'], res['test_metrics']
            print(f"{exp_name:<30} {train_m['accuracy']:<10.4f} {val_m['accuracy']:<10.4f} {test_m['accuracy']:<10.4f} "
                  f"{train_m['f1']:<10.4f} {val_m['f1']:<10.4f} {test_m['f1']:<10.4f}")
        else:
            print(f"{exp_name:<30} {'CRASHED':<10}")

if __name__ == "__main__":
    main()

