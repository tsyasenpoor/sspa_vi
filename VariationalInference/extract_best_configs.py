#!/usr/bin/env python3
"""
Extract best hyperparameters from Bayesian optimization runs.
Reads the latest best_params_vi_*.json for each prior/mode/experiment
and outputs a consolidated table + CLI arguments for quick_reference.py.
"""

import json
import os
import glob

RESULTS_ROOT = "/labs/Aguiar/SSPA_BRAY/results/sim_pathway/pbmc_bayes_opt"
EXPERIMENTS = ["exp0_easy", "exp1_medium", "exp2_hard", "exp3_intersectional"]
PRIORS = ["normal", "laplace"]
MODES = ["combined", "masked", "unmasked"]


def get_latest_best_params(prior, mode, exp):
    """Find and load the latest best_params_vi_*.json for a config."""
    exp_dir = os.path.join(RESULTS_ROOT, prior, mode, exp)
    if not os.path.isdir(exp_dir):
        return None
    param_files = sorted(glob.glob(os.path.join(exp_dir, "best_params_vi_*.json")))
    if not param_files:
        return None
    latest = param_files[-1]
    with open(latest) as f:
        data = json.load(f)
    data["_source_file"] = latest
    return data


def get_latest_best_config(prior, mode, exp):
    """Find and load the latest best_config_vi_*.json for a config."""
    exp_dir = os.path.join(RESULTS_ROOT, prior, mode, exp)
    if not os.path.isdir(exp_dir):
        return None
    config_files = sorted(glob.glob(os.path.join(exp_dir, "best_config_vi_*.json")))
    if not config_files:
        return None
    latest = config_files[-1]
    with open(latest) as f:
        return json.load(f)


def params_to_cli_args(params_data, config_data=None):
    """Convert best_params JSON to quick_reference.py CLI arguments."""
    bp = params_data["best_params"]
    fp = params_data.get("fixed_params", {})

    args = []
    # Prior-specific args
    v_prior = fp.get("v_prior", "normal")
    args.append(f"--v-prior {v_prior}")

    if v_prior == "laplace":
        if "b_v" in bp:
            args.append(f"--b-v {bp['b_v']}")
        elif config_data and "b_v" in config_data:
            args.append(f"--b-v {config_data['b_v']}")
    else:
        if "sigma_v" in bp:
            args.append(f"--sigma-v {bp['sigma_v']}")
        elif config_data and "sigma_v" in config_data:
            args.append(f"--sigma-v {config_data['sigma_v']}")

    # Common model params
    param_map = {
        "alpha_theta": "--alpha-theta",
        "alpha_beta": "--alpha-beta",
        "alpha_xi": "--alpha-xi",
        "alpha_eta": "--alpha-eta",
        "lambda_xi": "--lambda-xi",
        "lambda_eta": "--lambda-eta",
        "sigma_gamma": "--sigma-gamma",
        "theta_damping": "--theta-damping",
        "beta_damping": "--beta-damping",
        "v_damping": "--v-damping",
        "gamma_damping": "--gamma-damping",
        "xi_damping": "--xi-damping",
        "eta_damping": "--eta-damping",
    }

    # Try config_data first (has exact values), fall back to best_params
    source = config_data if config_data else bp
    for param, flag in param_map.items():
        if param in source:
            args.append(f"{flag} {source[param]}")
        elif param in bp:
            args.append(f"{flag} {bp[param]}")

    # Regression weight
    rw = fp.get("regression_weight", bp.get("regression_weight", 1.0))
    args.append(f"--regression-weight {rw}")

    # N factors
    n_factors = fp.get("n_factors", config_data.get("n_factors", 348) if config_data else 348)
    args.append(f"--n-factors {n_factors}")

    # N DRGPs for combined mode
    n_free = bp.get("n_free_factors", config_data.get("n_drgps") if config_data else None)
    if n_free is not None:
        args.append(f"--n-drgps {n_free}")

    return " \\\n    ".join(args)


def main():
    print("=" * 90)
    print("DRGP Best Hyperparameters from Bayesian Optimization (latest runs)")
    print("=" * 90)

    all_configs = []

    for prior in PRIORS:
        for mode in MODES:
            for exp in EXPERIMENTS:
                data = get_latest_best_params(prior, mode, exp)
                if data is None:
                    continue
                config = get_latest_best_config(prior, mode, exp)
                bvm = data["best_val_metrics"]
                print(f"\n--- {prior}/{mode}/{exp} ---")
                print(f"  Source: {os.path.basename(data['_source_file'])}")
                print(f"  Best trial: {data['best_trial']}/{data['n_trials']}")
                print(f"  Severity AUC: {bvm['val_severity_auc']:.4f}")
                print(f"  Outcome AUC: {bvm['val_outcome_auc']:.4f}")
                print(f"  Mean AUC: {data['best_auc']:.4f}")

                cli = params_to_cli_args(data, config)
                print(f"\n  CLI args:\n    {cli}")

                all_configs.append({
                    "prior": prior,
                    "mode": mode,
                    "experiment": exp,
                    "severity_auc": bvm["val_severity_auc"],
                    "outcome_auc": bvm["val_outcome_auc"],
                    "mean_auc": data["best_auc"],
                    "best_trial": data["best_trial"],
                    "cli_args": cli,
                    "source_file": data["_source_file"],
                    "params_data": data,
                    "config_data": config,
                })

    # Save consolidated configs
    output_path = os.path.join(
        "/labs/Aguiar/SSPA_BRAY/results/sim_pathway",
        "drgp_best_configs.json"
    )
    # Serialize without non-serializable fields
    serializable = []
    for c in all_configs:
        s = {k: v for k, v in c.items() if k not in ("params_data", "config_data")}
        s["best_params"] = c["params_data"]["best_params"]
        s["fixed_params"] = c["params_data"].get("fixed_params", {})
        if c["config_data"]:
            s["full_config"] = c["config_data"]
        serializable.append(s)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n\nSaved consolidated configs: {output_path}")

    return all_configs


if __name__ == "__main__":
    main()
