# Hyperparameter Optimization Guide

This guide explains how to use Bayesian optimization (via Optuna) to find the best hyperparameters for your VI model.

## Overview

The optimization process consists of three main steps:

1. **Run Bayesian Optimization** - Search for best hyperparameters using validation F1 score
2. **Review Results** - Analyze which hyperparameters matter most
3. **Train Final Model** - Use best parameters to train the final model on full data

## Scripts

### 1. `hyperparameter_optimization.py`
Main optimization script that uses Optuna to search the hyperparameter space.

**What it optimizes:**
- `n_factors` (20-100): Number of latent factors
- `alpha_theta`, `alpha_beta`, `alpha_xi`, `alpha_eta` (0.1-5.0): Gamma prior shape parameters
- `lambda_xi`, `lambda_eta` (0.5-5.0): Gamma prior rate parameters
- `sigma_v`, `sigma_gamma` (0.1-5.0): Gaussian prior variances
- `pi_v` (0.05-0.5): Prior probability of v being active
- `pi_beta` (0.01-0.2): Prior probability of beta being active
- `spike_variance_v` (1e-8 to 1e-4): Variance for spike component in v
- `spike_value_beta` (1e-8 to 1e-4): Value for spike component in beta
- **Damping parameters** (0.3-0.95): Control convergence speed

**Objective:** Maximize validation F1 score

### 2. `run_with_best_params.py`
Trains a final model using the best hyperparameters found by optimization.

## Quick Start

### Step 1: Run a Small Test (Recommended)

Start with a small number of trials to ensure everything works:

```bash
cd /home/user/sspa_vi/VariationalInference

python hyperparameter_optimization.py \
    --n_trials 10 \
    --max_iter 100 \
    --n_jobs 1 \
    --output_dir ./optuna_test
```

This will:
- Run 10 optimization trials
- Each trial trains for max 100 iterations
- Save results to `./optuna_test/`
- Take approximately 30-60 minutes

### Step 2: Run Full Optimization

Once the test works, run the full optimization:

```bash
python hyperparameter_optimization.py \
    --n_trials 100 \
    --max_iter 200 \
    --n_jobs 1 \
    --output_dir ./optuna_results
```

**For parallel optimization** (if you have multiple cores):
```bash
python hyperparameter_optimization.py \
    --n_trials 100 \
    --max_iter 200 \
    --n_jobs 4 \
    --output_dir ./optuna_results
```

**With timeout** (e.g., run for 24 hours max):
```bash
python hyperparameter_optimization.py \
    --n_trials 100 \
    --max_iter 200 \
    --timeout 86400 \
    --output_dir ./optuna_results
```

### Step 3: Review Results

After optimization completes, you'll find these files in the output directory:

```
optuna_results/
├── vi_optimization_YYYYMMDD_HHMMSS_best_params.json  # Best hyperparameters
├── vi_optimization_YYYYMMDD_HHMMSS_trials.csv        # All trial results
├── vi_optimization_YYYYMMDD_HHMMSS_results.pkl       # Full Optuna study object
├── vi_optimization_YYYYMMDD_HHMMSS_history.html      # Optimization history plot
└── vi_optimization_YYYYMMDD_HHMMSS_importance.html   # Parameter importance plot
```

**View the best parameters:**
```bash
cat optuna_results/vi_optimization_*_best_params.json
```

**View all trials:**
```bash
# Load in Python
import pandas as pd
df = pd.read_csv('optuna_results/vi_optimization_*_trials.csv')
df.sort_values('value', ascending=False).head(10)  # Top 10 trials
```

**Interactive visualizations:**
Open the HTML files in a browser to see:
- **history.html**: How F1 score improved over trials
- **importance.html**: Which hyperparameters matter most

### Step 4: Train Final Model with Best Parameters

```bash
python run_with_best_params.py \
    --study_file optuna_results/vi_optimization_*_best_params.json \
    --max_iter 300 \
    --output_dir ./best_model_results
```

This will:
- Load best hyperparameters
- Train final model with more iterations (300)
- Evaluate on train/val/test sets
- Save model and predictions

## Understanding the Results

### What is Bayesian Optimization?

Unlike grid search (which tries every combination), Bayesian optimization:

1. **Tries a few random hyperparameters** (first 5-10 trials)
2. **Builds a probabilistic model** of which hyperparameters work well
3. **Intelligently picks** the next hyperparameters to try
4. **Balances exploration vs exploitation** - tries new regions while refining good ones

### Key Metrics

- **Validation F1**: Primary metric being optimized
- **Validation Accuracy**: Also reported for reference
- **Validation AUC**: Also reported for reference
- **Final ELBO**: Convergence indicator (higher is better)

### Interpreting Parameter Importance

The `importance.html` plot shows which hyperparameters have the biggest impact on F1 score:

- **High importance**: Tuning this parameter significantly affects performance
- **Low importance**: This parameter doesn't matter much (could use default)

### Common Scenarios

**Scenario 1: Optimization finds much better parameters**
- Your validation F1 improves by >5%
- ✅ Use the optimized parameters!

**Scenario 2: Minimal improvement**
- Your validation F1 improves by <2%
- The original parameters were already pretty good
- Still worth using optimized params for marginal gains

**Scenario 3: Optimization fails**
- All trials have very low F1 scores
- Check for bugs or data issues
- Try narrowing the search ranges

## Advanced Usage

### Resume Interrupted Optimization

Optuna can resume from a saved study:

```python
import optuna
import pickle

# Load previous study
with open('optuna_results/vi_optimization_*_results.pkl', 'rb') as f:
    study = pickle.load(f)

# Continue optimization
study.optimize(objective, n_trials=50)  # Run 50 more trials
```

### Custom Search Ranges

Edit `hyperparameter_optimization.py` to adjust the ranges:

```python
# Example: Focus search on smaller n_factors
'n_factors': trial.suggest_int('n_factors', 10, 50, step=5),

# Example: Narrow sigma_v range based on prior knowledge
'sigma_v': trial.suggest_float('sigma_v', 0.5, 2.0, log=True),
```

### Two-Stage Optimization

For faster results, do a coarse search then refine:

**Stage 1: Coarse search**
```bash
# Wide ranges, fewer iterations per trial
python hyperparameter_optimization.py --n_trials 50 --max_iter 100
```

**Stage 2: Fine-tune**
```bash
# Edit script to narrow ranges around best params from Stage 1
python hyperparameter_optimization.py --n_trials 50 --max_iter 200
```

## Computational Considerations

### Time Estimates

- **Per trial**: 5-10 minutes (depends on data size and max_iter)
- **10 trials**: ~1 hour
- **50 trials**: ~5 hours
- **100 trials**: ~10 hours

### Memory Usage

- Each trial loads the full dataset
- Parallel jobs multiply memory usage
- For large datasets: use `--n_jobs 1` or `--n_jobs 2`

### Recommendations

- **Quick test**: 10 trials, 100 iterations
- **Standard**: 50-100 trials, 200 iterations
- **Thorough**: 100-200 trials, 300 iterations
- **Production**: Two-stage approach (coarse then fine)

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `--n_jobs` to 1, or reduce `--max_iter`

### Issue: Trials failing with errors
**Solution**: Check the error messages, may need to adjust search ranges

### Issue: All trials have similar F1 scores
**Solution**: The hyperparameters may not matter much, or ranges are too narrow

### Issue: F1 score is 0.0 for many trials
**Solution**: Some hyperparameter combinations don't converge. This is normal - Optuna learns to avoid them.

## Next Steps After Optimization

1. **Analyze parameter importance** - Focus future experiments on important parameters
2. **Check for overfitting** - Compare train/val/test F1 scores
3. **Ensemble models** - Train multiple models with top-10 hyperparameters and ensemble predictions
4. **Domain analysis** - Examine which genes/pathways are selected by the best model

## Files Created

### By `hyperparameter_optimization.py`:
- `*_best_params.json` - Best hyperparameters (human-readable)
- `*_trials.csv` - All trial results (for analysis)
- `*_results.pkl` - Full Optuna study (for programmatic access)
- `*_history.html` - Optimization progress plot
- `*_importance.html` - Parameter importance plot

### By `run_with_best_params.py`:
- `best_model.pkl` - Trained model object
- `train_predictions.csv` - Training set predictions
- `val_predictions.csv` - Validation set predictions
- `test_predictions.csv` - Test set predictions
- `summary.json` - Comprehensive results summary

## Questions?

Common questions:

**Q: How many trials should I run?**
A: 50-100 is usually sufficient. Diminishing returns after that.

**Q: Should I optimize damping parameters?**
A: Yes - they're included because they significantly affect convergence.

**Q: Can I run this on a cluster?**
A: Yes! Use `--n_jobs` to parallelize across cores.

**Q: What if I want to optimize for AUC instead of F1?**
A: Edit the `objective` function to return `val_auc` instead of `val_f1`.

**Q: How do I use different train/val/test splits?**
A: Edit the data loading section in both scripts to use your custom splits.
