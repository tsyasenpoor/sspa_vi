# Quick Start Guide - Hyperparameter Optimization

## What Was Created

I've implemented a complete Bayesian hyperparameter optimization system using Optuna. Here's what you have:

### Files Created:
1. **hyperparameter_optimization.py** - Main optimization script
2. **run_with_best_params.py** - Script to train final model with best parameters
3. **HYPERPARAMETER_OPTIMIZATION_GUIDE.md** - Comprehensive documentation
4. **install_requirements.sh** - Install Optuna and dependencies
5. **test_optimization.sh** - Quick test script

### What Gets Optimized:
All 12 hyperparameters you specified:
- `n_factors` (20-100)
- `alpha_theta`, `alpha_beta`, `alpha_xi`, `alpha_eta` (0.1-5.0)
- `lambda_xi`, `lambda_eta` (0.5-5.0)
- `sigma_v`, `sigma_gamma` (0.1-5.0)
- `pi_v` (0.05-0.5)
- `pi_beta` (0.01-0.2)
- `spike_variance_v`, `spike_value_beta` (1e-8 to 1e-4)
- Plus 6 damping parameters (0.3-0.95)

**Objective**: Maximize validation F1 score

## Installation (Already Done)

Dependencies have been installed:
- ✓ Optuna 4.6.0
- ✓ Plotly 6.5.0 (for visualizations)
- ✓ Pandas, scikit-learn, scipy

## How to Run

### Step 1: Quick Test (5-10 minutes)

Run a small test to ensure everything works:

```bash
cd /home/user/sspa_vi/VariationalInference

python hyperparameter_optimization.py \
    --n_trials 5 \
    --max_iter 50 \
    --n_jobs 1 \
    --output_dir ./optuna_test
```

### Step 2: Medium Run (~3-5 hours)

Once the test works, run a medium-sized optimization:

```bash
python hyperparameter_optimization.py \
    --n_trials 50 \
    --max_iter 150 \
    --n_jobs 1 \
    --output_dir ./optuna_medium
```

### Step 3: Full Optimization (~10-15 hours)

For production results:

```bash
python hyperparameter_optimization.py \
    --n_trials 100 \
    --max_iter 200 \
    --n_jobs 1 \
    --output_dir ./optuna_results
```

**Parallel execution** (if you have multiple cores):
```bash
python hyperparameter_optimization.py \
    --n_trials 100 \
    --max_iter 200 \
    --n_jobs 4 \
    --output_dir ./optuna_results
```

### Step 4: Review Results

After optimization completes:

```bash
# View best parameters
cat optuna_results/vi_optimization_*_best_params.json

# View all trials (sorted by F1)
python -c "
import pandas as pd
df = pd.read_csv('optuna_results/vi_optimization_*_trials.csv')
print(df.sort_values('value', ascending=False)[['number', 'value', 'user_attrs_val_accuracy', 'user_attrs_val_auc']].head(10))
"

# Open visualizations in browser
firefox optuna_results/*_history.html
firefox optuna_results/*_importance.html
```

### Step 5: Train Final Model

Use the best hyperparameters to train your final model:

```bash
python run_with_best_params.py \
    --study_file optuna_results/vi_optimization_*_best_params.json \
    --max_iter 300 \
    --output_dir ./best_model_results
```

## Command-Line Options

### hyperparameter_optimization.py

```
--n_trials       Number of trials to run (default: 100)
--max_iter       Max VI iterations per trial (default: 200)
--n_jobs         Parallel jobs (default: 1)
--output_dir     Where to save results (default: ./optuna_results)
--study_name     Custom study name (optional)
--timeout        Timeout in seconds (optional)
```

### run_with_best_params.py

```
--study_file     Path to *_best_params.json file (required)
--max_iter       Max iterations for final training (default: 300)
--output_dir     Where to save final results (default: ./best_model_results)
```

## Understanding the Output

### During Optimization

You'll see output like:
```
Trial 0: Testing hyperparameters
  n_factors: 50
  alpha_theta: 0.834521
  ...

Validation Results:
  F1 Score: 0.7234
  Accuracy: 0.7891
  AUC: 0.8123
  Final ELBO: -182345.21
```

### Best Trial Summary

At the end:
```
Best trial:
  Trial number: 42
  Validation F1: 0.8234
  Validation Accuracy: 0.8456
  Validation AUC: 0.8901

Best hyperparameters:
  n_factors: 60
  alpha_theta: 1.234567
  ...
```

### Files Generated

```
optuna_results/
├── vi_optimization_YYYYMMDD_HHMMSS_best_params.json    # ← Use this for final training
├── vi_optimization_YYYYMMDD_HHMMSS_trials.csv         # All trial results
├── vi_optimization_YYYYMMDD_HHMMSS_results.pkl        # Full study object
├── vi_optimization_YYYYMMDD_HHMMSS_history.html       # Optimization progress
└── vi_optimization_YYYYMMDD_HHMMSS_importance.html    # Parameter importance
```

## Tips for Success

### 1. Start Small
- Run 5-10 trials first to verify everything works
- Check that ELBO is converging and F1 scores are reasonable
- Then scale up to 50-100 trials

### 2. Monitor Progress
```bash
# In another terminal, watch the trials CSV file:
watch -n 10 'tail -20 optuna_results/*_trials.csv'
```

### 3. Resume if Interrupted
If optimization stops, you can load and continue:
```python
import optuna
import pickle

with open('optuna_results/vi_optimization_*_results.pkl', 'rb') as f:
    study = pickle.load(f)

# Continue with more trials
study.optimize(objective, n_trials=50)
```

### 4. Adjust Search Ranges
If trials are all failing or all similar, edit the ranges in `hyperparameter_optimization.py`:

```python
# Line ~70: Narrow the search range
'n_factors': trial.suggest_int('n_factors', 30, 70, step=10),  # Instead of 20-100
```

### 5. Two-Stage Strategy
For faster results:
1. **Coarse search**: 30 trials, 100 iterations each → Find good region
2. **Fine search**: Edit ranges based on stage 1, then 50 trials, 200 iterations

## Troubleshooting

### "ModuleNotFoundError: No module named 'optuna'"
Run: `./install_requirements.sh`

### "ModuleNotFoundError: No module named 'VariationalInference'"
Run from the correct directory:
```bash
cd /home/user/sspa_vi/VariationalInference
python hyperparameter_optimization.py ...
```

### "FileNotFoundError: ... sspa_bcell/df.pkl"
Check that your data files are at: `/labs/Aguiar/SSPA_BRAY/BRay/sspa_bcell/`

### All trials have F1 = 0.0
Some hyperparameter combinations don't work. This is normal - Optuna learns to avoid them.
If ALL trials fail, check your data and reduce search ranges.

### Out of memory
Reduce `--n_jobs` to 1 or reduce `--max_iter`

## Expected Performance

### Baseline (from quick_reference.py)
```python
# Current hyperparameters:
n_factors=50
alpha_theta=0.5
alpha_beta=2.0
# ... (hardcoded values)
```

### After Optimization
You should see:
- **F1 improvement**: 2-10% higher on validation set
- **More stable convergence**: Better damping parameters
- **Better sparsity**: Optimized spike-and-slab parameters

## What to Do with Results

1. **Compare train/val/test F1**
   - If val F1 >> test F1: may be overfitting, try more regularization
   - If train/val/test all similar: good generalization

2. **Check parameter importance**
   - Focus future work on high-importance parameters
   - Low-importance parameters can use defaults

3. **Ensemble top models**
   - Train models with top-5 hyperparameters
   - Average their predictions for better results

4. **Analyze learned factors**
   - Examine gene loadings (beta matrix)
   - See which programs drive classification (v weights)

## Next Steps

After getting best hyperparameters:

1. **Train final model** with more iterations (300-500)
2. **Analyze results**: Which genes/pathways selected?
3. **Cross-validation**: Use best params with different CV folds
4. **Sensitivity analysis**: How robust are results to hyperparameter changes?
5. **Biological interpretation**: What do the learned factors mean?

## Questions?

See **HYPERPARAMETER_OPTIMIZATION_GUIDE.md** for detailed documentation.

Good luck! 🚀
