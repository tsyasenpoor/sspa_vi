#!/bin/bash

# Quick test of the hyperparameter optimization script
# Runs only 3 trials with minimal iterations to verify it works

echo "=========================================="
echo "Testing Hyperparameter Optimization Setup"
echo "=========================================="
echo ""
echo "This will run 3 quick trials to verify everything works."
echo "Each trial will run for max 50 iterations."
echo "Expected time: ~5-10 minutes"
echo ""
echo "Starting test..."
echo ""

cd /home/user/sspa_vi/VariationalInference

python hyperparameter_optimization.py \
    --n_trials 3 \
    --max_iter 50 \
    --n_jobs 1 \
    --output_dir ./optuna_test \
    --study_name test_run

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check the results in: ./optuna_test/"
echo ""
echo "If successful, you can run the full optimization with:"
echo "  python hyperparameter_optimization.py --n_trials 100 --max_iter 200 --output_dir ./optuna_results"
echo ""
