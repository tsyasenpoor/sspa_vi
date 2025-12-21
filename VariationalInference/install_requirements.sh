#!/bin/bash

# Install required packages for hyperparameter optimization

echo "Installing Optuna and dependencies..."

pip install optuna
pip install plotly  # For Optuna visualizations
pip install kaleido  # For saving plots

echo ""
echo "Installation complete!"
echo ""
echo "Verifying installation..."
python -c "import optuna; print(f'✓ Optuna version: {optuna.__version__}')"
python -c "import plotly; print(f'✓ Plotly version: {plotly.__version__}')"

echo ""
echo "All dependencies installed successfully!"
