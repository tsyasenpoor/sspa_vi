"""
Generic Configuration for Variational Inference
================================================

This module provides a flexible configuration system for both VI and SVI models
that can be used with any single-cell dataset.

Usage:
    from VariationalInference.config import VIConfig, SVIConfig

    # Use VI defaults
    config = VIConfig(n_factors=50)

    # Use SVI defaults
    config = SVIConfig(n_factors=50, batch_size=128)

    # Load from JSON
    config = VIConfig.from_json('my_config.json')

    # Create model from config
    from VariationalInference.vi import VI
    from VariationalInference.svi import SVI

    model = VI(**config.model_params())  # for VIConfig
    model = SVI(**config.model_params())  # for SVIConfig
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json


@dataclass
class VIConfig:
    """
    Configuration for VI model and training.

    This dataclass holds all hyperparameters and settings for running
    variational inference experiments. It supports JSON serialization
    for reproducibility and experiment tracking.

    Parameters
    ----------
    n_factors : int
        Number of latent gene programs to discover. Required.

    Model Hyperparameters (Prior Parameters)
    -----------------------------------------
    alpha_theta : float, default=2.0
        Prior shape for sample factor activities.
    alpha_beta : float, default=2.0
        Prior shape for gene loadings.
    alpha_xi : float, default=2.0
        Prior shape for depth correction.
    alpha_eta : float, default=2.0
        Prior shape for gene scaling.
    lambda_xi : float, default=1.5
        Prior rate for depth correction.
    lambda_eta : float, default=1.5
        Prior rate for gene scaling.
    sigma_v : float, default=0.2
        Prior std for classification weights.
    sigma_gamma : float, default=0.5
        Prior std for auxiliary feature effects.
    pi_v : float, default=0.2
        Prior probability of v being active (slab).
    pi_beta : float, default=0.05
        Prior probability of beta being active (slab).

    Training Parameters
    -------------------
    max_iter : int, default=200
        Maximum training iterations.
    tol : float, default=10.0
        Absolute ELBO change tolerance for convergence.
    rel_tol : float, default=2e-4
        Relative ELBO change tolerance for convergence.
    elbo_freq : int, default=10
        Compute ELBO every N iterations.
    min_iter : int, default=50
        Minimum iterations before checking convergence.
    patience : int, default=5
        Consecutive small improvements before early stopping.

    Damping Parameters (Advanced)
    -----------------------------
    theta_damping : float, default=0.8
        Damping for theta updates (0=no update, 1=full update).
    beta_damping : float, default=0.8
        Damping for beta updates.
    v_damping : float, default=0.7
        Damping for v updates.
    gamma_damping : float, default=0.7
        Damping for gamma updates.
    xi_damping : float, default=0.9
        Damping for xi updates.
    eta_damping : float, default=0.9
        Damping for eta updates.

    Data Parameters
    ---------------
    label_column : str, default='t2dm'
        Column name in adata.obs for labels.
    aux_columns : list of str, default=None
        Column names in adata.obs for auxiliary features.
    train_ratio : float, default=0.7
        Proportion of data for training.
    val_ratio : float, default=0.15
        Proportion of data for validation.
    min_cells_expressing : float, default=0.02
        Minimum fraction of cells expressing a gene (filter threshold).

    Output Parameters
    -----------------
    output_dir : str, optional
        Directory to save results.
    cache_dir : str, optional
        Directory to cache preprocessed data.
    verbose : bool, default=True
        Whether to print progress during training.
    debug : bool, default=False
        Whether to print detailed debug information.
    """

    # Required
    n_factors: int

    # Model hyperparameters
    alpha_theta: float = 2.0
    alpha_beta: float = 2.0
    alpha_xi: float = 2.0
    alpha_eta: float = 2.0
    lambda_xi: float = 1.5
    lambda_eta: float = 1.5
    sigma_v: float = 0.2
    sigma_gamma: float = 0.5
    pi_v: float = 0.2
    pi_beta: float = 0.05
    spike_variance_v: float = 1e-6
    spike_value_beta: float = 1e-6

    # Training parameters
    max_iter: int = 200
    tol: float = 10.0
    rel_tol: float = 2e-4
    elbo_freq: int = 10
    min_iter: int = 50
    patience: int = 5

    # Damping parameters
    theta_damping: float = 0.8
    beta_damping: float = 0.8
    v_damping: float = 0.7
    gamma_damping: float = 0.7
    xi_damping: float = 0.9
    eta_damping: float = 0.9
    adaptive_damping: bool = True

    # Data parameters
    label_column: str = 't2dm'
    aux_columns: Optional[List[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    min_cells_expressing: float = 0.02

    # Output parameters
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    verbose: bool = True
    debug: bool = False

    # Random state (None = true random for scientific experiments)
    random_state: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")

    def model_params(self) -> Dict[str, Any]:
        """
        Get parameters for VI model constructor.

        Returns
        -------
        dict
            Parameters to pass to VI.__init__()
        """
        return {
            'n_factors': self.n_factors,
            'alpha_theta': self.alpha_theta,
            'alpha_beta': self.alpha_beta,
            'alpha_xi': self.alpha_xi,
            'alpha_eta': self.alpha_eta,
            'lambda_xi': self.lambda_xi,
            'lambda_eta': self.lambda_eta,
            'sigma_v': self.sigma_v,
            'sigma_gamma': self.sigma_gamma,
            'random_state': self.random_state,
            'pi_v': self.pi_v,
            'pi_beta': self.pi_beta,
            'spike_variance_v': self.spike_variance_v,
            'spike_value_beta': self.spike_value_beta,
        }

    def training_params(self) -> Dict[str, Any]:
        """
        Get parameters for VI.fit() method.

        Returns
        -------
        dict
            Parameters to pass to VI.fit()
        """
        return {
            'max_iter': self.max_iter,
            'tol': self.tol,
            'rel_tol': self.rel_tol,
            'elbo_freq': self.elbo_freq,
            'min_iter': self.min_iter,
            'patience': self.patience,
            'theta_damping': self.theta_damping,
            'beta_damping': self.beta_damping,
            'v_damping': self.v_damping,
            'gamma_damping': self.gamma_damping,
            'xi_damping': self.xi_damping,
            'eta_damping': self.eta_damping,
            'adaptive_damping': self.adaptive_damping,
            'verbose': self.verbose,
            'debug': self.debug,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        path : str
            Path to save JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VIConfig':
        """
        Create config from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with configuration values.

        Returns
        -------
        VIConfig
            Configuration object.
        """
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> 'VIConfig':
        """
        Load configuration from JSON file.

        Parameters
        ----------
        path : str
            Path to JSON configuration file.

        Returns
        -------
        VIConfig
            Configuration object.
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def copy(self, **updates) -> 'VIConfig':
        """
        Create a copy of this config with optional updates.

        Parameters
        ----------
        **updates
            Keyword arguments to update in the copy.

        Returns
        -------
        VIConfig
            New configuration object with updates applied.
        """
        d = self.to_dict()
        d.update(updates)
        return VIConfig.from_dict(d)


@dataclass
class SVIConfig:
    """
    Configuration for SVI (Stochastic Variational Inference) model and training.

    This dataclass holds all hyperparameters for running SVI experiments.
    SVI is preferred for large datasets as it processes mini-batches instead
    of the full dataset.

    Parameters
    ----------
    n_factors : int
        Number of latent gene programs to discover. Required.

    SVI-Specific Parameters
    -----------------------
    batch_size : int, default=128
        Size of mini-batches for stochastic updates.
    learning_rate : float, default=0.01
        Initial learning rate for global parameter updates.
    learning_rate_decay : float, default=0.75
        Decay exponent (kappa) for learning rate schedule.
    learning_rate_delay : float, default=1.0
        Delay parameter (tau) for learning rate schedule.
    learning_rate_min : float, default=1e-4
        Minimum learning rate to prevent stagnation in late training.
    warmup_epochs : int, default=5
        Number of epochs for learning rate warmup.
    local_iterations : int, default=5
        Number of iterations to optimize local parameters per mini-batch.
    regression_weight : float, default=1.0
        Weight for classification objective. Higher values make classification
        more influential on theta updates. Values around 1.0-10.0 are typical.

    Model Hyperparameters (Same as VI)
    ----------------------------------
    alpha_theta, alpha_beta, alpha_xi, alpha_eta : float, default=2.0
    lambda_xi, lambda_eta : float, default=1.5
    sigma_v : float, default=0.2
    sigma_gamma : float, default=0.5
    pi_v : float, default=0.2
    pi_beta : float, default=0.05

    Training Parameters
    -------------------
    max_epochs : int, default=100
        Maximum number of epochs (passes through data).
    elbo_freq : int, default=10
        Compute ELBO every N mini-batch iterations.
    min_epochs : int, default=10
        Minimum epochs before checking convergence.
    patience : int, default=5
        Early stopping patience.
    tol : float, default=10.0
        Absolute ELBO tolerance.
    rel_tol : float, default=2e-4
        Relative ELBO tolerance.
    """

    # Required
    n_factors: int

    # SVI-specific parameters
    batch_size: int = 128
    learning_rate: float = 0.01
    learning_rate_decay: float = 0.75
    learning_rate_delay: float = 1.0
    learning_rate_min: float = 1e-4
    warmup_epochs: int = 5
    local_iterations: int = 5
    regression_weight: float = 1.0

    # Model hyperparameters (same as VI)
    alpha_theta: float = 2.0
    alpha_beta: float = 2.0
    alpha_xi: float = 2.0
    alpha_eta: float = 2.0
    lambda_xi: float = 1.5
    lambda_eta: float = 1.5
    sigma_v: float = 0.2
    sigma_gamma: float = 0.5
    pi_v: float = 0.2
    pi_beta: float = 0.05
    spike_variance_v: float = 1e-6
    spike_value_beta: float = 1e-6

    # Training parameters
    max_epochs: int = 100
    elbo_freq: int = 10
    min_epochs: int = 10
    patience: int = 5
    tol: float = 10.0
    rel_tol: float = 2e-4

    # Data parameters
    label_column: str = 't2dm'
    aux_columns: Optional[List[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    min_cells_expressing: float = 0.02

    # Output parameters
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    verbose: bool = True
    debug: bool = False

    # Random state
    random_state: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0 < self.learning_rate_decay <= 1:
            raise ValueError("learning_rate_decay must be in (0, 1]")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")

    def model_params(self) -> Dict[str, Any]:
        """
        Get parameters for SVI model constructor.

        Returns
        -------
        dict
            Parameters to pass to SVI.__init__()
        """
        return {
            'n_factors': self.n_factors,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_decay': self.learning_rate_decay,
            'learning_rate_delay': self.learning_rate_delay,
            'learning_rate_min': self.learning_rate_min,
            'warmup_epochs': self.warmup_epochs,
            'local_iterations': self.local_iterations,
            'regression_weight': self.regression_weight,
            'alpha_theta': self.alpha_theta,
            'alpha_beta': self.alpha_beta,
            'alpha_xi': self.alpha_xi,
            'alpha_eta': self.alpha_eta,
            'lambda_xi': self.lambda_xi,
            'lambda_eta': self.lambda_eta,
            'sigma_v': self.sigma_v,
            'sigma_gamma': self.sigma_gamma,
            'random_state': self.random_state,
            'pi_v': self.pi_v,
            'pi_beta': self.pi_beta,
            'spike_variance_v': self.spike_variance_v,
            'spike_value_beta': self.spike_value_beta,
        }

    def training_params(self) -> Dict[str, Any]:
        """
        Get parameters for SVI.fit() method.

        Returns
        -------
        dict
            Parameters to pass to SVI.fit()
        """
        return {
            'max_epochs': self.max_epochs,
            'elbo_freq': self.elbo_freq,
            'min_epochs': self.min_epochs,
            'patience': self.patience,
            'tol': self.tol,
            'rel_tol': self.rel_tol,
            'verbose': self.verbose,
            'debug': self.debug,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SVIConfig':
        """Create config from dictionary."""
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> 'SVIConfig':
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def copy(self, **updates) -> 'SVIConfig':
        """Create a copy of this config with optional updates."""
        d = self.to_dict()
        d.update(updates)
        return SVIConfig.from_dict(d)


# Preset configurations for common use cases
PRESETS = {
    'default': VIConfig(n_factors=50),

    'small': VIConfig(
        n_factors=20,
        max_iter=100,
        min_iter=20,
    ),

    'large': VIConfig(
        n_factors=100,
        max_iter=300,
        min_iter=100,
    ),

    'conservative': VIConfig(
        n_factors=50,
        theta_damping=0.5,
        beta_damping=0.5,
        v_damping=0.3,
        gamma_damping=0.3,
    ),

    'fast': VIConfig(
        n_factors=50,
        max_iter=100,
        elbo_freq=20,
        min_iter=20,
        patience=3,
    ),
}

# SVI presets
SVI_PRESETS = {
    'default': SVIConfig(n_factors=50, batch_size=128),

    'small': SVIConfig(
        n_factors=20,
        batch_size=64,
        max_epochs=50,
        min_epochs=5,
    ),

    'large': SVIConfig(
        n_factors=100,
        batch_size=256,
        max_epochs=200,
        min_epochs=20,
    ),

    'fast': SVIConfig(
        n_factors=50,
        batch_size=256,
        max_epochs=50,
        elbo_freq=20,
        min_epochs=5,
        patience=3,
    ),
}


def get_preset(name: str, method: str = 'vi') -> Union[VIConfig, SVIConfig]:
    """
    Get a preset configuration by name.

    Parameters
    ----------
    name : str
        Preset name. One of: 'default', 'small', 'large', 'conservative', 'fast'
    method : str, default='vi'
        Inference method. Either 'vi' for batch VI or 'svi' for stochastic VI.

    Returns
    -------
    VIConfig or SVIConfig
        Configuration object.

    Examples
    --------
    >>> config = get_preset('fast', method='vi')
    >>> model = VI(**config.model_params())

    >>> config = get_preset('fast', method='svi')
    >>> model = SVI(**config.model_params())
    """
    if method.lower() == 'svi':
        if name not in SVI_PRESETS:
            raise ValueError(f"Unknown SVI preset '{name}'. Available: {list(SVI_PRESETS.keys())}")
        return SVI_PRESETS[name].copy()
    else:
        if name not in PRESETS:
            raise ValueError(f"Unknown VI preset '{name}'. Available: {list(PRESETS.keys())}")
        return PRESETS[name].copy()
