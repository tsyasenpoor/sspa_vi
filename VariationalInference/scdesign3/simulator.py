"""
ScDesign3 Simulator - Python interface to scDesign3 R package

This module provides a Python wrapper for running scDesign3 simulations,
handling data conversion between Python (AnnData) and R (SingleCellExperiment),
and loading results back into Python.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ScDesign3Config:
    """Configuration for scDesign3 simulation."""

    # Input data
    input_file: str
    celltype_column: Optional[str] = None
    pseudotime_column: Optional[str] = None
    spatial_columns: Optional[List[str]] = None

    # Simulation parameters
    n_cells: int = 0  # 0 means same as input
    n_genes: int = 0  # 0 means all genes
    gene_selection: str = "variable"  # "variable" or "random"
    min_cells_expressing: float = 0.0  # Filter genes

    # Model parameters
    family: str = "nb"  # "poisson", "nb", "zip", "zinb", "gaussian"
    copula: str = "gaussian"  # "gaussian" or "vine"
    mu_formula: Optional[str] = None  # Custom formula for mean model

    # Computation
    n_cores: int = 1
    seed: int = 42
    return_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            "input_file": str(self.input_file),
            "celltype_column": self.celltype_column or "",
            "pseudotime_column": self.pseudotime_column or "",
            "spatial_columns": self.spatial_columns or [],
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "gene_selection": self.gene_selection,
            "min_cells_expressing": self.min_cells_expressing,
            "family": self.family,
            "copula": self.copula,
            "mu_formula": self.mu_formula or "",
            "n_cores": self.n_cores,
            "seed": self.seed,
            "return_model": self.return_model,
        }


@dataclass
class ScDesign3Result:
    """Results from scDesign3 simulation."""

    counts: np.ndarray
    gene_names: List[str]
    cell_names: List[str]
    metadata: Optional[pd.DataFrame] = None
    simulation_info: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None

    @property
    def n_cells(self) -> int:
        """Number of simulated cells."""
        return self.counts.shape[1]

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.counts.shape[0]

    @property
    def shape(self) -> tuple:
        """Shape of count matrix (genes x cells)."""
        return self.counts.shape

    def to_anndata(self):
        """
        Convert results to AnnData object.

        Returns
        -------
        adata : AnnData
            AnnData object with simulated counts
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata is required: pip install anndata")

        # AnnData expects obs x var (cells x genes)
        X = self.counts.T  # Transpose from genes x cells to cells x genes

        # Create obs DataFrame
        if self.metadata is not None:
            obs = self.metadata.copy()
        else:
            obs = pd.DataFrame(index=self.cell_names)

        # Create var DataFrame
        var = pd.DataFrame(index=self.gene_names)

        # Create AnnData
        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
        )

        # Store simulation info in uns
        adata.uns["scdesign3_info"] = self.simulation_info

        return adata

    def to_dataframe(self, transpose: bool = True) -> pd.DataFrame:
        """
        Convert counts to pandas DataFrame.

        Parameters
        ----------
        transpose : bool
            If True, return cells x genes (default). If False, genes x cells.

        Returns
        -------
        df : pd.DataFrame
            Count matrix as DataFrame
        """
        if transpose:
            return pd.DataFrame(
                self.counts.T, index=self.cell_names, columns=self.gene_names
            )
        else:
            return pd.DataFrame(
                self.counts, index=self.gene_names, columns=self.cell_names
            )

    def save_pickle(self, output_dir: str, prefix: str = "scdesign3"):
        """
        Save results in pickle format compatible with existing sspa_vi loaders.

        Parameters
        ----------
        output_dir : str
            Directory to save files
        prefix : str
            Prefix for output files
        """
        import pickle

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save count matrix (cells x genes for compatibility)
        X = self.counts.T  # cells x genes
        with open(output_dir / f"{prefix}_X.pkl", "wb") as f:
            pickle.dump(X, f)

        # Save gene list
        with open(output_dir / "gene_list.txt", "w") as f:
            for gene in self.gene_names:
                f.write(f"{gene}\n")

        # Save metadata/features
        if self.metadata is not None:
            with open(output_dir / f"{prefix}_features.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

        # Save simulation info
        with open(output_dir / f"{prefix}_info.json", "w") as f:
            json.dump(self.simulation_info, f, indent=2)

        print(f"Saved scDesign3 results to {output_dir}")


class ScDesign3Simulator:
    """
    Python interface to scDesign3 for generating synthetic single-cell data.

    scDesign3 learns statistical models from reference datasets and generates
    synthetic data preserving gene expression distributions and correlations.

    Parameters
    ----------
    r_script_path : str, optional
        Path to the R script. If None, uses the bundled script.
    r_executable : str, optional
        Path to R executable. Default: "Rscript"

    Examples
    --------
    >>> simulator = ScDesign3Simulator()
    >>> result = simulator.simulate(
    ...     input_file="reference.h5ad",
    ...     celltype_column="cell_type",
    ...     n_cells=5000,
    ...     family="nb"
    ... )
    >>> adata = result.to_anndata()
    """

    def __init__(
        self,
        r_script_path: Optional[str] = None,
        r_executable: str = "Rscript",
    ):
        self.r_executable = r_executable

        # Default to bundled R script
        if r_script_path is None:
            self.r_script_path = Path(__file__).parent / "run_scdesign3.R"
        else:
            self.r_script_path = Path(r_script_path)

        # Verify R script exists
        if not self.r_script_path.exists():
            raise FileNotFoundError(
                f"R script not found: {self.r_script_path}\n"
                "Run setup_r_env.sh to install scDesign3."
            )

    def check_installation(self) -> bool:
        """
        Check if R and scDesign3 are properly installed.

        Returns
        -------
        bool
            True if installation is valid
        """
        try:
            result = subprocess.run(
                [
                    self.r_executable,
                    "-e",
                    "library(scDesign3); cat(packageVersion('scDesign3'))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                version = result.stdout.strip().split("\n")[-1]
                print(f"scDesign3 version {version} is installed")
                return True
            else:
                print(f"scDesign3 not found: {result.stderr}")
                return False
        except FileNotFoundError:
            print(f"R executable not found: {self.r_executable}")
            return False
        except subprocess.TimeoutExpired:
            print("Timeout checking R installation")
            return False

    def simulate(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        celltype_column: Optional[str] = None,
        pseudotime_column: Optional[str] = None,
        spatial_columns: Optional[List[str]] = None,
        n_cells: int = 0,
        n_genes: int = 0,
        gene_selection: str = "variable",
        min_cells_expressing: float = 0.0,
        family: str = "nb",
        copula: str = "gaussian",
        mu_formula: Optional[str] = None,
        n_cores: int = 1,
        seed: int = 42,
        return_model: bool = False,
        verbose: bool = True,
    ) -> ScDesign3Result:
        """
        Run scDesign3 simulation.

        Parameters
        ----------
        input_file : str
            Path to reference data (.h5ad, .rds, or .csv)
        output_dir : str, optional
            Directory to save outputs. If None, uses temp directory.
        celltype_column : str, optional
            Column in metadata containing cell type labels
        pseudotime_column : str, optional
            Column containing pseudotime values (for trajectory data)
        spatial_columns : list of str, optional
            Columns containing spatial coordinates
        n_cells : int
            Number of cells to generate. 0 = same as input.
        n_genes : int
            Number of genes to use. 0 = all genes.
        gene_selection : str
            How to select genes: "variable" or "random"
        min_cells_expressing : float
            Filter genes expressed in fewer than this fraction of cells
        family : str
            Distribution family: "poisson", "nb", "zip", "zinb", "gaussian"
        copula : str
            Copula type: "gaussian" (faster) or "vine" (more flexible)
        mu_formula : str, optional
            Custom formula for mean model (mgcv syntax)
        n_cores : int
            Number of CPU cores for parallel processing
        seed : int
            Random seed for reproducibility
        return_model : bool
            Whether to save fitted models (increases output size)
        verbose : bool
            Print progress messages

        Returns
        -------
        ScDesign3Result
            Object containing simulated counts, metadata, and info
        """
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="scdesign3_")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = ScDesign3Config(
            input_file=str(input_path.absolute()),
            celltype_column=celltype_column,
            pseudotime_column=pseudotime_column,
            spatial_columns=spatial_columns,
            n_cells=n_cells,
            n_genes=n_genes,
            gene_selection=gene_selection,
            min_cells_expressing=min_cells_expressing,
            family=family,
            copula=copula,
            mu_formula=mu_formula,
            n_cores=n_cores,
            seed=seed,
            return_model=return_model,
        )

        # Write config to temp file
        config_file = output_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        # Run R script
        if verbose:
            print("Running scDesign3 simulation...")
            print(f"  Input: {input_file}")
            print(f"  Output: {output_dir}")
            print(f"  Family: {family}, Copula: {copula}")

        cmd = [
            self.r_executable,
            str(self.r_script_path),
            str(config_file),
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout for large datasets
            )

            if verbose:
                print(result.stdout)

            if result.returncode != 0:
                print("STDERR:", result.stderr)
                raise RuntimeError(f"scDesign3 failed:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("scDesign3 timed out (>4 hours)")

        # Load results
        return self._load_results(output_path)

    def _load_results(self, output_dir: Path) -> ScDesign3Result:
        """Load simulation results from output directory."""
        output_dir = Path(output_dir)

        # Load counts
        counts_file = output_dir / "simulated_counts.csv"
        if not counts_file.exists():
            raise FileNotFoundError(f"Results not found: {counts_file}")

        counts_df = pd.read_csv(counts_file, index_col=0)
        counts = counts_df.values
        gene_names = list(counts_df.index)
        cell_names = list(counts_df.columns)

        # Load metadata if available
        meta_file = output_dir / "simulated_metadata.csv"
        if meta_file.exists():
            metadata = pd.read_csv(meta_file, index_col=0)
        else:
            metadata = None

        # Load simulation info
        info_file = output_dir / "simulation_info.json"
        if info_file.exists():
            with open(info_file) as f:
                simulation_info = json.load(f)
        else:
            simulation_info = {}

        return ScDesign3Result(
            counts=counts,
            gene_names=gene_names,
            cell_names=cell_names,
            metadata=metadata,
            simulation_info=simulation_info,
            output_dir=str(output_dir),
        )

    def simulate_from_anndata(
        self,
        adata,
        output_dir: Optional[str] = None,
        celltype_column: Optional[str] = None,
        **kwargs,
    ) -> ScDesign3Result:
        """
        Run simulation directly from an AnnData object.

        This is a convenience method that saves the AnnData to a temporary
        h5ad file before running the simulation.

        Parameters
        ----------
        adata : AnnData
            Reference AnnData object
        output_dir : str, optional
            Output directory
        celltype_column : str, optional
            Column in adata.obs for cell types
        **kwargs
            Additional arguments passed to simulate()

        Returns
        -------
        ScDesign3Result
            Simulation results
        """
        # Create temp directory for h5ad
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="scdesign3_")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save AnnData to h5ad
        h5ad_file = output_path / "reference.h5ad"
        adata.write_h5ad(h5ad_file)

        return self.simulate(
            input_file=str(h5ad_file),
            output_dir=str(output_path),
            celltype_column=celltype_column,
            **kwargs,
        )


def simulate_with_disease_signal(
    reference_file: str,
    output_dir: str,
    celltype_column: Optional[str] = None,
    n_cells: int = 0,
    n_disease_genes: int = 30,
    disease_fraction: float = 0.5,
    boost_factor: float = 2.0,
    family: str = "nb",
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate synthetic data with scDesign3 and add disease signal.

    This combines scDesign3's realistic data generation with the disease
    signal injection from synth_case.py.

    Parameters
    ----------
    reference_file : str
        Path to reference h5ad file
    output_dir : str
        Output directory
    celltype_column : str, optional
        Cell type column
    n_cells : int
        Number of cells to generate
    n_disease_genes : int
        Number of genes to be disease-associated
    disease_fraction : float
        Fraction of cells to be "diseased"
    boost_factor : float
        Multiplicative boost for disease genes
    family : str
        Distribution family for scDesign3
    seed : int
        Random seed
    **kwargs
        Additional arguments for ScDesign3Simulator.simulate()

    Returns
    -------
    dict
        Contains simulated data and ground truth information
    """
    from scipy.stats import poisson

    # Step 1: Generate realistic baseline with scDesign3
    simulator = ScDesign3Simulator()
    result = simulator.simulate(
        input_file=reference_file,
        output_dir=output_dir,
        celltype_column=celltype_column,
        n_cells=n_cells,
        family=family,
        seed=seed,
        **kwargs,
    )

    # Step 2: Add disease signal (similar to synth_case.py)
    rng = np.random.RandomState(seed)

    # Get count matrix (genes x cells)
    X_control = result.counts.copy()
    n_genes, n_cells_actual = X_control.shape

    # Select disease genes
    disease_gene_indices = rng.choice(n_genes, n_disease_genes, replace=False)
    disease_gene_names = [result.gene_names[i] for i in disease_gene_indices]

    # Select disease cells
    n_disease_cells = int(n_cells_actual * disease_fraction)
    disease_cell_indices = rng.choice(n_cells_actual, n_disease_cells, replace=False)

    # Apply multiplicative boost
    X_disease = X_control.copy()
    for i in disease_cell_indices:
        for j in disease_gene_indices:
            lambda_control = X_control[j, i]
            lambda_disease = lambda_control * boost_factor
            X_disease[j, i] = poisson.rvs(lambda_disease, random_state=rng)

    # Create labels
    y = np.zeros(n_cells_actual, dtype=int)
    y[disease_cell_indices] = 1

    # Update result with disease data
    result.counts = X_disease

    # Create ground truth info
    ground_truth = {
        "disease_gene_indices": disease_gene_indices.tolist(),
        "disease_gene_names": disease_gene_names,
        "disease_cell_indices": disease_cell_indices.tolist(),
        "n_disease_genes": n_disease_genes,
        "n_disease_cells": n_disease_cells,
        "boost_factor": boost_factor,
        "disease_fraction": disease_fraction,
    }

    # Save outputs
    output_path = Path(output_dir)

    # Save as pickle for compatibility with existing loaders
    result.save_pickle(output_dir, prefix="scdesign3")

    # Save labels
    import pickle
    with open(output_path / "y.pkl", "wb") as f:
        pickle.dump(y, f)

    # Save ground truth
    with open(output_path / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    return {
        "result": result,
        "y": y,
        "ground_truth": ground_truth,
    }
