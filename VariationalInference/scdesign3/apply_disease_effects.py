"""
Apply Disease Effects to scDesign3 Simulated Data

Implements the formula: λ = λ_sex + λ_pop + λ_disease × λ_g

For each cell in subgroup (s, p, d) and gene g:
    new_rate = μ_g × λ(s, p, d, g)

Where:
- μ_g is the baseline count from scDesign3
- λ_g = 1 for background genes, λ_g > 1 for disease-relevant genes
- s ∈ {0,1} = sex, p ∈ {0,1} = population, d ∈ {0,1} = disease status
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class DiseaseEffectConfig:
    """Configuration for disease effect simulation."""
    
    # Lambda values for sex effect (additive)
    lambda_s0: float = 1.0
    lambda_s1: float = 1.0
    
    # Lambda values for population effect (additive)
    lambda_p0: float = 1.0
    lambda_p1: float = 1.0
    
    # Lambda values for disease effect (multiplied by lambda_g)
    lambda_d0: float = 1.0
    lambda_d1: float = 10.0
    
    # Gene-specific amplification for disease genes
    lambda_g: float = 2.0
    
    # Number of disease-relevant genes
    n_disease_genes: int = 100
    
    # Disease gene selection method: "random", "high_var", "high_expr"
    gene_selection: str = "random"
    
    # Subgroup proportions (if None, balanced across 8 groups)
    subgroup_proportions: Optional[Dict[Tuple[int, int, int], float]] = None
    
    # Random seed
    seed: int = 42
    
    def compute_lambda(self, sex: int, pop: int, disease: int, is_disease_gene: bool) -> float:
        """Compute λ for a given cell-gene combination."""
        lam_s = self.lambda_s0 if sex == 0 else self.lambda_s1
        lam_p = self.lambda_p0 if pop == 0 else self.lambda_p1
        lam_d = self.lambda_d0 if disease == 0 else self.lambda_d1
        lam_g = self.lambda_g if is_disease_gene else 1.0
        return lam_s + lam_p + lam_d * lam_g
    
    def get_fold_change_summary(self) -> Dict:
        """Compute expected fold changes for different subgroups."""
        results = {}
        
        for s, p in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            key = f"s{s}_p{p}"
            
            # Disease genes
            lam_ctrl_dg = self.compute_lambda(s, p, 0, True)
            lam_case_dg = self.compute_lambda(s, p, 1, True)
            fc_dg = lam_case_dg / lam_ctrl_dg
            
            # Background genes
            lam_ctrl_bg = self.compute_lambda(s, p, 0, False)
            lam_case_bg = self.compute_lambda(s, p, 1, False)
            fc_bg = lam_case_bg / lam_ctrl_bg
            
            results[key] = {
                "disease_gene_fc": fc_dg,
                "background_gene_fc": fc_bg,
                "differential_fc": fc_dg / fc_bg,
            }
        
        return results
    
    def to_dict(self) -> Dict:
        return {
            "lambda_s0": self.lambda_s0,
            "lambda_s1": self.lambda_s1,
            "lambda_p0": self.lambda_p0,
            "lambda_p1": self.lambda_p1,
            "lambda_d0": self.lambda_d0,
            "lambda_d1": self.lambda_d1,
            "lambda_g": self.lambda_g,
            "n_disease_genes": self.n_disease_genes,
            "gene_selection": self.gene_selection,
            "seed": self.seed,
        }


# Predefined configurations matching paper experiments
PRESET_CONFIGS = {
    "exp0_easy": DiseaseEffectConfig(
        lambda_s0=1, lambda_s1=1, lambda_p0=1, lambda_p1=1,
        lambda_d0=1, lambda_d1=100, lambda_g=1.0,
        n_disease_genes=100,
    ),
    "exp1_sex_effect": DiseaseEffectConfig(
        lambda_s0=10, lambda_s1=1, lambda_p0=1, lambda_p1=1,
        lambda_d0=1, lambda_d1=100, lambda_g=1.0,
        n_disease_genes=100,
    ),
    "exp2_sex_pop_effect": DiseaseEffectConfig(
        lambda_s0=10, lambda_s1=1, lambda_p0=10, lambda_p1=1,
        lambda_d0=1, lambda_d1=100, lambda_g=1.0,
        n_disease_genes=100,
    ),
    # More realistic/challenging configurations
    "moderate": DiseaseEffectConfig(
        lambda_s0=2, lambda_s1=1, lambda_p0=2, lambda_p1=1,
        lambda_d0=1, lambda_d1=8, lambda_g=2.0,
        n_disease_genes=100,
    ),
    "challenging": DiseaseEffectConfig(
        lambda_s0=1.5, lambda_s1=1, lambda_p0=1.5, lambda_p1=1,
        lambda_d0=1, lambda_d1=4, lambda_g=2.0,
        n_disease_genes=100,
    ),
    "hard": DiseaseEffectConfig(
        lambda_s0=1.2, lambda_s1=1, lambda_p0=1.2, lambda_p1=1,
        lambda_d0=1, lambda_d1=2.5, lambda_g=1.5,
        n_disease_genes=50,
    ),
}


class DiseaseEffectSimulator:
    """Apply disease effects to scDesign3 baseline data."""
    
    def __init__(self, config: DiseaseEffectConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def load_baseline(self, input_dir: str) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
        """Load scDesign3 baseline data."""
        input_dir = Path(input_dir)
        
        # Load counts (genes x cells)
        counts_df = pd.read_csv(input_dir / "simulated_counts.csv", index_col=0)
        counts = counts_df.values.astype(np.float64)
        gene_names = list(counts_df.index)
        cell_names = list(counts_df.columns)
        
        # Load metadata
        metadata = pd.read_csv(input_dir / "simulated_metadata.csv", index_col=0)
        
        return counts, gene_names, cell_names, metadata
    
    def select_disease_genes(
        self, 
        counts: np.ndarray, 
        gene_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Select disease-relevant genes based on config."""
        n_genes = counts.shape[0]
        n_select = min(self.config.n_disease_genes, n_genes)
        
        if self.config.gene_selection == "random":
            indices = self.rng.choice(n_genes, n_select, replace=False)
        elif self.config.gene_selection == "high_var":
            gene_vars = counts.var(axis=1)
            indices = np.argsort(gene_vars)[-n_select:]
        elif self.config.gene_selection == "high_expr":
            gene_means = counts.mean(axis=1)
            # Select from top 50% expressed genes
            candidate_idx = np.where(gene_means > np.median(gene_means))[0]
            indices = self.rng.choice(candidate_idx, min(n_select, len(candidate_idx)), replace=False)
        else:
            raise ValueError(f"Unknown gene_selection: {self.config.gene_selection}")
        
        indices = np.sort(indices)
        names = [gene_names[i] for i in indices]
        return indices, names
    
    def assign_subgroups(
        self, 
        n_cells: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign cells to subgroups (sex, population, disease)."""
        if self.config.subgroup_proportions is not None:
            # Custom proportions
            groups = list(self.config.subgroup_proportions.keys())
            probs = list(self.config.subgroup_proportions.values())
            probs = np.array(probs) / sum(probs)  # normalize
            
            assignments = self.rng.choice(len(groups), n_cells, p=probs)
            sex = np.array([groups[i][0] for i in assignments])
            pop = np.array([groups[i][1] for i in assignments])
            disease = np.array([groups[i][2] for i in assignments])
        else:
            # Balanced design: equal cells per subgroup
            sex = self.rng.randint(0, 2, n_cells)
            pop = self.rng.randint(0, 2, n_cells)
            disease = self.rng.randint(0, 2, n_cells)
        
        return sex, pop, disease
    
    def apply_effects(
        self,
        counts: np.ndarray,
        disease_gene_indices: np.ndarray,
        sex: np.ndarray,
        pop: np.ndarray,
        disease: np.ndarray,
    ) -> np.ndarray:
        """
        Apply disease effects to count matrix.
        
        For each cell i and gene g:
            λ = compute_lambda(sex[i], pop[i], disease[i], g in disease_genes)
            new_count ~ Poisson(counts[g,i] * λ)
        """
        n_genes, n_cells = counts.shape
        new_counts = np.zeros_like(counts)
        
        # Create mask for disease genes
        is_disease_gene = np.zeros(n_genes, dtype=bool)
        is_disease_gene[disease_gene_indices] = True
        
        # Vectorized computation for efficiency
        for g in range(n_genes):
            is_dg = is_disease_gene[g]
            
            for i in range(n_cells):
                lam = self.config.compute_lambda(sex[i], pop[i], disease[i], is_dg)
                base_rate = counts[g, i]
                new_rate = base_rate * lam
                
                # Sample from Poisson
                if new_rate > 0:
                    new_counts[g, i] = self.rng.poisson(new_rate)
                else:
                    new_counts[g, i] = 0
        
        return new_counts.astype(np.int32)
    
    def apply_effects_vectorized(
        self,
        counts: np.ndarray,
        disease_gene_indices: np.ndarray,
        sex: np.ndarray,
        pop: np.ndarray,
        disease: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized version of apply_effects for performance.
        """
        n_genes, n_cells = counts.shape
        
        # Create disease gene mask
        is_disease_gene = np.zeros(n_genes, dtype=bool)
        is_disease_gene[disease_gene_indices] = True
        
        # Compute lambda matrix (genes x cells)
        lambda_matrix = np.ones((n_genes, n_cells), dtype=np.float64)
        
        cfg = self.config
        
        # Additive terms (same for all genes)
        lambda_s = np.where(sex == 0, cfg.lambda_s0, cfg.lambda_s1)  # (n_cells,)
        lambda_p = np.where(pop == 0, cfg.lambda_p0, cfg.lambda_p1)  # (n_cells,)
        
        # Disease term depends on gene type
        lambda_d = np.where(disease == 0, cfg.lambda_d0, cfg.lambda_d1)  # (n_cells,)
        
        # For background genes: λ = λ_s + λ_p + λ_d * 1
        bg_lambda = lambda_s + lambda_p + lambda_d  # (n_cells,)
        
        # For disease genes: λ = λ_s + λ_p + λ_d * λ_g
        dg_lambda = lambda_s + lambda_p + lambda_d * cfg.lambda_g  # (n_cells,)
        
        # Assign to matrix
        lambda_matrix[~is_disease_gene, :] = bg_lambda
        lambda_matrix[is_disease_gene, :] = dg_lambda
        
        # Compute new rates
        new_rates = counts * lambda_matrix
        
        # Sample from Poisson
        new_counts = self.rng.poisson(new_rates)
        
        return new_counts.astype(np.int32)
    
    def simulate(
        self,
        input_dir: str,
        output_dir: str,
        use_vectorized: bool = True,
    ) -> Dict:
        """
        Run complete simulation pipeline.
        
        Parameters
        ----------
        input_dir : str
            Directory with scDesign3 baseline data
        output_dir : str
            Output directory for modified data
        use_vectorized : bool
            Use vectorized computation (faster)
            
        Returns
        -------
        dict
            Contains modified data and ground truth
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading baseline from {input_dir}...")
        counts, gene_names, cell_names, metadata = self.load_baseline(input_dir)
        n_genes, n_cells = counts.shape
        print(f"  Loaded {n_genes} genes × {n_cells} cells")
        
        # Select disease genes
        print(f"Selecting {self.config.n_disease_genes} disease genes ({self.config.gene_selection})...")
        disease_gene_idx, disease_gene_names = self.select_disease_genes(counts, gene_names)
        print(f"  Selected genes: {disease_gene_names[:5]}... (showing first 5)")
        
        # Assign subgroups
        print("Assigning cells to subgroups...")
        sex, pop, disease = self.assign_subgroups(n_cells)
        
        # Print subgroup distribution
        from collections import Counter
        subgroup_counts = Counter(zip(sex, pop, disease))
        print("  Subgroup distribution:")
        for (s, p, d), count in sorted(subgroup_counts.items()):
            print(f"    sex={s}, pop={p}, disease={d}: {count} cells")
        
        # Apply effects
        print("Applying disease effects...")
        if use_vectorized:
            new_counts = self.apply_effects_vectorized(
                counts, disease_gene_idx, sex, pop, disease
            )
        else:
            new_counts = self.apply_effects(
                counts, disease_gene_idx, sex, pop, disease
            )
        
        # Compute summary statistics
        print("\nComputing effect summary...")
        fc_summary = self.config.get_fold_change_summary()
        for subgroup, stats in fc_summary.items():
            print(f"  {subgroup}: disease_gene_FC={stats['disease_gene_fc']:.2f}x, "
                  f"background_FC={stats['background_gene_fc']:.2f}x")
        
        # Empirical fold change for disease genes
        case_mask = disease == 1
        ctrl_mask = disease == 0
        
        dg_case_mean = new_counts[disease_gene_idx][:, case_mask].mean()
        dg_ctrl_mean = new_counts[disease_gene_idx][:, ctrl_mask].mean()
        bg_idx = ~np.isin(np.arange(n_genes), disease_gene_idx)
        bg_case_mean = new_counts[bg_idx][:, case_mask].mean()
        bg_ctrl_mean = new_counts[bg_idx][:, ctrl_mask].mean()
        
        print(f"\nEmpirical fold changes:")
        print(f"  Disease genes: {dg_case_mean:.4f} / {dg_ctrl_mean:.4f} = {dg_case_mean/dg_ctrl_mean:.2f}x")
        print(f"  Background genes: {bg_case_mean:.4f} / {bg_ctrl_mean:.4f} = {bg_case_mean/bg_ctrl_mean:.2f}x")
        
        # Save outputs
        print(f"\nSaving outputs to {output_dir}...")
        
        # Save count matrix
        counts_df = pd.DataFrame(new_counts, index=gene_names, columns=cell_names)
        counts_df.to_csv(output_dir / "modified_counts.csv")
        
        # Update metadata with subgroup assignments
        metadata["sex"] = sex
        metadata["population"] = pop
        metadata["disease"] = disease
        metadata.to_csv(output_dir / "modified_metadata.csv")
        
        # Save ground truth
        ground_truth = {
            "disease_gene_indices": disease_gene_idx.tolist(),
            "disease_gene_names": disease_gene_names,
            "n_disease_genes": len(disease_gene_idx),
            "config": self.config.to_dict(),
            "fold_change_summary": fc_summary,
            "empirical_fc": {
                "disease_genes_case": float(dg_case_mean),
                "disease_genes_ctrl": float(dg_ctrl_mean),
                "background_case": float(bg_case_mean),
                "background_ctrl": float(bg_ctrl_mean),
            },
            "subgroup_counts": {str(k): v for k, v in subgroup_counts.items()},
        }
        
        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)
        
        # Save disease labels for SVI
        y = disease.astype(np.int32)
        np.save(output_dir / "disease_labels.npy", y)
        
        # Save covariates
        covariates = pd.DataFrame({
            "sex": sex,
            "population": pop,
            "disease": disease,
        }, index=cell_names)
        covariates.to_csv(output_dir / "covariates.csv")
        
        print("Done!")
        
        return {
            "counts": new_counts,
            "gene_names": gene_names,
            "cell_names": cell_names,
            "metadata": metadata,
            "ground_truth": ground_truth,
            "disease_labels": y,
        }


def run_power_analysis(
    input_dir: str,
    configs: Dict[str, DiseaseEffectConfig],
    n_bootstrap: int = 100,
) -> pd.DataFrame:
    """
    Run power analysis for different configurations.
    
    Estimates detection power via bootstrap simulation.
    """
    from scipy import stats
    
    # Load baseline
    counts_df = pd.read_csv(Path(input_dir) / "simulated_counts.csv", index_col=0)
    counts = counts_df.values
    gene_means = counts.mean(axis=1)
    
    results = []
    
    for name, cfg in configs.items():
        # Theoretical fold changes
        fc_summary = cfg.get_fold_change_summary()
        
        # Use s=0, p=0 subgroup as reference
        dg_fc = fc_summary["s0_p0"]["disease_gene_fc"]
        bg_fc = fc_summary["s0_p0"]["background_gene_fc"]
        
        # Power calculation for median-expression gene
        median_mu = np.median(gene_means)
        n_per_group = 10000 // 8  # assuming balanced
        
        ctrl_rate = median_mu * (cfg.lambda_s0 + cfg.lambda_p0 + cfg.lambda_d0 * cfg.lambda_g)
        case_rate = median_mu * (cfg.lambda_s0 + cfg.lambda_p0 + cfg.lambda_d1 * cfg.lambda_g)
        
        # Z-test power
        se = np.sqrt(ctrl_rate/n_per_group + case_rate/n_per_group)
        z = (case_rate - ctrl_rate) / se
        power = 1 - stats.norm.cdf(1.96 - z)
        
        results.append({
            "config": name,
            "disease_gene_fc": dg_fc,
            "background_fc": bg_fc,
            "differential_fc": dg_fc / bg_fc,
            "ctrl_rate": ctrl_rate,
            "case_rate": case_rate,
            "z_statistic": z,
            "power_0.05": power,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply disease effects to scDesign3 data")
    parser.add_argument("--input", "-i", required=True, help="Input directory with baseline data")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--config", "-c", default="challenging", 
                       choices=list(PRESET_CONFIGS.keys()),
                       help="Configuration preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--power-analysis", action="store_true", 
                       help="Run power analysis for all configs")
    
    args = parser.parse_args()
    
    if args.power_analysis:
        print("Running power analysis...")
        results = run_power_analysis(args.input, PRESET_CONFIGS)
        print(results.to_string())
    else:
        config = PRESET_CONFIGS[args.config]
        config.seed = args.seed
        
        simulator = DiseaseEffectSimulator(config)
        result = simulator.simulate(args.input, args.output)
