#!/usr/bin/env python3
"""
Comprehensive DRGP Analysis across Multiple SVI Runs

This script analyzes ALL gene programs from multiple SVI runs to identify
reproducible consensus Disease-Related Gene Programs (DRGPs), including:
- Disease-risk programs (positive v coefficients)
- Protective programs (negative v coefficients)
- Nuisance/housekeeping programs (near-zero v coefficients)

Revised Pipeline:
1. Load ALL DRGPs from all runs (35 programs x 20 runs = 700 total)
2. Cluster by beta similarity (ignoring v during clustering)
3. For each cluster, compute run coverage and v statistics
4. Categorize consensus programs by reproducibility and v patterns
5. Report with effect sizes

Usage:
    python analyze_all_drgps.py /path/to/runs_folder --output results/all_drgps

Author: Generated for SVI analysis pipeline
"""

import argparse
import gzip
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


# =============================================================================
# Enums and Constants
# =============================================================================

class ProgramCategory(Enum):
    """Categories for consensus gene programs based on v coefficients."""
    DISEASE_RISK = "disease_risk"      # Reproducible + consistently positive v
    PROTECTIVE = "protective"           # Reproducible + consistently negative v
    NUISANCE = "nuisance"              # Reproducible + near-zero v
    NON_REPRODUCIBLE = "non_reproducible"  # Not enough run coverage


# Thresholds for categorization
V_EFFECT_THRESHOLD = 0.1  # |mean_v| > threshold to be considered non-nuisance
V_CONSISTENCY_THRESHOLD = 0.5  # |mean_v| / std_v ratio for consistency


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DRGP:
    """Represents a single Disease-Related Gene Program."""
    run_id: str
    program_id: str
    beta_vector: np.ndarray  # Full gene loading vector
    v_weight: float  # Classification weight (coefficient for disease)
    gene_names: list

    @property
    def full_id(self) -> str:
        return f"{self.run_id}_{self.program_id}"

    def get_binary_genes(self, threshold: float = 0.1) -> set:
        """Get genes above threshold as a set."""
        return set(np.array(self.gene_names)[self.beta_vector > threshold])

    def get_top_genes(self, n: int = 50) -> list:
        """Get top n genes by beta value."""
        indices = np.argsort(self.beta_vector)[::-1][:n]
        return [(self.gene_names[i], self.beta_vector[i]) for i in indices]


@dataclass
class DRGPCluster:
    """Represents a cluster of similar DRGPs across runs."""
    cluster_id: int
    members: list  # List of DRGP objects

    @property
    def run_coverage(self) -> int:
        """Number of unique runs contributing to this cluster."""
        return len(set(drgp.run_id for drgp in self.members))

    @property
    def run_ids(self) -> set:
        """Set of run IDs in this cluster."""
        return set(drgp.run_id for drgp in self.members)

    @property
    def has_duplicate_runs(self) -> bool:
        """Check if any run contributes more than one DRGP."""
        run_counts = {}
        for drgp in self.members:
            run_counts[drgp.run_id] = run_counts.get(drgp.run_id, 0) + 1
        return any(c > 1 for c in run_counts.values())

    def get_consensus_beta(self) -> np.ndarray:
        """Average beta vector across all members."""
        betas = np.array([drgp.beta_vector for drgp in self.members])
        return np.mean(betas, axis=0)

    def get_consensus_genes(self, threshold: float = 0.1, min_freq: float = 0.5) -> set:
        """Get genes that appear in at least min_freq fraction of members."""
        gene_counts = {}
        for drgp in self.members:
            for gene in drgp.get_binary_genes(threshold):
                gene_counts[gene] = gene_counts.get(gene, 0) + 1

        min_count = int(len(self.members) * min_freq)
        return set(g for g, c in gene_counts.items() if c >= min_count)

    def get_v_statistics(self) -> Dict[str, float]:
        """Compute v-weight statistics across cluster members."""
        v_weights = np.array([drgp.v_weight for drgp in self.members])
        return {
            'mean_v': np.mean(v_weights),
            'std_v': np.std(v_weights),
            'median_v': np.median(v_weights),
            'min_v': np.min(v_weights),
            'max_v': np.max(v_weights),
            'abs_mean_v': np.abs(np.mean(v_weights)),
            'sign_consistency': np.abs(np.mean(np.sign(v_weights)))  # 1.0 = all same sign
        }


@dataclass
class ConsensusProgram:
    """A consensus gene program derived from a cluster."""
    cluster_id: int
    n_contributing_runs: int
    consensus_beta: np.ndarray
    consensus_genes_union: set
    consensus_genes_intersection: set
    consensus_genes_majority: set  # Genes in >50% of members
    gene_names: list
    member_drgps: list
    quality_metrics: dict
    v_statistics: dict
    category: ProgramCategory

    @property
    def effect_size(self) -> float:
        """Effect size = |mean_v|"""
        return self.v_statistics['abs_mean_v']

    @property
    def mean_v(self) -> float:
        """Mean v coefficient across members."""
        return self.v_statistics['mean_v']

    def get_top_genes(self, n: int = 50) -> list:
        """Get top n genes by consensus beta."""
        indices = np.argsort(self.consensus_beta)[::-1][:n]
        return [(self.gene_names[i], self.consensus_beta[i]) for i in indices]

    def get_gene_dataframe(self, threshold: float = 0.05) -> pd.DataFrame:
        """Get DataFrame of genes above threshold with their consensus values."""
        mask = self.consensus_beta > threshold
        genes = np.array(self.gene_names)[mask]
        values = self.consensus_beta[mask]

        # Sort by value
        sorted_idx = np.argsort(values)[::-1]

        df = pd.DataFrame({
            'gene': genes[sorted_idx],
            'consensus_beta': values[sorted_idx],
            'in_majority': [g in self.consensus_genes_majority for g in genes[sorted_idx]]
        })
        return df


@dataclass
class ConsensusAnalysisResults:
    """Container for all analysis results."""
    all_drgps: list = field(default_factory=list)
    similarity_matrix_continuous: np.ndarray = None
    similarity_matrix_binary: np.ndarray = None
    linkage_matrix: np.ndarray = None
    clusters: list = field(default_factory=list)
    consensus_programs: list = field(default_factory=list)
    # Categorized programs
    disease_risk_programs: list = field(default_factory=list)
    protective_programs: list = field(default_factory=list)
    nuisance_programs: list = field(default_factory=list)
    non_reproducible_clusters: list = field(default_factory=list)
    enrichment_results: dict = field(default_factory=dict)


# =============================================================================
# Step 1: Load ALL DRGPs (no filtering)
# =============================================================================

def load_drgps_from_run(run_dir: Path, prefix: str = 'svi') -> list:
    """
    Load ALL DRGPs from a single run directory.

    Args:
        run_dir: Path to the run results directory
        prefix: File prefix (default 'svi')

    Returns:
        List of DRGP objects (all programs, not filtered)
    """
    gene_program_file = run_dir / f'{prefix}_gene_programs.csv.gz'

    if not gene_program_file.exists():
        # Try without compression
        gene_program_file = run_dir / f'{prefix}_gene_programs.csv'
        if not gene_program_file.exists():
            print(f"Warning: No gene program file found in {run_dir}")
            return []

    # Load the gene programs
    if str(gene_program_file).endswith('.gz'):
        df = pd.read_csv(gene_program_file, index_col=0, compression='gzip')
    else:
        df = pd.read_csv(gene_program_file, index_col=0)

    # Extract run ID from directory name
    run_id = run_dir.name

    # Identify v_weight columns and gene columns
    v_weight_cols = [c for c in df.columns if c.startswith('v_weight_')]
    gene_cols = [c for c in df.columns if not c.startswith('v_weight_')]

    drgps = []
    for program_id in df.index:
        row = df.loc[program_id]

        # Get beta vector (gene loadings)
        beta_vector = row[gene_cols].values.astype(float)

        # Get v_weight for class 1 (disease class)
        if 'v_weight_class1' in v_weight_cols:
            v_weight = row['v_weight_class1']
        elif len(v_weight_cols) > 0:
            v_weight = row[v_weight_cols[-1]]
        else:
            v_weight = 0.0  # No classification weight available

        drgp = DRGP(
            run_id=run_id,
            program_id=program_id,
            beta_vector=beta_vector,
            v_weight=v_weight,
            gene_names=list(gene_cols)
        )
        drgps.append(drgp)

    return drgps


def load_all_drgps(runs_folder: Path, prefix: str = 'svi') -> list:
    """
    Load ALL DRGPs from all run directories.

    Args:
        runs_folder: Path to folder containing all run subdirectories
        prefix: File prefix

    Returns:
        List of all DRGP objects (no filtering applied)
    """
    all_drgps = []

    # Find all run directories
    run_dirs = sorted([d for d in runs_folder.iterdir() if d.is_dir()])

    if len(run_dirs) == 0:
        if (runs_folder / f'{prefix}_gene_programs.csv.gz').exists():
            print(f"Found gene programs directly in {runs_folder}")
            drgps = load_drgps_from_run(runs_folder, prefix)
            all_drgps.extend(drgps)
            return all_drgps
        else:
            raise ValueError(f"No run directories found in {runs_folder}")

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        drgps = load_drgps_from_run(run_dir, prefix)
        if drgps:
            print(f"  {run_dir.name}: {len(drgps)} gene programs")
            all_drgps.extend(drgps)

    return all_drgps


# =============================================================================
# Step 2: Build Similarity Matrices (beta only, ignore v)
# =============================================================================

def compute_continuous_similarity(drgps: list, method: str = 'pearson') -> np.ndarray:
    """
    Compute pairwise similarity using continuous beta vectors.

    NOTE: This uses only beta (gene loadings), NOT v coefficients.

    Args:
        drgps: List of DRGP objects
        method: 'pearson' or 'cosine'

    Returns:
        N x N similarity matrix
    """
    beta_matrix = np.array([d.beta_vector for d in drgps])

    if method == 'pearson':
        sim_matrix = np.corrcoef(beta_matrix)
    elif method == 'cosine':
        sim_matrix = cosine_similarity(beta_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle NaN values
    sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)

    return sim_matrix


def compute_binary_similarity(drgps: list, threshold: float = 0.1) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity using binary gene sets.

    Args:
        drgps: List of DRGP objects
        threshold: Beta threshold for binary conversion

    Returns:
        N x N similarity matrix
    """
    n = len(drgps)
    gene_sets = [d.get_binary_genes(threshold) for d in drgps]

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if len(gene_sets[i]) == 0 and len(gene_sets[j]) == 0:
                sim = 1.0 if i == j else 0.0
            elif len(gene_sets[i]) == 0 or len(gene_sets[j]) == 0:
                sim = 0.0
            else:
                intersection = len(gene_sets[i] & gene_sets[j])
                union = len(gene_sets[i] | gene_sets[j])
                sim = intersection / union if union > 0 else 0.0
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return sim_matrix


# =============================================================================
# Step 3: Clustering (by beta similarity)
# =============================================================================

def perform_hierarchical_clustering(
    similarity_matrix: np.ndarray,
    method: str = 'average'
) -> np.ndarray:
    """
    Perform hierarchical clustering on the similarity matrix.

    Args:
        similarity_matrix: N x N similarity matrix
        method: Linkage method ('average', 'complete', 'ward', etc.)

    Returns:
        Linkage matrix from scipy
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Ensure non-negative distances
    distance_matrix = np.maximum(distance_matrix, 0)

    # Convert to condensed form
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method=method)

    return Z


def get_clusters(
    linkage_matrix: np.ndarray,
    drgps: list,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None
) -> list:
    """
    Extract clusters from hierarchical clustering.

    Args:
        linkage_matrix: Output from hierarchical clustering
        drgps: List of DRGP objects
        n_clusters: Number of clusters (mutually exclusive with distance_threshold)
        distance_threshold: Distance threshold for flat clusters

    Returns:
        List of DRGPCluster objects
    """
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    elif distance_threshold is not None:
        labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    else:
        labels = fcluster(linkage_matrix, 0.5, criterion='distance')

    # Group DRGPs by cluster
    cluster_dict = {}
    for drgp, label in zip(drgps, labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(drgp)

    clusters = [
        DRGPCluster(cluster_id=cid, members=members)
        for cid, members in sorted(cluster_dict.items())
    ]

    return clusters


def find_optimal_clusters(
    linkage_matrix: np.ndarray,
    drgps: list,
    n_runs: int,
    min_coverage: float = 0.5
) -> Tuple[list, float]:
    """
    Find optimal number of clusters based on run coverage.

    Args:
        linkage_matrix: Output from hierarchical clustering
        drgps: List of DRGP objects
        n_runs: Total number of runs
        min_coverage: Minimum fraction of runs a cluster should cover

    Returns:
        Tuple of (optimal clusters, optimal distance threshold)
    """
    best_threshold = 0.5
    best_score = 0
    best_clusters = None

    # Try different distance thresholds
    for threshold in np.arange(0.1, 0.9, 0.05):
        clusters = get_clusters(linkage_matrix, drgps, distance_threshold=threshold)

        # Score: number of clusters with good coverage, weighted by coverage
        reproducible = [c for c in clusters if c.run_coverage >= n_runs * min_coverage]
        if len(reproducible) > 0:
            score = sum(c.run_coverage for c in reproducible) / len(clusters)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_clusters = clusters

    if best_clusters is None:
        best_clusters = get_clusters(linkage_matrix, drgps, distance_threshold=0.5)

    return best_clusters, best_threshold


# =============================================================================
# Step 4: Categorize Programs by v Statistics
# =============================================================================

def categorize_program(
    v_stats: Dict[str, float],
    run_coverage: int,
    n_runs: int,
    min_coverage_fraction: float = 0.5,
    v_threshold: float = 0.1,
    consistency_threshold: float = 0.7
) -> ProgramCategory:
    """
    Categorize a consensus program based on reproducibility and v statistics.

    Args:
        v_stats: Dictionary with v-weight statistics (mean_v, std_v, etc.)
        run_coverage: Number of runs contributing to this cluster
        n_runs: Total number of runs
        min_coverage_fraction: Minimum coverage to be considered reproducible
        v_threshold: |mean_v| threshold to distinguish from nuisance
        consistency_threshold: Sign consistency threshold

    Returns:
        ProgramCategory enum value
    """
    # Check reproducibility first
    if run_coverage < n_runs * min_coverage_fraction:
        return ProgramCategory.NON_REPRODUCIBLE

    mean_v = v_stats['mean_v']
    abs_mean_v = v_stats['abs_mean_v']
    sign_consistency = v_stats['sign_consistency']

    # Check if effect is large enough and consistent
    is_consistent = sign_consistency >= consistency_threshold

    if abs_mean_v > v_threshold and is_consistent:
        if mean_v > 0:
            return ProgramCategory.DISEASE_RISK
        else:
            return ProgramCategory.PROTECTIVE
    else:
        # Either small effect or inconsistent sign = nuisance
        return ProgramCategory.NUISANCE


# =============================================================================
# Step 5: Assess Cluster Quality and Extract Consensus Programs
# =============================================================================

def assess_cluster_quality(
    cluster: DRGPCluster,
    similarity_matrix: np.ndarray,
    drgp_indices: dict,
    n_total_runs: int
) -> dict:
    """
    Compute quality metrics for a cluster.

    Args:
        cluster: DRGPCluster object
        similarity_matrix: Full similarity matrix
        drgp_indices: Dict mapping DRGP full_id to matrix index
        n_total_runs: Total number of runs

    Returns:
        Dictionary of quality metrics
    """
    # Get indices of cluster members
    indices = [drgp_indices[d.full_id] for d in cluster.members]

    # Compute within-cluster similarity (tightness)
    if len(indices) > 1:
        sub_sim = similarity_matrix[np.ix_(indices, indices)]
        triu_idx = np.triu_indices(len(indices), k=1)
        within_sims = sub_sim[triu_idx]
        tightness = np.mean(within_sims) if len(within_sims) > 0 else 1.0
        tightness_std = np.std(within_sims) if len(within_sims) > 0 else 0.0
    else:
        tightness = 1.0
        tightness_std = 0.0

    # Run coverage metrics
    run_coverage = cluster.run_coverage
    coverage_fraction = run_coverage / n_total_runs

    # Check for duplicate runs
    run_counts = {}
    for drgp in cluster.members:
        run_counts[drgp.run_id] = run_counts.get(drgp.run_id, 0) + 1
    max_duplicates = max(run_counts.values())

    # Get v-weight statistics
    v_stats = cluster.get_v_statistics()

    return {
        'cluster_id': cluster.cluster_id,
        'n_members': len(cluster.members),
        'run_coverage': run_coverage,
        'coverage_fraction': coverage_fraction,
        'has_duplicates': cluster.has_duplicate_runs,
        'max_duplicates': max_duplicates,
        'tightness': tightness,
        'tightness_std': tightness_std,
        **v_stats  # Include all v statistics
    }


def generate_cluster_report(
    clusters: list,
    similarity_matrix: np.ndarray,
    drgps: list,
    n_runs: int,
    min_coverage_fraction: float = 0.5,
    v_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Generate a summary report for all clusters with categorization.

    Args:
        clusters: List of DRGPCluster objects
        similarity_matrix: Full similarity matrix
        drgps: List of all DRGP objects
        n_runs: Total number of runs
        min_coverage_fraction: Minimum coverage for reproducibility
        v_threshold: Threshold for v categorization

    Returns:
        DataFrame with cluster quality metrics and categories
    """
    drgp_indices = {d.full_id: i for i, d in enumerate(drgps)}

    reports = []
    for cluster in clusters:
        quality = assess_cluster_quality(cluster, similarity_matrix, drgp_indices, n_runs)

        # Categorize
        category = categorize_program(
            v_stats=cluster.get_v_statistics(),
            run_coverage=cluster.run_coverage,
            n_runs=n_runs,
            min_coverage_fraction=min_coverage_fraction,
            v_threshold=v_threshold
        )
        quality['category'] = category.value

        reports.append(quality)

    df = pd.DataFrame(reports)
    df = df.sort_values('run_coverage', ascending=False)

    return df


def extract_consensus_programs(
    clusters: list,
    similarity_matrix: np.ndarray,
    drgps: list,
    n_runs: int,
    min_coverage_fraction: float = 0.5,
    threshold: float = 0.1,
    v_threshold: float = 0.1
) -> Tuple[List[ConsensusProgram], List[DRGPCluster]]:
    """
    Extract consensus programs from ALL clusters and categorize them.

    Args:
        clusters: List of DRGPCluster objects
        similarity_matrix: Full similarity matrix
        drgps: List of all DRGP objects
        n_runs: Total number of runs
        min_coverage_fraction: Minimum fraction of runs for reproducibility
        threshold: Beta threshold for binary gene sets
        v_threshold: Threshold for v categorization

    Returns:
        Tuple of (list of ConsensusProgram objects, list of non-reproducible clusters)
    """
    drgp_indices = {d.full_id: i for i, d in enumerate(drgps)}
    consensus_programs = []
    non_reproducible = []

    min_coverage = max(1, int(n_runs * min_coverage_fraction))

    for cluster in clusters:
        # Get quality metrics
        quality = assess_cluster_quality(cluster, similarity_matrix, drgp_indices, n_runs)
        v_stats = cluster.get_v_statistics()

        # Categorize
        category = categorize_program(
            v_stats=v_stats,
            run_coverage=cluster.run_coverage,
            n_runs=n_runs,
            min_coverage_fraction=min_coverage_fraction,
            v_threshold=v_threshold
        )

        if category == ProgramCategory.NON_REPRODUCIBLE:
            non_reproducible.append(cluster)
            continue

        # Compute consensus beta (average across members)
        consensus_beta = cluster.get_consensus_beta()
        gene_names = cluster.members[0].gene_names

        # Compute gene set union
        union_genes = set()
        for drgp in cluster.members:
            union_genes |= drgp.get_binary_genes(threshold)

        # Compute gene set intersection
        intersection_genes = cluster.members[0].get_binary_genes(threshold)
        for drgp in cluster.members[1:]:
            intersection_genes &= drgp.get_binary_genes(threshold)

        # Compute majority genes
        majority_genes = cluster.get_consensus_genes(threshold, min_freq=0.5)

        program = ConsensusProgram(
            cluster_id=cluster.cluster_id,
            n_contributing_runs=cluster.run_coverage,
            consensus_beta=consensus_beta,
            consensus_genes_union=union_genes,
            consensus_genes_intersection=intersection_genes,
            consensus_genes_majority=majority_genes,
            gene_names=gene_names,
            member_drgps=cluster.members,
            quality_metrics=quality,
            v_statistics=v_stats,
            category=category
        )
        consensus_programs.append(program)

    # Sort by effect size (|mean_v|) within each category
    consensus_programs.sort(key=lambda p: (p.category.value, -p.effect_size))

    return consensus_programs, non_reproducible


# =============================================================================
# Gene Set Enrichment (Placeholder)
# =============================================================================

def run_enrichment_analysis(
    consensus_programs: list,
    output_dir: Path,
    organism: str = 'human'
) -> dict:
    """
    Run gene set enrichment analysis on consensus programs.

    Args:
        consensus_programs: List of ConsensusProgram objects
        output_dir: Directory to save gene lists
        organism: Organism for enrichment

    Returns:
        Dictionary with gene list paths
    """
    enrichment_results = {}

    gene_lists_dir = output_dir / 'gene_lists'
    gene_lists_dir.mkdir(parents=True, exist_ok=True)

    for i, program in enumerate(consensus_programs):
        # Include category in directory name
        category_prefix = program.category.value
        program_dir = gene_lists_dir / f'{category_prefix}_program_{i+1}'
        program_dir.mkdir(exist_ok=True)

        # Save union genes
        union_file = program_dir / 'genes_union.txt'
        with open(union_file, 'w') as f:
            f.write('\n'.join(sorted(program.consensus_genes_union)))

        # Save majority genes
        majority_file = program_dir / 'genes_majority.txt'
        with open(majority_file, 'w') as f:
            f.write('\n'.join(sorted(program.consensus_genes_majority)))

        # Save intersection genes
        intersection_file = program_dir / 'genes_intersection.txt'
        with open(intersection_file, 'w') as f:
            f.write('\n'.join(sorted(program.consensus_genes_intersection)))

        # Save full gene ranking
        gene_df = program.get_gene_dataframe(threshold=0.0)
        ranking_file = program_dir / 'gene_ranking.csv'
        gene_df.to_csv(ranking_file, index=False)

        enrichment_results[f'{category_prefix}_program_{i+1}'] = {
            'category': category_prefix,
            'union_file': str(union_file),
            'majority_file': str(majority_file),
            'intersection_file': str(intersection_file),
            'ranking_file': str(ranking_file),
            'n_union_genes': len(program.consensus_genes_union),
            'n_majority_genes': len(program.consensus_genes_majority),
            'n_intersection_genes': len(program.consensus_genes_intersection),
            'effect_size': program.effect_size,
            'mean_v': program.mean_v
        }

    # Try to run enrichment if gseapy is available
    try:
        import gseapy as gp
        print("\nRunning gene set enrichment with gseapy...")

        for i, program in enumerate(consensus_programs):
            gene_list = list(program.consensus_genes_majority)
            if len(gene_list) < 5:
                print(f"  {program.category.value} Program {i+1}: Too few genes")
                continue

            try:
                category_prefix = program.category.value
                program_dir = gene_lists_dir / f'{category_prefix}_program_{i+1}'
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human',
                               'Reactome_2022', 'WikiPathway_2021_Human'],
                    organism=organism,
                    outdir=str(program_dir / 'enrichr'),
                    cutoff=0.05
                )
                enrichment_results[f'{category_prefix}_program_{i+1}']['enrichr'] = enr.results
                print(f"  {category_prefix} Program {i+1}: Enrichment complete")
            except Exception as e:
                print(f"  {category_prefix} Program {i+1}: Enrichment failed - {e}")

    except ImportError:
        print("\nNote: Install gseapy for automated enrichment analysis:")
        print("  pip install gseapy")

    return enrichment_results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_dendrogram(
    linkage_matrix: np.ndarray,
    drgps: list,
    output_path: Path,
    title: str = "DRGP Hierarchical Clustering"
):
    """Plot and save dendrogram."""
    fig, ax = plt.subplots(figsize=(20, 10))

    labels = [f"{d.run_id[-8:]}_{d.program_id}" for d in drgps]

    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=4,
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel('Distance (1 - Similarity)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    drgps: list,
    output_path: Path,
    title: str = "DRGP Similarity Matrix",
    cluster_labels: Optional[np.ndarray] = None
):
    """Plot and save similarity matrix heatmap."""
    fig, ax = plt.subplots(figsize=(16, 14))

    labels = [f"{d.run_id[-6:]}_{d.program_id}" for d in drgps]

    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        similarity_matrix = similarity_matrix[np.ix_(order, order)]
        labels = [labels[i] for i in order]

    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=3)
    plt.yticks(fontsize=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_cluster_summary_by_category(
    cluster_report: pd.DataFrame,
    output_path: Path
):
    """Plot cluster quality summary with category coloring."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Color mapping for categories
    category_colors = {
        'disease_risk': '#d62728',      # Red
        'protective': '#2ca02c',         # Green
        'nuisance': '#7f7f7f',           # Gray
        'non_reproducible': '#9467bd'    # Purple
    }

    colors = [category_colors.get(c, '#1f77b4') for c in cluster_report['category']]

    # 1. Run coverage by cluster
    ax = axes[0, 0]
    bars = ax.bar(range(len(cluster_report)), cluster_report['run_coverage'].values, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Run Coverage')
    ax.set_title('Run Coverage by Cluster')
    ax.axhline(y=cluster_report['run_coverage'].median(), color='k', linestyle='--', alpha=0.5)

    # 2. Mean v_weight by cluster (effect size)
    ax = axes[0, 1]
    ax.bar(range(len(cluster_report)), cluster_report['mean_v'].values,
           yerr=cluster_report['std_v'].values, capsize=2, color=colors)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean v Coefficient')
    ax.set_title('Disease Coefficient by Cluster\n(+ = risk, - = protective)')

    # 3. Tightness vs Coverage
    ax = axes[0, 2]
    scatter = ax.scatter(cluster_report['run_coverage'], cluster_report['tightness'],
                         c=colors, s=cluster_report['n_members'] * 20, alpha=0.6)
    ax.set_xlabel('Run Coverage')
    ax.set_ylabel('Tightness (Within-Cluster Similarity)')
    ax.set_title('Cluster Quality')

    # 4. Effect size (|mean_v|) distribution by category
    ax = axes[1, 0]
    categories = cluster_report['category'].unique()
    for i, cat in enumerate(categories):
        subset = cluster_report[cluster_report['category'] == cat]
        ax.bar(i, subset['abs_mean_v'].mean(),
               yerr=subset['abs_mean_v'].std(), capsize=5,
               color=category_colors.get(cat, '#1f77b4'), label=cat)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Mean |v| (Effect Size)')
    ax.set_title('Effect Size by Category')

    # 5. Category distribution
    ax = axes[1, 1]
    category_counts = cluster_report['category'].value_counts()
    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        colors=[category_colors.get(c, '#1f77b4') for c in category_counts.index],
        autopct='%1.0f%%',
        startangle=90
    )
    ax.set_title('Program Category Distribution')

    # 6. Sign consistency by category
    ax = axes[1, 2]
    for i, cat in enumerate(categories):
        subset = cluster_report[cluster_report['category'] == cat]
        ax.bar(i, subset['sign_consistency'].mean(),
               yerr=subset['sign_consistency'].std(), capsize=5,
               color=category_colors.get(cat, '#1f77b4'))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Sign Consistency')
    ax.set_title('v Coefficient Sign Consistency\n(1.0 = all same sign)')
    ax.axhline(y=0.7, color='k', linestyle='--', alpha=0.5, label='Threshold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_v_distribution(
    consensus_programs: list,
    output_path: Path
):
    """Plot distribution of v coefficients across all consensus programs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all v values by category
    category_colors = {
        ProgramCategory.DISEASE_RISK: '#d62728',
        ProgramCategory.PROTECTIVE: '#2ca02c',
        ProgramCategory.NUISANCE: '#7f7f7f'
    }

    # 1. Scatter plot: effect size vs run coverage
    ax = axes[0]
    for program in consensus_programs:
        ax.scatter(
            program.n_contributing_runs,
            program.mean_v,
            c=category_colors[program.category],
            s=100,
            alpha=0.7,
            label=program.category.value
        )
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.1, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Run Coverage')
    ax.set_ylabel('Mean v Coefficient')
    ax.set_title('Consensus Programs: Effect vs Reproducibility')

    # Create legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # 2. Box plot of v values by category
    ax = axes[1]
    data_by_category = {cat.value: [] for cat in ProgramCategory if cat != ProgramCategory.NON_REPRODUCIBLE}

    for program in consensus_programs:
        v_values = [d.v_weight for d in program.member_drgps]
        data_by_category[program.category.value].extend(v_values)

    # Filter out empty categories
    data_by_category = {k: v for k, v in data_by_category.items() if len(v) > 0}

    if data_by_category:
        bp = ax.boxplot(
            [data_by_category[k] for k in data_by_category.keys()],
            labels=list(data_by_category.keys()),
            patch_artist=True
        )

        colors = [category_colors.get(ProgramCategory(k), '#1f77b4') for k in data_by_category.keys()]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('v Coefficient')
    ax.set_title('v Coefficient Distribution by Category')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_consensus_program_genes(
    consensus_programs: list,
    output_dir: Path,
    top_n: int = 30
):
    """Plot top genes for each consensus program."""
    category_colors = {
        ProgramCategory.DISEASE_RISK: 'darkred',
        ProgramCategory.PROTECTIVE: 'darkgreen',
        ProgramCategory.NUISANCE: 'gray'
    }

    for i, program in enumerate(consensus_programs):
        fig, ax = plt.subplots(figsize=(10, 8))

        top_genes = program.get_top_genes(top_n)
        genes = [g[0] for g in top_genes]
        values = [g[1] for g in top_genes]

        # Color based on category and majority membership
        base_color = category_colors[program.category]
        colors = [base_color if g in program.consensus_genes_majority else 'lightgray'
                  for g in genes]

        y_pos = np.arange(len(genes))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Consensus Beta')

        category_label = program.category.value.replace('_', ' ').title()
        ax.set_title(
            f'{category_label} Program {i+1} - Top {top_n} Genes\n'
            f'(Runs: {program.n_contributing_runs}, '
            f'Effect: {program.effect_size:.3f}, '
            f'Mean v: {program.mean_v:.3f})'
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / f'{program.category.value}_program_{i+1}_top_genes.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close()


def plot_run_contribution_matrix(
    clusters: list,
    n_runs: int,
    run_ids: list,
    output_path: Path,
    cluster_categories: Optional[Dict[int, str]] = None
):
    """Plot which runs contribute to which clusters."""
    matrix = np.zeros((len(clusters), n_runs))

    for i, cluster in enumerate(clusters):
        for drgp in cluster.members:
            run_idx = run_ids.index(drgp.run_id)
            matrix[i, run_idx] += 1

    fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.4), max(8, len(clusters) * 0.3)))

    # Create labels with category if available
    if cluster_categories:
        ylabels = [f'C{c.cluster_id} ({cluster_categories.get(c.cluster_id, "?")[:4]})'
                   for c in clusters]
    else:
        ylabels = [f'Cluster {c.cluster_id}' for c in clusters]

    sns.heatmap(
        matrix,
        xticklabels=[r[-8:] for r in run_ids],
        yticklabels=ylabels,
        cmap='Blues',
        annot=True if len(clusters) < 50 else False,
        fmt='.0f',
        ax=ax
    )

    ax.set_xlabel('Run')
    ax.set_ylabel('Cluster')
    ax.set_title('Run Contribution to Clusters')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_category_summary(
    results: ConsensusAnalysisResults,
    output_path: Path
):
    """Create a summary figure showing all three program categories."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    category_colors = {
        'Disease Risk': '#d62728',
        'Protective': '#2ca02c',
        'Nuisance': '#7f7f7f'
    }

    # Count programs by category
    counts = {
        'Disease Risk': len(results.disease_risk_programs),
        'Protective': len(results.protective_programs),
        'Nuisance': len(results.nuisance_programs)
    }

    # 1. Category counts (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(counts.keys(), counts.values(),
                   color=[category_colors[k] for k in counts.keys()])
    ax1.set_ylabel('Number of Programs')
    ax1.set_title('Reproducible Programs by Category')
    for bar, count in zip(bars, counts.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(count), ha='center', va='bottom', fontsize=12)

    # 2. Effect size by category (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    effect_sizes = {
        'Disease Risk': [p.effect_size for p in results.disease_risk_programs],
        'Protective': [p.effect_size for p in results.protective_programs],
        'Nuisance': [p.effect_size for p in results.nuisance_programs]
    }
    effect_sizes = {k: v for k, v in effect_sizes.items() if v}

    if effect_sizes:
        bp = ax2.boxplot(list(effect_sizes.values()), labels=list(effect_sizes.keys()),
                         patch_artist=True)
        for patch, cat in zip(bp['boxes'], effect_sizes.keys()):
            patch.set_facecolor(category_colors[cat])
            patch.set_alpha(0.5)
    ax2.set_ylabel('|Mean v| (Effect Size)')
    ax2.set_title('Effect Size Distribution')

    # 3. Run coverage by category (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    coverages = {
        'Disease Risk': [p.n_contributing_runs for p in results.disease_risk_programs],
        'Protective': [p.n_contributing_runs for p in results.protective_programs],
        'Nuisance': [p.n_contributing_runs for p in results.nuisance_programs]
    }
    coverages = {k: v for k, v in coverages.items() if v}

    if coverages:
        bp = ax3.boxplot(list(coverages.values()), labels=list(coverages.keys()),
                         patch_artist=True)
        for patch, cat in zip(bp['boxes'], coverages.keys()):
            patch.set_facecolor(category_colors[cat])
            patch.set_alpha(0.5)
    ax3.set_ylabel('Number of Runs')
    ax3.set_title('Run Coverage Distribution')

    # 4-6: Top genes for each category (middle row)
    for idx, (cat_name, programs, color) in enumerate([
        ('Disease Risk', results.disease_risk_programs, '#d62728'),
        ('Protective', results.protective_programs, '#2ca02c'),
        ('Nuisance', results.nuisance_programs, '#7f7f7f')
    ]):
        ax = fig.add_subplot(gs[1, idx])
        if programs:
            # Get top genes from the best program
            best_prog = programs[0]
            top_genes = best_prog.get_top_genes(10)
            genes = [g[0] for g in top_genes]
            values = [g[1] for g in top_genes]

            ax.barh(range(len(genes)), values, color=color)
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Beta')
            ax.set_title(f'{cat_name} Program 1\n(n={best_prog.n_contributing_runs} runs, v={best_prog.mean_v:.3f})')
        else:
            ax.text(0.5, 0.5, 'No programs', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cat_name}: None')

    # 7: Summary statistics table (bottom row, spanning all columns)
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Create summary table
    table_data = []
    headers = ['Category', 'Programs', 'Mean Effect', 'Mean Coverage', 'Mean Genes (Majority)']

    for cat_name, programs in [
        ('Disease Risk', results.disease_risk_programs),
        ('Protective', results.protective_programs),
        ('Nuisance', results.nuisance_programs),
        ('Non-Reproducible', results.non_reproducible_clusters)
    ]:
        if programs:
            if isinstance(programs[0], DRGPCluster):
                # Non-reproducible clusters
                table_data.append([
                    cat_name,
                    len(programs),
                    'N/A',
                    f"{np.mean([c.run_coverage for c in programs]):.1f}",
                    'N/A'
                ])
            else:
                table_data.append([
                    cat_name,
                    len(programs),
                    f"{np.mean([p.effect_size for p in programs]):.3f}",
                    f"{np.mean([p.n_contributing_runs for p in programs]):.1f}",
                    f"{np.mean([len(p.consensus_genes_majority) for p in programs]):.0f}"
                ])
        else:
            table_data.append([cat_name, 0, 'N/A', 'N/A', 'N/A'])

    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the category cells
    for i, (cat_name, _) in enumerate([
        ('Disease Risk', results.disease_risk_programs),
        ('Protective', results.protective_programs),
        ('Nuisance', results.nuisance_programs),
        ('Non-Reproducible', results.non_reproducible_clusters)
    ]):
        color_map = {
            'Disease Risk': '#ffcccc',
            'Protective': '#ccffcc',
            'Nuisance': '#e0e0e0',
            'Non-Reproducible': '#e6ccff'
        }
        table[(i+1, 0)].set_facecolor(color_map.get(cat_name, 'white'))

    plt.suptitle('Comprehensive DRGP Analysis Summary', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_comprehensive_analysis(
    runs_folder: Path,
    output_dir: Path,
    prefix: str = 'svi',
    thresholds: list = [0.05, 0.1, 0.5],
    min_coverage_fraction: float = 0.5,
    v_threshold: float = 0.1,
    similarity_method: str = 'pearson',
    linkage_method: str = 'average'
) -> ConsensusAnalysisResults:
    """
    Run the comprehensive DRGP analysis pipeline on ALL programs.

    Args:
        runs_folder: Path to folder containing run subdirectories
        output_dir: Path to save results
        prefix: File prefix for SVI outputs
        thresholds: Beta thresholds for binary representation
        min_coverage_fraction: Minimum fraction of runs for reproducibility
        v_threshold: |mean_v| threshold for categorization
        similarity_method: 'pearson' or 'cosine'
        linkage_method: Linkage method for hierarchical clustering

    Returns:
        ConsensusAnalysisResults object with all results
    """
    results = ConsensusAnalysisResults()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE DRGP ANALYSIS (All Programs)")
    print("=" * 70)

    # Step 1: Load ALL DRGPs (no filtering)
    print("\n[Step 1] Loading ALL DRGPs from all runs...")
    results.all_drgps = load_all_drgps(runs_folder, prefix)

    if len(results.all_drgps) == 0:
        raise ValueError("No DRGPs found in any run!")

    run_ids = sorted(set(d.run_id for d in results.all_drgps))
    n_runs = len(run_ids)
    n_programs_per_run = len(results.all_drgps) // n_runs

    print(f"Total DRGPs loaded: {len(results.all_drgps)}")
    print(f"  - {n_runs} runs")
    print(f"  - {n_programs_per_run} programs per run")

    # Report v-weight distribution
    v_weights = [d.v_weight for d in results.all_drgps]
    print(f"\nv-weight distribution across all DRGPs:")
    print(f"  Mean: {np.mean(v_weights):.4f}")
    print(f"  Std:  {np.std(v_weights):.4f}")
    print(f"  Min:  {np.min(v_weights):.4f}")
    print(f"  Max:  {np.max(v_weights):.4f}")
    print(f"  Positive: {sum(1 for v in v_weights if v > 0)}")
    print(f"  Negative: {sum(1 for v in v_weights if v < 0)}")
    print(f"  Near-zero (|v| < {v_threshold}): {sum(1 for v in v_weights if abs(v) < v_threshold)}")

    # Step 2: Build similarity matrices (using beta only)
    print("\n[Step 2] Computing similarity matrices (beta only, ignoring v)...")
    print(f"  Continuous similarity ({similarity_method})...")
    results.similarity_matrix_continuous = compute_continuous_similarity(
        results.all_drgps, method=similarity_method
    )

    binary_results = {}
    for threshold in thresholds:
        print(f"  Binary similarity (Jaccard, threshold={threshold})...")
        sim_binary = compute_binary_similarity(results.all_drgps, threshold=threshold)
        binary_results[threshold] = sim_binary

    results.similarity_matrix_binary = binary_results[0.1]

    # Step 3: Hierarchical clustering (by beta similarity)
    print("\n[Step 3] Performing hierarchical clustering (by beta similarity)...")
    results.linkage_matrix = perform_hierarchical_clustering(
        results.similarity_matrix_continuous, method=linkage_method
    )

    min_coverage = max(1, int(n_runs * min_coverage_fraction))
    print(f"  Minimum coverage for reproducibility: {min_coverage} runs ({min_coverage_fraction*100:.0f}%)")

    results.clusters, optimal_threshold = find_optimal_clusters(
        results.linkage_matrix,
        results.all_drgps,
        n_runs,
        min_coverage=min_coverage_fraction
    )
    print(f"  Found {len(results.clusters)} clusters at distance threshold {optimal_threshold:.2f}")

    # Step 4: Generate cluster report with categorization
    print("\n[Step 4] Assessing cluster quality and categorizing...")
    cluster_report = generate_cluster_report(
        results.clusters,
        results.similarity_matrix_continuous,
        results.all_drgps,
        n_runs,
        min_coverage_fraction=min_coverage_fraction,
        v_threshold=v_threshold
    )

    # Print category distribution
    category_counts = cluster_report['category'].value_counts()
    print("\nCluster categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} clusters")

    # Save cluster report
    cluster_report.to_csv(output_dir / 'cluster_quality_report.csv', index=False)
    print(f"\nFull cluster report saved to: {output_dir / 'cluster_quality_report.csv'}")

    # Step 5: Extract consensus programs
    print(f"\n[Step 5] Extracting consensus programs (min coverage: {min_coverage})...")
    results.consensus_programs, results.non_reproducible_clusters = extract_consensus_programs(
        results.clusters,
        results.similarity_matrix_continuous,
        results.all_drgps,
        n_runs,
        min_coverage_fraction=min_coverage_fraction,
        threshold=0.1,
        v_threshold=v_threshold
    )

    # Separate by category
    results.disease_risk_programs = [p for p in results.consensus_programs
                                      if p.category == ProgramCategory.DISEASE_RISK]
    results.protective_programs = [p for p in results.consensus_programs
                                    if p.category == ProgramCategory.PROTECTIVE]
    results.nuisance_programs = [p for p in results.consensus_programs
                                  if p.category == ProgramCategory.NUISANCE]

    print(f"\nConsensus Program Summary:")
    print(f"  Disease Risk Programs: {len(results.disease_risk_programs)}")
    print(f"  Protective Programs:   {len(results.protective_programs)}")
    print(f"  Nuisance Programs:     {len(results.nuisance_programs)}")
    print(f"  Non-Reproducible:      {len(results.non_reproducible_clusters)} clusters")

    # Print details for each category
    for cat_name, programs in [
        ("DISEASE RISK", results.disease_risk_programs),
        ("PROTECTIVE", results.protective_programs),
        ("NUISANCE", results.nuisance_programs)
    ]:
        if programs:
            print(f"\n  --- {cat_name} PROGRAMS ---")
            for i, prog in enumerate(programs[:5]):  # Show top 5
                print(f"  Program {i+1}:")
                print(f"    Runs: {prog.n_contributing_runs}/{n_runs}")
                print(f"    Effect size (|v|): {prog.effect_size:.4f}")
                print(f"    Mean v: {prog.mean_v:.4f}")
                print(f"    Genes (majority): {len(prog.consensus_genes_majority)}")
                print(f"    Top 5 genes: {[g[0] for g in prog.get_top_genes(5)]}")

    # Step 6: Gene set enrichment
    print("\n[Step 6] Preparing gene lists for enrichment analysis...")
    results.enrichment_results = run_enrichment_analysis(
        results.consensus_programs,
        output_dir
    )

    # Step 7: Generate visualizations
    print("\n[Step 7] Generating visualizations...")

    # Dendrogram
    plot_dendrogram(
        results.linkage_matrix,
        results.all_drgps,
        output_dir / 'dendrogram.png',
        title=f'All DRGPs Hierarchical Clustering ({len(results.all_drgps)} programs)'
    )

    # Similarity matrix
    plot_similarity_matrix(
        results.similarity_matrix_continuous,
        results.all_drgps,
        output_dir / 'similarity_matrix_continuous.png',
        title=f'DRGP Beta Similarity Matrix ({similarity_method.capitalize()})'
    )

    # Clustered similarity matrix
    cluster_labels = np.zeros(len(results.all_drgps))
    for cluster in results.clusters:
        for drgp in cluster.members:
            idx = next(i for i, d in enumerate(results.all_drgps)
                       if d.full_id == drgp.full_id)
            cluster_labels[idx] = cluster.cluster_id

    plot_similarity_matrix(
        results.similarity_matrix_continuous,
        results.all_drgps,
        output_dir / 'similarity_matrix_clustered.png',
        title='DRGP Similarity Matrix (Clustered)',
        cluster_labels=cluster_labels
    )

    # Cluster summary with categories
    plot_cluster_summary_by_category(cluster_report, output_dir / 'cluster_summary.png')

    # V distribution plot
    if results.consensus_programs:
        plot_v_distribution(results.consensus_programs, output_dir / 'v_distribution.png')

    # Category summary
    plot_category_summary(results, output_dir / 'category_summary.png')

    # Run contribution matrix
    cluster_categories = {c.cluster_id: cluster_report.loc[cluster_report['cluster_id'] == c.cluster_id, 'category'].values[0]
                          for c in results.clusters if c.cluster_id in cluster_report['cluster_id'].values}
    plot_run_contribution_matrix(
        results.clusters,
        n_runs,
        run_ids,
        output_dir / 'run_contribution_matrix.png',
        cluster_categories=cluster_categories
    )

    # Consensus program gene plots
    if results.consensus_programs:
        plot_consensus_program_genes(
            results.consensus_programs,
            output_dir
        )

    # Step 8: Save comprehensive results
    print("\n[Step 8] Writing results to disk...")

    # Save consensus programs as CSV
    for i, prog in enumerate(results.consensus_programs):
        gene_df = prog.get_gene_dataframe(threshold=0.0)
        gene_df.to_csv(output_dir / f'{prog.category.value}_program_{i+1}_genes.csv', index=False)

    # Save summary JSON
    summary = {
        'n_runs': n_runs,
        'run_ids': run_ids,
        'n_total_drgps': len(results.all_drgps),
        'n_programs_per_run': n_programs_per_run,
        'n_clusters': len(results.clusters),
        'optimal_distance_threshold': optimal_threshold,
        'similarity_method': similarity_method,
        'linkage_method': linkage_method,
        'min_coverage_fraction': min_coverage_fraction,
        'v_threshold': v_threshold,
        'thresholds_tested': thresholds,
        'category_counts': {
            'disease_risk': len(results.disease_risk_programs),
            'protective': len(results.protective_programs),
            'nuisance': len(results.nuisance_programs),
            'non_reproducible': len(results.non_reproducible_clusters)
        },
        'consensus_programs': [
            {
                'category': prog.category.value,
                'cluster_id': prog.cluster_id,
                'n_contributing_runs': prog.n_contributing_runs,
                'effect_size': prog.effect_size,
                'mean_v': prog.mean_v,
                'std_v': prog.v_statistics['std_v'],
                'sign_consistency': prog.v_statistics['sign_consistency'],
                'n_genes_union': len(prog.consensus_genes_union),
                'n_genes_majority': len(prog.consensus_genes_majority),
                'n_genes_intersection': len(prog.consensus_genes_intersection),
                'top_10_genes': [g[0] for g in prog.get_top_genes(10)],
                'quality_metrics': prog.quality_metrics
            }
            for prog in results.consensus_programs
        ]
    }

    with gzip.open(output_dir / 'analysis_summary.json.gz', 'wt') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save similarity matrices
    np.savez_compressed(
        output_dir / 'similarity_matrices.npz',
        continuous=results.similarity_matrix_continuous,
        binary_0p1=binary_results[0.1],
        linkage=results.linkage_matrix,
        drgp_ids=[d.full_id for d in results.all_drgps]
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  - Analyzed {len(results.all_drgps)} DRGPs from {n_runs} runs")
    print(f"  - Found {len(results.clusters)} clusters")
    print(f"\n  Consensus Programs by Category:")
    print(f"    Disease Risk: {len(results.disease_risk_programs)}")
    print(f"    Protective:   {len(results.protective_programs)}")
    print(f"    Nuisance:     {len(results.nuisance_programs)}")
    print(f"    Discarded:    {len(results.non_reproducible_clusters)} (non-reproducible)")

    if results.disease_risk_programs:
        print(f"\n  Top Disease Risk Program:")
        top = results.disease_risk_programs[0]
        print(f"    - Appears in {top.n_contributing_runs}/{n_runs} runs")
        print(f"    - Effect size: {top.effect_size:.4f}")
        print(f"    - Top genes: {', '.join([g[0] for g in top.get_top_genes(5)])}")

    if results.protective_programs:
        print(f"\n  Top Protective Program:")
        top = results.protective_programs[0]
        print(f"    - Appears in {top.n_contributing_runs}/{n_runs} runs")
        print(f"    - Effect size: {top.effect_size:.4f} (mean_v: {top.mean_v:.4f})")
        print(f"    - Top genes: {', '.join([g[0] for g in top.get_top_genes(5)])}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive DRGP analysis across multiple SVI runs (all programs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python analyze_all_drgps.py /path/to/runs_folder --output results/all_drgps

    # With custom parameters
    python analyze_all_drgps.py /path/to/runs_folder \\
        --output results/all_drgps \\
        --min-coverage 0.5 \\
        --v-threshold 0.1 \\
        --similarity cosine

Program Categories:
    - disease_risk:     Reproducible + consistently positive v coefficient
    - protective:       Reproducible + consistently negative v coefficient
    - nuisance:         Reproducible + near-zero or inconsistent v
    - non_reproducible: Not enough run coverage (discarded)
        """
    )

    parser.add_argument(
        'runs_folder',
        type=Path,
        help='Path to folder containing SVI run subdirectories'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./all_drgps_results'),
        help='Output directory for results (default: ./all_drgps_results)'
    )

    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default='svi',
        help='File prefix for SVI outputs (default: svi)'
    )

    parser.add_argument(
        '--min-coverage', '-m',
        type=float,
        default=0.5,
        help='Minimum fraction of runs for reproducibility (default: 0.5)'
    )

    parser.add_argument(
        '--v-threshold', '-v',
        type=float,
        default=0.1,
        help='|mean_v| threshold to distinguish from nuisance (default: 0.1)'
    )

    parser.add_argument(
        '--similarity', '-s',
        type=str,
        choices=['pearson', 'cosine'],
        default='pearson',
        help='Similarity method for continuous vectors (default: pearson)'
    )

    parser.add_argument(
        '--linkage', '-l',
        type=str,
        choices=['average', 'complete', 'single', 'ward'],
        default='average',
        help='Linkage method for hierarchical clustering (default: average)'
    )

    parser.add_argument(
        '--thresholds', '-t',
        type=float,
        nargs='+',
        default=[0.05, 0.1, 0.5],
        help='Beta thresholds for binary gene set representation (default: 0.05 0.1 0.5)'
    )

    args = parser.parse_args()

    results = run_comprehensive_analysis(
        runs_folder=args.runs_folder,
        output_dir=args.output,
        prefix=args.prefix,
        thresholds=args.thresholds,
        min_coverage_fraction=args.min_coverage,
        v_threshold=args.v_threshold,
        similarity_method=args.similarity,
        linkage_method=args.linkage
    )

    return results


if __name__ == '__main__':
    main()
