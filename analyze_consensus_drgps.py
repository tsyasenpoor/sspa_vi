#!/usr/bin/env python3
"""
Consensus DRGP Analysis across Multiple SVI Runs

This script analyzes gene programs from multiple SVI runs to identify
reproducible, consensus Disease-Related Gene Programs (DRGPs).

Pipeline:
1. Load and filter positive-coefficient DRGPs from all runs
2. Create continuous (beta vectors) and binary (thresholded) representations
3. Build similarity matrices (Pearson/cosine for continuous, Jaccard for binary)
4. Hierarchical clustering to find reproducible programs
5. Assess cluster quality (run coverage, uniqueness, tightness)
6. Extract consensus gene programs
7. Gene set enrichment analysis (GO, Reactome)

Usage:
    python analyze_consensus_drgps.py /path/to/runs_folder --output results/consensus

Author: Generated for SVI analysis pipeline
"""

import argparse
import gzip
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


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


@dataclass
class ConsensusAnalysisResults:
    """Container for all analysis results."""
    all_drgps: list = field(default_factory=list)
    positive_drgps: list = field(default_factory=list)
    similarity_matrix_continuous: np.ndarray = None
    similarity_matrix_binary: np.ndarray = None
    linkage_matrix: np.ndarray = None
    clusters: list = field(default_factory=list)
    consensus_programs: list = field(default_factory=list)
    enrichment_results: dict = field(default_factory=dict)


# =============================================================================
# Step 1 & 2: Load and Filter DRGPs
# =============================================================================

def load_drgps_from_run(run_dir: Path, prefix: str = 'svi') -> list:
    """
    Load all DRGPs from a single run directory.

    Args:
        run_dir: Path to the run results directory
        prefix: File prefix (default 'svi')

    Returns:
        List of DRGP objects
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
        # Convention: class 1 is typically the disease/case class
        if 'v_weight_class1' in v_weight_cols:
            v_weight = row['v_weight_class1']
        elif len(v_weight_cols) > 0:
            # Use the last class if class1 not found
            v_weight = row[v_weight_cols[-1]]
        else:
            # If no v_weight, use mean of positive betas as proxy
            v_weight = np.mean(beta_vector[beta_vector > 0]) if np.any(beta_vector > 0) else 0

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
    Load DRGPs from all run directories.

    Args:
        runs_folder: Path to folder containing all run subdirectories
        prefix: File prefix

    Returns:
        List of all DRGP objects
    """
    all_drgps = []

    # Find all run directories
    run_dirs = sorted([d for d in runs_folder.iterdir() if d.is_dir()])

    if len(run_dirs) == 0:
        # Maybe the runs_folder itself contains the files directly
        # Check if gene program file exists in the folder
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


def filter_positive_drgps(drgps: list) -> list:
    """
    Filter to only positive-coefficient DRGPs.

    Positive coefficient indicates the program is positively associated
    with disease (case class).

    Args:
        drgps: List of all DRGPs

    Returns:
        List of positive-coefficient DRGPs only
    """
    positive = [d for d in drgps if d.v_weight > 0]
    print(f"Filtered to {len(positive)} positive-coefficient DRGPs "
          f"(from {len(drgps)} total)")
    return positive


# =============================================================================
# Step 3: Build Similarity Matrices
# =============================================================================

def compute_continuous_similarity(drgps: list, method: str = 'pearson') -> np.ndarray:
    """
    Compute pairwise similarity using continuous beta vectors.

    Args:
        drgps: List of DRGP objects
        method: 'pearson' or 'cosine'

    Returns:
        N x N similarity matrix
    """
    n = len(drgps)
    beta_matrix = np.array([d.beta_vector for d in drgps])

    if method == 'pearson':
        # Compute Pearson correlation
        sim_matrix = np.corrcoef(beta_matrix)
    elif method == 'cosine':
        sim_matrix = cosine_similarity(beta_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Handle NaN values (can occur with constant vectors)
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
# Step 4: Clustering
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
        # Default: use distance threshold of 0.5 (similarity > 0.5)
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
) -> tuple:
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
        # Fall back to default
        best_clusters = get_clusters(linkage_matrix, drgps, distance_threshold=0.5)

    return best_clusters, best_threshold


# =============================================================================
# Step 5: Cluster Quality Assessment
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
        # Get upper triangle (excluding diagonal)
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

    return {
        'cluster_id': cluster.cluster_id,
        'n_members': len(cluster.members),
        'run_coverage': run_coverage,
        'coverage_fraction': coverage_fraction,
        'has_duplicates': cluster.has_duplicate_runs,
        'max_duplicates': max_duplicates,
        'tightness': tightness,
        'tightness_std': tightness_std,
        'mean_v_weight': np.mean([d.v_weight for d in cluster.members]),
        'std_v_weight': np.std([d.v_weight for d in cluster.members])
    }


def generate_cluster_report(
    clusters: list,
    similarity_matrix: np.ndarray,
    drgps: list,
    n_runs: int
) -> pd.DataFrame:
    """
    Generate a summary report for all clusters.

    Args:
        clusters: List of DRGPCluster objects
        similarity_matrix: Full similarity matrix
        drgps: List of all DRGP objects
        n_runs: Total number of runs

    Returns:
        DataFrame with cluster quality metrics
    """
    # Build index mapping
    drgp_indices = {d.full_id: i for i, d in enumerate(drgps)}

    reports = []
    for cluster in clusters:
        quality = assess_cluster_quality(cluster, similarity_matrix, drgp_indices, n_runs)
        reports.append(quality)

    df = pd.DataFrame(reports)
    df = df.sort_values('run_coverage', ascending=False)

    return df


# =============================================================================
# Step 6: Extract Consensus Programs
# =============================================================================

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


def extract_consensus_programs(
    clusters: list,
    similarity_matrix: np.ndarray,
    drgps: list,
    n_runs: int,
    min_coverage: int = 10,
    threshold: float = 0.1
) -> list:
    """
    Extract consensus programs from reproducible clusters.

    Args:
        clusters: List of DRGPCluster objects
        similarity_matrix: Full similarity matrix
        drgps: List of all DRGP objects
        n_runs: Total number of runs
        min_coverage: Minimum run coverage to be considered reproducible
        threshold: Beta threshold for binary gene sets

    Returns:
        List of ConsensusProgram objects
    """
    drgp_indices = {d.full_id: i for i, d in enumerate(drgps)}
    consensus_programs = []

    for cluster in clusters:
        if cluster.run_coverage < min_coverage:
            continue

        # Get quality metrics
        quality = assess_cluster_quality(cluster, similarity_matrix, drgp_indices, n_runs)

        # Compute consensus beta (average across members)
        consensus_beta = cluster.get_consensus_beta()

        # Get gene names from first member
        gene_names = cluster.members[0].gene_names

        # Compute gene set union
        union_genes = set()
        for drgp in cluster.members:
            union_genes |= drgp.get_binary_genes(threshold)

        # Compute gene set intersection
        intersection_genes = cluster.members[0].get_binary_genes(threshold)
        for drgp in cluster.members[1:]:
            intersection_genes &= drgp.get_binary_genes(threshold)

        # Compute majority genes (in >50% of members)
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
            quality_metrics=quality
        )
        consensus_programs.append(program)

    # Sort by run coverage
    consensus_programs.sort(key=lambda p: p.n_contributing_runs, reverse=True)

    return consensus_programs


# =============================================================================
# Step 7: Gene Set Enrichment (Placeholder for external tools)
# =============================================================================

def run_enrichment_analysis(
    consensus_programs: list,
    output_dir: Path,
    organism: str = 'human'
) -> dict:
    """
    Run gene set enrichment analysis on consensus programs.

    This function prepares gene lists for enrichment analysis.
    For full enrichment, use external tools like:
    - gseapy (Python)
    - enrichR API
    - gprofiler2

    Args:
        consensus_programs: List of ConsensusProgram objects
        output_dir: Directory to save gene lists
        organism: Organism for enrichment ('human' or 'mouse')

    Returns:
        Dictionary with gene list paths
    """
    enrichment_results = {}

    gene_lists_dir = output_dir / 'gene_lists'
    gene_lists_dir.mkdir(parents=True, exist_ok=True)

    for i, program in enumerate(consensus_programs):
        program_dir = gene_lists_dir / f'consensus_program_{i+1}'
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

        enrichment_results[f'program_{i+1}'] = {
            'union_file': str(union_file),
            'majority_file': str(majority_file),
            'intersection_file': str(intersection_file),
            'ranking_file': str(ranking_file),
            'n_union_genes': len(program.consensus_genes_union),
            'n_majority_genes': len(program.consensus_genes_majority),
            'n_intersection_genes': len(program.consensus_genes_intersection)
        }

    # Try to run enrichment if gseapy is available
    try:
        import gseapy as gp
        print("\nRunning gene set enrichment with gseapy...")

        for i, program in enumerate(consensus_programs):
            gene_list = list(program.consensus_genes_majority)
            if len(gene_list) < 5:
                print(f"  Program {i+1}: Too few genes for enrichment")
                continue

            try:
                # Run Enrichr
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human',
                               'Reactome_2022', 'WikiPathway_2021_Human'],
                    organism=organism,
                    outdir=str(gene_lists_dir / f'consensus_program_{i+1}' / 'enrichr'),
                    cutoff=0.05
                )
                enrichment_results[f'program_{i+1}']['enrichr'] = enr.results
                print(f"  Program {i+1}: Enrichment complete")
            except Exception as e:
                print(f"  Program {i+1}: Enrichment failed - {e}")

    except ImportError:
        print("\nNote: Install gseapy for automated enrichment analysis:")
        print("  pip install gseapy")
        print("Gene lists have been saved for manual enrichment analysis.")

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

    # Create labels
    labels = [f"{d.run_id[-8:]}_{d.program_id}" for d in drgps]

    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6,
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
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create labels
    labels = [f"{d.run_id[-6:]}_{d.program_id}" for d in drgps]

    # If cluster labels provided, sort by cluster
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
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cluster_summary(
    cluster_report: pd.DataFrame,
    output_path: Path
):
    """Plot cluster quality summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Run coverage histogram
    ax = axes[0, 0]
    ax.bar(range(len(cluster_report)), cluster_report['run_coverage'].values)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Run Coverage')
    ax.set_title('Run Coverage by Cluster')
    ax.axhline(y=cluster_report['run_coverage'].median(), color='r',
               linestyle='--', label='Median')
    ax.legend()

    # Tightness vs Coverage
    ax = axes[0, 1]
    ax.scatter(cluster_report['run_coverage'], cluster_report['tightness'],
               s=cluster_report['n_members'] * 20, alpha=0.6)
    ax.set_xlabel('Run Coverage')
    ax.set_ylabel('Tightness (Mean Within-Cluster Similarity)')
    ax.set_title('Cluster Quality: Tightness vs Coverage')

    # Cluster size distribution
    ax = axes[1, 0]
    ax.bar(range(len(cluster_report)), cluster_report['n_members'].values)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Members')
    ax.set_title('Cluster Size Distribution')

    # Mean v_weight by cluster
    ax = axes[1, 1]
    ax.bar(range(len(cluster_report)), cluster_report['mean_v_weight'].values,
           yerr=cluster_report['std_v_weight'].values, capsize=3)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Mean Classification Weight')
    ax.set_title('Effect Size by Cluster')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_consensus_program_genes(
    consensus_programs: list,
    output_dir: Path,
    top_n: int = 30
):
    """Plot top genes for each consensus program."""
    for i, program in enumerate(consensus_programs):
        fig, ax = plt.subplots(figsize=(10, 8))

        top_genes = program.get_top_genes(top_n)
        genes = [g[0] for g in top_genes]
        values = [g[1] for g in top_genes]

        # Color by whether gene is in majority set
        colors = ['darkblue' if g in program.consensus_genes_majority else 'lightblue'
                  for g in genes]

        y_pos = np.arange(len(genes))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Consensus Beta')
        ax.set_title(f'Consensus Program {i+1} - Top {top_n} Genes\n'
                     f'(Contributing Runs: {program.n_contributing_runs}, '
                     f'Dark = in >50% of members)')

        plt.tight_layout()
        plt.savefig(output_dir / f'consensus_program_{i+1}_top_genes.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_run_contribution_matrix(
    clusters: list,
    n_runs: int,
    run_ids: list,
    output_path: Path
):
    """Plot which runs contribute to which clusters."""
    # Create binary matrix: clusters x runs
    matrix = np.zeros((len(clusters), n_runs))

    for i, cluster in enumerate(clusters):
        for drgp in cluster.members:
            run_idx = run_ids.index(drgp.run_id)
            matrix[i, run_idx] += 1

    fig, ax = plt.subplots(figsize=(max(12, n_runs * 0.4), max(8, len(clusters) * 0.3)))

    sns.heatmap(
        matrix,
        xticklabels=[r[-8:] for r in run_ids],
        yticklabels=[f'Cluster {c.cluster_id}' for c in clusters],
        cmap='Blues',
        annot=True,
        fmt='.0f',
        ax=ax
    )

    ax.set_xlabel('Run')
    ax.set_ylabel('Cluster')
    ax.set_title('Run Contribution to Clusters')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_consensus_analysis(
    runs_folder: Path,
    output_dir: Path,
    prefix: str = 'svi',
    thresholds: list = [0.05, 0.1, 0.5],
    min_coverage_fraction: float = 0.5,
    similarity_method: str = 'pearson',
    linkage_method: str = 'average'
) -> ConsensusAnalysisResults:
    """
    Run the full consensus DRGP analysis pipeline.

    Args:
        runs_folder: Path to folder containing run subdirectories
        output_dir: Path to save results
        prefix: File prefix for SVI outputs
        thresholds: Beta thresholds to test for binary representation
        min_coverage_fraction: Minimum fraction of runs for reproducible cluster
        similarity_method: 'pearson' or 'cosine' for continuous similarity
        linkage_method: Linkage method for hierarchical clustering

    Returns:
        ConsensusAnalysisResults object with all results
    """
    results = ConsensusAnalysisResults()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONSENSUS DRGP ANALYSIS")
    print("=" * 60)

    # Step 1: Load all DRGPs
    print("\n[Step 1] Loading DRGPs from all runs...")
    results.all_drgps = load_all_drgps(runs_folder, prefix)

    if len(results.all_drgps) == 0:
        raise ValueError("No DRGPs found in any run!")

    # Get number of runs
    run_ids = sorted(set(d.run_id for d in results.all_drgps))
    n_runs = len(run_ids)
    print(f"Total DRGPs loaded: {len(results.all_drgps)} from {n_runs} runs")

    # Step 2: Filter to positive DRGPs
    print("\n[Step 2] Filtering to positive-coefficient DRGPs...")
    results.positive_drgps = filter_positive_drgps(results.all_drgps)

    if len(results.positive_drgps) == 0:
        print("WARNING: No positive-coefficient DRGPs found!")
        print("Proceeding with all DRGPs instead...")
        results.positive_drgps = results.all_drgps

    # Step 3: Build similarity matrices
    print("\n[Step 3] Computing similarity matrices...")
    print(f"  Continuous similarity ({similarity_method})...")
    results.similarity_matrix_continuous = compute_continuous_similarity(
        results.positive_drgps, method=similarity_method
    )

    # Run for multiple thresholds
    binary_results = {}
    for threshold in thresholds:
        print(f"  Binary similarity (Jaccard, threshold={threshold})...")
        sim_binary = compute_binary_similarity(results.positive_drgps, threshold=threshold)
        binary_results[threshold] = sim_binary

    # Use default threshold for main analysis
    results.similarity_matrix_binary = binary_results[0.1]

    # Step 4: Hierarchical clustering
    print("\n[Step 4] Performing hierarchical clustering...")
    results.linkage_matrix = perform_hierarchical_clustering(
        results.similarity_matrix_continuous, method=linkage_method
    )

    # Find optimal clusters
    min_coverage = max(1, int(n_runs * min_coverage_fraction))
    print(f"  Minimum coverage for reproducibility: {min_coverage} runs ({min_coverage_fraction*100:.0f}%)")

    results.clusters, optimal_threshold = find_optimal_clusters(
        results.linkage_matrix,
        results.positive_drgps,
        n_runs,
        min_coverage=min_coverage_fraction
    )
    print(f"  Found {len(results.clusters)} clusters at distance threshold {optimal_threshold:.2f}")

    # Step 5: Assess cluster quality
    print("\n[Step 5] Assessing cluster quality...")
    cluster_report = generate_cluster_report(
        results.clusters,
        results.similarity_matrix_continuous,
        results.positive_drgps,
        n_runs
    )
    print(cluster_report.to_string())

    # Save cluster report
    cluster_report.to_csv(output_dir / 'cluster_quality_report.csv', index=False)

    # Step 6: Extract consensus programs
    print(f"\n[Step 6] Extracting consensus programs (min coverage: {min_coverage})...")
    results.consensus_programs = extract_consensus_programs(
        results.clusters,
        results.similarity_matrix_continuous,
        results.positive_drgps,
        n_runs,
        min_coverage=min_coverage,
        threshold=0.1
    )
    print(f"  Found {len(results.consensus_programs)} reproducible consensus programs")

    for i, prog in enumerate(results.consensus_programs):
        print(f"\n  Consensus Program {i+1}:")
        print(f"    Contributing runs: {prog.n_contributing_runs}/{n_runs}")
        print(f"    Genes (union): {len(prog.consensus_genes_union)}")
        print(f"    Genes (majority): {len(prog.consensus_genes_majority)}")
        print(f"    Genes (intersection): {len(prog.consensus_genes_intersection)}")
        print(f"    Top 5 genes: {[g[0] for g in prog.get_top_genes(5)]}")

    # Step 7: Gene set enrichment
    print("\n[Step 7] Preparing gene lists for enrichment analysis...")
    results.enrichment_results = run_enrichment_analysis(
        results.consensus_programs,
        output_dir
    )

    # Generate visualizations
    print("\n[Visualization] Generating plots...")

    # Dendrogram
    plot_dendrogram(
        results.linkage_matrix,
        results.positive_drgps,
        output_dir / 'dendrogram.png'
    )

    # Similarity matrices
    plot_similarity_matrix(
        results.similarity_matrix_continuous,
        results.positive_drgps,
        output_dir / 'similarity_matrix_continuous.png',
        title=f'DRGP Similarity Matrix ({similarity_method.capitalize()})'
    )

    # Get cluster labels for sorted heatmap
    cluster_labels = np.zeros(len(results.positive_drgps))
    for cluster in results.clusters:
        for drgp in cluster.members:
            idx = next(i for i, d in enumerate(results.positive_drgps)
                      if d.full_id == drgp.full_id)
            cluster_labels[idx] = cluster.cluster_id

    plot_similarity_matrix(
        results.similarity_matrix_continuous,
        results.positive_drgps,
        output_dir / 'similarity_matrix_clustered.png',
        title=f'DRGP Similarity Matrix (Clustered)',
        cluster_labels=cluster_labels
    )

    # Cluster summary
    plot_cluster_summary(cluster_report, output_dir / 'cluster_summary.png')

    # Run contribution matrix
    plot_run_contribution_matrix(
        results.clusters,
        n_runs,
        run_ids,
        output_dir / 'run_contribution_matrix.png'
    )

    # Consensus program gene plots
    if len(results.consensus_programs) > 0:
        plot_consensus_program_genes(
            results.consensus_programs,
            output_dir
        )

    # Save comprehensive results
    print("\n[Saving] Writing results to disk...")

    # Save consensus programs as CSV
    for i, prog in enumerate(results.consensus_programs):
        gene_df = prog.get_gene_dataframe(threshold=0.0)
        gene_df.to_csv(output_dir / f'consensus_program_{i+1}_genes.csv', index=False)

    # Save summary JSON
    summary = {
        'n_runs': n_runs,
        'run_ids': run_ids,
        'n_total_drgps': len(results.all_drgps),
        'n_positive_drgps': len(results.positive_drgps),
        'n_clusters': len(results.clusters),
        'n_consensus_programs': len(results.consensus_programs),
        'optimal_distance_threshold': optimal_threshold,
        'similarity_method': similarity_method,
        'linkage_method': linkage_method,
        'min_coverage_fraction': min_coverage_fraction,
        'thresholds_tested': thresholds,
        'consensus_programs': [
            {
                'cluster_id': prog.cluster_id,
                'n_contributing_runs': prog.n_contributing_runs,
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

    # Save full similarity matrix
    np.savez_compressed(
        output_dir / 'similarity_matrices.npz',
        continuous=results.similarity_matrix_continuous,
        binary_0p1=binary_results[0.1],
        linkage=results.linkage_matrix,
        drgp_ids=[d.full_id for d in results.positive_drgps]
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    print(f"  - Analyzed {len(results.positive_drgps)} positive DRGPs from {n_runs} runs")
    print(f"  - Found {len(results.clusters)} clusters")
    print(f"  - Identified {len(results.consensus_programs)} reproducible consensus programs")

    if len(results.consensus_programs) > 0:
        print(f"\nTop consensus program:")
        top = results.consensus_programs[0]
        print(f"  - Appears in {top.n_contributing_runs}/{n_runs} runs")
        print(f"  - Contains {len(top.consensus_genes_majority)} majority genes")
        print(f"  - Top genes: {', '.join([g[0] for g in top.get_top_genes(5)])}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze consensus gene programs across multiple SVI runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python analyze_consensus_drgps.py /path/to/runs_folder --output results/consensus

    # With custom parameters
    python analyze_consensus_drgps.py /path/to/runs_folder \\
        --output results/consensus \\
        --prefix svi \\
        --min-coverage 0.75 \\
        --similarity cosine \\
        --thresholds 0.05 0.1 0.2 0.5
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
        default=Path('./consensus_results'),
        help='Output directory for results (default: ./consensus_results)'
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
        help='Minimum fraction of runs for a cluster to be reproducible (default: 0.5)'
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

    # Run analysis
    results = run_consensus_analysis(
        runs_folder=args.runs_folder,
        output_dir=args.output,
        prefix=args.prefix,
        thresholds=args.thresholds,
        min_coverage_fraction=args.min_coverage,
        similarity_method=args.similarity,
        linkage_method=args.linkage
    )

    return results


if __name__ == '__main__':
    main()
