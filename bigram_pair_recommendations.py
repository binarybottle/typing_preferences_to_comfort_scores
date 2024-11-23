"""
Feature Space Module

This module provides functions for analyzing and visualizing feature space,
with specific application to bigram-pair (bigram-difference) features.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

def analyze_feature_space(
    feature_matrix: pd.DataFrame,
    output_paths: Dict[str, str],
    all_feature_differences: Dict,
    recommend_bigrams: bool = True,
    num_recommendations: int = 30,
    config: Optional[Dict] = None
) -> Dict:
    """
    Comprehensive feature space analysis including PCA projection, density analysis,
    and optional bigram recommendations.
    
    Args:
        feature_matrix: DataFrame of feature values for bigram pairs
        output_paths: Dictionary mapping output types to file paths
        all_feature_differences: Dictionary of all possible bigram pair differences
        recommend_bigrams: Whether to generate bigram recommendations
        num_recommendations: Number of bigram pairs to recommend
        config: Configuration dictionary containing feature grouping info
    
    Returns:
        Dictionary containing analysis results:
        - feature_space_metrics: Hull area and point density
        - feature_groups: Feature grouping information
        - recommendations: (optional) Suggested bigram pairs to test
    """
    results = {}
    
    # Get feature groups from config
    feature_groups = config['features']['groups']
    results['feature_groups'] = feature_groups

    # Create directories
    Path(output_paths['analysis']).parent.mkdir(parents=True, exist_ok=True)

    # PCA and scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Calculate metrics
    hull = ConvexHull(pca_result)
    hull_area = hull.area
    point_density = len(pca_result) / hull_area
    results['feature_space_metrics'] = {
        'hull_area': hull_area,
        'point_density': point_density
    }

    # Get axis limits for consistent plotting
    x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
    y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1

    # Plot PCA projection with consistent scale
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title('2D PCA projection of feature space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(output_paths['pca'])
    plt.close()

    # Identify underrepresented areas
    grid_points, distances = identify_underrepresented_areas(
        pca_result=pca_result,
        output_path=output_paths['underrepresented'],
        x_lims=(x_min, x_max),
        y_lims=(y_min, y_max)
    )

    if recommend_bigrams:
        recommendations = generate_bigram_recommendations(
            pca_result=pca_result,
            scaler=scaler,
            pca=pca,
            grid_points=grid_points,
            distances=distances,
            all_feature_differences=all_feature_differences,
            num_recommendations=num_recommendations
        )
        results['recommendations'] = recommendations
        
        # Save recommendations
        with open(output_paths['recommendations'], 'w') as f:
            for pair in recommendations:
                f.write(f"{pair[0][0]}{pair[0][1]}, {pair[1][0]}{pair[1][1]}\n")

        # Plot bigram graph
        plot_bigram_graph(feature_matrix.index, output_paths['bigram_graph'])
    
    # Save comprehensive analysis
    save_feature_space_analysis_results(results, output_paths['analysis'])

    return results

def save_feature_space_analysis_results(results: Dict, output_path: str) -> None:
    """
    Save feature space analysis results to a file.
    
    Args:
        results: Dictionary containing analysis results
        output_path: Path to save the analysis file
    """
    # [Implementation remains the same]

def identify_underrepresented_areas(
    pca_result: np.ndarray,
    output_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    num_grid: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify and visualize underrepresented areas in feature space.
    
    Args:
        pca_result: PCA-transformed feature data
        output_path: Path to save the visualization
        x_lims: (min, max) for x-axis
        y_lims: (min, max) for y-axis
        num_grid: Number of grid points for density estimation
    
    Returns:
        Tuple of (grid_points, distances) for use in recommendation generation
    """
    # [Implementation remains the same]

def generate_bigram_recommendations(
    pca_result: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    grid_points: np.ndarray,
    distances: np.ndarray,
    all_feature_differences: Dict,
    num_recommendations: int = 30
) -> List[Tuple]:
    """
    Generate recommendations for new bigram pairs to test.
    
    Uses PCA-space density analysis to identify underrepresented areas,
    then finds closest existing bigram pairs to generate recommendations.
    """
    # [Implementation remains the same]

def plot_bigram_graph(bigram_pairs: List[Tuple], output_path: str) -> None:
    """
    Create graph visualization of bigram relationships with minimal edge crossings.
    
    Organizes connected components vertically and uses Kamada-Kawai layout
    to minimize edge crossings within each component.
    """
    # [Implementation remains the same]