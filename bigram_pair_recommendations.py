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

def identify_underrepresented_areas(
    pca_result: np.ndarray,
    output_path: str,
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    num_grid: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify and visualize underrepresented areas in feature space.
    """
    try:
        # Create grid
        x = np.linspace(x_lims[0], x_lims[1], num_grid)
        y = np.linspace(y_lims[0], y_lims[1], num_grid)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        # Calculate distances to nearest points
        distances = np.zeros(len(grid_points))
        for i, point in enumerate(grid_points):
            dist_to_all = np.sqrt(np.sum((pca_result - point) ** 2, axis=1))
            distances[i] = np.min(dist_to_all)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                   c='black', alpha=0.6, label='Data points')
        plt.tricontourf(grid_points[:, 0], grid_points[:, 1], 
                       distances.reshape(-1))
        plt.colorbar(label='Distance to nearest point')
        plt.title('Underrepresented Areas in Feature Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        
        return grid_points, distances
        
    except Exception as e:
        logger.error(f"Error in underrepresented areas analysis: {str(e)}")
        return None

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
    """
    try:
        # Find points in underrepresented areas
        max_distances = np.argsort(distances)[-num_recommendations:]
        target_points = grid_points[max_distances]
        
        # Transform back to feature space
        target_features = pca.inverse_transform(target_points)
        target_features = scaler.inverse_transform(target_features)
        
        # Find closest existing bigram pairs
        recommendations = []
        for target in target_features:
            min_dist = float('inf')
            best_pair = None
            
            for pair, features in all_feature_differences.items():
                dist = np.sqrt(np.sum((np.array(features) - target) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    best_pair = pair
            
            if best_pair is not None:
                recommendations.append(best_pair)
        
        return recommendations[:num_recommendations]
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return None

def plot_bigram_graph(bigram_pairs: List[Tuple], output_path: str) -> None:
    """
    Create graph visualization of bigram relationships.
    """
    try:
        G = nx.Graph()
        
        # Add nodes and edges
        for pair in bigram_pairs:
            b1, b2 = pair
            G.add_edge(f"{b1[0]}{b1[1]}", f"{b2[0]}{b2[1]}")
        
        # Create layout
        pos = nx.kamada_kawai_layout(G)
        
        # Plot
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=1000, font_size=10, font_weight='bold')
        plt.title('Bigram Relationship Graph')
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting bigram graph: {str(e)}")

def save_feature_space_analysis_results(results: Dict, output_path: str) -> None:
    """
    Save feature space analysis results to a file.
    """
    try:
        with open(output_path, 'w') as f:
            f.write("=== Feature Space Analysis Results ===\n\n")
            
            # Write metrics
            if 'feature_space_metrics' in results:
                metrics = results['feature_space_metrics']
                f.write("Feature Space Metrics:\n")
                f.write(f"  Hull Area: {metrics['hull_area']:.3f}\n")
                f.write(f"  Point Density: {metrics['point_density']:.3f}\n\n")
            
            # Write feature groups
            if 'feature_groups' in results:
                groups = results['feature_groups']
                f.write("Feature Groups:\n")
                for group_name, features in groups.items():
                    f.write(f"\n{group_name}:\n")
                    for feature in features:
                        f.write(f"  - {feature}\n")
            
            # Write recommendations
            if 'recommendations' in results:
                f.write("\nBigram Pair Recommendations:\n")
                for i, pair in enumerate(results['recommendations'], 1):
                    f.write(f"{i}. {pair[0][0]}{pair[0][1]} - {pair[1][0]}{pair[1][1]}\n")
                    
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
    