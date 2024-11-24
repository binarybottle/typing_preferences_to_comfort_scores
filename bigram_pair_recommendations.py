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
from typing import Dict, List, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

def analyze_feature_space(
    feature_matrix: pd.DataFrame,
    output_paths: Dict[str, str],
    all_feature_differences: Dict[Tuple[str, str], np.ndarray],
    config: Dict[str, Any],
    recommend_bigrams: bool = True,
    num_recommendations: int = 30
) -> Dict[str, Any]:
    """Analyze feature space and generate recommendations."""
    try:
        results = {}
        
        logger.info("Starting feature space analysis")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Validate output paths
        required_paths = ['pca', 'underrepresented', 'recommendations', 'analysis']
        for path_key in required_paths:
            if path_key not in output_paths:
                raise ValueError(f"Missing required output path: {path_key}")
            # Create parent directories
            Path(output_paths[path_key]).parent.mkdir(parents=True, exist_ok=True)
        
        # Rest of the function remains the same until saving results
        
        # Save analysis results
        if results:
            logger.info("Saving analysis results")
            save_feature_space_analysis_results(results, output_paths['analysis'])
            logger.info("Analysis results saved")
            
        return results
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
def save_feature_space_analysis_results(results: Dict[str, Any], analysis_file_path: str) -> None:
    """Save feature space analysis results to file with proper error handling."""
    try:
        # Create directory if it doesn't exist
        analysis_dir = Path(analysis_file_path).parent
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving analysis results to {analysis_file_path}")
        
        with open(analysis_file_path, 'w') as f:
            f.write("Feature Space Analysis Results\n")
            f.write("============================\n\n")
            
            if 'feature_space_metrics' in results:
                metrics = results['feature_space_metrics']
                f.write("Feature Space Metrics:\n")
                f.write(f"Hull Area: {metrics['hull_area']:.3f}\n")
                f.write(f"Point Density: {metrics['point_density']:.3f}\n")
                f.write("Variance Explained:\n")
                f.write(f"  PC1: {metrics['variance_explained'][0]:.2%}\n")
                f.write(f"  PC2: {metrics['variance_explained'][1]:.2%}\n\n")
            
            if 'recommendations' in results:
                f.write("\nBigram Recommendations:\n")
                f.write("=====================\n\n")
                for i, rec in enumerate(results['recommendations'], 1):
                    f.write(f"{i}. Bigram: {rec['bigram']}\n")
                    f.write(f"   Distance: {rec['distance']:.3f}\n")
                    f.write("   Feature values:\n")
                    for name, value in rec['feature_values'].items():
                        f.write(f"     {name}: {value:.3f}\n")
                    f.write("\n")
                    
        logger.info("Analysis results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

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
    