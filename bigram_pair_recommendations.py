"""
Feature Space Module

This module provides functions for analyzing and visualizing feature space.

We apply this module to bigram-pair (bigram-difference) features.
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
from scipy.spatial.distance import cdist
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
    Comprehensive feature space analysis including multicollinearity and recommendations.
    """
    results = {}
    
    # Get feature groups from config
    feature_groups = config['features']['groups']

    # Add feature group information to results
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

    # Identify underrepresented areas with same scale
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

        # Plot bigram graph using all pairs from input data
        plot_bigram_graph(feature_matrix.index, output_paths['bigram_graph'])
    
    # Save comprehensive analysis including VIF results
    save_feature_space_analysis_results(results, output_paths['analysis'])

    return results

def plot_feature_space(feature_matrix: pd.DataFrame,
                       output_path: str) -> Tuple[PCA, ConvexHull]:
    """
    Create PCA visualization of the feature space.
    
    Args:
        feature_matrix: DataFrame containing feature values
        output_path: Path to save the plot
        
    Returns:
        Tuple containing fitted PCA object and computed ConvexHull
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title('2D PCA projection of feature space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Calculate convex hull
    hull = ConvexHull(pca_result)
    
    # Plot hull boundary
    for simplex in hull.simplices:
        plt.plot(pca_result[simplex, 0], pca_result[simplex, 1], 'r-', alpha=0.5)
    
    plt.savefig(output_path)
    plt.close()
    
    return pca, hull

def plot_feature_space_density(pca_result: np.ndarray,
                               output_path: str,
                               num_grid: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create density plot showing underrepresented areas in feature space.
    
    Args:
        pca_result: PCA-transformed feature data
        output_path: Path to save the plot
        num_grid: Number of grid points for density estimation
        
    Returns:
        Tuple containing grid points and distances
    """
    x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
    y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_grid),
                        np.linspace(y_min, y_max, num_grid))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Calculate distances
    distances = cdist(grid_points, pca_result).min(axis=1)
    distances = distances.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(distances, extent=[x_min, x_max, y_min, y_max],
               origin='lower', cmap='viridis')
    plt.colorbar(label='Distance to nearest data point')
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='red', s=20, alpha=0.5)
    plt.title('Underrepresented Areas in Feature Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    plt.savefig(output_path)
    plt.close()
    
    return grid_points, distances
    
def save_feature_space_analysis_results(results: Dict, output_path: str) -> None:
    """Save feature space analysis results to a file."""
    with open(output_path, 'w') as f:
        f.write(" ---- Feature Space Analysis ---- \n\n")
        
        # Basic metrics
        f.write("Feature Space Metrics:\n")
        f.write(f"Convex Hull Area: {results['feature_space_metrics']['hull_area']:.3f}\n")
        f.write(f"Point Density: {results['feature_space_metrics']['point_density']:.3f}\n\n")
        
        # Multicollinearity results if present
        if 'multicollinearity' in results:
            f.write("Multicollinearity Analysis:\n")
            f.write("\nVariance Inflation Factors:\n")
            for vif in results['multicollinearity']['vif']:
                f.write(f"{vif['Feature']:<20} VIF: {vif['VIF']:>8.3f} ({vif['Status']})\n")
            
            if results['multicollinearity']['high_correlations']:
                f.write("\nHigh Feature Correlations (>0.7):\n")
                for corr in results['multicollinearity']['high_correlations']:
                    f.write(f"{corr['Feature1']} - {corr['Feature2']}: {corr['Correlation']:.3f}\n")
        
        # Recommendations if present
        if 'recommendations' in results:
            f.write("\nRecommended Bigram Pairs for Further Testing:\n")
            for i, pair in enumerate(results['recommendations'], 1):
                f.write(f"{i:2d}. {pair[0][0]}{pair[0][1]} vs {pair[1][0]}{pair[1][1]}\n")

def identify_underrepresented_areas(pca_result: np.ndarray,
                                  output_path: str,
                                  x_lims: Tuple[float, float],
                                  y_lims: Tuple[float, float],
                                  num_grid: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Identify and visualize underrepresented areas in feature space."""
    xx, yy = np.meshgrid(np.linspace(x_lims[0], x_lims[1], num_grid),
                        np.linspace(y_lims[0], y_lims[1], num_grid))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances = cdist(grid_points, pca_result).min(axis=1)
    distances = distances.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(distances, extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
               origin='lower', cmap='viridis')
    plt.colorbar(label='Distance to nearest data point')
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='red', s=20, alpha=0.5)
    plt.title('Underrepresented Areas in Feature Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return grid_points, distances

def generate_bigram_recommendations(pca_result: np.ndarray,
                                  scaler: StandardScaler,
                                  pca: PCA,
                                  grid_points: np.ndarray,
                                  distances: np.ndarray,
                                  all_feature_differences: Dict,
                                  num_recommendations: int = 30) -> List[Tuple]:
    """Generate recommendations for new bigram pairs to collect data on."""
    # Sort grid points by distance (descending)
    sorted_indices = np.argsort(distances.ravel())[::-1]
    
    # Transform all grid points back to original feature space
    original_space_points = scaler.inverse_transform(pca.inverse_transform(grid_points))
    feature_points = np.abs(np.round(original_space_points))
    
    # Convert feature points to bigram pairs
    all_pairs = list(all_feature_differences.keys())
    all_values = np.array(list(all_feature_differences.values()))
    
    # Find closest existing bigram pairs to our generated points
    distances_to_existing = cdist(feature_points, all_values)
    
    # Get recommendations
    recommended_pairs = []
    seen = set()
    
    # Keep going through points until we have enough recommendations
    for idx in sorted_indices:
        closest_pair_idx = np.argmin(distances_to_existing[idx])
        pair = all_pairs[closest_pair_idx]
        
        # Convert to a hashable format for deduplication
        pair_tuple = tuple(sorted([tuple(pair[0]), tuple(pair[1])]))
        
        if pair_tuple not in seen:
            seen.add(pair_tuple)
            recommended_pairs.append(pair)
            
            if len(recommended_pairs) >= num_recommendations:
                break
    
    # If we still don't have enough recommendations, try the next closest pairs
    while len(recommended_pairs) < num_recommendations:
        for idx in sorted_indices:
            # Get indices of k closest pairs, sorted by distance
            closest_indices = np.argsort(distances_to_existing[idx])
            
            # Try each close pair until we find one we haven't used
            for pair_idx in closest_indices:
                pair = all_pairs[pair_idx]
                pair_tuple = tuple(sorted([tuple(pair[0]), tuple(pair[1])]))
                
                if pair_tuple not in seen:
                    seen.add(pair_tuple)
                    recommended_pairs.append(pair)
                    
                    if len(recommended_pairs) >= num_recommendations:
                        return recommended_pairs
                    break
                    
        # If we've tried everything and still don't have enough, break to avoid infinite loop
        if len(recommended_pairs) == len(seen):
            logger.warning(f"Could only generate {len(recommended_pairs)} unique recommendations")
            break
    
    return recommended_pairs

def plot_bigram_graph(bigram_pairs: List[Tuple], output_path: str) -> None:
    """
    Create graph visualization of bigram relationships with minimal edge crossings.
    
    Args:
        bigram_pairs: List of bigram pairs from the data
        output_path: Path to save the visualization
    """
    # Create graph
    G = nx.Graph()
    
    # Add all bigrams and their connections
    for pair in bigram_pairs:
        bigram1, bigram2 = pair
        # Add both bigrams as nodes
        G.add_node(''.join(bigram1))
        G.add_node(''.join(bigram2))
        # Add edge between them
        G.add_edge(''.join(bigram1), ''.join(bigram2))
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    # Sort components by size (largest first)
    components = sorted(components, key=len, reverse=True)
    
    # Calculate layout for each component separately
    pos = {}
    y_offset = 0
    max_width = 0
    
    for component in components:
        # Create subgraph for this component
        subgraph = G.subgraph(component)
        
        # Use kamada_kawai_layout for better edge crossing minimization
        # Scale and center each component independently
        component_pos = nx.kamada_kawai_layout(subgraph, scale=2.0)
        
        # Calculate component dimensions
        x_vals = [coord[0] for coord in component_pos.values()]
        y_vals = [coord[1] for coord in component_pos.values()]
        width = max(x_vals) - min(x_vals)
        height = max(y_vals) - min(y_vals)
        max_width = max(max_width, width)
        
        # Add offset to y coordinates to separate components
        for node in component:
            pos[node] = np.array([
                component_pos[node][0],
                component_pos[node][1] + y_offset
            ])
        
        # Update y_offset for next component
        # Add padding between components
        y_offset -= (height + 1.5)
    
    # Create the visualization
    plt.figure(figsize=(20, max(20, -y_offset)))
    
    # Draw edges first
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        alpha=0.5,
        width=1.0
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=1000,
        alpha=0.7
    )
    
    # Add labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold'
    )
    
    # Remove axes for cleaner look
    plt.axis('off')
    
    # Add title
    plt.title('Bigram Comparison Graph\n(Components arranged vertically, edge crossings minimized)',
              pad=20, fontsize=14)
    
    # Adjust layout to use full figure space
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


