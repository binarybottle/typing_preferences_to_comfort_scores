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
    """
    Analyze feature space and generate recommendations.
    
    Args:
        feature_matrix: DataFrame of feature values
        output_paths: Dictionary of output file paths
        all_feature_differences: Dictionary of precomputed feature differences
        config: Configuration dictionary
        recommend_bigrams: Whether to generate bigram recommendations
        num_recommendations: Number of bigram recommendations to generate
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    try:
        results = {}
        
        logger.info("Starting feature space analysis")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        # Convert feature matrix to numpy array for PCA
        X = feature_matrix.values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.debug("Features normalized")
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Calculate hull area and point density
        hull = ConvexHull(X_pca)
        point_density = len(X_pca) / hull.area
        
        results['feature_space_metrics'] = {
            'hull_area': float(hull.area),
            'point_density': float(point_density),
            'variance_explained': pca.explained_variance_ratio_.tolist()
        }
        logger.debug(f"Feature space metrics calculated: {results['feature_space_metrics']}")
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Feature Space PCA')
        plt.savefig(output_paths['pca'], dpi=300, bbox_inches='tight')
        plt.close()
        
        if recommend_bigrams:
            logger.info("Generating bigram recommendations")
            # Find underrepresented regions
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(X_pca)
            
            # Generate grid of points
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            
            # Get density estimates
            positions = np.vstack([xx.ravel(), yy.ravel()]).T
            density = np.exp(kde.score_samples(positions))
            
            # Find low density regions
            low_density_mask = density < np.percentile(density, 10)
            low_density_points = positions[low_density_mask]
            
            # Plot underrepresented regions with density heatmap
            plot_underrepresented_regions(
                X_pca=X_pca,
                density=density,
                positions=positions,
                pca=pca,
                output_path=output_paths['underrepresented']
            )
            
            # Generate recommendations
            recommendations = []
            feature_names = feature_matrix.columns.tolist()
            
            for point in low_density_points[:num_recommendations]:
                # Reshape point to 2D array for inverse transform
                point_2d = point.reshape(1, -1)
                
                # Project back to original feature space
                feature_point = pca.inverse_transform(point_2d)
                feature_point = scaler.inverse_transform(feature_point)
                
                # Find nearest existing bigram
                distances = np.linalg.norm(X - feature_point, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_bigram = feature_matrix.index[nearest_idx]
                
                recommendations.append({
                    'bigram': nearest_bigram,
                    'distance': float(distances[nearest_idx]),
                    'feature_values': {
                        name: float(value) 
                        for name, value in zip(feature_names, feature_point.flatten())
                    }
                })
            
            # Sort recommendations by distance
            recommendations.sort(key=lambda x: x['distance'])
            recommendations = recommendations[:num_recommendations]
            results['recommendations'] = recommendations
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            # Save recommendations in both formats
            save_bigram_recommendations(
                recommendations,
                output_paths['recommendations'].replace('recommended_bigram_pairs_scores.txt', 
                                                        'recommended_bigram_pairs.txt')
            )
            
            # Create bigram graph
            plot_bigram_graph(
                feature_matrix=feature_matrix,
                recommendations=recommendations,
                output_path=output_paths['bigram_graph']
            )
        
        # Save full analysis
        with open(output_paths['analysis'], 'w') as f:
            f.write("Feature Space Analysis Results\n")
            f.write("============================\n\n")
            
            # Write metrics
            metrics = results['feature_space_metrics']
            f.write("Feature Space Metrics:\n")
            f.write(f"Hull Area: {metrics['hull_area']:.3f}\n")
            f.write(f"Point Density: {metrics['point_density']:.3f}\n")
            f.write("Variance Explained:\n")
            f.write(f"  PC1: {metrics['variance_explained'][0]:.2%}\n")
            f.write(f"  PC2: {metrics['variance_explained'][1]:.2%}\n\n")
            
            # Write feature importance (based on PCA loadings)
            f.write("Feature Importance (PCA Loadings):\n")
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=feature_names
            )
            for feature in feature_names:
                f.write(f"{feature}:\n")
                f.write(f"  PC1: {feature_importance.loc[feature, 'PC1']:.3f}\n")
                f.write(f"  PC2: {feature_importance.loc[feature, 'PC2']:.3f}\n")
            f.write("\n")
            
            if 'recommendations' in results:
                f.write("Top Recommended Bigrams:\n")
                for i, rec in enumerate(results['recommendations'][:10], 1):
                    f.write(f"{i}. {rec['bigram'][0]}{rec['bigram'][1]} "
                           f"(distance: {rec['distance']:.3f})\n")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in feature space analysis: {str(e)}")
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

def save_bigram_recommendations(recommendations: List[Dict], output_base_path: str) -> None:
    """Save bigram recommendations in two formats."""

    # Save simple pairs list
    with open(output_base_path, 'w') as f:
        for rec in recommendations:
            bigram = rec['bigram']
            f.write(f"{format_bigram_pair(bigram)}\n")

    # Save detailed scores
    scores_path = output_base_path.replace('.txt', '_scores.txt')
    with open(scores_path, 'w') as f:
        f.write("Bigram Recommendations with Scores:\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. Bigram: {rec['bigram']}\n")
            f.write(f"   Distance: {rec['distance']:.3f}\n")
            f.write("   Feature values:\n")
            for name, value in rec['feature_values'].items():
                f.write(f"     {name}: {value:.3f}\n")
            f.write("\n")

def format_bigram_pair(bigram_pair: Tuple[Tuple[str, str], Tuple[str, str]]) -> str:
    """Format a pair of bigram tuples as a comma-separated string."""
    return f"{bigram_pair[0][0]}{bigram_pair[0][1]}, {bigram_pair[1][0]}{bigram_pair[1][1]}"

def plot_underrepresented_regions(
    X_pca: np.ndarray,
    density: np.ndarray,
    positions: np.ndarray,
    pca: PCA,
    output_path: str
) -> None:
    """Plot underrepresented regions with density heatmap."""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with viridis colormap
    xx = positions[:, 0].reshape(100, 100)
    yy = positions[:, 1].reshape(100, 100)
    zz = density.reshape(100, 100)
    
    plt.contourf(xx, yy, zz, levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    
    # Plot existing points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
               c='white', alpha=0.7, s=50,
               edgecolors='black', linewidth=1,
               label='Existing bigrams')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Feature Space Density with Existing Bigrams')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_bigram_recommendations(
    pca_result: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    grid_points: np.ndarray,
    distances: np.ndarray,
    all_feature_differences: Dict,
    num_recommendations: int = 30
) -> List[Dict]:
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
                # Format as a dictionary with properly formatted bigrams
                recommendations.append({
                    'bigram': best_pair,
                    'formatted': f"{format_bigram(best_pair[0])}, {format_bigram(best_pair[1])}",
                    'distance': min_dist
                })
        
        return recommendations[:num_recommendations]
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return None

def plot_bigram_graph(
    feature_matrix: pd.DataFrame,
    recommendations: List[Dict] = None,
    output_path: str = None
) -> None:
    try:
        import networkx as nx
        
        G = nx.Graph()
        
        # Add edges from the feature matrix index
        for bigram_pair in feature_matrix.index:
            chosen_bigram = ''.join(bigram_pair[0])
            unchosen_bigram = ''.join(bigram_pair[1])
            G.add_edge(chosen_bigram, unchosen_bigram)
            
        # Try different layout algorithms for fewer crossings
        # pos = nx.spring_layout(G, k=2, iterations=100)  # Increase k and iterations
        pos = nx.kamada_kawai_layout(G)  # Often better for avoiding crossings
        # pos = nx.circular_layout(G)  # Try circular layout
        
        plt.figure(figsize=(15, 15))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             alpha=0.3,
                             width=0.5)
        
        # Draw nodes and create legend elements
        legend_elements = []
        
        if recommendations:
            node_size = 50
            recommended_bigrams = set()
            for rec in recommendations:
                bigram = rec['bigram']
                recommended_bigrams.add(''.join(bigram[0]))
                recommended_bigrams.add(''.join(bigram[1]))
            
            # Regular nodes
            regular_nodes = [node for node in G.nodes() if node not in recommended_bigrams]
            if regular_nodes:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=regular_nodes,
                                     node_color='lightblue',
                                     node_size=node_size,
                                     alpha=0.5)
                legend_elements.append(plt.scatter([], [], c='lightblue', alpha=0.5,
                                       s=node_size, label='Existing bigrams'))
            
            # Recommended nodes
            recommended_nodes = [node for node in G.nodes() if node in recommended_bigrams]
            if recommended_nodes:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=recommended_nodes,
                                     node_color='red',
                                     node_size=node_size,
                                     alpha=0.5)
                legend_elements.append(plt.scatter([], [], c='red', alpha=0.5,
                                       s=node_size, label='Recommended bigrams'))
        else:
            nx.draw_networkx_nodes(G, pos,
                                 node_color='lightblue',
                                 node_size=node_size,
                                 alpha=0.5)
            legend_elements.append(plt.scatter([], [], c='lightblue', alpha=0.6,
                                            s=node_size, label='Bigrams'))
        
        # Add edge to legend
        legend_elements.append(plt.Line2D([0], [0], color='lightgray', alpha=0.5,
                               label='Appears together in data'))
        
        # Add labels with small offset
        label_pos = {node: (coord[0], coord[1] + 0.03) for node, coord in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=8, font_weight='regular')
        
        plt.title('Bigram Pair Network')
        plt.axis('off')
        
        # Add legend
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))
        
        plt.margins(0.2)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'n_bigrams': G.number_of_nodes(),
            'n_connections': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
        }
        
    except Exception as e:
        logger.error(f"Error creating bigram graph: {str(e)}")
        return None
        
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

    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
    