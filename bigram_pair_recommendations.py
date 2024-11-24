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

    # Save simple pair list
    pairs_path = output_base_path
    with open(pairs_path, 'w') as f:
        for rec in recommendations:
            bigram = rec['bigram']
            f.write(f"{bigram[0]}{bigram[1]}\n")

def plot_underrepresented_regions(
    X_pca: np.ndarray,
    density: np.ndarray,
    positions: np.ndarray,
    pca: PCA,
    output_path: str
) -> None:
    """Plot underrepresented regions with distance heatmap."""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    xx = positions[:, 0].reshape(100, 100)
    yy = positions[:, 1].reshape(100, 100)
    zz = density.reshape(100, 100)
    
    plt.contourf(xx, yy, zz, levels=20, cmap='YlOrRd_r')
    plt.colorbar(label='Density')
    
    # Plot existing points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
               c='blue', alpha=0.5, s=50, label='Existing bigrams')
    
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

def plot_bigram_graph(
    feature_matrix: pd.DataFrame,
    recommendations: List[Dict],
    output_path: str,
    top_n: int = 50  # Number of existing bigrams to show
) -> None:
    """
    Plot a graph visualization of bigram relationships.
    
    Nodes are individual characters, edges represent bigrams.
    Edge thickness represents frequency/strength of relationship.
    Recommended bigrams are highlighted differently.
    """
    try:
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add existing bigrams
        for idx in feature_matrix.index[:top_n]:
            if isinstance(idx, tuple) and len(idx) == 2:
                char1, char2 = idx
                # Add nodes
                G.add_node(char1, type='existing')
                G.add_node(char2, type='existing')
                # Add edge
                G.add_edge(char1, char2, type='existing', weight=1.0)
        
        # Add recommended bigrams
        for rec in recommendations:
            bigram = rec['bigram']
            if isinstance(bigram, tuple) and len(bigram) == 2:
                char1, char2 = bigram
                # Add nodes if not exist
                if char1 not in G:
                    G.add_node(char1, type='recommended')
                if char2 not in G:
                    G.add_node(char2, type='recommended')
                # Add edge
                G.add_edge(char1, char2, type='recommended', 
                          weight=2.0, distance=rec['distance'])
        
        # Set up plot
        plt.figure(figsize=(12, 8))
        
        # Use spring layout
        pos = nx.spring_layout(G, k=1.5)
        
        # Draw existing bigrams
        nx.draw_networkx_edges(G, pos,
                             edgelist=[(u, v) for u, v, d in G.edges(data=True) 
                                      if d['type'] == 'existing'],
                             edge_color='gray', alpha=0.5)
        
        # Draw recommended bigrams
        nx.draw_networkx_edges(G, pos,
                             edgelist=[(u, v) for u, v, d in G.edges(data=True) 
                                      if d['type'] == 'recommended'],
                             edge_color='red', alpha=0.7, width=2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             node_color='lightblue',
                             node_size=500)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Bigram Relationships Graph\n(Red edges: recommended bigrams)')
        plt.axis('off')
        
        # Add legend
        plt.plot([], [], 'gray', alpha=0.5, label='Existing bigrams')
        plt.plot([], [], 'red', alpha=0.7, linewidth=2, label='Recommended bigrams')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bigram graph saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating bigram graph: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

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
    