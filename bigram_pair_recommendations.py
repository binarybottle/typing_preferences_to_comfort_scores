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
    try:
        results = {}
        X = feature_matrix.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        hull = ConvexHull(X_pca)
        point_density = len(X_pca) / hull.area
        
        results['feature_space_metrics'] = {
            'hull_area': float(hull.area),
            'point_density': float(point_density),
            'variance_explained': pca.explained_variance_ratio_.tolist()
        }
        
        if recommend_bigrams:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(X_pca)
            
            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            
            positions = np.vstack([xx.ravel(), yy.ravel()]).T
            density = np.exp(kde.score_samples(positions))
            
            low_density_mask = density < np.percentile(density, 10)
            low_density_points = positions[low_density_mask]
            
            recommendations = []
            seen_bigrams = set()
            feature_names = feature_matrix.columns.tolist()
            
            for point in low_density_points:
                if len(recommendations) >= num_recommendations:
                    break
                    
                point_2d = point.reshape(1, -1)
                feature_point = pca.inverse_transform(point_2d)
                feature_point = scaler.inverse_transform(feature_point)
                
                k = 5
                distances = np.linalg.norm(X - feature_point, axis=1)
                nearest_indices = np.argsort(distances)[:k]
                
                for idx in nearest_indices:
                    bigram = feature_matrix.index[idx]
                    sorted_bigram = tuple(sorted([
                        ''.join(bigram[0]), 
                        ''.join(bigram[1])
                    ]))
                    
                    if sorted_bigram not in seen_bigrams:
                        seen_bigrams.add(sorted_bigram)
                        recommendations.append({
                            'bigram': bigram,
                            'distance': float(distances[idx]),
                            'feature_values': {
                                name: float(value) 
                                for name, value in zip(feature_names, feature_point.flatten())
                            },
                            'pca_coords': point
                        })
                        break
            
            recommendations.sort(key=lambda x: x['distance'])
            recommendations = recommendations[:num_recommendations]
            results['recommendations'] = recommendations
            
            # Save recommendations
            with open(output_paths['recommendations'].replace('_scores.txt', '.txt'), 'w') as f:
                for rec in recommendations:
                    bigram = rec['bigram']
                    f.write(f"{bigram[0][0]}{bigram[0][1]}, {bigram[1][0]}{bigram[1][1]}\n")
                    
            with open(output_paths['recommendations'], 'w') as f:
                f.write("Bigram Recommendations with Scores:\n\n")
                for i, rec in enumerate(recommendations, 1):
                    bigram = rec['bigram']
                    f.write(f"{i}. {bigram[0][0]}{bigram[0][1]}, {bigram[1][0]}{bigram[1][1]}\n")
                    f.write(f"   Distance: {rec['distance']:.3f}\n")
                    f.write("   Feature values:\n")
                    for name, value in rec['feature_values'].items():
                        f.write(f"     {name}: {value:.3f}\n")
                    f.write("\n")
            
            with open(output_paths['analysis'], 'w') as f:
                f.write("Feature Space Analysis Results\n")
                f.write("============================\n\n")
                
                metrics = results['feature_space_metrics']
                f.write("Feature Space Metrics:\n")
                f.write(f"Hull Area: {metrics['hull_area']:.3f}\n")
                f.write(f"Point Density: {metrics['point_density']:.3f}\n")
                f.write("Variance Explained:\n")
                f.write(f"  PC1: {metrics['variance_explained'][0]:.2%}\n")
                f.write(f"  PC2: {metrics['variance_explained'][1]:.2%}\n\n")
                
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
                
                f.write("Top Recommended Bigrams:\n")
                for i, rec in enumerate(recommendations[:10], 1):
                    bigram = rec['bigram']
                    f.write(f"{i}. {bigram[0][0]}{bigram[0][1]}, {bigram[1][0]}{bigram[1][1]} ")
                    f.write(f"(distance: {rec['distance']:.3f})\n")
            
            plot_bigram_graph_variants(
                feature_matrix=feature_matrix,
                recommendations=recommendations,
                output_base_path=output_paths['bigram_graph'].replace('.png', '')
            )
            
            plot_underrepresented_variants(
                X_pca=X_pca,
                density=density,
                positions=positions,
                pca=pca,
                scaler=scaler,  # Pass scaler instance
                bigram_pairs=feature_matrix.index.tolist(),
                recommendations=recommendations,
                output_base_path=output_paths['underrepresented'].replace('.png', '')
            )
            
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

def plot_underrepresented_variants(
    X_pca: np.ndarray,
    density: np.ndarray,
    positions: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,  # Added scaler parameter
    bigram_pairs: List[Tuple],
    recommendations: List[Dict] = None,
    output_base_path: str = None
) -> None:
    # Basic plot (original version)
    plt.figure(figsize=(12, 8))
    plt.contourf(positions[:, 0].reshape(100, 100), 
                positions[:, 1].reshape(100, 100), 
                density.reshape(100, 100), 
                levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
               c='white', alpha=0.7, s=50,
               edgecolors='black', linewidth=1,
               label='Existing bigram pairs')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Feature Space Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_base_path + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot with recommended bigram pairs
    if recommendations:
        plt.figure(figsize=(15, 10))
        plt.contourf(positions[:, 0].reshape(100, 100), 
                    positions[:, 1].reshape(100, 100), 
                    density.reshape(100, 100), 
                    levels=20, cmap='viridis')
        plt.colorbar(label='Density')
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                   c='white', alpha=0.7, s=50,
                   edgecolors='black', linewidth=1,
                   label='Existing bigram pairs')
        
        # Plot recommended bigram pairs in red with labels
        for rec in recommendations:
            point_2d = pca.transform(scaler.transform(
                np.array([list(rec['feature_values'].values())])
            ))[0]
            plt.scatter(point_2d[0], point_2d[1], 
                       c='red', s=50, alpha=0.7,
                       edgecolors='black', linewidth=1)
            bigram = rec['bigram']
            label = f"{bigram[0][0]}{bigram[0][1]}-{bigram[1][0]}{bigram[1][1]}"
            plt.annotate(label, (point_2d[0], point_2d[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red', alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Feature Space Density with Recommended Bigram Pairs')
        plt.legend(['Existing pairs', 'Recommended pairs'])
        plt.tight_layout()
        plt.savefig(output_base_path + '_with_recommendations.png', dpi=300, bbox_inches='tight')
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

def plot_bigram_graph_variants(
    feature_matrix: pd.DataFrame,
    recommendations: List[Dict] = None,
    output_base_path: str = None
) -> None:
    # Basic graph - only existing pairs
    G_basic = nx.Graph()
    for bigram_pair in feature_matrix.index:
        chosen_bigram = ''.join(bigram_pair[0])
        unchosen_bigram = ''.join(bigram_pair[1])
        G_basic.add_edge(chosen_bigram, unchosen_bigram)
    
    pos = nx.kamada_kawai_layout(G_basic)
    
    # Plot basic graph
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_edges(G_basic, pos, edge_color='gray', alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G_basic, pos, node_color='lightblue', node_size=50, alpha=0.5)
    
    label_pos = {node: (coord[0], coord[1] + 0.03) for node, coord in pos.items()}
    nx.draw_networkx_labels(G_basic, label_pos, font_size=8)
    
    plt.title('Bigram Pair Network')
    plt.axis('off')
    plt.legend(['Existing bigrams'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.margins(0.2)
    plt.savefig(output_base_path + '_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graph with recommendations
    if recommendations:
        G_full = G_basic.copy()
        new_bigrams = set()
        recommended_edges = []
        
        # Add recommended pairs
        for rec in recommendations:
            bigram = rec['bigram']
            bg1, bg2 = ''.join(bigram[0]), ''.join(bigram[1])
            
            if bg1 not in G_basic:
                new_bigrams.add(bg1)
            if bg2 not in G_basic:
                new_bigrams.add(bg2)
                
            G_full.add_edge(bg1, bg2)
            recommended_edges.append((bg1, bg2))
        
        plt.figure(figsize=(15, 15))
        
        # Draw existing edges
        nx.draw_networkx_edges(G_full, pos, 
                             edgelist=G_basic.edges(),
                             edge_color='gray',
                             alpha=0.3,
                             width=0.5)
        
        # Draw recommended edges
        nx.draw_networkx_edges(G_full, pos,
                             edgelist=recommended_edges,
                             edge_color='red',
                             alpha=0.5,
                             width=1.0)
        
        # Draw existing nodes
        existing_nodes = [n for n in G_full.nodes() if n not in new_bigrams]
        nx.draw_networkx_nodes(G_full, pos,
                             nodelist=existing_nodes,
                             node_color='lightblue',
                             node_size=50,
                             alpha=0.5)
        
        # Draw new nodes
        if new_bigrams:
            nx.draw_networkx_nodes(G_full, pos,
                                 nodelist=list(new_bigrams),
                                 node_color='red',
                                 node_size=50,
                                 alpha=0.5)
        
        nx.draw_networkx_labels(G_full, label_pos, font_size=8)
        
        plt.title('Bigram Pair Network with Recommendations')
        plt.axis('off')
        legend_elements = [
            plt.scatter([], [], c='lightblue', alpha=0.5, s=50, label='Existing bigrams'),
            plt.scatter([], [], c='red', alpha=0.5, s=50, label='New bigrams'),
            plt.Line2D([0], [0], color='gray', alpha=0.5, label='Existing pairs'),
            plt.Line2D([0], [0], color='red', alpha=0.5, label='Recommended pairs')
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.margins(0.2)
        plt.savefig(output_base_path + '_with_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
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
    