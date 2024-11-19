"""
Visualization Module

This module provides functions for visualizing keyboard layout analysis results,
including feature space analysis, timing relationships, and model diagnostics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import arviz as az
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
import logging

from bigram_features import qwerty_bigram_frequency

logger = logging.getLogger(__name__)

def set_plotting_style():
    """Set consistent style for all plots."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def plot_timing_frequency_relationship(bigram_data: pd.DataFrame,
                                     bigrams: List[str],
                                     bigram_frequencies_array: np.ndarray,
                                     output_path: str) -> Dict:
    """
    Plot relationship between normalized bigram frequency and typing time.
    
    Args:
        bigram_data: DataFrame containing bigram timing data
        bigrams: List of bigrams ordered by frequency
        bigram_frequencies_array: Array of corresponding frequency values
        output_path: Path to save the plot
        
    Returns:
        Dictionary containing correlation statistics
    """
    # Extract chosen bigrams and their timing
    frequencies = []
    timings = []
    bigram_list = []  # To track which bigrams we're analyzing
    
    # Process each row
    for _, row in bigram_data.iterrows():
        try:
            chosen_bigram = row['chosen_bigram'].lower().strip()
            
            if len(chosen_bigram) == 2:  # Ensure it's a valid bigram
                freq = qwerty_bigram_frequency(chosen_bigram[0], chosen_bigram[1], 
                                             bigrams, bigram_frequencies_array)
                frequencies.append(freq)
                timings.append(row['chosen_bigram_time'])
                bigram_list.append(chosen_bigram)
                
        except Exception as e:
            logger.error(f"Error processing row - chosen_bigram: {row.get('chosen_bigram', 'N/A')}, "
                      f"error: {str(e)}")
            continue
    
    frequencies = np.array(frequencies)
    timings = np.array(timings)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(frequencies) | np.isnan(timings))
    frequencies = frequencies[valid_mask]
    timings = timings[valid_mask]
    bigram_list = [b for i, b in enumerate(bigram_list) if valid_mask[i]]
    
    plt.figure(figsize=(12, 5))
    
    # Raw frequency plot
    plt.subplot(1, 2, 1)
    plt.scatter(frequencies, timings, alpha=0.5, label='Data points')
    plt.xlabel('Normalized Bigram Frequency')
    plt.ylabel('Typing Time (ms)')
    plt.title('Raw Frequency vs Typing Time')
    
    # Add labels for extreme points
    n_labels = 5
    extreme_indices = np.argsort(frequencies)[-n_labels:]
    for idx in extreme_indices:
        plt.annotate(bigram_list[idx], 
                    (frequencies[idx], timings[idx]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Calculate raw correlation
    raw_correlation, raw_p_value = stats.pearsonr(frequencies, timings)
    
    plt.text(0.05, 0.95,
             f'r: {raw_correlation:.3f}\np: {raw_p_value:.3e}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Log frequency plot
    plt.subplot(1, 2, 2)
    log_frequencies = np.log10(frequencies + 1e-10)
    plt.scatter(log_frequencies, timings, alpha=0.5, label='Data points')
    
    # Add regression line
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    reg = LinearRegression().fit(log_frequencies.reshape(-1, 1), timings)
    x_range = np.linspace(log_frequencies.min(), log_frequencies.max(), 100)
    plt.plot(x_range,
             reg.predict(x_range.reshape(-1, 1)),
             color='red',
             label='Regression line')
    
    plt.xlabel('Log10(Normalized Frequency)')
    plt.ylabel('Typing Time (ms)')
    plt.title('Log Frequency vs Typing Time')
    plt.legend()
    
    # Calculate correlations for log-transformed data
    log_correlation, log_p_value = stats.pearsonr(log_frequencies, timings)
    r2 = r2_score(timings, reg.predict(log_frequencies.reshape(-1, 1)))
    
    plt.text(0.05, 0.95,
             f'r: {log_correlation:.3f}\nR²: {r2:.3f}\np: {log_p_value:.3e}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return {
        'raw_correlation': raw_correlation,
        'raw_p_value': raw_p_value,
        'log_correlation': log_correlation,
        'log_p_value': log_p_value,
        'r2': r2,
        'regression_coefficient': reg.coef_[0],
        'intercept': reg.intercept_
    }

def plot_timing_by_frequency_groups(bigram_data: pd.DataFrame,
                                  bigrams: List[str],
                                  bigram_frequencies_array: np.ndarray,
                                  n_groups: int = 4,
                                  output_base_path: str = "output/timing_groups") -> Dict:
    """
    Create violin and box plots comparing typing times across frequency groups.
    
    Args:
        bigram_data: DataFrame containing timing data and bigram pairs
        bigrams: List of bigrams ordered by frequency
        bigram_frequencies_array: Array of corresponding frequency values
        n_groups: Number of frequency groups to create
        output_base_path: Base path for saving plots
        
    Returns:
        Dictionary containing group statistics and ANOVA results
    """
    # First get frequencies for each chosen bigram
    freqs = []
    times = []
    bigram_texts = []
    
    for _, row in bigram_data.iterrows():
        chosen_bigram = row['chosen_bigram'].lower().strip()
        if len(chosen_bigram) == 2:
            freq = qwerty_bigram_frequency(chosen_bigram[0], chosen_bigram[1], 
                                           bigrams, bigram_frequencies_array)
            freqs.append(freq)
            times.append(row['chosen_bigram_time'])
            bigram_texts.append(chosen_bigram)
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'bigram': bigram_texts,
        'frequency': freqs,
        'timing': times,
    })
    
    # Create frequency groups
    analysis_df['frequency_group'] = pd.qcut(analysis_df['frequency'], n_groups, 
                                           labels=['Very Low', 'Low', 'High', 'Very High'])

    # Calculate summary statistics by group (fix pandas warning)
    group_stats = analysis_df.groupby('frequency_group', observed=True)['timing'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    # Add frequency ranges for each group (fix pandas warning)
    freq_ranges = analysis_df.groupby('frequency_group', observed=True)['frequency'].agg(['min', 'max']).round(4)
    group_stats['freq_range'] = freq_ranges.apply(lambda x: f"{x['min']:.4f} - {x['max']:.4f}", axis=1)
    
    # Perform one-way ANOVA (fix pandas warning)
    groups = [group['timing'].values for name, group in analysis_df.groupby('frequency_group', observed=True)]
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Box plot (fix matplotlib warning)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=analysis_df, x='frequency_group', y='timing')
    plt.title('Typing Time Distribution by Frequency Group')
    plt.xlabel('Frequency Group')
    plt.ylabel('Typing Time (ms)')
    
    # Add sample sizes to x-axis labels (fix matplotlib warning)
    sizes = analysis_df['frequency_group'].value_counts()
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick.get_text()}\n(n={sizes[tick.get_text()]})' 
                        for tick in ax.get_xticklabels()])
    
    plt.savefig(f'{output_base_path}_boxplot.png')
    plt.close()
    
    # Violin plot (fix matplotlib warning)
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=analysis_df, x='frequency_group', y='timing')
    plt.title('Typing Time Distribution by Frequency Group')
    plt.xlabel('Frequency Group')
    plt.ylabel('Typing Time (ms)')
    
    # Add sample sizes to x-axis labels (fix matplotlib warning)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{tick.get_text()}\n(n={sizes[tick.get_text()]})' 
                        for tick in ax.get_xticklabels()])
    
    plt.savefig(f'{output_base_path}_violin.png')
    plt.close()
    
    return {
        'group_stats': group_stats,
        'anova_f_stat': f_stat,
        'anova_p_value': p_value
    }

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

def analyze_feature_space(feature_matrix: pd.DataFrame,
                         output_dir: str,
                         all_feature_differences: Dict,  # Add this parameter
                         check_multicollinearity: bool = True,
                         recommend_bigrams: bool = True,
                         num_recommendations: int = 30) -> Dict:
    """
    Comprehensive feature space analysis including multicollinearity and recommendations.
    
    Args:
        feature_matrix: DataFrame of features
        output_dir: Directory for output files
        check_multicollinearity: Whether to check for multicollinearity
        recommend_bigrams: Whether to generate bigram recommendations
        num_recommendations: Number of bigram pairs to recommend
    """ 
    results = {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check multicollinearity if requested
    if check_multicollinearity:
        multicollinearity_results = check_multicollinearity_vif(feature_matrix)
        results['multicollinearity'] = multicollinearity_results
        
        # Save VIF results
        vif_df = pd.DataFrame(multicollinearity_results['vif'])
        vif_df.to_csv(f"{output_dir}/vif_results.csv", index=False)
        
        # Save correlation results
        if multicollinearity_results['high_correlations']:
            corr_df = pd.DataFrame(multicollinearity_results['high_correlations'])
            corr_df.to_csv(f"{output_dir}/high_correlations.csv", index=False)
    
    # Standardize features and perform PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Plot PCA projection
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.title('2D PCA projection of feature space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(f'{output_dir}/feature_space_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate feature space metrics
    hull = ConvexHull(pca_result)
    hull_area = hull.area
    point_density = len(pca_result) / hull_area
    
    results['feature_space_metrics'] = {
        'hull_area': hull_area,
        'point_density': point_density
    }
    
    # Identify underrepresented areas
    grid_points, distances = identify_underrepresented_areas(
        pca_result,
        output_path=f'{output_dir}/underrepresented_areas.png'
    )
    
    # Generate recommendations if requested
    if recommend_bigrams:
        recommendations = generate_bigram_recommendations(
            pca_result=pca_result,
            scaler=scaler,
            pca=pca,
            grid_points=grid_points,
            distances=distances,
            all_feature_differences=all_feature_differences,  # Pass this through
            num_recommendations=num_recommendations
        )
        results['recommendations'] = recommendations
        
        # Save recommendations
        with open(f"{output_dir}/bigram_recommendations.txt", 'w') as f:
            f.write("Recommended bigram pairs to collect data for:\n\n")
            for i, pair in enumerate(recommendations, 1):
                f.write(f"{i}. {pair[0][0]}{pair[0][1]} vs {pair[1][0]}{pair[1][1]}\n")
    
    return results

def check_multicollinearity_vif(feature_matrix: pd.DataFrame) -> Dict:
    """Check for multicollinearity using Variance Inflation Factor."""
    results = {
        'vif': [],
        'high_correlations': []
    }
    
    # Add constant term
    X = sm.add_constant(feature_matrix)
    
    # Calculate VIF for each feature
    for i, column in enumerate(X.columns):
        if column != 'const':
            try:
                vif = variance_inflation_factor(X.values, i)
                results['vif'].append({
                    'Feature': column,
                    'VIF': vif,
                    'Status': 'High multicollinearity' if vif > 5 else 'Acceptable'
                })
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {column}: {str(e)}")
    
    # Calculate correlation matrix
    corr_matrix = feature_matrix.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:  # Threshold for high correlation
                results['high_correlations'].append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    return results

def identify_underrepresented_areas(pca_result: np.ndarray,
                                  output_path: str,
                                  num_grid: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Identify and visualize underrepresented areas in feature space."""
    x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
    y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_grid),
                        np.linspace(y_min, y_max, num_grid))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances = cdist(grid_points, pca_result).min(axis=1)
    distances = distances.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(distances, extent=[x_min, x_max, y_min, y_max],
               origin='lower', cmap='viridis')
    plt.colorbar(label='Distance to nearest data point')
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='red', s=20, alpha=0.5)
    plt.title('Underrepresented Areas in Feature Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
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
    
    # Select the grid points with the largest distances
    selected_points = grid_points[sorted_indices[:num_recommendations]]
    
    # Transform back to original feature space
    original_space_points = scaler.inverse_transform(pca.inverse_transform(selected_points))
    
    # Round and take absolute values
    feature_points = np.abs(np.round(original_space_points))
    
    # Convert feature points to bigram pairs
    all_pairs = list(all_feature_differences.keys())
    all_values = np.array(list(all_feature_differences.values()))
    
    # Find closest existing bigram pairs to our generated points
    distances_to_existing = cdist(feature_points, all_values)
    closest_indices = np.argmin(distances_to_existing, axis=1)
    
    # Get the corresponding bigram pairs
    recommended_pairs = []
    seen = set()
    
    for idx in closest_indices:
        pair = all_pairs[idx]
        # Convert to a hashable format for deduplication
        pair_tuple = tuple(sorted([tuple(pair[0]), tuple(pair[1])]))
        if pair_tuple not in seen:
            seen.add(pair_tuple)
            recommended_pairs.append(pair)
            if len(recommended_pairs) >= num_recommendations:
                break
    
    return recommended_pairs

def plot_bigram_graph(bigram_pairs: List[Tuple[str, str]],
                      output_path: str) -> None:
    """
    Plot a graph showing bigram connectivity.
    
    Args:
        bigram_pairs: List of bigram pairs
        output_path: Path to save the plot
    """
    # Create graph
    G = nx.Graph()
    
    # Add edges
    for bigram1, bigram2 in bigram_pairs:
        bigram1_str = ''.join(bigram1)
        bigram2_str = ''.join(bigram2)
        G.add_edge(bigram1_str, bigram2_str)
    
    # Get connected components
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    plt.figure(figsize=(14, 14))
    
    # Layout positioning
    pos = {}
    grid_size = int(np.ceil(np.sqrt(len(components))))
    spacing = 5.0
    
    # Position components
    for i, component in enumerate(components):
        component_pos = nx.spring_layout(component, k=1.0, seed=i)
        
        x_offset = (i % grid_size) * spacing
        y_offset = (i // grid_size) * spacing
        
        for node in component_pos:
            component_pos[node][0] += x_offset
            component_pos[node][1] += y_offset
            
        pos.update(component_pos)
    
    # Draw graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            font_weight='bold', node_size=500, font_size=14,
            edge_color='gray', linewidths=1.5, width=2.0)
    
    plt.title("Bigram Connectivity Graph", fontsize=20)
    plt.savefig(output_path)
    plt.close()

def plot_model_diagnostics(trace: az.InferenceData,
                         output_base_path: str,
                         inference_method: str = "mcmc") -> None:
    """
    Create diagnostic plots for the Bayesian model.
    
    Args:
        trace: ArviZ InferenceData object
        output_base_path: Base path for saving plots
        inference_method: Type of inference used ('mcmc' or 'variational')
    """
    # Check what groups are available in the trace
    available_groups = list(trace.groups())
    logger.info(f"Available groups in trace: {available_groups}")

    # Get variables excluding participant-specific ones
    var_names = [v for v in trace.posterior.data_vars 
                 if not v.startswith('participant_')]
    logger.info(f"Plotting diagnostics for {len(var_names)} variables")

    # Plot main variables separately
    main_vars = ['row_sum', 'engram_sum', 'freq', 'sigma']
    try:
        # Posterior plots for main variables
        az.plot_posterior(trace, var_names=main_vars)
        plt.suptitle(f'Posterior Distributions - Main Variables ({inference_method})')
        plt.tight_layout()
        plt.savefig(f'{output_base_path}_posterior_main.png')
        plt.close()

        # Forest plot for main variables
        az.plot_forest(trace, var_names=main_vars)
        plt.suptitle(f'Forest Plot - Main Variables ({inference_method})')
        plt.tight_layout()
        plt.savefig(f'{output_base_path}_forest_main.png')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create main variable plots: {str(e)}")

    # Participant effects summary
    try:
        participant_vars = [v for v in trace.posterior.data_vars 
                          if v.startswith('participant_')]
        if participant_vars:
            # Create summary statistics
            participant_summary = az.summary(trace, var_names=participant_vars)
            participant_summary.to_csv(f'{output_base_path}_participant_effects.csv')
            
            # Plot distribution
            plt.figure(figsize=(10, 6))
            means = [participant_summary.loc[var, 'mean'] for var in participant_vars]
            plt.hist(means, bins=30)
            plt.title(f'Distribution of Participant Effects ({inference_method})')
            plt.xlabel('Effect Size')
            plt.ylabel('Count')
            plt.savefig(f'{output_base_path}_participant_effects_dist.png')
            plt.close()
    except Exception as e:
        logger.warning(f"Could not create participant effects plots: {str(e)}")

    # Create summary statistics
    try:
        summary = az.summary(trace)
        summary.to_csv(f'{output_base_path}_summary.csv')
    except Exception as e:
        logger.warning(f"Could not create summary statistics: {str(e)}")

    # MCMC-specific plots
    if inference_method == 'mcmc' and 'sample_stats' in available_groups:
        try:
            # Trace plot for main variables
            az.plot_trace(trace, var_names=main_vars)
            plt.suptitle('Trace Plots - Main Variables')
            plt.tight_layout()
            plt.savefig(f'{output_base_path}_trace_main.png')
            plt.close()

            # Energy plot if available
            try:
                az.plot_energy(trace)
                plt.suptitle('Energy Plot')
                plt.tight_layout()
                plt.savefig(f'{output_base_path}_energy.png')
                plt.close()
            except Exception as e:
                logger.warning(f"Could not create energy plot: {str(e)}")

            # Pair plot for main variables
            az.plot_pair(trace, var_names=main_vars)
            plt.suptitle('Pair Plot - Main Variables')
            plt.tight_layout()
            plt.savefig(f'{output_base_path}_pair_main.png')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create MCMC-specific plots: {str(e)}")

    # Variational inference specific diagnostics
    elif inference_method == 'variational':
        try:
            # Could add VI-specific diagnostics here if needed
            logger.info("No additional VI-specific diagnostics implemented")
            pass
        except Exception as e:
            logger.warning(f"Could not create VI-specific plots: {str(e)}")

    logger.info(f"Diagnostic plots creation completed for {inference_method} inference")

def plot_sensitivity_analysis(parameter_estimates: Dict[str, pd.DataFrame],
                              design_features: List[str],
                              control_features: List[str],
                              output_path: str) -> None:
    """
    Create visualization of sensitivity analysis results.
    
    Args:
        parameter_estimates: Dictionary of parameter estimates
        design_features: List of design features
        control_features: List of control features
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Design features plot
    plt.subplot(2, 1, 1)
    data = []
    labels = []
    for feature in design_features:
        means = [est.loc[feature, 'mean'] for est in parameter_estimates.values()]
        data.append(means)
        labels.append(feature)
    
    plt.boxplot(data, labels=labels)
    plt.title('Design Feature Effect Estimates')
    plt.ylabel('Effect Size')
    plt.grid(True, alpha=0.3)
    
    # Control features plot
    if control_features:
        plt.subplot(2, 1, 2)
        data = []
        labels = []
        for feature in control_features:
            means = [est.loc[feature, 'mean'] for est in parameter_estimates.values()]
            data.append(means)
            labels.append(feature)
        
        plt.boxplot(data, labels=labels)
        plt.title('Control Feature Effect Estimates')
        plt.ylabel('Effect Size')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
