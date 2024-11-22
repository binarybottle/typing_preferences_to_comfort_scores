"""
Bigram Frequency & Timing Module

This module provides functions for analyzing and visualizing bigram frequency and timing relationships.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import logging

from bigram_feature_definitions import qwerty_bigram_frequency

logger = logging.getLogger(__name__)

def plot_frequency_timing_relationship(bigram_data: pd.DataFrame,
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

def save_timing_analysis(timing_results: Dict,
                        group_comparison_results: Dict,
                        output_path: str) -> None:
    """
    Save timing and frequency analysis results to a file.
    
    Args:
        timing_results: Dictionary containing timing-frequency correlation statistics
        group_comparison_results: Dictionary containing group comparison statistics
        output_path: Path to save the analysis results
    """
    with open(output_path, 'w') as f:
        # Write timing-frequency relationship results
        f.write("=== Timing-Frequency Relationship Analysis ===\n\n")
        f.write("Raw Correlation Analysis:\n")
        f.write(f"Correlation coefficient: {timing_results['raw_correlation']:.3f}\n")
        f.write(f"P-value: {timing_results['raw_p_value']:.3e}\n\n")
        
        f.write("Log-transformed Analysis:\n")
        f.write(f"Correlation coefficient: {timing_results['log_correlation']:.3f}\n")
        f.write(f"R-squared: {timing_results['r2']:.3f}\n")
        f.write(f"P-value: {timing_results['log_p_value']:.3e}\n")
        f.write(f"Regression coefficient: {timing_results['regression_coefficient']:.3f}\n")
        f.write(f"Intercept: {timing_results['intercept']:.3f}\n\n")
        
        # Write group comparison results
        f.write("=== Frequency Group Analysis ===\n\n")
        f.write("Group Statistics:\n")
        f.write(group_comparison_results['group_stats'].to_string())
        f.write("\n\nOne-way ANOVA Results:\n")
        f.write(f"F-statistic: {group_comparison_results['anova_f_stat']:.3f}\n")
        f.write(f"P-value: {group_comparison_results['anova_p_value']:.3e}\n")
