"""
Bigram Frequency & Timing Module

This module provides functions for analyzing and visualizing bigram frequency and timing relationships.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
import os
from typing import List, Dict, Any, Tuple

from data_processing import ProcessedData

logger = logging.getLogger(__name__)

def plot_frequency_timing_relationship(
    bigram_data: ProcessedData,
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    output_path: str
) -> Dict[str, Any]:
    """
    Analyze and plot the relationship between bigram frequencies and typing times.
    
    Args:
        bigram_data: ProcessedData object containing bigram pairs and timing data
        bigrams: List of all possible bigrams
        bigram_frequencies_array: Array of frequency values for each bigram
        output_path: Where to save the plot
        
    Returns:
        Dictionary containing correlation statistics and analysis results
    """
    # [rest of implementation stays the same until results]
    
    # Convert frequencies to log scale for additional analysis
    log_frequencies = np.log10(frequencies + 1e-10)  # Add small constant to avoid log(0)
    
    # Calculate log-transformed correlation
    log_correlation, log_p_value = stats.spearmanr(log_frequencies, typing_times)
    
    # Perform linear regression on log-transformed data
    slope, intercept = np.polyfit(log_frequencies, typing_times, 1)
    y_pred = slope * log_frequencies + intercept
    r2 = 1 - np.sum((typing_times - y_pred)**2) / np.sum((typing_times - np.mean(typing_times))**2)
    
    # Prepare results with both raw and log-transformed analysis
    results = {
        'raw_correlation': correlation,
        'raw_p_value': p_value,
        'log_correlation': log_correlation,
        'log_p_value': log_p_value,
        'r2': r2,
        'regression_coefficient': slope,
        'intercept': intercept,
        'n_samples': len(typing_times),
        'mean_time': np.mean(typing_times),
        'std_time': np.std(typing_times),
        'mean_freq': np.mean(frequencies),
        'std_freq': np.std(frequencies)
    }
    
    return results

def plot_timing_by_frequency_groups(
    bigram_data: ProcessedData,
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    n_groups: int = 4,
    output_base_path: str
) -> Dict[str, Any]:
    """
    Analyze typing times across different frequency groups.
    
    Args:
        bigram_data: ProcessedData object containing bigram pairs and timing data
        bigrams: List of all possible bigrams
        bigram_frequencies_array: Array of frequency values for each bigram
        n_groups: Number of frequency groups to create
        output_base_path: Base path for saving visualizations
        
    Returns:
        Dictionary containing group comparison statistics and analysis results
    """
    # [implementation stays the same until results]
    
    # Calculate more detailed group statistics
    group_stats = pd.DataFrame({
        'Size': [len(g) for g in group_times],
        'Mean': [np.mean(g) for g in group_times],
        'Std': [np.std(g) for g in group_times],
        'Min': [np.min(g) for g in group_times],
        'Max': [np.max(g) for g in group_times],
        'Freq_Range_Low': [freq_percentiles[i] for i in range(n_groups)],
        'Freq_Range_High': [freq_percentiles[i+1] for i in range(n_groups)]
    })
    
    results = {
        'group_stats': group_stats,
        'anova_f_stat': f_stat,
        'anova_p_value': p_value
    }
    
    return results

def save_timing_analysis(
    timing_results: Dict[str, Any],
    group_comparison_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save timing and frequency analysis results to a file.
    
    Args:
        timing_results: Dictionary containing timing-frequency correlation statistics
                       from plot_frequency_timing_relationship()
        group_comparison_results: Dictionary containing group comparison statistics
                                from plot_timing_by_frequency_groups()
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
        
        f.write("Summary Statistics:\n")
        f.write(f"Number of samples: {timing_results['n_samples']}\n")
        f.write(f"Mean typing time: {timing_results['mean_time']:.2f} ms\n")
        f.write(f"Std typing time: {timing_results['std_time']:.2f} ms\n")
        f.write(f"Mean frequency: {timing_results['mean_freq']:.2e}\n")
        f.write(f"Std frequency: {timing_results['std_freq']:.2e}\n\n")
        
        # Write group comparison results
        f.write("=== Frequency Group Analysis ===\n\n")
        f.write("Group Statistics:\n")
        f.write(group_comparison_results['group_stats'].to_string())
        f.write("\n\nOne-way ANOVA Results:\n")
        f.write(f"F-statistic: {group_comparison_results['anova_f_stat']:.3f}\n")
        f.write(f"P-value: {group_comparison_results['anova_p_value']:.3e}\n")