"""
Bigram Frequency & Timing Analysis Module

This module analyzes the relationship between bigram frequencies 
and typing times. It handles data where:
1. Input contains pairs of bigrams (chosen vs. unchosen)
2. Each bigram pair has an associated average typing time
3. Each bigram has an associated frequency from a corpus

The analysis ensures each unique bigram is counted only once,
with its timing averaged across all instances where it appears
(whether as chosen or unchosen in the original pairs).
"""
import numpy as np
from scipy import stats
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

from data_processing import ProcessedData

logger = logging.getLogger(__name__)

def plot_frequency_timing_relationship(
    bigram_data: Any,
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    output_path: str
) -> Dict[str, Any]:
    """
    Analyze and plot the relationship between bigram frequencies and typing times.

    Data Structure Handling:
    - Input bigram_data contains pairs of bigrams ((char1,char2), (char3,char4))
    - Each pair has one average typing time (already averaged from chosen/unchosen times)
    - Each unique bigram's final timing is averaged across all its appearances
    
    Args:
        bigram_data: ProcessedData object containing:
            - bigram_pairs: List of tuples, each containing two bigram tuples
            - typing_times: Array of average typing times for each pair
        bigrams: List of all possible bigrams from corpus
        bigram_frequencies_array: Array of frequency values for each bigram
        output_path: Where to save the visualization
        
    Returns:
        Dictionary containing:
            - raw_correlation: Spearman correlation using raw frequencies
            - raw_p_value: P-value for raw correlation
            - log_correlation: Spearman correlation using log frequencies
            - log_p_value: P-value for log correlation
            - r2: R-squared value for log-linear fit
            - regression_coefficient: Slope of log-linear fit
            - intercept: Intercept of log-linear fit
            - n_unique_bigrams: Number of unique bigrams analyzed
            - mean_time: Mean typing time across all bigrams
            - std_time: Standard deviation of typing times
            - mean_freq: Mean frequency across all bigrams
            - std_freq: Standard deviation of frequencies
            
    Example input structure:
        bigram_pairs = [
            (('t','h'), ('f','r')),  # Each pair is two bigrams
            (('i','n'), ('o','n')),
            ...
        ]
        typing_times = [250, 300, ...]  # Already averaged for each pair
    """
    # Create mapping from bigram to its frequency
    bigram_to_freq = {}
    for (b1, b2), freq in zip(bigrams, bigram_frequencies_array):
        bigram_to_freq[(b1, b2)] = freq
        bigram_to_freq[(b2, b1)] = freq  # Store reversed version too
    
    # Dictionary to collect timing data for each unique bigram
    bigram_timing_data = {}
    
    logger.info("\n=== Processing Bigram Pairs ===")
    logger.info(f"Total pairs to process: {len(bigram_data.bigram_pairs)}")
    
    # Debug value ranges
    logger.info("Timing data summary:")
    logger.info(f"Min time: {np.min(bigram_data.typing_times)}")
    logger.info(f"Max time: {np.max(bigram_data.typing_times)}")
    logger.info(f"Mean time: {np.mean(bigram_data.typing_times)}")
    
    # Process each bigram pair
    for i, pair in enumerate(bigram_data.bigram_pairs):
        (b1a, b1b), (b2a, b2b) = pair
        first_bigram = (b1a, b1b)
        second_bigram = (b2a, b2b)
        current_time = bigram_data.typing_times[i]
        
        # Skip invalid timing data
        if not np.isfinite(current_time):
            continue
            
        for bigram in [first_bigram, second_bigram]:
            if bigram in bigram_to_freq:
                if bigram not in bigram_timing_data:
                    bigram_timing_data[bigram] = []
                bigram_timing_data[bigram].append(float(current_time))  # Ensure float
    
    # Compute averages and collect frequency data
    typing_times = []
    frequencies = []
    matched_bigrams = []
    
    for bigram, times in bigram_timing_data.items():
        if times:  # Only process if we have valid times
            avg_time = np.mean(times)
            if np.isfinite(avg_time):  # Only include finite values
                typing_times.append(avg_time)
                frequencies.append(float(bigram_to_freq[bigram]))  # Ensure float
                matched_bigrams.append(bigram)
    
    if not typing_times:
        error_msg = "No valid bigram timing data found"
        logger.error(error_msg)
        return {'error': error_msg, 'n_samples': 0}

    # Convert to numpy arrays and ensure float type
    typing_times = np.array(typing_times, dtype=float)
    frequencies = np.array(frequencies, dtype=float)
    
    # Debug data before correlation
    logger.info("\n=== Data Validation ===")
    logger.info(f"Number of valid pairs: {len(typing_times)}")
    logger.info(f"Any NaN in times: {np.isnan(typing_times).any()}")
    logger.info(f"Any NaN in frequencies: {np.isnan(frequencies).any()}")
    logger.info(f"Time range: [{np.min(typing_times):.2f}, {np.max(typing_times):.2f}]")
    logger.info(f"Frequency range: [{np.min(frequencies):.6f}, {np.max(frequencies):.6f}]")
    
    # Calculate correlations with explicit null checks
    try:
        # Calculate raw correlation
        correlation, p_value = stats.spearmanr(frequencies, typing_times, nan_policy='omit')
        
        # Calculate log-transformed correlation
        log_frequencies = np.log10(frequencies + 1e-10)  # Add small constant to avoid log(0)
        log_correlation, log_p_value = stats.spearmanr(log_frequencies, typing_times, nan_policy='omit')
        
        # Linear regression on log-transformed data
        mask = np.isfinite(log_frequencies) & np.isfinite(typing_times)
        slope, intercept = np.polyfit(log_frequencies[mask], typing_times[mask], 1)
        y_pred = slope * log_frequencies + intercept
        
        # Calculate R-squared
        ss_res = np.sum((typing_times[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((typing_times[mask] - np.mean(typing_times[mask])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(log_frequencies, typing_times, alpha=0.5)
        plt.plot(log_frequencies, y_pred, 'r-', 
                label=f'Linear fit (R² = {r2:.3f})\nSpearman r = {correlation:.3f}')
        plt.xlabel('Log10(Bigram Frequency)')
        plt.ylabel('Average Typing Time (ms)')
        plt.title(f'Bigram Frequency vs Average Typing Time\n(n={len(typing_times)} unique bigrams)')
        plt.legend()
        plt.grid(True)
        
        # Add some sample points with labels
        for i in range(min(5, len(matched_bigrams))):
            bigram = matched_bigrams[i]
            plt.annotate(
                f"{''.join(bigram)}", 
                (log_frequencies[i], typing_times[i]),
                xytext=(5, 5), textcoords='offset points'
            )
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log matching results
        logger.info("\n=== ANALYSIS RESULTS ===")
        logger.info(f"Unique bigrams matched: {len(matched_bigrams)}")
        logger.info(f"Sample of matched bigrams:")
        for i in range(min(5, len(matched_bigrams))):
            bigram = matched_bigrams[i]
            logger.info(f"  {''.join(bigram)}: freq={bigram_to_freq[bigram]:.6f}, "
                       f"avg_time={np.mean(bigram_timing_data[bigram]):.2f}ms, "
                       f"n_observations={len(bigram_timing_data[bigram])}")
        
        # Prepare results
        results = {
            'raw_correlation': float(correlation),
            'raw_p_value': float(p_value),
            'log_correlation': float(log_correlation),
            'log_p_value': float(log_p_value),
            'r2': float(r2),
            'regression_coefficient': float(slope),
            'intercept': float(intercept),
            'n_samples': len(typing_times),
            'n_unique_bigrams': len(matched_bigrams),
            'mean_time': float(np.mean(typing_times)),
            'std_time': float(np.std(typing_times)),
            'mean_freq': float(np.mean(frequencies)),
            'std_freq': float(np.std(frequencies))
        }
        
        logger.info(f"Raw correlation: {correlation:.3f} (p = {p_value:.3e})")
        logger.info(f"Log correlation: {log_correlation:.3f} (p = {log_p_value:.3e})")
        logger.info(f"R-squared: {r2:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in correlation calculation: {str(e)}")
        logger.error("Dumping data summaries for debugging:")
        logger.error(f"Typing times summary: {pd.Series(typing_times).describe()}")
        logger.error(f"Frequencies summary: {pd.Series(frequencies).describe()}")
        return {
            'error': f"Correlation calculation failed: {str(e)}",
            'n_samples': len(typing_times)
        }
    
def plot_timing_by_frequency_groups(
    bigram_data: ProcessedData,
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    output_base_path: str,
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze typing times across different frequency groups.
    
    Args:
        bigram_data: ProcessedData object containing bigram pairs and timing data
        bigrams: List of all possible bigrams
        bigram_frequencies_array: Array of frequency values for each bigram
        output_base_path: Base path for saving visualizations
        n_groups: Number of frequency groups to create (min: 2, default: 4)
        
    Returns:
        Dictionary containing group comparison statistics and analysis results
    """
    # Validate n_groups parameter
    if not isinstance(n_groups, int):
        raise ValueError(f"n_groups must be an integer, got {type(n_groups)}")
    if n_groups < 2:
        logger.warning(f"n_groups={n_groups} is too small, setting to minimum value of 2")
        n_groups = 2
    if n_groups > 10:
        logger.warning(f"n_groups={n_groups} is quite large, this may result in sparse groups")
    
    # Create mapping from bigram to its frequency (including reversed pairs)
    bigram_to_freq = {}
    for (b1, b2), freq in zip(bigrams, bigram_frequencies_array):
        bigram_to_freq[(b1, b2)] = freq
        bigram_to_freq[(b2, b1)] = freq
    
    # Extract timing data and corresponding frequencies
    timing_data = []
    freq_data = []
    
    for i, pair in enumerate(bigram_data.bigram_pairs):
        if pair in bigram_to_freq:
            timing_data.append(bigram_data.typing_times[i])
            freq_data.append(bigram_to_freq[pair])
    
    if not timing_data:
        logger.error("No matching bigrams found between input data and frequency data")
        return {
            'error': 'No matching bigrams found',
            'n_samples': 0
        }
    
    # Convert to numpy arrays
    timing_data = np.array(timing_data)
    freq_data = np.array(freq_data)
    
    # Calculate frequency percentiles for grouping
    freq_percentiles = np.percentile(freq_data, np.linspace(0, 100, n_groups + 1))
    
    # Group timing data by frequency
    group_times = []
    group_freqs = []
    for i in range(n_groups):
        mask = (freq_data >= freq_percentiles[i]) & (freq_data <= freq_percentiles[i + 1])
        group_times.append(timing_data[mask])
        group_freqs.append(freq_data[mask])
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*group_times)
    
    # Calculate group statistics
    group_stats = pd.DataFrame({
        'Size': [len(g) for g in group_times],
        'Mean_Time': [np.mean(g) for g in group_times],
        'Std_Time': [np.std(g) for g in group_times],
        'Mean_Freq': [np.mean(f) for f in group_freqs],
        'Min_Time': [np.min(g) if len(g) > 0 else np.nan for g in group_times],
        'Max_Time': [np.max(g) if len(g) > 0 else np.nan for g in group_times],
        'Freq_Range_Low': freq_percentiles[:-1],
        'Freq_Range_High': freq_percentiles[1:]
    })
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    plt.boxplot(group_times, labels=[f'G{i+1}' for i in range(n_groups)])
    plt.title(f'Typing Times by Frequency Group\n(n={len(timing_data)} pairs)')
    plt.ylabel('Typing Time (ms)')
    plt.xlabel('Frequency Group')
    
    # Mean plot with error bars
    plt.subplot(1, 2, 2)
    means = [np.mean(g) for g in group_times]
    stds = [np.std(g) for g in group_times]
    plt.errorbar(range(1, n_groups + 1), means, yerr=stds, fmt='o-')
    plt.title('Mean Typing Time by Group')
    plt.ylabel('Mean Typing Time (ms)')
    plt.xlabel('Frequency Group')
    
    plt.tight_layout()
    plt.savefig(f"{output_base_path}/group_analysis_{n_groups}_groups.png")
    plt.close()
    
    # Save detailed group statistics
    group_stats.to_csv(f"{output_base_path}/group_stats_{n_groups}_groups.csv")
    
    results = {
        'group_stats': group_stats,
        'anova_f_stat': f_stat,
        'anova_p_value': p_value,
        'n_groups': n_groups,
        'total_samples': len(timing_data),
        'output_files': {
            'plot': f"group_analysis_{n_groups}_groups.png",
            'stats': f"group_stats_{n_groups}_groups.csv"
        }
    }
    
    return results

def save_timing_analysis(
    timing_results: Optional[Dict[str, Any]],
    group_comparison_results: Optional[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save timing and frequency analysis results to a file.
    Handles cases where only one type of analysis is available or where analyses failed.
    
    Args:
        timing_results: Dictionary containing timing-frequency correlation statistics
                       from plot_frequency_timing_relationship(), or None
        group_comparison_results: Dictionary containing group comparison statistics
                                from plot_timing_by_frequency_groups(), or None
        output_path: Path to save the analysis results
    """
    with open(output_path, 'w') as f:
        # Write timing-frequency relationship results if available
        f.write("=== Timing-Frequency Relationship Analysis ===\n\n")
        if timing_results is None:
            f.write("No timing-frequency analysis results available.\n\n")
        elif 'error' in timing_results:
            f.write(f"Analysis Error: {timing_results['error']}\n")
            f.write(f"Number of samples: {timing_results.get('n_samples', 0)}\n\n")
        else:
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
        
        # Write group comparison results if available
        f.write("=== Frequency Group Analysis ===\n\n")
        if group_comparison_results is None:
            f.write("No group comparison analysis results available.\n")
        elif 'error' in group_comparison_results:
            f.write(f"Analysis Error: {group_comparison_results['error']}\n")
            f.write(f"Number of samples: {group_comparison_results.get('n_samples', 0)}\n")
        else:
            f.write("Group Statistics:\n")
            f.write(group_comparison_results['group_stats'].to_string())
            f.write("\n\nOne-way ANOVA Results:\n")
            f.write(f"F-statistic: {group_comparison_results['anova_f_stat']:.3f}\n")
            f.write(f"P-value: {group_comparison_results['anova_p_value']:.3e}\n")