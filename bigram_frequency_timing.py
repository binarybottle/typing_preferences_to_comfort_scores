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
    bigram_data: pd.DataFrame,
    bigrams: Dict[str, int],
    bigram_frequencies_array: np.ndarray,
    output_path: str,
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze and plot relationship between bigram frequency and typing time.
    
    Args:
        bigram_data: DataFrame with bigram typing data
        bigrams: Dictionary mapping bigrams to indices
        bigram_frequencies_array: Array of bigram frequencies
        output_path: Path to save the plot
        n_groups: Number of frequency groups for analysis
        
    Returns:
        Dictionary containing correlation results and group analysis
    """
    try:
        # Calculate correlation between frequency and timing
        correlation, p_value = stats.spearmanr(
            bigram_frequencies_array,
            bigram_data['typing_time']
        )
        
        # Fit regression line
        X = bigram_frequencies_array.reshape(-1, 1)
        y = bigram_data['typing_time'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(bigram_frequencies_array, 
                   bigram_data['typing_time'],
                   alpha=0.5)
        
        # Add regression line
        plt.plot(X, reg.predict(X), color='red', 
                label=f'R² = {r2:.3f}')
        
        plt.xlabel('Bigram Frequency')
        plt.ylabel('Typing Time (ms)')
        plt.title('Bigram Frequency vs Typing Time')
        plt.legend()
        
        # Add correlation info
        plt.text(0.05, 0.95, 
                f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze timing groups
        group_results = analyze_timing_groups(
            bigram_data=bigram_data,
            bigram_frequencies_array=bigram_frequencies_array,
            n_groups=n_groups
        )
        
        # Prepare results
        results = {
            'raw_correlation': correlation,
            'raw_p_value': p_value,
            'r2': r2,
            'n_unique_bigrams': len(bigrams),
            'group_analysis': group_results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in frequency timing analysis: {str(e)}")
        return {'error': str(e)}

def plot_timing_by_frequency_groups(
    bigram_data: pd.DataFrame,
    bigram_frequencies_array: np.ndarray,
    group_results: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Create visualizations of timing differences between frequency groups.
    
    Args:
        bigram_data: DataFrame with bigram typing data
        bigram_frequencies_array: Array of bigram frequencies
        group_results: Results from group analysis
        output_dir: Directory to save plots
    """
    try:
        if not group_results:
            logger.warning("No group results available for plotting")
            return
            
        # Create boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=bigram_data, x='freq_group', y='typing_time')
        plt.xlabel('Frequency Group')
        plt.ylabel('Typing Time (ms)')
        plt.title('Typing Time Distribution by Frequency Group')
        
        # Add group size annotations
        for i, stats in group_results['group_stats'].items():
            plt.text(i, plt.ylim()[0], f"n={stats['n_bigrams']}", 
                    ha='center', va='top')
        
        plt.savefig(os.path.join(output_dir, 'timing_by_group_box.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=bigram_data, x='freq_group', y='typing_time')
        plt.xlabel('Frequency Group')
        plt.ylabel('Typing Time (ms)')
        plt.title('Typing Time Distribution by Frequency Group')
        
        plt.savefig(os.path.join(output_dir, 'timing_by_group_violin.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating group plots: {str(e)}")

def save_timing_analysis(
    timing_results: Dict[str, Any],
    group_comparison_results: Optional[Dict[str, Any]],
    output_path: str
) -> None:
    """Save timing analysis results to file."""
    with open(output_path, 'w') as f:
        f.write("Bigram Timing Analysis Results\n")
        f.write("============================\n\n")
        
        # Write correlation results
        f.write("Overall Correlation Analysis:\n")
        f.write("--------------------------\n")
        f.write(f"Raw correlation: {timing_results['raw_correlation']:.3f}")
        f.write(f" (p = {timing_results['raw_p_value']:.3e})\n")
        f.write(f"R-squared: {timing_results['r2']:.3f}\n")
        f.write(f"Number of unique bigrams: {timing_results['n_unique_bigrams']}\n\n")
        
        # Write group comparison results if available
        if group_comparison_results:
            f.write("Group Comparison Analysis:\n")
            f.write("------------------------\n")
            f.write("Frequency Groups:\n")
            
            # Write group statistics
            for group_num, stats in group_comparison_results['group_stats'].items():
                f.write(f"\nGroup {group_num}:\n")
                f.write(f"  Frequency range: {stats['freq_range']}\n")
                f.write(f"  Mean timing: {stats['mean_timing']:.3f} ms\n")
                f.write(f"  Std timing: {stats['std_timing']:.3f} ms\n")
                f.write(f"  Number of bigrams: {stats['n_bigrams']}\n")
            
            # Write ANOVA results
            f.write("\nANOVA Results:\n")
            f.write(f"F-statistic: {group_comparison_results['f_stat']:.3f}\n")
            f.write(f"p-value: {group_comparison_results['p_value']:.3e}\n")
            
            if 'post_hoc' in group_comparison_results:
                f.write("\nPost-hoc Analysis:\n")
                f.write(group_comparison_results['post_hoc'].to_string())
                f.write("\n")
        else:
            f.write("\nNo group comparison analysis results available.\n")
            f.write("Note: Group analysis requires sufficient data in each frequency group.\n")
            f.write("Consider adjusting group parameters or collecting more data.\n")
