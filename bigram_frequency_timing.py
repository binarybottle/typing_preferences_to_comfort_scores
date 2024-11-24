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
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import logging
import os

from data_processing import ProcessedData

logger = logging.getLogger(__name__)

def plot_frequency_timing_relationship(
    bigram_data: 'ProcessedData',
    bigrams: list,
    bigram_frequencies_array: np.ndarray,
    output_path: str,
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze and plot relationship between bigram frequency and typing time.
    """
    try:
        # Create dictionary mapping bigrams to their frequencies
        bigram_frequencies = {
            b: freq for b, freq in zip(bigrams, bigram_frequencies_array)
        }
        
        # Create DataFrame with both chosen and unchosen bigrams and their timing
        timing_data = []
        
        # Process both chosen and unchosen bigrams
        for i, bigram_tuple in enumerate(bigram_data.bigram_pairs):
            chosen_bigram = ''.join(bigram_tuple[0])
            unchosen_bigram = ''.join(bigram_tuple[1])
            timing = bigram_data.typing_times[i]
            
            # Add both bigrams with their timing
            timing_data.append({'bigram': chosen_bigram, 'timing': timing})
            timing_data.append({'bigram': unchosen_bigram, 'timing': timing})
        
        df = pd.DataFrame(timing_data)
        
        # Calculate timing statistics for each unique bigram
        aggregated_timings = df.groupby('bigram').agg({
            'timing': ['median', 'mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        aggregated_timings.columns = ['bigram', 'median_timing', 'mean_timing', 
                                    'std_timing', 'n_occurrences']

        # Add frequencies and create frequency groups
        aggregated_timings['frequency'] = aggregated_timings['bigram'].map(bigram_frequencies)
        aggregated_timings = aggregated_timings.dropna(subset=['frequency'])
        aggregated_timings['freq_group'] = pd.qcut(
            aggregated_timings['frequency'], 
            n_groups, 
            labels=[f"Group {i+1}" for i in range(n_groups)]
        )
        
        # Calculate correlation between frequency and timing
        correlation, p_value = stats.spearmanr(
            aggregated_timings['frequency'],
            aggregated_timings['median_timing']
        )
        
        # Fit regression line
        X = aggregated_timings['frequency'].values.reshape(-1, 1)
        y = aggregated_timings['median_timing'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(aggregated_timings['frequency'], 
                   aggregated_timings['median_timing'],
                   alpha=0.5,
                   s=aggregated_timings['n_occurrences'] * 10)  # Size by occurrence count
        
        # Add regression line
        plt.plot(X, reg.predict(X), color='red', 
                label=f'R² = {r2:.3f}')
        
        plt.xlabel('Bigram Frequency')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Bigram Frequency vs Median Typing Time')
        plt.legend()
        
        # Add correlation info
        plt.text(0.05, 0.95, 
                f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create directory for additional plots if it doesn't exist
        output_dir = os.path.dirname(output_path)
        
        # Generate additional visualizations
        plot_timing_by_frequency_groups(
            aggregated_data=aggregated_timings,
            output_dir=output_dir
        )
        
        # Analyze groups
        group_stats = {}
        for group_label in aggregated_timings['freq_group'].unique():
            group_data = aggregated_timings[aggregated_timings['freq_group'] == group_label]
            if len(group_data) > 0:
                group_stats[group_label] = {
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'mean_timing': float(group_data['median_timing'].mean()),
                    'std_timing': float(group_data['median_timing'].std()),
                    'n_bigrams': len(group_data),
                    'total_occurrences': int(group_data['n_occurrences'].sum())
                }
        
        # Perform ANOVA on median timings
        if len(group_stats) >= 2:
            groups = [group['median_timing'].values for _, group in 
                    aggregated_timings.groupby('freq_group', observed=True) if len(group) > 0]
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            f_stat, p_value = None, None
            
        # Prepare results
        results = {
            'correlation': correlation,
            'correlation_p_value': p_value,
            'r2': r2,
            'n_unique_bigrams': len(aggregated_timings),
            'group_stats': group_stats,
            'anova_f_stat': f_stat,
            'anova_p_value': p_value,
            'total_occurrences': int(aggregated_timings['n_occurrences'].sum())
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in frequency timing analysis: {str(e)}")
        return {'error': str(e)}
    
def plot_timing_by_frequency_groups(
    aggregated_data: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Create visualizations of timing differences between frequency groups.
    """
    try:
        # Create boxplot of median timings by frequency group
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=aggregated_data, x='freq_group', y='median_timing')
        plt.xlabel('Frequency Group')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Median Typing Times by Frequency Group')
        
        # Add group size annotations
        for i in range(len(aggregated_data['freq_group'].unique())):
            group_data = aggregated_data[aggregated_data['freq_group'] == i]
            n_bigrams = len(group_data)
            n_occurrences = group_data['n_occurrences'].sum()
            plt.text(i, plt.ylim()[0], 
                    f"n={n_bigrams}\n(occ={n_occurrences})", 
                    ha='center', va='top')
        
        plt.savefig(os.path.join(output_dir, 'timing_by_group_box.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create violin plot for more detailed distribution view
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=aggregated_data, x='freq_group', y='median_timing')
        plt.xlabel('Frequency Group')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Median Typing Times by Frequency Group')
        
        plt.savefig(os.path.join(output_dir, 'timing_by_group_violin.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating group plots: {str(e)}")

def analyze_timing_groups(
    bigram_data: 'ProcessedData',
    bigram_frequencies: Dict[str, float],
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze timing differences between frequency groups.
    """
    try:
        # First create a DataFrame with all bigram occurrences
        timing_data = []
        
        # Process both chosen and unchosen bigrams from each pair
        for i, bigram_tuple in enumerate(bigram_data.bigram_pairs):
            chosen_bigram = ''.join(bigram_tuple[0])
            unchosen_bigram = ''.join(bigram_tuple[1])
            timing = bigram_data.typing_times[i]
            
            # Add both bigrams with their timing
            timing_data.append({'bigram': chosen_bigram, 'timing': timing})
            timing_data.append({'bigram': unchosen_bigram, 'timing': timing})
        
        # Create DataFrame and aggregate timings by bigram
        df = pd.DataFrame(timing_data)
        aggregated_timings = df.groupby('bigram').agg({
            'timing': ['median', 'mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        aggregated_timings.columns = ['bigram', 'median_timing', 'mean_timing', 
                                    'std_timing', 'n_occurrences']
        
        # Add frequencies for bigrams that have frequency data
        aggregated_timings['frequency'] = aggregated_timings['bigram'].map(bigram_frequencies)
        
        # Remove bigrams without frequency data
        aggregated_timings = aggregated_timings.dropna(subset=['frequency'])
        
        # Create frequency groups
        aggregated_timings['freq_group'] = pd.qcut(
            aggregated_timings['frequency'], 
            n_groups, 
            labels=False
        )
        
        # Calculate group statistics
        group_stats = {}
        for group in range(n_groups):
            group_data = aggregated_timings[aggregated_timings['freq_group'] == group]
            if len(group_data) > 0:
                group_stats[group] = {
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'mean_timing': float(group_data['median_timing'].mean()),
                    'std_timing': float(group_data['median_timing'].std()),
                    'n_bigrams': len(group_data),
                    'total_occurrences': int(group_data['n_occurrences'].sum())
                }
        
        # Perform ANOVA on median timings if we have enough groups
        if len(group_stats) >= 2:
            groups = [group['median_timing'].values for _, group in 
                     aggregated_timings.groupby('freq_group') if len(group) > 0]
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            f_stat, p_value = None, None
        
        return {
            'group_stats': group_stats,
            'f_stat': float(f_stat) if f_stat is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
            'n_unique_bigrams': len(aggregated_timings),
            'total_occurrences': int(aggregated_timings['n_occurrences'].sum()),
            'aggregated_data': aggregated_timings
        }
        
    except Exception as e:
        logger.error(f"Error in timing group analysis: {str(e)}")
        return {
            'error': str(e),
            'trace': str(e.__traceback__)
        }
    
def save_timing_analysis(
    timing_results: Dict[str, Any],
    group_comparison_results: Optional[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save timing analysis results to file.
    """
    try:
        with open(output_path, 'w') as f:
            f.write("Bigram Timing Analysis Results\n")
            f.write("============================\n\n")
            
            # Write correlation results
            f.write("Overall Correlation Analysis:\n")
            f.write("--------------------------\n")
            f.write(f"Correlation: {timing_results['correlation']:.3f}")
            f.write(f" (p = {timing_results['correlation_p_value']:.3e})\n")            
            f.write(f"R-squared: {timing_results['r2']:.3f}\n")
            f.write(f"Number of unique bigrams: {timing_results['n_unique_bigrams']}\n\n")
            
            # Write group comparison results if available
            if timing_results.get('group_stats'):
                f.write("Group Comparison Analysis:\n")
                f.write("------------------------\n")
                f.write("Frequency Groups:\n")
                
                # Write group statistics
                for group_label, stats in timing_results['group_stats'].items():
                    f.write(f"\n{group_label}:\n")
                    f.write(f"  Frequency range: {stats['freq_range']}\n")
                    f.write(f"  Mean timing: {stats['mean_timing']:.3f} ms\n")
                    f.write(f"  Std timing: {stats['std_timing']:.3f} ms\n")
                    f.write(f"  Number of bigrams: {stats['n_bigrams']}\n")
                    f.write(f"  Total occurrences: {stats['total_occurrences']}\n")
                
                # Write ANOVA results if available
                if timing_results.get('anova_f_stat') is not None:
                    f.write("\nANOVA Results:\n")
                    f.write(f"F-statistic: {timing_results['anova_f_stat']:.3f}\n")
                    f.write(f"p-value: {timing_results['anova_p_value']:.3e}\n")
            else:
                f.write("\nNo group comparison analysis results available.\n")
                f.write("Note: Group analysis requires sufficient data in each frequency group.\n")
                f.write("Consider adjusting group parameters or collecting more data.\n")
                
    except Exception as e:
        logger.error(f"Error saving timing analysis: {str(e)}")
