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
    bigram_data: pd.DataFrame,
    bigrams: list,
    bigram_frequencies_array: np.ndarray,
    output_path: str,
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze and plot relationship between bigram frequency and typing time distributions.
    Creates three plots:
    1. Distribution plot showing median times with error bars and sample sizes
    2. Median bigram typing times
    3. Minimum (fastest) bigram typing times
    """
    try:

        print("Input types:")
        print("bigram_data type:", type(bigram_data))
        print("bigrams type:", type(bigrams))
        print("bigram_frequencies_array type:", type(bigram_frequencies_array))
        print("\nFirst few values:")
        print("bigram_data head:\n", bigram_data.head())
        print("bigrams[:5]:", bigrams[:5])
        print("bigram_frequencies_array[:5]:", bigram_frequencies_array[:5])
        print("Shape of input data:", bigram_data.shape)
        print("Sample of bigrams:", bigrams[:5])
        print("Sample of frequencies:", bigram_frequencies_array[:5])
        
        # Create dictionary mapping bigrams to their frequencies
        bigram_frequencies = {
            b: freq for b, freq in zip(bigrams, bigram_frequencies_array)
        }
        print("Created bigram frequencies dictionary")
        
        # Create DataFrames for chosen and unchosen bigrams
        chosen_df = bigram_data[['chosen_bigram', 'chosen_bigram_time']].copy()
        unchosen_df = bigram_data[['unchosen_bigram', 'unchosen_bigram_time']].copy()
        
        print("Created separate dataframes")
        print("Chosen shape:", chosen_df.shape)
        print("Unchosen shape:", unchosen_df.shape)
        
        # Rename columns to match
        chosen_df.columns = ['bigram', 'timing']
        unchosen_df.columns = ['bigram', 'timing']
        
        # Combine data
        df = pd.concat([chosen_df, unchosen_df], axis=0)
        print("Combined data shape:", df.shape)

        # Create long format dataframe directly
        timing_data = pd.concat([
            pd.DataFrame({
                'bigram': bigram_data['chosen_bigram'],
                'timing': bigram_data['chosen_bigram_time']
            }),
            pd.DataFrame({
                'bigram': bigram_data['unchosen_bigram'], 
                'timing': bigram_data['unchosen_bigram_time']
            })
        ])

        # Calculate timing statistics for each unique bigram
        aggregated_timings = timing_data.groupby('bigram').agg({
            'timing': ['median', 'mean', 'std', 'count', 'min']
        }).reset_index()

        # Flatten column names
        aggregated_timings.columns = ['bigram', 'median_timing', 'mean_timing', 
                                    'std_timing', 'n_occurrences', 'min_timing']

        # Add frequencies and create frequency groups
        aggregated_timings['frequency'] = aggregated_timings['bigram'].map(bigram_frequencies)
        aggregated_timings = aggregated_timings.dropna(subset=['frequency'])
        aggregated_timings['freq_group'] = pd.qcut(
            aggregated_timings['frequency'], 
            n_groups, 
            labels=[f"Group {i+1}" for i in range(n_groups)]
        )

        # Calculate correlations
        correlations = {
            'median': stats.spearmanr(aggregated_timings['frequency'], 
                                      aggregated_timings['median_timing']),
            'mean': stats.spearmanr(aggregated_timings['frequency'], 
                                    aggregated_timings['mean_timing']),
            'min': stats.spearmanr(aggregated_timings['frequency'],
                                   aggregated_timings['min_timing'])
        }  

        #--------------------------
        # Plot 1: Distribution plot
        #--------------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        scatter = plt.scatter(aggregated_timings['frequency'], 
                            aggregated_timings['median_timing'],
                            alpha=0.5,
                            s=aggregated_timings['n_occurrences'] / 10)
        
        plt.errorbar(aggregated_timings['frequency'], 
                    aggregated_timings['median_timing'],
                    yerr=aggregated_timings['std_timing'],
                    fmt='none',
                    alpha=0.2)

        # Need to modify regression visualization for log scale
        x_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_log = LinearRegression().fit(x_log, aggregated_timings['median_timing'].values)
        r2_log = reg_log.score(x_log, aggregated_timings['median_timing'].values)

        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            2
        )
        plt.plot(x_range, 
                reg_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_log:.3f}')

        # Add legend for sample sizes
        legend_elements = [plt.scatter([],[], s=n/10, 
                                     label=f'n={n} samples', 
                                     alpha=0.5)
                           for n in [10, 50, 100, 500, 1000]]
        plt.legend(handles=legend_elements, 
                  title="Number of timing samples",
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left')

        plt.xlabel('Bigram Frequency')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Typing Times vs. Frequency')

        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['median'][0]:.3f}\n"
                f"p-value: {correlations['median'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')

        dist_plot_path = output_path.replace('.png', '_distribution.png')
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #----------------------
        # Plot 2: Minimum times
        #----------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        
        # Scatter plot
        plt.scatter(aggregated_timings['frequency'], 
                   aggregated_timings['min_timing'],
                   alpha=0.8)

        # Add bigram labels to each point
        for _, row in aggregated_timings.iterrows():
            plt.annotate(row['bigram'], 
                        (row['frequency'], row['min_timing']),
                        xytext=(3,3),  # 3 points offset
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)

        # Add regression line for minimum times with log scale
        x_min_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_min_log = LinearRegression().fit(
            x_min_log,
            aggregated_timings['min_timing'].values
        )
        r2_min_log = reg_min_log.score(
            x_min_log,
            aggregated_timings['min_timing'].values
        )
        
        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            100
        )
        
        plt.plot(x_range,
                reg_min_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_min_log:.3f}')

        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Minimum Typing Time (ms)')
        plt.title('Fastest Times vs. Frequency')
        
        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['min'][0]:.3f}\n"
                f"p-value: {correlations['min'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()

        min_plot_path = output_path.replace('.png', '_minimum.png')
        plt.savefig(min_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #---------------------
        # Plot 3: Median times
        #---------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        
        # Scatter plot
        plt.scatter(aggregated_timings['frequency'], 
                   aggregated_timings['median_timing'],
                   alpha=0.8)

        # Add bigram labels to each point
        for _, row in aggregated_timings.iterrows():
            plt.annotate(row['bigram'], 
                        (row['frequency'], row['median_timing']),
                        xytext=(3,3),  # 3 points offset
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)

        # Add regression line for median times with log scale
        x_med_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_med_log = LinearRegression().fit(
            x_med_log,
            aggregated_timings['median_timing'].values
        )
        r2_med_log = reg_med_log.score(
            x_med_log,
            aggregated_timings['median_timing'].values
        )
        
        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            100
        )
        
        plt.plot(x_range,
                reg_med_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_med_log:.3f}')

        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Median Times vs. Frequency')
        
        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['median'][0]:.3f}\n"
                f"p-value: {correlations['median'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()

        med_plot_path = output_path.replace('.png', '_median.png')
        plt.savefig(med_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #--------------
        # Perform ANOVA
        #--------------
        groups = [group_data['median_timing'].values 
                 for _, group_data in aggregated_timings.groupby('freq_group')]
        f_stat, anova_p = stats.f_oneway(*groups)

        # Post-hoc analysis if ANOVA is significant
        post_hoc_results = None
        if anova_p < 0.05 and len(groups) > 2:
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                post_hoc = pairwise_tukeyhsd(
                    aggregated_timings['median_timing'],
                    aggregated_timings['freq_group']
                )
                post_hoc_results = pd.DataFrame(
                    data=post_hoc._results_table.data[1:],
                    columns=post_hoc._results_table.data[0]
                )
            except Exception as e:
                print(f"Could not perform post-hoc analysis: {str(e)}")

        #----------------
        # Prepare results
        #----------------
        results = {
            'correlation_median': correlations['median'][0],
            'correlation_median_p': correlations['median'][1],
            'correlation_mean': correlations['mean'][0],
            'correlation_mean_p': correlations['mean'][1],
            'correlation_min': correlations['min'][0],
            'correlation_min_p': correlations['min'][1],
            'r2': r2_log,
            'r2_min': r2_min_log,
            'n_unique_bigrams': len(aggregated_timings),
            'total_occurrences': aggregated_timings['n_occurrences'].sum(),
            'anova_f_stat': float(f_stat),
            'anova_p_value': float(anova_p),
            'post_hoc': post_hoc_results,
            'group_stats': {
                str(group): {
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'median_timing': float(group_data['median_timing'].mean()),
                    'mean_timing': float(group_data['mean_timing'].mean()),
                    'min_timing': float(group_data['min_timing'].min()),
                    'timing_std': float(group_data['std_timing'].mean()),
                    'n_bigrams': len(group_data),
                    'total_occurrences': int(group_data['n_occurrences'].sum())
                }
                for group, group_data in aggregated_timings.groupby('freq_group')
            }
        }

        return results

    except Exception as e:
        print(f"Error in frequency timing analysis: {str(e)}")
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
    """Save timing analysis results to file."""
    try:
        with open(output_path, 'w') as f:
            f.write("Bigram Timing Analysis Results\n")
            f.write("============================\n\n")
            
            # Write correlation results
            f.write("Overall Correlation Analysis:\n")
            f.write("--------------------------\n")
            f.write(f"Median correlation: {timing_results['correlation_median']:.3f}")
            f.write(f" (p = {timing_results['correlation_median_p']:.3e})\n")
            f.write(f"Mean correlation: {timing_results['correlation_mean']:.3f}")
            f.write(f" (p = {timing_results['correlation_mean_p']:.3e})\n")
            f.write(f"R-squared: {timing_results['r2']:.3f}\n")
            f.write(f"Number of unique bigrams: {timing_results['n_unique_bigrams']}\n")
            f.write(f"Total timing instances: {timing_results['total_occurrences']}\n\n")
            
            # Write ANOVA results
            f.write("ANOVA Results:\n")
            f.write("-------------\n")
            f.write(f"F-statistic: {timing_results['anova_f_stat']:.3f}\n")
            f.write(f"p-value: {timing_results['anova_p_value']:.3e}\n\n")
            
            # Write group statistics
            f.write("Frequency Group Analysis:\n")
            f.write("----------------------\n")
            for group, stats in timing_results['group_stats'].items():
                f.write(f"\n{group}:\n")
                f.write(f"  Frequency range: {stats['freq_range']}\n")
                f.write(f"  Median typing time: {stats['median_timing']:.3f} ms\n")
                f.write(f"  Mean typing time: {stats['mean_timing']:.3f} ms\n")
                f.write(f"  Timing std dev: {stats['timing_std']:.3f} ms\n")
                f.write(f"  Number of unique bigrams: {stats['n_bigrams']}\n")
                f.write(f"  Total timing instances: {stats['total_occurrences']}\n")
            
            # Write post-hoc results if available
            if timing_results['post_hoc'] is not None:
                f.write("\nPost-hoc Analysis (Tukey HSD):\n")
                f.write("---------------------------\n")
                f.write(timing_results['post_hoc'].to_string())
                f.write("\n")
                
    except Exception as e:
        logger.error(f"Error saving timing analysis: {str(e)}")


