#!/usr/bin/env python3
"""
Analyze feature selection metrics and trained model to determine optimal importance threshold
and visualize feature relationships. 

The elbow point (knee) calculation in the code uses the KneeLocator from the kneed library
and locates a steep drop in the log-scaled values.

Usage (from within the engram3/engram3 directory):
    python features/analyze_features.py --metrics ../output/data/feature_metrics.csv [--model ../output/data/feature_selection_model.pkl]

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

def load_data(metrics_path: str, model_path: str = None):
    """Load feature metrics and optionally the model."""
    print(f"\nLoading metrics from {metrics_path}")
    df = pd.read_csv(metrics_path)
    
    model = None
    if model_path:
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            
        # Get feature weights from saved model state
        if 'feature_weights' in model_dict:
            weights = model_dict['feature_weights']
            df['final_weight'] = df['feature_name'].map(
                lambda x: weights.get(x, (0,0))[0]
            )
            df['final_weight_std'] = df['feature_name'].map(
                lambda x: weights.get(x, (0,0))[1]
            )
        else:
            print("Warning: No feature weights found in model file")
            
        model = model_dict
    
    return df, model

def analyze_model_weights(df: pd.DataFrame, model):
    """Analyze relationship between importance scores and final model weights."""
    print("\n=== Model Weight Analysis ===")
    
    if 'final_weight' not in df.columns:
        print("No model weights available")
        return
        
    # Correlation between importance and final weights
    corr, p = spearmanr(df['selected_importance'], df['final_weight'].abs())
    print(f"\nCorrelation between importance and |weight|: {corr:.3f} (p={p:.3f})")
    
    # Plot importance vs weights
    plt.figure(figsize=(10, 6))
    plt.scatter(df['selected_importance'], df['final_weight'].abs(), alpha=0.6)
    
    # Add feature labels for top features
    for _, row in df.nlargest(5, 'selected_importance').iterrows():
        plt.annotate(row['feature_name'], 
                    (row['selected_importance'], abs(row['final_weight'])))
    
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Absolute Feature Weight in Model')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Feature Importance vs Final Model Weight')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('importance_vs_weights.png', dpi=300)
    print("\nSaved importance vs weights plot to 'importance_vs_weights.png'")
    
    # Compare feature rankings
    df['importance_rank'] = df['selected_importance'].rank(ascending=False)
    df['weight_rank'] = df['final_weight'].abs().rank(ascending=False)
    df['rank_diff'] = (df['importance_rank'] - df['weight_rank']).abs()
    
    print("\nFeatures with largest rank differences:")
    print(df.nlargest(5, 'rank_diff')[
        ['feature_name', 'importance_rank', 'weight_rank', 'rank_diff']
    ])

def analyze_model_uncertainty(df: pd.DataFrame, model):
    """Analyze model uncertainty and feature stability."""
    print("\n=== Model Uncertainty Analysis ===")
    
    if 'final_weight_std' not in df.columns:
        print("No model uncertainty information available")
        return
        
    # Calculate coefficient of variation for final weights
    df['weight_cv'] = df['final_weight_std'] / df['final_weight'].abs()
    
    print("\nFeature stability (lower CV = more stable):")
    stability_stats = df.sort_values('weight_cv')[
        ['feature_name', 'final_weight', 'final_weight_std', 'weight_cv']
    ]
    print(stability_stats.head().to_string())
    
    # Plot weight distributions for top features
    plt.figure(figsize=(12, 6))
    top_features = df.nlargest(10, 'selected_importance')
    
    x = np.arange(len(top_features))
    plt.errorbar(x, top_features['final_weight'], 
                yerr=top_features['final_weight_std'],
                fmt='o', capsize=5)
    
    plt.xticks(x, top_features['feature_name'], rotation=45, ha='right')
    plt.ylabel('Feature Weight (with std)')
    plt.title('Weight Distributions for Top Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_stability.png', dpi=300)
    print("\nSaved feature stability plot to 'feature_stability.png'")

def analyze_feature_standardization(df: pd.DataFrame, model):
    """Analyze feature standardization statistics from model."""
    print("\n=== Feature Standardization Analysis ===")
    
    if not model or 'feature_stats' not in model:
        print("No feature standardization statistics available")
        return
        
    stats_df = pd.DataFrame.from_dict(
        model['feature_stats'], 
        orient='index'
    )
    
    print("\nFeature standardization statistics:")
    print(stats_df)
    
    # Plot original feature distributions
    plt.figure(figsize=(12, 6))
    plt.scatter(stats_df['mean'], stats_df['std'], alpha=0.6)
    
    for idx in stats_df.index:
        plt.annotate(idx, (stats_df.loc[idx, 'mean'], stats_df.loc[idx, 'std']))
    
    plt.xlabel('Feature Mean')
    plt.ylabel('Feature Std Dev')
    plt.title('Feature Distributions Before Standardization')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300)
    print("\nSaved feature distribution plot to 'feature_distributions.png'")

def analyze_importance_distribution(df: pd.DataFrame, importance_threshold: float = 0.00001):
    """Analyze distribution of importance scores and suggest thresholds."""
    print("\n=== Importance Score Analysis ===")
    
    df_sorted = df.sort_values('selected_importance', ascending=False)
    importance_scores = df_sorted['selected_importance'].values
    n_scores = len(importance_scores)
    x_points = np.arange(n_scores)
    
    # Calculate statistics
    mean = df_sorted['selected_importance'].mean()
    median = df_sorted['selected_importance'].median()
    std = df_sorted['selected_importance'].std()
    mad = np.median(np.abs(importance_scores - median))
    
    print(f"\nImportance score statistics:")
    print(f"Mean: {mean:.6f}")
    print(f"Median: {median:.6f}") 
    print(f"Std: {std:.6f}")
    print(f"MAD: {mad:.6f}")
    
    # Find gaps and elbow
    gaps = df_sorted['selected_importance'].diff().sort_values(ascending=False)
    print("\nLargest gaps in importance scores:")
    print(gaps.head().to_frame('gap'))
    
    try:
        importance_scores_log = np.log10(importance_scores)
        kneedle = KneeLocator(
            x_points, importance_scores_log,
            S=1.0, curve='concave', direction='decreasing'
        )
        if kneedle.knee is not None:
            knee_value = importance_scores[kneedle.knee]
            print(f"\nDetected elbow point at importance score: {knee_value:.6f}")
    except Exception as e:
        print(f"\nCould not detect elbow point: {e}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(importance_scores, marker='o', linewidth=1, markersize=4)
    
    # Plot threshold lines
    percentiles = [25, 50, 75]
    #colors = ['purple', 'blue', 'cyan']
    #for p, c in zip(percentiles, colors):
    #    threshold = np.percentile(importance_scores, p)
    #    plt.axhline(y=threshold, color=c, linestyle='--', 
    #               label=f'{p}th percentile ({threshold:.6f})')
    
    plt.axhline(y=median, color='blue', linestyle='--',
                label=f'Median ({median:.6f})')
    plt.axhline(y=median + mad, color='gray', linestyle=':',
                label=f'Median + 1MAD ({median + mad:.6f})')
    plt.axhline(y=median - mad, color='gray', linestyle=':',
                label=f'Median - 1MAD ({median - mad:.6f})')
    #plt.axhline(y=importance_threshold, color='r', linestyle='--',
    #            label=f'Current threshold ({importance_threshold:.6f})')
    
    if kneedle.knee is not None:
        plt.axhline(y=knee_value, color='r', linestyle='--',
                   label=f'Detected elbow ({knee_value:.6f})')
    
    plt.yscale('log')
    plt.xlabel('Feature rank')
    plt.ylabel('Importance score (log scale)')
    plt.title('Feature Importance Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('importance_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved importance distribution plot to 'importance_distribution.png'")
    
    # Print threshold suggestions
    print("\nSuggested thresholds:")
    for p in percentiles:
        threshold = np.percentile(importance_scores, p)
        n_features = sum(importance_scores >= threshold)
        print(f"{p}th percentile: {threshold:.6f} ({n_features} features)")
    
    n_med = sum(importance_scores >= median)
    n_mad = sum(importance_scores >= median - mad)
    print(f"Median: {median:.6f} ({n_med} features)")
    print(f"Median - MAD: {median - mad:.6f} ({n_mad} features)")
        
    return df_sorted

def analyze_feature_consistency(df: pd.DataFrame):
    """Analyze relationship between effect magnitude and consistency."""
    print("\n=== Feature Consistency Analysis ===")
    
    # Create consistency plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['effect_magnitude'], df['effect_std'], 
                alpha=0.6, c=df['selected_importance'], cmap='viridis')
    
    plt.colorbar(label='Importance score')
    plt.xlabel('Effect magnitude')
    plt.ylabel('Effect standard deviation')
    plt.title('Effect Magnitude vs. Standard Deviation')
    
    # Add diagonal lines showing different std/magnitude ratios
    max_val = max(df['effect_magnitude'].max(), df['effect_std'].max())
    x = np.linspace(0, max_val, 100)
    ratios = [1, 2, 5]
    for ratio in ratios:
        plt.plot(x, ratio * x, '--', alpha=0.3, color='gray', 
                label=f'std/magnitude = {ratio}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_consistency.png', dpi=300)
    print("\nSaved feature consistency plot to 'feature_consistency.png'")
    
    # Analyze correlations
    corr, p = spearmanr(df['effect_magnitude'], df['effect_std'])
    print(f"\nCorrelation between magnitude and std: {corr:.3f} (p={p:.3f})")
    
    # Look for clusters in magnitude-std space
    X = StandardScaler().fit_transform(
        df[['effect_magnitude', 'effect_std']]
    )
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    
    cluster_stats = pd.DataFrame({
        'cluster': range(3),
        'size': pd.Series(kmeans.labels_).value_counts().sort_index(),
        'avg_importance': [df[kmeans.labels_ == i]['selected_importance'].mean() 
                         for i in range(3)]
    }).sort_values('avg_importance', ascending=False)
    
    print("\nFeature clusters by magnitude-std relationship:")
    print(cluster_stats)

def analyze_feature_interactions(df: pd.DataFrame):
    """Analyze interactions between features."""
    print("\n=== Feature Interaction Analysis ===")
    
    # Get base features and interaction features
    base_features = df[~df['feature_name'].str.contains('_x_')]
    interaction_features = df[df['feature_name'].str.contains('_x_')]
    
    print(f"\nBase features: {len(base_features)}")
    print(f"Interaction features: {len(interaction_features)}")
    
    if len(interaction_features) > 0:
        # Compare importance of base vs interaction features
        print("\nImportance statistics by feature type:")
        print("\nBase features:")
        print(base_features['selected_importance'].describe())
        print("\nInteraction features:")
        print(interaction_features['selected_importance'].describe())
        
        # Plot comparison
        plt.figure(figsize=(8, 6))
        plt.boxplot([
            base_features['selected_importance'],
            interaction_features['selected_importance']
        ], labels=['Base features', 'Interaction features'])
        plt.yscale('log')
        plt.ylabel('Importance score (log scale)')
        plt.title('Feature Importance by Type')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_types.png', dpi=300)
        print("\nSaved feature type comparison plot to 'feature_types.png'")

def main():
    importance_threshold = 0.00001  #config.feature_selection['importance_threshold']

    parser = argparse.ArgumentParser(description='Analyze feature selection metrics')
    parser.add_argument('--metrics', required=True, help='Path to feature_metrics.csv')
    parser.add_argument('--model', help='Path to feature_selection_model.pkl')
    args = parser.parse_args()
    
    # Load data
    df, model = load_data(args.metrics, args.model)
    
    # Print basic statistics
    print("\nDataset summary:")
    print(f"Total features: {len(df)}")
    print(f"Selected features: {sum(df['selected'])}")
    
    # Analyze importance distribution
    df_sorted = analyze_importance_distribution(df, importance_threshold)
    
    # Analyze feature consistency
    analyze_feature_consistency(df)
    
    # Analyze feature interactions
    analyze_feature_interactions(df)
    
    # Model-specific analyses if model is provided
    if model is not None:
        analyze_model_weights(df, model)
        analyze_model_uncertainty(df, model)
        analyze_feature_standardization(df, model)
    
    # Save annotated metrics
    output_path = Path('analyzed_metrics.csv')
    df_sorted.to_csv(output_path, index=False)
    print(f"\nSaved annotated metrics to {output_path}")

if __name__ == '__main__':
    main()