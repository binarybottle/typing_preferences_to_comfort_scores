import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

class FeatureMetricsVisualizer:
    """Handles visualization and analysis of feature metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
    def plot_feature_metrics(self, metrics_dict: Dict[str, Dict[str, float]]):
        """Plot comprehensive feature metrics visualization."""
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([
            {
                'feature': feature,
                **metrics
            }
            for feature, metrics in metrics_dict.items()
        ])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # 1. Correlation vs Mutual Information scatter
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation_mi_scatter(metrics_df, ax1)
        
        # 2. Model Effect with uncertainties
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_model_effects(metrics_df, ax2)
        
        # 3. Combined Score comparison
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_combined_scores(metrics_df, ax3)
        
        plt.tight_layout()
        return fig
        
    def _plot_correlation_mi_scatter(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot correlation vs mutual information scatter plot."""
        scatter = ax.scatter(df['correlation'].abs(), 
                           df['mutual_information'],
                           c=df['combined_score'],
                           cmap='viridis',
                           s=100,
                           alpha=0.6)
        
        # Add feature labels
        for _, row in df.iterrows():
            ax.annotate(row['feature'], 
                       (abs(row['correlation']), row['mutual_information']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
            
        ax.set_xlabel('|Correlation|')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Feature Relationship Measures')
        plt.colorbar(scatter, ax=ax, label='Combined Score')
        
    def _plot_model_effects(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot model effects with error bars."""
        # Sort by absolute effect size
        df['abs_effect'] = abs(df['model_effect_mean'])
        df_sorted = df.sort_values('abs_effect', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        
        # Plot horizontal error bars
        ax.barh(y_pos, 
                df_sorted['model_effect_mean'],
                xerr=df_sorted['model_effect_std'],
                alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['feature'], fontsize=8)
        ax.set_xlabel('Model Effect')
        ax.set_title('Feature Effects with Uncertainty')
        
    def _plot_combined_scores(self, df: pd.DataFrame, ax: plt.Axes):
        """Plot combined scores with component breakdown."""
        # Sort by combined score
        df_sorted = df.sort_values('combined_score', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        width = 0.35
        
        # Plot stacked bars for score components
        ax.barh(y_pos, df_sorted['correlation'].abs(), width,
                label='|Correlation|', alpha=0.6)
        ax.barh(y_pos, df_sorted['mutual_information'], width,
                left=df_sorted['correlation'].abs(),
                label='Mutual Information', alpha=0.6)
        
        # Add normalized model effect
        normalized_effect = (df_sorted['model_effect_mean'].abs() / 
                           df_sorted['model_effect_std'].clip(lower=1e-10))
        ax.barh(y_pos, normalized_effect, width,
                left=df_sorted['correlation'].abs() + df_sorted['mutual_information'],
                label='Normalized Model Effect', alpha=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['feature'], fontsize=8)
        ax.set_xlabel('Score Components')
        ax.set_title('Feature Score Breakdown')
        ax.legend(loc='lower right')
        
    def plot_feature_evolution(self, metrics_history: List[Dict]):
        """Plot how feature metrics evolve during selection process."""
        # Convert history to DataFrame
        history_df = pd.DataFrame(metrics_history)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot metric evolution
        for metric in ['correlation', 'mutual_information', 'model_effect']:
            axes[0].plot(history_df.index, history_df[metric],
                        label=metric, marker='o')
        
        axes[0].set_xlabel('Selection Step')
        axes[0].set_ylabel('Metric Value')
        axes[0].set_title('Feature Metric Evolution')
        axes[0].legend()
        
        # Plot performance metrics
        for metric in ['accuracy', 'transitivity_score']:
            axes[1].plot(history_df.index, history_df[metric],
                        label=metric, marker='o')
            
        axes[1].set_xlabel('Selection Step')
        axes[1].set_ylabel('Performance')
        axes[1].set_title('Model Performance Evolution')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
        
    def save_metrics_report(self, metrics_dict: Dict[str, Dict[str, float]],
                           output_file: str):
        """Generate and save a detailed metrics report."""
        report_df = pd.DataFrame([
            {
                'Feature': feature,
                'Correlation': metrics['correlation'],
                'Mutual Information': metrics['mutual_information'],
                'Model Effect': metrics['model_effect_mean'],
                'Effect Std': metrics['model_effect_std'],
                'Combined Score': metrics['combined_score']
            }
            for feature, metrics in metrics_dict.items()
        ])
        
        # Sort by combined score
        report_df = report_df.sort_values('Combined Score', ascending=False)
        
        # Save to CSV
        report_df.to_csv(self.output_dir / output_file, index=False)