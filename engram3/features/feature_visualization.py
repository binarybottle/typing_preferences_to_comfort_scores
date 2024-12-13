# engram3/features/feature_visualization.py
"""
Visualization module for feature metrics and model analysis.

Provides comprehensive visualization capabilities including:
  - Feature importance metrics visualization
  - PCA-based feature space exploration
  - Feature selection process tracking
  - Model performance monitoring
  - Feature correlation analysis
  - Effect size visualization
  - Uncertainty quantification plots
  - Detailed metric reports generation

Supports analysis workflow through:
  - Interactive plotting interfaces
  - Configurable output formats
  - Multiple visualization perspectives
  - Temporal evolution tracking
  - Component-wise breakdowns
  - Statistical summaries
  - Export functionality for reports
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from collections import defaultdict

from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.utils.visualization import PlottingUtils

class FeatureMetricsVisualizer:
    """Handles all feature-related visualization and tracking."""
    
    def __init__(self, config: Dict):
        self.output_dir = Path(config['data']['visualization']['output_dir'])
        self.plotting = PlottingUtils(self.output_dir)
        
        # Create subdirectories
        self.dirs = {
            'iterations': self.output_dir / 'iterations',
            'features': self.output_dir / 'features',
            'performance': self.output_dir / 'performance',
            'space': self.output_dir / 'feature_space'
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        self.iteration_metrics = defaultdict(list)
    
    def plot_feature_metrics(self, metrics_dict: Dict[str, Dict[str, float]]):
        """Plot comprehensive feature metrics visualization."""
        fig, axes = self.plotting.create_figure(figsize=(15, 10))
        self.plotting.setup_axis(axes, title="Feature Metrics")

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
    
    def plot_feature_space(self, model: 'PreferenceModel', 
                        dataset: 'PreferenceDataset',
                        title: str = "Feature Space"):
        """Plot 2D PCA of feature space with comfort scores."""
        feature_vectors = []
        comfort_scores = []
        uncertainties = []
        
        for pref in dataset.preferences:
            feat1 = [pref.features1[f] for f in model.selected_features]
            feat2 = [pref.features2[f] for f in model.selected_features]
            
            score1, unc1 = model.get_bigram_comfort_scores(pref.bigram1)
            score2, unc2 = model.get_bigram_comfort_scores(pref.bigram2)
            
            feature_vectors.extend([feat1, feat2])
            comfort_scores.extend([score1, score2])
            uncertainties.extend([unc1, unc2])
            
        X = StandardScaler().fit_transform(feature_vectors)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                           c=comfort_scores,
                           s=np.array(uncertainties) * 500,
                           alpha=0.6,
                           cmap='RdYlBu')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.colorbar(scatter, label='Comfort Score')
        plt.title(title)
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

    def save_iteration(self, iteration: int, model: 'PreferenceModel',
                      dataset: PreferenceDataset, metrics: Dict[str, float]):
        """Save metrics and generate visualizations for current iteration."""
        self._save_metrics(iteration, metrics)
        self._plot_feature_space(iteration, model, dataset)
        self._plot_feature_impacts(iteration, model)
        self._plot_performance_tracking()
        self._update_tracking_plots()

    def _save_metrics(self, iteration: int, metrics: Dict[str, float]):
        metrics['iteration'] = iteration
        df = pd.DataFrame([metrics])
        file_path = self.dirs['iterations'] / f'iteration_{iteration}_metrics.csv'
        df.to_csv(file_path, index=False)
        
        for key, value in metrics.items():
            self.iteration_metrics[key].append(value)

    def plot_feature_impacts(self, model: 'PreferenceModel'):
        """Plot feature weights and their interactions."""
        weights = model.get_feature_weights()
        
        # Feature weights plot
        fig1 = plt.figure(figsize=(12, 6))
        sns.barplot(x=list(weights.keys()), y=list(weights.values()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Weights')
        plt.tight_layout()
        
        # Interaction heatmap
        interaction_features = [f for f in model.selected_features if '_x_' in f]
        if interaction_features:
            fig2 = plt.figure(figsize=(10, 8))
            interaction_matrix = np.zeros((len(model.selected_features), 
                                        len(model.selected_features)))
            for i, f1 in enumerate(model.selected_features):
                for j, f2 in enumerate(model.selected_features):
                    interaction = f"{f1}_x_{f2}"
                    if interaction in weights:
                        interaction_matrix[i, j] = abs(weights[interaction])
            
            sns.heatmap(interaction_matrix, 
                       xticklabels=model.selected_features,
                       yticklabels=model.selected_features,
                       cmap='YlOrRd')
            plt.title('Feature Interactions')
            plt.tight_layout()
            return fig1, fig2
        
        return fig1, None

    def plot_performance_history(self):
        """Plot performance metrics over iterations."""
        metrics_df = pd.DataFrame(self.iteration_metrics)
        
        fig = plt.figure(figsize=(12, 6))
        for col in metrics_df.columns:
            if col != 'iteration':
                plt.plot(metrics_df['iteration'], metrics_df[col], 
                        label=col, marker='o')
        
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
