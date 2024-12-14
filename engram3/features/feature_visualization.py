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
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from engram3.utils.config import Config, VisualizationSettings
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engram3.model import PreferenceModel
from engram3.data import PreferenceDataset
from engram3.utils.visualization import PlottingUtils
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureMetricsVisualizer:
    """Handles all feature-related visualization and tracking."""
    
    DEFAULT_FIGURE_SIZE = (12, 8)
    DEFAULT_DPI = 300
    DEFAULT_ALPHA = 0.6
    MI_BINS = 20

    def __init__(self, config: Union[Dict, Config]):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Configuration dictionary or Config object
        """
        if isinstance(config, dict):
            self.config = Config(**config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError(f"Config must be a dictionary or Config object, got {type(config)}")

        # Use root-level visualization settings
        self.dpi = self.config.visualization.dpi
        self.output_dir = Path(self.config.paths.plots_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set other visualization parameters
        self.figure_size = self.DEFAULT_FIGURE_SIZE
        self.alpha = self.DEFAULT_ALPHA
        self.color_map = 'viridis'
        
    def plot_feature_metrics(self, model: 'PreferenceModel', metrics_dict: Dict[str, Dict[str, float]]) -> Figure:
        """
        Plot comprehensive feature metrics visualization.
        
        Args:
            model: PreferenceModel instance
            metrics_dict: Dictionary of feature metrics
            
        Returns:
            matplotlib Figure object
        """
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([
            {
                'feature': feature,
                **metrics
            }
            for feature, metrics in metrics_dict.items()
        ])
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # 1. Correlation vs Mutual Information scatter
        ax1 = fig.add_subplot(gs[0, 0])
        self.plotting.setup_axis(
            ax1,
            title="Feature Correlations and Mutual Information",
            xlabel="Absolute Correlation",
            ylabel="Mutual Information"
        )
        self._plot_correlation_mi_scatter(metrics_df, ax1)
        
        # 2. Model Effect with uncertainties
        ax2 = fig.add_subplot(gs[0, 1])
        self.plotting.setup_axis(
            ax2,
            title="Feature Effects and Uncertainties",
            xlabel="Effect Size",
            ylabel="Features"
        )
        self._plot_model_effects(metrics_df, ax2)
        
        # 3. Combined Score comparison
        ax3 = fig.add_subplot(gs[1, :])
        self.plotting.setup_axis(
            ax3,
            title="Combined Feature Scores",
            xlabel="Score",
            ylabel="Features"
        )
        self._plot_combined_scores(metrics_df, ax3)
        
        plt.tight_layout()
        
        # Save if configured
        if getattr(self.config, 'visualization', {}).get('save_plots', False):
            self.plotting.save_figure(
                fig,
                "feature_metrics.png",
                dpi=getattr(self.config, 'visualization', {}).get('dpi', 300)
            )
        
        return fig
    
    def plot_feature_space(self, model: 'PreferenceModel', dataset: 'PreferenceDataset', title: str = "Feature Space") -> Figure:
        """Plot 2D PCA of feature space with comfort scores."""
        feature_vectors = []
        bigram_labels = []
        valid_indices = []
        
        # Extract features and handle NaN values
        for i, pref in enumerate(dataset.preferences):
            try:
                # Get feature vectors and handle NaN values
                feat1 = [pref.features1.get(f, 0.0) for f in model.selected_features]
                feat2 = [pref.features2.get(f, 0.0) for f in model.selected_features]
                
                # Check for NaN values
                if any(pd.isna(v) for v in feat1) or any(pd.isna(v) for v in feat2):
                    continue
                    
                feature_vectors.extend([feat1, feat2])
                bigram_labels.extend([pref.bigram1, pref.bigram2])
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Skipping preference due to feature error: {e}")
                continue

        if not feature_vectors:
            raise ValueError("No valid feature vectors after filtering")

        # Convert to numpy array and standardize features
        X = np.array(feature_vectors)
        
        # Additional NaN check
        if np.any(np.isnan(X)):
            logger.warning(f"Found {np.sum(np.isnan(X))} NaN values after conversion")
            # Replace NaN with 0
            X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Calculate explained variance
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c='lightblue', s=100, alpha=self.alpha,
                            edgecolor='darkblue')
        
        # Add labels with offset
        for i, label in enumerate(bigram_labels):
            offset = 0.02 * (max(X_2d[:, 0]) - min(X_2d[:, 0]))
            ax.annotate(label, 
                    (X_2d[i, 0] + offset, X_2d[i, 1] + offset),
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Add edges for paired bigrams
        for i in range(0, len(bigram_labels), 2):
            # Create curved line between points
            mid_point = [(X_2d[i, 0] + X_2d[i+1, 0])/2,
                        (X_2d[i, 1] + X_2d[i+1, 1])/2]
            # Add some curvature
            mid_point[1] += 0.05 * (max(X_2d[:, 1]) - min(X_2d[:, 1]))
            
            curve = plt.matplotlib.patches.ConnectionPatch(
                xyA=(X_2d[i, 0], X_2d[i, 1]),
                xyB=(X_2d[i+1, 0], X_2d[i+1, 1]),
                coordsA="data", coordsB="data",
                axesA=ax, axesB=ax,
                color='gray', alpha=0.3,
                connectionstyle="arc3,rad=0.2")
            ax.add_patch(curve)
        
        # Set labels and title
        ax.set_xlabel(f'PC1 ({var1:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({var2:.1f}% variance)')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add some padding to the limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        padding = 0.1
        ax.set_xlim(xlim[0] - padding * (xlim[1] - xlim[0]),
                    xlim[1] + padding * (xlim[1] - xlim[0]))
        ax.set_ylim(ylim[0] - padding * (ylim[1] - ylim[0]),
                    ylim[1] + padding * (ylim[1] - ylim[0]))
        
        plt.tight_layout()
        
        # Log summary
        logger.info(f"\nFeature space visualization:")
        logger.info(f"  Total preferences: {len(dataset.preferences)}")
        logger.info(f"  Valid preferences: {len(valid_indices)}")
        logger.info(f"  Features used: {len(model.selected_features)}")
        logger.info(f"  Variance explained: {var1 + var2:.1f}%")
        
        return fig

    def _plot_feature_space(self, iteration: int, model: 'PreferenceModel', 
                        dataset: PreferenceDataset) -> None:
        """Plot feature space for current iteration."""
        fig = self.plot_feature_space(model, dataset)
        self.plotting.save_figure(
            fig,
            f"feature_space_iteration_{iteration}.png"
        )

    def _plot_performance_tracking(self) -> None:
        """Plot performance tracking metrics."""
        fig = self.plot_performance_history()
        self.plotting.save_figure(
            fig,
            "performance_tracking.png"
        )

    def _plot_correlation_mi_scatter(self, df: pd.DataFrame, ax: plt.Axes) -> None:
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
        
    def _plot_model_effects(self, df: pd.DataFrame, ax: plt.Axes) -> None:
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
        
    def _plot_combined_scores(self, df: pd.DataFrame, ax: plt.Axes) -> None:
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
        report_df.to_csv(self.config.paths.metrics_dir / output_file, index=False)

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

    def plot_feature_impacts(self, model: 'PreferenceModel') -> Tuple[Figure, Optional[Figure]]:
        """Plot feature weights and their interactions."""
        try:
            weights = model.get_feature_weights()
            if not weights:
                logger.warning("No feature weights available")
                return plt.figure(), None
            
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
        
        except Exception as e:
                logger.error(f"Error plotting feature impacts: {str(e)}")
                return plt.figure(), None

    def _plot_feature_impacts(self, iteration: int, model: 'PreferenceModel') -> None:
        """Plot feature impacts for current iteration."""
        fig1, fig2 = self.plot_feature_impacts(model)
        self.plotting.save_figure(
            fig1,
            f"feature_weights_iteration_{iteration}.png"
        )
        if fig2 is not None:
            self.plotting.save_figure(
                fig2,
                f"feature_interactions_iteration_{iteration}.png"
            )

    def plot_performance_history(self) -> Figure:
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
