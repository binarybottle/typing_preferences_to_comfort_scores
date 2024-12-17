# engram3/utils/visualization.py
"""
Plotting utilities module for data visualization.

Provides standardized plotting functionality with:
  - Consistent styling and formatting
  - Common figure/axis setup
  - Automated output management
  - Figure saving utilities
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union
from pathlib import Path
from adjustText import adjust_text
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from typing import TYPE_CHECKING, Optional, Tuple, Union
if TYPE_CHECKING:
    from engram3.model import PreferenceModel
    from engram3.data import PreferenceDataset
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class PlottingUtils:
    """Centralized plotting utilities with common setup and styling."""
    
    def __init__(self, plots_dir: Union[str, Path]):
        """Initialize PlottingUtils.
        
        Args:
            plots_dir: Directory path for saving plots
        """
        self.output_dir = Path(plots_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_axis(self, ax: plt.Axes, 
                   title: Optional[str] = None,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   legend: bool = True,
                   grid: bool = True) -> plt.Axes:
        """Common axis setup with consistent styling."""
        if title:
            ax.set_title(title, pad=20)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        return ax
        
    def create_figure(self, 
                     figsize: Tuple[int, int] = (10, 6),
                     title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create figure with common settings."""
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            fig.suptitle(title)
        return fig, ax
        
    def save_figure(self, 
                    fig: plt.Figure,
                    filename: str,
                    dpi: int = 300,
                    bbox_inches: str = 'tight') -> None:
        """Save figure with standard settings."""
        filepath = self.config.paths.plots_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)

def plot_feature_space(model: 'PreferenceModel', 
                        dataset: 'PreferenceDataset', 
                        title: str = "Feature Space",
                        figure_size: Tuple[int, int] = (12, 8),
                        alpha: float = 0.6) -> Figure:
        """Plot 2D feature space with log-transformed features and improved label handling."""
        feature_vectors = []
        bigram_labels = []
        unique_bigrams = set()  # Track unique bigrams
        
        # Separate main and control features
        control_features = model.config.features.control_features
        main_features = [f for f in model.selected_features 
                        if f not in control_features]
        
        # Extract features and handle duplicates
        for pref in dataset.preferences:
            try:
                if not main_features:
                    logger.warning("No main features available for visualization")
                    return create_empty_plot(title, "No main features available for visualization")
                    
                # Get only main features for PCA visualization
                feat1 = [pref.features1.get(f, 0.0) for f in main_features]
                feat2 = [pref.features2.get(f, 0.0) for f in main_features]
                
                # Only add if bigram not seen before
                if pref.bigram1 not in unique_bigrams:
                    feature_vectors.append(feat1)
                    bigram_labels.append(pref.bigram1)
                    unique_bigrams.add(pref.bigram1)
                if pref.bigram2 not in unique_bigrams:
                    feature_vectors.append(feat2)
                    bigram_labels.append(pref.bigram2)
                    unique_bigrams.add(pref.bigram2)
                    
            except Exception as e:
                logger.warning(f"Skipping preference due to feature error: {e}")
                continue
                
        if not feature_vectors:
            logger.warning("No valid feature vectors for visualization")
            return create_empty_plot(title, "No valid data for visualization")
            
        X = np.array(feature_vectors)
        
        if X.shape[0] < 2 or X.shape[1] < 1:
            logger.warning(f"Insufficient data for PCA: {X.shape[0]} samples, {X.shape[1]} features")
            return create_empty_plot(title, f"Insufficient data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Log transform to spread out points
        epsilon = 1e-10
        X_transformed = np.sign(X) * np.log1p(np.abs(X) + epsilon)
        
        # Standardize
        X_transformed = StandardScaler().fit_transform(X_transformed)
        
        # Fit PCA with appropriate number of components
        n_components = min(2, X_transformed.shape[0], X_transformed.shape[1])
        pca = PCA(n_components=n_components, svd_solver='auto')
        X_2d = pca.fit_transform(X_transformed)
        
        # If we only got 1 component, add a zero second component
        if X_2d.shape[1] == 1:
            X_2d = np.column_stack([X_2d, np.zeros_like(X_2d)])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Plot points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c='lightblue', s=100, alpha=alpha,
                            edgecolor='darkblue', label='Bigrams')
        
        # Add labels with increased initial offset
        texts = []
        offset = 0.05
        for i, label in enumerate(bigram_labels):
            texts.append(ax.text(X_2d[i, 0] + offset, 
                            X_2d[i, 1] + offset,
                            label,
                            fontsize=8,
                            alpha=0.7))
        
        # Adjust label positions with more spacing
        try:
            adjust_text(texts, 
                    ax=ax,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                    expand_points=(2.0, 2.0))
        except Exception as e:
            logger.warning(f"Could not adjust text positions: {e}")
        
        # Customize plot
        if hasattr(pca, 'explained_variance_ratio_'):
            var1 = pca.explained_variance_ratio_[0] * 100
            var2 = pca.explained_variance_ratio_[1] * 100 if len(pca.explained_variance_ratio_) > 1 else 0
            ax.set_xlabel(f'PC1 (log-transformed, {var1:.1f}% variance)')
            ax.set_ylabel(f'PC2 (log-transformed, {var2:.1f}% variance)')
        
        # Update title to indicate control
        control_note = f"\n(Effects shown after controlling for {', '.join(control_features)})" if control_features else ""
        ax.set_title(f"{title}\n{len(unique_bigrams)} unique bigrams{control_note}")
        ax.grid(True, alpha=0.3)
        
        # Add note about feature transformation
        transform_text = ("Feature space transformation:\n"
                        "1. Log transform: sign(x) * log(|x| + ε)\n"
                        "2. Standardize\n"
                        "3. PCA projection")
        ax.text(0.02, 0.98, transform_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        return fig

def create_empty_plot(title: str, message: str) -> Figure:
    """Create an empty plot with an error message."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, message,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
