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

    # Extract features and handle duplicates
    for pref in dataset.preferences:
        try:
            feat1 = [pref.features1.get(f, 0.0) for f in model.selected_features]
            feat2 = [pref.features2.get(f, 0.0) for f in model.selected_features]
            
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

    X = np.array(feature_vectors)
    
    # Log transform to spread out points
    epsilon = 1e-10
    X_transformed = np.sign(X) * np.log1p(np.abs(X) + epsilon)
    
    # Standardize
    X_transformed = StandardScaler().fit_transform(X_transformed)
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_transformed)
    
    # Create figure
    fig, ax = plt.subplots(figsize=self.figure_size)
    
    # Plot points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                        c='lightblue', s=100, alpha=self.alpha,
                        edgecolor='darkblue', label='Bigrams')
    
    # Smart label placement to avoid overlaps
    texts = []
    
    # Add labels with increased initial offset
    offset = 0.05  # Increased offset for initial label placement
    for i, label in enumerate(bigram_labels):
        texts.append(ax.text(X_2d[i, 0] + offset, 
                        X_2d[i, 1] + offset,
                        label,
                        fontsize=8,
                        alpha=0.7))
    
    # Adjust label positions with more spacing
    adjust_text(texts, 
            ax=ax,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
            expand_points=(2.0, 2.0))  # Increased spacing
    
    # Add edges for paired bigrams
    processed_pairs = set()
    for pref in dataset.preferences:
        pair = (pref.bigram1, pref.bigram2)
        if pair not in processed_pairs:
            try:
                idx1 = bigram_labels.index(pref.bigram1)
                idx2 = bigram_labels.index(pref.bigram2)
                
                # Create curved line between points
                curve = plt.matplotlib.patches.ConnectionPatch(
                    xyA=(X_2d[idx1, 0], X_2d[idx1, 1]),
                    xyB=(X_2d[idx2, 0], X_2d[idx2, 1]),
                    coordsA="data", coordsB="data",
                    axesA=ax, axesB=ax,
                    color='gray', alpha=0.2,
                    connectionstyle="arc3,rad=0.2")
                ax.add_patch(curve)
                processed_pairs.add(pair)
            except ValueError:
                continue

    # Customize plot
    var1, var2 = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 (log-transformed, {var1:.1f}% variance)')
    ax.set_ylabel(f'PC2 (log-transformed, {var2:.1f}% variance)')
    ax.set_title(f"{title}\n{len(unique_bigrams)} unique bigrams")
    ax.grid(True, alpha=0.3)
    
    # Add transformation explanation
    transform_text = ("Feature space transformation:\n"
                    "sign(x) * log(|x| + ε)")
    ax.text(0.02, 0.98, transform_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    
    return fig

