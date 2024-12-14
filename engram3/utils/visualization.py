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
from typing import Optional, Tuple, Union
from pathlib import Path

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