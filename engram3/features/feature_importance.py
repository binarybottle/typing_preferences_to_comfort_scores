# engram3/features/feature_importance.py
"""
Feature importance calculation and selection for keyboard layout preference analysis.

The FeatureImportanceCalculator class:
- Manages feature evaluation pipeline
- Handles feature standardization and validation
- Provides detailed diagnostic information
- Supports both base and interaction features
- Implements efficient caching mechanisms
- Produces detailed evaluation reports

Used by the main feature selection pipeline to identify optimal feature sets
for the keyboard layout preference model.
"""
import numpy as np
from typing import Dict, List, Union
from pathlib import Path

from engram3.utils.config import Config, FeatureSelectionSettings
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engram3.model import PreferenceModel
from engram3.data import PreferenceDataset
from engram3.utils.caching import CacheManager
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureImportanceCalculator:
    """Centralized feature importance calculation."""
    
    def __init__(self, config: Union[Dict, Config], model: 'PreferenceModel'):
        """Initialize the feature importance calculator.
        
        Args:
            config: Either a dictionary or Config instance containing configuration
            model: Reference to the PreferenceModel instance
        """
        logger.debug("Initializing FeatureImportanceCalculator")
        
        # Handle different config input types
        if isinstance(config, dict):
            feature_selection_config = config.get('feature_selection', {})
            self.features = config.get('features', {})
            # Convert to FeatureSelectionSettings
            logger.debug(f"Converting dict config: {feature_selection_config}")
            self.config = FeatureSelectionSettings(**feature_selection_config)
        else:
            # We already have a properly validated Config instance
            logger.debug("Using existing Config instance")
            self.config = config.feature_selection  # Already a FeatureSelectionSettings instance
            self.features = config.features
        
        logger.debug(f"Features from config: {self.features}")
        logger.debug(f"Feature selection config: {self.config}")
        
        # Store model reference
        self.model = model
           
        # Initialize caches
        self.metric_cache = CacheManager(max_size=10000)
        self.feature_values_cache = CacheManager(max_size=10000)

        # Ensure Stan model executable has proper permissions
        if hasattr(model, 'model') and hasattr(model.model, 'exe_file'):
            try:
                import os
                exe_path = Path(model.model.exe_file)
                if exe_path.exists():
                    os.chmod(str(exe_path), 0o755)
            except Exception as e:
                logger.warning(f"Could not set Stan model executable permissions: {str(e)}")

        # Initialize tracking variables
        self._init_tracking_variables()

        # Calculate baseline accuracy once
        self._baseline_accuracy = self._compute_baseline_accuracy(model.dataset, model)

    def _init_tracking_variables(self):
        """Initialize all tracking variables for feature importance calculation."""
        # Global tracking
        self._max_effect_seen = 0.0
        self._max_consistency_seen = 0.0
        self._baseline_accuracy = None  # Will be set on first evaluation

        # Separate tracking for control vs main features
        self._max_effect = {
            'control': 0.0,  # For normalizing control features
            'main': 0.0      # For normalizing main features
        }
        self._max_consistency = {
            'control': 0.0,
            'main': 0.0
        }
        self._baseline_accuracy = None

    def _compute_baseline_accuracy(self, dataset: PreferenceDataset, model: 'PreferenceModel') -> float:
        """Compute baseline model accuracy using only control features."""
        try:
            if not dataset.preferences:
                raise ValueError("Empty dataset")
                
            # Create baseline model with only control features
            baseline_model = type(model)(config=model.config)
            baseline_model.feature_extractor = model.feature_extractor
            baseline_model.is_baseline_model = True  # Flag this as baseline
            
            # Use only control features
            control_features = list(model.config.features.control_features)
            if not control_features:
                logger.warning("No control features defined, using random chance baseline")
                return 0.5  # Random chance baseline
                
            # Initialize model properly
            baseline_model.feature_names = control_features
            baseline_model.selected_features = control_features
                
            try:
                baseline_model.fit(dataset, control_features, 
                                fit_purpose="Computing baseline accuracy")
                metrics = baseline_model.evaluate(dataset)
                accuracy = metrics.get('accuracy')
                
                if accuracy is None:
                    logger.warning("Could not compute baseline accuracy, using random chance")
                    return 0.5
                    
                logger.info(f"Baseline accuracy: {accuracy:.4f}")
                return float(accuracy)
                
            finally:
                baseline_model.cleanup()
                    
        except Exception as e:
            logger.error(f"Error computing baseline accuracy: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return 0.5  # Return random chance on error

    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'feature_values_cache'):
            self.feature_values_cache.clear()

