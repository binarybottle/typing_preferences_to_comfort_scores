# engram3/features/importance.py
"""
Feature importance calculation module for keyboard layout preference analysis.

Provides comprehensive feature evaluation through three independent metrics:
  - model_effect: Feature's impact based on model weights
  - effect_consistency: Feature's stability across cross-validation splits  
  - predictive_power: Model performance improvement from feature

Key functionality:
  - Pairwise feature comparison using all three metrics
  - Round-robin tournament selection in context of previously selected features
  - Efficient caching of feature computations
  - Special handling for interaction features
  - Support for typing time and other keyboard-specific features
  - Cross-validation based stability assessment
  - Detailed metric reporting and logging
  - Error handling and fallback mechanisms

The module centers around the FeatureImportanceCalculator class which:
  - Evaluates individual features using multiple independent metrics
  - Compares features based on wins across all metrics
  - Handles feature value computation and caching
  - Generates detailed metrics reports
  - Provides robust error handling
"""
import numpy as np
from typing import Dict, Union
from pathlib import Path
from collections import defaultdict

from engram3.utils.config import Config, FeatureSelectionConfig
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
        logger.debug("Initializing FeatureImportanceCalculator")
        if isinstance(config, dict):
            feature_selection_config = config.get('feature_selection', {})
            self.features = config.get('features', {})
        else:
            feature_selection_config = config.feature_selection.dict()
            self.features = config.features
        
        logger.debug(f"Features from config: {self.features}")
        self.config = FeatureSelectionConfig(**feature_selection_config)
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
                                
    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'feature_values_cache'):
            self.feature_values_cache.clear()
                                                            
    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel') -> Dict[str, float]:
        """Calculate all metrics for a feature."""
        try:
            logger.debug(f"Feature being evaluated: {feature}")
            
            # Create new model instance here, before the try block
            model_single = type(model)(config=model.config)
            model_single.feature_extractor = model.feature_extractor
            
            # If it's a control feature, return special metrics
            if feature in model.config.features.control_features:
                weights = model.get_feature_weights(include_control=True)
                if feature in weights:
                    return {
                        'model_effect': 0.0,
                        'effect_consistency': 0.0,
                        'predictive_power': 0.0,
                        'weight': weights[feature][0],
                        'weight_std': weights[feature][1]
                    }
                return self._get_default_metrics()
            
            # For non-control features, determine features to test
            features_to_test = [feature]  # Start with base feature
            
            # Add interactions if any
            for f1, f2 in model.config.features.interactions:
                if feature in (f1, f2):
                    interaction_name = f"{f1}_x_{f2}"
                    if interaction_name in model.selected_features:
                        features_to_test.append(interaction_name)
                    else:
                        alt_name = f"{f2}_x_{f1}"
                        if alt_name in model.selected_features:
                            features_to_test.append(alt_name)

            try:
                logger.info("Fitting model with features: %s", features_to_test)
                # Fit and evaluate with feature(s)
                model_single.fit(dataset, features_to_test)

                logger.info("Evaluating model...")
                metrics_single = model_single.evaluate(dataset)
                logger.info(f"Evaluation metrics: {metrics_single}")
                
                acc_single = metrics_single.get('accuracy', 0.5)
                logger.info(f"Model accuracy: {acc_single:.4f}")
                
                # Calculate improvement over random chance (0.5)
                improvement = acc_single - 0.5
                logger.info(f"Raw improvement over random: {improvement:.4f}")
                
                # Normalize improvement to [0,1] range
                normalized_improvement = np.clip(improvement / 0.5, 0, 1)
                logger.info(f"Normalized improvement: {normalized_improvement:.4f}")
                
                # Return metrics dictionary instead of just the normalized improvement
                return {
                    'model_effect': abs(normalized_improvement),
                    'effect_consistency': normalized_improvement,
                    'predictive_power': normalized_improvement
                }
                
            finally:
                # Clean up
                if hasattr(model_single, 'fit_result'):
                    del model_single.fit_result
                del model_single
                
        except Exception as e:
            logger.error(f"Error calculating predictive power for {feature}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self._get_default_metrics()  # Return metrics dictionary instead of float
            
    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset, 
                                    include_interactions: bool = False) -> float:
        """
        Calculate consistency of feature effect across cross-validation splits.
        Returns a value between 0 (inconsistent) and 1 (consistent).
        """
        try:
            n_splits = 5
            effects = []
            interaction_effects = defaultdict(list)
            
            for train_idx, val_idx in self.model._get_cv_splits(dataset, n_splits):
                train_data = dataset._create_subset_dataset(train_idx)
                
                features_to_test = []
                
                # Handle control features as main features for testing
                if feature in self.model.config.features.control_features:
                    features_to_test.append(feature)
                else:
                    features_to_test.append(feature)
                    if include_interactions:
                        for f1, f2 in self.model.config.features.interactions:
                            if feature in (f1, f2):
                                interaction_name = f"{f1}_x_{f2}"
                                if interaction_name in self.model.selected_features:
                                    features_to_test.append(interaction_name)
                                else:
                                    alt_name = f"{f2}_x_{f1}"
                                    if alt_name in self.model.selected_features:
                                        features_to_test.append(alt_name)
                                                        
                self.model.fit(train_data, features_to_test)
                weights = self.model.get_feature_weights()
                
                # Get base feature effect
                if feature in weights:
                    effects.append(weights[feature][0])
                
                # Get interaction effects
                if include_interactions:
                    for f in features_to_test:
                        if '_x_' in f and feature in f:
                            interaction_effects[f].append(weights.get(f, (0.0, 0.0))[0])
            
            if not effects and not interaction_effects:
                return 0.0
            
            # Calculate consistency for base feature
            effects = np.array(effects)
            base_consistency = 0.0
            if len(effects) > 0:
                mean_abs_effect = np.mean(np.abs(effects))
                if mean_abs_effect > 0:
                    mad = np.mean(np.abs(effects - np.mean(effects)))
                    base_consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
            
            # Calculate consistency for interactions
            interaction_consistencies = []
            for effects in interaction_effects.values():
                if effects:
                    effects = np.array(effects)
                    mean_abs_effect = np.mean(np.abs(effects))
                    if mean_abs_effect > 0:
                        mad = np.mean(np.abs(effects - np.mean(effects)))
                        consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
                        interaction_consistencies.append(consistency)
            
            # Combine base and interaction consistencies
            if interaction_consistencies:
                return float(np.mean([base_consistency] + interaction_consistencies))
            return float(base_consistency)
            
        except Exception as e:
            logger.error(f"Error calculating effect consistency: {str(e)}")
            return 0.0
                    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics dictionary with zero values."""
        return {
            'model_effect': 0.0,
            'effect_consistency': 0.0,
            'predictive_power': 0.0
        }
 
        