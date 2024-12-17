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

from engram3.utils.config import (Config, FeatureSelectionConfig)
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
        if isinstance(config, dict):
            feature_selection_config = config.get('feature_selection', {})
        else:
            feature_selection_config = config.feature_selection.dict()
        self.config = FeatureSelectionConfig(**feature_selection_config)
        
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
            # Check for discriminative power using differences
            feature_data = model._get_feature_data(feature, dataset)
            if np.std(feature_data['differences']) == 0:
                logger.warning(f"Feature {feature} shows no discrimination between preferences - skipping evaluation")
                return {
                    'model_effect': 0.0,
                    'effect_consistency': 0.0,
                    'predictive_power': 0.0
                }

            logger.debug(f"\nEvaluating feature: {feature}")
            metrics = {}
            
            # Model Effect (normalized by max observed effect)
            try:
                # Create temporary model to evaluate this feature alone
                temp_model = type(model)(config=model.config)
                temp_model.feature_extractor = model.feature_extractor
                temp_model.fit(dataset, [feature])
                
                weights = temp_model.get_feature_weights()
                effect = abs(weights.get(feature, (0.0, 0.0))[0])
                metrics['model_effect'] = effect
                logger.debug(f"Model effect: {effect:.4f}")
            except Exception as e:
                logger.error(f"Error calculating model effect for {feature}: {str(e)}")
                metrics['model_effect'] = 0.0

            # Effect Consistency
            try:
                consistency = self._calculate_effect_consistency(feature, dataset)
                metrics['effect_consistency'] = consistency
                logger.debug(f"Effect consistency: {consistency:.4f}")
            except Exception as e:
                logger.error(f"Error calculating effect consistency for {feature}: {str(e)}")
                metrics['effect_consistency'] = 0.0

            # Predictive Power
            try:
                pred_power = self._calculate_predictive_power(feature, dataset, model)
                metrics['predictive_power'] = pred_power
                logger.debug(f"Predictive power: {pred_power:.4f}")
            except Exception as e:
                logger.error(f"Error calculating predictive power for {feature}: {str(e)}")
                metrics['predictive_power'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            return self._get_default_metrics()
                                                        
    def _calculate_predictive_power(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel') -> float:
        """Calculate how much a feature improves model predictions over random chance."""
        try:
            logger.info(f"\nCalculating predictive power for feature: {feature}")
            
            # Create new model instance for this feature
            model_single = type(model)(config=model.config)
            model_single.feature_extractor = model.feature_extractor
            
            try:
                logger.info("Fitting single feature model...")
                # Fit and evaluate with single feature
                model_single.fit(dataset, [feature])
                
                logger.info("Evaluating single feature model...")
                metrics_single = model_single.evaluate(dataset)
                logger.info(f"Single feature evaluation metrics: {metrics_single}")
                
                acc_single = metrics_single.get('accuracy', 0.5)
                logger.info(f"Single feature accuracy: {acc_single:.4f}")
                
                # Calculate improvement over random chance (0.5)
                improvement = acc_single - 0.5
                logger.info(f"Raw improvement over random: {improvement:.4f}")
                
                # Normalize improvement to [0,1] range
                normalized_improvement = np.clip(improvement / 0.5, 0, 1)
                logger.info(f"Normalized improvement: {normalized_improvement:.4f}")
                
                if normalized_improvement == 0.0:
                    logger.warning(f"Got zero improvement for feature {feature}. Check if model evaluation succeeded.")
                
                return float(normalized_improvement)
                
            finally:
                # Clean up
                if hasattr(model_single, 'fit_result'):
                    del model_single.fit_result
                del model_single
                
        except Exception as e:
            logger.error(f"Error calculating predictive power for {feature}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return 0.0
                        
    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset) -> float:
        """
        Calculate consistency of feature effect across cross-validation splits.
        Returns a value between 0 (inconsistent) and 1 (consistent).
        """
        try:
            n_splits = 5
            effects = []
            
            for train_idx, val_idx in self.model._get_cv_splits(dataset, n_splits):
                train_data = dataset._create_subset_dataset(train_idx)
                self.model.fit(train_data, [feature])
                weights = self.model.get_feature_weights()
                if feature in weights:
                    effects.append(weights[feature][0])
                    
            if not effects:
                return 0.0
                
            effects = np.array(effects)
            
            # Calculate consistency using absolute values
            mean_abs_effect = np.mean(np.abs(effects))
            if mean_abs_effect == 0:
                return 0.0
                
            # Use mean absolute deviation instead of std
            mad = np.mean(np.abs(effects - np.mean(effects)))
            
            # Normalize to [0,1] range
            consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
            
            return float(consistency)
            
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
 
        