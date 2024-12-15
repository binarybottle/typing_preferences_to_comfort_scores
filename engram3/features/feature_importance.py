# engram3/features/importance.py
"""
Feature importance calculation module for keyboard layout preference analysis.

Provides comprehensive feature evaluation through:
  - Calculation of multiple importance metrics:
    - model_effect: Feature's normalized impact based on model weights
    - effect_consistency: Feature's stability across cross-validation splits  
    - predictive_power: Model performance improvement from feature
    - combined importance score: Weighted combination of metrics

Key functionality:
  - Efficient caching of feature computations
  - Special handling for interaction features
  - Support for typing time and other keyboard-specific features
  - Cross-validation based stability assessment
  - Detailed metric reporting and logging
  - Error handling and fallback mechanisms

The module centers around the FeatureImportanceCalculator class which:
  - Evaluates individual features using multiple metrics
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

    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'feature_values_cache'):
            self.feature_values_cache.clear()

    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel') -> Dict[str, float]:
        """Calculate all metrics for a feature."""
        try:
            logger.debug(f"\nEvaluating feature: {feature}")
            metrics = {}
            
            # Model Effect (normalized by max observed effect)
            try:
                weights = model.get_feature_weights()
                effect = abs(weights.get(feature, (0.0, 0.0))[0])
                max_effect = max(abs(w[0]) for w in weights.values())
                effect_norm = effect / max_effect if max_effect > 0 else 0.0
                metrics['model_effect'] = effect_norm
                logger.debug(f"Model effect: {effect_norm:.4f}")
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

            # Calculate importance score using normalized metrics
            try:
                importance_score = self._calculate_combined_score(**metrics)
                metrics['importance_score'] = importance_score
                logger.debug(f"Final importance score: {importance_score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating importance score for {feature}: {str(e)}")
                metrics['importance_score'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            return self._get_default_metrics()

    def _calculate_combined_score(self, **metrics) -> float:
        """Calculate combined importance score."""
        try:
            weights = self.config.metric_weights
            return (
                weights['model_effect'] * metrics.get('model_effect', 0.0) +
                weights['effect_consistency'] * (1.0 - metrics.get('effect_consistency', 0.0)) +
                weights['predictive_power'] * metrics.get('predictive_power', 0.0)
            )
        except Exception as e:
            logger.error(f"Error calculating combined score: {str(e)}")
            return 0.0

    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset) -> float:
        """Calculate consistency of feature effect across cross-validation splits."""
        try:
            n_splits = 5  # Could move to config if needed
            effects = []
            
            # Use model's CV splitting method
            for train_idx, val_idx in self.model._get_cv_splits(dataset, n_splits):
                train_data = dataset._create_subset_dataset(train_idx)
                self.model.fit(train_data, [feature])
                weights = self.model.get_feature_weights()
                if feature in weights:
                    effects.append(weights[feature][0])
                    
            if not effects:
                return 0.0
                
            # Calculate consistency as 1 - coefficient of variation
            return 1.0 - (np.std(effects) / (abs(np.mean(effects)) + 1e-10))
            
        except Exception as e:
            logger.error(f"Error calculating effect consistency: {str(e)}")
            return 0.0

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics dictionary with zero values."""
        return {
            'model_effect': 0.0,
            'effect_consistency': 0.0,
            'predictive_power': 0.0,
            'importance_score': 0.0
        }
 
        