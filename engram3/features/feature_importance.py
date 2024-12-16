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

    def _calculate_predictive_power(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel') -> float:
        """
        Calculate how much a feature improves model predictions.
        """
        try:
            # Get current feature set and performance
            current_features = model.selected_features
            if feature in current_features:
                features_without = [f for f in current_features if f != feature]
                features_with = current_features
            else:
                features_without = current_features
                features_with = current_features + [feature]

            # Ensure we have at least one feature
            if not features_without:
                features_without = [feature]
            if not features_with:
                features_with = [feature]

            # Validate feature lists
            if len(features_without) == 0 or len(features_with) == 0:
                logger.error(f"Invalid feature sets: without={features_without}, with={features_with}")
                return 0.0

            try:
                # Measure performance without feature
                model_without = type(model)(config=model.config)
                model_without.feature_extractor = model.feature_extractor
                model_without.feature_names = features_without  # Set feature names before fitting
                model_without.fit(dataset, features_without)
                metrics_without = model_without.evaluate(dataset)
                acc_without = metrics_without.get('accuracy', 0.0)

                # Measure performance with feature
                model_with = type(model)(config=model.config)
                model_with.feature_extractor = model.feature_extractor
                model_with.feature_names = features_with  # Set feature names before fitting
                model_with.fit(dataset, features_with)
                metrics_with = model_with.evaluate(dataset)
                acc_with = metrics_with.get('accuracy', 0.0)

                # Calculate improvement
                improvement = acc_with - acc_without
                
                # Normalize improvement to [0,1] range
                max_improvement = 0.5  # Maximum expected improvement from one feature
                normalized_improvement = np.clip(improvement / max_improvement, 0, 1)

                return float(normalized_improvement)

            except Exception as e:
                logger.error(f"Error during model fitting: {str(e)}")
                return 0.0
            finally:
                # Clean up
                if 'model_without' in locals():
                    del model_without
                if 'model_with' in locals():
                    del model_with

        except Exception as e:
            logger.error(f"Error calculating predictive power for {feature}: {str(e)}")
            return 0.0
                                
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
 
        