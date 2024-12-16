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
            # Check for zero variance
            feature_data = model._get_feature_data(feature, dataset)
            if np.std(feature_data['values']) == 0:
                logger.warning(f"Feature {feature} has zero variance - skipping evaluation")
                return {
                    'combined_score': 0.0,
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

            # Calculate combined score using normalized metrics
            try:
                combined_score = self._calculate_combined_score(**metrics)
                metrics['combined_score'] = combined_score
                logger.debug(f"Final combined score: {combined_score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating combined score for {feature}: {str(e)}")
                metrics['combined_score'] = 0.0

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            return self._get_default_metrics()
        
    def _calculate_combined_score(self, **metrics) -> float:
        """
        Calculate combined score from individual metrics for feature importance evaluation.
        
        Args:
            **metrics: Keyword arguments containing:
                - model_effect: Absolute effect size
                - effect_consistency: Cross-validation based consistency
                - predictive_power: Improvement in model predictions
                
        Returns:
            float: Combined score between 0 and 1, used to evaluate feature importance
        """
        try:
            # Get individual components with defaults
            effect = metrics.get('model_effect', 0.0)
            consistency = metrics.get('effect_consistency', 0.0)
            predictive = metrics.get('predictive_power', 0.0)
            
            # Normalize effect size
            max_effect = 1.0  # Adjust based on your data
            normalized_effect = min(abs(effect) / max_effect, 1.0)
            
            # Calculate weighted score
            weights = {
                'effect': 0.4,      # Weight for effect size
                'consistency': 0.3,  # Weight for consistency
                'predictive': 0.3    # Weight for predictive power
            }
            
            score = (
                weights['effect'] * normalized_effect +
                weights['consistency'] * max(0, consistency) +  # Clip negative consistency
                weights['predictive'] * predictive
            )
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            logger.error(f"Error calculating combined score: {str(e)}")
            return 0.0
                                        
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
            'combined_score': 0.0
        }
 
        