# engram3/features/feature_importance.py
"""
Feature importance calculation and selection for keyboard layout preference analysis.

Core functionality:
- Statistical evaluation of individual features using multiple metrics:
  * model_effect: Direct impact on predictions
  * effect_consistency: Stability across cross-validation splits
  * predictive_power: Incremental prediction improvement
- Interaction feature analysis and evaluation
- Round-robin tournament selection process
- Cross-validation based stability assessment
- Performance optimization via result caching
- Comprehensive metric reporting

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
from collections import defaultdict
import copy

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

    def reset_normalization_factors(self):
        """Reset global normalization tracking at start of feature selection."""
        self._max_effect = {'control': 0.0, 'main': 0.0}
        self._max_consistency = {'control': 0.0, 'main': 0.0}
        self._baseline_accuracy = None

    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'feature_values_cache'):
            self.feature_values_cache.clear()

    def _compute_baseline_accuracy(self, dataset: PreferenceDataset, model: 'PreferenceModel') -> float:
        """Compute baseline model accuracy using only control features."""
        try:
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
        
    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, 
                        model: 'PreferenceModel', all_features: List[str],
                        current_selected_features: List[str]) -> Dict[str, float]:
        """Evaluate a feature's importance using multiple metrics."""
        try:
            # Input validation and logging
            logger.info(f"\nEvaluating feature: {feature}")
            logger.info(f"Current selected features: {current_selected_features}")
            
            # Create evaluation context with ALL currently selected features
            evaluation_features = list(dict.fromkeys(
                current_selected_features +  # Already includes control features
                [feature]                    # Add feature being evaluated
            ))
            logger.info(f"Evaluation context: {evaluation_features}")
            
            # Fit model with proper context
            eval_model = type(model)(config=model.config)
            eval_model.feature_extractor = model.feature_extractor
            eval_model.feature_names = evaluation_features
            eval_model.selected_features = evaluation_features
            
            try:
                eval_model.fit(dataset, evaluation_features,
                            fit_purpose=f"Feature evaluation for {feature}")

                # Get feature effect
                weights = eval_model.get_feature_weights()
                if feature not in weights:
                    logger.warning(f"No weight found for feature {feature}")
                    return self._get_default_metrics()
                    
                effect_mean, effect_std = weights[feature]
                
                # Calculate effect consistency
                consistency = self._calculate_effect_consistency(
                    feature, dataset, eval_model, current_selected_features)
                
                # Calculate predictive power
                predictive_power = self._calculate_predictive_power(
                    feature, dataset, eval_model, self._baseline_accuracy)
                
                metrics_dict = {
                    'model_effect': abs(effect_mean),  # Use absolute effect
                    'effect_consistency': consistency,
                    'predictive_power': predictive_power
                }
                
                logger.debug(f"Metrics for {feature}: {metrics_dict}")
                return metrics_dict
                
            finally:
                eval_model.cleanup()
                
        except Exception as e:
            logger.error(f"Error evaluating {feature}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self._get_default_metrics()

    def _calculate_predictive_power(self, feature: str, dataset: PreferenceDataset, 
                                model: 'PreferenceModel', baseline_accuracy: float) -> float:
        """Calculate predictive power as improvement over baseline."""
        try:
            # Ensure we have a valid baseline
            if baseline_accuracy is None:
                logger.warning("No baseline accuracy provided, computing new baseline")
                baseline_accuracy = self._compute_baseline_accuracy(dataset, model)
            
            if baseline_accuracy is None:
                logger.error("Could not compute baseline accuracy")
                return 0.0

            # Get accuracy with this feature
            metrics = model.evaluate(dataset)
            feature_accuracy = metrics.get('accuracy')
            
            if feature_accuracy is None:
                logger.error(f"Could not get accuracy for feature {feature}")
                return 0.0
                
            # Calculate improvement over baseline
            improvement = feature_accuracy - baseline_accuracy
            
            # Log values for debugging
            logger.info(f"Predictive power calculation:")
            logger.info(f"  Baseline accuracy: {baseline_accuracy:.4f}")
            logger.info(f"  Feature accuracy: {feature_accuracy:.4f}")
            logger.info(f"  Improvement: {improvement:.4f}")
            
            # Normalize to [0,1] range assuming max 0.5 improvement possible
            normalized_power = np.clip(improvement / 0.5, 0, 1)
            
            return float(normalized_power)
            
        except Exception as e:
            logger.error(f"Error calculating predictive power: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return 0.0
        
    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset, 
                                    model: 'PreferenceModel', current_features: List[str]) -> float:
        """Calculate consistency of feature effect across cross-validation splits."""
        try:
            logger.info(f"\nCalculating consistency for {feature}:")
            effects = []
            interaction_effects = defaultdict(list)
            
            # Get cross-validation splits
            cv_splits = model._get_cv_splits(dataset, n_splits=5)
            
            for split_idx, (train_idx, val_idx) in enumerate(cv_splits, 1):
                train_data = dataset._create_subset_dataset(train_idx)
                
                # Define features to use in this split
                features_to_test = list(current_features)
                if feature not in features_to_test:
                    features_to_test.append(feature)
                                        
                # Create new model instance with proper state
                split_model = type(model)(config=model.config)
                # Copy necessary attributes
                split_model.feature_extractor = model.feature_extractor
                split_model.feature_names = features_to_test  # Add this
                split_model.selected_features = features_to_test  # Add this
                
                try:
                    split_model.fit(train_data, features_to_test, 
                                fit_purpose=f"Cross-validation split {split_idx}/5 for {feature}")

                    weights = split_model.get_feature_weights()
                    
                    # Get feature effect
                    if feature in weights:
                        effect = weights[feature][0]
                        effects.append(effect)
                        logger.debug(f"Split {split_idx}: effect = {effect:.4f}")
                    else:
                        logger.warning(f"Feature {feature} missing from weights in split {split_idx}")
                        logger.debug(f"Available weights: {list(weights.keys())}")
                    
                    # Handle n-way interactions
                    if '_x_' in feature:
                        components = feature.split('_x_')
                        for component in components:
                            if component in weights and component != feature:
                                interaction_effects[f'{feature}_with_{component}'].append(
                                    weights[component][0])
                
                except Exception as e:
                    logger.error(f"Error in split {split_idx}: {str(e)}")
                    continue
                finally:
                    split_model.cleanup()
            
            if not effects:
                logger.warning(f"No effects collected for {feature}")
                return 0.0
            
            # Calculate consistency metrics
            effects = np.array(effects)
            logger.info(f"Raw effects across folds: {effects}")
            mean_effect = np.mean(effects)
            mean_abs_effect = np.mean(np.abs(effects))
            logger.info(f"Mean effect: {mean_effect:.4f}")
            logger.info(f"Mean absolute effect: {mean_abs_effect:.4f}")

            # Calculate magnitude consistency
            if mean_abs_effect > 0:
                mad = np.mean(np.abs(effects - mean_effect))
                magnitude_consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
            else:
                magnitude_consistency = 0.0
            
            # Calculate direction consistency
            sign_consistency = np.mean(np.sign(effects) == np.sign(mean_effect))
            
            # Combine both metrics
            final_consistency = magnitude_consistency * sign_consistency
            
            logger.info(f"Magnitude consistency: {magnitude_consistency:.4f}")
            logger.info(f"Sign consistency: {sign_consistency:.4f}")
            logger.info(f"Final consistency score: {final_consistency:.4f}")
            
            return float(final_consistency)
                
        except Exception as e:
            logger.error(f"Error calculating effect consistency: {str(e)}")
            return 0.0
                
    def clear_caches(self):
        """Clear all caches."""
        if hasattr(self, 'metric_cache'):
            self.metric_cache.clear()
        if hasattr(self, 'feature_values_cache'):
            self.feature_values_cache.clear()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics dictionary with zero values."""
        return {
            'model_effect': 0.0,
            'effect_consistency': 0.0,
            'predictive_power': 0.0
        }
 
