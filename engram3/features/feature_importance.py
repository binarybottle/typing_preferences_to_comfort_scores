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
                logger.warning("No control features defined, using dummy feature")
                return 0.5  # Random chance baseline
                
            try:
                baseline_model.fit(dataset, control_features, 
                                fit_purpose="Computing baseline accuracy")
                metrics = baseline_model.evaluate(dataset)
                return float(metrics['accuracy'])
                
            finally:
                baseline_model.cleanup()
                
        except Exception as e:
            logger.error(f"Error computing baseline accuracy: {str(e)}")
            return 0.5  # Return random chance on error

    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel',
                        all_features: List[str], current_selected_features: List[str]) -> Dict[str, float]:
        """Evaluate a feature's importance using multiple metrics."""
        try:
            # Input validation
            if not dataset or not model:
                logger.error("Missing required parameters")
                return self._get_default_metrics()
            if feature not in all_features:
                logger.error(f"Feature {feature} not in all_features")
                return self._get_default_metrics()

            # Compute baseline once and cache it
            if self._baseline_accuracy is None:
                self._baseline_accuracy = self._compute_baseline_accuracy(dataset, model)
            
            # Log evaluation header
            logger.info(f"\n{'='*32}")
            if '_x_' in feature:
                components = feature.split('_x_')
                logger.info(f"EVALUATING {len(components)}-WAY INTERACTION:")
                logger.info(f"  {' × '.join(components)}")
                # Check if all components are available
                missing_components = [c for c in components 
                                    if c not in current_selected_features 
                                    and c not in model.config.features.control_features]
                if missing_components:
                    logger.info(f"Missing components: {missing_components}")
                    return self._get_default_metrics()
            else:
                logger.info(f"EVALUATING BASE FEATURE:")
                logger.info(f"  {feature}")
                
            logger.info(f"Context: {', '.join(current_selected_features)}")
            logger.info(f"{'-'*32}")
            
            # Setup evaluation model
            eval_model = type(model)(config=model.config)
            eval_model.feature_extractor = model.feature_extractor
            eval_features = (list(model.config.features.control_features) + 
                            current_selected_features + [feature])
            eval_model.feature_names = eval_features
            eval_model.selected_features = eval_features
            
            try:
                # Fit evaluation model
                eval_model.fit(dataset, eval_features, 
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
            # Get accuracy with this feature
            metrics = model.evaluate(dataset)
            feature_accuracy = metrics['accuracy']
            
            # Calculate improvement over baseline
            improvement = feature_accuracy - baseline_accuracy
            
            # Normalize to [0,1] range assuming max 0.5 improvement possible
            normalized_power = np.clip(improvement / 0.5, 0, 1)
            
            return float(normalized_power)
            
        except Exception as e:
            logger.error(f"Error calculating predictive power: {str(e)}")
            return 0.0

    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset, 
                                    model: 'PreferenceModel', current_features: List[str]) -> float:
        """Calculate consistency of feature effect across cross-validation splits."""
        try:
            n_splits = 5
            effects = []
            interaction_effects = defaultdict(list)
            
            # Get cross-validation splits
            cv_splits = model._get_cv_splits(dataset, n_splits)
            
            for split_idx, (train_idx, val_idx) in enumerate(cv_splits, 1):
                train_data = dataset._create_subset_dataset(train_idx)
                
                features_to_test = list(current_features)
                if feature not in features_to_test:
                    features_to_test.append(feature)
                                        
                # Create new model instance for this split
                split_model = type(model)(config=model.config)
                split_model.feature_extractor = model.feature_extractor
                
                try:
                    logger.info(f"\nCalculating consistency for {feature}:")
                    split_model.fit(train_data, features_to_test, 
                                fit_purpose=f"Cross-validation split {split_idx}/5 for {feature}")

                    weights = split_model.get_feature_weights()
                    
                    # Get feature effect
                    if feature in weights:
                        effects.append(weights[feature][0])  # Store raw effect (not absolute)
                    
                    # Handle n-way interactions
                    if '_x_' in feature:
                        components = feature.split('_x_')
                        # Track component effects
                        for component in components:
                            if component in weights and component != feature:
                                interaction_effects[f'{feature}_with_{component}'].append(
                                    weights[component][0])  # Store raw effects
                
                except Exception as e:
                    logger.error(f"Error in split {split_idx}: {str(e)}")
                    continue
                finally:
                    split_model.cleanup()
            
            if not effects and not interaction_effects:
                return 0.0
            
            # Calculate consistency for base feature
            effects = np.array(effects)
            base_consistency = 0.0
            if len(effects) > 0:
                # Calculate magnitude consistency
                logger.info(f"  Raw effects across folds: {effects}")
                mean_effect = np.mean(effects)  # Use raw mean, not absolute
                mean_abs_effect = np.mean(np.abs(effects))
                logger.info(f"  Mean effect: {mean_effect:.4f}")
                logger.info(f"  Mean absolute effect: {mean_abs_effect:.4f}")

                if mean_abs_effect > 0:
                    mad = np.mean(np.abs(effects - mean_effect))  # Deviation from raw mean
                    magnitude_consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
                else:
                    magnitude_consistency = 0.0
                    
                # Calculate direction consistency
                sign_consistency = np.mean(np.sign(effects) == np.sign(mean_effect))
                
                # Combine both metrics - a feature must be consistent in both magnitude and direction
                base_consistency = magnitude_consistency * sign_consistency

                logger.info(f"  Magnitude consistency: {magnitude_consistency:.4f}")
                logger.info(f"  Sign consistency: {sign_consistency:.4f}")
                logger.info(f"  Final consistency score: {base_consistency:.4f}")
            else:
                logger.warning(f"  No effects collected for {feature}")
            
            # Calculate consistency for interactions and components
            interaction_consistencies = []
            for effects in interaction_effects.values():
                if effects:
                    effects = np.array(effects)
                    mean_effect = np.mean(effects)
                    mean_abs_effect = np.mean(np.abs(effects))
                    if mean_abs_effect > 0:
                        mad = np.mean(np.abs(effects - mean_effect))
                        magnitude_consistency = 1.0 - np.clip(mad / mean_abs_effect, 0, 1)
                        sign_consistency = np.mean(np.sign(effects) == np.sign(mean_effect))
                        consistency = magnitude_consistency * sign_consistency
                        interaction_consistencies.append(consistency)
            
            # Combine consistencies with proper weighting
            if interaction_consistencies:
                if '_x_' in feature:
                    n_components = len(feature.split('_x_'))
                    weights = [n_components if i == 0 else 1 
                            for i in range(len(interaction_consistencies))]
                    return float(np.average([base_consistency] + interaction_consistencies, 
                                        weights=weights))
                return float(np.mean([base_consistency] + interaction_consistencies))
            
            return float(base_consistency)
            
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
 
