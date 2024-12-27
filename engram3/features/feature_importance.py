# engram3/features/feature_importance.py
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
                                                            
    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel',
                                all_features: List[str], current_selected_features: List[str]) -> Dict[str, float]:
            """Evaluate a feature's importance using multiple metrics."""
            try:
                # Only log the start of evaluation and baseline computation once
                if self._baseline_accuracy is None:
                    logger.info("\nComputing initial baseline model...")
                    base_model = type(model)(config=model.config)
                    base_model.feature_extractor = model.feature_extractor
                    base_model.feature_names = list(model.config.features.control_features)
                    base_model.selected_features = list(model.config.features.control_features)
                    base_model.is_baseline_model = True
                    
                    base_model.fit(dataset, base_model.selected_features)
                    base_metrics = base_model.evaluate(dataset)
                    self._baseline_accuracy = base_metrics.get('accuracy', 0.5)
                    logger.info(f"Baseline accuracy: {self._baseline_accuracy:.4f}")

                # Show clear feature evaluation header
                logger.info(f"\n{'='*32}")
                # Add better interaction display
                if '_x_' in feature:
                    components = feature.split('_x_')
                    logger.info(f"EVALUATING {len(components)}-WAY INTERACTION:")
                    logger.info(f"  {' × '.join(components)}")
                else:
                    logger.info(f"EVALUATING BASE FEATURE:")
                    logger.info(f"  {feature}")
                logger.info(f"Context: {', '.join(current_selected_features)}")
                logger.info(f"{'-'*32}")

                # Set up test features silently
                control_features = list(model.config.features.control_features)
                test_features = control_features.copy()
                is_control_feature = feature in control_features
                
                if not is_control_feature:
                    test_features.append(feature)
                    # Handle higher-order interactions
                    for interaction in model.config.features.interactions:
                        # Check if feature is part of this interaction
                        if feature in interaction:
                            # Only add interaction if all other components are already selected
                            other_components = [f for f in interaction if f != feature]
                            if all(f in current_selected_features for f in other_components):
                                interaction_name = '_x_'.join(sorted(interaction))
                                if interaction_name in all_features:
                                    test_features.append(interaction_name)

                # Fit model silently
                model_single = type(model)(config=model.config)
                model_single.feature_extractor = model.feature_extractor
                model_single.feature_names = test_features
                model_single.selected_features = test_features
                model_single.is_baseline_model = len(test_features) == len(control_features)
                model_single.fit(dataset, test_features)

                # Calculate metrics
                weights = model_single.get_feature_weights()
                raw_effect = abs(weights.get(feature, (0.0, 0.0))[0])
                category = 'control' if is_control_feature else 'main'
                
                # Update and normalize effect
                self._max_effect[category] = max(self._max_effect[category], raw_effect)
                self._max_effect_seen = max(self._max_effect_seen, raw_effect)
                model_effect = raw_effect / self._max_effect_seen if self._max_effect_seen > 0 else 0.0

                # Calculate consistency
                raw_consistency = self._calculate_effect_consistency(
                    feature, dataset, current_selected_features + control_features)
                self._max_consistency[category] = max(self._max_consistency[category], raw_consistency)
                self._max_consistency_seen = max(self._max_consistency_seen, raw_consistency)
                effect_consistency = raw_consistency / self._max_consistency_seen if self._max_consistency_seen > 0 else 0.0

                # Calculate predictive power with detailed logging
                metrics = model_single.evaluate(dataset)
                test_accuracy = metrics.get('accuracy', 0.5)
                epsilon = 1e-10

                logger.info("\nPredictive Power Details:")
                logger.info(f"  Test accuracy:      {test_accuracy:.6f}")
                logger.info(f"  Baseline accuracy:  {self._baseline_accuracy:.6f}")
                logger.info(f"  Raw difference:     {test_accuracy - self._baseline_accuracy:+.6f}")

                # Calculate improvement
                raw_improvement = max(0.0, test_accuracy - self._baseline_accuracy)
                logger.info(f"  Clipped improve:    {raw_improvement:.6f}")
                
                # Calculate maximum possible improvement
                max_possible = 1.0 - self._baseline_accuracy
                logger.info(f"  Max possible:       {max_possible:.6f}")

                # Calculate final power
                predictive_power = raw_improvement / max_possible if max_possible > epsilon else 0.0
                logger.info(f"  Final power:        {predictive_power:.6f}")

                # Also log AUC if available
                if 'auc' in metrics:
                    logger.info(f"  AUC:               {metrics['auc']:.6f}")
                
                # Log any improvements in other metrics
                for metric_name, value in metrics.items():
                    if metric_name not in ['accuracy', 'auc']:
                        logger.info(f"  {metric_name}:          {value:.6f}")

                # Log results clearly
                logger.info("Results:")
                logger.info(f"  Effect:      {model_effect:.4f} (raw: {raw_effect:.4f})")
                logger.info(f"  Consistency: {effect_consistency:.4f} (raw: {raw_consistency:.4f})")
                logger.info(f"  Power:       {predictive_power:.4f} (acc: {test_accuracy:.4f})")
                weight, std = weights.get(feature, (0.0, 0.0))
                logger.info(f"  Weight:      {weight:.4f} ± {std:.4f}")
                logger.info(f"{'='*32}\n")

                # Cache raw values
                self.metric_cache.set(f"{feature}_raw_effect", raw_effect)
                self.metric_cache.set(f"{feature}_raw_consistency", raw_consistency)
                self.metric_cache.set(f"{feature}_raw_improvement", raw_improvement)
                self.metric_cache.set(f"{feature}_max_possible", max_possible)
                self.metric_cache.set(f"{feature}_test_accuracy", test_accuracy)

                return {
                    'model_effect': model_effect,
                    'effect_consistency': effect_consistency,
                    'predictive_power': predictive_power,
                    'weight': weight,
                    'weight_std': std,
                    'test_accuracy': test_accuracy,
                    'baseline_accuracy': self._baseline_accuracy
                }
                            
            except Exception as e:
                logger.error(f"Error evaluating {feature}: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                return self._get_default_metrics()
                                                                                                                
    def _format_interaction_name(self, feature: str) -> str:
            """Format interaction name for display."""
            if '_x_' in feature:
                components = feature.split('_x_')
                return f"{feature} ({' × '.join(components)})"

    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset, 
                                    current_features: List[str]) -> float:
        """
        Calculate consistency of feature effect across cross-validation splits.
        Handles both base features and higher-order interactions.
        
        Args:
            feature: Feature to evaluate
            dataset: Dataset for evaluation
            current_features: List of currently selected features including controls
            
        Returns:
            float: Consistency score between 0 (inconsistent) and 1 (consistent)
        """
        try:
            n_splits = 5
            effects = []
            interaction_effects = defaultdict(list)
            
            for train_idx, val_idx in self.model._get_cv_splits(dataset, n_splits):
                train_data = dataset._create_subset_dataset(train_idx)
                
                # Use all current features when evaluating consistency
                features_to_test = list(current_features)  # Copy to avoid modifying input
                if feature not in features_to_test:
                    features_to_test.append(feature)
                                                        
                self.model.fit(train_data, features_to_test)
                weights = self.model.get_feature_weights()
                
                # Get feature effect
                if feature in weights:
                    effects.append(weights[feature][0])
                
                # Handle n-way interactions
                for f in weights:
                    if '_x_' in f:
                        components = f.split('_x_')
                        # Check if this feature participates in the interaction
                        if feature in components:
                            interaction_effects[f].append(weights[f][0])
                            # If this is an interaction being evaluated, also track component effects
                            if f == feature:
                                for component in components:
                                    if component in weights and component != feature:
                                        interaction_effects[f'{feature}_with_{component}'].append(
                                            weights[component][0])
            
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
            
            # Calculate consistency for interactions and components
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
                # Weight the consistencies by the number of interacting features
                if '_x_' in feature:
                    n_components = len(feature.split('_x_'))
                    # Give more weight to higher-order interaction consistency
                    weights = [n_components if i == 0 else 1 for i in range(len(interaction_consistencies))]
                    return float(np.average([base_consistency] + interaction_consistencies, 
                                        weights=weights))
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
 
