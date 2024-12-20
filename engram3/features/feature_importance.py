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
from typing import Dict, List, Union
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
                                                            
    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, model: 'PreferenceModel', 
                        all_features: List[str], current_selected_features: List[str]) -> Dict[str, float]:
        """Calculate globally-normalized metrics that work for both selection and analysis."""
        try:
            # Fit model with current feature set + candidate
            test_features = current_selected_features + [feature]
            model_single = type(model)(config=model.config)
            model_single.fit(dataset, test_features)
            
            # Get weights for normalization scale across ALL features
            all_effects = []
            for f in all_features:
                # Get raw weight magnitude for feature AND any interactions it's part of
                f_weight = abs(model_single.get_feature_weights().get(f, (0.0, 0.0))[0])
                all_effects.append(f_weight)
                # Add interaction weights if any
                for f1, f2 in model.config.features.interactions:
                    if f in (f1, f2):
                        int_name = f"{f1}_x_{f2}"
                        alt_name = f"{f2}_x_{f1}"
                        if int_name in all_features:
                            int_weight = abs(model_single.get_feature_weights().get(int_name, (0.0, 0.0))[0])
                            all_effects.append(int_weight)
                        elif alt_name in all_features:
                            int_weight = abs(model_single.get_feature_weights().get(alt_name, (0.0, 0.0))[0])
                            all_effects.append(int_weight)

            # Normalize model effect against global maximum
            raw_effect = abs(model_single.get_feature_weights().get(feature, (0.0, 0.0))[0])
            max_effect = max(all_effects) if all_effects else 1.0
            model_effect = raw_effect / max_effect if max_effect > 0 else 0.0
            
            # Calculate predictive power against global baseline
            metrics = model_single.evaluate(dataset)
            accuracy = metrics.get('accuracy', 0.5)
            baseline_accuracy = 0.5  # Random chance baseline
            max_possible = 1.0 - baseline_accuracy
            predictive_power = (accuracy - baseline_accuracy) / max_possible if max_possible > 0 else 0.0
            
            # Effect consistency through CV (already globally normalized)
            effect_consistency = self._calculate_effect_consistency(
                feature, dataset, current_selected_features)
            
            # Get actual weights
            weights = model_single.get_feature_weights()
            weight, weight_std = weights.get(feature, (0.0, 0.0))
            
            return {
                'model_effect': model_effect,
                'effect_consistency': effect_consistency,
                'predictive_power': predictive_power,
                'weight': weight,
                'weight_std': weight_std
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {feature}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self._get_default_metrics()
                                                    
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
 
        