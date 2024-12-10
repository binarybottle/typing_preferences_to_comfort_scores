# engram3/models/bayesian.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings 
from scipy import stats 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
from collections import defaultdict

from engram3.data import PreferenceDataset

logger = logging.getLogger(__name__)

class BayesianPreferenceModel():
    
    def __init__(self, config: Optional[Dict] = None):
        self.feature_weights = None
        self.config = config or {}
        self.n_samples = self.config.get('model', {}).get('n_samples', 10000)
        self.n_chains = self.config.get('model', {}).get('chains', 8)
        self.target_accept = self.config.get('model', {}).get('target_accept', 0.85)
        
    def fit(self, dataset: PreferenceDataset) -> None:
        """Fit the model to training data."""
        try:
            # Get feature names once
            feature_names = dataset.get_feature_names()
            self.feature_weights = {name: 0.0 for name in feature_names}
            
            # Calculate feature differences and handle None/NaN
            for feature in feature_names:
                diffs = []
                prefs = []
                for pref in dataset.preferences:
                    val1 = pref.features1.get(feature)
                    val2 = pref.features2.get(feature)
                    
                    # Skip if either value is None/NaN
                    if val1 is None or val2 is None:
                        continue
                    
                    try:
                        diff = float(val1) - float(val2)
                        if not np.isnan(diff):
                            diffs.append(diff)
                            prefs.append(1.0 if pref.preferred else -1.0)
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error calculating difference for feature {feature}: {e}")
                        continue
                
                if diffs:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        correlation = stats.spearmanr(diffs, prefs).correlation
                        self.feature_weights[feature] = (
                            correlation if not np.isnan(correlation) else 0.0
                        )
                        
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise

    def predict_preference(self, bigram1: str, bigram2: str) -> float:
        """Predict preference probability."""
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before making predictions")
            
        try:
            # Extract features for both bigrams
            from ..features.extraction import extract_bigram_features, extract_same_letter_features
            from ..features.keymaps import (
                column_map, row_map, finger_map,
                engram_position_values, row_position_values
            )
            
            # Get features for each bigram
            if bigram1[0] == bigram1[1]:  # Same-letter bigram
                features1 = extract_same_letter_features(
                    bigram1[0], 
                    column_map, finger_map,
                    engram_position_values, row_position_values
                )
            else:
                features1 = extract_bigram_features(
                    bigram1[0], bigram1[1],
                    column_map, row_map, finger_map,
                    engram_position_values, row_position_values
                )
                
            if bigram2[0] == bigram2[1]:  # Same-letter bigram
                features2 = extract_same_letter_features(
                    bigram2[0],
                    column_map, finger_map,
                    engram_position_values, row_position_values
                )
            else:
                features2 = extract_bigram_features(
                    bigram2[0], bigram2[1],
                    column_map, row_map, finger_map,
                    engram_position_values, row_position_values
                )
            
            # Calculate scores using feature weights
            score1 = sum(self.feature_weights.get(f, 0.0) * features1.get(f, 0.0) 
                        for f in self.feature_weights)
            score2 = sum(self.feature_weights.get(f, 0.0) * features2.get(f, 0.0) 
                        for f in self.feature_weights)
            
            # Convert to probability using sigmoid
            return 1 / (1 + np.exp(-(score1 - score2)))
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 0.5  # Return uncertainty in case of error
        
    def get_feature_weights(self) -> Dict[str, float]:
        """Get the learned feature weights."""
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before getting weights")
        return self.feature_weights.copy()
        
    def cross_validate(
        self, 
        dataset: PreferenceDataset, 
        n_splits: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation with participant-based splits."""
        if n_splits is None:
            n_splits = self.config.get('model', {}).get('cross_validation', {}).get('n_splits', 5)
        
        # Get feature names upfront
        feature_names = dataset.get_feature_names()
        logger.debug(f"Features for cross-validation: {feature_names}")
        
        # Initialize metrics storage
        metrics = defaultdict(list)
        #feature_effects = {feature: [] for feature in feature_names}  # Initialize for all features
        
        # Get participant IDs and prepare data arrays
        participant_ids = np.array([p.participant_id for p in dataset.preferences])
        all_preferences = []
        all_features1 = []
        all_features2 = []
        
        for pref in dataset.preferences:
            all_preferences.append(float(pref.preferred))
            feat1 = [pref.features1[f] for f in feature_names]
            feat2 = [pref.features2[f] for f in feature_names]
            all_features1.append(feat1)
            all_features2.append(feat2)
            
        X1 = np.array(all_features1)
        X2 = np.array(all_features2)
        y = np.array(all_preferences)
        
        # Perform cross-validation
        cv = GroupKFold(n_splits=n_splits)
        
        feature_effects = defaultdict(list)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X1, y, groups=participant_ids)):
            try:
                logger.info(f"Processing fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets with all required arguments
                train_data = self._create_subset_dataset(
                    dataset, train_idx, X1, X2, y, participant_ids)
                val_data = self._create_subset_dataset(
                    dataset, val_idx, X1, X2, y, participant_ids)
                
                if len(train_data.preferences) == 0 or len(val_data.preferences) == 0:
                    logger.warning(f"Empty split in fold {fold + 1}, skipping")
                    continue
                
                # Fit model on training data
                self.fit(train_data)
                
                # Get predictions on validation set
                val_predictions = []
                val_true = []
                
                for pref in val_data.preferences:
                    try:
                        pred = self.predict_preference(pref.bigram1, pref.bigram2)
                        if not np.isnan(pred):  # Skip NaN predictions
                            val_predictions.append(pred)
                            val_true.append(1.0 if pref.preferred else 0.0)
                    except Exception as e:
                        logger.warning(f"Prediction failed for {pref.bigram1}-{pref.bigram2}: {str(e)}")
                        continue
                
                if not val_predictions:  # Skip fold if no valid predictions
                    logger.warning(f"No valid predictions in fold {fold + 1}, skipping")
                    continue
                
                val_predictions = np.array(val_predictions)
                val_true = np.array(val_true)
                
                # Calculate metrics
                metrics['accuracy'].append(accuracy_score(val_true, val_predictions > 0.5))
                metrics['auc'].append(roc_auc_score(val_true, val_predictions))
                
                # Get feature weights and store them properly
                weights = self.get_feature_weights()
                if weights:  # Add debug logging
                    logger.debug(f"Fold {fold + 1} weights: {weights}")
                    for feature, weight in weights.items():
                        if not np.isnan(weight):
                            feature_effects[feature].append(weight)
                else:
                    logger.warning(f"No weights obtained in fold {fold + 1}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        # Process feature effects with better error handling
        processed_effects = {}
        for feature, effects in feature_effects.items():
            if effects:  # Only process if we have effects
                processed_effects[feature] = {
                    'mean': float(np.mean(effects)),
                    'std': float(np.std(effects)),
                    'values': effects
                }
                logger.debug(f"Feature {feature}: {len(effects)} effects, mean={processed_effects[feature]['mean']:.3f}")
            else:
                logger.warning(f"No effects collected for feature {feature}")
        
        results = {
            'metrics': metrics,
            'feature_effects': processed_effects,
            'stability': self._calculate_stability_metrics(feature_effects)
        }
        
        logger.debug(f"Cross-validation complete. Features with effects: {list(processed_effects.keys())}")
        return results
            
    def _create_subset_dataset(
        self, 
        dataset: PreferenceDataset,
        indices: np.ndarray,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        participants: np.ndarray
    ) -> PreferenceDataset:
        """Create a new dataset from a subset of indices."""
        subset = PreferenceDataset.__new__(PreferenceDataset)
        subset.preferences = [dataset.preferences[i] for i in indices]
        subset.participants = set(participants[indices])
        subset.feature_names = dataset.get_feature_names()
        # Copy over needed attributes from original dataset
        subset.file_path = dataset.file_path
        subset.all_bigrams = dataset.all_bigrams
        subset.all_bigram_features = dataset.all_bigram_features
        return subset
    
    def _calculate_stability_metrics(
        self, 
        feature_effects: Dict[str, List[float]],
        config: Optional[Dict] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature stability metrics with proper error handling.
        
        Args:
            feature_effects: Dictionary mapping feature names to lists of effects across folds
            config: Configuration dictionary with validation settings
            
        Returns:
            Dictionary containing stability metrics for each feature
        """
        stability = {}
        
        # Get validation settings from config
        if config and 'feature_evaluation' in config:
            validation_config = config['feature_evaluation'].get('validation', {})
            perform_stability_check = validation_config.get('perform_stability_check', True)
            outlier_detection = validation_config.get('outlier_detection', True)
            outlier_threshold = validation_config.get('outlier_threshold', 3.0)
        else:
            perform_stability_check = True
            outlier_detection = True
            outlier_threshold = 3.0
        
        for feature, effects in feature_effects.items():
            if not effects:  # Handle empty lists
                stability[feature] = {
                    'effect_mean': 0.0,
                    'effect_std': 0.0,
                    'effect_cv': float('inf'),
                    'sign_consistency': 0.0,
                    'relative_range': float('inf'),
                    'is_stable': False,
                    'n_outliers': 0
                }
                continue
            
            # Convert to numpy array and remove any NaN values
            effects = np.array(effects)
            effects = effects[~np.isnan(effects)]  # Remove NaN values
            
            if len(effects) == 0:  # Skip if no valid effects
                stability[feature] = {
                    'effect_mean': 0.0,
                    'effect_std': 0.0,
                    'effect_cv': float('inf'),
                    'sign_consistency': 0.0,
                    'relative_range': float('inf'),
                    'is_stable': False,
                    'n_outliers': 0,
                    'n_samples': 0
                }
                continue
            
            # Outlier detection if enabled
            if outlier_detection and len(effects) > 1:  # Need at least 2 points
                effects_std = np.std(effects)
                if effects_std > 0:  # Only calculate z-scores if std > 0
                    z_scores = np.abs((effects - np.mean(effects)) / effects_std)
                    outliers = z_scores > outlier_threshold
                    n_outliers = np.sum(outliers)
                    
                    if n_outliers > 0:
                        logger.warning(f"Feature '{feature}' has {n_outliers} outlier effects")
                        effects = effects[~outliers]
                else:
                    n_outliers = 0
            else:
                n_outliers = 0
            
            if len(effects) == 0:  # Check again after outlier removal
                continue
            
            # Calculate basic statistics
            mean_effect = np.mean(effects)
            std_effect = np.std(effects)
            
            # Calculate stability metrics with error handling
            try:
                effect_cv = std_effect / mean_effect if mean_effect != 0 else float('inf')
            except:
                effect_cv = float('inf')
                
            try:
                sign_consistency = np.mean(np.sign(effects) == np.sign(mean_effect))
            except:
                sign_consistency = 0.0
                
            try:
                relative_range = ((np.max(effects) - np.min(effects)) / 
                                np.abs(mean_effect)) if mean_effect != 0 else float('inf')
            except:
                relative_range = float('inf')
            
            # Determine if feature is stable based on metrics
            is_stable = (
                effect_cv < 1.0 and  # Coefficient of variation less than 100%
                sign_consistency > 0.8 and  # Consistent sign in >80% of folds
                relative_range < 2.0  # Range less than 2x mean effect
            ) if perform_stability_check else True
            
            stability[feature] = {
                'effect_mean': float(mean_effect),
                'effect_std': float(std_effect),
                'effect_cv': float(effect_cv),
                'sign_consistency': float(sign_consistency),
                'relative_range': float(relative_range),
                'is_stable': bool(is_stable),
                'n_outliers': int(n_outliers),
                'n_samples': len(effects)
            }
            
            # Log stability information
            if not is_stable:
                reasons = []
                if effect_cv >= 1.0:
                    reasons.append("high variation")
                if sign_consistency <= 0.8:
                    reasons.append("inconsistent sign")
                if relative_range >= 2.0:
                    reasons.append("large range")
                logger.warning(f"Feature '{feature}' is unstable: {', '.join(reasons)}")
        
        return stability
