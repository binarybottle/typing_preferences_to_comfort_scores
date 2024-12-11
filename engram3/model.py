# engram3/models/bayesian.py
"""
Preference learning model for analyzing bigram typing preferences.
Handles feature selection, cross-validation, and model evaluation.
"""
import numpy as np
import warnings 
from scipy import stats 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mutual_info_score, accuracy_score, roc_auc_score
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from engram3.data import PreferenceDataset

logger = logging.getLogger(__name__)

class PreferenceModel():
    
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
            from .features.extraction import extract_bigram_features, extract_same_letter_features
            from .features.keymaps import (
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

        # Get feature names including interactions
        feature_names = dataset.get_feature_names()  # This should include interaction features
        logger.debug(f"Features for cross-validation (including interactions): {feature_names}")        
        # Initialize metrics storage
        metrics = defaultdict(list)
        
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
                
                # Create train/val datasets
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
                        if not np.isnan(pred):
                            val_predictions.append(pred)
                            val_true.append(1.0 if pref.preferred else 0.0)
                    except Exception as e:
                        logger.warning(f"Prediction failed for {pref.bigram1}-{pref.bigram2}: {str(e)}")
                        continue
                
                if not val_predictions:
                    logger.warning(f"No valid predictions in fold {fold + 1}, skipping")
                    continue
                
                val_predictions = np.array(val_predictions)
                val_true = np.array(val_true)
                
                # Calculate metrics
                metrics['accuracy'].append(accuracy_score(val_true, val_predictions > 0.5))
                metrics['auc'].append(roc_auc_score(val_true, val_predictions))
                
                # Store feature weights
                weights = self.get_feature_weights()
                if weights:
                    logger.debug(f"Fold {fold + 1} weights: {weights}")
                    for feature, weight in weights.items():
                        if not np.isnan(weight):
                            feature_effects[feature].append(weight)
                else:
                    logger.warning(f"No weights obtained in fold {fold + 1}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue

        # Process feature effects and calculate metrics
        processed_effects = {}
        importance_metrics = {}
        for feature in feature_names:
            effects = feature_effects.get(feature, [])
            if effects:
                # Calculate effect statistics
                effects = np.array(effects)
                mean_effect = float(np.mean(effects))
                std_effect = float(np.std(effects))
                
                processed_effects[feature] = {
                    'mean': mean_effect,
                    'std': std_effect,
                    'values': effects.tolist()
                }

                # Calculate feature importance metrics
                diffs = []
                prefs = []
                for pref in dataset.preferences:
                    try:
                        diff = pref.features1[feature] - pref.features2[feature]
                        if not np.isnan(diff):
                            diffs.append(diff)
                            prefs.append(1.0 if pref.preferred else -1.0)
                    except (KeyError, TypeError):
                        continue

                if diffs:
                    diffs = np.array(diffs)
                    prefs = np.array(prefs)
                    
                    # Calculate correlation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        correlation = stats.spearmanr(diffs, prefs).correlation
                        if np.isnan(correlation):
                            correlation = 0.0

                    # Calculate mutual information
                    feat_diff_bin = diffs > np.median(diffs)
                    mutual_info = mutual_info_score(feat_diff_bin, prefs > 0)

                    # Calculate combined score
                    combined_score = (
                        0.4 * abs(mean_effect) +
                        0.3 * abs(correlation) +
                        0.3 * mutual_info
                    )

                    importance_metrics[feature] = {
                        'correlation': correlation,
                        'mutual_info': mutual_info,
                        'combined_score': combined_score
                    }

        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(feature_effects)

        # Determine selected features
        importance_threshold = self.config['feature_evaluation']['thresholds']['importance']
        selected_features = [
            feature for feature in feature_names
            if importance_metrics.get(feature, {}).get('combined_score', 0.0) >= importance_threshold
        ]

        # Log results
        logger.info(f"\nFeature selection results:")
        logger.info(f"Selected {len(selected_features)} features")
        
        logger.info("\nSelected features:")
        for feature in sorted(selected_features, 
                            key=lambda x: importance_metrics[x]['combined_score'], 
                            reverse=True):
            score = importance_metrics[feature]['combined_score']
            logger.info(f"- {feature} (importance: {score:.3f})")
        
        logger.info("\nNon-selected features:")
        non_selected = set(feature_names) - set(selected_features)
        for feature in sorted(non_selected, 
                            key=lambda x: importance_metrics[x]['combined_score'], 
                            reverse=True):
            score = importance_metrics[feature].get('combined_score', 0.0)
            logger.info(f"- {feature} (importance: {score:.3f})")

        # Save metrics to CSV
        metrics_file = Path(self.config['feature_evaluation']['metrics_file'])
        self.save_metrics_csv(metrics_file, processed_effects, importance_metrics, stability_metrics, selected_features)

        return {
            'metrics': metrics,
            'selected_features': selected_features,
            'feature_effects': processed_effects,
            'importance_metrics': importance_metrics,
            'stability_metrics': stability_metrics
        }

    def save_metrics_csv(self, csv_file: str, 
                        processed_effects: Dict, 
                        importance_metrics: Dict,
                        stability_metrics: Dict,
                        selected_features: List[str]) -> None:
        """Save all feature metrics to CSV."""            
        with open(csv_file, 'w') as f:
            # Write header
            header = [
                "feature_name",
                "selected",
                "combined_score",
                "model_effect_mean",
                "model_effect_std",
                "correlation",
                "mutual_information",
                "effect_cv",
                "relative_range",
                "sign_consistency"
            ]
            f.write(','.join(header) + '\n')
            
            # Get all features (base + interactions)
            all_features = sorted(processed_effects.keys(), 
                                key=lambda x: (1 if '_x_' in x else 0, x))
            
            # Write data for each feature
            for feature in all_features:
                effects = processed_effects.get(feature, {})
                importance = importance_metrics.get(feature, {})
                stability = stability_metrics.get(feature, {})
                
                # Determine if feature was selected based on importance threshold
                selected = "1" if feature in selected_features else "0"
                
                values = [
                    feature,
                    selected,
                    f"{importance.get('combined_score', 0.0):.6f}",
                    f"{effects.get('mean', 0.0):.6f}",
                    f"{effects.get('std', 0.0):.6f}",
                    f"{importance.get('correlation', 0.0):.6f}",
                    f"{importance.get('mutual_info', 0.0):.6f}",
                    f"{stability.get('effect_cv', 0.0):.6f}",
                    f"{stability.get('relative_range', 0.0):.6f}",
                    f"{stability.get('sign_consistency', 0.0):.6f}"
                ]
                f.write(','.join(values) + '\n')
            
            logger.info(f"Saved feature metrics to {csv_file}")

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
        logger.debug("\nStarting stability metric calculations")
        logger.debug(f"Number of features: {len(feature_effects)}")
        for feature, effects in feature_effects.items():
            logger.debug(f"\nFeature '{feature}':")
            logger.debug(f"Raw effects: {effects}")
            logger.debug(f"Number of effects: {len(effects)}")
            if len(effects) > 0:
                logger.debug(f"Range: [{min(effects)}, {max(effects)}]")

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
            if not effects:  # Skip empty effects
                continue
                
            effects = np.array(effects)
            effects = effects[~np.isnan(effects)]  # Remove NaN values
            
            if len(effects) == 0:
                continue
                
            mean_effect = np.mean(effects)
            std_effect = np.std(effects)
            
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
                    
                        # Recalculate statistics after outlier removal
            mean_effect = np.mean(effects)  # This is crucial - recalculate with cleaned data
            std_effect = np.std(effects)
            logger.debug(f"After outlier removal for {feature}:")
            logger.debug(f"Number of effects: {len(effects)}")
            logger.debug(f"Effects: {effects}")
            logger.debug(f"Mean effect: {mean_effect}")
            logger.debug(f"Std effect: {std_effect}")
            
            # Calculate stability metrics with error handling
            try:
                effect_cv = std_effect / abs(mean_effect) if abs(mean_effect) > 1e-10 else float('inf')
                logger.debug(f"{feature} effect_cv calculation:")
                logger.debug(f"std_effect: {std_effect}, mean_effect: {mean_effect}, cv: {effect_cv}")
            except Exception as e:
                logger.error(f"CV calculation failed for {feature}: {str(e)}")
                effect_cv = float('inf')

            try:
                sign_consistency = np.mean(np.sign(effects) == np.sign(mean_effect))
                logger.debug(f"{feature} sign_consistency calculation:")
                logger.debug(f"sign_consistency: {sign_consistency}")
            except Exception as e:
                logger.error(f"Sign consistency calculation failed for {feature}: {str(e)}")
                sign_consistency = 0.0

            try:
                relative_range = ((np.max(effects) - np.min(effects)) / 
                                abs(mean_effect)) if abs(mean_effect) > 1e-10 else float('inf')
                logger.debug(f"{feature} relative_range calculation:")
                logger.debug(f"max: {np.max(effects)}, min: {np.min(effects)}, range: {relative_range}")
            except Exception as e:
                logger.error(f"Range calculation failed for {feature}: {str(e)}")
                relative_range = float('inf')
   
            # NOW we can determine stability
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

