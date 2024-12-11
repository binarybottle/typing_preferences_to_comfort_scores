# engram3/model.py
"""
Preference learning model for analyzing bigram typing preferences.
Handles feature selection, cross-validation, and model evaluation.

  - Models latent comfort scores directly through features
  - Includes participant random effects
  - Uses proper Bayesian inference for uncertainty estimation
  - Maintains transitivity through Bradley-Terry structure
  - Can predict preferences for new participants
  - Can estimate absolute comfort scores for any bigram

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
import pymc as pm
import aesara.tensor as at

from engram3.data import PreferenceDataset

logger = logging.getLogger(__name__)

class PreferenceModel:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_weights = None
        self.participant_effects = None
        self.model = None
        
    def fit(self, dataset: PreferenceDataset, features: Optional[List[str]] = None):
        """
        Fit hierarchical feature-based Bradley-Terry model.
        """
        feature_names = features if features is not None else dataset.get_feature_names()
        n_features = len(feature_names)
        n_participants = len(dataset.participants)
        
        # Extract feature matrices and preferences
        X1, X2, y, participant_ids = self._prepare_data(dataset, feature_names)
        
        with pm.Model() as self.model:
            # Global feature weights (fixed effects)
            feature_weights = pm.Normal('feature_weights', 
                                     mu=0, sigma=1, 
                                     shape=n_features)
            
            # Participant-level random effects
            participant_effects = pm.Normal('participant_effects',
                                         mu=0, sigma=1,
                                         shape=n_participants)
            
            # Compute latent comfort scores for each bigram
            comfort_score1 = at.dot(X1, feature_weights)
            comfort_score2 = at.dot(X2, feature_weights)
            
            # Add participant effects
            comfort_score1 = comfort_score1 + participant_effects[participant_ids]
            comfort_score2 = comfort_score2 + participant_effects[participant_ids]
            
            # Bradley-Terry probability
            p = pm.math.sigmoid(comfort_score1 - comfort_score2)
            
            # Likelihood
            pm.Bernoulli('likelihood', p=p, observed=y)
            
            # Fit model using NUTS sampler
            self.trace = pm.sample(
                draws=self.config.get('model', {}).get('n_samples', 1000),
                chains=self.config.get('model', {}).get('chains', 4),
                target_accept=self.config.get('model', {}).get('target_accept', 0.8)
            )
    
    def predict_preference(self, bigram1: str, bigram2: str, 
                         participant_id: Optional[str] = None) -> Tuple[float, float]:
        """
        Predict preference probability with uncertainty.
        Returns (mean probability, standard deviation)
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before making predictions")
            
        # Extract features
        feat1 = self._extract_features(bigram1)
        feat2 = self._extract_features(bigram2)
        
        # Get samples from posterior
        feature_weights_samples = self.trace.posterior['feature_weights'].values
        
        if participant_id is not None:
            participant_effects = self.trace.posterior['participant_effects'].values
            participant_idx = self._get_participant_index(participant_id)
            comfort1 = np.dot(feat1, feature_weights_samples.T) + participant_effects[:, :, participant_idx]
            comfort2 = np.dot(feat2, feature_weights_samples.T) + participant_effects[:, :, participant_idx]
        else:
            comfort1 = np.dot(feat1, feature_weights_samples.T)
            comfort2 = np.dot(feat2, feature_weights_samples.T)
        
        # Calculate probabilities
        probs = 1 / (1 + np.exp(-(comfort1 - comfort2)))
        
        return float(np.mean(probs)), float(np.std(probs))
    
    def get_bigram_comfort_scores(self, bigram: str) -> Tuple[float, float]:
        """
        Get estimated comfort score for a bigram with uncertainty.
        Returns (mean score, standard deviation)
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before getting comfort scores")
            
        features = self._extract_features(bigram)
        feature_weights_samples = self.trace.posterior['feature_weights'].values
        comfort_scores = np.dot(features, feature_weights_samples.T)
        
        return float(np.mean(comfort_scores)), float(np.std(comfort_scores))
            
    def get_feature_weights(self) -> Dict[str, float]:
        """Get the learned feature weights including interactions."""
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before getting weights")
        return self.feature_weights.copy()
  
    def cross_validate(
        self, 
        dataset: PreferenceDataset, 
        n_splits: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform cross-validation with multiple validation strategies."""
        if n_splits is None:
            n_splits = self.config.get('model', {}).get('cross_validation', {}).get('n_splits', 5)
        
        feature_names = dataset.get_feature_names()
        logger.debug(f"Features for cross-validation (including interactions): {feature_names}")
        
        metrics = defaultdict(list)
        feature_effects = defaultdict(list)
        
        # 1. Participant-based Cross-validation
        participant_ids = np.array([p.participant_id for p in dataset.preferences])
        cv = GroupKFold(n_splits=n_splits)
        
        # Prepare data arrays once
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
        for fold, (train_idx, val_idx) in enumerate(cv.split(X1, y, groups=participant_ids)):
            try:
                logger.info(f"Processing fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets
                train_data = dataset._create_subset_dataset(train_idx)
                val_data = dataset._create_subset_dataset(val_idx)
                
                if len(train_data.preferences) == 0 or len(val_data.preferences) == 0:
                    logger.warning(f"Empty split in fold {fold + 1}, skipping")
                    continue
                
                # Fit Bayesian model on training data
                self.fit(train_data)
                
                # Get predictions with uncertainty on validation set
                val_predictions = []
                val_uncertainties = []
                val_true = []
                
                for pref in val_data.preferences:
                    try:
                        pred_mean, pred_std = self.predict_preference(
                            pref.bigram1, pref.bigram2)
                        if not np.isnan(pred_mean):
                            val_predictions.append(pred_mean)
                            val_uncertainties.append(pred_std)
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
                metrics['mean_uncertainty'].append(np.mean(val_uncertainties))
                
                # Store feature weights with uncertainty
                weights = self.get_feature_weights()
                if weights:
                    logger.debug(f"Fold {fold + 1} weights: {weights}")
                    for feature, (weight_mean, weight_std) in weights.items():
                        if not np.isnan(weight_mean):
                            feature_effects[feature].append({
                                'mean': weight_mean,
                                'std': weight_std
                            })
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
                # Calculate effect statistics considering uncertainty
                effect_means = [e['mean'] for e in effects]
                effect_stds = [e['std'] for e in effects]
                
                mean_effect = float(np.mean(effect_means))
                mean_uncertainty = float(np.mean(effect_stds))
                
                processed_effects[feature] = {
                    'mean': mean_effect,
                    'std': mean_uncertainty,
                    'values': effect_means
                }
                
                # Calculate feature importance incorporating uncertainty
                importance_metrics[feature] = self._calculate_feature_importance(
                    feature, dataset, mean_effect, mean_uncertainty)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(feature_effects)
        
        # Determine selected features using Bayesian criteria
        selected_features = self._select_features_bayesian(
            importance_metrics, stability_metrics)
        
        # Log results
        self._log_feature_selection_results(
            selected_features, importance_metrics)
        
        # Save metrics to CSV
        self.save_metrics_csv(
            self.config['feature_evaluation']['metrics_file'],
            processed_effects, importance_metrics, 
            stability_metrics, selected_features)
        
        return {
            'metrics': metrics,
            'selected_features': selected_features,
            'feature_effects': processed_effects,
            'importance_metrics': importance_metrics,
            'stability_metrics': stability_metrics,
            'fold_uncertainties': dict(metrics['mean_uncertainty'])
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

