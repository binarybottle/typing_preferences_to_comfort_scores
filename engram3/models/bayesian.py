# engram3/models/bayesian.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings 
from scipy import stats 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

from engram3.data import PreferenceDataset
from engram3.models.base import PreferenceModel

logger = logging.getLogger(__name__)

class BayesianPreferenceModel(PreferenceModel):
    def __init__(self):
        self.feature_weights = None
        
    def fit(self, dataset: PreferenceDataset) -> None:
        """
        Fit the Bayesian preference model.
        
        Args:
            dataset: PreferenceDataset containing training data
        """
        try:
            # For now, implement a simple version that learns feature weights
            feature_names = dataset.get_feature_names()
            self.feature_weights = {name: 0.0 for name in feature_names}
            
            # Simple weight calculation based on preference correlations
            for feature in feature_names:
                diffs = []
                prefs = []
                for pref in dataset.preferences:
                    try:
                        diff = pref.features1[feature] - pref.features2[feature]
                        diffs.append(diff)
                        prefs.append(1.0 if pref.preferred else -1.0)
                    except KeyError:
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
        """
        Predict preference probability for bigram1 over bigram2.
        
        Returns:
            Float between 0 and 1, probability of preferring bigram1
        """
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before making predictions")
            
        try:
            # Extract features for both bigrams
            from ..features.extraction import extract_bigram_features
            from ..features.definitions import (
                column_map, row_map, finger_map,
                engram_position_values, row_position_values
            )
            
            # Get features for each bigram
            features1 = extract_bigram_features(
                bigram1[0], bigram1[1],
                column_map, row_map, finger_map,
                engram_position_values, row_position_values
            )
            
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
    
    def cross_validate(self, dataset: PreferenceDataset, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation with participant-based splits.
        
        Args:
            dataset: PreferenceDataset containing all data
            n_splits: Number of cross-validation folds
            
        Returns:
            Dictionary containing:
            - metrics: Dict of lists containing scores for each fold
            - feature_effects: Dict of feature effect sizes across folds
            - stability: Dict of feature stability metrics
        """
        feature_effects = {name: [] for name in dataset.get_feature_names()}
        logger.info(f"Starting cross-validation with {n_splits} splits")
    
        metrics = {
            'accuracy': [], 
            'auc': [],
            'log_likelihood': []
        }
                
        # Convert participants to array for splitting
        participants = np.array([p.participant_id for p in dataset.preferences])
        
        # Prepare data arrays
        all_preferences = []
        all_features1 = []
        all_features2 = []
        
        for pref in dataset.preferences:
            all_preferences.append(float(pref.preferred))
            feat1 = [pref.features1[f] for f in dataset.get_feature_names()]
            feat2 = [pref.features2[f] for f in dataset.get_feature_names()]
            all_features1.append(feat1)
            all_features2.append(feat2)
            
        X1 = np.array(all_features1)
        X2 = np.array(all_features2)
        y = np.array(all_preferences)
        
        group_kfold = GroupKFold(n_splits=n_splits)
                
        for fold, (train_idx, val_idx) in enumerate(
            group_kfold.split(X1, y, participants)
        ):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Create train/validation datasets
            train_data = self._create_subset_dataset(
                dataset, train_idx, X1, X2, y, participants)
            val_data = self._create_subset_dataset(
                dataset, val_idx, X1, X2, y, participants)
            
            try:
                # Fit model on training data
                self.fit(train_data)
                
                # Get predictions on validation set
                val_predictions = []
                val_true = []
                
                for pref in val_data.preferences:
                    pred = self.predict_preference(pref.bigram1, pref.bigram2)
                    val_predictions.append(pred)
                    val_true.append(float(pref.preferred))
                
                val_predictions = np.array(val_predictions)
                val_true = np.array(val_true)
                
                # Calculate metrics
                metrics['accuracy'].append(
                    accuracy_score(val_true, val_predictions > 0.5))
                metrics['auc'].append(
                    roc_auc_score(val_true, val_predictions))
                
                # Store feature effects
                weights = self.get_feature_weights()
                for feature, weight in weights.items():
                    feature_effects[feature].append(weight)
                    
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        for feature, effects in feature_effects.items():
            if not effects:
                logger.warning(f"No effects collected for feature: {feature}")
            else:
                logger.debug(f"Feature {feature}: collected {len(effects)} effects")
            
        # Calculate stability metrics
        stability = self._calculate_stability_metrics(feature_effects)
        
        # Calculate summary statistics
        results = {
            'metrics': {
                name: {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'values': list(scores)
                }
                for name, scores in metrics.items()
            },
            'feature_effects': {
                name: {
                    'mean': float(np.mean(effects)),
                    'std': float(np.std(effects)),
                    'values': list(effects)
                }
                for name, effects in feature_effects.items()
            },
            'stability': stability
        }
        
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
        return subset
    
    def _calculate_stability_metrics(
        self, 
        feature_effects: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature stability metrics with proper error handling.
        
        Args:
            feature_effects: Dictionary mapping feature names to lists of effects across folds
            
        Returns:
            Dictionary containing stability metrics for each feature
        """
        stability = {}
        
        for feature, effects in feature_effects.items():
            if not effects:  # Handle empty lists
                stability[feature] = {
                    'effect_mean': 0.0,
                    'effect_std': 0.0,
                    'effect_cv': float('inf'),
                    'sign_consistency': 0.0,
                    'relative_range': float('inf')
                }
                continue
                
            effects = np.array(effects)
            mean_effect = np.mean(effects)
            std_effect = np.std(effects)
            
            # Calculate metrics with proper error handling
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
            
            stability[feature] = {
                'effect_mean': float(mean_effect),
                'effect_std': float(std_effect),
                'effect_cv': float(effect_cv),
                'sign_consistency': float(sign_consistency),
                'relative_range': float(relative_range)
            }
        
        return stability