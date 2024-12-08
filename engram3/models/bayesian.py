# engram3/models/bayesian.py
from typing import Dict, List
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import logging

from engram3.data import PreferenceDataset
from engram3.models.base import PreferenceModel

logger = logging.getLogger(__name__)

class BayesianPreferenceModel(PreferenceModel):
    def cross_validate(self, dataset: PreferenceDataset, n_splits: int = 5) -> Dict[str, List[float]]:
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
        
        logger.info(f"Starting {n_splits}-fold cross-validation")
        
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