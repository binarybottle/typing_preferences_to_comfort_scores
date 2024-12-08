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
    
    def __init__(self, config: Optional[Dict] = None):
        self.feature_weights = None
        self.config = config or {}
        self.n_samples = self.config.get('feature_evaluation', {}).get('n_samples', 10000)
        self.n_chains = self.config.get('feature_evaluation', {}).get('chains', 8)
        self.target_accept = self.config.get('feature_evaluation', {}).get('target_accept', 0.85)
        
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
    
    def cross_validate(
        self, 
        dataset: PreferenceDataset, 
        n_splits: Optional[int] = None
    ) -> Dict[str, Any]:
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
            - fold_details: Optional detailed fold information
            - raw_metrics: Optional raw prediction data
        """
        # Use config n_splits if not specified
        if n_splits is None:
            n_splits = self.config.get('feature_evaluation', {}).get('n_splits', 5)
        
        # Get validation settings from config
        validation_config = self.config.get('feature_evaluation', {}).get('validation', {})
        reporting_config = self.config.get('feature_evaluation', {}).get('reporting', {})
        
        min_samples = validation_config.get('min_training_samples', 1000)
        min_val_samples = validation_config.get('min_validation_samples', 200)
        perform_outlier_detection = validation_config.get('outlier_detection', True)
        outlier_threshold = validation_config.get('outlier_threshold', 3.0)
        save_fold_details = reporting_config.get('save_fold_details', True)
        save_raw_metrics = reporting_config.get('save_raw_metrics', True)
        
        # Validate dataset size
        if len(dataset.preferences) < min_samples:
            logger.warning(f"Dataset has fewer than {min_samples} samples")
        
        expected_val_size = len(dataset.preferences) // n_splits
        if expected_val_size < min_val_samples:
            logger.warning(f"Expected validation set size ({expected_val_size}) "
                        f"is less than minimum ({min_val_samples})")
        
        # Initialize storage
        feature_effects = {name: [] for name in dataset.get_feature_names()}
        metrics = {
            'accuracy': [], 
            'auc': [],
            'log_likelihood': []
        }
        
        logger.info(f"Starting {n_splits}-fold cross-validation")
        
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
        
        # Initialize fold details if needed
        if save_fold_details:
            fold_details = []
        
        # Initialize raw metrics if needed
        if save_raw_metrics:
            raw_metrics = {
                'predictions': [],
                'true_values': [],
                'fold_indices': []
            }
        
        # Perform cross-validation
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
                
                # Handle outliers if enabled
                if perform_outlier_detection:
                    z_scores = np.abs((val_predictions - np.mean(val_predictions)) / np.std(val_predictions))
                    outliers = z_scores > outlier_threshold
                    if np.any(outliers):
                        logger.warning(f"Found {np.sum(outliers)} outlier predictions in fold {fold + 1}")
                        val_predictions_clean = val_predictions[~outliers]
                        val_true_clean = val_true[~outliers]
                    else:
                        val_predictions_clean = val_predictions
                        val_true_clean = val_true
                else:
                    val_predictions_clean = val_predictions
                    val_true_clean = val_true
                
                # Calculate metrics
                fold_accuracy = accuracy_score(val_true_clean, val_predictions_clean > 0.5)
                fold_auc = roc_auc_score(val_true_clean, val_predictions_clean)
                
                metrics['accuracy'].append(fold_accuracy)
                metrics['auc'].append(fold_auc)
                
                # Store feature effects
                weights = self.get_feature_weights()
                for feature, weight in weights.items():
                    feature_effects[feature].append(weight)
                
                # Save fold details if enabled
                if save_fold_details:
                    fold_details.append({
                        'fold': fold + 1,
                        'train_size': len(train_data.preferences),
                        'val_size': len(val_data.preferences),
                        'n_outliers': np.sum(outliers) if perform_outlier_detection else 0,
                        'metrics': {
                            'accuracy': float(fold_accuracy),
                            'auc': float(fold_auc)
                        },
                        'feature_weights': {
                            f: float(w) for f, w in weights.items()
                        }
                    })
                
                # Save raw metrics if enabled
                if save_raw_metrics:
                    raw_metrics['predictions'].extend(val_predictions.tolist())
                    raw_metrics['true_values'].extend(val_true.tolist())
                    raw_metrics['fold_indices'].append({
                        'train': train_idx.tolist(),
                        'val': val_idx.tolist()
                    })
                    
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        # Log feature effects summary
        for feature, effects in feature_effects.items():
            if not effects:
                logger.warning(f"No effects collected for feature: {feature}")
            else:
                logger.debug(f"Feature {feature}: collected {len(effects)} effects")
            
        # Calculate stability metrics
        stability = self._calculate_stability_metrics(feature_effects, config=self.config)
        
        # Prepare results
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
        
        # Add optional detailed results
        if save_fold_details:
            results['fold_details'] = fold_details
        if save_raw_metrics:
            results['raw_metrics'] = raw_metrics
        
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
            
            effects = np.array(effects)
            
            # Outlier detection if enabled
            if outlier_detection:
                z_scores = np.abs((effects - np.mean(effects)) / np.std(effects))
                outliers = z_scores > outlier_threshold
                n_outliers = np.sum(outliers)
                
                if n_outliers > 0:
                    logger.warning(f"Feature '{feature}' has {n_outliers} outlier effects")
                    # Optionally remove outliers for stability calculations
                    effects = effects[~outliers]
            else:
                n_outliers = 0
            
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
