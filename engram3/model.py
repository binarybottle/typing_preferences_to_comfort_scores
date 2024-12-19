# model.py
"""
Hierarchical Bayesian model for keyboard layout preference learning.

Core functionality:
  - Bayesian inference using Bradley-Terry preference model structure:
    - Feature-based comfort score estimation
    - Participant-level random effects
    - Uncertainty quantification via MCMC sampling
    - Stan backend integration

Key components:
  1. Sequential Feature Selection:
    - Round-robin comparison of features
    - Context-aware evaluation with previously selected features
    - Three independent metrics:
      * Model effect magnitude
      * Effect consistency across cross-validation
      * Predictive power
    - Feature interaction handling

  2. Model Operations:
    - Efficient data preparation for Stan
    - MCMC sampling with diagnostics
    - Cross-validation with participant grouping
    - Feature weight extraction and caching

  3. Prediction Pipeline:
    - Bigram comfort score estimation
    - Preference probability prediction
    - Uncertainty quantification
    - Cached predictions for efficiency

  4. Evaluation Mechanisms:
    - Classification metrics (accuracy, AUC)
    - Effect size estimation
    - Cross-validation stability
    - Model diagnostics
    - Transitivity checks

  5. Output Handling:
    - Detailed metric reporting
    - State serialization
    - Comprehensive logging
    - CSV export capabilities

The PreferenceModel class provides a complete pipeline for learning and predicting
keyboard layout preferences while maintaining efficiency through caching and
proper error handling.

Example:
    >>> model = PreferenceModel(config)
    >>> model.fit(dataset)
    >>> prediction = model.predict_preference("th", "he")
    >>> print(f"Preference probability: {prediction.probability:.2f} ± {prediction.uncertainty:.2f}")

Notes:
    - Features are pre-normalized except typing_time
    - Uses participant-based cross-validation splits
    - Thread-safe Stan implementation
    - Comprehensive error handling
    - Detailed logging system
"""
import cmdstanpy
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import time  # Needed for computation_time in predict_preference
import pickle
import copy
from pathlib import Path
from itertools import combinations

from engram3.utils.config import (
    Config, NotFittedError, FeatureError,
    ModelPrediction, StabilityMetrics
)
from engram3.data import PreferenceDataset
from engram3.features.feature_importance import FeatureImportanceCalculator
from engram3.utils.visualization import PlottingUtils
from engram3.utils.caching import CacheManager
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class PreferenceModel:
    # Class variable for cache storage
    _feature_data_cache_: Dict[str, Dict[str, np.ndarray]] = {}

    #--------------------------------------------
    # Core model methods
    #--------------------------------------------   
    def __init__(self, config: Union[Dict, Config] = None):
        """Initialize PreferenceModel."""
        # First, validate and set config
        if config is None:
            raise ValueError("Config is required")
        
        # Set config first before using it
        if isinstance(config, dict):
            self.config = Config(**config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError(f"Config must be a dictionary or Config object, got {type(config)}")
            
        # Initialize basic attributes
        self.fit_result = None
        self.feature_names = None
        self.selected_features = []
        self.interaction_metadata = {}
        self.dataset = None
        self.feature_weights = None
        self.feature_extractor = None
        # Initialize caches
        self.feature_cache = CacheManager()  # Single feature cache
        self.prediction_cache = CacheManager()
        
        try:
            # Initialize Stan model
            model_path = Path(__file__).parent / "models" / "preference_model.stan"
            if not model_path.exists():
                raise FileNotFoundError(f"Stan model file not found: {model_path}")
            
            logger.info(f"Loading Stan model from {model_path}")
            
            import cmdstanpy
            self.model = cmdstanpy.CmdStanModel(
                stan_file=str(model_path),
                cpp_options={'STAN_THREADS': True},
                stanc_options={'warn-pedantic': True}
            )
            
            # Set permissions if needed
            if hasattr(self.model, 'exe_file'):
                exe_path = Path(self.model.exe_file)
                if exe_path.exists():
                    exe_path.chmod(0o755)
                    
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

        # Initialize visualization components
        self.plotting = PlottingUtils(self.config.paths.plots_dir)
        
        # Initialize importance calculator last (needs fully initialized self)
        self.importance_calculator = FeatureImportanceCalculator(self.config, self)

    # Property decorators
    @property
    def feature_scale(self) -> float:
        """Prior scale for feature weights."""
        return self.config.model.feature_scale
        
    @property
    def participant_scale(self) -> float:
        """Prior scale for participant effects."""
        return self.config.model.participant_scale
        
    @property
    def _feature_data_cache(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Access feature data cache with initialization check."""
        return self._feature_data_cache_

    @_feature_data_cache.setter
    def _feature_data_cache(self, value: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Set feature data cache."""
        self._feature_data_cache_ = value

    # Remaining class methods
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self.feature_cache.clear()
        self.prediction_cache.clear()

    def fit(self, dataset: PreferenceDataset, features: Optional[List[str]] = None) -> None:
        try:
            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor
            
            # Validate features
            if not features:
                features = dataset.get_feature_names()
            if not features:
                raise ValueError("No features provided and none available from dataset")
            
            # Store feature names
            self.feature_names = features
            
            # Prepare data
            stan_data = self.prepare_data(dataset, features)
            
            # Validate Stan data
            if stan_data['F'] < 1:
                raise ValueError(f"Invalid number of features: {stan_data['F']}")
                
            # Fit using Stan
            self.fit_result = self.model.sample(
                data=stan_data,
                chains=self.config.model.chains,
                iter_warmup=self.config.model.warmup,
                iter_sampling=self.config.model.n_samples,
                adapt_delta=self.config.model.adapt_delta,
                max_treedepth=self.config.model.max_treedepth
            )
            
            # Check diagnostics
            self._check_diagnostics()
            
            # Update feature weights
            self._update_feature_weights()
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def _fit_stan(self, stan_data: Dict) -> None:
        """Fit model using Stan backend."""
        logger.info("Starting Stan sampling...")
        self.fit_result = self.model.sample(
            data=stan_data,
            chains=self.config.model.chains,
            iter_warmup=self.config.model.warmup,
            iter_sampling=self.config.model.n_samples,
            adapt_delta=getattr(self.config.model, 'adapt_delta', 0.95),
            max_treedepth=getattr(self.config.model, 'max_treedepth', 12),
            show_progress=True,
            refresh=50
        )

    def _update_feature_weights(self) -> None:
        """Update feature weights from fitted model for both main and control features."""
        try:
            self.feature_weights = {}
            self.selected_features = []  # Reset selected features
            
            if hasattr(self, 'fit'):  # Stan backend
                # Get main feature weights (beta)
                beta = self.fit_result.stan_variable('beta')
                main_features = [f for f in self.feature_names 
                            if f not in self.config.features.control_features]
                for i, feature in enumerate(main_features):
                    self.feature_weights[feature] = (
                        float(np.mean(beta[:, i])),
                        float(np.std(beta[:, i]))
                    )
                    self.selected_features.append(feature)  # Add main features
                
                # Get control feature weights (gamma)
                if self.config.features.control_features:
                    gamma = self.fit_result.stan_variable('gamma')
                    for i, feature in enumerate(self.config.features.control_features):
                        self.feature_weights[feature] = (
                            float(np.mean(gamma[:, i])),
                            float(np.std(gamma[:, i]))
                        )
                        self.selected_features.append(feature)  # Add control features
                
                logger.debug("Updated weights:")
                for feature, (mean, std) in self.feature_weights.items():
                    feature_type = "control" if feature in self.config.features.control_features else "main"
                    logger.debug(f"  {feature} ({feature_type}): {mean:.4f} ± {std:.4f}")
                    
            elif hasattr(self, 'trace'):  # PyMC backend
                # Note: PyMC backend would need similar separation of main/control features
                main_features = [f for f in self.feature_names 
                            if f not in self.config.features.control_features]
                for i, feature in enumerate(main_features):
                    weights = self.trace.get_values('feature_weights')[:, i]
                    self.feature_weights[feature] = (
                        float(np.mean(weights)),
                        float(np.std(weights))
                    )
                    self.selected_features.append(feature)  # Add main features
                
                if self.config.features.control_features:
                    for i, feature in enumerate(self.config.features.control_features):
                        weights = self.trace.get_values('control_weights')[:, i]
                        self.feature_weights[feature] = (
                            float(np.mean(weights)),
                            float(np.std(weights))
                        )
                        self.selected_features.append(feature)  # Add control features
                        
        except Exception as e:
            logger.error(f"Error updating feature weights: {str(e)}")
            logger.error("Traceback:", exc_info=True)
                                                
    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        try:
            if not hasattr(self, 'fit_result') or self.fit_result is None:
                logger.error("Model must be fit before evaluation")
                raise RuntimeError("Model must be fit before evaluation")
                
            predictions = []
            actuals = []
            
            logger.info(f"Evaluating model on {len(dataset.preferences)} preferences")
            
            for pref in dataset.preferences:
                try:
                    # Get prediction probability - now handling ModelPrediction object
                    prediction = self.predict_preference(pref.bigram1, pref.bigram2)
                    if not np.isnan(prediction.probability):
                        predictions.append(prediction.probability)
                        actuals.append(1.0 if pref.preferred else 0.0)
                except Exception as e:
                    logger.debug(f"Skipping preference in evaluation due to: {str(e)}")
                    continue
                    
            if not predictions:
                logger.warning("No valid predictions for evaluation")
                return {
                    'accuracy': 0.5,  # Random chance
                    'auc': 0.5,
                    'n_evaluated': 0
                }
            
            logger.info(f"Made predictions for {len(predictions)} preferences")
                
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            accuracy = np.mean((predictions > 0.5) == actuals)
            auc = roc_auc_score(actuals, predictions)
            
            logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'n_evaluated': len(predictions)
            }
                
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return {
                'accuracy': 0.5,
                'auc': 0.5,
                'n_evaluated': 0
            }
                
    def cross_validate(self, dataset: PreferenceDataset, n_splits: Optional[int] = None) -> Dict[str, Any]:
        """Perform cross-validation with multiple validation strategies."""
        feature_names = dataset.get_feature_names()
        logger.debug(f"Features for cross-validation (including interactions): {feature_names}")
        
        metrics = defaultdict(list)
        feature_effects = defaultdict(list)
        
        # Get CV splits using shared method
        for fold, (train_idx, val_idx) in enumerate(self._get_cv_splits(dataset, n_splits)):
            try:
                # Clear caches before each fold to prevent memory buildup
                self.clear_caches()
                logger.info(f"Processing fold {fold + 1}/{n_splits}")
                
                # Create train/val datasets
                train_data = dataset._create_subset_dataset(train_idx)
                val_data = dataset._create_subset_dataset(val_idx)
                
                if len(train_data.preferences) == 0 or len(val_data.preferences) == 0:
                    logger.warning(f"Empty split in fold {fold + 1}, skipping")
                    continue
                
                # Fit Bayesian model on training data
                self.fit_result(train_data)
                
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
                        
            finally:
                # Clear caches after cross-validation
                self.clear_caches()
                    
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
                importance_metrics[feature] = self.importance_calculator.evaluate_feature(
                    feature=feature,
                    dataset=dataset,
                    model=self
                )

        # Determine selected features using Bayesian criteria
        selected_features = self.selected_features
        
        # Log results
        self._log_feature_selection_results(
            selected_features, importance_metrics)
        
        # Save metrics to CSV
        self.save_metrics_report(
            metrics_dict={feature: {
                **processed_effects.get(feature, {}),
                **importance_metrics.get(feature, {}),
            } for feature in feature_names},
            output_file=self.config.feature_selection.metrics_file
        )
        
        return {
            'metrics': metrics,
            'selected_features': selected_features,
            'feature_effects': processed_effects,
            'importance_metrics': importance_metrics,
            'fold_uncertainties': dict(metrics['mean_uncertainty'])
        }

    def _log_feature_selection_results(self, 
                                    selected_features: List[str],
                                    importance_metrics: Dict[str, Dict]) -> None:
        """Log feature selection results."""
        logger.info("\nSelected Features:")
        for feature in selected_features:
            metrics = importance_metrics.get(feature, {})
            logger.info(f"  {feature}:")

    #--------------------------------------------
    # Data preparation and feature methods
    #--------------------------------------------
    def _get_feature_data(self, feature: str, dataset: Optional[PreferenceDataset] = None) -> Dict[str, np.ndarray]:
        """
        Centralized method for getting feature data with caching.
        Only normalizes typing_time, assumes other features (including control features) are pre-normalized.
        """
        dataset = dataset or self.dataset
        cache_key = f"{dataset.file_path}_{feature}"
        
        if hasattr(self, '_feature_data_cache') and cache_key in self._feature_data_cache:
            return self._feature_data_cache[cache_key]
            
        if '_x_' in feature:
            # Handle interaction features (already normalized through components)
            feat1, feat2 = feature.split('_x_')
            data1 = self._get_feature_data(feat1, dataset)
            data2 = self._get_feature_data(feat2, dataset)
            
            values = data1['values'] * data2['values']
            differences = data1['differences'] * data2['differences']
            raw_features = {}
            for bigram in set(data1['raw_features']) | set(data2['raw_features']):
                raw_features[bigram] = data1['raw_features'].get(bigram, 0.0) * data2['raw_features'].get(bigram, 0.0)
            
        else:
            # Handle base features
            values = []
            differences = []
            raw_features = {}
            
            if feature == 'typing_time':
                # Only typing_time needs normalization
                all_times = []
                for pref in dataset.preferences:
                    if pref.typing_time1 is not None:
                        all_times.append(pref.typing_time1)
                    if pref.typing_time2 is not None:
                        all_times.append(pref.typing_time2)
                
                time_mean = np.mean(all_times)
                time_std = np.std(all_times)
                
                for pref in dataset.preferences:
                    time1 = pref.typing_time1
                    time2 = pref.typing_time2
                    
                    # Use mean for None values
                    if time1 is None or time2 is None:
                        valid_times = [t for t in [time1, time2] if t is not None]
                        mean_time = np.mean(valid_times) if valid_times else time_mean
                        time1 = mean_time if time1 is None else time1
                        time2 = mean_time if time2 is None else time2
                    
                    # Z-score normalize timing values
                    time1_norm = (time1 - time_mean) / time_std
                    time2_norm = (time2 - time_mean) / time_std
                    
                    values.extend([time1_norm, time2_norm])
                    differences.append(time1_norm - time2_norm)
                    raw_features[pref.bigram1] = time1_norm
                    raw_features[pref.bigram2] = time2_norm
                    
            else:
                # All other features (including control features) are already normalized
                for pref in dataset.preferences:
                    feat1 = self.feature_extractor.extract_bigram_features(
                        pref.bigram1[0], pref.bigram1[1]).get(feature, 0.0)
                    feat2 = self.feature_extractor.extract_bigram_features(
                        pref.bigram2[0], pref.bigram2[1]).get(feature, 0.0)
                    
                    values.extend([feat1, feat2])
                    differences.append(feat1 - feat2)
                    raw_features[pref.bigram1] = feat1
                    raw_features[pref.bigram2] = feat2
                    
        result = {
            'values': np.array(values),
            'differences': np.array(differences),
            'raw_features': raw_features
        }
        
        if not hasattr(self, '_feature_data_cache'):
            self._feature_data_cache = {}
        self._add_to_cache(self._feature_data_cache, cache_key, result)
        
        return result

    def _add_to_cache(self, cache: Dict, key: str, value: Any) -> None:
        """Add item to cache with size management."""
        if len(cache) > 1000:  # Arbitrary limit
            # Remove oldest 10% of entries
            remove_count = len(cache) // 10
            for k in list(cache.keys())[:remove_count]:
                del cache[k]
        cache[key] = value
    
    def prepare_data(self, dataset: PreferenceDataset, features: Optional[List[str]] = None) -> Dict:
        """
        Prepare data for Stan model with proper validation and cleaning.
        Handles both main features and control features separately.
        """
        try:
            # Get feature names and separate control features
            self.feature_names = features if features is not None else dataset.get_feature_names()
            control_features = self.config.features.control_features
            
            # Main features are those not in control_features
            main_features = [f for f in self.feature_names if f not in control_features]
            
            if not main_features:
                raise ValueError("No main features provided for model fitting")
            
            logger.info(f"Preparing data:")
            logger.info(f"  Main features ({len(main_features)}): {main_features}")
            logger.info(f"  Control features ({len(control_features)}): {control_features}")

            # Debug logging for interaction features
            for feature in self.feature_names:
                if '_x_' in feature:
                    feature_data = self._compute_feature_values(feature, dataset)
                    logger.debug(f"Interaction feature {feature} stats:")
                    logger.debug(f"  mean: {np.mean(feature_data):.4f}")
                    logger.debug(f"  std: {np.std(feature_data):.4f}")
                    logger.debug(f"  range: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")

            # Create participant ID mapping
            participant_ids = sorted(list(dataset.participants))
            participant_map = {pid: i+1 for i, pid in enumerate(participant_ids)}

            # Handle typing_time normalization
            if 'typing_time' in self.feature_names:
                all_times = []
                for pref in dataset.preferences:
                    if pref.typing_time1 is not None:
                        all_times.append(pref.typing_time1)
                    if pref.typing_time2 is not None:
                        all_times.append(pref.typing_time2)
                time_mean = np.mean(all_times)
                time_std = np.std(all_times)
                logger.info(f"Typing time normalization parameters - mean: {time_mean:.3f}, std: {time_std:.3f}")

            # Initialize arrays for both main and control features
            X1, X2 = [], []  # Main feature matrices
            C1, C2 = [], []  # Control feature matrices
            participant = []
            y = []

            skipped_count = 0
            for pref in dataset.preferences:
                try:
                    # Process main features
                    features1_main, features2_main = [], []
                    for feature in main_features:
                        feat1 = pref.features1.get(feature)
                        feat2 = pref.features2.get(feature)

                        if feature == 'typing_time':
                            if feat1 is None or feat2 is None:
                                valid_times = [t for t in [pref.typing_time1, pref.typing_time2] if t is not None]
                                mean_time = np.mean(valid_times) if valid_times else time_mean
                                feat1 = mean_time if feat1 is None else feat1
                                feat2 = mean_time if feat2 is None else feat2
                            feat1 = (feat1 - time_mean) / time_std
                            feat2 = (feat2 - time_mean) / time_std
                        else:
                            feat1 = 0.0 if feat1 is None else feat1
                            feat2 = 0.0 if feat2 is None else feat2

                        features1_main.append(feat1)
                        features2_main.append(feat2)

                    # Process control features
                    features1_control, features2_control = [], []
                    for feature in control_features:
                        feat1 = pref.features1.get(feature, 0.0)
                        feat2 = pref.features2.get(feature, 0.0)
                        features1_control.append(feat1)
                        features2_control.append(feat2)

                    X1.append(features1_main)
                    X2.append(features2_main)
                    C1.append(features1_control)
                    C2.append(features2_control)
                    participant.append(participant_map[pref.participant_id])
                    y.append(1 if pref.preferred else 0)

                except Exception as e:
                    logger.warning(f"Skipping preference due to error: {str(e)}")
                    skipped_count += 1
                    continue

            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} preferences due to invalid features")

            if not X1:
                raise ValueError("No valid preferences after data preparation")

            # Convert to numpy arrays
            X1 = np.array(X1, dtype=np.float64)
            X2 = np.array(X2, dtype=np.float64)
            C1 = np.array(C1, dtype=np.float64)
            C2 = np.array(C2, dtype=np.float64)
            participant = np.array(participant, dtype=np.int32)
            y = np.array(y, dtype=np.int32)

            # Validation
            if X1.shape[1] != len(main_features):
                raise ValueError(f"Main feature dimension mismatch: {X1.shape[1]} != {len(main_features)}")
            if C1.shape[1] != len(control_features):
                raise ValueError(f"Control feature dimension mismatch: {C1.shape[1]} != {len(control_features)}")
            if np.any(np.isnan(X1)) or np.any(np.isnan(X2)) or np.any(np.isnan(C1)) or np.any(np.isnan(C2)):
                raise ValueError("NaN values found in feature matrices")

            # Log dimensions
            logger.info(f"Data dimensions:")
            logger.info(f"  N (preferences): {len(y)}")
            logger.info(f"  P (participants): {len(participant_ids)}")
            logger.info(f"  F (main features): {len(main_features)}")
            logger.info(f"  C (control features): {len(control_features)}")

            # Prepare Stan data
            stan_data = {
                'N': len(y),
                'P': len(participant_ids),
                'F': len(main_features),
                'C': len(control_features),
                'X1': X1,
                'X2': X2,
                'C1': C1,
                'C2': C2,
                'participant': participant,
                'y': y,
                'feature_scale': self.config.model.feature_scale,
                'participant_scale': self.config.model.participant_scale
            }

            # Log statistics
            logger.info("\nFeature summary statistics:")
            for i, feature in enumerate(main_features):
                logger.info(f"{feature} (main):")
                logger.info(f"  X1 - mean: {np.mean(X1[:, i]):.3f}, std: {np.std(X1[:, i]):.3f}")
                logger.info(f"  X2 - mean: {np.mean(X2[:, i]):.3f}, std: {np.std(X2[:, i]):.3f}")
            for i, feature in enumerate(control_features):
                logger.info(f"{feature} (control):")
                logger.info(f"  C1 - mean: {np.mean(C1[:, i]):.3f}, std: {np.std(C1[:, i]):.3f}")
                logger.info(f"  C2 - mean: {np.mean(C2[:, i]):.3f}, std: {np.std(C2[:, i]):.3f}")

            return stan_data

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
                                
    def _extract_features(self, bigram: str) -> Dict[str, float]:
        """
        Extract features for a bigram using feature extractor with caching.
        
        Args:
            bigram: Two-character string to extract features for
            
        Returns:
            Dictionary mapping feature names to their values
            
        Raises:
            NotFittedError: If feature extractor is not initialized
            ValueError: If bigram is not exactly 2 characters
            FeatureError: If feature extraction fails
        """
        # Input validation
        if not isinstance(bigram, str):
            raise ValueError(f"Bigram must be a string, got {type(bigram)}")
            
        if len(bigram) != 2:
            raise ValueError(f"Bigram must be exactly 2 characters, got '{bigram}'")
            
        if not self.feature_extractor:
            raise NotFittedError("Feature extractor not initialized. Call fit() first.")
            
        try:
            # Use cache if available
            cache_key = str(bigram)  # Ensure string key for cache
            cached_features = self.feature_cache.get(cache_key)
            if cached_features is not None:
                return cached_features.copy()  # Return copy to prevent modification
            
            # Extract base features
            try:
                features = self.feature_extractor.extract_bigram_features(
                    char1=bigram[0],
                    char2=bigram[1]
                )
            except Exception as e:
                raise FeatureError(f"Feature extraction failed for bigram '{bigram}': {str(e)}")
                
            # Validate extracted features
            if not isinstance(features, dict):
                raise FeatureError(f"Feature extractor returned {type(features)}, expected dict")
                
            if not features:
                logger.warning(f"No features extracted for bigram '{bigram}'")
                
            # Cache result
            self.feature_cache.set(cache_key, features.copy())
            
            return features.copy()
            
        except (NotFittedError, ValueError, FeatureError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error extracting features for bigram '{bigram}': {str(e)}")
            raise FeatureError(f"Feature extraction failed: {str(e)}")
        
    def get_bigram_comfort_scores(self, bigram: str) -> Tuple[float, float]:
        """
        Get comfort score and uncertainty for a single bigram.
        Accounts for both main features and control features.
        
        Returns:
            Tuple of (mean score, standard deviation)
        """
        try:
            # Extract features for bigram
            features = self._extract_features(bigram)
            
            # Get feature weights from posterior, including control features
            weights = self.get_feature_weights(include_control=True)
            
            # Separate main and control features
            main_features = [f for f in self.selected_features 
                            if f not in self.config.features.control_features]
            control_features = self.config.features.control_features
            
            # Calculate main feature contribution
            main_score = 0.0
            main_uncertainty = 0.0
            for feature in main_features:
                if feature in weights:
                    value = features.get(feature, 0.0)
                    weight_mean, weight_std = weights[feature]
                    main_score += value * weight_mean
                    main_uncertainty += (value * weight_std) ** 2
            
            # Calculate control feature contribution
            control_score = 0.0
            control_uncertainty = 0.0
            for feature in control_features:
                if feature in weights:
                    value = features.get(feature, 0.0)
                    weight_mean, weight_std = weights[feature]
                    control_score += value * weight_mean
                    control_uncertainty += (value * weight_std) ** 2
            
            # Combine scores and uncertainties
            total_score = main_score + control_score
            total_uncertainty = np.sqrt(main_uncertainty + control_uncertainty)
            
            return float(total_score), float(total_uncertainty)
            
        except Exception as e:
            logger.error(f"Error calculating comfort scores: {str(e)}")
            return 0.0, 1.0
        
    def get_feature_weights(self, include_control: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        Get feature weights from fitted model.
        
        Args:
            include_control: Whether to include control feature weights in output
            
        Returns:
            Dictionary mapping feature names to (mean, std) tuples
        """
        if not self.fit_result:
            raise NotFittedError("Model must be fit before getting weights")
            
        if self.feature_weights is not None:
            if include_control:
                return self.feature_weights.copy()
            else:
                # Filter out control features
                return {k: v for k, v in self.feature_weights.items() 
                    if k not in self.config.features.control_features}
            
        try:
            summary = self.fit_result.summary()
            logger.info("\nAvailable features in summary:")
            logger.info(f"Summary index: {list(summary.index)}")
            
            weights: Dict[str, Tuple[float, float]] = {}
            
            # Process main features (beta parameters)
            logger.info("\nProcessing main features:")
            main_features = [f for f in self.feature_names 
                            if f not in self.config.features.control_features]
            for feature in main_features:
                logger.info(f"Processing main feature: {feature}")
                param_name = f"beta[{main_features.index(feature) + 1}]"
                if param_name in summary.index:
                    mean_val = float(summary.loc[param_name, 'mean'])
                    std_val = float(summary.loc[param_name, 'sd'])
                    logger.info(f"Found weights - mean: {mean_val}, std: {std_val}")
                    weights[feature] = (mean_val, std_val)
                else:
                    logger.warning(f"Feature {feature} not found in model summary")
            
            # Process control features (gamma parameters) if requested
            if include_control:
                logger.info("\nProcessing control features:")
                control_features = self.config.features.control_features
                for feature in control_features:
                    logger.info(f"Processing control feature: {feature}")
                    param_name = f"gamma[{control_features.index(feature) + 1}]"
                    if param_name in summary.index:
                        mean_val = float(summary.loc[param_name, 'mean'])
                        std_val = float(summary.loc[param_name, 'sd'])
                        logger.info(f"Found weights - mean: {mean_val}, std: {std_val}")
                        weights[feature] = (mean_val, std_val)
                    else:
                        logger.warning(f"Control feature {feature} not found in model summary")
            
            # Cache the results
            logger.info(f"\nFinal weights dictionary: {weights}")
            self.feature_weights = weights
            
            if include_control:
                return weights.copy()
            else:
                # Filter out control features for return
                return {k: v for k, v in weights.items() 
                    if k not in self.config.features.control_features}
            
        except Exception as e:
            logger.error(f"Error extracting feature weights: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise FeatureError(f"Failed to extract feature weights: {str(e)}")
                                            
    #--------------------------------------------
    # Feature selection and evaluation methods
    #--------------------------------------------                          
    def select_features(self, dataset: PreferenceDataset, all_features: List[str]) -> List[str]:
        """
        Select features using round-robin comparison, evaluating each candidate
        in the context of previously selected features plus control features.
        Control features are always included but not selected.
        """
        # Initialize with control features - these are always included
        self.selected_features = list(self.config.features.control_features)
        main_features = [f for f in all_features if f not in self.config.features.control_features]
        
        while True:
            remaining_features = [f for f in main_features 
                                if f not in self.selected_features]
            
            if not remaining_features:
                break
                
            logger.info(f"\nEvaluating {len(remaining_features)} remaining main features...")
            logger.info(f"Current feature set: {self.selected_features}")
            
            # Evaluate all remaining features in context of current set
            feature_metrics = {}
            for feature in remaining_features:
                # Create candidate feature set with current feature
                candidate_features = self.selected_features + [feature]
                
                # Fit model with candidate feature set
                self.fit(dataset, candidate_features)
                
                # Evaluate this feature in context of current set
                metrics = self.importance_calculator.evaluate_feature(
                    feature=feature,
                    dataset=dataset,
                    model=self
                )
                feature_metrics[feature] = metrics
            
            # Compare each feature against all others
            win_counts = {f: 0 for f in remaining_features}
            for f1, f2 in combinations(remaining_features, 2):
                if self._is_feature_better(feature_metrics[f1], feature_metrics[f2]):
                    win_counts[f1] += 1
                else:
                    win_counts[f2] += 1
                    
            # Select feature with most wins
            best_feature = max(win_counts.items(), key=lambda x: x[1])[0]
            best_metrics = feature_metrics[best_feature]
            
            logger.info(f"\nFeature comparison results:")
            for feature, wins in win_counts.items():
                metrics = feature_metrics[feature]
                logger.info(f"\n{feature}:")
                logger.info(f"  Wins: {wins}")
                logger.info(f"  Effect: {metrics['model_effect']:.3f}")
                logger.info(f"  Consistency: {metrics['effect_consistency']:.3f}")
                logger.info(f"  Predictive power: {metrics['predictive_power']:.3f}")
            
            if (best_metrics['effect_consistency'] > 0.5 or 
                best_metrics['predictive_power'] > 0.1):
                self.selected_features.append(best_feature)
                self.fit(dataset, self.selected_features)
                logger.info(f"\nSelected {best_feature} with {win_counts[best_feature]} wins")
                
                logger.info("\nCurrent model state:")
                weights = self.get_feature_weights()
                for feat in [f for f in self.selected_features 
                            if f not in self.config.features.control_features]:
                    w, s = weights.get(feat, (0.0, 0.0))
                    logger.info(f"  {feat}: {w:.3f} ± {s:.3f}")
            else:
                break
        
        # Ensure at least one main feature is selected
        if not any(f not in self.config.features.control_features 
                for f in self.selected_features):
            logger.warning("No main features selected, forcing selection of best main feature")
            best_main = max(feature_metrics.items(), 
                        key=lambda x: x[1]['model_effect'])[0]
            self.selected_features.append(best_main)
            self.fit(dataset, self.selected_features)
        
        return self.selected_features

    def _is_feature_better(self, metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> bool:
        """
        Compare two features based on their metrics.
        Returns True if feature A is better than feature B.
        
        Args:
            metrics_a: Dictionary of metrics for feature A
            metrics_b: Dictionary of metrics for feature B
            
        Returns:
            bool: True if feature A wins the comparison
        """
        effect_a = abs(metrics_a['model_effect'])
        effect_b = abs(metrics_b['model_effect'])
        
        consistency_a = metrics_a['effect_consistency']
        consistency_b = metrics_b['effect_consistency']
        
        predictive_a = metrics_a['predictive_power']
        predictive_b = metrics_b['predictive_power']
        
        # Count how many measures are better
        better_measures = 0
        if effect_a > effect_b: better_measures += 1
        if consistency_a > consistency_b: better_measures += 1
        if predictive_a > predictive_b: better_measures += 1
        
        # Feature A is better if it wins on majority of measures
        return better_measures >= 2
            
    def _calculate_feature_sparsity(self) -> float:
        """Calculate proportion of meaningful feature weights."""
        try:
            weights = self.get_feature_weights()
            if not weights:
                return 0.0
                
            # Count weights above threshold
            threshold = getattr(self.config, 'model', {}).get('sparsity_threshold', 0.1)
            significant_weights = sum(1 for w, _ in weights.values() if abs(w) > threshold)
            
            return significant_weights / len(weights)
            
        except Exception as e:
            logger.warning(f"Error calculating feature sparsity: {str(e)}")
            return 0.0

    def _check_transitivity(self) -> float:
        """Check transitivity violations more efficiently."""
        try:
            # Pre-compute all predictions to avoid redundant calculations
            predictions = {}
            for pref in self.dataset.preferences:
                key = (pref.bigram1, pref.bigram2)
                if key not in predictions:
                    pred, _ = self.predict_preference(key[0], key[1])
                    predictions[key] = pred
                    predictions[(key[1], key[0])] = 1 - pred
                    
            violations = 0
            total = 0
            
            # Check transitivity using pre-computed predictions
            seen_triples = set()
            for i, pref_i in enumerate(self.dataset.preferences[:-2]):
                for j, pref_j in enumerate(self.dataset.preferences[i+1:-1]):
                    for pref_k in self.dataset.preferences[j+1:]:
                        bigrams = {pref_i.bigram1, pref_i.bigram2,
                                pref_j.bigram1, pref_j.bigram2,
                                pref_k.bigram1, pref_k.bigram2}
                        
                        if len(bigrams) < 3:
                            continue
                            
                        triple = tuple(sorted(bigrams))
                        if triple in seen_triples:
                            continue
                        seen_triples.add(triple)
                        
                        pred_ij = predictions[(pref_i.bigram1, pref_i.bigram2)]
                        pred_jk = predictions[(pref_j.bigram1, pref_j.bigram2)]
                        pred_ik = predictions[(pref_i.bigram1, pref_k.bigram2)]
                        
                        if ((pred_ij > 0.5 and pred_jk > 0.5 and pred_ik < 0.5) or
                            (pred_ij < 0.5 and pred_jk < 0.5 and pred_ik > 0.5)):
                            violations += 1
                        total += 1
            
            return 1.0 - (violations / total) if total > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error checking transitivity: {str(e)}")
            return 0.0
                
    def _compute_feature_values(self, feature: str, dataset: PreferenceDataset) -> np.ndarray:
        """
        Compute feature values without caching.
        
        Args:
            feature: Feature name or interaction feature (containing '_x_')
            dataset: PreferenceDataset containing preferences
            
        Returns:
            np.ndarray of feature differences between bigram pairs
        """
        try:
            # Handle interaction features
            if '_x_' in feature:
                components = feature.split('_x_')
                # Get component values
                component_values = []
                for comp in components:
                    values = self._compute_single_feature_values(comp, dataset)
                    component_values.append(values)
                
                # Multiply components for interaction
                feature_diffs = component_values[0]
                for values in component_values[1:]:
                    feature_diffs = feature_diffs * values
                return feature_diffs
                
            else:
                # Handle single features
                return self._compute_single_feature_values(feature, dataset)
                
        except Exception as e:
            logger.error(f"Error computing feature values for {feature}: {str(e)}")
            return np.zeros(len(dataset.preferences))

    def _compute_single_feature_values(self, feature: str, dataset: PreferenceDataset) -> np.ndarray:
        """
        Compute values for a single (non-interaction) feature.
        
        Args:
            feature: Feature name (e.g., 'typing_time', 'same_finger', etc.)
            dataset: PreferenceDataset containing preferences
            
        Returns:
            np.ndarray of feature differences between bigram pairs
        """
        try:
            # Initialize array for feature differences
            feature_diffs = np.zeros(len(dataset.preferences))
            
            # For each preference, compute feature difference
            for i, pref in enumerate(dataset.preferences):
                # Get feature values for both bigrams
                value1 = pref.features1.get(feature, 0.0)
                value2 = pref.features2.get(feature, 0.0)
                
                # Special handling for typing_time
                if feature == 'typing_time':
                    # Use mean time if either value is None
                    if value1 is None or value2 is None:
                        valid_times = [t for t in [pref.typing_time1, pref.typing_time2] 
                                    if t is not None]
                        mean_time = np.mean(valid_times) if valid_times else 0.0
                        value1 = mean_time if value1 is None else value1
                        value2 = mean_time if value2 is None else value2
                
                # Compute difference
                feature_diffs[i] = value1 - value2
                
            # Debug logging
            logger.debug(f"Feature {feature} difference stats:")
            logger.debug(f"  mean: {np.mean(feature_diffs):.4f}")
            logger.debug(f"  std: {np.std(feature_diffs):.4f}")
            logger.debug(f"  range: [{np.min(feature_diffs):.4f}, {np.max(feature_diffs):.4f}]")
            
            return feature_diffs
            
        except Exception as e:
            logger.error(f"Error computing single feature values for {feature}: {str(e)}")
            return np.zeros(len(dataset.preferences))
                                                                                                                            
    def save_metrics_report(self, metrics_dict: Dict[str, Dict[str, float]], output_file: str):
        """Generate and save a detailed metrics report."""
        report_df = pd.DataFrame([
            {
                'Feature': feature,
                'Model Effect': metrics['model_effect'],
                'Effect Consistency': metrics['effect_consistency'],
                'Predictive Power': metrics['predictive_power']
            }
            for feature, metrics in metrics_dict.items()
        ])
        
        # Save to CSV
        report_df.to_csv(output_file, index=False)

    #--------------------------------------------
    # Statistical and model diagnostic methods
    #--------------------------------------------
    def predict_preference(self, bigram1: str, bigram2: str) -> ModelPrediction:
        """
        Predict preference probability and uncertainty for a bigram pair.
        Accounts for both main features and control features.
        
        Args:
            bigram1: First bigram to compare
            bigram2: Second bigram to compare
            
        Returns:
            ModelPrediction containing probability, uncertainty and metadata
            
        Raises:
            NotFittedError: If model hasn't been fit
            ValueError: If bigrams are invalid
        """
        if not self.fit_result:
            raise NotFittedError("Model must be fit before making predictions")
            
        try:
            # Validate inputs
            if len(bigram1) != 2 or len(bigram2) != 2:
                raise ValueError(f"Invalid bigram lengths: {bigram1}, {bigram2}")
                
            # Check cache
            cache_key = (bigram1, bigram2)
            cached_prediction = self.prediction_cache.get(cache_key)
            if cached_prediction is not None:
                return cached_prediction
                
            # Get comfort scores (these now include control features)
            start_time = time.perf_counter()
            score1, unc1 = self.get_bigram_comfort_scores(bigram1)
            score2, unc2 = self.get_bigram_comfort_scores(bigram2)
            
           # Separate main and control feature effects
            main_features = [f for f in self.selected_features 
                            if f not in self.config.features.control_features]
            control_features = self.config.features.control_features

            if not main_features:
                raise ValueError("No main features available for prediction")

            # Extract base features
            features1 = self.feature_extractor.extract_bigram_features(bigram1[0], bigram1[1])
            features2 = self.feature_extractor.extract_bigram_features(bigram2[0], bigram2[1])            
            # Add interaction features
            for f1, f2 in self.config.features.interactions:

                # Validate base features exist
                if f1 not in features1 or f2 not in features1:
                    logger.warning(f"Missing base features for interaction: {f1}, {f2}")
                    continue

                # Check both possible orderings of the interaction
                interaction_name1 = f"{f1}_x_{f2}"
                interaction_name2 = f"{f2}_x_{f1}"
                
                if interaction_name1 in self.selected_features:
                    features1[interaction_name1] = features1[f1] * features1[f2]
                    features2[interaction_name1] = features2[f1] * features2[f2]
                elif interaction_name2 in self.selected_features:
                    features1[interaction_name2] = features1[f1] * features1[f2]
                    features2[interaction_name2] = features2[f1] * features2[f2]

            logger.debug(f"Available features in bigram1: {list(features1.keys())}")
            logger.debug(f"Available features in bigram2: {list(features2.keys())}")

            # Get posterior samples for both main and control effects
            n_samples = self.config.model.n_samples
            
            # Get main feature effects
            beta_samples = self.fit_result.stan_variable('beta')
            main_diffs = np.array([features1[f] - features2[f] for f in main_features]).reshape(1, -1)  # Add batch dimension
            main_effects = np.dot(main_diffs, beta_samples.T)  # Now shapes will align correctly
            
            # Get control feature effects
            if control_features:
                gamma_samples = self.fit_result.stan_variable('gamma')
                control_diffs = np.array([features1[f] - features2[f] for f in control_features]).reshape(1, -1)
                control_effects = np.dot(control_diffs, gamma_samples.T)
                
                # Combine effects
                total_effects = main_effects + control_effects
            else:
                total_effects = main_effects
            
            # Transform to probabilities
            probs = 1 / (1 + np.exp(-total_effects))
            
            # Create prediction object
            prediction = ModelPrediction(
                probability=float(np.mean(probs)),
                uncertainty=float(np.std(probs)),
                features_used=list(self.selected_features),  # Includes both main and control features
                computation_time=time.perf_counter() - start_time
            )
            
            # Cache result
            self.prediction_cache.set(cache_key, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting preference: {str(e)}")
            return ModelPrediction(
                probability=0.5,
                uncertainty=1.0,
                features_used=[],
                computation_time=0.0
            )
        
    def _check_diagnostics(self) -> None:
        """Check MCMC diagnostics with proper type handling."""
        try:
            if hasattr(self.fit_result, 'diagnose'):
                diagnostic_info = self.fit_result.diagnose()
                logger.debug("Diagnostic Information:")
                logger.debug(diagnostic_info)
            
            summary = self.fit_result.summary()
            rhat_col = next((col for col in summary.columns 
                            if any(x in col.lower() 
                                for x in ['r_hat', 'rhat', 'r-hat'])), None)
            
            if rhat_col:
                rhat = summary[rhat_col].astype(float)
                if (rhat > 1.1).any():
                    logger.warning("Some parameters have high R-hat (>1.1)")
                    high_rhat_params = summary.index[rhat > 1.1]
                    logger.warning(f"Parameters with high R-hat: {high_rhat_params}")
                    
                    # Call diagnose() when there are convergence issues
                    if hasattr(self.fit_result, 'diagnose'):
                        logger.info("Running detailed diagnostics...")
                        self.fit_result.diagnose()
                        
        except Exception as e:
            logger.warning(f"Error in diagnostics: {str(e)}")
                                        
    def _compute_model_metrics(self) -> Dict[str, float]:
        """Compute model-specific performance metrics."""
        metrics = {}
        try:
            # Classification metrics
            y_true = []
            y_pred = []
            uncertainties = []
            
            for pref in self.dataset.preferences:
                pred_prob, pred_std = self.predict_preference(
                    pref.bigram1, pref.bigram2)
                
                y_true.append(1.0 if pref.preferred else 0.0)
                y_pred.append(pred_prob)
                uncertainties.append(pred_std)
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
            metrics['auc'] = roc_auc_score(y_true, y_pred)
            metrics['uncertainty'] = float(np.mean(uncertainties))
            metrics['transitivity'] = self._check_transitivity()
            metrics['feature_sparsity'] = self._calculate_feature_sparsity()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing model metrics: {str(e)}")
            return {}

    def _calculate_model_metrics(self, feature: str, dataset: PreferenceDataset) -> Dict[str, float]:
        """
        Calculate model-specific metrics for a feature.
        
        Args:
            feature: Feature to evaluate
            dataset: Dataset to evaluate on
            
        Returns:
            Dict containing model-specific metrics:
            - model_effect: Absolute effect size
            - effect_consistency: Cross-validation based consistency
            - predictive_power: Improvement in model predictions
            - feature_weight: Current weight in model
            - weight_uncertainty: Uncertainty in weight estimate
        """
        try:
            metrics = {}
            
            # Get feature weight and uncertainty
            weights = self.get_feature_weights()
            if feature in weights:
                weight_mean, weight_std = weights[feature]
                metrics['feature_weight'] = weight_mean
                metrics['weight_uncertainty'] = weight_std
                metrics['model_effect'] = abs(weight_mean)
            else:
                metrics['feature_weight'] = 0.0
                metrics['weight_uncertainty'] = 1.0
                metrics['model_effect'] = 0.0
                
            # Calculate effect consistency through cross-validation
            metrics['effect_consistency'] = self._calculate_effect_consistency(feature, dataset)
            
            # Calculate predictive power
            metrics['predictive_power'] = self._calculate_predictive_power(feature, dataset, self)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating model metrics for {feature}: {str(e)}")
            return {
                'feature_weight': 0.0,
                'weight_uncertainty': 1.0,
                'model_effect': 0.0,
                'effect_consistency': 1.0,
                'predictive_power': 0.0  # Add this line
            }
                
    #--------------------------------------------
    # Cross-validation and splitting methods
    #--------------------------------------------
    def _get_cv_splits(self, dataset: PreferenceDataset, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits preserving participant structure."""
        from sklearn.model_selection import KFold
        
        # Get unique participants
        participants = list(dataset.participants)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        for train_part_idx, val_part_idx in kf.split(participants):
            # Get participant IDs for each split
            train_participants = {participants[i] for i in train_part_idx}
            
            # Get preference indices for each split
            train_indices = [i for i, pref in enumerate(dataset.preferences)
                           if pref.participant_id in train_participants]
            val_indices = [i for i, pref in enumerate(dataset.preferences)
                          if pref.participant_id not in train_participants]
            
            splits.append((np.array(train_indices), np.array(val_indices)))
            
        return splits
                            
    #--------------------------------------------
    # Output and visualization methods
    #--------------------------------------------
    def save(self, path: Path) -> None:
        """Save model state to file."""
        save_dict = {
            'config': self.config,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'feature_weights': self.feature_weights,
            'fit_result': self.fit_result,
            'interaction_metadata': self.interaction_metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
            
    @classmethod
    def load(cls, path: Path) -> 'PreferenceModel':
        """Load model state from file."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
            
        model = cls(config=save_dict['config'])
        model.feature_names = save_dict['feature_names']
        model.selected_features = save_dict['selected_features']
        model.feature_weights = save_dict['feature_weights']
        model.fit_result = save_dict['fit_result']
        model.interaction_metadata = save_dict['interaction_metadata']
        
        return model