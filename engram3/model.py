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
  1. Feature Management:
    - Dynamic feature extraction and caching
    - Interaction feature computation
    - Feature selection with cross-validation
    - Comfort score estimation

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
    >>> prob, unc = model.predict_preference("th", "he")
    >>> print(f"Preference probability: {prob:.2f} ± {unc:.2f}")

Notes:
    - Uses LRU caching for feature computations
    - Lazy feature computation
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
from typing import (
    Dict, List, Optional, Tuple, Any, Union
)
import pandas as pd
import time  # Needed for computation_time in predict_preference
import pickle
import copy

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
        self.feature_cache = CacheManager()
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
        self.feature_values_cache.clear()

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
        """Update feature weights from fitted model."""
        try:
            self.feature_weights = {}
            
            if hasattr(self, 'fit'):  # Stan backend
                beta = self.fit_result.stan_variable('beta')
                for i, feature in enumerate(self.feature_names):
                    self.feature_weights[feature] = (
                        float(np.mean(beta[:, i])),
                        float(np.std(beta[:, i]))
                    )
            elif hasattr(self, 'trace'):  # PyMC backend
                for i, feature in enumerate(self.feature_names):
                    weights = self.trace.get_values('feature_weights')[:, i]
                    self.feature_weights[feature] = (
                        float(np.mean(weights)),
                        float(np.std(weights))
                    )
                    
        except Exception as e:
            logger.error(f"Error updating feature weights: {str(e)}")
                        
    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        try:
            if not hasattr(self, 'fit') or self.fit_result is None:
                raise RuntimeError("Model must be fit before evaluation")
                
            predictions = []
            actuals = []
            
            for pref in dataset.preferences:
                try:
                    # Get prediction probability
                    pred_mean, _ = self.predict_preference(pref.bigram1, pref.bigram2)
                    if not np.isnan(pred_mean):
                        predictions.append(pred_mean)
                        actuals.append(1.0 if pref.preferred else 0.0)
                except Exception as e:
                    logger.debug(f"Skipping preference in evaluation due to: {str(e)}")
                    continue
                    
            if not predictions:
                logger.warning("No valid predictions for evaluation")
                return {
                    'accuracy': 0.0,
                    'auc': 0.5,
                    'n_evaluated': 0
                }
                
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            accuracy = np.mean((predictions > 0.5) == actuals)
            auc = roc_auc_score(actuals, predictions)
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'n_evaluated': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return {
                'accuracy': 0.0,
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

        # Calculate stability metrics
        stability_metrics = {}
        for feature in feature_names:
            stability_metrics[feature] = self._calculate_stability_metrics(dataset, feature)

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
                **stability_metrics.get(feature, {})
            } for feature in feature_names},
            output_file=self.config.feature_selection.metrics_file
        )
        
        return {
            'metrics': metrics,
            'selected_features': selected_features,
            'feature_effects': processed_effects,
            'importance_metrics': importance_metrics,
            'stability_metrics': stability_metrics,
            'fold_uncertainties': dict(metrics['mean_uncertainty'])
        }

    def select_features(self, dataset: PreferenceDataset) -> List[str]:
        """Select features using centralized importance calculator."""
        logger.info("Starting feature selection...")
        self.selected_features = []  # Start empty
        base_features = self.config.features.base_features
        all_features = self.config.features.get_all_features()
        
        # Initial model fit with base features to get started
        self.fit(dataset, base_features)
        
        iteration = 0
        while iteration < self.config.feature_selection.n_iterations:
            # Evaluate remaining features
            feature_metrics = {}
            for feature in [f for f in all_features if f not in self.selected_features]:
                metrics = self.importance_calculator.evaluate_feature(
                    feature=feature,
                    dataset=dataset,
                    model=self  # Changed from self.model to self
                )
                feature_metrics[feature] = metrics
                                    
            # Get importance scores and threshold
            importance_scores = [m['importance_score'] for m in feature_metrics.values()]
            threshold = self._calculate_adaptive_threshold(importance_scores)
            
            # Select best feature that passes criteria
            passing_features = [
                (f, m['importance_score']) 
                for i, (f, m) in enumerate(feature_metrics.items())
                if m['importance_score'] >= threshold
            ]
            
            if not passing_features:
                break
                
            best_feature, best_score = max(passing_features, key=lambda x: x[1])
            self.selected_features.append(best_feature)
            
            # Refit model with updated feature set
            self.fit(dataset, self.selected_features)
            iteration += 1
            
        return self.selected_features
               
    def _log_feature_selection_results(self, 
                                    selected_features: List[str],
                                    importance_metrics: Dict[str, Dict]) -> None:
        """Log feature selection results."""
        logger.info("\nSelected Features:")
        for feature in selected_features:
            metrics = importance_metrics.get(feature, {})
            logger.info(f"  {feature}:")
            logger.info(f"    Score: {metrics.get('importance_score', 0.0):.3f}")

    #--------------------------------------------
    # Data preparation and feature methods
    #--------------------------------------------
    def _get_feature_data(self, feature: str, dataset: Optional[PreferenceDataset] = None) -> Dict[str, np.ndarray]:
        """
        Centralized method for getting feature data with caching.
        
        Args:
            feature: Feature name to extract
            dataset: Optional dataset, defaults to self.dataset
            
        Returns:
            Dict containing:
                'values': Feature values array
                'differences': Feature differences array
                'raw_features': Raw feature dictionary
        """
        dataset = dataset or self.dataset
        cache_key = f"{dataset.file_path}_{feature}"
        
        if hasattr(self, '_feature_data_cache') and cache_key in self._feature_data_cache:
            return self._feature_data_cache[cache_key]
            
        if '_x_' in feature:
            # Handle interaction features
            feat1, feat2 = feature.split('_x_')
            data1 = self._get_feature_data(feat1, dataset)
            data2 = self._get_feature_data(feat2, dataset)
            
            values = data1['values'] * data2['values']
            differences = data1['differences'] * data2['differences']
            
        else:
            # Handle base features
            values = []
            differences = []
            raw_features = {}
            
            for pref in dataset.preferences:
                feat1 = self.feature_extractor.extract_bigram_features(
                    pref.bigram1[0], pref.bigram1[1]).get(feature, 0.0)
                feat2 = self.feature_extractor.extract_bigram_features(
                    pref.bigram2[0], pref.bigram2[1]).get(feature, 0.0)
                
                values.append(feat1)
                values.append(feat2)
                differences.append(feat1 - feat2)
                raw_features[pref.bigram1] = feat1
                raw_features[pref.bigram2] = feat2
                
        result = {
            'values': np.array(values),
            'differences': np.array(differences),
            'raw_features': raw_features
        }
        
        # Cache with size limit
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
        
        Args:
            dataset: PreferenceDataset containing preferences
            features: Optional list of features to use. If None, uses all features.
            
        Returns:
            Dictionary containing data formatted for Stan model
            
        Raises:
            ValueError: If feature dimensions don't match or data is invalid
        """
        try:
            # Get feature names
            self.feature_names = features if features is not None else dataset.get_feature_names()
            
            # Log feature information
            logger.info(f"Preparing data with {len(self.feature_names)} features:")
            for f in self.feature_names:
                logger.info(f"  - {f}")
            
            # Create participant ID mapping
            participant_ids = sorted(list(dataset.participants))
            participant_map = {pid: i+1 for i, pid in enumerate(participant_ids)}
            
            # Initialize arrays
            X1 = []  # Features for first bigram in each preference
            X2 = []  # Features for second bigram in each preference
            participant = []  # Participant IDs
            y = []  # Preference outcomes (1 if first bigram preferred, 0 otherwise)
            
            skipped_count = 0
            # Extract feature values and preferences
            for pref in dataset.preferences:
                try:
                    # Get features for both bigrams
                    features1 = []
                    features2 = []
                    for feature in self.feature_names:
                        feat1 = pref.features1.get(feature)
                        feat2 = pref.features2.get(feature)
                        
                        # Special handling for typing_time
                        if feature == 'typing_time':
                            if feat1 is None or feat2 is None:
                                # Get mean of valid times from this preference
                                valid_times = [t for t in [pref.typing_time1, pref.typing_time2] if t is not None]
                                mean_time = np.mean(valid_times) if valid_times else 0.0
                                feat1 = mean_time if feat1 is None else feat1
                                feat2 = mean_time if feat2 is None else feat2
                        
                        # For other features, use 0.0 for None
                        else:
                            feat1 = 0.0 if feat1 is None else feat1
                            feat2 = 0.0 if feat2 is None else feat2
                        
                        features1.append(feat1)
                        features2.append(feat2)

                    X1.append(features1)
                    X2.append(features2)
                    participant.append(participant_map[pref.participant_id])
                    y.append(1 if pref.preferred else 0)  # Integer values for Stan
                    
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
            participant = np.array(participant, dtype=np.int32)
            y = np.array(y, dtype=np.int32)
            
            # Validate dimensions and types
            if X1.shape[1] != len(self.feature_names):
                raise ValueError(f"Feature dimension mismatch: {X1.shape[1]} != {len(self.feature_names)}")
            
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"y must be integer type, got {y.dtype}")
            
            # Check for NaN values
            if np.any(np.isnan(X1)) or np.any(np.isnan(X2)):
                raise ValueError("NaN values found in feature matrices")
            
            # Log data dimensions
            logger.info(f"Data dimensions:")
            logger.info(f"  N (preferences): {len(y)}")
            logger.info(f"  P (participants): {len(participant_ids)}")
            logger.info(f"  F (features): {len(self.feature_names)}")
            logger.info(f"  X1 shape: {X1.shape}")
            logger.info(f"  X2 shape: {X2.shape}")
            
            # Prepare Stan data
            stan_data = {
                'N': len(y),
                'P': len(participant_ids),
                'F': len(self.feature_names),
                'X1': X1,
                'X2': X2,
                'participant': participant,
                'y': y,
                'feature_scale': self.config.model.feature_scale,
                'participant_scale': self.config.model.participant_scale
            }
            
            # Log summary statistics
            logger.info("\nFeature summary statistics:")
            for i, feature in enumerate(self.feature_names):
                logger.info(f"{feature}:")
                logger.info(f"  X1 - mean: {np.mean(X1[:, i]):.3f}, std: {np.std(X1[:, i]):.3f}")
                logger.info(f"  X2 - mean: {np.mean(X2[:, i]):.3f}, std: {np.std(X2[:, i]):.3f}")
            
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
        
        Returns:
            Tuple of (mean score, standard deviation)
        """
        try:
            # Extract features for bigram
            features = self._extract_features(bigram)
            
            # Get feature weights from posterior
            weights = self.get_feature_weights()
            
            # Calculate comfort score
            score = 0.0
            uncertainty = 0.0
            
            for feature, value in features.items():
                if feature in weights:
                    weight_mean, weight_std = weights[feature]
                    score += value * weight_mean
                    uncertainty += (value * weight_std) ** 2
                    
            uncertainty = np.sqrt(uncertainty)
            
            return float(score), float(uncertainty)
            
        except Exception as e:
            logger.error(f"Error calculating comfort scores: {str(e)}")
            return 0.0, 1.0

    def get_feature_weights(self) -> Dict[str, Tuple[float, float]]:
        """
        Get the learned feature weights including interactions.
        
        Returns:
            Dict mapping feature names to (mean, std) tuples of weights
            
        Raises:
            NotFittedError: If model hasn't been fit yet
            FeatureError: If error occurs extracting weights
        """
        if not self.fit_result:
            raise NotFittedError("Model must be fit before getting weights")
            
        # Use cached weights if available
        if self.feature_weights is not None:
            return self.feature_weights.copy()
            
        try:
            summary = self.fit_result.summary()
            weights: Dict[str, Tuple[float, float]] = {}
            
            for feature in self.selected_features:
                if feature in summary.index:
                    weights[feature] = (
                        float(summary.loc[feature, 'mean']),
                        float(summary.loc[feature, 'sd'])
                    )
                else:
                    logger.warning(f"Feature {feature} not found in model summary")
                    
            # Cache the results
            self.feature_weights = weights
            return weights.copy()
            
        except AttributeError:
            raise NotFittedError("Model fitting did not complete successfully")
        except KeyError as e:
            raise FeatureError(f"Error accessing feature weights: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting feature weights: {str(e)}")
            raise FeatureError(f"Failed to extract feature weights: {str(e)}")
                            
    #--------------------------------------------
    # Feature selection and evaluation methods
    #--------------------------------------------                          
    def _calculate_stability_metrics(self, dataset: PreferenceDataset,
                                feature: str) -> StabilityMetrics:
        """
        Calculate stability metrics for a feature using cross-validation.
        
        Args:
            dataset: PreferenceDataset to evaluate
            feature: Name of feature to assess stability
            
        Returns:
            Dict containing stability metrics:
            - effect_cv: Coefficient of variation of feature effects
            - sign_consistency: Consistency of effect direction
            - relative_range: Range of effects relative to mean
        """
        try:
            # Get n_splits from config or use default
            n_splits = getattr(self.config, 'model', {}).get('cross_validation', {}).get('n_splits', 5)
            
            # Perform cross-validation using shared splitting method
            cv_effects = []
            
            # Get train/val splits using common method
            for train_idx, val_idx in self._get_cv_splits(dataset, n_splits):
                # Create train dataset
                train_data = dataset._create_subset_dataset(train_idx)
                
                # Fit model on training data
                self.fit_result(train_data, self.selected_features + [feature])
                
                # Get feature effect
                weights = self.get_feature_weights()
                if feature in weights:
                    effect = weights[feature][0]  # Get mean effect
                    cv_effects.append(effect)
            
            cv_effects = np.array(cv_effects)
            
            if len(cv_effects) == 0:
                raise ValueError(f"No valid effects calculated for feature {feature}")
            
            # Calculate stability metrics
            metrics = {
                'effect_cv': np.std(cv_effects) / (abs(np.mean(cv_effects)) + 1e-10),
                'sign_consistency': np.mean(np.sign(cv_effects) == np.sign(np.mean(cv_effects))),
                'relative_range': (np.max(cv_effects) - np.min(cv_effects)) / (abs(np.mean(cv_effects)) + 1e-10)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating stability metrics for {feature}: {str(e)}")
            return {
                'effect_cv': float('inf'),
                'sign_consistency': 0.0,
                'relative_range': float('inf')
            }

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

    def _calculate_adaptive_threshold(self, importance_scores: np.ndarray) -> float:
        """
        Calculate adaptive threshold for feature importance scores.
        
        Args:
            importance_scores: Array of importance scores
            
        Returns:
            float: Adaptive threshold value
        """
        try:
            scores = np.array(importance_scores)
            if len(scores) == 0:
                return 0.0
                
            # Remove any invalid values
            scores = scores[~np.isnan(scores)]
            
            # Use various statistical measures to determine threshold
            mean = np.mean(scores)
            std = np.std(scores)
            
            # Use simple threshold based on mean and std
            threshold = mean + 0.1 * std  # Using fixed 0.1 factor
                    
            # Ensure threshold is not too extreme
            min_threshold = np.percentile(scores, 25)
            max_threshold = np.percentile(scores, 75)
            threshold = np.clip(threshold, min_threshold, max_threshold)
                
            logger.debug(f"Calculated adaptive threshold: {threshold:.3f}")
            return float(threshold)
                
        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {str(e)}")
            return 0.1  # Simple default threshold
        
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
            feature: Feature name
            dataset: PreferenceDataset containing preferences
            
        Returns:
            np.ndarray of feature differences between bigram pairs
        """
        values = []
        
        # Special handling for typing_time
        if feature == 'typing_time':
            # Calculate mean typing time once
            valid_times = []
            for pref in dataset.preferences:
                if pref.typing_time1 is not None:
                    valid_times.append(pref.typing_time1)
                if pref.typing_time2 is not None:
                    valid_times.append(pref.typing_time2)
            mean_time = np.mean(valid_times) if valid_times else 0.0
            
            # Use mean time for None values
            for pref in dataset.preferences:
                val1 = mean_time if pref.typing_time1 is None else pref.typing_time1
                val2 = mean_time if pref.typing_time2 is None else pref.typing_time2
                values.append(val1 - val2)
                
        # All other features
        else:
            for pref in dataset.preferences:
                val1 = pref.features1.get(feature, 0.0)
                val2 = pref.features2.get(feature, 0.0)
                val1 = 0.0 if val1 is None else val1
                val2 = 0.0 if val2 is None else val2
                values.append(val1 - val2)
                
        return np.array(values)
                                                                                                                     
    def save_metrics_report(self, metrics_dict: Dict[str, Dict[str, float]], output_file: str):
        """Generate and save a detailed metrics report."""
        report_df = pd.DataFrame([
            {
                'Feature': feature,
                'Model Effect': metrics['model_effect'],
                'Effect Consistency': metrics['effect_consistency'],
                'Predictive Power': metrics['predictive_power'],
                'Combined Score': metrics['importance_score']
            }
            for feature, metrics in metrics_dict.items()
        ])
        
        # Sort by combined score
        report_df = report_df.sort_values('Combined Score', ascending=False)
        
        # Save to CSV
        report_df.to_csv(output_file, index=False)

    #--------------------------------------------
    # Statistical and model diagnostic methods
    #--------------------------------------------
    def predict_preference(self, bigram1: str, bigram2: str) -> ModelPrediction:
        """
        Predict preference probability and uncertainty for a bigram pair.
        
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
                
            # Get comfort scores
            start_time = time.perf_counter()
            score1, unc1 = self.get_bigram_comfort_scores(bigram1)
            score2, unc2 = self.get_bigram_comfort_scores(bigram2)
            
            # Get posterior samples
            n_samples = self.config.model.n_samples
            samples = np.random.normal(
                loc=score1 - score2,
                scale=np.sqrt(unc1**2 + unc2**2),
                size=n_samples
            )
            
            # Transform to probabilities
            probs = 1 / (1 + np.exp(-samples))
            
            # Create prediction object
            prediction = ModelPrediction(
                probability=float(np.mean(probs)),
                uncertainty=float(np.std(probs)),
                features_used=list(self.selected_features),
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
                    f"{stability.get('effect_cv', 0.0):.6f}",
                    f"{stability.get('relative_range', 0.0):.6f}",
                    f"{stability.get('sign_consistency', 0.0):.6f}"
                ]
                f.write(','.join(values) + '\n')
            
            logger.info(f"Saved feature metrics to {csv_file}")

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