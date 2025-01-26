# model.py
"""
Hierarchical Bayesian model for keyboard layout preference learning.

Core functionality:
  - Bayesian inference using Bradley-Terry preference model structure:
    - Feature-based comfort score estimation
    - Participant-level random effects
    - Uncertainty quantification via MCMC sampling
    - Stan backend integration
    - Automated resource management

Key components:
  1. Sequential Feature Selection:
    - Round-robin comparison of features
    - Context-aware evaluation with previously selected features
    - Three independent metrics:
      * Model effect magnitude
      * Effect consistency across cross-validation
      * Predictive power
    - Feature interaction handling
    - Control feature separation

  2. Model Operations:
    - Efficient data preparation for Stan
    - MCMC sampling with diagnostics
    - Cross-validation with participant grouping
    - Feature weight extraction and caching
    - Memory usage optimization

  3. Prediction Pipeline:
    - Bigram comfort score estimation
    - Preference probability prediction
    - Uncertainty quantification
    - Cached predictions for efficiency
    - Baseline model handling

  4. Evaluation Mechanisms:
    - Classification metrics (accuracy, AUC)
    - Effect size estimation
    - Cross-validation stability
    - Model diagnostics
    - Transitivity checks
    - Parameter convergence monitoring

  5. Output Handling:
    - Detailed metric reporting
    - State serialization
    - Comprehensive logging
    - CSV export capabilities
    - Diagnostic information export

Classes:
    PreferenceModel: Main class implementing the Bayesian preference learning pipeline
        Methods:
            fit(): Train model on preference data
            predict_preference(): Generate predictions for bigram pairs
            evaluate(): Compute model performance metrics
            cross_validate(): Perform cross-validation analysis
            save()/load(): Model serialization

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
    - Automatic resource cleanup
    - Memory usage monitoring
    - Temporary file management

Dependencies:
    - cmdstanpy: Stan model compilation and sampling
    - numpy: Numerical computations
    - pandas: Data management
    - sklearn: Evaluation metrics
    - psutil: Resource monitoring
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
import traceback
from collections import defaultdict
import tempfile
import shutil
import psutil
from tenacity import retry, stop_after_attempt, wait_fixed
import gc
from contextlib import contextmanager

from engram3.utils.config import Config, NotFittedError, FeatureError, ModelPrediction
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
        if config is None:
            raise ValueError("Config is required")
        
        self.config = config if isinstance(config, Config) else Config(**config)
        
        # Initialize basic attributes
        self.fit_result = None
        self.feature_names = None
        self.selected_features = []
        self.interaction_metadata = {}
        self.dataset = None
        self.feature_weights = None
        self.feature_extractor = None
        self.is_baseline_model = False
        self._importance_calculator = None  # Use underscore for private attribute
        
        # Initialize caches
        self.feature_cache = CacheManager()
        self.prediction_cache = CacheManager()
        
        # Initialize visualization
        self.plotting = PlottingUtils(self.config.paths.plots_dir)

        # Initialize Stan model
        try:
            model_path = Path(__file__).parent / "models" / "preference_model.stan"
            if not model_path.exists():
                raise FileNotFoundError(f"Stan model file not found: {model_path}")
                
            output_dir = Path(self.config.paths.root_dir) / "stan_temp"
            output_dir.mkdir(parents=True, exist_ok=True)
                
            self.model = cmdstanpy.CmdStanModel(
                stan_file=str(model_path),
                cpp_options={'STAN_THREADS': True},
                stanc_options={'warn-pedantic': True}
            )
                
            if hasattr(self.model, 'exe_file'):
                exe_path = Path(self.model.exe_file)
                if exe_path.exists():
                    exe_path.chmod(0o755)

        except Exception as e:
            print(f"Error initializing Stan model: {str(e)}")
            raise

    @property 
    def importance_calculator(self):
        if self._importance_calculator is None:
            self._importance_calculator = FeatureImportanceCalculator(self.config, self)
        return self._importance_calculator

    @importance_calculator.setter
    def importance_calculator(self, value):
        self._importance_calculator = value
               
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

    @contextmanager
    def _create_temp_model(self):
        """Context manager for temporary model instances."""
        temp_model = type(self)(config=self.config)
        try:
            yield temp_model
        finally:
            temp_model.cleanup()

    # Remaining class methods
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self.feature_cache.clear()
        self.prediction_cache.clear()

    def _check_memory(self, required_gb=8):
        """Check available system memory"""
        mem = psutil.virtual_memory()
        if mem.available < required_gb * 1024**3:
            raise RuntimeError(f"Insufficient memory: {mem.available/(1024**3):.1f}GB free, need {required_gb}GB")

    def _check_temp_space(self):
        required_mb = self.config.model.required_temp_mb  # Default value is already 2000
        temp_dir = Path(tempfile.gettempdir())
        total, used, free = shutil.disk_usage(temp_dir)
        free_mb = free // (1024 * 1024)
        if free_mb < required_mb:
            raise RuntimeError(f"Insufficient temporary space: {free_mb}MB free, need {required_mb}MB")
                    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _sample_with_retry(self, **kwargs):
        """Attempt sampling with retries on failure"""
        return self.model.sample(**kwargs)

    def fit(self, dataset: PreferenceDataset, features: Optional[List[str]] = None, fit_purpose: str = None) -> None:
        """
        Fit the model to the given dataset.
        
        Args:
            dataset: Dataset to fit
            features: Optional list of features to use (default: all available)
            fit_purpose: Description of what this fit is for (logging purposes)
        """
        try:

            # Initial validation
            if not dataset or not hasattr(dataset, 'preferences') or len(dataset.preferences) == 0:
                raise ValueError("Empty dataset")
            if features is not None and len(features) == 0:
                raise ValueError("Empty features list")

            # Resource checks
            self._check_memory()
            self._check_temp_space()

            # Clear previous results
            for attr in ['fit_result', 'feature_weights']:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Clean temp directories
            temp_dirs = [tempfile.gettempdir(), self.config.paths.stan_temp]
            for temp_dir in temp_dirs:
                for pattern in ["preference_model*", "stanModel*", "stan_*"]:
                    for file in Path(temp_dir).glob(pattern):
                        try:
                            if file.is_file():
                                file.unlink()
                            elif file.is_dir():
                                shutil.rmtree(file)
                        except OSError:
                            pass

            if fit_purpose:
                logger.info(f"\nFitting model: {fit_purpose}")

            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor
            
            # Validate features
            if not features:
                features = dataset.get_feature_names()
            if not features:
                raise ValueError("No features provided and none available from dataset")
            
            # Ensure proper feature handling
            control_features = self.config.features.control_features
            main_features = [f for f in features if f not in control_features]
            
            # Store unique features (no duplicates)
            self.feature_names = list(dict.fromkeys(main_features + control_features))
            self.selected_features = list(dict.fromkeys(main_features + control_features))

            # Prepare data
            stan_data = self.prepare_data(dataset, features)
            
            # Validate Stan data
            if stan_data['F'] < 1:
                raise ValueError(f"Invalid number of features: {stan_data['F']}")

            sampling_msg = f"Starting sampling with {self.config.model.chains} chains, " + \
                                f"{self.config.model.warmup} warmup iterations, " + \
                                f"{self.config.model.n_samples} sampling iterations"
            if fit_purpose:
                sampling_msg += f" ({fit_purpose})"
            logger.info(sampling_msg)

            # Log Stan configuration
            logger.info("Stan configuration:")
            logger.info(f"  Chains: {self.config.model.chains}")
            logger.info(f"  Warmup: {self.config.model.warmup}")
            logger.info(f"  Samples: {self.config.model.n_samples}")
            logger.info(f"  Max treedepth: {self.config.model.max_treedepth}")
            logger.info(f"  Adapt delta: {self.config.model.adapt_delta}")

            # Force garbage collection before sampling
            gc.collect()

            try:
                self.fit_result = self._sample_with_retry(
                    data=stan_data,
                    chains=self.config.model.chains,
                    iter_warmup=self.config.model.warmup,
                    iter_sampling=self.config.model.n_samples,
                    adapt_delta=self.config.model.adapt_delta,
                    max_treedepth=self.config.model.max_treedepth,
                    show_progress=True,
                    refresh=max(1, min(self.config.model.n_samples // 10, 100)),
                    save_warmup=False
                )
            except Exception as stan_error:
                logger.error("Stan sampling error details:")
                for f in Path(self.config.paths.stan_temp).glob("*-std*.txt"):
                    logger.error(f"\n{f.name}:")
                    logger.error(f.read_text())
                raise RuntimeError(f"Stan sampling failed: {str(stan_error)}")

            # Force garbage collection after sampling
            gc.collect()

            self._check_diagnostics()
            self._update_feature_weights()

        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def _update_feature_weights(self) -> None:
        """Update feature weights from fitted model for both main and control features."""
        try:
            self.feature_weights = {}
            
            if not hasattr(self, 'fit_result'):
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Get main feature weights (beta) if any exist
            main_features = list(dict.fromkeys(
                [f for f in self.feature_names if f not in self.config.features.control_features]
            ))
            
            if main_features:  # Only process beta if we have main features
                beta = self.fit_result.stan_variable('beta')
                logger.debug(f"Beta shape: {beta.shape}")
                logger.debug(f"Main features: {main_features}")
                
                if beta.shape[1] != len(main_features):
                    raise ValueError(f"Beta shape mismatch: {beta.shape[1]} != {len(main_features)}")
                
                # Update weights for main features
                for i, feature in enumerate(main_features):
                    self.feature_weights[feature] = (
                        float(np.mean(beta[:, i])),
                        float(np.std(beta[:, i]))
                    )
            
            # Get control feature weights (gamma) if any exist
            control_features = list(dict.fromkeys(self.config.features.control_features))
            if control_features:
                try:
                    gamma = self.fit_result.stan_variable('gamma')
                    if gamma.shape[1] != len(control_features):
                        raise ValueError(f"Gamma shape mismatch: {gamma.shape[1]} != {len(control_features)}")
                        
                    for i, feature in enumerate(control_features):
                        self.feature_weights[feature] = (
                            float(np.mean(gamma[:, i])),
                            float(np.std(gamma[:, i]))
                        )
                except Exception as e:
                    logger.error(f"Error processing control features: {str(e)}")
                    raise

            logger.debug("Updated weights:")
            for feature, (mean, std) in self.feature_weights.items():
                feature_type = "control" if feature in control_features else "main"
                logger.debug(f"  {feature} ({feature_type}): {mean:.4f} ± {std:.4f}")
                    
        except Exception as e:
            logger.error(f"Error updating feature weights: {str(e)}")
            raise
                                                                            
    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        logger.info("=== Starting model evaluation ===")
        logger.info(f"Model state:")
        logger.info(f"  Feature names: {list(dict.fromkeys(self.feature_names)) if hasattr(self, 'feature_names') else 'No feature_names'}")
        logger.info(f"  Selected features: {list(dict.fromkeys(self.selected_features)) if hasattr(self, 'selected_features') else 'No selected_features'}")
        logger.info(f"  Feature weights: {self.feature_weights if hasattr(self, 'feature_weights') else 'No feature_weights'}")
        
        try:
            if not hasattr(self, 'fit_result') or self.fit_result is None:
                logger.error("Model must be fit before evaluation")
                raise RuntimeError("Model must be fit before evaluation")
                
            predictions = []
            actuals = []
            feature_values = defaultdict(list)  # Track feature values
            
            logger.info(f"Evaluating model on {len(dataset.preferences)} preferences")
            
            # Sample some predictions for debugging
            debug_samples = []
            
            for pref_idx, pref in enumerate(dataset.preferences):
                try:
                    # Get prediction probability - now handling ModelPrediction object
                    prediction = self.predict_preference(pref.bigram1, pref.bigram2)
                    if not np.isnan(prediction.probability):
                        predictions.append(prediction.probability)
                        actuals.append(1.0 if pref.preferred else 0.0)
                        
                        # Track feature values for debugging
                        for feature in self.feature_names:
                            val1 = pref.features1.get(feature, 0.0)
                            val2 = pref.features2.get(feature, 0.0)
                            feature_values[feature].append(val1 - val2)
                        
                        # Store some samples for debugging
                        if pref_idx < 5:  # First 5 examples
                            debug_samples.append({
                                'bigrams': (pref.bigram1, pref.bigram2),
                                'features': {f: (pref.features1.get(f, 0.0), 
                                            pref.features2.get(f, 0.0))
                                        for f in self.feature_names},
                                'prediction': prediction.probability,
                                'actual': 1.0 if pref.preferred else 0.0
                            })
                            
                except Exception as e:
                    logger.debug(f"Skipping preference in evaluation due to: {str(e)}")
                    continue
                    
            if not predictions:
                logger.warning("No valid predictions for evaluation")
                return {
                    'accuracy': 0.5,
                    'auc': 0.5,
                    'n_evaluated': 0
                }
            
            logger.info(f"Made predictions for {len(predictions)} preferences")
            
            # Log feature value statistics
            logger.debug("\nFeature value distributions:")
            for feature in self.feature_names:
                values = np.array(feature_values[feature])
                logger.debug(f"{feature}:")
                logger.debug(f"  Mean: {np.mean(values):.4f}")
                logger.debug(f"  Std: {np.std(values):.4f}")
                logger.debug(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
            
            # Log some example predictions
            logger.debug("\nExample predictions:")
            for sample in debug_samples:
                logger.debug(f"\nBigrams: {sample['bigrams']}")
                logger.debug(f"Features: {sample['features']}")
                logger.debug(f"Prediction: {sample['prediction']:.4f}")
                logger.debug(f"Actual: {sample['actual']}")
                
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            accuracy = np.mean((predictions > 0.5) == actuals)
            auc = roc_auc_score(actuals, predictions)
            
            logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            logger.debug(f"Prediction stats - Mean: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")
            
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
                        prediction = self.predict_preference(pref.bigram1, pref.bigram2)
                        pred_prob = prediction.probability
                        pred_std = prediction.uncertainty
                        if not np.isnan(pred_prob):
                            val_predictions.append(pred_prob)
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
                    model=self,
                    all_features=feature_names,  # We have this from earlier in the method
                    current_selected_features=self.selected_features
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
                
                if all_times:  # Only calculate if we have valid times
                    time_mean = np.mean(all_times)
                    time_std = np.std(all_times)
                else:
                    time_mean = 0.0
                    time_std = 1.0
                
                for pref in dataset.preferences:
                    time1 = pref.typing_time1
                    time2 = pref.typing_time2
                    
                    # Replace None values with mean
                    time1 = time_mean if time1 is None else time1
                    time2 = time_mean if time2 is None else time2
                    
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
            Prepare data for Stan model with standardization, validation and cleaning.
            Handles both main features and control features separately.
            """
            try:
                processed_features = set()

                # Get feature names and separate control features
                self.feature_names = list(dict.fromkeys(
                    features if features is not None else dataset.get_feature_names()
                ))
                control_features = self.config.features.control_features
                
                # Main features are those not in control_features - deduplicate
                main_features = list(dict.fromkeys(
                    [f for f in self.feature_names if f not in control_features]
                ))
                
                # Check if this is a control-only model
                is_control_only = not main_features and all(f in control_features for f in self.feature_names)
                
                if not is_control_only and not main_features:
                    raise ValueError("No main features provided for model fitting")
                    
                logger.info(f"Preparing data:")
                logger.info(f"  Main features ({len(main_features)}): {main_features}")
                logger.info(f"  Control features ({len(control_features)}): {control_features}")
                
                # Create participant ID mapping
                participant_ids = sorted(list(dataset.participants))
                participant_map = {pid: i+1 for i, pid in enumerate(participant_ids)}
                
                # Track feature statistics for standardization
                feature_stats = defaultdict(lambda: {'values': [], 'mean': None, 'std': None})
                
                # First pass: collect values for standardization
                logger.info("First pass: collecting values for standardization")
                for pref in dataset.preferences:
                    # Collect main feature values
                    if not is_control_only:
                        for feature in main_features:

                            if feature not in processed_features:
                                logger.debug(f"Processing base feature: {feature}")
                                processed_features.add(feature)

                            if '_x_' in feature:
                                # Split interaction name into component features
                                # e.g., 'same_finger_x_sum_finger_values_x_rows_apart' -> 
                                # ['same_finger', 'sum_finger_values', 'rows_apart']
                                components = feature.split('_x_')
                                
                                # For each component feature, collect its values for both bigrams in the preference
                                # This maintains statistics for the base features themselves
                                for component in components:
                                    # Get values for this component from both bigrams in the preference
                                    feat1 = pref.features1.get(component, 0.0)  # Value for first bigram
                                    feat2 = pref.features2.get(component, 0.0)  # Value for second bigram
                                    # Store both values for computing statistics on the base feature
                                    feature_stats[component]['values'].extend([feat1, feat2])
                                
                                # Now compute the interaction value for each bigram
                                # For each bigram, multiply together all its component feature values
                                feat1_interaction = 1.0  # Will hold product of all components for first bigram
                                feat2_interaction = 1.0  # Will hold product of all components for second bigram
                                for component in components:
                                    # Multiply each component's value into the running product for each bigram
                                    feat1_interaction *= pref.features1.get(component, 0.0)  # First bigram's product
                                    feat2_interaction *= pref.features2.get(component, 0.0)  # Second bigram's product
                                
                                # Store both interaction values (one per bigram) for computing interaction statistics
                                feature_stats[feature]['values'].extend([feat1_interaction, feat2_interaction])
                            else:
                                # For base features, simply get and store the value from each bigram
                                feat1 = pref.features1.get(feature, 0.0)  # Value from first bigram
                                feat2 = pref.features2.get(feature, 0.0)  # Value from second bigram
                                if feature == 'typing_time' and (feat1 is None or feat2 is None):
                                    feat1 = 0.0
                                    feat2 = 0.0
                                feature_stats[feature]['values'].extend([feat1, feat2])

                    # Collect control feature values
                    for feature in control_features:

                        if feature not in processed_features:
                            logger.debug(f"Processing base feature: {feature}")
                            processed_features.add(feature)

                        feat1 = pref.features1.get(feature, 0.0)
                        feat2 = pref.features2.get(feature, 0.0)
                        feature_stats[feature]['values'].extend([feat1, feat2])
                
                # Calculate standardization parameters
                logger.info("Computing standardization parameters")
                for feature, stats in feature_stats.items():
                    values = np.array(stats['values'])

                    # Filter out None values before calculating statistics
                    values = np.array([v for v in values if v is not None])

                    if len(values) > 0:
                        stats['mean'] = float(np.mean(values))
                        stats['std'] = float(np.std(values))
                    else:
                        # Handle case where all values are None
                        stats['mean'] = 0.0
                        stats['std'] = 1.0

                    logger.debug(f"Feature {feature} stats - mean: {stats['mean']}, std: {stats['std']}")

                # Second pass: build standardized matrices
                processed_features.clear()  # Reset before second pass
                logger.info("Second pass: building standardized matrices")
                X1, X2 = [], []  # Main feature matrices
                C1, C2 = [], []  # Control feature matrices
                participant = []
                y = []
                
                skipped_count = 0
                for pref in dataset.preferences:
                    try:
                        # Process main features
                        if is_control_only:
                            features1_main = [0.0]  # Dummy feature
                            features2_main = [0.0]
                        else:
                            features1_main = []
                            features2_main = []
                            for feature in main_features:

                                if feature not in processed_features:
                                    logger.debug(f"Processing base feature: {feature}")
                                    processed_features.add(feature)

                                try:

                                    if '_x_' in feature:
                                        # Add detailed logging
                                        logger.debug(f"Processing interaction feature: {feature}")
                                        components = feature.split('_x_')
                                        logger.debug(f"Components: {components}")
                                        
                                        # Get standardized base features with error checking
                                        for component in components:
                                            if component not in feature_stats:
                                                logger.error(f"Missing statistics for component: {component}")
                                                raise ValueError(f"Missing statistics for component: {component}")
                                        
                                        # Initialize products for interaction
                                        feat1_interaction = 1.0
                                        feat2_interaction = 1.0
                                        
                                        # Multiply standardized values for each component
                                        for component in components:
                                            feat1 = pref.features1.get(component, 0.0)
                                            feat2 = pref.features2.get(component, 0.0)
                                            if component == 'typing_time' and (feat1 is None or feat2 is None):
                                                feat1 = 0.0
                                                feat2 = 0.0
                                            
                                            feat1_base = (feat1 - feature_stats[component]['mean']) / feature_stats[component]['std']
                                            feat2_base = (feat2 - feature_stats[component]['mean']) / feature_stats[component]['std']
                                            
                                            feat1_interaction *= feat1_base
                                            feat2_interaction *= feat2_base
                                        
                                        features1_main.append(feat1_interaction)
                                        features2_main.append(feat2_interaction)
                                    else:
                                        # Add logging for base features
                                        logger.debug(f"Processing base feature: {feature}")
                                        if feature not in feature_stats:
                                            logger.error(f"Missing statistics for feature: {feature}")
                                            raise ValueError(f"Missing statistics for feature: {feature}")
                                        
                                        feat1 = pref.features1.get(feature, 0.0)
                                        feat2 = pref.features2.get(feature, 0.0)
                                        if feature == 'typing_time' and (feat1 is None or feat2 is None):
                                            feat1 = 0.0
                                            feat2 = 0.0

                                        # Always standardize, even if we've set values to zero
                                        feat1 = (feat1 - feature_stats[feature]['mean']) / feature_stats[feature]['std']
                                        feat2 = (feat2 - feature_stats[feature]['mean']) / feature_stats[feature]['std']
                                        
                                        features1_main.append(feat1)
                                        features2_main.append(feat2)

                                except Exception as e:
                                    logger.error(f"Error processing feature {feature}: {str(e)}")
                                    raise  # Re-raise to be caught by outer try-except
                                    
                        # Process control features with error checking
                        features1_control = []
                        features2_control = []
                        for feature in control_features:

                            if feature not in processed_features:
                                logger.debug(f"Processing base feature: {feature}")
                                processed_features.add(feature)

                            if feature not in feature_stats:
                                logger.error(f"Missing statistics for control feature: {feature}")
                                raise ValueError(f"Missing statistics for control feature: {feature}")
                                
                            feat1 = (pref.features1.get(feature, 0.0) - feature_stats[feature]['mean']) / feature_stats[feature]['std']
                            feat2 = (pref.features2.get(feature, 0.0) - feature_stats[feature]['mean']) / feature_stats[feature]['std']
                            features1_control.append(feat1)
                            features2_control.append(feat2)
                            
                        # Add arrays only if we got here without errors
                        X1.append(features1_main)
                        X2.append(features2_main)
                        C1.append(features1_control)
                        C2.append(features2_control)
                        participant.append(participant_map[pref.participant_id])
                        y.append(1 if pref.preferred else 0)
                        
                    except Exception as e:
                        logger.warning(f"Skipping preference due to error: {str(e)}")
                        logger.warning(f"Preference details: bigram1={pref.bigram1}, bigram2={pref.bigram2}")
                        logger.warning(f"Features1: {pref.features1}")
                        logger.warning(f"Features2: {pref.features2}")
                        skipped_count += 1
                        continue

                if skipped_count == len(dataset.preferences):
                    logger.error("All preferences were skipped due to errors")
                    raise ValueError("No valid preferences remained after processing")
                elif skipped_count > 0:
                    logger.warning(f"Skipped {skipped_count} preferences due to invalid features")

                # Convert to numpy arrays
                X1 = np.array(X1, dtype=np.float64)
                X2 = np.array(X2, dtype=np.float64)
                C1 = np.array(C1, dtype=np.float64)
                C2 = np.array(C2, dtype=np.float64)
                participant = np.array(participant, dtype=np.int32)
                y = np.array(y, dtype=np.int32)

                # Check if we have any valid data
                if len(X1) == 0:
                    logger.error("No valid preferences after processing")
                    raise ValueError("No valid preferences remained after processing")

                # Reshape arrays if needed
                if is_control_only:
                    if len(X1) > 0 and X1.ndim == 1:
                        X1 = X1.reshape(-1, 1)
                        X2 = X2.reshape(-1, 1)
                else:
                    if len(X1) > 0 and X1.ndim == 1 and len(main_features) > 0:
                        X1 = X1.reshape(-1, len(main_features))
                        X2 = X2.reshape(-1, len(main_features))

                if len(C1) > 0 and C1.ndim == 1 and len(control_features) > 0:
                    C1 = C1.reshape(-1, len(control_features))
                    C2 = C2.reshape(-1, len(control_features))

                # Verify dimensions and check for NaNs
                if not is_control_only:
                    if X1.shape[1] != len(main_features):
                        raise ValueError(f"Main feature dimension mismatch: {X1.shape[1]} != {len(main_features)}")
                else:
                    if X1.shape[1] != 1:
                        raise ValueError("Invalid dummy feature dimension for control-only model")
                        
                if C1.shape[1] != len(control_features):
                    raise ValueError(f"Control feature dimension mismatch: {C1.shape[1]} != {len(control_features)}")
                    
                if np.any(np.isnan(X1)) or np.any(np.isnan(X2)) or np.any(np.isnan(C1)) or np.any(np.isnan(C2)):
                    raise ValueError("NaN values found in feature matrices")

                # Log dimensions and statistics
                logger.info(f"\nData dimensions:")
                logger.info(f"  N (preferences): {len(y)}")
                logger.info(f"  P (participants): {len(participant_ids)}")
                if is_control_only:
                    logger.info("  Using dummy main feature for control-only model")
                logger.info(f"  F (main features): {1 if is_control_only else len(main_features)}")
                logger.info(f"  C (control features): {len(control_features)}")

                # Log original and standardized statistics
                logger.info("\nFeature statistics:")
                if not is_control_only:
                    for i, feature in enumerate(main_features):
                        logger.info(f"\n{feature} (main):")
                        if feature in feature_stats:
                            logger.info(f"  Original - mean: {feature_stats[feature]['mean']:.3f}, "
                                    f"std: {feature_stats[feature]['std']:.3f}")
                        logger.info(f"  Standardized - mean: {np.mean(X1[:, i]):.3f}, "
                                f"std: {np.std(X1[:, i]):.3f}")

                for i, feature in enumerate(control_features):
                    logger.info(f"\n{feature} (control):")
                    logger.info(f"  Original - mean: {feature_stats[feature]['mean']:.3f}, "
                            f"std: {feature_stats[feature]['std']:.3f}")
                    logger.info(f"  Standardized - mean: {np.mean(C1[:, i]):.3f}, "
                            f"std: {np.std(C1[:, i]):.3f}")

                return {
                    'N': len(y),
                    'P': len(participant_ids),
                    'F': 1 if is_control_only else len(main_features),
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
            
            # Add interaction features
            for interaction in self.config.features.interactions:
                # Get name for this interaction (sorted for consistency)
                interaction_name = '_x_'.join(sorted(interaction))
                
                # Compute interaction value by multiplying component features
                interaction_value = 1.0
                for component in interaction:
                    if component not in features:
                        logger.warning(f"Component feature '{component}' missing for interaction in bigram '{bigram}'")
                        interaction_value = 0.0
                        break
                    interaction_value *= features[component]
                    
                # Add interaction to features
                features[interaction_name] = interaction_value
            
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

    def calculate_storage_requirements(self, dataset: PreferenceDataset, all_features: List[str]) -> int:
        """
        Calculate approximate storage requirements for the entire feature selection process.
        
        Returns:
            Required storage in MB
        """
        bytes_per_param = 8  # Double precision float
        n_participants = len(dataset.participants)
        n_preferences = len(dataset.preferences)
        n_features = len(all_features)
        n_chains = self.config.model.chains
        n_samples = self.config.model.warmup + self.config.model.n_samples
        
        logger.info(f"Storage calculation details:")
        logger.info(f"  Participants: {n_participants}")
        logger.info(f"  Preferences: {n_preferences}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Chains: {n_chains}")
        logger.info(f"  Total samples: {n_samples}")
        
        # Storage per model fit
        params_per_chain = (n_participants * n_features) + n_preferences + n_features
        chain_storage = params_per_chain * bytes_per_param * n_chains * n_samples
        
        logger.info(f"  Parameters per chain: {params_per_chain}")
        logger.info(f"  Storage per full model fit: {chain_storage / (1024 * 1024):.1f} MB")
        
        # Number of model fits needed
        total_fits = 1 + n_features + (5 * n_features)
        logger.info(f"  Total model fits needed: {total_fits}")
        logger.info(f"    - Initial baseline: 1")
        logger.info(f"    - Feature evaluations: {n_features}")
        logger.info(f"    - Cross-validation fits: {5 * n_features}")
        
        # Total storage including overhead for temp files
        total_bytes = chain_storage * total_fits * 1.5  # 50% overhead
        total_mb = int(total_bytes / (1024 * 1024))
        
        logger.info(f"  Total storage needed (with 50% overhead): {total_mb} MB")
        
        return total_mb

    def check_total_storage(self, dataset: PreferenceDataset, all_features: List[str]) -> None:
        """
        Check if enough storage is available for the entire feature selection process.
        Raises RuntimeError if insufficient space.
        """
        required_mb = self.calculate_storage_requirements(dataset, all_features)
        
        # Check temp directory space
        temp_dir = Path(tempfile.gettempdir())
        total, used, free = shutil.disk_usage(temp_dir)
        free_mb = free // (1024 * 1024)
        
        logger.info(f"Storage requirements:")
        logger.info(f"  Required space: {required_mb} MB")
        logger.info(f"  Available space: {free_mb} MB")
        
        if free_mb < required_mb:
            raise RuntimeError(
                f"Insufficient storage space for feature selection.\n"
                f"Required: {required_mb} MB\n"
                f"Available: {free_mb} MB\n"
                f"Consider:\n"
                f"- Freeing up temporary space\n"
                f"- Reducing number of chains ({self.config.model.chains})\n"
                f"- Reducing number of samples ({self.config.model.n_samples})\n"
                f"- Reducing number of features to evaluate ({len(all_features)})"
            )
                                                
    #--------------------------------------------
    # Feature selection and evaluation methods
    #--------------------------------------------  
    def select_features(self, dataset: PreferenceDataset, all_features: List[str]) -> List[str]:
        """Select features using round-robin comparison with threshold elimination."""
        self.dataset = dataset  # Set dataset first
        
        # Preprocess dataset to handle NaN values consistently
        logger.info("Preprocessing dataset for feature selection...")
        valid_prefs = []
        for pref in dataset.preferences:
            # Check all main features (not just typing_time) for NaN values
            valid = True
            for feature in all_features:
                if (pref.features1.get(feature) is None or 
                    pref.features2.get(feature) is None):
                    valid = False
                    break
            if valid:
                valid_prefs.append(pref)
        
        # Create preprocessed dataset
        processed_dataset = PreferenceDataset.__new__(PreferenceDataset)
        processed_dataset.preferences = valid_prefs
        processed_dataset.participants = {p.participant_id for p in valid_prefs}
        processed_dataset.file_path = dataset.file_path
        processed_dataset.config = dataset.config
        processed_dataset.control_features = dataset.control_features
        processed_dataset.feature_extractor = dataset.feature_extractor
        processed_dataset.feature_names = dataset.feature_names
        processed_dataset.all_bigrams = dataset.all_bigrams
        processed_dataset.all_bigram_features = dataset.all_bigram_features
        
        # Copy the maps to make cleaner logs and 
        # prevent potential issues if these attributes are referenced elsewhere
        processed_dataset.column_map = getattr(dataset, 'column_map', None)
        processed_dataset.row_map = getattr(dataset, 'row_map', None)
        processed_dataset.finger_map = getattr(dataset, 'finger_map', None)
        processed_dataset.engram_position_values = getattr(dataset, 'engram_position_values', None)
        processed_dataset.row_position_values = getattr(dataset, 'row_position_values', None)

        logger.info(f"Preprocessed dataset:")
        logger.info(f"  Original size: {len(dataset.preferences)} preferences")
        logger.info(f"  After filtering: {len(processed_dataset.preferences)} preferences")
        logger.info(f"  Participants: {len(processed_dataset.participants)}")
        
        # Initialize selection and tracking
        self.selected_features = []
        selected_so_far = []  # Track selected features separately
        control_features = self.config.features.control_features
        base_features = [f for f in all_features if f not in control_features and '_x_' not in f]
        interaction_features = [f for f in all_features if f not in control_features and '_x_' in f]

        max_rounds = len(base_features) + len(interaction_features)
        thresholds = self.config.feature_selection.thresholds
        min_metrics_passed = self.config.feature_selection.min_metrics_passed

        # First evaluate control features
        for feature in control_features:
            metrics = self.importance_calculator.evaluate_feature(
                feature=feature,
                dataset=processed_dataset,
                model=self,
                all_features=all_features,
                current_selected_features=selected_so_far  # Use tracking list consistently
            )
            if sum([
                metrics['model_effect'] >= thresholds.model_effect,
                metrics['effect_consistency'] >= thresholds.effect_consistency,
                metrics['predictive_power'] >= thresholds.predictive_power
            ]) >= min_metrics_passed:
                self.selected_features.append(feature)
                selected_so_far.append(feature)  # Update tracking list



        for round_num in range(1, max_rounds + 1):
            remaining_features = base_features if base_features else interaction_features
            if not remaining_features:
                break
                
            # Evaluate all remaining features
            feature_metrics = {}
            for feature in remaining_features:
                metrics = self.importance_calculator.evaluate_feature(
                    feature=feature,
                    dataset=processed_dataset,
                    model=self,
                    all_features=all_features,
                    current_selected_features=self.selected_features
                )

                passes = [
                    metrics['model_effect'] >= thresholds.model_effect,
                    metrics['effect_consistency'] >= thresholds.effect_consistency,
                    metrics['predictive_power'] >= thresholds.predictive_power
                ]
                metrics_passed = sum(passes)

                logger.info(f"\nFeature {feature} metrics:")
                logger.info(f"  Model effect: {metrics['model_effect']:.3f} (threshold: {thresholds.model_effect})")
                logger.info(f"  Effect consistency: {metrics['effect_consistency']:.3f} (threshold: {thresholds.effect_consistency})")
                logger.info(f"  Predictive power: {metrics['predictive_power']:.3f} (threshold: {thresholds.predictive_power})")
                logger.info(f"  Metrics passed: {metrics_passed}/{min_metrics_passed}")

                if metrics_passed >= min_metrics_passed:
                    feature_metrics[feature] = metrics
                    
            # Remove features that didn't meet thresholds
            if base_features:
                base_features = list(feature_metrics.keys())
            else:
                interaction_features = list(feature_metrics.keys())

            # Select best remaining feature (single scoring block)
            if feature_metrics:
                weights = self.config.feature_selection.metric_weights
                best_score = float('-inf')
                best_feature = None
                
                for feature, metrics in feature_metrics.items():
                    score = (
                        metrics['model_effect'] * weights['model_effect'] +
                        metrics['effect_consistency'] * weights['effect_consistency'] + 
                        metrics['predictive_power'] * weights['predictive_power']
                    )
                    logger.info(f"Feature {feature} score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                
                if best_feature is not None:
                    logger.info(f"Selected feature {best_feature} with score {best_score:.4f}")
                    self.selected_features.append(best_feature)
                    selected_so_far.append(best_feature)  # Update tracking list
                    
                    # Remove selected feature
                    if best_feature in base_features:
                        base_features.remove(best_feature)
                    if best_feature in interaction_features:
                        interaction_features.remove(best_feature)

        return self.selected_features

    def _select_best_feature(self, feature_metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Select best feature based on metrics."""
        if not feature_metrics:
            return None
            
        thresholds = self.config.feature_selection.thresholds
        min_metrics = self.config.feature_selection.min_metrics_passed
        
        # Count wins for each feature
        win_counts = {f: 0 for f in feature_metrics}
        for f1, f2 in combinations(feature_metrics.keys(), 2):
            if self._is_feature_better(feature_metrics[f1], feature_metrics[f2]):
                win_counts[f1] += 1
            else:
                win_counts[f2] += 1
        
        # Select feature with most wins that meets thresholds
        best_feature = max(win_counts.items(), key=lambda x: x[1])[0]
        metrics = feature_metrics[best_feature]
        
        # Print metrics for best feature
        print(f"\nBest feature metrics ({best_feature}):")
        print(f"  Model effect: {metrics['model_effect']:.3f} (threshold: {thresholds.model_effect})")
        print(f"  Effect consistency: {metrics['effect_consistency']:.3f} (threshold: {thresholds.effect_consistency})")
        print(f"  Predictive power: {metrics['predictive_power']:.3f} (threshold: {thresholds.predictive_power})")
        
        metrics_passed = (
            (metrics['model_effect'] >= thresholds.model_effect) +
            (metrics['effect_consistency'] >= thresholds.effect_consistency) +
            (metrics['predictive_power'] >= thresholds.predictive_power)
        )
        
        if metrics_passed >= min_metrics:
            return best_feature
        return None

    def _get_evaluation_context(self, feature: str, current_selected: List[str], 
                            control_features: List[str]) -> Optional[List[str]]:
        """Get evaluation context for a feature."""
        if '_x_' in feature:
            components = feature.split('_x_')
            if not all(c in current_selected for c in components):
                return None
        
        # Build context with no duplicates
        context = list(dict.fromkeys(
            control_features +  # Control features first
            current_selected +  # Currently selected features
            [feature]          # Feature being evaluated
        ))
        
        return context
                                                                                                
    def _is_feature_better(self, metrics_a: Dict[str, float], metrics_b: Dict[str, float]) -> bool:
        """
        Compare two features based on their metrics and thresholds.
        Returns True if feature A is better than feature B.
        """
        thresholds = self.config.feature_selection.thresholds
        min_metrics = self.config.feature_selection.min_metrics_passed
        
        # Count passed metrics for each feature
        metrics_passed_a = (
            (metrics_a['model_effect'] >= thresholds.model_effect) +
            (metrics_a['effect_consistency'] >= thresholds.effect_consistency) +
            (metrics_a['predictive_power'] >= thresholds.predictive_power)
        )
        
        metrics_passed_b = (
            (metrics_b['model_effect'] >= thresholds.model_effect) +
            (metrics_b['effect_consistency'] >= thresholds.effect_consistency) +
            (metrics_b['predictive_power'] >= thresholds.predictive_power)
        )
        
        # If one feature meets minimum metrics and other doesn't, choose the one that does
        if metrics_passed_a >= min_metrics and metrics_passed_b < min_metrics:
            return True
        if metrics_passed_b >= min_metrics and metrics_passed_a < min_metrics:
            return False
        
        # If neither or both meet minimum metrics, compare their scores
        score_a = metrics_a['model_effect'] + metrics_a['effect_consistency'] + metrics_a['predictive_power']
        score_b = metrics_b['model_effect'] + metrics_b['effect_consistency'] + metrics_b['predictive_power']
        
        return score_a > score_b
                                                                                                                                                       
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
        """Predict preference probability and uncertainty for a bigram pair."""
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
                
            start_time = time.perf_counter()
            
            # Log model state
            logger.debug(f"All features available: {self.feature_names if hasattr(self, 'feature_names') else 'No feature_names'}")
            logger.debug(f"Feature weights: {self.feature_weights if hasattr(self, 'feature_weights') else 'No feature_weights'}")
            logger.debug(f"Selected features: {self.selected_features}")
            logger.debug(f"Control features: {self.config.features.control_features}")
            
            # Separate main and control feature effects
            main_features = list(dict.fromkeys([f for f in self.selected_features 
                                    if f not in self.config.features.control_features]))
            control_features = list(dict.fromkeys(self.config.features.control_features))

            # Determine if this is a baseline evaluation
            is_baseline = getattr(self, 'is_baseline_model', False)
            logger.debug(f"Model type: {'baseline' if is_baseline else 'main'}")

            # For non-baseline models, require main features
            main_features = [f for f in self.selected_features 
                            if f not in self.config.features.control_features]
            control_features = self.config.features.control_features

            # For non-baseline models, require main features  
            if not main_features and not is_baseline:
                logger.error("Prediction failed due to missing main features:")
                logger.error(f"  All features available: {self.feature_names}")
                logger.error(f"  Selected features: {self.selected_features}")
                logger.error(f"  Control features: {control_features}")
                logger.error(f"  Feature weights: {self.feature_weights}")
                raise ValueError("No main features available for non-baseline prediction")
                
            # Extract base features
            features1 = self.feature_extractor.extract_bigram_features(bigram1[0], bigram1[1])
            features2 = self.feature_extractor.extract_bigram_features(bigram2[0], bigram2[1])
            
            # Add interaction features
            for interaction in self.config.features.interactions:
                # Skip if any component is missing
                if any(f not in features1 for f in interaction):
                    logger.warning(f"Missing base features for interaction: {interaction}")
                    continue
                    
                # Create interaction name by joining all components
                interaction_name = '_x_'.join(sorted(interaction))
                
                if interaction_name in self.selected_features:
                    # Compute product of all component features
                    features1[interaction_name] = np.prod([features1[f] for f in interaction])
                    features2[interaction_name] = np.prod([features2[f] for f in interaction])
                
            logger.debug(f"Available features in bigram1: {list(features1.keys())}")
            logger.debug(f"Available features in bigram2: {list(features2.keys())}")
            
            # Calculate predictions
            total_effect = 0.0
            uncertainty = 0.0
            
            if main_features:
                logger.debug(f"Computing main feature effects for: {main_features}")
                beta_samples = self.fit_result.stan_variable('beta')
                main_diffs = np.array([features1[f] - features2[f] for f in main_features])
                main_effects = np.dot(main_diffs, beta_samples.T)
                total_effect += np.mean(main_effects)
                uncertainty += np.std(main_effects)
                
            if control_features:
                logger.debug(f"Computing control feature effects for: {control_features}")
                gamma_samples = self.fit_result.stan_variable('gamma')
                control_diffs = np.array([features1[f] - features2[f] for f in control_features])
                control_effects = np.dot(control_diffs, gamma_samples.T)
                total_effect += np.mean(control_effects)
                uncertainty += np.std(control_effects)
                
            # Transform to probability
            probability = 1 / (1 + np.exp(-total_effect))
            
            logger.debug(f"Prediction results:")
            logger.debug(f"  Total effect: {total_effect:.4f}")
            logger.debug(f"  Uncertainty: {uncertainty:.4f}")
            logger.debug(f"  Probability: {probability:.4f}")
            
            prediction = ModelPrediction(
                probability=float(probability),
                uncertainty=float(uncertainty),
                features_used=self.selected_features,
                computation_time=time.perf_counter() - start_time
            )
            # Assertion to catch invalid returns
            assert isinstance(prediction, ModelPrediction), f"Invalid prediction type: {type(prediction)}"

            self.prediction_cache.set(cache_key, prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting preference: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")  # Add this to see full stack trace
            return ModelPrediction(
                probability=0.5,
                uncertainty=1.0,
                features_used=[],
                computation_time=0.0
            )
                
    def _check_diagnostics(self) -> None:
            """Check MCMC diagnostics with concise reporting."""
            try:
                # Get summary statistics
                summary = self.fit_result.summary()
                
                # Check R-hat convergence
                rhat_col = next((col for col in summary.columns 
                            if any(x in col.lower() for x in ['r_hat', 'rhat', 'r-hat'])), 
                            None)
                
                if rhat_col:
                    rhat = summary[rhat_col].astype(float)
                    high_rhat_mask = rhat > 1.1
                    if high_rhat_mask.any():
                        n_high_rhat = high_rhat_mask.sum()
                        n_total = len(rhat)
                        logger.warning(f"{n_high_rhat}/{n_total} parameters have high R-hat (>1.1)")

                # Check treedepth and divergences
                if hasattr(self.fit_result, 'method_variables'):
                    method_vars = self.fit_result.method_variables()
                    
                    # Check maximum treedepth
                    treedepth = method_vars.get('treedepth__', None)
                    if treedepth is not None:
                        max_treedepth = self.config.model.max_treedepth
                        max_treedepth_hits = (treedepth >= max_treedepth).mean()
                        if max_treedepth_hits > 0.1:  # More than 10% hitting max
                            logger.warning(f"{max_treedepth_hits*100:.1f}% of iterations hit max treedepth")
                    
                    # Check divergences
                    divergent = method_vars.get('divergent__', None)
                    if divergent is not None:
                        n_divergent = divergent.sum()
                        if n_divergent > 0:
                            divergence_rate = n_divergent / len(divergent) * 100
                            logger.warning(f"{divergence_rate:.1f}% divergent transitions")

                # If there are any issues, run full diagnostics but suppress output
                if hasattr(self.fit_result, 'diagnose'):
                    diagnostic_info = self.fit_result.diagnose()
                    # Store diagnostic info but don't display it
                    self._diagnostic_info = diagnostic_info
                            
            except Exception as e:
                logger.warning(f"Error in diagnostics: {str(e)}")
                                                                                        
    #--------------------------------------------
    # Cross-validation and splitting methods
    #--------------------------------------------
    def _get_cv_splits(self, dataset: PreferenceDataset, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits preserving participant structure with validation."""
        from sklearn.model_selection import KFold
        
        # Check for dataset size consistency
        if not hasattr(self, '_dataset_size'):
            self._dataset_size = len(dataset.preferences)
            self._dataset_participants = len(dataset.participants)
        else:
            if len(dataset.preferences) != self._dataset_size:
                raise ValueError(f"Dataset size changed: {len(dataset.preferences)} vs {self._dataset_size}")
            if len(dataset.participants) != self._dataset_participants:
                raise ValueError(f"Participant count changed: {len(dataset.participants)} vs {self._dataset_participants}")
        
        # Get unique participants and their preferences
        participants = list(dataset.participants)
        participant_to_indices = {}
        for i, pref in enumerate(dataset.preferences):
            if pref.participant_id not in participant_to_indices:
                participant_to_indices[pref.participant_id] = []
            participant_to_indices[pref.participant_id].append(i)
        
        # Create participant splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        total_preferences = len(dataset.preferences)
        used_indices = set()
        
        for train_part_idx, val_part_idx in kf.split(participants):
            # Get participant IDs for each split
            train_participants = {participants[i] for i in train_part_idx}
            val_participants = {participants[i] for i in val_part_idx}
            
            # Verify participant split
            if train_participants & val_participants:
                raise ValueError("Participant overlap detected in CV splits")
            
            # Get preference indices for each split
            train_indices = []
            val_indices = []
            
            for participant, indices in participant_to_indices.items():
                if participant in train_participants:
                    train_indices.extend(indices)
                elif participant in val_participants:
                    val_indices.extend(indices)
            
            # Verify preference split
            if set(train_indices) & set(val_indices):
                raise ValueError("Preference index overlap detected in CV splits")
            
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            
            # Update used indices tracking
            used_indices.update(train_indices)
            used_indices.update(val_indices)
            
            splits.append((train_indices, val_indices))
        
        # Verify all preferences are used
        if len(used_indices) != total_preferences:
            raise ValueError(f"Not all preferences used in CV splits: {len(used_indices)} vs {total_preferences}")
            
        return splits
                            
    #--------------------------------------------
    # Output and visualization methods
    #--------------------------------------------
    def save(self, path: Path) -> None:
        """Save model state to file."""
        logger.info("=== Saving model state ===")
        logger.info(f"Selected features: {self.selected_features}")
        logger.info(f"Feature names: {self.feature_names}")
        logger.info(f"Feature weights: {self.feature_weights}")
        
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
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'PreferenceModel':
        """Load model state from file."""
        logger.info("=== Loading model state ===")
        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)
            
            model = cls(config=save_dict['config'])
            try:
                model.feature_names = save_dict['feature_names']
                model.selected_features = save_dict['selected_features']
                model.feature_weights = save_dict['feature_weights']
                model.fit_result = save_dict['fit_result']
                model.interaction_metadata = save_dict['interaction_metadata']
                return model
            except Exception:
                model.cleanup()
                raise
        except Exception:
            logger.error(f"Error loading model from {path}")
            raise

    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            # Clean Stan temp directory if it exists in config
            if hasattr(self.config.paths, 'stan_temp'):
                output_dir = Path(self.config.paths.stan_temp)
                if output_dir.exists():
                    for file in output_dir.glob("preference_model-*"):
                        try:
                            file.unlink()
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {file}: {str(e)}")
            # Clear caches
            self.clear_caches()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Suppress errors during garbage collection

