# model.py
"""
Bayesian preference learning model for keyboard layout evaluation.
Core functionality:
  - Feature-based preference modeling using Bradley-Terry framework
  - Hierarchical model with participant-level random effects
  - MCMC sampling via Stan backend
  - Feature selection pipeline

Key components:
  1. Model Structure:
    - Feature-based comfort scoring
    - Participant-specific random effects
    - Hierarchical Bayesian inference
    - Control vs. main feature handling

  2. Feature Selection:
    - Cross-validated importance scoring
    - Effect magnitude and consistency metrics
    - Participant-aware validation splits
    - Control feature separation

  3. Prediction Pipeline:
    - Preference probability estimation
    - Uncertainty quantification
    - Comfort score calculation
    - Feature interaction handling

  4. Resource Management:
    - Stan model compilation and cleanup
    - Memory usage optimization
    - Temporary file handling
    - Feature data caching

  5. Model Operations:
    - Data preprocessing and normalization
    - MCMC sampling with diagnostics
    - Cross-validation
    - Model serialization

Classes:
    PreferenceModel: Main class implementing the preference learning pipeline
        Methods:
            fit(): Train model on preference data
            predict_preference(): Generate predictions for bigram pairs
            evaluate(): Compute model performance metrics
            select_features(): Perform feature selection
            save()/load(): Model serialization

Dependencies:
    - cmdstanpy: Stan model interface
    - numpy: Numerical operations
    - sklearn: Evaluation metrics
    - pandas: Data management
    - psutil: Resource monitoring
"""
import cmdstanpy
import numpy as np
from sklearn.metrics import roc_auc_score
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
import os
import shutil
import psutil
from tenacity import retry, stop_after_attempt, wait_fixed
import gc

from engram3.utils.config import Config, NotFittedError, FeatureError, ModelPrediction
from engram3.data import PreferenceDataset
from engram3.utils.visualization import PlottingUtils
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
        
        self._is_fitted = False  # Private attribute to store the fitted state
        self.config = config if isinstance(config, Config) else Config(**config)
        self.reset_state()
        
        # Initialize visualization
        self.plotting = PlottingUtils(self.config.paths.plots_dir)

        # Initialize Stan model
        try:
            model_path = Path(__file__).parent / "models" / "preference_model.stan"
            if not model_path.exists():
                raise FileNotFoundError(f"Stan model file not found: {model_path}")
                
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
            logger.error(f"Error initializing Stan model: {str(e)}")
            raise

    def reset_state(self):
        """Reset all model state variables to initial values."""
        # Core model state
        self.fit_result = None 
        self.is_fitted = False
        self.feature_names = None
        self.selected_features = []
        
        # Feature related state
        self.feature_weights = None
        self.feature_stats = {}
        self.interaction_metadata = {}
        
        # Dataset related state
        self.dataset = None
        self.feature_extractor = None
        
        # Cache related state
        if hasattr(self, 'processed_dataset'):
            delattr(self, 'processed_dataset')
        if hasattr(self, '_feature_data_cache'):
            self._feature_data_cache.clear()
        if hasattr(self, '_temp_models'):
            self._temp_models = []
            
        # Computation state
        self.is_baseline_model = False
        self._importance_calculator = None
        
        # Clear any other temporary attributes
        attrs_to_clear = [
            '_dataset_size',
            '_dataset_participants',
            'processed_dataset',
            '_feature_importance_cache'
        ]
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
                
        # Force garbage collection
        gc.collect()
        
    def fit(self, dataset: PreferenceDataset, features: Optional[List[str]] = None,
        fit_purpose: Optional[str] = None) -> None:
        """Fit model with specified features."""
        try:
            self._check_memory_usage()  # Check before fitting

            # Reset state and store feature names
            self.reset_state()
            self.dataset = dataset  # Store dataset
            self.processed_dataset = dataset  
            self.feature_names = features
            
            if fit_purpose:
                logger.info(f"Fitting model: {fit_purpose}")

            # Store dataset and feature extractor
            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor
            
            # Validate and process features
            if not features:
                features = dataset.get_feature_names()
            if not features:
                raise ValueError("No features provided and none available from dataset")
            
            # Process features
            control_features = self.config.features.control_features
            main_features = [f for f in features if f not in control_features]
            
            if main_features:
                logger.info("Features:")
                for feat in main_features:
                    logger.info(f"  - {feat} (main)")
            if control_features:
                for feat in control_features:
                    logger.info(f"  - {feat} (control)")

            # Prepare feature matrices for Stan
            processed_data = self._prepare_feature_matrices(
                dataset=dataset,
                main_features=main_features,
                control_features=control_features
            )

            # Log key dimensions without overwhelming output
            logger.debug(f"Data dims: N={processed_data['N']}, P={processed_data['P']}, " 
                        f"F={processed_data['F']}, C={processed_data['C']}")

            # Store feature information
            self.feature_names = main_features + list(control_features)
            self.selected_features = main_features + list(control_features)

            # Stan sampling
            self.fit_result = self._sample_with_retry(
                data=processed_data,
                chains=self.config.model.chains,
                iter_warmup=self.config.model.warmup,
                iter_sampling=self.config.model.n_samples,
                adapt_delta=self.config.model.adapt_delta,
                max_treedepth=self.config.model.max_treedepth,
                refresh=None
            )

            # Check diagnostics but only log if issues found
            if hasattr(self.fit_result, 'diagnostic_summary'):
                diagnostics = self.fit_result.diagnostic_summary()
                if any(d > 0 for d in diagnostics.values()):
                    logger.warning(f"Sampling diagnostics: {diagnostics}")

            # Update weights and set fitted flag only after successful sampling
            self._update_feature_weights()
            self.is_fitted = True

        except Exception as e:
            self.is_fitted = False  # Ensure flag is False if fit fails
            logger.error(f"Error in fit: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise
        
        finally:
            #self.cleanup()  # Clean up after fitting
            self._check_memory_usage()
                    
    def evaluate(self, dataset: PreferenceDataset) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if not hasattr(self, 'fit_result'):
                raise ValueError("Model not fitted")
            
            # Get predictions
            predictions = []
            true_labels = []
            
            for pref in dataset.preferences:
                # Get prediction
                pred = self.predict_preference(pref.bigram1, pref.bigram2)
                
                # Store logit for AUC calculation
                logit = -np.log(1/pred.probability - 1)
                predictions.append(logit)
                true_labels.append(float(pref.preferred))
                
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)
            
            # Compute metrics
            accuracy = np.mean((predictions > 0) == true_labels)  # Use 0 as threshold for logits
            auc = roc_auc_score(true_labels, predictions)
            
            logger.info(f"Prediction distribution:")
            logger.info(f"  Mean: {np.mean(predictions):.4f}")
            logger.info(f"  Std: {np.std(predictions):.4f}")
            logger.info(f"  Min: {np.min(predictions):.4f}")
            logger.info(f"  Max: {np.max(predictions):.4f}")
            
            return {
                'accuracy': float(accuracy),
                'auc': float(auc),
                'mean_pred': float(np.mean(predictions)),
                'std_pred': float(np.std(predictions))
            }
        
        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            return {'accuracy': 0.5, 'auc': 0.5, 'mean_pred': 0.0, 'std_pred': 1.0}

    def predict_preference(self, bigram1: str, bigram2: str) -> ModelPrediction:
        """Predict preference between two bigrams."""
        if not self.is_fitted:
            raise NotFittedError("Model must be fit before making predictions")
            
        try:
            # Get features for both bigrams
            features1 = self.feature_extractor.extract_bigram_features(bigram1[0], bigram1[1])
            features2 = self.feature_extractor.extract_bigram_features(bigram2[0], bigram2[1])

            # Initialize feature matrices
            X1, X2 = [], []  # Main features
            C1, C2 = [], []  # Control features
            
            # Split features into main and control
            control_features = self.config.features.control_features
            main_features = [f for f in self.feature_names if f not in control_features]
            
            # Process main features
            if not main_features:
                X1 = []  # Empty list for no main features
                X2 = []
            else:
                for feature in main_features:
                    feat1 = features1.get(feature, 0.0)
                    feat2 = features2.get(feature, 0.0)

                    if feature == 'typing_time' and (feat1 is None or feat2 is None):
                        feat1 = feat2 = 0.0
                        
                    # Use feature stats if available, otherwise use raw values
                    if hasattr(self, 'feature_stats') and feature in self.feature_stats:
                        mean = self.feature_stats[feature]['mean']
                        std = self.feature_stats[feature]['std']
                        feat1 = (feat1 - mean) / std
                        feat2 = (feat2 - mean) / std
                        
                    X1.append(feat1)
                    X2.append(feat2)
                
            # Process control features
            for feature in control_features:
                feat1 = features1.get(feature, 0.0)
                feat2 = features2.get(feature, 0.0)
                
                # Use feature stats if available, otherwise use raw values
                if hasattr(self, 'feature_stats') and feature in self.feature_stats:
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    feat1 = (feat1 - mean) / std
                    feat2 = (feat2 - mean) / std
                    
                C1.append(feat1)
                C2.append(feat2)

            # Convert to arrays with error checking
            try:
                X1 = np.array(X1, dtype=np.float64).reshape(1, -1)
                X2 = np.array(X2, dtype=np.float64).reshape(1, -1)
                C1 = np.array(C1, dtype=np.float64).reshape(1, -1)
                C2 = np.array(C2, dtype=np.float64).reshape(1, -1)
            except Exception as e:
                logger.error(f"Error converting feature arrays: {str(e)}")
                raise ValueError(f"Feature conversion failed: {str(e)}")

            # Verify array shapes
            if X1.shape[1] != len(main_features):
                raise ValueError(f"Main feature shape mismatch: {X1.shape[1]} != {len(main_features)}")
            if C1.shape[1] != len(control_features):
                raise ValueError(f"Control feature shape mismatch: {C1.shape[1]} != {len(control_features)}")

            try:
                # Get model predictions
                y_pred = self.fit_result.stan_variable('y_pred')
                probability = 1 / (1 + np.exp(-y_pred))
                uncertainty = np.std(y_pred) if isinstance(y_pred, np.ndarray) else 0.0

                # Ensure valid probability
                probability = np.clip(probability, 0.001, 0.999)
                
                return ModelPrediction(
                    probability=float(np.mean(probability)),
                    uncertainty=float(uncertainty),
                    features_used=main_features + list(control_features),
                    computation_time=0.0
                )

            except Exception as e:
                logger.error(f"Error in prediction calculation: {str(e)}")
                raise ValueError(f"Prediction calculation failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error in predict_preference: {str(e)}")
            # Return balanced prediction on error
            return ModelPrediction(
                probability=0.5,
                uncertainty=1.0,
                features_used=[],
                computation_time=0.0
            )
                            
    #--------------------------------------------
    # Resource management methods
    #--------------------------------------------
    def cleanup(self):
        """Clean up model resources."""
        # Don't clear fit_result if model is fitted
        if not self.is_fitted:
            if hasattr(self, 'fit_result'):
                self._cleanup_run_directory(active_only=True)
                del self.fit_result
        
        if hasattr(self, '_feature_data_cache'):
            self._feature_data_cache.clear()
            
        # Clean up temp models
        self.cleanup_temp_models()
        # Don't reset state if model is fitted
        if not self.is_fitted:
            self.reset_state()
        gc.collect()
        
    def cleanup_temp_models(self):
        """Clean up temporary models created during evaluation."""
        if hasattr(self, '_temp_models'):
            models_to_remove = []
            for model in self._temp_models:
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                models_to_remove.append(model)
                
            for model in models_to_remove:
                if model in self._temp_models:
                    self._temp_models.remove(model)
            
            # Clear list after cleanup
            self._temp_models = []

    def _cleanup_run_directory(self, run_dir: Optional[Path] = None, active_only: bool = False) -> None:
        """Cleanup only Stan run directories that we own.
        
        Args:
            run_dir: Optional specific run directory to clean
            active_only: If True, only clean active run directory
        """
        try:
            # Get active run directory if it exists
            active_run_dir = None
            if hasattr(self, 'fit_result') and hasattr(self.fit_result, '_run_dir'):
                active_run_dir = Path(self.fit_result._run_dir)
                
            # If cleaning specific directory
            if run_dir is not None:
                if active_only and run_dir == active_run_dir:
                    return
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                return
                
            # Clean up engram temp folder
            engram_temp = Path.home() / '.engram_temp'
            if engram_temp.exists():
                cleaned_space = 0
                for item in engram_temp.iterdir():
                    try:
                        # Skip active run directory
                        if active_only and active_run_dir and item == active_run_dir:
                            continue
                            
                        # Only clean run_* directories
                        if item.is_dir() and item.name.startswith('run_'):
                            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                            shutil.rmtree(item)
                            cleaned_space += size
                    except Exception as e:
                        logger.debug(f"Could not remove {item}: {e}")
                        
                if cleaned_space > 0:
                    logger.info(f"Cleaned up {cleaned_space / (1024*1024):.1f}MB from temp directories")
                        
        except Exception as e:
            logger.error(f"Error during temp cleanup: {e}")

    def _create_temp_model(self):
        """Create a temporary model copy for evaluation."""
        temp_model = type(self)(config=self.config)
        temp_model.feature_extractor = self.feature_extractor
        temp_model.feature_stats = self.feature_stats.copy()  # Make a copy of stats

        # Store model reference in parent
        if not hasattr(self, '_temp_models'):
            self._temp_models = []
        self._temp_models.append(temp_model)

        class ModelContext:
            def __init__(self, model, parent_models):
                self.model = model
                self._parent_models = parent_models
                
            def __enter__(self):
                return self.model
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # On context exit, mark this model as ready for cleanup
                if hasattr(self.model, 'ready_for_cleanup'):
                    self.model.ready_for_cleanup = True

        return ModelContext(temp_model, self._temp_models)

    def _add_to_cache(self, cache: Dict, key: str, value: Any) -> None:
        """Add item to cache with size management."""
        # Set maximum cache size (adjust as needed)
        MAX_CACHE_SIZE = 1000  
        
        if len(cache) >= MAX_CACHE_SIZE:
            # Remove oldest 20% of entries when cache is full
            remove_count = MAX_CACHE_SIZE // 5
            for k in list(cache.keys())[:remove_count]:
                del cache[k]
                
        cache[key] = value

    def _check_memory_usage(self):
        """Monitor memory usage and clean up if needed."""
        import psutil
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        # If using more than 80% memory, force cleanup
        if memory_percent > 80:
            logger.warning(f"High memory usage ({memory_percent:.1f}%). Cleaning up...")
            self.cleanup()

    def check_disk_space(self, required_mb: int = 2000) -> bool:
        """
        Check if there's enough disk space available.
        
        Args:
            required_mb: Required space in megabytes
        Returns:
            bool: True if enough space available
        """
        try:
            # Check temp directory space
            temp_dir = tempfile.gettempdir()
            stats = shutil.disk_usage(temp_dir)
            available_mb = stats.free / (1024 * 1024)  # Convert to MB
            
            return available_mb >= required_mb
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False
        
    #--------------------------------------------
    # Feature handling methods
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
            
            return features.copy()
            
        except (NotFittedError, ValueError, FeatureError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error extracting features for bigram '{bigram}': {str(e)}")
            raise FeatureError(f"Feature extraction failed: {str(e)}")
        
    def get_feature_weights(self, include_control: bool = True) -> Dict[str, Tuple[float, float]]:
        """Get feature weights and their uncertainties."""

        if not hasattr(self, 'fit_result') or self.fit_result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not hasattr(self.fit_result, 'stan_variable'):
            raise ValueError("Invalid Stan result - missing stan_variable method")
                
        try:
            weights = {}
            
            # Get main feature weights (beta)
            main_features = [f for f in self.feature_names 
                            if f not in self.config.features.control_features]
            
            if main_features:  # Only process beta if we have main features
                beta = self.fit_result.stan_variable('beta')
                
                if beta.shape[1] != len(main_features):
                    raise ValueError(f"Beta shape mismatch: {beta.shape[1]} != {len(main_features)}")
                
                for i, feature in enumerate(main_features):
                    weights[feature] = (
                        float(np.mean(beta[:, i])),
                        float(np.std(beta[:, i]))
                    )
            
            # Get control feature weights (gamma) if requested
            if include_control:
                control_features = self.config.features.control_features
                if control_features:
                    gamma = self.fit_result.stan_variable('gamma')
                    
                    if gamma.shape[1] != len(control_features):
                        raise ValueError(f"Gamma shape mismatch: {gamma.shape[1]} != {len(control_features)}")
                    
                    for i, feature in enumerate(control_features):
                        weights[feature] = (
                            float(np.mean(gamma[:, i])),
                            float(np.std(gamma[:, i]))
                        )
            
            return weights
            
        except Exception as e:
            logger.error(f"Error getting feature weights: {str(e)}")
            return {}
                                                        
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
        
    #--------------------------------------------
    # Data preparation methods
    #--------------------------------------------
    def _prepare_feature_matrices(self, dataset: PreferenceDataset,
                                main_features: List[str],
                                control_features: List[str]) -> Dict[str, Any]:
        """Prepare feature matrices for Stan model."""
        try:
            # Collect all feature values
            feature_values = defaultdict(list)
            for feature in main_features + control_features:
                for pref in dataset.preferences:
                    feat1 = pref.features1.get(feature, 0.0)
                    feat2 = pref.features2.get(feature, 0.0)
                    if feature == 'typing_time' and (feat1 is None or feat2 is None):
                        feat1 = 0.0
                        feat2 = 0.0
                    feature_values[feature].extend([feat1, feat2])

            # Calculate statistics and store them
            self.feature_stats = {}
            
            # Process feature statistics
            for feature in main_features + control_features:
                values = np.array([v for v in feature_values[feature] if v is not None])
                mean = float(np.mean(values)) if len(values) > 0 else 0.0
                std = float(np.std(values)) if len(values) > 0 else 1.0
                self.feature_stats[feature] = {'mean': mean, 'std': std}
                
                logger.info(f"{feature} ({'main' if feature in main_features else 'control'}): original - mean: {mean:.3f}, std: {std:.3f}")

            # Initialize matrices
            X1 = np.zeros((len(dataset.preferences), 0), dtype=np.float64)  # Empty if no main features
            X2 = np.zeros((len(dataset.preferences), 0), dtype=np.float64)
            C1 = []
            C2 = []
            participant = []
            y = []

            # Process preferences
            for pref in dataset.preferences:
                # Process main features if any exist
                if main_features:
                    features1_main = []
                    features2_main = []
                    for feature in main_features:
                        feat1 = pref.features1.get(feature, 0.0)
                        feat2 = pref.features2.get(feature, 0.0)
                        if feature == 'typing_time' and (feat1 is None or feat2 is None):
                            feat1 = feat2 = 0.0
                        mean = self.feature_stats[feature]['mean']
                        std = self.feature_stats[feature]['std']
                        features1_main.append((feat1 - mean) / std)
                        features2_main.append((feat2 - mean) / std)
                    X1 = np.array([features1_main for _ in range(len(dataset.preferences))])
                    X2 = np.array([features2_main for _ in range(len(dataset.preferences))])

                # Process control features
                features1_control = []
                features2_control = []
                for feature in control_features:
                    feat1 = pref.features1.get(feature, 0.0)
                    feat2 = pref.features2.get(feature, 0.0)
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    features1_control.append((feat1 - mean) / std)
                    features2_control.append((feat2 - mean) / std)
                C1.append(features1_control)
                C2.append(features2_control)

                # Add participant and response data
                participant.append(pref.participant_id)
                y.append(1 if pref.preferred else 0)

            # Convert to numpy arrays
            C1 = np.array(C1, dtype=np.float64)
            C2 = np.array(C2, dtype=np.float64)

            # Process participant IDs
            unique_participants = sorted(set(participant))
            participant_map = {pid: i for i, pid in enumerate(unique_participants)}
            participant_indices = [participant_map[p] for p in participant]
            participant = np.array(participant_indices, dtype=np.int32)

            # Convert response vector
            y = np.array(y, dtype=np.int32)

            # Log dimensions and stats
            logger.info(f"Number of unique participants: {len(unique_participants)}")
            logger.info(f"Participant ID range: 0 to {len(unique_participants) - 1}")
            logger.info(f"Matrix shapes:")
            logger.info(f"  Main features: {len(main_features)}")
            logger.info(f"  Control features: {len(control_features)}")
            logger.info(f"  X1, X2: {X1.shape}, {X2.shape}")
            logger.info(f"  C1, C2: {C1.shape}, {C2.shape}")

            # Log standardization results
            for feature in main_features + control_features:
                try:
                    mean = np.mean(np.concatenate([
                        X1[:, main_features.index(feature)] if feature in main_features 
                        else C1[:, control_features.index(feature)],
                        X2[:, main_features.index(feature)] if feature in main_features
                        else C2[:, control_features.index(feature)]
                    ]))
                    std = np.std(np.concatenate([
                        X1[:, main_features.index(feature)] if feature in main_features
                        else C1[:, control_features.index(feature)],
                        X2[:, main_features.index(feature)] if feature in main_features
                        else C2[:, control_features.index(feature)]
                    ]))
                    logger.info(f"{feature} standardized - mean: {mean:.3f}, std: {std:.3f}")
                except Exception as e:
                    logger.warning(f"Error calculating standardization stats for {feature}: {str(e)}")

            # Return prepared data with 1-based participant indices for Stan
            return {
                'N': len(y),
                'P': len(unique_participants),
                'F': X1.shape[1],  # Can be 0 when no main features
                'C': C1.shape[1],  # Number of control features (always > 0)
                'has_main_features': 1 if main_features else 0,  # Flag for Stan
                'participant': participant + 1,  # Stan uses 1-based indexing
                'X1': X1,
                'X2': X2,
                'C1': C1,
                'C2': C2,
                'y': y,
                'feature_scale': self.config.model.feature_scale,
                'participant_scale': self.config.model.participant_scale
            }

        except Exception as e:
            logger.error(f"Error preparing feature matrices: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise
                                                                                                        
    def _update_feature_weights(self) -> None:
        """Update feature weights from fitted model for both main and control features."""
        try:
            self.feature_weights = {}
            
            if not hasattr(self, 'fit_result') or self.fit_result is None:
                raise ValueError("Model not fitted. Call fit() first.")
            
            # Verify Stan variables exist and have expected shapes
            if not hasattr(self.fit_result, 'stan_variable'):
                raise ValueError("Invalid Stan result - missing stan_variable method")
                        
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
            
        finally:
            # Clean up the run directory after we're done with the results
            if hasattr(self.fit_result, '_run_dir'):
                try:
                    run_dir = Path(self.fit_result._run_dir)
                    if run_dir.exists():
                        import shutil
                        shutil.rmtree(run_dir)
                except Exception as e:
                    logger.warning(f"Could not clean up run directory: {e}")
                                                                                                
    def _sample_with_retry(self, **kwargs):
        """Attempt sampling with retries and disk space checks."""
        run_dir = None
        try:
            # Create temp directory in user's home directory
            temp_dir = Path.home() / '.engram_temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.chmod(0o755)

            # Create a unique subfolder for this run
            import uuid
            run_dir = temp_dir / f"run_{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Override CmdStanPy's temp directory setting
            import cmdstanpy.utils.filesystem as csfs
            original_tmpdir = csfs._TMPDIR
            csfs._TMPDIR = str(run_dir)
            
            # Add output directory to kwargs
            kwargs['output_dir'] = str(run_dir)
            
            # Check available space
            required_mb = self.config.model.required_temp_mb
            if not self.check_disk_space(required_mb):
                raise OSError(f"Insufficient disk space. Need {required_mb}MB.")
            
            # Attempt sampling
            logger.info(f"Starting Stan sampling in {run_dir}...")
            result = self.model.sample(**kwargs)
            
            if not hasattr(result, 'stan_variable'):
                raise RuntimeError("Sampling failed to produce valid results")

            # Store run_dir in result to prevent premature cleanup
            result._run_dir = run_dir
                
            return result
            
        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            # Clean up failed run directory
            if run_dir:
                self._cleanup_run_directory(run_dir)
            raise
            
        finally:
            # Restore original temp directory
            if 'original_tmpdir' in locals():
                csfs._TMPDIR = original_tmpdir
                                                                            
    #--------------------------------------------
    # Feature selection methods
    #--------------------------------------------  
    def select_features(self, dataset: PreferenceDataset, all_features: List[str]) -> List[str]:
        """Select features by evaluating their importance for prediction."""
        try:
            self._check_memory_usage()
            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor  # Ensure feature_extractor is set

            # Preprocess dataset
            logger.info("Preprocessing dataset for feature selection...")
            valid_prefs = []
            for pref in dataset.preferences:
                valid = True
                for feature in all_features:
                    if (pref.features1.get(feature) is None or 
                        pref.features2.get(feature) is None):
                        valid = False
                        break
                if valid:
                    valid_prefs.append(pref)
            
            # Create processed dataset
            self.dataset = PreferenceDataset.__new__(PreferenceDataset)
            self.dataset.preferences = valid_prefs
            self.dataset.participants = {p.participant_id for p in valid_prefs}
            self.dataset.file_path = dataset.file_path
            self.dataset.config = dataset.config
            self.dataset.control_features = dataset.control_features
            self.dataset.feature_extractor = dataset.feature_extractor
            self.dataset.feature_names = dataset.feature_names
            self.dataset.all_bigrams = dataset.all_bigrams
            self.dataset.all_bigram_features = dataset.all_bigram_features
            
            # Initialize feature sets
            control_features = self.config.features.control_features
            candidate_features = [f for f in all_features if f not in control_features]
            selected_features = list(control_features)  # Start with control features

            # Log initial state
            logger.info("Starting feature selection:")
            logger.info(f"  Control features: {selected_features}")
            logger.info(f"  Candidate features: {len(candidate_features)}")
            
            # Keep selecting features until no more useful ones found
            round_num = 1
            while candidate_features:
                logger.info(f"Selection round {round_num}")
                logger.info(f"  Current features: {selected_features}")
                logger.info(f"  Remaining candidates: {len(candidate_features)}")
                
                best_feature = None
                best_importance = -float('inf')
                
                # Evaluate each candidate
                for feature in candidate_features:
                    importance = self._calculate_feature_importance(
                        feature=feature,
                        dataset=self.dataset,
                        current_features=selected_features
                    )

                    # If feature helps predictions more than our threshold
                    if importance > self.config.feature_selection.importance_threshold:
                        if importance > best_importance:
                            best_importance = importance
                            best_feature = feature
                
                if best_feature is None:
                    logger.info("No remaining features improve predictions sufficiently.")
                    break
                    
                # Add best feature and continue
                logger.info(f"\nSelected feature {best_feature} with importance {best_importance:.4f}")
                selected_features.append(best_feature)
                candidate_features.remove(best_feature)
                round_num += 1

            logger.info("Feature selection complete:")
            logger.info(f"Selected features: {selected_features}")
            self.selected_features = selected_features
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in select_features: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

        finally:
            self.cleanup()

    def _calculate_feature_importance(self, feature: str, dataset: PreferenceDataset, 
                                current_features: List[str]) -> float:
        """Calculate feature importance based on prediction improvement.
        
        Returns:
            float: Importance score using bounded consistency metric (0 to effect_magnitude)
        """
        logger.info(f"\nCalculating importance for feature: {feature}")
        logger.info(f"Current features: {current_features}")
        
        if not self.is_fitted:
            logger.info("Model is not fitted. Attempting to fit now...")
            self.fit(dataset, features=current_features)
            
        try:
            # Verify dataset and feature extractor
            if dataset is None or dataset.feature_extractor is None:
                logger.error("Invalid dataset or missing feature extractor")
                return -float('inf')
                
            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor
            
            # Get cross-validation splits
            cv_splits = self._get_cv_splits(dataset, n_splits=5)
            cv_aligned_effects = []
            
            # Process each fold
            for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
                logger.info(f"\nProcessing fold {fold}/5 for {feature}")
                train_data = dataset._create_subset_dataset(train_idx)
                val_data = dataset._create_subset_dataset(val_idx)
                logger.info(f"Train set size: {len(train_data.preferences)} preferences")
                logger.info(f"Validation set size: {len(val_data.preferences)} preferences")
                
                try:
                    # Create and train both models for this fold
                    with self._create_temp_model() as fold_baseline_model, \
                        self._create_temp_model() as fold_feature_model:
                        
                        # Ensure feature extractors are set
                        fold_baseline_model.feature_extractor = dataset.feature_extractor
                        fold_feature_model.feature_extractor = dataset.feature_extractor
                        
                        # Train baseline model
                        logger.info(f"Training baseline model for fold {fold}")
                        fold_baseline_model.fit(train_data, features=current_features)
                        
                        # Check if baseline model fitted - now checking fit_result existence
                        if not hasattr(fold_baseline_model, 'fit_result'):
                            logger.warning(f"No fit result for baseline model in fold {fold}")
                            continue
                            
                        # Train feature model
                        logger.info(f"Training feature model for fold {fold}")
                        fold_feature_model.fit(train_data, features=current_features + [feature])
                        
                        # Check if feature model fitted - now checking fit_result existence
                        if not hasattr(fold_feature_model, 'fit_result'):
                            logger.warning(f"No fit result for feature model in fold {fold}")
                            continue
                        
                        # Calculate effects for validation set
                        fold_effects = []
                        for pref in val_data.preferences:
                            try:
                                # Get baseline prediction
                                base_pred = fold_baseline_model.predict_preference(pref.bigram1, pref.bigram2)
                                if base_pred.probability <= 0 or base_pred.probability >= 1:
                                    logger.warning(f"Invalid baseline probability for {pref.bigram1}-{pref.bigram2}")
                                    continue
                                base_logit = np.log(base_pred.probability / (1 - base_pred.probability))
                                
                                # Get feature model prediction
                                feat_pred = fold_feature_model.predict_preference(pref.bigram1, pref.bigram2)
                                if feat_pred.probability <= 0 or feat_pred.probability >= 1:
                                    logger.warning(f"Invalid feature probability for {pref.bigram1}-{pref.bigram2}")
                                    continue
                                feat_logit = np.log(feat_pred.probability / (1 - feat_pred.probability))
                                
                                # Calculate aligned effect
                                effect = feat_logit - base_logit
                                aligned_effect = effect if pref.preferred else -effect
                                fold_effects.append(aligned_effect)
                                
                            except Exception as e:
                                logger.warning(f"Error processing preference {pref.bigram1}-{pref.bigram2}: {str(e)}")
                                continue
                                
                        # Process fold results
                        if fold_effects:
                            mean_fold_effect = np.mean(fold_effects)
                            logger.info(f"Mean effect for fold {fold}: {mean_fold_effect:.4f}")
                            cv_aligned_effects.extend(fold_effects)
                        else:
                            logger.warning(f"No valid effects calculated for fold {fold}")
                        
                except Exception as e:
                    logger.warning(f"Error processing fold {fold}: {str(e)}")
                    continue
                    
            # Calculate metrics if we have effects
            if cv_aligned_effects:
                cv_aligned_effects = np.array(cv_aligned_effects)
                mean_aligned_effect = float(np.mean(cv_aligned_effects))
                effect_std = float(np.std(cv_aligned_effects))
                effect_magnitude = abs(mean_aligned_effect)
                
                # Calculate different consistency metrics
                # 1. Unbounded negative
                consistency_unbounded = 1 - (effect_std / (effect_magnitude + 1e-6))
                
                # 2. Bounded [0,1] using inverse ratio
                consistency_bounded = 1 / (1 + (effect_std / effect_magnitude))
                
                # 3. Bounded [0,1] using capped ratio
                consistency_capped = 1 - min(1, effect_std / effect_magnitude)
                
                # 4. Bounded [0,1] using sigmoid of ratio
                consistency_sigmoid = 1 / (1 + np.exp(effect_std / effect_magnitude - 1))
                
                # Calculate importance scores using different methods
                importance_original = effect_magnitude * max(0, consistency_unbounded)
                importance_bounded = effect_magnitude * consistency_bounded  
                importance_capped = effect_magnitude * consistency_capped
                importance_sigmoid = effect_magnitude * consistency_sigmoid
                
                # Log detailed metrics
                logger.info(f"\nFeature importance analysis for '{feature}':")
                logger.info(f"Raw metrics:")
                logger.info(f"  Total effects analyzed: {len(cv_aligned_effects)}")
                logger.info(f"  Mean aligned effect: {mean_aligned_effect:.4f}")
                logger.info(f"  Effect magnitude: {effect_magnitude:.4f}")
                logger.info(f"  Effect std dev: {effect_std:.4f}")
                logger.info(f"  Std/Magnitude ratio: {(effect_std/effect_magnitude):.4f}")
                
                logger.info(f"\nConsistency metrics:")
                logger.info(f"  unbounded ratio: 1 - (std/magnitude) = {consistency_unbounded:.5f}")
                logger.info(f"  min-capped ratio: 1 - min(1, std/magnitude) = {consistency_capped:.5f}")
                logger.info(f"  inverse-bounded: 1/(1 + std/magnitude) = {consistency_bounded:.5f}")
                logger.info(f"  sygmoid-bounded: 1/(1 + exp(std/magnitude - 1)) = {consistency_sigmoid:.5f}")
                
                logger.info(f"\nImportance scores = effect magnitude times:")
                logger.info(f"  max(0, unbounded consistency): {importance_original:.5f}")
                logger.info(f"  min-capped consistency: {importance_capped:.5f}")
                logger.info(f"  inverse-bounded consistency: {importance_bounded:.5f}")
                logger.info(f"  sigmoid-bounded consistency: {importance_sigmoid:.5f}")
                
                # Return selected importance measure
                selected_importance = importance_bounded
                return selected_importance
                
            else:
                logger.warning("No valid effects calculated")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return 0.0
        
        finally:
            # Ensure cleanup even if error occurs
            self.cleanup_temp_models()
                        
    #--------------------------------------------
    # Cross-validation methods
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
    # Serialization methods
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
                raise
        except Exception:
            logger.error(f"Error loading model from {path}")
            raise

    #--------------------------------------------
    # Properties
    #--------------------------------------------
    @property
    def feature_scale(self) -> float:
        """Prior scale for feature weights."""
        return self.config.model.feature_scale
        
    @property
    def participant_scale(self) -> float:
        """Prior scale for participant effects."""
        return self.config.model.participant_scale
        
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, value: bool):
        """Set the fitted state of the model."""
        self._is_fitted = value
