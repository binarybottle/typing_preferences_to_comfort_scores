# model.py
"""
Bayesian hierarchical model for keyboard layout preference learning. 

Core components:

1. Model Architecture:
- Stan-based Bayesian hierarchical model 
- Participant-specific random effects
- Configurable feature interactions
- Feature importance estimation
- Control feature handling
- Single-key score derivation

2. Training Pipeline:
- Feature preprocessing and standardization
- MCMC sampling with convergence checks
- Cross-validation with participant awareness
- Model state persistence
- Memory usage monitoring

3. Prediction Systems:
- Bigram comfort score estimation
- Single-key comfort score estimation
- Uncertainty quantification
- Feature interaction handling
- Temporary model management
- Basic prediction caching

4. Resource Management:
- Memory usage monitoring
- Disk space verification
- Temporary file cleanup
- Cache size limits
- Error recovery

Core Features:
- Cross-validated feature selection
- Bayesian uncertainty estimation
- Resource-aware computation
- Comprehensive error handling
- Dual-level comfort score prediction (bigrams and single keys)
"""
import random
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
import gc
from datetime import datetime, timedelta
import matplotlib as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from bigram_typing_preferences_to_comfort_scores.utils.config import Config, NotFittedError, FeatureError, ModelPrediction
from bigram_typing_preferences_to_comfort_scores.data import PreferenceDataset
from bigram_typing_preferences_to_comfort_scores.utils.visualization import PlottingUtils
from bigram_typing_preferences_to_comfort_scores.utils.logging import LoggingManager
from bigram_typing_preferences_to_comfort_scores.features.keymaps import (
    finger_map,
    engram_position_values,
    row_position_values
)
logger = LoggingManager.getLogger(__name__)

def set_all_seeds(base_seed: int):
    """
    Set all seeds for reproducibility.
    Using different derived seeds for different components.
    Args:
        base_seed: Base seed from config.yaml to derive all other seeds
    """
    # Create different seeds for different components
    np_seed = base_seed
    python_seed = base_seed + 1
    stan_seed = base_seed + 2
    cv_seed = base_seed + 3
    
    # Set seeds
    np.random.seed(np_seed)
    random.seed(python_seed)
    os.environ['PYTHONHASHSEED'] = str(python_seed)
    
    return {
        'numpy': np_seed,
        'python': python_seed,
        'stan': stan_seed,
        'cv': cv_seed
    }

class ModelPrediction:
    """Holds prediction results."""
    def __init__(self, probability: float, uncertainty: float, 
                 features_used: List[str] = None, computation_time: float = 0.0):
        self.probability = probability
        self.uncertainty = uncertainty
        self.features_used = features_used or []
        self.computation_time = computation_time

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
        
        # First convert config if needed
        self.config = config if isinstance(config, Config) else Config(**config)
        
        # Then set seeds using the proper config object
        self.seeds = set_all_seeds(self.config.data.splits['random_seed'])
        
        self._is_fitted = False
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
        
        # Add storage for feature importance metrics
        self.feature_importance_metrics = {}

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
            sampling_params = {
                'chains': self.config.model.chains,
                'iter_warmup': self.config.model.warmup,
                'iter_sampling': self.config.model.n_samples,
                'adapt_delta': self.config.model.adapt_delta,
                'max_treedepth': self.config.model.max_treedepth,
                'seed': self.seeds['stan'], 
                'refresh': None
            }
            
            # Use sampling params in _sample_with_retry
            self.fit_result = self._sample_with_retry(
                data=processed_data,
                **sampling_params
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

            # Get feature weights from model
            control_features = self.config.features.control_features
            main_features = [f for f in self.feature_names if f not in control_features]
            
            # Initialize arrays for each feature group
            X1 = np.zeros((1, len(main_features))) if main_features else None
            X2 = np.zeros((1, len(main_features))) if main_features else None
            C1 = np.zeros((1, len(control_features)))
            C2 = np.zeros((1, len(control_features)))

            # Process main features if they exist
            if main_features and 'beta' in self.fit_result.stan_variables():
                for i, feature in enumerate(main_features):
                    feat1 = features1.get(feature, 0.0)
                    feat2 = features2.get(feature, 0.0)
                    
                    # Standardize using stored stats
                    if feature in self.feature_stats:
                        mean = self.feature_stats[feature]['mean']
                        std = self.feature_stats[feature]['std']
                        if abs(std) < 1e-8:
                            logger.warning(f"Feature '{feature}' has near-zero standard deviation ({std}).")
                            std = 1e-8
                        feat1 = (feat1 - mean) / std
                        feat2 = (feat2 - mean) / std
                        
                    X1[0, i] = feat1
                    X2[0, i] = feat2

            # Process control features
            for i, feature in enumerate(control_features):
                feat1 = features1.get(feature, 0.0)
                feat2 = features2.get(feature, 0.0)
                
                if feature in self.feature_stats:
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    if abs(std) < 1e-8:
                        logger.warning(f"Feature '{feature}' has near-zero standard deviation ({std}).")
                        std = 1e-8
                    feat1 = (feat1 - mean) / std
                    feat2 = (feat2 - mean) / std
                    
                C1[0, i] = feat1
                C2[0, i] = feat2

            # Get model weights and number of samples
            gamma = self.fit_result.stan_variable('gamma')  # Control feature weights
            n_samples = len(gamma)
            logits = np.zeros(n_samples)
            
            # Calculate logits for each MCMC sample
            for i in range(n_samples):
                logit = 0.0
                # Add main features contribution if they exist
                if main_features and 'beta' in self.fit_result.stan_variables():
                    beta = self.fit_result.stan_variable('beta')
                    main_diff = np.dot(X1 - X2, beta[i])
                    logit += main_diff
                
                # Add control features contribution
                control_diff = np.dot(C1 - C2, gamma[i])
                logit += control_diff
                logits[i] = logit

            # Calculate probability and uncertainty
            probabilities = 1 / (1 + np.exp(-logits))
            mean_prob = float(np.mean(probabilities))
            uncertainty = float(np.std(probabilities))

            return ModelPrediction(
                probability=mean_prob,
                uncertainty=uncertainty,
                features_used=main_features + list(control_features),
                computation_time=0.0
            )

        except Exception as e:
            logger.error(f"Error in predict_preference: {str(e)}, available variables: {self.fit_result.stan_variables()}")
            return ModelPrediction(
                probability=0.5,
                uncertainty=1.0,
                features_used=[],
                computation_time=0.0
            )
                
    def predict_comfort_score(self, bigram: str) -> ModelPrediction:
        try:
            score, uncertainty = self.get_bigram_comfort_scores(bigram)
            return ModelPrediction(probability=score, uncertainty=uncertainty)
        except Exception as e:
            logger.error(f"Error predicting comfort score for {bigram}: {str(e)}")
            return ModelPrediction(probability=0.0, uncertainty=1.0)

    @staticmethod
    def normalize_keymap_values(values_dict):
        """
        Normalize keymap values to [0,1] range where 1 = most discomfort
        """
        values = np.array(list(values_dict.values()))
        min_val = values.min()
        max_val = values.max()
        return {k: (v - min_val)/(max_val - min_val) for k, v in values_dict.items()}

    @staticmethod
    def are_significantly_different(score_a, std_a, score_b, std_b, confidence_level=0.95):
        """
        Determine if two scores are significantly different using a simpler criterion
        """
        # Calculate the difference in terms of pooled standard deviations
        diff = abs(score_a - score_b)
        pooled_std = np.sqrt(std_a**2 + std_b**2)
        
        # Use 2 standard deviations as threshold (roughly 95% confidence)
        return diff > 2 * pooled_std

    def plot_key_scores(self, df, exclude_keys=None, title_suffix="", save_suffix=""):
        """Create comfort score plot, optionally excluding certain keys."""
        if exclude_keys:
            df = df[~df['key'].isin(exclude_keys)].copy()
        
        fig = Figure(figsize=self.config.visualization.figure_size)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        x = range(len(df))
        ax.errorbar(x, 
                    df['comfort_score'],
                    yerr=df['uncertainty'],
                    fmt='o',
                    capsize=5,
                    alpha=self.config.visualization.alpha)
        
        # Only compare adjacent keys
        df = df.sort_values('comfort_score', ascending=False)
        adjacent_differences = []

        print("\nAnalyzing adjacent pairs:")
        for i in range(len(df)-1):
            key1, key2 = df.iloc[i]['key'], df.iloc[i+1]['key']
            score1, std1 = df.iloc[i]['comfort_score'], df.iloc[i]['uncertainty']
            score2, std2 = df.iloc[i+1]['comfort_score'], df.iloc[i+1]['uncertainty']
            
            diff = abs(score1 - score2)
            pooled_std = np.sqrt(std1**2 + std2**2)
            is_different = diff > 1 * pooled_std
            
            print(f"Pair {key1}-{key2}:")
            print(f"  Scores: {score1:.3f} ± {std1:.3f} vs {score2:.3f} ± {std2:.3f}")
            print(f"  Difference: {diff:.3f}, Pooled std: {pooled_std:.3f}")
            print(f"  Significantly different: {is_different}")
            
            if is_different:
                adjacent_differences.append((key1, key2))

        print(f"\nNumber of adjacent differences found: {len(adjacent_differences)}")
        print("Adjacent differences:", adjacent_differences)

        # Add significance bars with distinct heights
        y_min = df['comfort_score'].min()
        y_max = df['comfort_score'].max()
        plot_height = y_max - y_min

        # Get the x positions for the keys (should be 0 through len(df)-1)
        x_positions = {key: i for i, key in enumerate(df['key'])}

        for i, (key1, key2) in enumerate(adjacent_differences):
            # Use actual x positions instead of index lookups
            x1 = x_positions[key1]
            x2 = x_positions[key2]
            bar_y = y_max + (0.1 + i * 0.05) * plot_height
            ax.hlines(y=bar_y,
                    xmin=x1, xmax=x2,
                    color='red', alpha=0.7,
                    linewidth=2)

        # Adjust y-axis limits to show all lines
        ax.set_ylim(y_min - 0.1 * plot_height, y_max + (0.1 + len(adjacent_differences) * 0.05) * plot_height)

        ax.set_xticks(x)
        ax.set_xticklabels(df['key'])
        ax.set_ylabel('Comfort Score')
        ax.set_xlabel('Key')
        ax.set_title(f'Keyboard Key Comfort Scores with Uncertainty{title_suffix}')
        ax.grid(True, alpha=0.3)
        
        plot_file = Path(self.config.paths.plots_dir) / f"key_comfort_scores{save_suffix}.png"
        fig.savefig(plot_file, dpi=self.config.visualization.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_file}")

    def predict_key_scores(self):
        """Predict comfort scores for individual keys using normalized values and model weights."""
        try:
            from bigram_typing_preferences_to_comfort_scores.features.keymaps import (
                finger_map,
                engram_position_values,
                row_position_values
            )
            
            # Get covariance matrix from fitted model
            if hasattr(self.fit_result, 'stan_variables'):
                # Get MCMC samples for all parameters
                beta_samples = self.fit_result.stan_variable('beta')
                # Calculate covariance matrix from MCMC samples
                feature_covariance = np.cov(beta_samples.T)
                logger.info("\nFeature covariance matrix shape: %s", feature_covariance.shape)
            
            # Normalize keymap values
            norm_finger = self.normalize_keymap_values(finger_map)
            norm_engram = self.normalize_keymap_values(engram_position_values)
            norm_row = self.normalize_keymap_values(row_position_values)

            logger.info("\nNormalized value ranges:")
            logger.info(f"finger_map: {min(norm_finger.values()):.3f} to {max(norm_finger.values()):.3f}")
            logger.info(f"engram_pos: {min(norm_engram.values()):.3f} to {max(norm_engram.values()):.3f}")
            logger.info(f"row_pos: {min(norm_row.values()):.3f} to {max(norm_row.values()):.3f}")

            # Get model weights
            feature_weights = self.get_feature_weights()
            
            # MODIFIED CODE: Only include single-key features that exist in the model
            base_single_key_features = [
                'sum_finger_values',
                'sum_engram_position_values', 
                'sum_row_position_values'
            ]
            
            # Only add interaction features if they exist in the model weights
            single_key_features = []
            for feature in base_single_key_features:
                if feature in feature_weights:
                    single_key_features.append(feature)
                    
            # Add interaction features only if they exist
            potential_interactions = [
                'sum_finger_values_x_sum_row_position_values',
                'sum_engram_position_values_x_sum_finger_values'
            ]
            
            for feature in potential_interactions:
                if feature in feature_weights:
                    single_key_features.append(feature)
            
            # Get feature indices for covariance lookup
            feature_indices = {f: i for i, f in enumerate(self.feature_names)
                            if f in single_key_features}
            
            relevant_weights = {
                feature: feature_weights[feature] 
                for feature in single_key_features
                if feature in feature_weights
            }
            
            logger.info("\nUse these single-key relevant weights from model:")
            for feature, (weight, std) in relevant_weights.items():
                logger.info(f"{feature}: {weight:.3f} ± {std:.3f}")
            
            layout_chars = self.config.data.layout['chars']
            results = []

            # Create new consecutive indices just for our single-key features
            single_key_indices = {feature: idx for idx, feature in enumerate(single_key_features)}
            
            logger.info("\nSingle key feature indices:")
            logger.info(single_key_indices)

            # Get the subset of the covariance matrix we need
            full_indices = [feature_indices[f] for f in single_key_features]
            reduced_covariance = feature_covariance[np.ix_(full_indices, full_indices)]

            # After creating reduced covariance:
            logger.info("\nReduced covariance matrix:")
            logger.info("Features: %s", single_key_features)
            logger.info("\n".join([f"{row}" for row in reduced_covariance]))

            for key in layout_chars:
                key_values = {
                    'sum_finger_values': norm_finger[key],
                    'sum_engram_position_values': norm_engram[key],
                    'sum_row_position_values': norm_row[key],
                    'sum_finger_values_x_sum_row_position_values': 
                        norm_finger[key] * norm_row[key],
                    'sum_engram_position_values_x_sum_finger_values': 
                        norm_engram[key] * norm_finger[key]
                }
                
                score = 0.0
                feature_values = np.zeros(len(single_key_features))  # Now size 5
                
                for feature, value in key_values.items():
                    if feature in relevant_weights:
                        weight, _ = relevant_weights[feature]
                        score += value * weight
                        if feature in single_key_indices:  # Use new indices
                            feature_values[single_key_indices[feature]] = value

                # Calculate variance with reduced covariance matrix
                score_variance = feature_values.T @ reduced_covariance @ feature_values
                score_std = np.sqrt(max(0, score_variance))

                results.append({
                    'key': key,
                    'comfort_score': score,
                    'uncertainty': score_std,
                    'normalized_values': key_values
                })
            
            # Create DataFrame and sort by comfort score
            df = pd.DataFrame(results)
            df = df.sort_values('comfort_score', ascending=False)
            
            # Log feature covariances
            logger.info("\nFeature covariances:")
            for i, feat1 in enumerate(single_key_features):
                if feat1 not in feature_indices:
                    continue
                idx1 = feature_indices[feat1]
                for j, feat2 in enumerate(single_key_features):
                    if feat2 not in feature_indices:
                        continue
                    idx2 = feature_indices[feat2]
                    cov = feature_covariance[idx1, idx2]
                    if abs(cov) > 1e-6:  # Only show non-zero covariances
                        logger.info(f"{feat1} - {feat2}: {cov:.6f}")

            # Create full plot
            self.plot_key_scores(df)
            
            # Create filtered plot
            exclude_keys = ['t', 'g', 'b']
            self.plot_key_scores(df, exclude_keys=exclude_keys,
                                 title_suffix=" (Filtered)",
                                 save_suffix="_filtered")

            return df

        except Exception as e:
            logger.error(f"Error in predict_key_scores: {str(e)}")
            raise
                                                                
    #--------------------------------------------
    # Resource management methods
    #--------------------------------------------
    def cleanup(self, preserve_features: bool = False):
        """Clean up model resources."""
        # Store features if needed
        saved_state = None
        if preserve_features:
            saved_state = {
                'feature_names': getattr(self, 'feature_names', None),
                'selected_features': getattr(self, 'selected_features', None),
                'feature_extractor': getattr(self, 'feature_extractor', None)
            }
        
        # Don't clear fit_result if model is fitted
        if not self.is_fitted:
            if hasattr(self, 'fit_result'):
                self._cleanup_run_directory(active_only=True)
                del self.fit_result
        
        if hasattr(self, '_feature_data_cache'):
            self._feature_data_cache.clear()
                        
        # Don't reset state if model is fitted
        if not self.is_fitted:
            self.reset_state()
            
            # Restore saved state
            if preserve_features and saved_state:
                for key, value in saved_state.items():
                    if value is not None:
                        setattr(self, key, value)
                
        gc.collect()
        
    def _create_temp_model(self):
        """Create a temporary model with proper cleanup."""
        temp_model = type(self)(config=self.config)
        
        # Copy over key state
        temp_model.feature_extractor = self.feature_extractor
        temp_model.feature_stats = (self.feature_stats.copy() 
                                if hasattr(self, 'feature_stats') and self.feature_stats is not None 
                                else {})
        
        # Copy feature-related state
        if hasattr(self, 'feature_names'):
            temp_model.feature_names = self.feature_names.copy() if self.feature_names else []
        if hasattr(self, 'selected_features'):
            temp_model.selected_features = self.selected_features.copy() if self.selected_features else []

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
                # Immediate cleanup of Stan results
                if hasattr(self.model, 'fit_result'):
                    if hasattr(self.model.fit_result, '_run_dir'):
                        shutil.rmtree(self.model.fit_result._run_dir)
                    del self.model.fit_result
                
                # Clear any cached data
                if hasattr(self.model, '_feature_data_cache'):
                    self.model._feature_data_cache.clear()
                
                # Remove from parent's temp models list
                if self.model in self._parent_models:
                    self._parent_models.remove(self.model)
                
                gc.collect()

        return ModelContext(temp_model, self._temp_models)

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
        
        # Copy over key state
        temp_model.feature_extractor = self.feature_extractor
        
        # Safely copy feature stats
        temp_model.feature_stats = (self.feature_stats.copy() 
                                if hasattr(self, 'feature_stats') and self.feature_stats is not None 
                                else {})
        
        # Copy feature-related state
        if hasattr(self, 'feature_names'):
            temp_model.feature_names = self.feature_names.copy() if self.feature_names else []
        if hasattr(self, 'selected_features'):
            temp_model.selected_features = self.selected_features.copy() if self.selected_features else []

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
        """Monitor memory usage with more aggressive cleanup."""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        if memory_percent > 75:  # Lower threshold for earlier intervention
            logger.warning(f"High memory usage ({memory_percent:.1f}%). Cleaning up...")
            # Clear feature cache
            if hasattr(self, '_feature_data_cache'):
                self._feature_data_cache.clear()
            
            # Clear Stan results if not needed
            if hasattr(self, 'fit_result') and not self.is_fitted:
                if hasattr(self.fit_result, '_run_dir'):
                    shutil.rmtree(self.fit_result._run_dir)
                del self.fit_result
            
            gc.collect()

    def _check_memory_before_fold(self) -> bool:
        """Check memory status and clean up if needed. Returns True if safe to proceed."""
        try:
            mem = psutil.virtual_memory()
            if mem.percent > 90:  # Increased from 80% to 90%
                logger.warning(f"High memory usage ({mem.percent}%). Cleaning up...")
                
                # Clear feature cache
                if hasattr(self, '_feature_data_cache'):
                    self._feature_data_cache.clear()
                
                # Force garbage collection
                gc.collect()
                
                # Wait a bit and check again
                time.sleep(5)
                mem = psutil.virtual_memory()
                
                if mem.percent > 95:  # Only skip if really high (95%)
                    logger.error(f"Critical memory usage ({mem.percent}%), must skip fold")
                    return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return True  # Continue on error rather than skip
                        
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
        
    def _get_progress_stats(self, features_done: int, total_features: int, 
                        start_time: datetime, current_feature: str) -> Dict[str, Any]:
        """Calculate progress statistics."""
        now = datetime.now()
        elapsed = now - start_time
        
        if features_done > 0:
            avg_time_per_feature = elapsed / features_done
            features_left = total_features - features_done
            est_time_left = avg_time_per_feature * features_left
        else:
            est_time_left = timedelta(0)
            avg_time_per_feature = timedelta(0)
        
        return {
            'features_done': features_done,
            'features_left': total_features - features_done,
            'percent_done': (features_done / total_features) * 100,
            'elapsed_time': elapsed,
            'est_time_left': est_time_left,
            'avg_time_per_feature': avg_time_per_feature,
            'current_feature': current_feature
        }

    def _print_progress(self, stats: Dict[str, Any]):
        """Print progress information."""
        logger.info("\n=== Progress Update ===")
        logger.info(f"Current feature: {stats['current_feature']}")
        logger.info(f"Progress: {stats['features_done']}/{stats['features_done'] + stats['features_left']} features " +
                    f"({stats['percent_done']:.1f}%)")
        logger.info(f"Time elapsed: {str(stats['elapsed_time']).split('.')[0]}")
        logger.info(f"Est. time remaining: {str(stats['est_time_left']).split('.')[0]}")
        if stats['features_done'] > 0:
            logger.info(f"Avg. time per feature: {str(stats['avg_time_per_feature']).split('.')[0]}")
        logger.info("=" * 30 + "\n")
        
    #--------------------------------------------
    # Feature handling methods
    #--------------------------------------------  
    def _get_feature_data(self, feature: str, dataset: Optional[PreferenceDataset] = None) -> Dict[str, np.ndarray]:
        """
        Centralized method for getting feature data with caching.
        All features are assumed to be complete (no missing values).
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
            raw_features = {}
            for bigram in set(data1['raw_features']) | set(data2['raw_features']):
                raw_features[bigram] = data1['raw_features'].get(bigram, 0.0) * data2['raw_features'].get(bigram, 0.0)
            
        else:
            # Handle base features
            values = []
            differences = []
            raw_features = {}
            
            # Process all features uniformly since we know there are no missing values
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

    def extract_features(self, bigram: str) -> Dict[str, float]:
        """
        Public method to extract features for a bigram.
        
        Args:
            bigram: Two-character string to extract features for
            
        Returns:
            Dictionary mapping feature names to their values
            
        Raises:
            NotFittedError: If feature extractor is not initialized
            ValueError: If bigram is not valid
        """
        return self._extract_features(bigram)

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

    def _save_feature_metrics(self, feature: str, metrics: Dict[str, float], metrics_file: Path):
        """Save metrics for a single feature to CSV, creating or appending as needed."""
        # Add metadata to metrics
        metrics_row = {
            'feature_name': feature,
            'selected': 0,  # Default to not selected
            'round': 1,     # Default to first round
            **metrics      # Include all calculated metrics
        }
        
        df_row = pd.DataFrame([metrics_row])
        
        try:
            if metrics_file.exists():
                # Append to existing file
                df_row.to_csv(metrics_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                df_row.to_csv(metrics_file, index=False)
                
            logger.info(f"Saved metrics for feature '{feature}' to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics for feature '{feature}': {e}")

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
                    feature_values[feature].extend([feat1, feat2])

            # Calculate statistics and store them
            self.feature_stats = {}
            
            # Process feature statistics
            for feature in main_features + control_features:
                values = np.array(feature_values[feature])
                mean = float(np.mean(values))
                std = float(np.std(values)) if len(values) > 1 else 1.0
                self.feature_stats[feature] = {'mean': mean, 'std': std}
                
                logger.info(f"{feature} ({'main' if feature in main_features else 'control'}): original - mean: {mean:.3f}, std: {std:.3f}")

            # Initialize matrices with correct dimensions from the start
            X1 = np.zeros((len(dataset.preferences), len(main_features)), dtype=np.float64)
            X2 = np.zeros((len(dataset.preferences), len(main_features)), dtype=np.float64)
            C1 = np.zeros((len(dataset.preferences), len(control_features)), dtype=np.float64)
            C2 = np.zeros((len(dataset.preferences), len(control_features)), dtype=np.float64)
            participant = []
            y = []

            # Process preferences
            for i, pref in enumerate(dataset.preferences):
                # Process main features
                for j, feature in enumerate(main_features):
                    feat1 = pref.features1.get(feature, 0.0)
                    feat2 = pref.features2.get(feature, 0.0)
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    if abs(std) < 1e-8:
                        logger.warning(f"Feature '{feature}' has near-zero standard deviation ({std}).")
                        std = 1e-8
                    X1[i, j] = (feat1 - mean) / std
                    X2[i, j] = (feat2 - mean) / std

                # Process control features
                for j, feature in enumerate(control_features):
                    feat1 = pref.features1.get(feature, 0.0)
                    feat2 = pref.features2.get(feature, 0.0)
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    if abs(std) < 1e-8:
                        logger.warning(f"Feature '{feature}' has near-zero standard deviation ({std}).")
                        std = 1e-8
                    C1[i, j] = (feat1 - mean) / std
                    C2[i, j] = (feat2 - mean) / std

                # Add participant and response data
                participant.append(pref.participant_id)
                y.append(1 if pref.preferred else 0)

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
            logger.debug(f"Matrix shapes:")
            logger.debug(f"  Main features: {len(main_features)}")
            logger.debug(f"  Control features: {len(control_features)}")
            logger.debug(f"  X1, X2: {X1.shape}, {X2.shape}")
            logger.debug(f"  C1, C2: {C1.shape}, {C2.shape}")

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
                'F': X1.shape[1],  # Number of main features
                'C': C1.shape[1],  # Number of control features
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
        start_time = datetime.now()
        
        try:
            self._check_memory_usage()
            self.dataset = dataset
            self.feature_extractor = dataset.feature_extractor
            metrics_file = Path(self.config.feature_selection.metrics_file)

            # Initialize feature importance metrics storage
            self.feature_importance_metrics = {}

            # Load existing progress if any
            existing_metrics = {}
            if metrics_file.exists():
                try:
                    existing_df = pd.read_csv(metrics_file)
                    logger.info(f"Found existing metrics file with {len(existing_df)} features")
                    existing_metrics = {row['feature_name']: dict(row) 
                                    for _, row in existing_df.iterrows()}
                except Exception as e:
                    logger.warning(f"Error reading existing metrics: {e}")

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
            current_features = list(control_features)  # Features to use in current round
            selected_features = list(control_features)  # All selected features

            # Log initial state
            logger.info("\n=== Starting Feature Selection ===")
            logger.info(f"Initial control features: {control_features}")
            logger.info(f"Initial candidate features: {len(candidate_features)} features")
            logger.info(f"Previously evaluated: {len(existing_metrics)} features")
            logger.info(f"Importance threshold: {self.config.feature_selection.importance_threshold}")
            
            # Keep selecting features until no more useful ones found
            round_num = 1
            while candidate_features:
                logger.info(f"\n=== Selection Round {round_num} ===")
                logger.info(f"Round {round_num} - Starting features:")
                logger.info(f"  Current features: {current_features}")
                logger.info(f"  Selected features: {selected_features}")
                logger.info(f"  Candidates remaining: {len(candidate_features)} features")
                
                best_feature = None
                best_importance = -float('inf')
                best_metrics = None
                
                # Evaluate each candidate
                features_done = len(existing_metrics)
                total_features = len(candidate_features)
                
                for feature in candidate_features:
                    # Skip if already evaluated or selected
                    if feature in current_features or feature in selected_features:
                        logger.warning(f"Feature '{feature}' is already selected, skipping...")
                        continue

                    # Check if we already have metrics for this feature
                    if feature in existing_metrics:
                        metrics = existing_metrics[feature]
                        logger.info(f"Using existing metrics for '{feature}'")
                    else:
                        # Print progress
                        elapsed = datetime.now() - start_time
                        if features_done > 0:
                            time_per_feature = elapsed / features_done
                            est_remaining = time_per_feature * (total_features - features_done)
                        else:
                            est_remaining = timedelta(0)
                        
                        logger.info("\n=== Progress Update ===")
                        logger.info(f"Evaluating: {feature}")
                        logger.info(f"Progress: {features_done}/{total_features} features " +
                                f"({(features_done/total_features*100):.1f}%)")
                        logger.info(f"Time elapsed: {str(elapsed).split('.')[0]}")
                        logger.info(f"Est. remaining: {str(est_remaining).split('.')[0]}")
                        
                        # Evaluate feature
                        metrics = self._calculate_feature_importance(
                            feature=feature,
                            dataset=self.dataset,
                            current_features=current_features
                        )
                        
                        # Add metadata
                        metrics.update({
                            'feature_name': feature,
                            'round': round_num,
                            'n_components': len(feature.split('_x_')),
                            'selected': 0  # Will update to 1 if selected
                        })
                        
                        # Save metrics immediately
                        metrics_df = pd.DataFrame([metrics])
                        if metrics_file.exists():
                            metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
                        else:
                            metrics_df.to_csv(metrics_file, index=False)
                        
                        existing_metrics[feature] = metrics
                        features_done += 1
                    
                    # Store metrics and check if best
                    self.feature_importance_metrics[feature] = metrics
                    importance = metrics['selected_importance']
                    if importance > self.config.feature_selection.importance_threshold:
                        if importance > best_importance:
                            best_importance = importance
                            best_feature = feature
                            best_metrics = metrics
                            logger.info(f"\nFound new best feature '{feature}':")
                            logger.info(f"  Importance = {importance:.6f}")
                            logger.info(f"  Effect magnitude: {metrics['effect_magnitude']:.6f}")
                            logger.info(f"  Effect std dev: {metrics['effect_std']:.6f}")

                    # Cleanup after each feature
                    gc.collect()

                # After finding best feature
                if best_feature is not None:
                    # Update selected status in metrics file
                    if metrics_file.exists():
                        df = pd.read_csv(metrics_file)
                        df.loc[df['feature_name'] == best_feature, 'selected'] = 1
                        df.to_csv(metrics_file, index=False)
                    
                    selected_features.append(best_feature)
                    current_features = selected_features.copy()
                    candidate_features.remove(best_feature)
                    round_num += 1
                else:
                    logger.info("\nNo remaining features improve predictions sufficiently.")
                    break

            # Log final results
            logger.info("\n=== Feature Selection Complete ===")
            total_time = datetime.now() - start_time
            logger.info(f"Total time: {str(total_time).split('.')[0]}")
            main_features = [f for f in selected_features if f not in control_features]
            logger.info(f"Selected {len(main_features)} features:")
            for feat in main_features:
                logger.info(f"  - {feat}")
            
            self.selected_features = selected_features
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in select_features: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            raise

        finally:
            try:
                # Clean up all remaining temp models
                if hasattr(self, '_temp_models'):
                    for model in list(self._temp_models):
                        try:
                            if hasattr(model, 'cleanup'):
                                model.cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up temp model: {e}")
                    self._temp_models.clear()
                
                # Force garbage collection
                gc.collect()
                
                # Clean up main model
                self.cleanup()
                
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")
                                                                                            
    def _calculate_feature_importance_parallel_cv(self, feature: str, dataset: PreferenceDataset, 
                                current_features: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on prediction improvement."""
        logger.info(f"\nCalculating importance for feature: {feature}")
        logger.info(f"Current features: {current_features}")
        
        try:
            # Add memory check at start
            self._check_memory_usage()

            # Verify dataset and feature extractor
            if dataset is None or dataset.feature_extractor is None:
                logger.error("Invalid dataset or missing feature extractor")
                return {
                    'effect_magnitude': 0.0,
                    'effect_std': 0.0,
                    'std_magnitude_ratio': float('inf'),
                    'n_effects': 0,
                    'mean_aligned_effect': 0.0,
                    'consistency_unbounded': 0.0,
                    'consistency_bounded': 0.0,
                    'consistency_capped': 0.0,
                    'consistency_sigmoid': 0.0,
                    'importance_unbounded': 0.0,
                    'importance_bounded': 0.0,
                    'importance_capped': 0.0,
                    'importance_sigmoid': 0.0,
                    'selected_importance': 0.0
                }

            # Use passed dataset directly instead of storing in self
            cv_splits = self._get_cv_splits(dataset, n_splits=5)
            cv_aligned_effects = []
                
            # Process each fold
            for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
                logger.info(f"Processing fold {fold}/5 for {feature}")
                train_data = dataset._create_subset_dataset(train_idx)
                val_data = dataset._create_subset_dataset(val_idx)
                logger.info(f"Train set size: {len(train_data.preferences)} preferences")
                logger.info(f"Validation set size: {len(val_data.preferences)} preferences")
                
                try:
                    # Create and train both models for this fold
                    with self._create_temp_model() as fold_baseline_model, \
                         self._create_temp_model() as fold_feature_model:
                        
                        # Set up baseline model
                        fold_baseline_model.feature_extractor = dataset.feature_extractor
                        fold_baseline_model.selected_features = current_features  # Changed from saved_features
                        fold_baseline_model.feature_names = current_features     # Changed from saved_features
                        
                        # Train baseline model
                        logger.info(f"Training baseline model for fold {fold}")
                        fold_baseline_model.fit(train_data, features=current_features)  # Changed from saved_features
                        
                        # Check if baseline model fitted
                        if not hasattr(fold_baseline_model, 'fit_result'):
                            logger.warning(f"No fit result for baseline model in fold {fold}")
                            continue
                            
                        # Set up and train feature model
                        fold_feature_model.feature_extractor = dataset.feature_extractor
                        fold_feature_model.selected_features = current_features + [feature]  # Changed from saved_features
                        fold_feature_model.feature_names = current_features + [feature]     # Changed from saved_features
                        
                        logger.info(f"Training feature model for fold {fold}")
                        fold_feature_model.fit(train_data, features=current_features + [feature])  # Changed from saved_features
                                                
                        # Check if feature model fitted
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
                finally:
                    # Force cleanup after each fold
                    if hasattr(self, '_feature_data_cache'):
                        self._feature_data_cache.clear()
                    gc.collect()
                    self._check_memory_usage()
                    
            # Calculate metrics if we have effects
            if cv_aligned_effects:
                cv_aligned_effects = np.array(cv_aligned_effects)
                mean_aligned_effect = float(np.mean(cv_aligned_effects))
                effect_std = float(np.std(cv_aligned_effects))
                effect_magnitude = abs(mean_aligned_effect)
                std_magnitude_ratio = effect_std / effect_magnitude if effect_magnitude > 0 else float('inf')
                                
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
                importance_unbounded = effect_magnitude * max(0, consistency_unbounded)
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
                logger.info(f"  sygmoid-bounded: 1/(1 + exp(std/magnitude - 1)) = {consistency_sigmoid:.5f}")
                logger.info(f"  inverse-bounded: 1/(1 + std/magnitude) = {consistency_bounded:.5f}")
                
                logger.info(f"\nImportance scores = effect magnitude multiplied by:")
                logger.info(f"  max(0, unbounded consistency): {importance_unbounded:.5f}")
                logger.info(f"  min-capped consistency: {importance_capped:.5f}")
                logger.info(f"  sigmoid-bounded consistency: {importance_sigmoid:.5f}")
                logger.info(f"  inverse-bounded consistency: {importance_bounded:.5f}")
                
                return {
                    'effect_magnitude': effect_magnitude,
                    'effect_std': effect_std,
                    'std_magnitude_ratio': std_magnitude_ratio,
                    'n_effects': len(cv_aligned_effects),
                    'mean_aligned_effect': mean_aligned_effect,
                    'consistency_unbounded': consistency_unbounded,
                    'consistency_bounded': consistency_bounded,
                    'consistency_capped': consistency_capped,
                    'consistency_sigmoid': consistency_sigmoid,
                    'importance_unbounded': importance_unbounded,
                    'importance_bounded': importance_bounded,
                    'importance_capped': importance_capped,
                    'importance_sigmoid': importance_sigmoid,
                    'selected_importance': importance_bounded  # Keep using bounded for selection
                }
            else:
                logger.warning("No valid effects calculated")
                return {
                    'effect_magnitude': 0.0,
                    'effect_std': 0.0,
                    'std_magnitude_ratio': float('inf'),
                    'n_effects': 0,
                    'mean_aligned_effect': 0.0,
                    'consistency_unbounded': 0.0,
                    'consistency_bounded': 0.0,
                    'consistency_capped': 0.0,
                    'consistency_sigmoid': 0.0,
                    'importance_unbounded': 0.0,
                    'importance_bounded': 0.0,
                    'importance_capped': 0.0,
                    'importance_sigmoid': 0.0,
                    'selected_importance': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {
                'effect_magnitude': 0.0,
                'effect_std': 0.0,
                'std_magnitude_ratio': float('inf'),
                'n_effects': 0,
                'mean_aligned_effect': 0.0,
                'consistency_unbounded': 0.0,
                'consistency_bounded': 0.0,
                'consistency_capped': 0.0,
                'consistency_sigmoid': 0.0,
                'importance_unbounded': 0.0,
                'importance_bounded': 0.0,
                'importance_capped': 0.0,
                'importance_sigmoid': 0.0,
                'selected_importance': 0.0
            }
        finally:
            # Final cleanup
            gc.collect()

    def _calculate_feature_importance(self, feature: str, dataset: PreferenceDataset, 
                                current_features: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on prediction improvement."""
        logger.info(f"\n=== Feature Importance Calculation for '{feature}' ===")
        logger.info(f"Base features: {current_features}")
        logger.info(f"Number of CV folds: 5")
        
        try:
            # Reset numpy seed before each feature calculation
            np.random.seed(self.seeds['numpy'])

            # Verify dataset and feature extractor
            if dataset is None or dataset.feature_extractor is None:
                logger.error("Invalid dataset or missing feature extractor")
                return self._get_default_metrics()

            cv_splits = self._get_cv_splits(dataset, n_splits=5)
            
            # Initialize aggregation variables
            all_effects = []
            total_effect = 0
            n_valid_effects = 0
            
            # Process each fold sequentially
            for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
                logger.info(f"Processing fold {fold}/5 for {feature}")
                
                # Check memory before starting fold
                if not self._check_memory_before_fold():
                    logger.warning(f"Insufficient memory for fold {fold}, skipping...")
                    continue
                
                try:
                    # Create fold datasets
                    train_data = dataset._create_subset_dataset(train_idx)
                    val_data = dataset._create_subset_dataset(val_idx)
                    logger.info(f"│  • Train set: {len(train_data.preferences)} preferences")
                    logger.info(f"│  • Val set: {len(val_data.preferences)} preferences")
                    
                    # Process fold with temporary models
                    with self._create_temp_model() as fold_baseline_model, \
                        self._create_temp_model() as fold_feature_model:
                        
                        # Train baseline model
                        logger.info(f"│  • Training baseline model...")
                        fold_baseline_model.feature_extractor = dataset.feature_extractor
                        fold_baseline_model.selected_features = current_features
                        fold_baseline_model.feature_names = current_features
                        fold_baseline_model.fit(train_data, features=current_features)
                        
                        # Check memory after baseline model
                        if not self._check_memory_before_fold():
                            logger.warning("Memory high after baseline model, skipping feature model")
                            continue
                        
                        # Train feature model
                        logger.info(f"│  • Training feature model...")
                        fold_feature_model.feature_extractor = dataset.feature_extractor
                        fold_feature_model.selected_features = current_features + [feature]
                        fold_feature_model.feature_names = current_features + [feature]
                        fold_feature_model.fit(train_data, features=current_features + [feature])
                        
                        # Calculate effects for validation set
                        logger.info(f"│  • Calculating validation effects...")
                        fold_effects = []
                        for pref in val_data.preferences:
                            try:
                                # Calculate effect for each preference
                                base_pred = fold_baseline_model.predict_preference(pref.bigram1, pref.bigram2)
                                feat_pred = fold_feature_model.predict_preference(pref.bigram1, pref.bigram2)
                                
                                # Convert to logits and calculate effect
                                base_logit = np.log(base_pred.probability / (1 - base_pred.probability))
                                feat_logit = np.log(feat_pred.probability / (1 - feat_pred.probability))
                                effect = feat_logit - base_logit
                                aligned_effect = effect if pref.preferred else -effect
                                
                                fold_effects.append(aligned_effect)
                                
                            except Exception as e:
                                logger.debug(f"│    Error processing preference: {str(e)}")
                                continue
                        
                        # Aggregate fold results
                        if fold_effects:
                            mean_fold_effect = np.mean(fold_effects)
                            total_effect += mean_fold_effect
                            all_effects.extend(fold_effects)
                            n_valid_effects += len(fold_effects)
                            
                            logger.info(f"│  • Fold {fold} complete: {len(fold_effects)} effects, mean = {mean_fold_effect:.4f}")
                    
                except Exception as e:
                    logger.error(f"│  ✗ Error in fold {fold}: {str(e)}")
                    continue
                    
                finally:
                    # Clean up fold resources
                    if hasattr(self, '_feature_data_cache'):
                        self._feature_data_cache.clear()
                    gc.collect()
                    
                    # Brief pause between folds
                    time.sleep(2)
                
                # Visual separator between folds
                if fold < 5:
                    logger.info("├─────────────────────────────────────────────")
                else:
                    logger.info("└─────────────────────────────────────────────")

            # Calculate final metrics from aggregated data
            if all_effects:
                all_effects = np.array(all_effects)
                mean_effect = np.mean(all_effects)
                effect_std = np.std(all_effects)
                effect_magnitude = abs(mean_effect)
                
                # Calculate consistency metrics
                std_magnitude_ratio = effect_std / effect_magnitude if effect_magnitude > 0 else float('inf')
                consistency = 1 / (1 + std_magnitude_ratio)  # Bounded [0,1]
                importance = effect_magnitude * consistency
                
                logger.info("\nFinal metrics:")
                logger.info(f"• Total valid effects: {n_valid_effects}")
                logger.info(f"• Mean effect: {mean_effect:.4f}")
                logger.info(f"• Effect magnitude: {effect_magnitude:.4f}")
                logger.info(f"• Effect std dev: {effect_std:.4f}")
                logger.info(f"• Consistency score: {consistency:.4f}")
                logger.info(f"• Importance score: {importance:.4f}")
                
                return {
                    'effect_magnitude': effect_magnitude,
                    'effect_std': effect_std,
                    'std_magnitude_ratio': std_magnitude_ratio,
                    'mean_aligned_effect': mean_effect,
                    'consistency': consistency,
                    'selected_importance': importance,
                    'n_effects': n_valid_effects
                }
                
            return self._get_default_metrics()
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics dictionary with zero values."""
        return {
            'effect_magnitude': 0.0,
            'effect_std': 0.0,
            'std_magnitude_ratio': float('inf'),
            'mean_aligned_effect': 0.0,
            'consistency': 0.0,
            'selected_importance': 0.0,
            'n_effects': 0
        }
        
    #--------------------------------------------
    # Cross-validation methods
    #--------------------------------------------
    def _get_cv_splits(self, dataset: PreferenceDataset, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits preserving participant structure with validation."""
        # Use specific CV seed
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seeds['cv'])
        
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
            'interaction_metadata': self.interaction_metadata,
            'feature_importance_metrics': self.feature_importance_metrics,
            'feature_extractor': self.feature_extractor,
            'feature_stats': getattr(self, 'feature_stats', {}),
            'is_fitted': self._is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def _convert_legacy_config(config_obj):
        """Temporary helper to convert old saved Config objects to dictionary format."""
        try:
            # Manual conversion of the nested structure
            return {
                'paths': dict(config_obj.paths),
                'model': dict(config_obj.model),
                'feature_selection': dict(config_obj.feature_selection),
                'features': dict(config_obj.features),
                'data': dict(config_obj.data),
                'recommendations': dict(config_obj.recommendations),
                'logging': dict(config_obj.logging),
                'visualization': dict(config_obj.visualization)
            }
        except AttributeError:
            # If already a dict, return as is
            return config_obj

    @classmethod
    def load(cls, path: Path) -> 'PreferenceModel':
        """Load model state from file."""
        logger.info("=== Loading model state ===")
        try:
            with open(path, 'rb') as f:
                save_dict = pickle.load(f)
            
            # Convert legacy config format
            config_data = cls._convert_legacy_config(save_dict['config'])
            config = Config(**config_data)
            
            model = cls(config=config)

            try:
                # Core model attributes
                model.feature_names = save_dict['feature_names']
                model.selected_features = save_dict['selected_features']
                model.feature_weights = save_dict['feature_weights']
                model.fit_result = save_dict['fit_result']
                model.interaction_metadata = save_dict['interaction_metadata']
                model.feature_importance_metrics = save_dict.get('feature_importance_metrics', {})
                
                # Feature extraction related attributes
                model.feature_extractor = save_dict.get('feature_extractor')
                model.feature_stats = save_dict.get('feature_stats', {})
                
                # Model state
                model._is_fitted = save_dict.get('is_fitted', False)
                
                if model.feature_extractor is None:
                    logger.warning("No feature extractor found in saved model state")
                
                return model
            except Exception as e:
                logger.error(f"Error restoring model attributes: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
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
