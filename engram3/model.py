# model.py
"""
Hierarchical Bayesian preference learning model implementation.

Uses Bradley-Terry structure to model typing preferences through:
  - Feature-based comfort scores
  - Participant-level random effects
  - Full posterior uncertainty quantification

Handles feature selection, model training, and prediction:
  - Models latent comfort scores directly through features
  - Includes participant random effects
  - Uses proper Bayesian inference for uncertainty estimation
  - Maintains transitivity through Bradley-Terry structure
  - Provides relative preference predictions between bigram pairs
  - Supports feature importance analysis and interaction detection
"""
import cmdstanpy
import numpy as np
import cmdstanpy 
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import logging

from engram3.data import PreferenceDataset
from engram3.features.visualization import FeatureMetricsVisualizer
from engram3.features.extraction import extract_bigram_features

logger = logging.getLogger(__name__)

class PreferenceModel:

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.fit = None
        self.feature_names = None
        self.selected_features = []
        self.dataset = None  # Add dataset attribute
        
        # Initialize output directory from config
        if config and 'data' in config and 'output_dir' in config['data']:
            self.output_dir = Path(config['data']['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path('output')
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize visualizer with proper output directory
        self.visualizer = FeatureMetricsVisualizer(str(self.output_dir)) if config else None
        
        try:
            # Keep original Stan initialization that was working
            model_path = Path(__file__).parent / "models" / "preference_model.stan"
            if not model_path.exists():
                raise FileNotFoundError(f"Stan model file not found: {model_path}")
            
            logger.info(f"Loading Stan model from {model_path}")
            
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

    def _extract_features(self, bigram: str) -> Dict[str, float]:
        """
        Extract features for a bigram using dataset's feature extraction configuration.
        """
        if not self.dataset:
            raise ValueError("Dataset not initialized. Call fit_model first.")
            
        if len(bigram) != 2:
            raise ValueError(f"Bigram must be exactly 2 characters, got {bigram}")
        
        try:
            # Use dataset's precomputed features if available
            if hasattr(self.dataset, 'all_bigram_features') and bigram in self.dataset.all_bigram_features:
                return self.dataset.all_bigram_features[bigram]
                    
            # Extract individual characters as strings
            char1, char2 = str(bigram[0]), str(bigram[1])
            
            # Special handling for same-character bigrams
            if char1 == char2:
                logger.debug(f"Processing same-character bigram: {bigram}")
                
            # Ensure we have the necessary maps
            required_maps = ['column_map', 'row_map', 'finger_map', 
                            'engram_position_values', 'row_position_values']
            for map_name in required_maps:
                if not hasattr(self.dataset, map_name):
                    raise ValueError(f"Dataset missing required map: {map_name}")
            
            features = extract_bigram_features(
                char1, char2,
                self.dataset.column_map,
                self.dataset.row_map,
                self.dataset.finger_map,
                self.dataset.engram_position_values,
                self.dataset.row_position_values
            )
            
            # Validate features
            if not features:
                raise ValueError(f"No features extracted for bigram {bigram}")
                
            return features
                
        except Exception as e:
            logger.error(f"Error extracting features for bigram {bigram}: {str(e)}")
            # Return empty features rather than raising to avoid breaking visualization
            # Use selected_features if available, otherwise return empty dict
            if hasattr(self, 'selected_features') and self.selected_features:
                return {feature: 0.0 for feature in self.selected_features}
            return {}
            
    def fit_model(self, dataset: PreferenceDataset, features: Optional[List[str]] = None) -> None:
        """Fit the model to data."""
        try:
            # Store dataset reference
            self.dataset = dataset
            
            # Rest of the existing fit_model code...
            stan_data = self.prepare_data(dataset, features)
            
            logger.info(f"Preparing Stan data:")
            logger.info(f"N (preferences): {stan_data['N']}")
            logger.info(f"P (participants): {stan_data['P']}")
            logger.info(f"F (features): {stan_data['F']}")
            logger.info(f"Selected features: {self.feature_names}")
            
            self.fit = self.model.sample(
                data=stan_data,
                chains=self.config.get('model', {}).get('chains', 4),
                iter_warmup=self.config.get('model', {}).get('warmup', 1000),
                iter_sampling=self.config.get('model', {}).get('samples', 1000),
                show_progress=True,
                refresh=50,
                adapt_delta=0.9,
                max_treedepth=12,
                show_console=True
            )
            
            # Check MCMC diagnostics
            self._check_diagnostics()

            self.feature_weights = {}
            beta = self.fit.stan_variable('beta')
            for i, feature in enumerate(self.feature_names):
                self.feature_weights[feature] = (
                    float(np.mean(beta[:, i])),  # mean
                    float(np.std(beta[:, i]))    # std
                )
                
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def select_features(self, dataset: PreferenceDataset, 
                        initial_features: Optional[List[str]] = None) -> List[str]:
        """Select features with visualization support."""
        # Store dataset reference
        self.dataset = dataset
        
        logger.info("Starting feature selection...")
        
        # Initialize with base features
        self.selected_features = initial_features or self.config['features']['base_features']
        logger.info(f"Initial features: {self.selected_features}")
                
        # Initialize feature metrics visualizer
        metrics_history = []

        # Get all possible features from dataset
        all_features = dataset.get_feature_names()
        
        iteration = 0
        while iteration < self.config.get('feature_selection', {}).get('n_iterations', 10):
            # Fit model with current features
            self.fit_model(dataset, self.selected_features)
            
            # Get current model performance metrics
            base_metrics = self._compute_model_metrics()
            
            # Evaluate each potential new feature
            feature_scores = {}
            feature_metrics = {}
            for feature in all_features:
                if feature not in self.selected_features:
                    # Calculate feature importance metrics
                    metrics = self.calculate_feature_metrics(dataset, feature)
                    feature_metrics[feature] = metrics
                    
                    importance = self._evaluate_feature_importance(
                        dataset, feature, self.selected_features)
                    
                    feature_scores[feature] = importance
                    logger.debug(f"Feature {feature} importance: {importance:.3f}")
            
            # Generate visualization of current feature metrics
            if feature_metrics:  # Only if we have features to evaluate
                fig = visualizer.plot_feature_metrics(feature_metrics)
                fig.savefig(self.output_dir / f'feature_metrics_iter_{iteration}.png')
                plt.close(fig)
            
            if not feature_scores:
                logger.info("No more features to evaluate")
                break
                    
            # Select best feature if it meets threshold
            best_feature, best_score = max(feature_scores.items(), 
                                        key=lambda x: x[1])
            
            threshold = self.config.get('feature_selection', {}).get(
                'importance_threshold', 0.1)
            
            if best_score < threshold:
                logger.info("No remaining features meet importance threshold")
                break
                    
            self.selected_features.append(best_feature)
            logger.info(f"Added feature: {best_feature} (score: {best_score:.3f})")
            
            # Store metrics for history
            metrics_history.append({
                'iteration': iteration,
                'feature': best_feature,
                'score': best_score,
                **base_metrics,
                **feature_metrics[best_feature]
            })
            
            # Check if we should add interaction with existing features
            self._add_interactions(dataset, self.selected_features, best_feature)
        
            # Add visualization
            if self.visualizer:
                metrics = {
                    'n_features': len(self.selected_features),
                    'accuracy': base_metrics['accuracy'],
                    'transitivity_score': base_metrics['transitivity'],
                    'feature_score': best_score,
                    'n_interactions': len([f for f in self.selected_features if '_x_' in f])
                }
                self.visualizer.save_iteration(iteration, self, dataset, metrics)
            
            iteration += 1
        
        # Generate final evolution plot
        if metrics_history:
            fig = visualizer.plot_feature_evolution(metrics_history)
            fig.savefig(self.output_dir / 'feature_evolution.png')
            plt.close(fig)
            
        logger.info(f"Feature selection completed with {len(self.selected_features)} features")
        return self.selected_features
 
    def prepare_data(self, dataset: PreferenceDataset, features: Optional[List[str]] = None) -> Dict:
        """Prepare data for Stan model with proper validation and cleaning."""
        # Get feature names
        self.feature_names = features if features is not None else dataset.get_feature_names()
        
        # Log feature information
        logger.info(f"Preparing data with {len(self.feature_names)} features:")
        for f in self.feature_names:
            logger.info(f"  - {f}")
        
        # Create participant ID mapping
        participant_ids = sorted(list(dataset.participants))
        participant_map = {pid: i+1 for i, pid in enumerate(participant_ids)}
        
        # Prepare data matrices
        X1 = []  # features for first bigram
        X2 = []  # features for second bigram
        y = []   # preferences (1 if first bigram preferred)
        participant = []  # participant IDs
        
        skipped_count = 0
        for pref in dataset.preferences:
            try:
                # Extract feature vectors
                feat1 = [pref.features1.get(f, 0.0) for f in self.feature_names]
                feat2 = [pref.features2.get(f, 0.0) for f in self.feature_names]
                
                # Check for null/nan values
                if any(pd.isna(x) for x in feat1) or any(pd.isna(x) for x in feat2):
                    skipped_count += 1
                    continue
                    
                X1.append(feat1)
                X2.append(feat2)
                y.append(1 if pref.preferred else 0)
                participant.append(participant_map[pref.participant_id])
                
            except KeyError as e:
                logger.warning(f"Missing feature in preference: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} preferences due to missing or invalid features")
        
        # Convert to numpy arrays
        X1 = np.array(X1, dtype=np.float64)
        X2 = np.array(X2, dtype=np.float64)
        y = np.array(y, dtype=np.int32)
        participant = np.array(participant, dtype=np.int32)
        
        # Validate arrays
        if X1.size == 0 or X2.size == 0:
            raise ValueError("No valid data points after cleaning")
            
        if np.any(np.isnan(X1)) or np.any(np.isnan(X2)):
            raise ValueError("NaN values found in feature matrices after cleaning")
        
        # Log data dimensions
        logger.info(f"Data dimensions:")
        logger.info(f"  N (preferences): {len(y)}")
        logger.info(f"  P (participants): {len(participant_ids)}")
        logger.info(f"  F (features): {len(self.feature_names)}")
        logger.info(f"  X1 shape: {X1.shape}")
        logger.info(f"  X2 shape: {X2.shape}")
        
        # Log summary statistics
        logger.info("\nFeature summary statistics:")
        for i, feature in enumerate(self.feature_names):
            logger.info(f"{feature}:")
            logger.info(f"  X1 - mean: {np.mean(X1[:, i]):.3f}, std: {np.std(X1[:, i]):.3f}")
            logger.info(f"  X2 - mean: {np.mean(X2[:, i]):.3f}, std: {np.std(X2[:, i]):.3f}")
        
        # Prepare Stan data
        stan_data = {
            'N': len(y),
            'P': len(participant_ids),
            'F': len(self.feature_names),
            'X1': X1,
            'X2': X2,
            'participant': participant,
            'y': y,
            'feature_scale': self.config.get('model', {}).get('feature_scale', 2.0),
            'participant_scale': self.config.get('model', {}).get('participant_scale', 1.0)
        }
        
        return stan_data
                
    def calculate_feature_metrics(self, 
                                dataset: PreferenceDataset,
                                feature: str) -> Dict[str, float]:
        """
        Calculate comprehensive feature metrics including correlation
        and mutual information.
        """
        try:
            # Extract feature differences and preferences
            feature_diffs = []
            preferences = []
            
            for pref in dataset.preferences:
                # Calculate feature difference
                value1 = pref.features1.get(feature, 0.0)
                value2 = pref.features2.get(feature, 0.0)
                feature_diffs.append(value1 - value2)
                
                # Convert preference to numeric
                preferences.append(1.0 if pref.preferred else -1.0)
                
            feature_diffs = np.array(feature_diffs)
            preferences = np.array(preferences)
            
            # Calculate correlation
            correlation = self._calculate_correlation(feature_diffs, preferences)
            
            # Calculate mutual information
            mi_score = self._calculate_mutual_information(feature_diffs, preferences)
            
            # Get model effect if available
            if hasattr(self, 'fit') and self.fit is not None:
                weights = self.get_feature_weights()
                if feature in weights:
                    effect_mean, effect_std = weights[feature]
                else:
                    effect_mean, effect_std = 0.0, 0.0
            else:
                effect_mean, effect_std = 0.0, 0.0
            
            metrics = {
                'feature_name': feature,
                'correlation': correlation,
                'mutual_information': mi_score,
                'model_effect_mean': effect_mean,
                'model_effect_std': effect_std,
                'combined_score': self._calculate_combined_score(
                    correlation, mi_score, effect_mean, effect_std)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for feature {feature}: {str(e)}")
            return {
                'feature_name': feature,
                'correlation': 0.0,
                'mutual_information': 0.0,
                'model_effect_mean': 0.0,
                'model_effect_std': 0.0,
                'combined_score': 0.0
            }

    def _calculate_correlation(self, 
                            feature_diffs: np.ndarray, 
                            preferences: np.ndarray) -> float:
        """
        Calculate Spearman correlation between feature differences and preferences.
        Uses Spearman's rho as it captures monotonic relationships and is robust
        to outliers.
        """
        try:
            # Remove any pairs with NaN values
            mask = ~(np.isnan(feature_diffs) | np.isnan(preferences))
            if not np.any(mask):
                return 0.0
                
            feature_diffs = feature_diffs[mask]
            preferences = preferences[mask]
            
            if len(feature_diffs) < 2:  # Need at least 2 points for correlation
                return 0.0
                
            correlation, _ = spearmanr(feature_diffs, preferences)
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating correlation: {str(e)}")
            return 0.0

    def _calculate_mutual_information(self,
                                    feature_diffs: np.ndarray,
                                    preferences: np.ndarray,
                                    n_bins: int = 20) -> float:
        """
        Calculate mutual information between feature differences and preferences.
        Discretizes continuous feature differences using binning.
        """
        try:
            # Remove any pairs with NaN values
            mask = ~(np.isnan(feature_diffs) | np.isnan(preferences))
            if not np.any(mask):
                return 0.0
                
            feature_diffs = feature_diffs[mask]
            preferences = preferences[mask]
            
            if len(feature_diffs) < 2:  # Need at least 2 points
                return 0.0
                
            # Discretize feature differences
            bins = np.linspace(np.min(feature_diffs), np.max(feature_diffs), n_bins)
            binned_diffs = np.digitize(feature_diffs, bins)
            
            # Calculate mutual information
            mi = mutual_info_score(binned_diffs, preferences)
            return float(mi)
            
        except Exception as e:
            logger.warning(f"Error calculating mutual information: {str(e)}")
            return 0.0

    def _calculate_combined_score(self,
                                correlation: float,
                                mutual_info: float,
                                effect_mean: float,
                                effect_std: float) -> float:
        """
        Calculate combined importance score using correlation, mutual information,
        and model effect.
        """
        try:
            # Get weights from config
            weights = self.config.get('feature_evaluation', {}).get('importance_weights', {
                'model_effect': 0.4,
                'correlation': 0.3,
                'mutual_info': 0.3
            })
            
            # Normalize effect size by its uncertainty
            effect_score = abs(effect_mean) / (effect_std + 1e-10)  # Add small constant to avoid division by zero
            
            # Calculate weighted score
            score = (
                weights['model_effect'] * effect_score +
                weights['correlation'] * abs(correlation) +
                weights['mutual_info'] * mutual_info
            )
            
            return float(score)
            
        except Exception as e:
            logger.warning(f"Error calculating combined score: {str(e)}")
            return 0.0
        
    def _check_diagnostics(self) -> None:
        """Check MCMC diagnostics with proper type handling."""
        try:
            # Get summary dataframe
            summary_df = self.fit.summary()
            
            # Log column names for debugging
            logger.debug(f"Available columns in summary: {list(summary_df.columns)}")
            
            # Check R-hat (should be close to 1)
            rhat_col = next((col for col in summary_df.columns if 'rhat' in col.lower()), None)
            if rhat_col:
                rhat = summary_df[rhat_col]
                if np.any(rhat > 1.1):
                    logger.warning("Some parameters have R-hat > 1.1, indicating potential convergence issues")
                    problematic = summary_df[rhat > 1.1].index
                    logger.warning(f"Parameters with high R-hat: {problematic}")
            
            # Check effective sample size
            ess_col = next((col for col in summary_df.columns if 'ess' in col.lower()), None)
            if ess_col:
                n_eff = summary_df[ess_col]
                if np.any(n_eff < 100):
                    logger.warning("Some parameters have low effective sample size")
                    problematic = summary_df[n_eff < 100].index
                    logger.warning(f"Parameters with low ESS: {problematic}")
            
            # Add additional diagnostics
            logger.info("\nSampling diagnostics:")
            logger.info(f"Number of chains: {self.fit.chains}")
            # Properly get number of draws
            n_draws = self.fit.draws_pd().shape[0] if hasattr(self.fit, 'draws_pd') else None
            logger.info(f"Samples per chain: {n_draws // self.fit.chains if n_draws else 'Unknown'}")
            logger.info(f"Total samples: {n_draws if n_draws else 'Unknown'}")
            
            if hasattr(self.fit, 'time'):
                logger.info("\nTiming information:")
                timing = self.fit.time()
                if isinstance(timing, dict):
                    logger.info(f"Warmup time: {timing.get('warmup', 0):.1f} seconds")
                    logger.info(f"Sampling time: {timing.get('sampling', 0):.1f} seconds")
                
        except Exception as e:
            logger.warning(f"Error checking diagnostics: {str(e)}")
            logger.warning("Continuing without full diagnostic checks")

    def _calculate_predictive_impact(self, dataset: PreferenceDataset,
                                candidate_feature: str) -> float:
        """
        Calculate how much a feature improves predictive performance
        using posterior predictive checks.
        """
        try:
            # Get posterior predictive probabilities with feature
            p_pred_with = self.fit.stan_variable('p_pred')
            elpd_with = np.mean(np.log(p_pred_with))
            
            # Approximate ELPD without feature using importance sampling
            # First get base model predictions
            base_features = [f for f in self.selected_features if f != candidate_feature]
            if not base_features:
                return 0.0  # No comparison possible for first feature
                
            # Fit temporary model without candidate feature
            temp_model = PreferenceModel(self.config)
            temp_model.fit_model(dataset, base_features)
            
            if not hasattr(temp_model, 'fit') or temp_model.fit is None:
                logger.warning(f"Could not fit comparison model for feature {candidate_feature}")
                return 0.0
                
            p_pred_without = temp_model.fit.stan_variable('p_pred')
            elpd_without = np.mean(np.log(p_pred_without))
            
            # Calculate improvement
            improvement = elpd_with - elpd_without
            
            logger.debug(f"Predictive impact for {candidate_feature}: {improvement:.3f}")
            return float(improvement)
            
        except Exception as e:
            logger.warning(f"Error calculating predictive impact: {str(e)}")
            return 0.0
            
    def get_feature_weights(self) -> Dict[str, Tuple[float, float]]:
        """Get posterior means and standard deviations of feature weights."""
        if not hasattr(self, 'feature_weights'):
            raise RuntimeError("Model must be fit before getting weights")
        return self.feature_weights
    
    def _compute_model_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive model performance metrics.
        
        Returns:
            Dictionary containing:
            - accuracy: Classification accuracy on preferences
            - transitivity: Score for transitivity preservation
            - auc: Area under ROC curve
            - log_likelihood: Model log likelihood
            - feature_sparsity: Proportion of non-zero feature weights
            - uncertainty: Average predictive uncertainty
        """
        if not self.fit:
            raise RuntimeError("Model must be fit before computing metrics")
            
        try:
            metrics = {}
            
            # Get predictions for all preferences
            y_true = []
            y_pred = []
            uncertainties = []
            
            for pref in self.dataset.preferences:
                pred_prob, pred_std = self.predict_preference(
                    pref.bigram1, pref.bigram2)
                
                y_true.append(1.0 if pref.preferred else 0.0)
                y_pred.append(pred_prob)
                uncertainties.append(pred_std)
            
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred > 0.5)
            metrics['auc'] = roc_auc_score(y_true, y_pred)
            
            # Transitivity score
            transitivity_results = self.dataset.check_transitivity()
            metrics['transitivity'] = 1.0 - transitivity_results['violation_rate']
            
            # Model uncertainty
            metrics['uncertainty'] = float(np.mean(uncertainties))
            
            # Feature sparsity (proportion of meaningful weights)
            weights = self.get_feature_weights()
            weight_vals = np.array([w for w, _ in weights.values()])
            metrics['feature_sparsity'] = float(np.mean(np.abs(weight_vals) > 0.1))
            
            # Log likelihood
            if hasattr(self.fit, 'log_likelihood'):
                metrics['log_likelihood'] = float(self.fit.log_likelihood())
            
            # Additional Stan-specific diagnostics if available
            if hasattr(self.fit, 'summary'):
                summary = self.fit.summary()
                metrics['r_hat_max'] = float(summary['r_hat'].max())
                metrics['n_eff_min'] = float(summary['n_eff'].min())
                
            logger.debug("Computed model metrics:")
            for name, value in metrics.items():
                logger.debug(f"{name}: {value:.3f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            logger.debug("Traceback:", exc_info=True)
            return {
                'accuracy': 0.0,
                'transitivity': 0.0,
                'auc': 0.5,
                'uncertainty': 1.0,
                'feature_sparsity': 0.0,
                'log_likelihood': -np.inf
            }

    def predict_preference(self, bigram1: str, bigram2: str) -> Tuple[float, float]:
        """
        Predict preference probability and uncertainty for a bigram pair.
        """
        try:
            # Validate inputs
            if not isinstance(bigram1, str) or not isinstance(bigram2, str):
                raise ValueError(f"Bigrams must be strings, got {type(bigram1)} and {type(bigram2)}")
                
            if len(bigram1) != 2 or len(bigram2) != 2:
                raise ValueError(f"Bigrams must be exactly 2 characters, got '{bigram1}' and '{bigram2}'")
                
            # Get comfort scores for both bigrams
            score1, unc1 = self.get_bigram_comfort_scores(bigram1)
            score2, unc2 = self.get_bigram_comfort_scores(bigram2)
            
            # Safely get number of samples
            if not hasattr(self.fit, 'chains') or not hasattr(self.fit, 'draws'):
                n_samples = 1000  # Default fallback
            else:
                try:
                    n_samples = self.fit.chains * self.fit.draws
                except TypeError:
                    n_samples = 1000  # Fallback if multiplication fails
            
            # Get samples from posterior
            samples = np.random.normal(
                loc=score1 - score2,
                scale=np.sqrt(unc1**2 + unc2**2),
                size=n_samples
            )
            
            # Transform to probabilities
            probs = 1 / (1 + np.exp(-samples))
            
            return float(np.mean(probs)), float(np.std(probs))
            
        except Exception as e:
            logger.error(f"Error predicting preference: {str(e)}")
            return 0.5, 1.0
                
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
              
    def _evaluate_feature_importance(self, dataset: PreferenceDataset,
                                candidate_feature: str,
                                current_features: List[str]) -> float:
        """
        Enhanced feature importance evaluation incorporating all metrics.
        """
        try:
            # Get comprehensive metrics
            metrics = self.calculate_feature_metrics(dataset, candidate_feature)
            
            # Get weights from config
            weights = self.config.get('feature_selection', {}).get('metric_weights', {
                'correlation': 0.2,
                'mutual_information': 0.2,
                'model_effect': 0.4,
                'stability': 0.2
            })
            
            # Calculate normalized scores
            correlation_score = abs(metrics['correlation'])
            mi_score = metrics['mutual_information']
            
            # Model effect score (normalized by uncertainty)
            effect_score = abs(metrics['model_effect_mean']) / (metrics['model_effect_std'] + 1e-10)
            
            # Calculate stability score
            stability_metrics = self._calculate_stability_metrics(dataset, candidate_feature)
            stability_score = (
                (1.0 - stability_metrics['effect_cv']) +  # Lower CV is better
                stability_metrics['sign_consistency'] +    # Higher consistency is better
                (1.0 - stability_metrics['relative_range'])  # Lower range is better
            ) / 3.0
            
            # Combine scores
            importance_score = (
                weights['correlation'] * correlation_score +
                weights['mutual_information'] * mi_score +
                weights['model_effect'] * effect_score +
                weights['stability'] * stability_score
            )
            
            # Log detailed scores
            logger.debug(f"\nDetailed scores for {candidate_feature}:")
            logger.debug(f"Correlation score: {correlation_score:.3f}")
            logger.debug(f"Mutual information score: {mi_score:.3f}")
            logger.debug(f"Effect score: {effect_score:.3f}")
            logger.debug(f"Stability score: {stability_score:.3f}")
            logger.debug(f"Final importance score: {importance_score:.3f}")
            
            return float(importance_score)
            
        except Exception as e:
            logger.error(f"Error evaluating feature importance: {str(e)}")
            return 0.0
        
    def _add_interactions(self, dataset: PreferenceDataset,
                        selected_features: List[str],
                        new_feature: str) -> None:
        """
        Evaluate and potentially add interactions with new feature.
        """
        # Only consider interactions with base features
        base_features = [f for f in selected_features 
                        if '_x_' not in f and f != new_feature]
        
        for feature in base_features:
            interaction = f"{new_feature}_x_{feature}"
            
            # Evaluate interaction importance
            importance = self._evaluate_feature_importance(
                dataset, interaction, selected_features)
            
            # Add if meets threshold
            threshold = self.config.get('feature_selection', {}).get(
                'interaction_threshold', 0.15)
            
            if importance >= threshold:
                selected_features.append(interaction)
                logger.info(
                    f"Added interaction: {interaction} (score: {importance:.3f})")

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
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get the learned feature weights including interactions."""
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before getting weights")
        return self.feature_weights.copy()
  
    def cross_validate(self, dataset: PreferenceDataset, n_splits: Optional[int] = None):
        """Perform cross-validation with multiple validation strategies."""
        feature_names = dataset.get_feature_names()
        logger.debug(f"Features for cross-validation (including interactions): {feature_names}")
        
        metrics = defaultdict(list)
        feature_effects = defaultdict(list)
        
        # Get CV splits using shared method
        for fold, (train_idx, val_idx) in enumerate(self._get_cv_splits(dataset, n_splits)):
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
        stability_metrics = {}
        for feature in feature_names:
            stability_metrics[feature] = self._calculate_stability_metrics(dataset, feature)

        # Determine selected features using Bayesian criteria
        selected_features = self._select_features_bayesian(
            importance_metrics, stability_metrics)
        
        # Log results
        self._log_feature_selection_results(
            selected_features, importance_metrics)
        
        # Save metrics to CSV
        if self.visualizer:
            self.visualizer.save_metrics_report(
                metrics_dict={feature: {
                    **processed_effects.get(feature, {}),
                    **importance_metrics.get(feature, {}),
                    **stability_metrics.get(feature, {})
                } for feature in feature_names},
                output_file=self.config['feature_evaluation']['metrics_file']
            )
        
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
    
    def _calculate_stability_metrics(self, dataset: PreferenceDataset,
                                feature: str) -> Dict[str, float]:
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
            n_splits = self.config.get('model', {}).get('cross_validation', {}).get('n_splits', 5)
            
            # Perform cross-validation using shared splitting method
            cv_effects = []
            
            # Get train/val splits using common method
            for train_idx, val_idx in self._get_cv_splits(dataset, n_splits):
                # Create train dataset
                train_data = dataset._create_subset_dataset(train_idx)
                
                # Fit model on training data
                self.fit_model(train_data, self.selected_features + [feature])
                
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
                
    def _get_cv_splits(self, dataset: PreferenceDataset, n_splits: Optional[int] = None) -> GroupKFold:
        """Get cross-validation splits while preserving participant grouping.
        
        Args:
            dataset: PreferenceDataset to split
            n_splits: Number of CV folds, defaults to config value if not specified
            
        Returns:
            Iterator of train/validation indices from GroupKFold
        """
        if n_splits is None:
            n_splits = self.config.get('model', {}).get('cross_validation', {}).get('n_splits', 5)
        
        participant_ids = np.array([p.participant_id for p in dataset.preferences])
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(np.zeros(len(participant_ids)), groups=participant_ids)


class FeatureVisualization:
    """Handles visualization and tracking of feature selection process."""
    
    def __init__(self, config: Dict):
        self.output_dir = Path(config['data']['output_dir']) / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'iterations': self.output_dir / 'iterations',
            'features': self.output_dir / 'features',
            'performance': self.output_dir / 'performance',
            'space': self.output_dir / 'feature_space'
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)
            
        self.iteration_metrics = defaultdict(list)

    def save_iteration(self, iteration: int, model: 'PreferenceModel',
                      dataset: PreferenceDataset, metrics: Dict[str, float]):
        """Save metrics and generate visualizations for current iteration."""
        self._save_metrics(iteration, metrics)
        self._plot_feature_space(iteration, model, dataset)
        self._plot_feature_impacts(iteration, model)
        self._plot_performance_tracking()
        self._update_tracking_plots()

    def _save_metrics(self, iteration: int, metrics: Dict[str, float]):
        metrics['iteration'] = iteration
        df = pd.DataFrame([metrics])
        file_path = self.dirs['iterations'] / f'iteration_{iteration}_metrics.csv'
        df.to_csv(file_path, index=False)
        
        for key, value in metrics.items():
            self.iteration_metrics[key].append(value)

    def plot_feature_space(self, model: 'PreferenceModel', 
                          dataset: PreferenceDataset,
                          title: str = "Feature Space"):
        """Plot 2D PCA of feature space with comfort scores."""
        feature_vectors = []
        comfort_scores = []
        uncertainties = []
        
        for pref in dataset.preferences:
            feat1 = [pref.features1[f] for f in model.selected_features]
            feat2 = [pref.features2[f] for f in model.selected_features]
            
            score1, unc1 = model.get_bigram_comfort_scores(pref.bigram1)
            score2, unc2 = model.get_bigram_comfort_scores(pref.bigram2)
            
            feature_vectors.extend([feat1, feat2])
            comfort_scores.extend([score1, score2])
            uncertainties.extend([unc1, unc2])
            
        X = StandardScaler().fit_transform(feature_vectors)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                           c=comfort_scores,
                           s=np.array(uncertainties) * 500,
                           alpha=0.6,
                           cmap='RdYlBu')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.colorbar(scatter, label='Comfort Score')
        plt.title(title)
        plt.tight_layout()
        
        return fig

    def plot_feature_impacts(self, model: 'PreferenceModel'):
        """Plot feature weights and their interactions."""
        weights = model.get_feature_weights()
        
        # Feature weights plot
        fig1 = plt.figure(figsize=(12, 6))
        sns.barplot(x=list(weights.keys()), y=list(weights.values()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Weights')
        plt.tight_layout()
        
        # Interaction heatmap
        interaction_features = [f for f in model.selected_features if '_x_' in f]
        if interaction_features:
            fig2 = plt.figure(figsize=(10, 8))
            interaction_matrix = np.zeros((len(model.selected_features), 
                                        len(model.selected_features)))
            for i, f1 in enumerate(model.selected_features):
                for j, f2 in enumerate(model.selected_features):
                    interaction = f"{f1}_x_{f2}"
                    if interaction in weights:
                        interaction_matrix[i, j] = abs(weights[interaction])
            
            sns.heatmap(interaction_matrix, 
                       xticklabels=model.selected_features,
                       yticklabels=model.selected_features,
                       cmap='YlOrRd')
            plt.title('Feature Interactions')
            plt.tight_layout()
            return fig1, fig2
        
        return fig1, None

    def plot_performance_history(self):
        """Plot performance metrics over iterations."""
        metrics_df = pd.DataFrame(self.iteration_metrics)
        
        fig = plt.figure(figsize=(12, 6))
        for col in metrics_df.columns:
            if col != 'iteration':
                plt.plot(metrics_df['iteration'], metrics_df[col], 
                        label=col, marker='o')
        
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
