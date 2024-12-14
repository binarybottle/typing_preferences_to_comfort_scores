# engram3/features/importance.py
"""
Feature importance calculation module for hierarchical Bayesian preference learning.

Provides comprehensive feature importance analysis through:
  - Model-independent metrics (correlation, mutual information)
  - Model-dependent metrics (effect magnitude, consistency)
  - Bootstrap-based feature stability assessment
  - Combined importance scoring with configurable weights
  - Cross-validation for effect consistency estimation
  - Proper error handling and logging

Supports feature selection pipeline by:
  - Calculating base feature importance
  - Estimating feature stability
  - Providing inclusion probabilities
  - Computing effect sizes and uncertainties
  - Combining multiple metrics into unified scores
"""
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from statsmodels.stats.multitest import fdrcorrection
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Tuple, Union, TypedDict, TYPE_CHECKING
from sklearn.linear_model import LogisticRegression

from engram3.utils.config import (Config, FeatureSelectionConfig, MetricWeights)
if TYPE_CHECKING:
    from engram3.model import PreferenceModel
from engram3.data import PreferenceDataset
from engram3.utils.caching import CacheManager
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureImportanceCalculator:
    """Centralized feature importance calculation."""

    def __init__(self, config: Union[Dict, Config]):
        if isinstance(config, dict):
            feature_selection_config = config.get('feature_selection', {})
        else:
            # It's a Config object, access feature_selection directly
            feature_selection_config = config.feature_selection.dict()

        self.config = FeatureSelectionConfig(**feature_selection_config)
        
        # Get statistical parameters from config
        self.mi_bins = self.config.statistical_testing.mi_bins
        self.n_permutations = self.config.statistical_testing.n_permutations
        self.n_bootstrap = self.config.statistical_testing.n_bootstrap

        # Initialize metric cache
        from engram3.utils.caching import CacheManager
        self.metric_cache = CacheManager(max_size=10000)
        
    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'metric_cache'):
            self.metric_cache.clear()

    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, 
                        model: Optional['PreferenceModel'] = None) -> Dict[str, float]:
        """Main public interface for evaluating feature importance."""
        if not isinstance(feature, str):
            raise ValueError(f"Feature must be a string, got {type(feature)}")
        
        if not isinstance(dataset, PreferenceDataset):
            raise ValueError(f"Dataset must be PreferenceDataset, got {type(dataset)}")
    
        cache_key = self.metric_cache.get_cache_key(dataset.file_path, feature, id(dataset))
        cached_metrics = self.metric_cache.get(cache_key)
        if cached_metrics is not None:
            return cached_metrics.copy()
            
        try:
            # Calculate base metrics
            metrics = self._calculate_base_metrics(feature, dataset)
            
            # Add model-dependent metrics if model provided
            if model is not None:
                # Use model's method instead of calculating here
                model_metrics = model._calculate_model_metrics(feature, dataset)
                metrics.update(model_metrics)
                
            # Calculate combined score
            metrics['importance_score'] = self._calculate_combined_score(**metrics)
            
            # Calculate statistical significance
            metrics['p_value'] = self._calculate_significance(feature, dataset)
            
            # Cache results
            self.metric_cache.set(cache_key, metrics.copy())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            return self._get_default_metrics()

    def calculate_adaptive_threshold(self, importance_scores: np.ndarray) -> float:
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
                median = np.median(scores)
                
                # Get threshold from config or use default
                base_threshold = (
                    self.config.thresholds.get('importance', 0.1) 
                    if hasattr(self.config, 'thresholds') 
                    else 0.1
                )
                
                # Adaptive threshold based on distribution
                if len(scores) > 1:
                    # Use mean + std if distribution is well-behaved
                    if std > 0:
                        threshold = mean + base_threshold * std
                    else:
                        threshold = median
                        
                    # Ensure threshold is not too extreme
                    min_threshold = np.percentile(scores, 25)  # 25th percentile
                    max_threshold = np.percentile(scores, 75)  # 75th percentile
                    threshold = np.clip(threshold, min_threshold, max_threshold)
                else:
                    threshold = base_threshold
                    
                logger.debug(f"Calculated adaptive threshold: {threshold:.3f}")
                return float(threshold)
                
            except Exception as e:
                logger.error(f"Error calculating adaptive threshold: {str(e)}")
                # Fall back to base threshold from config
                return (
                    self.config.thresholds.get('importance', 0.1) 
                    if hasattr(self.config, 'thresholds') 
                    else 0.1
                )
            
    def fdr_correct(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Perform False Discovery Rate (FDR) correction on p-values.
        
        Args:
            p_values: Array of p-values to correct
            alpha: Significance level
            
        Returns:
            Boolean array indicating which tests are significant
        """
        try:
            from statsmodels.stats.multitest import fdrcorrection
            
            # Ensure p-values are valid
            p_values = np.array(p_values)
            p_values[np.isnan(p_values)] = 1.0
            p_values = np.clip(p_values, 0, 1)
            
            # Perform FDR correction
            significant, _ = fdrcorrection(p_values, alpha=alpha)
            return significant
            
        except Exception as e:
                logger.error(f"Error in FDR correction: {str(e)}")
                # Fall back to simple threshold
                return p_values < alpha
                                                                
    @staticmethod
    def compute_interaction_lr(feature_values: List[np.ndarray], 
                             responses: np.ndarray,
                             min_effect_size: float) -> Tuple[float, float]:
        """
        Compute interaction significance using likelihood ratio test.
        
        Args:
            feature_values: List of feature value arrays [feature1_values, feature2_values]
            responses: Binary response values (0/1)
            min_effect_size: Minimum effect size threshold
            
        Returns:
            Tuple of (interaction_value, p_value)
        """
        from sklearn.linear_model import LogisticRegression
        from scipy import stats
        
        try:
            # Create interaction term
            X_main = np.column_stack(feature_values)
            X_interaction = X_main[:, 0] * X_main[:, 1]
            
            # Fit model without interaction
            model_main = LogisticRegression(random_state=42)
            model_main.fit(X_main, responses)
            ll_main = model_main.score(X_main, responses) * len(responses)
            
            # Fit model with interaction
            X_full = np.column_stack([X_main, X_interaction])
            model_full = LogisticRegression(random_state=42)
            model_full.fit(X_full, responses)
            ll_full = model_full.score(X_full, responses) * len(responses)
            
            # Calculate likelihood ratio statistic
            lr_stat = 2 * (ll_full - ll_main)
            
            # Get p-value
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
            
            # Get interaction coefficient
            interaction_value = model_full.coef_[0][-1]
            
            # Return zero effect if below minimum threshold
            if abs(interaction_value) < min_effect_size:
                return 0.0, 1.0
                
            return float(interaction_value), float(p_value)
            
        except Exception as e:
            logger.error(f"Error in interaction calculation: {str(e)}")
            return 0.0, 1.0
                
    def _calculate_interaction_score(self, metrics: Dict[str, float]) -> float:
        """Calculate feature interaction score."""
        weights = self.config.feature_selection.metric_weights
        return (
            weights['correlation'] * abs(metrics['correlation']) +
            weights['mutual_information'] * metrics['mutual_information']
        )
        
    def _calculate_base_feature_score(self, metrics: Dict[str, float]) -> float:
        """Calculate base feature importance score."""
        weights = self.config.feature_selection['metric_weights']
        return (
            weights['inclusion_probability'] * metrics['inclusion_probability'] +
            weights['effect_magnitude'] * metrics['effect_magnitude'] +
            weights['effect_consistency'] * (1.0 - metrics['effect_consistency']) +
            weights['correlation'] * abs(metrics['correlation']) +
            weights['mutual_information'] * metrics['mutual_information']
        )
        
    def _calculate_inclusion_probability(self,
                                     feature: str,
                                     dataset: 'PreferenceDataset',
                                     n_bootstrap: Optional[int] = None) -> float:
        """Calculate probability of feature being selected across bootstrap samples."""
        n_bootstrap = n_bootstrap or self.DEFAULT_BOOTSTRAP_SAMPLES

        try:
            n_samples = len(dataset.preferences)
            inclusion_count = 0
            
            for _ in range(n_bootstrap):
                try:
                    bootstrap_indices = np.random.choice(
                        n_samples,
                        size=n_samples,
                        replace=True
                    )
                    bootstrap_data = dataset._create_subset_dataset(bootstrap_indices)
                    importance = self._calculate_basic_importance(feature, bootstrap_data)
                    threshold = self.config.feature_selection.thresholds['importance']
                    
                    if importance > threshold:
                        inclusion_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error in bootstrap iteration: {str(e)}")
                    continue
                    
            return inclusion_count / n_bootstrap
            
        except Exception as e:
            logger.error(f"Error calculating inclusion probability: {str(e)}")
            return 0.0
        
    def _calculate_correlation(self, feature_diffs: np.ndarray, preferences: np.ndarray) -> float:
        """Calculate correlation handling constant inputs."""
        try:
            # Check for constant inputs
            if len(set(feature_diffs)) <= 1 or len(set(preferences)) <= 1:
                logger.debug(f"Constant input detected for correlation calculation")
                return 0.0
                
            correlation, _ = spearmanr(feature_diffs, preferences)
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception as e:
            logger.warning(f"Error calculating correlation: {str(e)}")
            return 0.0
        
    def _calculate_mutual_information(self, feature: str, dataset: 'PreferenceDataset') -> float:
        """Calculate mutual information between feature and preferences."""
        try:
            # Use consolidated extraction method
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
            
            # Convert preferences to 0/1 for MI calculation
            preferences = (preferences + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Discretize feature differences for MI calculation
            bins = np.linspace(min(feature_diffs), max(feature_diffs), self.MI_BINS)
            binned_diffs = np.digitize(feature_diffs, bins)
            
            mi_score = mutual_info_score(binned_diffs, preferences)
            return float(mi_score)
            
        except Exception as e:
            logger.error(f"Error calculating mutual information for {feature}: {str(e)}")
            return 0.0

    def _calculate_basic_importance(self, feature: str, dataset: 'PreferenceDataset') -> float:
        """Calculate basic importance score without model-dependent metrics."""
        try:
            correlation = self._calculate_correlation(feature, dataset)
            mutual_info = self._calculate_mutual_information(feature, dataset)
            
            # Use configured weights or defaults
            weights = getattr(self.config, 'feature_selection', {}).get('metric_weights', {
                'correlation': 0.5,
                'mutual_information': 0.5
            })
            
            return (weights['correlation'] * abs(correlation) + 
                    weights['mutual_information'] * mutual_info)
                    
        except Exception as e:
            logger.error(f"Error calculating basic importance for {feature}: {str(e)}")
            return 0.0

    def _calculate_combined_score(self, **metrics) -> float:
        """Calculate combined importance score using all metrics."""
        try:
            weights = self.config.metric_weights  # Access directly from config
            
            return (
                weights['correlation'] * abs(metrics.get('correlation', 0.0)) +
                weights['mutual_information'] * metrics.get('mutual_information', 0.0) +
                weights['effect_magnitude'] * metrics.get('effect_magnitude', 0.0) +
                weights['effect_consistency'] * (1.0 - metrics.get('effect_consistency', 0.0)) +
                weights['inclusion_probability'] * metrics.get('inclusion_probability', 0.0)
            )
        except Exception as e:
            logger.error(f"Error calculating combined score: {str(e)}")
            return 0.0
                        
    def _calculate_significance(self, feature: str, dataset: PreferenceDataset) -> float:
        """
        Calculate statistical significance of feature using permutation test.
        
        Args:
            feature: Feature to test
            dataset: Dataset to evaluate on
            
        Returns:
            p-value from permutation test
        """
        try:
            # Get original correlation
            original_corr = self._calculate_correlation(feature, dataset)
            
            # Perform permutation test
            n_permutations = self.N_PERMUTATIONS
            null_corrs = []
            
            preferences = [pref.preferred for pref in dataset.preferences]
            for _ in range(n_permutations):
                # Shuffle preferences
                np.random.shuffle(preferences)
                
                # Calculate correlation with shuffled preferences
                feature_diffs = []
                for pref, shuffled_pref in zip(dataset.preferences, preferences):
                    value1 = pref.features1.get(feature, 0.0)
                    value2 = pref.features2.get(feature, 0.0)
                    feature_diffs.append(value1 - value2)
                    
                corr, _ = spearmanr(feature_diffs, preferences)
                if not np.isnan(corr):
                    null_corrs.append(abs(corr))
                    
            # Calculate p-value
            null_corrs = np.array(null_corrs)
            p_value = np.mean(null_corrs >= abs(original_corr))
            
            return float(p_value)
            
        except Exception as e:
            logger.error(f"Error calculating significance for {feature}: {str(e)}")
            return 1.0
        
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics dictionary with zero values."""
        return {
            'correlation': 0.0,
            'mutual_information': 0.0,
            'effect_magnitude': 0.0,
            'effect_consistency': 0.0,
            'inclusion_probability': 0.0,
            'importance_score': 0.0,
            'p_value': 1.0
        }

    def _calculate_base_metrics(self, feature: str, dataset: PreferenceDataset) -> Dict[str, float]:
        """Calculate base metrics independent of model."""
        feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
        
        return {
            'correlation': self._calculate_correlation(feature_diffs, preferences),
            'mutual_information': self._calculate_mutual_information(feature, dataset)
        }

    def _extract_feature_differences(self, feature: str, dataset: PreferenceDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature differences and preferences from dataset."""
        if not dataset.preferences:
            raise ValueError("Dataset contains no preferences")
            
        feature_diffs = []
        preferences = []
        
        for pref in dataset.preferences:
            try:
                value1 = pref.features1.get(feature, 0.0)
                value2 = pref.features2.get(feature, 0.0)
                # Handle None values
                if value1 is None or value2 is None:
                    continue
                feature_diffs.append(float(value1) - float(value2))
                preferences.append(1.0 if pref.preferred else -1.0)
            except AttributeError as e:
                logger.warning(f"Invalid preference format: {str(e)}")
                continue
                
        if not feature_diffs:
            raise ValueError(f"No valid feature differences extracted for {feature}")
            
        return np.array(feature_diffs), np.array(preferences)
