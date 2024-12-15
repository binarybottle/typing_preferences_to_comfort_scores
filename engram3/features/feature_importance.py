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

    def __init__(self, config: Union[Dict, Config], model: 'PreferenceModel'):
        if isinstance(config, dict):
            feature_selection_config = config.get('feature_selection', {})
        else:
            feature_selection_config = config.feature_selection.dict()
        self.config = FeatureSelectionConfig(**feature_selection_config)
        
        # Store model reference
        self.model = model
        
        # Get statistical parameters from config
        self.mi_bins = self.config.statistical_testing.mi_bins
        self.n_permutations = self.config.statistical_testing.n_permutations
        self.n_bootstrap = self.config.statistical_testing.n_bootstrap
        
        # Initialize caches
        self.metric_cache = CacheManager(max_size=10000)
        self.feature_values_cache = CacheManager(max_size=10000)

    def _get_feature_values(self, feature: str, dataset: PreferenceDataset) -> np.ndarray:
        """Get feature values from dataset with caching."""
        cache_key = f"{dataset.file_path}_{feature}"
        cached_values = self.feature_values_cache.get(cache_key)
        if cached_values is not None:
            return cached_values

        values = self._compute_feature_values(feature, dataset)
        self.feature_values_cache.set(cache_key, values)
        return values

    def _compute_feature_values(self, feature: str, dataset: PreferenceDataset) -> np.ndarray:
        """Compute feature values without caching."""
        if '_x_' in feature:  # Handle interaction features
            f1, f2 = feature.split('_x_')
            return self._get_feature_values(f1, dataset) * self._get_feature_values(f2, dataset)
            
        values = []
        for pref in dataset.preferences:
            val1 = pref.features1.get(feature, 0.0)
            val2 = pref.features2.get(feature, 0.0)
            values.append(val1 - val2)
        return np.array(values)

    def __del__(self):
        """Ensure cache is cleared on deletion."""
        if hasattr(self, 'metric_cache'):
            self.metric_cache.clear()
        
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
        
    def _calculate_inclusion_probability(self, feature: str, dataset: PreferenceDataset) -> float:
        """Calculate probability of feature being selected across bootstrap samples."""
        try:
            n_samples = len(dataset.preferences)
            inclusion_count = 0
            
            for _ in range(self.n_bootstrap):
                # Create bootstrap sample
                bootstrap_indices = np.random.choice(
                    n_samples,
                    size=n_samples,
                    replace=True
                )
                bootstrap_data = dataset._create_subset_dataset(bootstrap_indices)
                
                # Calculate feature importance on bootstrap sample
                importance = self._calculate_basic_importance(feature, bootstrap_data)
                threshold = self.config.thresholds['importance']
                
                if importance > threshold:
                    inclusion_count += 1
                    
            return inclusion_count / self.n_bootstrap
            
        except Exception as e:
            logger.error(f"Error calculating inclusion probability for {feature}: {str(e)}")
            return 0.0
                
    def _calculate_correlation(self, feature: str, dataset: PreferenceDataset) -> float:
        """Calculate correlation between feature and preferences."""
        try:
            # Extract feature differences and preferences
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
                    
            # Ensure we're working with numpy arrays
            feature_diffs = np.array(feature_diffs, dtype=float)
            preferences = np.array(preferences, dtype=float)
            
            # Handle None/NaN values
            feature_diffs = np.nan_to_num(feature_diffs, 0.0)
            preferences = np.nan_to_num(preferences, 0.0)
            
            # Calculate correlation if we have valid data
            if len(feature_diffs) > 0 and len(set(feature_diffs)) > 1:
                correlation, _ = spearmanr(feature_diffs, preferences)
                return float(correlation) if not np.isnan(correlation) else 0.0
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation for feature {feature}: {str(e)}")
            return 0.0
                        
    def _extract_feature_differences(self, feature: str, dataset: PreferenceDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature differences and preferences from dataset."""
        try:
            # Handle interaction features
            if '_x_' in feature:
                components = feature.split('_x_')
                # Get component values
                component_values = []
                for comp in components:
                    values = []
                    if comp == 'typing_time':
                        # Calculate mean typing time once for the dataset
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
                    else:
                        for pref in dataset.preferences:
                            val1 = pref.features1.get(comp, 0.0)
                            val2 = pref.features2.get(comp, 0.0)
                            val1 = 0.0 if val1 is None else val1
                            val2 = 0.0 if val2 is None else val2
                            values.append(val1 - val2)
                    component_values.append(np.array(values))
                
                # Multiply components for interaction
                feature_diffs = component_values[0]
                for values in component_values[1:]:
                    feature_diffs = feature_diffs * values
            else:
                # Handle base features
                if feature == 'typing_time':
                    # Calculate mean typing time once for the dataset
                    valid_times = []
                    for pref in dataset.preferences:
                        if pref.typing_time1 is not None:
                            valid_times.append(pref.typing_time1)
                        if pref.typing_time2 is not None:
                            valid_times.append(pref.typing_time2)
                    mean_time = np.mean(valid_times) if valid_times else 0.0
                    
                    # Use mean time for None values
                    feature_diffs = []
                    for pref in dataset.preferences:
                        val1 = mean_time if pref.typing_time1 is None else pref.typing_time1
                        val2 = mean_time if pref.typing_time2 is None else pref.typing_time2
                        feature_diffs.append(val1 - val2)
                else:
                    feature_diffs = []
                    for pref in dataset.preferences:
                        val1 = pref.features1.get(feature, 0.0)
                        val2 = pref.features2.get(feature, 0.0)
                        val1 = 0.0 if val1 is None else val1
                        val2 = 0.0 if val2 is None else val2
                        feature_diffs.append(val1 - val2)
                
                feature_diffs = np.array(feature_diffs)
            
            # Get preferences
            preferences = np.array([1.0 if pref.preferred else -1.0 
                                for pref in dataset.preferences])
            
            # Ensure we have valid data
            if len(feature_diffs) == 0 or len(preferences) == 0:
                logger.warning(f"No valid data for feature {feature}, returning dummy arrays")
                return np.array([0.0]), np.array([0.0])
                
            return feature_diffs, preferences
            
        except Exception as e:
            logger.error(f"Error extracting feature differences for {feature}: {str(e)}")
            logger.error("Returning dummy arrays")
            return np.array([0.0]), np.array([0.0])

    def _calculate_mutual_information(self, feature: str, dataset: PreferenceDataset) -> float:
        """Calculate mutual information between feature and preferences."""
        try:
            # Get values with proper None handling
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
            
            # Handle None/NaN values
            feature_diffs = np.nan_to_num(feature_diffs, 0.0)
            preferences = np.nan_to_num(preferences, 0.0)
            
            # Discretize feature differences using config parameter
            bins = np.histogram_bin_edges(feature_diffs, bins=self.mi_bins)
            binned_diffs = np.digitize(feature_diffs, bins)
            
            return float(mutual_info_score(binned_diffs, (preferences + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating mutual information for {feature}: {str(e)}")
            return 0.0

    def _calculate_basic_importance(self, feature: str, dataset: 'PreferenceDataset') -> float:
        """Calculate basic importance score without model-dependent metrics."""
        try:
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
            correlation = self._calculate_correlation(feature_diffs, preferences)  # Pass arrays
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
            # Get original feature differences and preferences
            feature_diffs, original_preferences = self._extract_feature_differences(feature, dataset)
            
            # Calculate original correlation
            original_corr, _ = spearmanr(feature_diffs, original_preferences)
            if np.isnan(original_corr):
                return 1.0
            
            # Perform permutation test using config parameter
            n_permutations = self.n_permutations  # Use instance attribute
            null_corrs = []
            
            preferences = original_preferences.copy()  # Work with copy
            for _ in range(n_permutations):
                # Shuffle preferences
                np.random.shuffle(preferences)
                
                # Calculate correlation with shuffled preferences
                corr, _ = spearmanr(feature_diffs, preferences)
                if not np.isnan(corr):
                    null_corrs.append(abs(corr))
                    
            if not null_corrs:
                return 1.0
                
            # Calculate p-value
            null_corrs = np.array(null_corrs)
            p_value = np.mean(null_corrs >= abs(original_corr))
            
            return float(p_value)
            
        except Exception as e:
            logger.error(f"Error calculating significance for {feature}: {str(e)}")
            return 1.0
        
    def _calculate_effect_consistency(self, feature: str, dataset: PreferenceDataset) -> float:
        """Calculate consistency of feature effect across cross-validation splits."""
        try:
            n_splits = 5  # Could move to config if needed
            effects = []
            
            # Use model's CV splitting method
            for train_idx, val_idx in self.model._get_cv_splits(dataset, n_splits):
                train_data = dataset._create_subset_dataset(train_idx)
                self.model.fit(train_data, [feature])
                weights = self.model.get_feature_weights()
                if feature in weights:
                    effects.append(weights[feature][0])
                    
            if not effects:
                return 0.0
                
            # Calculate consistency as 1 - coefficient of variation
            return 1.0 - (np.std(effects) / (abs(np.mean(effects)) + 1e-10))
            
        except Exception as e:
            logger.error(f"Error calculating effect consistency: {str(e)}")
            return 0.0
                
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
        """Calculate base metrics for a feature with detailed error logging."""
        metrics = {}
        feature_diffs = None
        preferences = None

        try:
            # Get feature differences and preferences first
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
            logger.debug(f"Feature {feature} value range: [{np.min(feature_diffs)}, {np.max(feature_diffs)}]")
            
            # Calculate correlation
            correlation, p_value = spearmanr(feature_diffs, preferences)
            metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            logger.debug(f"Correlation for {feature}: {metrics['correlation']:.3f} (p={p_value:.3f})")

            # Calculate mutual information
            bins = np.histogram_bin_edges(feature_diffs, bins=self.mi_bins)
            binned_diffs = np.digitize(feature_diffs, bins)
            mi_score = mutual_info_score(binned_diffs, (preferences + 1) / 2)
            metrics['mutual_information'] = float(mi_score)
            logger.debug(f"Mutual information for {feature}: {metrics['mutual_information']:.3f}")

            # Calculate effect magnitude
            effect_magnitude = np.mean(np.abs(feature_diffs))
            metrics['effect_magnitude'] = float(effect_magnitude)
            logger.debug(f"Effect magnitude for {feature}: {metrics['effect_magnitude']:.3f}")

            # Calculate effect consistency through cross-validation
            cv_effects = []
            for train_idx, val_idx in self.model._get_cv_splits(dataset, 5):
                train_data = dataset._create_subset_dataset(train_idx)
                self.model.fit(train_data, [feature])
                weights = self.model.get_feature_weights()
                if feature in weights:
                    cv_effects.append(weights[feature][0])
            
            if cv_effects:
                effect_consistency = 1.0 - (np.std(cv_effects) / (abs(np.mean(cv_effects)) + 1e-10))
                metrics['effect_consistency'] = float(effect_consistency)
                logger.debug(f"Effect consistency for {feature}: {metrics['effect_consistency']:.3f}")
            else:
                metrics['effect_consistency'] = 0.0
                logger.warning(f"No CV effects calculated for {feature}")

            # Calculate inclusion probability
            inclusion_count = 0
            for _ in range(self.n_bootstrap):
                bootstrap_indices = np.random.choice(len(dataset.preferences), 
                                                  size=len(dataset.preferences), 
                                                  replace=True)
                bootstrap_data = dataset._create_subset_dataset(bootstrap_indices)
                importance = self._calculate_basic_importance(feature, bootstrap_data)
                if importance > self.config.thresholds['importance']:
                    inclusion_count += 1
            
            metrics['inclusion_probability'] = float(inclusion_count / self.n_bootstrap)
            logger.debug(f"Inclusion probability for {feature}: {metrics['inclusion_probability']:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for {feature}: {str(e)}")
            logger.error(f"Feature differences shape: {feature_diffs.shape if feature_diffs is not None else None}")
            logger.error(f"Preferences shape: {preferences.shape if preferences is not None else None}")
            logger.error("Traceback:", exc_info=True)
            return self._get_default_metrics()

    def evaluate_feature(self, feature: str, dataset: PreferenceDataset, 
                        model: Optional['PreferenceModel'] = None) -> Dict[str, float]:
        """Calculate all metrics for a feature."""
        try:
            logger.debug(f"\nEvaluating feature: {feature}")

            # Get feature differences and preferences first
            feature_diffs, preferences = self._extract_feature_differences(feature, dataset)
            logger.debug(f"Feature differences sample: {feature_diffs[:5]} ...")
            logger.debug(f"Preferences sample: {preferences[:5]} ...")

            # Calculate each metric individually with error handling
            metrics = {}
            
            # 1. Correlation and p-value
            try:
                correlation = self._calculate_correlation(feature_diffs, preferences)
                p_value = self._calculate_significance(feature, dataset)
                metrics['correlation'] = correlation
                metrics['p_value'] = p_value
                logger.debug(f"Correlation: {correlation:.4f}")
                logger.debug(f"P-value: {p_value:.4f}")
            except Exception as e:
                logger.error(f"Error calculating correlation for {feature}: {str(e)}")
                metrics['correlation'] = 0.0
                metrics['p_value'] = 1.0

            # 2. Mutual Information
            try:
                mi_score = self._calculate_mutual_information(feature, dataset)
                metrics['mutual_information'] = mi_score
                logger.debug(f"Mutual information: {mi_score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating mutual information for {feature}: {str(e)}")
                metrics['mutual_information'] = 0.0

            # 3. Effect Magnitude
            try:
                effect = np.mean(np.abs(feature_diffs))
                metrics['effect_magnitude'] = effect
                logger.debug(f"Effect magnitude: {effect:.4f}")
            except Exception as e:
                logger.error(f"Error calculating effect magnitude for {feature}: {str(e)}")
                metrics['effect_magnitude'] = 0.0

            # 4. Effect Consistency
            try:
                consistency = self._calculate_effect_consistency(feature, dataset)
                metrics['effect_consistency'] = consistency
                logger.debug(f"Effect consistency: {consistency:.4f}")
            except Exception as e:
                logger.error(f"Error calculating effect consistency for {feature}: {str(e)}")
                metrics['effect_consistency'] = 0.0

            # 5. Inclusion Probability
            try:
                inclusion_prob = self._calculate_inclusion_probability(feature, dataset)
                metrics['inclusion_probability'] = inclusion_prob
                logger.debug(f"Inclusion probability: {inclusion_prob:.4f}")
            except Exception as e:
                logger.error(f"Error calculating inclusion probability for {feature}: {str(e)}")
                metrics['inclusion_probability'] = 0.0

            # Calculate combined importance score last
            try:
                importance_score = self._calculate_combined_score(**metrics)
                metrics['importance_score'] = importance_score
                logger.debug(f"Final importance score: {importance_score:.4f}")
                
                # Log weights used for combined score
                logger.debug("Weights used for scoring:")
                weights = self.config.metric_weights
                for metric, weight in weights.items():
                    logger.debug(f"  {metric}: {weight:.3f}")
            except Exception as e:
                logger.error(f"Error calculating importance score for {feature}: {str(e)}")
                metrics['importance_score'] = 0.0

            # Check if all metrics are zeros/ones
            if all(v in (0, 1) for v in metrics.values()):
                logger.warning(f"All metrics are 0/1 for feature {feature} - possible calculation error")
            
            # Final summary log
            logger.debug("\nFinal metrics summary for feature %s:", feature)
            for metric_name, value in metrics.items():
                logger.debug(f"  {metric_name}: {value:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating feature {feature}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self._get_default_metrics()
               
    def _calculate_interaction_effect(self, feature1: str, feature2: str, dataset: PreferenceDataset) -> float:
        """Calculate interaction effect between features."""
        try:
            # Get feature values with None handling
            diffs1, prefs1 = self._extract_feature_differences(feature1, dataset)
            diffs2, _ = self._extract_feature_differences(feature2, dataset)
            
            # Handle None/NaN values
            diffs1 = np.nan_to_num(diffs1, 0.0)
            diffs2 = np.nan_to_num(diffs2, 0.0)
            
            # Calculate interaction
            interaction = diffs1 * diffs2
            
            # Calculate correlation
            return self._calculate_correlation(interaction, prefs1)
            
        except Exception as e:
            logger.warning(f"Error calculating interaction effect: {str(e)}")
            return 0.0
        