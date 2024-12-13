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
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_score
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Tuple, Union, TypedDict

from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.utils.caching import CacheManager
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureSelectionConfig(BaseModel):
    """Validate feature selection configuration."""
    metric_weights: Dict[str, float]
    thresholds: Dict[str, float]
    n_bootstrap: int = 100
    
    @validator('metric_weights')
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Metric weights must sum to 1.0, got {total}")
        return v
    
    @validator('thresholds')
    def thresholds_must_be_positive(cls, v: Dict[str, float]) -> Dict[str, float]:
        if any(t <= 0 for t in v.values()):
            raise ValueError("All thresholds must be positive")
        return v

class FeatureSelectionMetricWeights(BaseModel):
    """
    Configuration for feature importance metric weights.
    All weights should sum to 1.0.
    """
    correlation: float
    mutual_information: float
    effect_magnitude: float
    effect_consistency: float
    inclusion_probability: float

class FeatureSelectionThresholds(BaseModel):
    importance: float
    stability: float

class MetricWeights(TypedDict):
    """Type definition for metric weights."""
    correlation: float
    mutual_information: float
    effect_magnitude: float
    effect_consistency: float
    inclusion_probability: float

class FeatureImportanceCalculator:
    """Centralized feature importance calculation."""
    MI_BINS: int = 20  # Number of bins for mutual information calculation
    N_PERMUTATIONS: int = 1000  # Number of permutations for significance test
    DEFAULT_BOOTSTRAP_SAMPLES: int = 100  # Default number of bootstrap samples
    
    def __init__(self, config: Dict):
        self.config = FeatureSelectionConfig(**config.get('feature_selection', {}))
        self.metric_cache: CacheManager[str, Dict[str, float]] = CacheManager()

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
                                                
    def _calculate_interaction_score(self, metrics: Dict[str, float]) -> float:
        """Calculate feature interaction score."""
        weights = self.config['feature_selection']['metric_weights']
        return (
            weights['correlation'] * abs(metrics['correlation']) +
            weights['mutual_information'] * metrics['mutual_information']
        )
        
    def _calculate_base_feature_score(self, metrics: Dict[str, float]) -> float:
        """Calculate base feature importance score."""
        weights = self.config['feature_selection']['metric_weights']
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
                    threshold = self.config['feature_selection']['thresholds']['importance']
                    
                    if importance > threshold:
                        inclusion_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error in bootstrap iteration: {str(e)}")
                    continue
                    
            return inclusion_count / n_bootstrap
            
        except Exception as e:
            logger.error(f"Error calculating inclusion probability: {str(e)}")
            return 0.0
        
    def _calculate_correlation(self, 
                             feature_diffs: Union[str, np.ndarray],
                             preferences: Union[PreferenceDataset, np.ndarray]) -> float:
        """
        Calculate correlation between feature and preferences.
        
        Args:
            feature_diffs: Either feature name or pre-computed differences
            preferences: Either dataset or pre-computed preferences
            
        Returns:
            Spearman correlation coefficient
        """
        try:
            # Handle different input types
            if isinstance(feature_diffs, str) and isinstance(preferences, PreferenceDataset):
                # Use consolidated extraction method
                feature_diffs, preferences = self._extract_feature_differences(feature_diffs, preferences)
                
            # Calculate correlation
            correlation, _ = spearmanr(feature_diffs, preferences)
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
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
            weights = self.config.get('feature_selection', {}).get('metric_weights', {
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
            required_weights = {'correlation', 'mutual_information', 'effect_magnitude', 
                            'effect_consistency', 'inclusion_probability'}
            weights: MetricWeights = self.config['feature_selection']['metric_weights']
            
            # Validate all required weights are present
            missing = required_weights - set(weights.keys())
            if missing:
                raise ValueError(f"Missing required weights: {missing}")
                
            return (weights['correlation'] * abs(metrics['correlation']) +
                    weights['mutual_information'] * metrics['mutual_information'] +
                    weights['effect_magnitude'] * metrics['effect_magnitude'] +
                    weights['effect_consistency'] * (1 - metrics['effect_consistency']) +
                    weights['inclusion_probability'] * metrics['inclusion_probability'])
                    
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
                feature_diffs.append(value1 - value2)
                preferences.append(1.0 if pref.preferred else -1.0)
            except AttributeError as e:
                logger.warning(f"Invalid preference format: {str(e)}")
                continue
                
        if not feature_diffs:
            raise ValueError(f"No valid feature differences extracted for {feature}")
            
        return np.array(feature_diffs), np.array(preferences)
