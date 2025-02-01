"""
Bigram pair recommendation system for keyboard layout optimization.

The system balances two key objectives:
1. Maximizing information gain: Select pairs that will best improve model predictions
2. Ensuring comprehensive coverage: Maintain broad exploration of the typing space

This approach helps build a robust preference model while avoiding sampling biases.

Core Components:
  1. Model-Driven Selection:
    - Identifies high-uncertainty regions
    - Estimates information gain potential
    - Prioritizes impactful pairs
    
  2. Coverage Management:
    - Tracks feature space exploration
    - Identifies undersampled regions
    - Maintains sampling diversity
    
  3. Recommendation Pipeline:
    - Filters unsuitable candidates
    - Applies multi-criteria scoring
    - Ensures recommendation diversity
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set, Union, Optional, Any
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.neighbors import KernelDensity
import logging

from engram3.utils.config import Config
from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.utils.logging import LoggingManager

logger = LoggingManager.getLogger(__name__)

class BigramRecommender:
    def __init__(self, 
                dataset: PreferenceDataset, 
                model: PreferenceModel, 
                config: Union[Dict, Config]):
        """
        Initialize recommender with dataset, model and configuration.
        
        Validates inputs and sets up state tracking for recommendations.
        """
        # Core components
        self.dataset = dataset
        self.model = model
        self.config = config if isinstance(config, Config) else Config(**config)
        self.feature_extractor = model.feature_extractor
        
        # Validate core requirements
        if not dataset.preferences:
            raise ValueError("Dataset contains no preferences")
        if not hasattr(model, 'predict_comfort_score'):
            raise ValueError("Model missing required prediction method")
            
        # Configuration
        rec_config = self.config.recommendations
        self.n_recommendations = rec_config.n_recommendations
        self.max_candidates = rec_config.max_candidates
        self.weights = self._validate_weights(rec_config.weights)
        
        # Thresholds
        self.comfort_threshold = rec_config.min_comfort_score
        self.min_feature_coverage = rec_config.min_feature_coverage
        self.diversity_threshold = rec_config.min_diversity

        # Initialize state tracking
        self._comfort_range = None
        self._feature_coverage = None
        self._model_entropy = None
        self._existing_pairs: Set[Tuple[str, str]] = set()
        
        # Caching
        self._score_cache: Dict = {}
        self._feature_cache: Dict = {}
        
        # Layout information
        self.layout_chars = self.config.data.layout["chars"]

    #--------------------------------------------
    # Core class structure and basic functions
    #--------------------------------------------
    def recommend_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate recommended bigram pairs.
        
        Process:
        1. Analyze current state (model uncertainty and data coverage)
        2. Generate and filter candidate pairs
        3. Score candidates based on information gain and coverage
        4. Select diverse, high-value subset
        
        Returns:
            List of recommended bigram pairs
        """
        try:
            self._initialize_state()
            
            candidates = self._generate_candidates()
            if not candidates:
                raise ValueError("No valid candidates generated")
                
            filtered = self._filter_candidates(candidates)
            if not filtered:
                raise ValueError("No candidates passed filtering criteria")
                
            scored = self._score_candidates(filtered)
            recommendations = self._select_diverse_subset(scored)
            
            self._save_recommendations(recommendations)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def _initialize_state(self) -> None:
        """
        Initialize and validate system state.
        
        Sets up:
        - Existing pair tracking
        - Comfort score range
        - Feature space coverage
        - Model entropy baseline
        """
        logger.info("Initializing recommendation state...")
        
        # Track existing pairs (both directions)
        self._existing_pairs = {
            (p.bigram1, p.bigram2) for p in self.dataset.preferences
        }
        self._existing_pairs.update(
            (p.bigram2, p.bigram1) for p in self.dataset.preferences
        )
        
        # Calculate comfort score range
        self._comfort_range = self._analyze_comfort_range()
        
        # Analyze feature space
        self._feature_coverage = self._analyze_feature_coverage()
        
        # Get model entropy baseline
        self._model_entropy = self._calculate_model_entropy()
        
        logger.info(f"State initialized with {len(self._existing_pairs)} existing pairs")

    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate scoring weights sum to 1.0 and contain required components."""
        required = {'information_gain', 'coverage_value'}
        
        if not weights:
            raise ValueError("No scoring weights provided")
            
        missing = required - set(weights.keys())
        if missing:
            raise ValueError(f"Missing required weights: {missing}")
            
        if not np.isclose(sum(weights.values()), 1.0, rtol=1e-5):
            raise ValueError(f"Weights must sum to 1.0")
            
        return weights
    
    #--------------------------------------------
    # Filtering and scoring system
    #--------------------------------------------
    def _generate_candidates(self) -> List[Tuple[str, str]]:
        """
        Generate candidate bigram pairs.
        
        Generates all possible pairs excluding:
        - Same-character bigrams
        - Already evaluated pairs
        - Reversed versions of existing pairs
        
        Randomly samples if exceeding max_candidates limit.
        """
        # Generate possible bigrams
        bigrams = []
        for c1, c2 in combinations(self.layout_chars, 2):
            if c1 != c2:  # Exclude same-character bigrams
                bigrams.extend([(c1 + c2), (c2 + c1)])
            
        # Generate candidate pairs
        candidates = [
            (b1, b2) for b1, b2 in combinations(bigrams, 2)
            if (b1, b2) not in self._existing_pairs 
            and (b2, b1) not in self._existing_pairs
        ]
        
        # Sample if too many
        if len(candidates) > self.max_candidates:
            candidates = list(np.random.choice(
                candidates, 
                self.max_candidates, 
                replace=False
            ))
        
        logger.info(f"Generated {len(candidates)} candidate pairs")
        return candidates

    def _filter_candidates(self, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Filter candidates based on basic criteria.
        
        Criteria:
        1. Minimum comfort threshold - avoid clearly uncomfortable pairs
        2. Feature space novelty - ensure sufficient difference from existing data
        
        This quick filtering step reduces the number of pairs needing detailed scoring.
        """
        filtered = []
        for pair in candidates:
            try:
                if self._meets_basic_criteria(pair):
                    filtered.append(pair)
            except Exception as e:
                logger.debug(f"Error filtering pair {pair}: {str(e)}")
                continue
                
        logger.info(f"Filtered to {len(filtered)} valid candidates")
        return filtered

    def _meets_basic_criteria(self, pair: Tuple[str, str]) -> bool:
        """
        Check if pair meets basic suitability criteria.
        
        A pair is suitable if:
        1. Both bigrams have acceptable predicted comfort scores
        2. The pair adds sufficient novelty to the feature space
        """
        try:
            # Check comfort predictions
            pred1 = self.model.predict_comfort_score(pair[0])
            pred2 = self.model.predict_comfort_score(pair[1])
            
            if min(pred1.score, pred2.score) < self.comfort_threshold:
                return False
            
            # Check feature space novelty
            features1 = self._get_features(pair[0])
            features2 = self._get_features(pair[1])
            
            density = self._estimate_density([features1, features2])
            if density > (1 - self.min_feature_coverage):
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Error checking criteria for {pair}: {str(e)}")
            return False

    def _score_candidates(self, candidates: List[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], float, Dict[str, float]]]:
        """
        Score candidates using information gain and coverage metrics.
        
        Scoring components:
        1. Information gain: Potential model improvement from pair
        2. Coverage value: Contribution to feature space exploration
        
        Returns sorted list of (pair, total_score, detailed_scores).
        """
        scored = []
        
        for pair in candidates:
            try:
                # Check cache first
                if pair in self._score_cache:
                    scored.append(self._score_cache[pair])
                    continue
                
                # Calculate component scores
                scores = {
                    'information_gain': self._calculate_information_gain(pair),
                    'coverage_value': self._calculate_coverage_value(pair)
                }
                
                # Calculate total score using configuration weights
                total_score = sum(self.weights[k] * v for k, v in scores.items())
                
                result = (pair, total_score, scores)
                self._score_cache[pair] = result
                scored.append(result)
                
            except Exception as e:
                logger.debug(f"Error scoring pair {pair}: {str(e)}")
                continue
        
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _select_diverse_subset(self, scored_candidates: List[Tuple]) -> List[Tuple[str, str]]:
        """
        Select diverse subset of recommendations.
        
        Process:
        1. Start with highest scoring candidates
        2. For each additional selection:
           - Check diversity against already selected pairs
           - Only add if sufficiently different
        3. Continue until either:
           - Reached desired number of recommendations
           - No more sufficiently diverse candidates available
        """
        if not scored_candidates:
            return []
            
        selected = []
        candidates = scored_candidates.copy()
        
        while len(selected) < self.n_recommendations and candidates:
            # Add highest scoring remaining candidate
            best = candidates.pop(0)
            selected.append(best[0])
            
            # Filter remaining candidates for diversity
            candidates = [
                c for c in candidates 
                if self._is_sufficiently_diverse(c[0], selected)
            ]
            
        logger.info(f"Selected {len(selected)} diverse recommendations")
        return selected

    def _is_sufficiently_diverse(self, pair: Tuple[str, str], 
    
                               selected: List[Tuple[str, str]]) -> bool:
        """
        Check if pair is sufficiently different from already selected pairs.
        
        Uses feature-based similarity calculation with configurable threshold.
        """
        if not selected:
            return True
            
        try:
            features = self._get_features(pair[0]), self._get_features(pair[1])
            
            for sel_pair in selected:
                sel_features = (
                    self._get_features(sel_pair[0]), 
                    self._get_features(sel_pair[1])
                )
                
                similarity = self._calculate_similarity(features, sel_features)
                if similarity > (1 - self.diversity_threshold):
                    return False
                    
            return True
            
        except Exception as e:
            logger.debug(f"Error checking diversity: {str(e)}")
            return False

    #--------------------------------------------
    # Analysis and calculation methods
    #--------------------------------------------
    def _calculate_information_gain(self, pair: Tuple[str, str]) -> float:
        """
        Calculate expected information gain from evaluating this pair.
        
        Information gain is estimated by:
        1. Current model entropy
        2. Expected entropy after hypothetical observation
        3. Difference represents potential model improvement
        """
        try:
            pred1 = self.model.predict_comfort_score(pair[0])
            pred2 = self.model.predict_comfort_score(pair[1])
            
            # Higher uncertainty means more potential information gain
            uncertainty_component = (pred1.uncertainty + pred2.uncertainty) / 2
            
            # Calculate expected entropy reduction
            expected_entropy = self._estimate_posterior_entropy(pair, pred1, pred2)
            entropy_reduction = self._model_entropy - expected_entropy
            
            # Combine components (both normalized to [0,1] range)
            return float(0.4 * uncertainty_component + 0.6 * entropy_reduction)
            
        except Exception as e:
            logger.debug(f"Error calculating information gain: {str(e)}")
            return 0.0

    def _calculate_coverage_value(self, pair: Tuple[str, str]) -> float:
        """
        Calculate how well pair improves feature space coverage.
        
        Coverage value considers:
        1. Distance to existing data points
        2. Local density in feature space
        3. Distribution of feature values
        """
        try:
            features1 = self._get_features(pair[0])
            features2 = self._get_features(pair[1])
            
            # Calculate minimum distances to existing data
            distances = self._calculate_distances([features1, features2])
            mean_distance = np.mean(distances)
            
            # Calculate local density penalty
            density = self._estimate_density([features1, features2])
            density_penalty = 1 - density
            
            # Combine with weight toward distance (prefer exploring empty regions)
            return float(0.7 * mean_distance + 0.3 * density_penalty)
            
        except Exception as e:
            logger.debug(f"Error calculating coverage: {str(e)}")
            return 0.0

    def _analyze_comfort_range(self) -> Tuple[float, float]:
        """Calculate current comfort score range from existing data."""
        scores = []
        for pref in self.dataset.preferences:
            pred1 = self.model.predict_comfort_score(pref.bigram1)
            pred2 = self.model.predict_comfort_score(pref.bigram2)
            scores.extend([pred1.score, pred2.score])
            
        scores = np.array(scores)
        return float(np.min(scores)), float(np.max(scores))

    def _analyze_feature_coverage(self) -> np.ndarray:
        """
        Analyze current feature space coverage.
        
        Returns matrix of feature vectors for all existing bigrams.
        """
        features = []
        for pref in self.dataset.preferences:
            features.append(self._get_features(pref.bigram1))
            features.append(self._get_features(pref.bigram2))
        return np.array(features)

    def _calculate_model_entropy(self) -> float:
        """Calculate current model entropy."""
        try:
            # Use model's entropy calculation if available
            if hasattr(self.model, 'calculate_entropy'):
                return float(self.model.calculate_entropy())
            
            # Fallback: estimate from prediction uncertainties
            uncertainties = []
            for pref in self.dataset.preferences:
                pred1 = self.model.predict_comfort_score(pref.bigram1)
                pred2 = self.model.predict_comfort_score(pref.bigram2)
                uncertainties.extend([pred1.uncertainty, pred2.uncertainty])
            
            return float(np.mean(uncertainties))
            
        except Exception as e:
            logger.debug(f"Error calculating entropy: {str(e)}")
            return 1.0  # Conservative fallback

    def _estimate_posterior_entropy(self, pair: Tuple[str, str], 
                                 pred1: Any, pred2: Any) -> float:
        """
        Estimate model entropy after hypothetical observation.
        
        Uses model's built-in method if available, otherwise estimates
        based on current predictions and uncertainties.
        """
        try:
            # Use model's method if available
            if hasattr(self.model, 'estimate_posterior_entropy'):
                return float(self.model.estimate_posterior_entropy(pair, pred1, pred2))
            
            # Fallback: simple estimate based on uncertainties
            current_uncertainty = self._model_entropy
            improvement = (pred1.uncertainty + pred2.uncertainty) / 4
            return float(current_uncertainty - improvement)
            
        except Exception as e:
            logger.debug(f"Error estimating posterior entropy: {str(e)}")
            return self._model_entropy

    def _get_features(self, bigram: str) -> np.ndarray:
        """Get cached or compute feature vector for bigram."""
        if bigram in self._feature_cache:
            return self._feature_cache[bigram]
            
        features = self.model.extract_features(bigram)
        self._feature_cache[bigram] = features
        return features

    def _calculate_distances(self, features: List[np.ndarray]) -> np.ndarray:
        """Calculate minimum distances to existing data points."""
        try:
            return np.min(
                np.linalg.norm(
                    self._feature_coverage[:, np.newaxis] - features, 
                    axis=2
                ), 
                axis=0
            )
        except Exception as e:
            logger.debug(f"Error calculating distances: {str(e)}")
            return np.array([0.0])

    def _estimate_density(self, features: List[np.ndarray]) -> float:
        """Estimate local density in feature space."""
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
            kde.fit(self._feature_coverage)
            return float(np.mean(np.exp(kde.score_samples(features))))
        except Exception as e:
            logger.debug(f"Error estimating density: {str(e)}")
            return 1.0  # Conservative fallback (assume dense region)

    def _calculate_similarity(self, features1: Tuple[np.ndarray, np.ndarray],
                            features2: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate feature-based similarity between pairs."""
        try:
            similarities = [
                np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                for f1, f2 in zip(features1, features2)
            ]
            return float(np.mean(similarities))
        except Exception as e:
            logger.debug(f"Error calculating similarity: {str(e)}")
            return 1.0  # Conservative fallback (assume similar)

    def _save_recommendations(self, recommendations: List[Tuple[str, str]]) -> None:
        """Save recommendations and analysis metadata."""
        try:
            results = {
                'recommendations': recommendations,
                'metadata': {
                    'comfort_range': self._comfort_range,
                    'model_entropy': float(self._model_entropy),
                    'n_existing_pairs': len(self._existing_pairs)
                },
                'scores': {
                    str(pair): self._score_cache.get(pair, {})
                    for pair in recommendations
                }
            }
            
            output_path = Path(self.config.recommendations.recommendations_file)
            
            # Save basic recommendations
            pd.DataFrame(recommendations, columns=['bigram1', 'bigram2']).to_csv(
                output_path, index=False
            )
            
            # Save detailed results
            detailed_path = output_path.parent / 'recommendation_details.json'
            import json
            with open(detailed_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")

    #--------------------------------------------
    # Visualization method
    #--------------------------------------------
    def visualize_recommendations(self, recommended_pairs: List[Tuple[str, str]]):
        """
        Visualize how recommendations complement existing data in feature space.
        
        Creates two plots:
        1. PCA projection showing existing data and recommendations
        2. Feature coverage comparison before/after recommendations
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            # Get features for existing and recommended pairs
            existing_features = self._feature_coverage
            recommended_features = []
            for b1, b2 in recommended_pairs:
                recommended_features.append(self._get_features(b1))
                recommended_features.append(self._get_features(b2))
            recommended_features = np.array(recommended_features)
            
            # Combine for PCA
            all_features = np.vstack([existing_features, recommended_features])
            
            # Fit PCA
            pca = PCA(n_components=2)
            all_projected = pca.fit_transform(all_features)
            
            # Split back into existing and recommended
            n_existing = len(existing_features)
            existing_projected = all_projected[:n_existing]
            recommended_projected = all_projected[n_existing:]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: PCA projection
            ax1.scatter(existing_projected[:, 0], existing_projected[:, 1], 
                    alpha=0.5, label='Existing Data', color='blue')
            ax1.scatter(recommended_projected[:, 0], recommended_projected[:, 1],
                    alpha=0.7, label='Recommendations', color='red')
            ax1.set_title('Feature Space Coverage')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Feature coverage comparison
            feature_names = list(self.model.get_feature_weights(include_control=True).keys())
            existing_coverage = np.percentile(existing_features, [25, 50, 75], axis=0)
            recommended_coverage = np.percentile(recommended_features, [25, 50, 75], axis=0)
            
            x = np.arange(len(feature_names))
            width = 0.35
            
            # Plot existing coverage
            ax2.bar(x - width/2, existing_coverage[1], width, 
                    label='Existing', color='blue', alpha=0.5)
            ax2.vlines(x - width/2, existing_coverage[0], existing_coverage[2],
                    color='blue', alpha=0.3)
                    
            # Plot recommended coverage
            ax2.bar(x + width/2, recommended_coverage[1], width,
                    label='Recommended', color='red', alpha=0.5)
            ax2.vlines(x + width/2, recommended_coverage[0], recommended_coverage[2],
                    color='red', alpha=0.3)
            
            ax2.set_title('Feature Value Distribution')
            ax2.set_xticks(x)
            ax2.set_xticklabels(feature_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_path = Path(self.config.paths.plots_dir) / 'recommendation_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved recommendation visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing recommendations: {str(e)}")

            