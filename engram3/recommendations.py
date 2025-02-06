# recommendations.py
"""
Bigram pair recommendation system for keyboard layout optimization.

The system uses a weighted multi-objective approach to balance:
1. Information Gain (φ ≈ 0.618):
   - Model uncertainty for each bigram
   - Expected entropy reduction
   - Potential for model improvement

2. Feature Space Coverage (1-φ ≈ 0.382):
   - Distance to existing data points 
   - Penalty for closeness to already selected recommendations
   - Maximize minimum distance between points

The process iteratively selects bigram pairs by:
1. First selecting the point furthest from existing data
2. Then selecting points that maximize the minimum distance to all points (both existing and selected)
3. Applying an exponential penalty for closeness to already selected recommendations

This maximum-minimum distance approach optimizes for point separation to achieve good coverage
of the feature space. Combined with information gain weighting, this helps select pairs that
will both improve model accuracy and explore under-sampled regions of the feature space.

The algorithm steps are:

1. Generate all possible bigram pairs from the layout characters except for:
   - Same-character bigrams
   - Pairs already evaluated

2. Project data to 2D PCA space:
   - Project both existing data and candidate pairs
   - Preserve key feature relationships
   - Enable efficient distance calculations

3. Select diverse pairs:
   - Pick first pair furthest from existing data
   - Iteratively select pairs maximizing minimum distance to all points
   - Apply exponential penalty for closeness to already selected pairs
   - Balance exploration vs local density

4. Visualize results:
   - Plot PCA projections of all points
   - Show feature space coverage
   - Display feature value distributions
"""
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union, Optional, Any
import numpy as np
import pandas as pd
from itertools import combinations
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
        """Generate recommendations by maximizing minimum distances to all points."""
        try:
            self._initialize_state()
            
            candidates = self._generate_candidates()
            if not candidates:
                raise ValueError("No valid candidates generated")
                
            # Project to PCA space as before
            existing_differences = []
            for pref in self.dataset.preferences:
                diff = self._get_pair_features(pref.bigram1, pref.bigram2)
                existing_differences.append(diff)
            existing_differences = np.array(existing_differences)

            candidate_differences = []
            for pair in candidates:
                diff = self._get_pair_features(pair[0], pair[1])
                candidate_differences.append(diff)
            candidate_differences = np.array(candidate_differences)

            pca = PCA(n_components=2)
            all_differences = np.vstack([existing_differences, candidate_differences])
            pca.fit(all_differences)
            
            existing_2d = pca.transform(existing_differences)
            candidate_2d = pca.transform(candidate_differences)
            
            # Initialize selections
            selected = []
            selected_points = []
            remaining_candidates = list(zip(candidates, candidate_2d))
            
            logger.info(f"Starting selection with {len(remaining_candidates)} candidates")

            # First selection: point furthest from existing data
            if remaining_candidates:
                distances = []
                for pair, point in remaining_candidates:
                    min_dist = np.min(np.linalg.norm(existing_2d - point, axis=1))
                    distances.append((min_dist, pair, point))
                
                best_dist, best_pair, best_point = max(distances)
                selected.append(best_pair)
                selected_points.append(best_point)
                remaining_candidates = [(p, pt) for p, pt in remaining_candidates if p != best_pair]

            # Subsequent selections: maximize minimum distance to ALL points
            while len(selected) < self.n_recommendations and remaining_candidates:
                # Consider both existing and selected points
                all_points = np.vstack([existing_2d, selected_points])
                
                # Find candidate that maximizes minimum distance to all points
                max_min_dist = -float('inf')
                best_candidate = None
                best_point = None
                
                for pair, point in remaining_candidates:
                    # Calculate minimum distance to any existing or selected point
                    distances = np.linalg.norm(all_points - point, axis=1)
                    min_dist = np.min(distances)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_candidate = pair
                        best_point = point
                
                if best_candidate is None:
                    break
                    
                selected.append(best_candidate)
                selected_points.append(best_point)
                remaining_candidates = [(p, pt) for p, pt in remaining_candidates if p != best_candidate]
                
                logger.info(f"Selected {len(selected)}/{self.n_recommendations} recommendations")

            return selected

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def _initialize_state(self) -> None:
        """Initialize state with calibrated comfort threshold."""
        logger.info("Initializing recommendation state...")
        
        # Track existing pairs
        self._existing_pairs = {
            (p.bigram1, p.bigram2) for p in self.dataset.preferences
        }
        self._existing_pairs.update(
            (p.bigram2, p.bigram1) for p in self.dataset.preferences
        )
        logger.info(f"State initialized with {len(self._existing_pairs)} existing pairs")

        # Calibrate comfort threshold
        all_scores = []
        for pref in self.dataset.preferences:
            pred1 = self.model.predict_comfort_score(pref.bigram1)
            pred2 = self.model.predict_comfort_score(pref.bigram2)
            all_scores.extend([pred1.probability, pred2.probability])
        
        self.comfort_threshold = np.median(all_scores)
        logger.info(f"Calibrated comfort threshold to median: {self.comfort_threshold:.3f}")
        logger.info(f"Score distribution - min: {np.min(all_scores):.3f}, max: {np.max(all_scores):.3f}")

        # Initialize feature coverage matrix
        self._feature_coverage = self._analyze_feature_coverage()
        logger.info(f"Initialized feature coverage matrix with shape: {self._feature_coverage.shape}")
        
        # Calculate model entropy baseline
        self._model_entropy = self._calculate_model_entropy()
        logger.info(f"Initial model entropy: {self._model_entropy:.3f}")
                                        
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
    # Scoring system
    #--------------------------------------------
    def _generate_candidates(self) -> List[Tuple[str, str]]:
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
            indices = np.random.choice(
                len(candidates), 
                self.max_candidates, 
                replace=False
            )
            candidates = [candidates[i] for i in indices]
        
        logger.info(f"Generated {len(candidates)} candidate pairs")
        return candidates
           
    def _score_candidates(self, candidates: List[Tuple[str, str]], 
                        selected: List[Tuple[str, str]]) -> List[Tuple[Tuple[str, str], float, Dict[str, float]]]:
        """Score candidates using information gain and coverage metrics."""
        scored = []
        
        for pair in candidates:
            try:
                # Calculate scores considering currently selected pairs
                scores = {
                    'information_gain': self._calculate_information_gain(pair),
                    'coverage_value': self._calculate_coverage_value(pair, selected)
                }
                
                total_score = sum(self.weights[k] * v for k, v in scores.items())
                scored.append((pair, total_score, scores))
                
            except Exception as e:
                logger.debug(f"Error scoring pair {pair}: {str(e)}")
                continue
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
        
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

    def _calculate_coverage_value(self, pair: Tuple[str, str], selected: List[Tuple[str, str]]) -> float:
        try:
            pair_features = np.concatenate([
                self._get_features(pair[0]),
                self._get_features(pair[1])
            ])

            # Base distance to existing data
            existing_distances = []
            for pref in self.dataset.preferences:
                existing_features = np.concatenate([
                    self._get_features(pref.bigram1),
                    self._get_features(pref.bigram2)
                ])
                existing_distances.append(np.linalg.norm(pair_features - existing_features))
            
            base_score = np.min(existing_distances)

            # Strong penalty for similarity to already selected recommendations
            if selected:
                selected_distances = []
                for sel_pair in selected:
                    sel_features = np.concatenate([
                        self._get_features(sel_pair[0]),
                        self._get_features(sel_pair[1])
                    ])
                    dist = np.linalg.norm(pair_features - sel_features)
                    selected_distances.append(dist)
                
                # Exponential penalty for nearby recommendations
                min_selected_dist = np.min(selected_distances)
                selection_penalty = np.exp(-min_selected_dist)
                base_score *= (1.0 - selection_penalty)

            return float(base_score)

        except Exception as e:
            logger.error(f"Error calculating coverage: {str(e)}")
            return 0.0
                                                        
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

    def _get_features(self, bigram: str) -> np.ndarray:
        """Get feature vector for bigram."""
        if bigram in self._feature_cache:
            return self._feature_cache[bigram]
            
        features_dict = self.model.extract_features(bigram)
        # Convert dictionary to array in consistent order
        feature_names = list(self.model.get_feature_weights(include_control=True).keys())
        features_array = np.array([features_dict[name] for name in feature_names])
        
        self._feature_cache[bigram] = features_array
        return features_array

    def _get_pair_features(self, bigram1: str, bigram2: str) -> np.ndarray:
        """Get absolute feature differences between two bigrams."""
        features1 = self._get_features(bigram1)
        features2 = self._get_features(bigram2)
        return np.abs(features1 - features2)

    def _analyze_feature_coverage(self) -> np.ndarray:
        """Analyze current feature space coverage."""
        features = []
        for pref in self.dataset.preferences:
            # Get features for each bigram in existing pairs
            features1 = self._get_features(pref.bigram1)
            features2 = self._get_features(pref.bigram2)
            # Concatenate features from both bigrams
            pair_features = np.concatenate([features1, features2])
            features.append(pair_features)
        return np.array(features)
        
    #--------------------------------------------
    # Visualization method
    #--------------------------------------------
    def visualize_recommendations(self, recommended_pairs: List[Tuple[str, str]]):
        """Visualize feature differences with separate figures."""
        try:
            # Get feature differences for existing and recommended pairs
            existing_differences = []
            for pref in self.dataset.preferences:
                diff = self._get_pair_features(pref.bigram1, pref.bigram2)
                existing_differences.append(diff)
            existing_differences = np.array(existing_differences)
            logger.info(f"Existing differences shape: {existing_differences.shape}")

            recommended_differences = []
            for b1, b2 in recommended_pairs:
                diff = self._get_pair_features(b1, b2)
                recommended_differences.append(diff)
            recommended_differences = np.array(recommended_differences)
            logger.info(f"Recommended differences shape: {recommended_differences.shape}")

            # Generate all possible pair differences
            chars = self.config.data.layout['chars']
            all_bigrams = [c1 + c2 for c1 in chars for c2 in chars if c1 != c2]
            
            possible_differences = []
            possible_pairs = list(combinations(all_bigrams, 2))
            
            logger.info(f"Generating features for all {len(possible_pairs)} possible pairs...")
            for b1, b2 in possible_pairs:
                diff = self._get_pair_features(b1, b2)
                possible_differences.append(diff)
            possible_differences = np.array(possible_differences)
            logger.info(f"Possible differences shape: {possible_differences.shape}")

            # 1. PCA Space Plot
            fig1 = plt.figure(figsize=(12, 8))
            
            # Fit PCA on everything
            pca = PCA(n_components=2)
            all_features = np.vstack([existing_differences, recommended_differences, possible_differences])
            all_projected = pca.fit_transform(all_features)

            # Split back into groups
            n_existing = len(existing_differences)
            n_recommended = len(recommended_differences)
            existing_projected = all_projected[:n_existing]
            recommended_projected = all_projected[n_existing:n_existing + n_recommended]
            possible_projected = all_projected[n_existing + n_recommended:]

            plt.scatter(possible_projected[:, 0], possible_projected[:, 1],
                    alpha=0.05, label=f'Possible Pairs ({len(possible_pairs)})', color='gray')
            plt.scatter(existing_projected[:, 0], existing_projected[:, 1],
                    alpha=0.5, label='Existing Data', color='blue')
            plt.scatter(recommended_projected[:, 0], recommended_projected[:, 1],
                    alpha=0.7, label='Recommendations', color='red')
            
            plt.title('Feature Difference Space Coverage')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path1 = Path(self.config.paths.plots_dir) / 'feature_space_coverage.png'
            plt.savefig(output_path1, dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Feature Distributions Plot
            fig2 = plt.figure(figsize=(12, 8))
            
            feature_names = list(self.model.get_feature_weights(include_control=True).keys())
            x = np.arange(len(feature_names))
            width = 0.25
            
            # Calculate percentiles for all three groups
            def get_percentiles(features):
                return np.percentile(features, [25, 50, 75], axis=0)
            
            existing_stats = get_percentiles(existing_differences)
            recommended_stats = get_percentiles(recommended_differences)
            possible_stats = get_percentiles(possible_differences)
            
            # Plot distributions
            plt.bar(x - width, existing_stats[1], width, 
                label='Existing', color='blue', alpha=0.5)
            plt.vlines(x - width, existing_stats[0], existing_stats[2],
                    color='blue', alpha=0.3)
            
            plt.bar(x, recommended_stats[1], width,
                label='Recommended', color='red', alpha=0.5)
            plt.vlines(x, recommended_stats[0], recommended_stats[2],
                    color='red', alpha=0.3)
            
            plt.bar(x + width, possible_stats[1], width,
                label='Possible Range', color='gray', alpha=0.3)
            plt.vlines(x + width, possible_stats[0], possible_stats[2],
                    color='gray', alpha=0.2)
            
            plt.title('Feature Difference Distributions')
            plt.xticks(x, feature_names, rotation=45, ha='right')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path2 = Path(self.config.paths.plots_dir) / 'feature_distributions.png'
            plt.savefig(output_path2, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved visualizations to {output_path1} and {output_path2}")

        except Exception as e:
            logger.error(f"Error visualizing recommendations: {str(e)}")
            raise
    