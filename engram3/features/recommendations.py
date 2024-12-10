# engram3/recommendations.py

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
from itertools import combinations
import warnings
from scipy import stats

from engram3.data import PreferenceDataset
from engram3.features.extraction import extract_bigram_features

logger = logging.getLogger(__name__)

class BigramRecommender:
    """Generate recommended bigram pairs for data collection."""
    
    def __init__(self, dataset: PreferenceDataset, config: Dict, selected_features: List[str]):
        self.dataset = dataset
        self.config = config
        
        # If no selected features, use features above minimum threshold
        if not selected_features:
            logger.warning("No selected features provided. Using features above minimum threshold.")
            min_importance = config['feature_evaluation'].get('min_importance_threshold', 0.03)
            self.selected_features = [
                f for f in dataset.get_feature_names()
                if f in ['rows_apart', 'angle_apart', 'adj_finger_diff_row', 'sum_row_position_values']
                or f in ['sum_finger_values', 'sum_engram_position_values']  # Add key features
            ]
            if not self.selected_features:
                logger.warning("Using all available features as fallback.")
                self.selected_features = dataset.get_feature_names()
        else:
            self.selected_features = selected_features
            
        self.n_recommendations = config['recommendations']['n_recommendations']
        logger.info(f"Using {len(self.selected_features)} features for recommendations")
                
    def get_uncertainty_pairs(self) -> List[Tuple[str, str]]:
        """Generate pairs from regions of high feature uncertainty."""
        # Use sparse regions from analysis
        sparse_points = self._find_sparse_regions(self.dataset)
        
        # Generate all possible bigrams
        all_bigrams = self._generate_all_possible_bigrams()
        
        # Score bigrams by feature similarity to sparse points
        scored_pairs = []
        for b1, b2 in combinations(all_bigrams, 2):
            features1 = self._extract_features(b1)
            features2 = self._extract_features(b2)
            
            # Calculate average distance to sparse points
            uncertainty_score = self._calculate_uncertainty_score(
                features1, features2, sparse_points)
            
            scored_pairs.append(((b1, b2), uncertainty_score))
            
        # Sort by uncertainty score and return top N
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, score in scored_pairs[:self.n_recommendations]]

    def get_interaction_pairs(self) -> List[Tuple[str, str]]:
        """Generate pairs that test important feature interactions."""
        # Get feature interaction strengths from model
        interaction_strengths = self._get_feature_interaction_strengths()
        
        # Generate pairs that maximize difference in interaction features
        scored_pairs = []
        all_bigrams = self._generate_all_possible_bigrams()
        
        for b1, b2 in combinations(all_bigrams, 2):
            features1 = self._extract_features(b1)
            features2 = self._extract_features(b2)
            
            interaction_score = self._calculate_interaction_score(
                features1, features2, interaction_strengths)
            
            scored_pairs.append(((b1, b2), interaction_score))
            
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, score in scored_pairs[:self.n_recommendations]]

    def get_transitivity_pairs(self) -> List[Tuple[str, str]]:
        """Generate pairs that would help validate transitivity."""
        # Build preference graph
        pref_graph = self._build_preference_graph()
        
        # Find potential transitivity chains
        scored_pairs = []
        for a in pref_graph:
            for b in pref_graph.get(a, set()):
                # Look for c where we have a>b, b>c but no direct a:c comparison
                for c in pref_graph.get(b, set()):
                    if c not in pref_graph.get(a, set()) and a not in pref_graph.get(c, set()):
                        score = self._calculate_transitivity_score(a, b, c, pref_graph)
                        scored_pairs.append(((a, c), score))
        
        # Sort by score and return top N
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, score in scored_pairs[:self.n_recommendations]]

    def _extract_features(self, bigram: str) -> Dict[str, float]:
        """Extract selected features for a bigram."""
        return extract_bigram_features(
            bigram[0], bigram[1],
            self.dataset.column_map,
            self.dataset.row_map,
            self.dataset.finger_map,
            self.dataset.engram_position_values,
            self.dataset.row_position_values
        )

    def _find_sparse_regions(dataset: PreferenceDataset) -> List[Dict[str, float]]:
        """
        Identify sparse regions in feature space.
        
        Args:
            dataset: PreferenceDataset containing preferences
            
        Returns:
            List of dictionaries representing points in sparse regions
        """
        # Get all feature vectors
        vectors = []
        feature_names = dataset.get_feature_names()
        
        for pref in dataset.preferences:
            for features in [pref.features1, pref.features2]:
                try:
                    # Check for None or NaN values
                    if any(features.get(f) is None for f in feature_names):
                        continue
                        
                    vector = [features[f] for f in feature_names]
                    if any(np.isnan(v) for v in vector):
                        continue
                        
                    vectors.append(vector)
                    
                except KeyError as e:
                    logger.warning(f"Missing feature in preference: {str(e)}")
                    continue
        
        if not vectors:
            logger.warning("No valid feature vectors found")
            return []
        
        vectors = np.array(vectors)
        
        # Find points with few neighbors
        sparse_points = []
        for i, v in enumerate(vectors):
            try:
                distances = np.linalg.norm(vectors - v, axis=1)
                n_neighbors = (distances < np.percentile(distances, 10)).sum()
                if n_neighbors < 5:  # Arbitrary threshold
                    sparse_points.append(dict(zip(feature_names, v)))
            except Exception as e:
                logger.warning(f"Error processing vector {i}: {str(e)}")
                continue
                
        logger.info(f"Found {len(sparse_points)} points in sparse regions "
                f"from {len(vectors)} valid vectors")
        
        return sparse_points

    def _calculate_uncertainty_score(self, features1: Dict, features2: Dict, 
                                   sparse_points: List[Dict]) -> float:
        """Score how well a pair covers uncertain regions."""
        # Implementation details for scoring uncertainty coverage
        pass

    def _calculate_interaction_score(self, features1: Dict, features2: Dict,
                                   interaction_strengths: Dict) -> float:
        """Score how well a pair tests feature interactions."""
        # Implementation details for scoring interaction testing
        pass

    def _calculate_transitivity_score(self, a: str, b: str, c: str, 
                                    pref_graph: Dict) -> float:
        """Score how valuable a pair would be for transitivity testing."""
        # Implementation details for scoring transitivity testing
        pass

    def _build_preference_graph(self) -> Dict[str, Set[str]]:
        """Build graph of existing preferences."""
        pref_graph = {}
        
        # Build graph from dataset preferences
        for pref in self.dataset.preferences:
            if pref.preferred:
                better, worse = pref.bigram1, pref.bigram2
            else:
                better, worse = pref.bigram2, pref.bigram1
                
            if better not in pref_graph:
                pref_graph[better] = set()
            pref_graph[better].add(worse)
        
        return pref_graph

    def _generate_all_possible_bigrams(self) -> List[str]:
        """Generate all valid bigram combinations."""
        # Implementation to generate valid bigrams
        pass

    def _calculate_uncertainty_score(self, features1: Dict, features2: Dict, 
                               sparse_points: List[Dict]) -> float:
        """
        Score how well a pair covers uncertain regions.
        Higher score = pair better samples uncertain feature space.
        """
        # Convert feature dicts to vectors using selected features only
        vec1 = np.array([features1[f] for f in self.selected_features])
        vec2 = np.array([features2[f] for f in self.selected_features])
        
        # Convert sparse points to array
        sparse_vecs = np.array([[p[f] for f in self.selected_features] 
                            for p in sparse_points])
        
        # Calculate minimum distances to sparse points for each bigram
        distances1 = np.min(np.linalg.norm(sparse_vecs - vec1, axis=1))
        distances2 = np.min(np.linalg.norm(sparse_vecs - vec2, axis=1))
        
        # Score based on:
        # 1. How close at least one bigram is to sparse region
        # 2. How different the bigrams are from each other
        pair_distance = np.linalg.norm(vec1 - vec2)
        min_sparse_distance = min(distances1, distances2)
        
        # Combine metrics: want small distance to sparse region but large pair difference
        uncertainty_score = pair_distance * np.exp(-min_sparse_distance)
        
        return float(uncertainty_score)

    def _calculate_interaction_score(self, features1: Dict, features2: Dict,
                                interaction_strengths: Dict[Tuple[str, str], float]) -> float:
        """
        Score how well a pair tests feature interactions.
        Higher score = pair better tests important interactions.
        """
        score = 0.0
        
        # For each interaction pair
        for (feat1, feat2), strength in interaction_strengths.items():
            if feat1 in features1 and feat2 in features1:
                # Calculate interaction difference between bigrams
                interaction1 = features1[feat1] * features1[feat2]
                interaction2 = features2[feat1] * features2[feat2]
                diff = abs(interaction1 - interaction2)
                
                # Weight difference by interaction strength
                score += diff * strength
        
        return float(score)

    def _calculate_transitivity_score(self, a: str, b: str, c: str, 
                                    pref_graph: Dict[str, Set[str]]) -> float:
        """
        Score how valuable a pair would be for transitivity testing.
        Higher score = pair better validates transitivity.
        """
        # Count existing preference chains involving these bigrams
        n_chains = 0
        chain_strength = 0.0
        
        # Look for chains a>b>c
        if b in pref_graph.get(a, set()):
            for d in pref_graph.get(b, set()):
                if d != c:  # Don't count the chain we're testing
                    n_chains += 1
                    # Add strength based on path length
                    chain_strength += 1.0 / len(pref_graph.get(a, set())) + \
                                    1.0 / len(pref_graph.get(b, set()))

        # Add helper scores
        def _get_feature_difference(bigram1: str, bigram2: str) -> float:
            """Calculate feature space distance between bigrams."""
            features1 = self._extract_features(bigram1)
            features2 = self._extract_features(bigram2)
            vec1 = np.array([features1[f] for f in self.selected_features])
            vec2 = np.array([features2[f] for f in self.selected_features])
            return float(np.linalg.norm(vec1 - vec2))

        # Final score combines:
        # 1. Number of related chains
        # 2. Strength of existing preferences
        # 3. Feature space distances
        feature_distances = _get_feature_difference(a, c)
        
        score = (n_chains + 1) * chain_strength * feature_distances
        
        return float(score)

    def _get_feature_interaction_strengths(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate interaction strengths between features based on existing preferences.
        Returns dict mapping feature pairs to interaction strength scores.
        """
        interaction_strengths = {}
        
        # For each feature pair
        for feat1, feat2 in combinations(self.selected_features, 2):
            # Calculate correlation between interaction terms and preferences
            interaction_values = []
            preference_values = []
            
            for pref in self.dataset.preferences:
                # Calculate interaction term differences
                int1 = pref.features1[feat1] * pref.features1[feat2]
                int2 = pref.features2[feat1] * pref.features2[feat2]
                interaction_values.append(int1 - int2)
                
                # Get preference direction
                preference_values.append(1.0 if pref.preferred else -1.0)
                
            # Calculate correlation
            interaction_values = np.array(interaction_values)
            preference_values = np.array(preference_values)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlation = stats.spearmanr(interaction_values, preference_values)[0]
                
            if not np.isnan(correlation):
                interaction_strengths[(feat1, feat2)] = abs(correlation)
        
        return interaction_strengths

    def _generate_all_possible_bigrams(self) -> List[str]:
        """Generate all valid bigram combinations."""
        # Get unique letters from existing bigrams
        letters = set()
        for pref in self.dataset.preferences:
            letters.update(pref.bigram1)
            letters.update(pref.bigram2)
        
        # Generate all valid combinations
        valid_bigrams = []
        for l1, l2 in combinations(sorted(letters), 2):
            # Check if bigram appears in dataset (as validation)
            bigram = l1 + l2
            reverse_bigram = l2 + l1
            if any(bigram in (p.bigram1, p.bigram2) or 
                reverse_bigram in (p.bigram1, p.bigram2)
                for p in self.dataset.preferences):
                valid_bigrams.append(bigram)
        
        return valid_bigrams