# engram3/recommendations.py

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd
from itertools import combinations
import warnings
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
import random

from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.features.extraction import extract_bigram_features

logger = logging.getLogger(__name__)

class BigramRecommender:
    """Generate recommended bigram pairs for data collection."""
    
    def __init__(self, dataset: PreferenceDataset, model: PreferenceModel, config: Dict):
        self.dataset = dataset
        self.model = model
        self.config = config
        self.selected_features = list(model.get_feature_weights().keys())
        self.n_recommendations = config['recommendations']['n_recommendations']
        self.layout_chars = config['data']['layout']['chars']

    def score_pair(self, bigram1: str, bigram2: str) -> float:
        """
        Score a bigram pair based on model uncertainty.
        Higher score = model is more uncertain about preference.
        """
        try:
            # Get model's prediction probability
            pred_prob = self.model.predict_preference(bigram1, bigram2)
            
            # Higher score when prediction is closer to 0.5 (uncertain)
            uncertainty = 1 - abs(pred_prob - 0.5) * 2
            
            return float(uncertainty)
            
        except Exception as e:
            logger.warning(f"Error scoring pair {bigram1}-{bigram2}: {e}")
            return 0.0

    def get_recommended_pairs(self) -> List[Tuple[str, str]]:
        """Get recommended bigram pairs based on multiple criteria."""
        # Get candidate pairs
        candidate_pairs = self._generate_candidate_pairs()
        
        # Score each candidate pair
        scored_pairs = []
        weights = self.config.get('recommendations', {}).get('weights', {
            'uncertainty': 0.4,
            'interaction': 0.3,
            'transitivity': 0.3
        })
        
        for pair in candidate_pairs:
            try:
                uncertainty_score = self.score_pair(*pair)
                interaction_score = self._calculate_interaction_score(*pair)
                transitivity_score = self._calculate_transitivity_score(*pair)
                
                total_score = (
                    weights['uncertainty'] * uncertainty_score +
                    weights['interaction'] * interaction_score +
                    weights['transitivity'] * transitivity_score
                )
                
                scored_pairs.append((pair, total_score))
            except Exception as e:
                logger.warning(f"Error scoring pair {pair}: {str(e)}")
                continue
            
        # Filter and return top pairs
        return self._filter_top_pairs(scored_pairs)

    def visualize_recommendations(self, recommended_pairs: List[Tuple[str, str]]):
        """
        Create two plots: current data and current+recommended pairs
        Uses PCA to project feature space into 2D for visualization
        """
        # Get feature vectors for all bigrams
        feature_vectors = []
        bigram_labels = []
        
        for pref in self.dataset.preferences:
            # Get feature vectors and handle NaN values
            try:
                feat1 = [pref.features1.get(f, 0.0) for f in self.selected_features]
                feat2 = [pref.features2.get(f, 0.0) for f in self.selected_features]
                
                # Replace NaN with 0.0
                feat1 = [0.0 if pd.isna(x) else x for x in feat1]
                feat2 = [0.0 if pd.isna(x) else x for x in feat2]
                
                feature_vectors.extend([feat1, feat2])
                bigram_labels.extend([pref.bigram1, pref.bigram2])
            except Exception as e:
                logger.warning(f"Skipping preference due to feature error: {e}")
                continue

        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Verify no NaN values remain
        if np.any(np.isnan(X)):
            logger.warning("NaN values found in feature matrix after preprocessing")
            # Replace any remaining NaNs with 0
            X = np.nan_to_num(X, nan=0.0)
        
        # Fit PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Current data
        # Plot nodes
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], c='lightblue', s=100)
        
        # Add labels
        for i, label in enumerate(bigram_labels):
            ax1.annotate(label, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
        
        # Plot edges for existing pairs
        for i in range(0, len(bigram_labels), 2):
            ax1.plot([X_2d[i, 0], X_2d[i+1, 0]], 
                    [X_2d[i, 1], X_2d[i+1, 1]], 
                    'gray', alpha=0.5)
        
        ax1.set_title("Current Bigram Pairs")
        
        # Plot 2: Current + Recommended
        # Copy first plot
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], c='lightblue', s=100)
        for i, label in enumerate(bigram_labels):
            ax2.annotate(label, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
        
        # Plot existing edges
        for i in range(0, len(bigram_labels), 2):
            ax2.plot([X_2d[i, 0], X_2d[i+1, 0]], 
                    [X_2d[i, 1], X_2d[i+1, 1]], 
                    'gray', alpha=0.5)
        
        # Add recommended pairs
        for b1, b2 in recommended_pairs:
            try:
                # Get feature vectors
                feat1 = [self._extract_features(b1).get(f, 0.0) for f in self.selected_features]
                feat2 = [self._extract_features(b2).get(f, 0.0) for f in self.selected_features]
                
                # Replace NaN with 0.0
                feat1 = [0.0 if pd.isna(x) else x for x in feat1]
                feat2 = [0.0 if pd.isna(x) else x for x in feat2]
                
                # Project to 2D
                new_points = pca.transform(np.array([feat1, feat2]))
                
                # Plot new points
                ax2.scatter(new_points[:, 0], new_points[:, 1], c='red', s=100)
                
                # Add labels
                for i, label in enumerate([b1, b2]):
                    ax2.annotate(label, (new_points[i, 0], new_points[i, 1]), 
                                fontsize=8, color='red')
                
                # Add edge
                ax2.plot([new_points[0, 0], new_points[1, 0]], 
                        [new_points[0, 1], new_points[1, 1]], 
                        'red', linewidth=2)
            except Exception as e:
                logger.warning(f"Error plotting recommended pair {b1}-{b2}: {e}")
                continue
        
        ax2.set_title("Current + Recommended Pairs")
        
        plt.tight_layout()
        
        # Save plot if configured
        if self.config.get('recommendations', {}).get('save_plots', False):
            output_dir = Path(self.config['data']['output_dir']) / 'plots'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'bigram_recommendations.png')
            logger.info(f"Saved plot to {output_dir / 'bigram_recommendations.png'}")
        
        plt.show()
        
        # Print feature space coverage statistics
        print("\nFeature Space Coverage:")
        print(f"Variance explained by 2D projection: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"Number of existing pairs: {len(self.dataset.preferences)}")
        print(f"Number of recommended pairs: {len(recommended_pairs)}")
        
    def _calculate_interaction_score(self, bigram1: str, bigram2: str) -> float:
        """
        Score how well a pair tests feature interactions.
        Uses interaction features from the model's feature weights.
        """
        try:
            # Get feature weights from model
            feature_weights = self.model.get_feature_weights()
            
            # Extract features for both bigrams
            features1 = self._extract_features(bigram1)
            features2 = self._extract_features(bigram2)
            
            interaction_score = 0.0
            # Look for interaction features (those with '_x_' in name)
            interaction_features = [f for f in feature_weights.keys() if '_x_' in f]
            
            for interaction_feature in interaction_features:
                try:
                    # Split interaction feature name into component features
                    components = interaction_feature.split('_x_')
                    if len(components) != 2:
                        continue
                    feat1, feat2 = components
                    
                    if feat1 in features1 and feat2 in features1:
                        # Calculate interaction difference
                        int1 = features1[feat1] * features1[feat2]
                        int2 = features2[feat1] * features2[feat2]
                        weight = feature_weights[interaction_feature]
                        interaction_score += abs(int1 - int2) * abs(weight)
                except ValueError:
                    continue
            
            # Normalize score
            if interaction_features:
                interaction_score /= len(interaction_features)
                
            return float(interaction_score)
            
        except Exception as e:
            logger.warning(f"Error calculating interaction score: {str(e)}")
            return 0.0

    def _calculate_transitivity_score(self, bigram1: str, bigram2: str) -> float:
        """
        Score how valuable a pair would be for transitivity testing.
        Higher score = pair better validates transitivity.
        """
        try:
            pref_graph = self._build_preference_graph()
            
            # Look for potential transitive chains
            score = 0.0
            for intermediate in pref_graph:
                if (bigram1 in pref_graph.get(intermediate, set()) and 
                    intermediate in pref_graph.get(bigram2, set())):
                    # Found potential chain, increase score
                    score += 1.0
                    
            return float(score)
            
        except Exception as e:
            logger.warning(f"Error calculating transitivity score: {str(e)}")
            return 0.0
            
    def _filter_top_pairs(self, scored_pairs: List[Tuple[Tuple[str, str], float]]) -> List[Tuple[str, str]]:
        # Sort by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        selected_pairs = []
        covered_features = set()
        
        for pair, score in scored_pairs:
            # Extract features for this pair
            features1 = self._extract_features(pair[0])
            features2 = self._extract_features(pair[1])
            
            # Calculate feature coverage
            new_features = set()
            for feat, val1 in features1.items():
                val2 = features2[feat]
                if abs(val1 - val2) > 0.1:  # Significant difference threshold
                    new_features.add(feat)
                    
            # Add pair if it covers new features or has high score
            if len(new_features - covered_features) > 0 or score > 0.8:
                selected_pairs.append(pair)
                covered_features.update(new_features)
                
            if len(selected_pairs) >= self.n_recommendations:
                break
                
        return selected_pairs
    
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

    def _generate_candidate_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate all possible bigram pairs from layout characters.
        Excludes pairs that already exist in the dataset.
        """
        # Get existing pairs
        existing_pairs = set()
        for pref in self.dataset.preferences:
            pair = (pref.bigram1, pref.bigram2)
            existing_pairs.add(pair)
            existing_pairs.add((pair[1], pair[0]))  # Add reverse pair too
            
        # Generate all possible bigrams first
        possible_bigrams = []
        for char1, char2 in combinations(self.layout_chars, 2):
            bigram = char1 + char2
            possible_bigrams.append(bigram)
            # Add reverse bigram
            reverse_bigram = char2 + char1
            possible_bigrams.append(reverse_bigram)
            
        # Generate all possible pairs of bigrams
        candidate_pairs = []
        for b1, b2 in combinations(possible_bigrams, 2):
            pair = (b1, b2)
            # Skip if pair (or its reverse) already exists in dataset
            if pair not in existing_pairs and (b2, b1) not in existing_pairs:
                candidate_pairs.append(pair)
                
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs "
                f"from {len(self.layout_chars)} characters")
        
        # Optionally limit number of candidates for computational efficiency
        max_candidates = self.config.get('recommendations', {}).get('max_candidates', 1000)
        if len(candidate_pairs) > max_candidates:
            logger.info(f"Randomly sampling {max_candidates} candidates")
            candidate_pairs = random.sample(candidate_pairs, max_candidates)
            
        return candidate_pairs
    
    def _build_preference_graph(self) -> Dict[str, Set[str]]:
        """Build graph of existing preferences."""
        pref_graph = {}
        
        # Build graph from dataset preferences
        for pref in self.dataset.preferences:
            better = pref.bigram1 if pref.preferred else pref.bigram2
            worse = pref.bigram2 if pref.preferred else pref.bigram1
                
            if better not in pref_graph:
                pref_graph[better] = set()
            pref_graph[better].add(worse)
        
        return pref_graph



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