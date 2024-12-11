# engram3/recommendations.py

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
from itertools import combinations
import warnings
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx

from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.features.extraction import extract_bigram_features

logger = logging.getLogger(__name__)

class BigramRecommender:
    """Generate recommended bigram pairs for data collection."""
    
    def __init__(self, dataset: PreferenceDataset, model: PreferenceModel, config: Dict):
        self.dataset = dataset
        self.model = model  # Use existing trained model
        self.config = config
        self.selected_features = model.get_feature_weights().keys()
        self.n_recommendations = config['recommendations']['n_recommendations']

    def score_pair(self, bigram1: str, bigram2: str) -> float:
        # Get model's prediction confidence
        pred_prob = self.model.predict_preference(bigram1, bigram2)
        # Higher score for predictions closer to 0.5 (uncertain)
        uncertainty = 1 - abs(pred_prob - 0.5) * 2
        return uncertainty

    def get_recommended_pairs(self) -> List[Tuple[str, str]]:
        # Weights for different selection criteria
        weights = {
            'uncertainty': 0.4,  # Model prediction uncertainty
            'interaction': 0.3,  # Feature interaction coverage
            'transitivity': 0.3  # Transitivity validation
        }
        
        scored_pairs = []
        candidate_pairs = self._generate_candidate_pairs()
        
        for pair in candidate_pairs:
            uncertainty_score = self.score_pair(*pair)
            interaction_score = self._calculate_interaction_score(*pair)
            transitivity_score = self._calculate_transitivity_score(*pair)
            
            total_score = (
                weights['uncertainty'] * uncertainty_score +
                weights['interaction'] * interaction_score +
                weights['transitivity'] * transitivity_score
            )
            scored_pairs.append((pair, total_score))
            
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
                # Get feature vectors for existing pairs
                feat1 = [pref.features1[f] for f in self.selected_features]
                feat2 = [pref.features2[f] for f in self.selected_features]
                feature_vectors.extend([feat1, feat2])
                bigram_labels.extend([pref.bigram1, pref.bigram2])

            # Convert to numpy array
            X = np.array(feature_vectors)
            
            # Fit PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Current data
            G1 = nx.Graph()
            
            # Add nodes
            for i, (x, y) in enumerate(X_2d):
                G1.add_node(bigram_labels[i], pos=(x, y))
            
            # Add edges for existing pairs
            for pref in self.dataset.preferences:
                G1.add_edge(pref.bigram1, pref.bigram2)
            
            # Draw network
            pos1 = nx.get_node_attributes(G1, 'pos')
            nx.draw(G1, pos1, ax=ax1, node_size=100, 
                node_color='lightblue', with_labels=True, 
                font_size=8)
            ax1.set_title("Current Bigram Pairs")
            
            # Plot 2: Current + Recommended
            G2 = G1.copy()
            
            # Add recommended pairs
            for b1, b2 in recommended_pairs:
                # Get feature vectors
                feat1 = [self._extract_features(b1)[f] for f in self.selected_features]
                feat2 = [self._extract_features(b2)[f] for f in self.selected_features]
                
                # Project to 2D
                new_points = pca.transform(np.array([feat1, feat2]))
                
                # Add to graph
                G2.add_node(b1, pos=tuple(new_points[0]))
                G2.add_node(b2, pos=tuple(new_points[1]))
                G2.add_edge(b1, b2, color='red', weight=2)
            
            # Draw network
            pos2 = nx.get_node_attributes(G2, 'pos')
            
            # Draw existing edges
            nx.draw_networkx_edges(G2, pos2, ax=ax2, 
                                edgelist=[(u,v) for (u,v) in G2.edges() 
                                        if (u,v) not in recommended_pairs],
                                edge_color='gray')
            
            # Draw recommended edges
            nx.draw_networkx_edges(G2, pos2, ax=ax2,
                                edgelist=recommended_pairs,
                                edge_color='red', width=2)
            
            # Draw nodes and labels
            nx.draw_networkx_nodes(G2, pos2, ax=ax2, 
                                node_size=100, node_color='lightblue')
            nx.draw_networkx_labels(G2, pos2, ax=ax2, font_size=8)
            
            ax2.set_title("Current + Recommended Pairs")
            
            plt.tight_layout()
            plt.show()

            # Print feature space coverage statistics
            print("\nFeature Space Coverage:")
            print(f"Variance explained by 2D projection: {pca.explained_variance_ratio_.sum():.2%}")
            print(f"Number of existing pairs: {len(self.dataset.preferences)}")
            print(f"Number of recommended pairs: {len(recommended_pairs)}")

    def _calculate_interaction_score(self, bigram1: str, bigram2: str) -> float:
        # Get feature weights from model
        feature_weights = self.model.get_feature_weights()
        
        # Extract features for both bigrams
        features1 = self._extract_features(bigram1)
        features2 = self._extract_features(bigram2)
        
        interaction_score = 0.0
        for feat1, feat2 in self._get_interaction_pairs():
            if f"{feat1}_x_{feat2}" in feature_weights:
                # Calculate interaction difference
                int1 = features1[feat1] * features1[feat2]
                int2 = features2[feat1] * features2[feat2]
                weight = feature_weights[f"{feat1}_x_{feat2}"]
                interaction_score += abs(int1 - int2) * abs(weight)
                
        return interaction_score

    def _calculate_transitivity_score(self, bigram1: str, bigram2: str) -> float:
        pref_graph = self._build_preference_graph()
        
        # Look for potential transitive chains
        score = 0.0
        for intermediate in pref_graph:
            if (bigram1 in pref_graph.get(intermediate, set()) and 
                intermediate in pref_graph.get(bigram2, set())):
                # Found potential chain, increase score
                score += 1.0
                
        return score
    
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