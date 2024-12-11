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
    def __init__(self, dataset: PreferenceDataset, model: PreferenceModel, config: Dict):
        self.dataset = dataset
        self.model = model
        self.config = config
        self.selected_features = list(model.get_feature_weights().keys())
        self.n_recommendations = config['recommendations']['n_recommendations']
        self.layout_chars = config['data']['layout']['chars']

    def get_recommended_pairs(self) -> List[Tuple[str, str]]:
        """Get recommended bigram pairs based on multiple criteria."""
        candidate_pairs = self._generate_candidate_pairs()
        
        # Score each candidate pair
        scored_pairs = []
        weights = self.config.get('recommendations', {}).get('weights', {
            'prediction_uncertainty': 0.4,
            'comfort_uncertainty': 0.3,
            'feature_space': 0.3
        })
        
        for pair in candidate_pairs:
            try:
                # 1. Get prediction uncertainty
                pred_mean, pred_std = self.model.predict_preference(*pair)
                prediction_score = pred_std  # Higher uncertainty = higher score
                
                # 2. Get comfort score uncertainties
                comfort1_mean, comfort1_std = self.model.get_bigram_comfort_scores(pair[0])
                comfort2_mean, comfort2_std = self.model.get_bigram_comfort_scores(pair[1])
                comfort_score = (comfort1_std + comfort2_std) / 2
                
                # 3. Calculate feature space coverage score
                feature_score = self._calculate_feature_space_score(pair)
                
                # Combine scores
                total_score = (
                    weights['prediction_uncertainty'] * prediction_score +
                    weights['comfort_uncertainty'] * comfort_score +
                    weights['feature_space'] * feature_score
                )
                
                scored_pairs.append((pair, total_score))
                
            except Exception as e:
                logger.warning(f"Error scoring pair {pair}: {str(e)}")
                continue
        
        # Filter and return top pairs
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Log recommendation details
        logger.info("\nTop recommended pairs:")
        for pair, score in scored_pairs[:self.n_recommendations]:
            pred_mean, pred_std = self.model.predict_preference(*pair)
            logger.info(f"{pair[0]}-{pair[1]}: score={score:.3f} "
                       f"(pred={pred_mean:.3f}±{pred_std:.3f})")
        
        return [pair for pair, _ in scored_pairs[:self.n_recommendations]]

    def _calculate_feature_space_score(self, pair: Tuple[str, str]) -> float:
        """
        Calculate how well a pair covers sparse regions of feature space.
        """
        try:
            # Get features for both bigrams
            features1 = self._extract_features(pair[0])
            features2 = self._extract_features(pair[1])
            
            # Get feature vectors for existing data
            existing_vectors = []
            for pref in self.dataset.preferences:
                feat1 = [pref.features1.get(f, 0.0) for f in self.selected_features]
                feat2 = [pref.features2.get(f, 0.0) for f in self.selected_features]
                existing_vectors.extend([feat1, feat2])
            
            existing_vectors = np.array(existing_vectors)
            
            # Calculate minimum distances to existing points
            new_vector1 = np.array([features1.get(f, 0.0) for f in self.selected_features])
            new_vector2 = np.array([features2.get(f, 0.0) for f in self.selected_features])
            
            dist1 = np.min(np.linalg.norm(existing_vectors - new_vector1, axis=1))
            dist2 = np.min(np.linalg.norm(existing_vectors - new_vector2, axis=1))
            
            # Return average distance (higher = better coverage of sparse regions)
            return (dist1 + dist2) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating feature space score: {str(e)}")
            return 0.0
          
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

        # Convert to numpy array and standardize features
        X = np.array(feature_vectors)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
        
        # Verify no NaN values remain
        if np.any(np.isnan(X)):
            logger.warning("NaN values found in feature matrix after preprocessing")
            X = np.nan_to_num(X, nan=0.0)
        
        # Fit PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        # Calculate explained variance for axis labels
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        def setup_ax(ax):
            """Set up common axis properties"""
            ax.set_xlabel(f'PC1 ({var1:.1f}% variance)')
            ax.set_ylabel(f'PC2 ({var2:.1f}% variance)')
            ax.grid(True, alpha=0.3)
            
            # Add equal aspect ratio to prevent distortion
            ax.set_aspect('equal')
            
            # Add some padding to the limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            padding = 0.1
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            ax.set_xlim(xlim[0] - x_range*padding, xlim[1] + x_range*padding)
            ax.set_ylim(ylim[0] - y_range*padding, ylim[1] + y_range*padding)
        
        # Plot 1: Current data
        # Plot nodes
        scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], 
                            c='lightblue', s=100, alpha=0.6,
                            edgecolor='darkblue')
        
        # Add labels with offset for better readability
        for i, label in enumerate(bigram_labels):
            offset = 0.02 * (max(X_2d[:, 0]) - min(X_2d[:, 0]))
            ax1.annotate(label, 
                        (X_2d[i, 0] + offset, X_2d[i, 1] + offset),
                        fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Plot edges for existing pairs with curved lines
        for i in range(0, len(bigram_labels), 2):
            # Create curved line between points
            mid_point = [(X_2d[i, 0] + X_2d[i+1, 0])/2,
                        (X_2d[i, 1] + X_2d[i+1, 1])/2]
            # Add some curvature
            mid_point[1] += 0.05 * (max(X_2d[:, 1]) - min(X_2d[:, 1]))
            
            curve = plt.matplotlib.patches.ConnectionPatch(
                xyA=(X_2d[i, 0], X_2d[i, 1]),
                xyB=(X_2d[i+1, 0], X_2d[i+1, 1]),
                coordsA="data", coordsB="data",
                axesA=ax1, axesB=ax1,
                color='gray', alpha=0.3,
                connectionstyle="arc3,rad=0.2")
            ax1.add_patch(curve)
        
        ax1.set_title("Current Bigram Pairs", pad=20)
        setup_ax(ax1)
        
        # Plot 2: Current + Recommended
        scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1],
                            c='lightblue', s=100, alpha=0.6,
                            edgecolor='darkblue')
        
        # Copy labels from first plot
        for i, label in enumerate(bigram_labels):
            offset = 0.02 * (max(X_2d[:, 0]) - min(X_2d[:, 0]))
            ax2.annotate(label,
                        (X_2d[i, 0] + offset, X_2d[i, 1] + offset),
                        fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Add recommended pairs
        recommended_points = []
        recommended_labels = []
        
        for b1, b2 in recommended_pairs:
            try:
                # Get feature vectors
                feat1 = [self._extract_features(b1).get(f, 0.0) for f in self.selected_features]
                feat2 = [self._extract_features(b2).get(f, 0.0) for f in self.selected_features]
                
                # Standardize using same parameters as training data
                feat1 = (feat1 - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
                feat2 = (feat2 - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
                
                # Project to 2D
                new_points = pca.transform(np.array([feat1, feat2]))
                recommended_points.extend(new_points)
                recommended_labels.extend([b1, b2])
                
                # Add curved connection
                mid_point = [(new_points[0, 0] + new_points[1, 0])/2,
                            (new_points[0, 1] + new_points[1, 1])/2]
                mid_point[1] += 0.05 * (max(X_2d[:, 1]) - min(X_2d[:, 1]))
                
                curve = plt.matplotlib.patches.ConnectionPatch(
                    xyA=(new_points[0, 0], new_points[0, 1]),
                    xyB=(new_points[1, 0], new_points[1, 1]),
                    coordsA="data", coordsB="data",
                    axesA=ax2, axesB=ax2,
                    color='red', alpha=0.5,
                    connectionstyle="arc3,rad=0.2")
                ax2.add_patch(curve)
                
            except Exception as e:
                logger.warning(f"Error plotting recommended pair {b1}-{b2}: {e}")
                continue
        
        if recommended_points:
            recommended_points = np.array(recommended_points)
            scatter_rec = ax2.scatter(recommended_points[:, 0], recommended_points[:, 1],
                                    c='red', s=100, alpha=0.6,
                                    edgecolor='darkred', label='Recommended')
            
            # Add labels for recommended points
            for i, label in enumerate(recommended_labels):
                offset = 0.02 * (max(X_2d[:, 0]) - min(X_2d[:, 0]))
                ax2.annotate(label,
                            (recommended_points[i, 0] + offset, 
                            recommended_points[i, 1] + offset),
                            fontsize=8, color='darkred',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        ax2.set_title("Current + Recommended Pairs", pad=20)
        ax2.legend()
        setup_ax(ax2)
        
        plt.tight_layout()
        
        # Save plot if configured
        if self.config.get('recommendations', {}).get('save_plots', False):
            output_dir = Path(self.config['data']['output_dir']) / 'plots'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'bigram_recommendations.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_dir / 'bigram_recommendations.png'}")
        
        plt.show()
        
        # Print feature space coverage statistics
        print("\nFeature Space Coverage:")
        print(f"Total variance explained by 2D projection: {(var1 + var2):.1f}%")
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