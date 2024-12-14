# recommendations.py
"""
Recommendation system for selecting informative bigram pairs.
Implements BigramRecommender class which:
- Generates candidate bigram pairs
- Scores pairs using multiple criteria (uncertainty, coverage, etc.)
- Visualizes recommendations in feature space
- Exports detailed recommendation data
"""
from pathlib import Path
from typing import List, Tuple, Dict, Set, Union
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

from engram3.utils.config import Config, RecommendationsConfig
from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.features.feature_extraction import FeatureExtractor
from engram3.utils.visualization import PlottingUtils
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class BigramRecommender:

    def __init__(self, dataset: PreferenceDataset, model: PreferenceModel, config: Union[Dict, Config]):
        """Initialize BigramRecommender.
        
        Args:
            dataset: Dataset containing preferences
            model: Trained preference model
            config: Configuration object or dictionary
        """
        # Handle config
        if isinstance(config, dict):
            self.config = Config(**config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise ValueError(f"Config must be a dictionary or Config object, got {type(config)}")

        # Store basic attributes
        self.dataset = dataset
        self.model = model
        self.feature_extractor = model.feature_extractor
        self.selected_features = list(model.get_feature_weights().keys())
        
        # Get configuration values
        self.n_recommendations = config.recommendations.n_recommendations
        self.max_candidates = config.recommendations.max_candidates
        self.weights = config.recommendations.weights
        
        # Set up paths and utilities
        self.output_dir = Path(config.paths.plots_dir)
        self.layout_chars = config.data.layout["chars"]
        
        # Initialize components
        self.importance_calculator = model.importance_calculator
        self.plotting = PlottingUtils(config.paths.plots_dir)

    def get_recommended_pairs(self) -> List[Tuple[str, str]]:
        """
        Get recommended bigram pairs using comprehensive scoring metrics.
        
        Returns:
            List of tuples containing recommended bigram pairs
            
        Raises:
            FileNotFoundError: If feature metrics file not found
            ValueError: If configuration is invalid
            RuntimeError: If scoring or visualization fails
        """
        try:
            # Validate configuration
            if not getattr(self.config, 'recommendations'):
                raise ValueError("Missing recommendations configuration")
                
            # Get candidate pairs
            candidate_pairs = self._generate_candidate_pairs()
            logger.info(f"Generated {len(candidate_pairs)} candidate pairs")
            
            # Get scoring weights with validation
            weights = self._get_scoring_weights()
            
            # Load and validate feature metrics
            feature_metrics = self._load_feature_metrics()
            
            # Score candidate pairs
            scored_pairs: List[Tuple[Tuple[str, str], float, Dict[str, float]]] = []
            
            for pair in candidate_pairs:
                try:
                    # Calculate comprehensive scores
                    scores_detail = self._calculate_pair_scores(
                        pair=pair,
                        feature_metrics=feature_metrics,
                        weights=weights
                    )
                    
                    if scores_detail['total'] > 0:  # Only keep positive scores
                        scored_pairs.append((pair, scores_detail['total'], scores_detail))
                        
                except Exception as e:
                    logger.warning(
                        f"Error scoring pair {pair}: {str(e)}\n"
                        f"Skipping pair"
                    )
                    continue
            
            if not scored_pairs:
                raise RuntimeError("No valid scored pairs generated")
                
            # Sort by score and get top recommendations
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            n_recommendations = min(
                self.config.recommendations.n_recommendations,
                len(scored_pairs)
            )
            
            recommended_pairs = scored_pairs[:n_recommendations]
            
            # Save detailed results
            try:
                self._save_recommendation_results(recommended_pairs)
            except Exception as e:
                logger.error(f"Failed to save recommendation results: {str(e)}")
            
            # Generate visualizations if enabled
            if self.config['recommendations'].get('save_plots', False):
                try:
                    self.visualize_recommendations([pair for pair, _, _ in recommended_pairs])
                except Exception as e:
                    logger.error(f"Failed to generate visualizations: {str(e)}")
            
            # Log summary
            logger.info(f"\nGenerated {n_recommendations} recommendations:")
            for pair, score, _ in recommended_pairs[:5]:  # Show top 5
                logger.info(f"  {pair[0]}-{pair[1]}: {score:.4f}")
            
            return [pair for pair, _, _ in recommended_pairs]
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            raise

    def _get_scoring_weights(self) -> Dict[str, float]:
        """Get and validate scoring weights from config."""
        weights = getattr(self.config, 'recommendations', {}).get('weights', {
            'prediction_uncertainty': 0.25,
            'comfort_uncertainty': 0.15,
            'feature_space': 0.15,
            'correlation': 0.075,
            'mutual_information': 0.075,
            'stability': 0.1,
            'interaction_testing': 0.1,
            'transitivity': 0.1
        })
        
        # Validate weights sum to 1
        total = sum(weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
            
        return weights

    def _load_feature_metrics(self) -> pd.DataFrame:
        """Load and validate feature metrics from file."""
        metrics_file = Path(self.config.feature_selection.metrics_file)
        
        if not metrics_file.exists():
            raise FileNotFoundError(
                f"Feature metrics file not found: {metrics_file}"
            )
            
        feature_metrics = pd.read_csv(metrics_file)
        
        required_columns = {'feature_name', 'correlation', 'mutual_information'}
        missing_cols = required_columns - set(feature_metrics.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns in metrics file: {missing_cols}")
            
        return feature_metrics

    def visualize_recommendations(self, recommended_pairs: List[Tuple[str, str]]):
        """
        Create two plots: current data and current+recommended pairs
        Uses PCA to project feature space into 2D for visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        self.plotting.setup_axis(ax1, title="Current Bigram Pairs")
        self.plotting.setup_axis(ax2, title="Current + Recommended Pairs")

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
        
        # Save plot
        plt.savefig(config.paths.plots_dir / 'bigram_recommendations.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {config.paths.plots_dir / 'bigram_recommendations.png'}")
        
        plt.show()
        
        # Print feature space coverage statistics
        print("\nFeature Space Coverage:")
        print(f"Total variance explained by 2D projection: {(var1 + var2):.1f}%")
        print(f"Number of existing pairs: {len(self.dataset.preferences)}")
        print(f"Number of recommended pairs: {len(recommended_pairs)}")
            
    def _calculate_pair_scores(self, pair: Tuple[str, str], 
                             feature_metrics: pd.DataFrame,
                             weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate all scores for a bigram pair."""
        try:
            # 1. Get prediction uncertainty
            pred_mean, pred_std = self.model.predict_preference(*pair)
            prediction_score = pred_std
            
            # 2. Get comfort score uncertainties
            comfort1_mean, comfort1_std = self.model.get_bigram_comfort_scores(pair[0])
            comfort2_mean, comfort2_std = self.model.get_bigram_comfort_scores(pair[1])
            comfort_score = (comfort1_std + comfort2_std) / 2
            
            # 3. Calculate feature space coverage score
            feature_score = self._calculate_feature_space_score(pair)
            
            # 4. Get correlation and mutual information scores
            correlation_score = self._get_metric_difference(pair, feature_metrics, 'correlation')
            mi_score = self._get_metric_difference(pair, feature_metrics, 'mutual_information')
            
            # 5. Get stability score from effect CV
            stability_score = self._get_metric_average(pair, feature_metrics, 'effect_cv')
            
            # 6. Calculate interaction testing score
            interaction_score = self._calculate_interaction_score(pair[0], pair[1])
            
            # 7. Calculate transitivity testing score
            transitivity_score = self._calculate_transitivity_score(pair[0], pair[1])
            
            # Combine all scores
            total_score = sum(
                weight * score for weight, score in [
                    (weights['prediction_uncertainty'], prediction_score),
                    (weights['comfort_uncertainty'], comfort_score),
                    (weights['feature_space'], feature_score),
                    (weights['correlation'], correlation_score),
                    (weights['mutual_information'], mi_score),
                    (weights['stability'], 1.0 - stability_score),
                    (weights['interaction_testing'], interaction_score),
                    (weights['transitivity'], transitivity_score)
                ]
            )
            
            return {
                'prediction_uncertainty': prediction_score,
                'comfort_uncertainty': comfort_score,
                'feature_space': feature_score,
                'correlation': correlation_score,
                'mutual_information': mi_score,
                'stability': 1.0 - stability_score,
                'interaction_testing': interaction_score,
                'transitivity': transitivity_score,
                'total': total_score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating scores for pair {pair}: {str(e)}")
            return {k: 0.0 for k in weights.keys() | {'total'}}

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
          
    def _calculate_interaction_score(self, bigram1: str, bigram2: str) -> float:
        """
        Calculate interaction score for bigram pair.
        
        Args:
            bigram1: First bigram to evaluate
            bigram2: Second bigram to evaluate
            
        Returns:
            float: Interaction score between the bigrams
            
        Notes:
            Uses importance calculator to evaluate feature interactions
            and returns normalized score between 0 and 1
        """
        try:
            # Input validation
            if not isinstance(bigram1, str) or not isinstance(bigram2, str):
                raise ValueError(f"Bigrams must be strings, got {type(bigram1)} and {type(bigram2)}")
                
            if len(bigram1) != 2 or len(bigram2) != 2:
                raise ValueError(f"Bigrams must be exactly 2 characters, got '{bigram1}' and '{bigram2}'")
                
            # Get features for both bigrams using model's feature extractor
            features1 = self.model._extract_features(bigram1)
            features2 = self.model._extract_features(bigram2)
            
            # Calculate feature differences for importance metrics
            feature_diffs = {
                name: features1.get(name, 0.0) - features2.get(name, 0.0)
                for name in self.model.selected_features
            }
            
            # Get metrics using importance calculator
            metrics = self.model.importance_calculator.calculate_feature_importance(
                feature_diffs=feature_diffs,
                dataset=self.dataset,
                model=self.model
            )
            
            # Calculate interaction score
            interaction_score = self.model.importance_calculator.calculate_interaction_score(metrics)
            
            # Log detailed metrics for debugging
            logger.debug(f"Interaction metrics for {bigram1}-{bigram2}:")
            for metric_name, value in metrics.items():
                logger.debug(f"  {metric_name}: {value:.4f}")
            
            return float(interaction_score)
            
        except ValueError as e:
            # Re-raise validation errors with context
            raise ValueError(f"Invalid bigrams for interaction score: {str(e)}")
        except Exception as e:
            logger.warning(
                f"Error calculating interaction score for {bigram1}-{bigram2}: {str(e)}\n"
                f"Returning zero score"
            )
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
                        
    def _generate_candidate_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible bigram pairs from layout characters."""
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
        
        # Optionally limit number of candidates
        max_candidates = self.config.max_candidates  # Direct attribute access
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

    def _get_metric_difference(self, pair: Tuple[str, str], 
                            metrics_df: pd.DataFrame, 
                            metric_name: str) -> float:
        """Get difference in metric values between bigrams."""
        try:
            value1 = metrics_df[metrics_df['feature_name'] == pair[0]][metric_name].iloc[0]
            value2 = metrics_df[metrics_df['feature_name'] == pair[1]][metric_name].iloc[0]
            return abs(value1 - value2)
        except Exception:
            return 0.0

    def _get_metric_average(self, pair: Tuple[str, str],
                        metrics_df: pd.DataFrame,
                        metric_name: str) -> float:
        """Get average metric value for bigram pair."""
        try:
            value1 = metrics_df[metrics_df['feature_name'] == pair[0]][metric_name].iloc[0]
            value2 = metrics_df[metrics_df['feature_name'] == pair[1]][metric_name].iloc[0]
            return (value1 + value2) / 2
        except Exception:
            return 0.0
                
    def _save_recommendation_results(self, scored_pairs: List[Tuple]):
        """Save detailed recommendation results."""
        # Log detailed scoring information
        logger.info("\nTop recommended pairs with scores:")
        for pair, total_score, scores_detail in scored_pairs[:self.n_recommendations]:
            logger.info(f"\nPair: {pair[0]}-{pair[1]}")
            logger.info(f"Total score: {total_score:.3f}")
            for metric, score in scores_detail.items():
                logger.info(f"  {metric}: {score:.3f}")
        
        # Save detailed scores
        scores_df = pd.DataFrame([
            {
                'bigram1': pair[0],
                'bigram2': pair[1],
                **scores_detail
            }
            for pair, _, scores_detail in scored_pairs[:self.n_recommendations]
        ])
        
        # Save to output files
        scores_df.to_csv(self.config.paths.root_dir / 'detailed_scores.csv', index=False)
        
        # Simple pair list for backward compatibility
        pd.DataFrame([
            {'bigram1': pair[0], 'bigram2': pair[1]}
            for pair, _, _ in scored_pairs[:self.n_recommendations]
        ]).to_csv(self.config.recommendations.recommendations_file, index=False)

    
