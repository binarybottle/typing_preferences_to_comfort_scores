# comprehensive_pair_recommender.py
# python comprehensive_pair_recommender.py --config config.yaml --count 100 --output comprehensive_pairs.csv

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from typing_preferences_to_comfort_scores.utils.config import Config
from typing_preferences_to_comfort_scores.data import PreferenceDataset
from typing_preferences_to_comfort_scores.model import PreferenceModel
from typing_preferences_to_comfort_scores.recommendations import BigramRecommender
from typing_preferences_to_comfort_scores.features.feature_extraction import FeatureExtractor, FeatureConfig
from typing_preferences_to_comfort_scores.features.features import angles
from typing_preferences_to_comfort_scores.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array

# I renamed the repository/module, so need to symlink the module name in the pickle files
# Create an alias from old to new package name
import sys
import importlib
sys.modules['engram3'] = importlib.import_module('typing_preferences_to_comfort_scores')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

# Add a custom method to BigramRecommender to generate comprehensive recommendations
def recommend_comprehensive_pairs(recommender, include_existing=False):
    """
    Generate recommendations covering the entire feature space, optionally ignoring existing data.
    
    Args:
        recommender: BigramRecommender instance
        include_existing: Whether to consider existing data points when calculating distances
    
    Returns:
        List of recommended bigram pairs
    """
    try:
        recommender._initialize_state()
        
        candidates = recommender._generate_candidates()
        if not candidates:
            raise ValueError("No valid candidates generated")
            
        # Calculate features for each candidate pair
        candidate_differences = []
        for pair in candidates:
            diff = recommender._get_pair_features(pair[0], pair[1])
            candidate_differences.append(diff)
        candidate_differences = np.array(candidate_differences)

        # Get features for existing pairs if needed
        if include_existing:
            existing_differences = []
            for pref in recommender.dataset.preferences:
                if not any(c in recommender.excluded_chars for c in pref.bigram1 + pref.bigram2):
                    diff = recommender._get_pair_features(pref.bigram1, pref.bigram2)
                    existing_differences.append(diff)
            existing_differences = np.array(existing_differences)
            
            # Project to PCA space with existing data included
            pca = PCA(n_components=2)
            all_features = np.vstack([existing_differences, candidate_differences]) if len(existing_differences) > 0 else candidate_differences
            all_projected = pca.fit_transform(all_features)
            
            n_existing = len(existing_differences)
            existing_projected = all_projected[:n_existing] if n_existing > 0 else np.array([])
            candidate_projected = all_projected[n_existing:]
        else:
            # Project just the candidates to PCA space
            pca = PCA(n_components=2)
            candidate_projected = pca.fit_transform(candidate_differences)
            existing_projected = np.array([])
        
        # Initialize selections
        selected = []
        selected_points = []
        remaining_candidates = list(zip(candidates, candidate_projected))
        
        # First selection: furthest from origin (most extreme point)
        distances = [np.linalg.norm(point) for _, point in remaining_candidates]
        best_idx = np.argmax(distances)
        best_pair, best_point = remaining_candidates[best_idx]
        selected.append(best_pair)
        selected_points.append(best_point)
        remaining_candidates.pop(best_idx)
        
        # Subsequent selections: maximize minimum distance to already selected points
        # (optionally considering existing data)
        while len(selected) < recommender.n_recommendations and remaining_candidates:
            max_min_dist = -float('inf')
            best_idx = -1
            
            for i, (pair, point) in enumerate(remaining_candidates):
                # Calculate minimum distance to any selected point
                min_dist = float('inf')
                
                # Check distance to already selected points
                for sel_point in selected_points:
                    dist = np.linalg.norm(point - sel_point)
                    min_dist = min(min_dist, dist)
                
                # Optionally check distance to existing data points
                if include_existing and len(existing_projected) > 0:
                    for ex_point in existing_projected:
                        dist = np.linalg.norm(point - ex_point)
                        min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                best_pair, best_point = remaining_candidates[best_idx]
                selected.append(best_pair)
                selected_points.append(best_point)
                remaining_candidates.pop(best_idx)
                
                if len(selected) % 10 == 0:
                    print(f"Selected {len(selected)}/{recommender.n_recommendations} pairs")

        return selected

    except Exception as e:
        print(f"Error generating comprehensive recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive bigram recommendations')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--count', type=int, default=200, help='Number of recommendations to generate')
    parser.add_argument('--excluded_chars', type=str, default='', help='Characters to exclude (no spaces)')
    parser.add_argument('--model_path', type=str, help='Path to trained feature selection model')
    parser.add_argument('--output', type=str, default='comprehensive_bigram_pairs.csv', help='Output file path')
    parser.add_argument('--include_existing', action='store_true', help='Whether to consider existing data points')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override number of recommendations in config
    config.recommendations.n_recommendations = args.count
    
    # Set excluded characters
    excluded_chars = list(args.excluded_chars) if args.excluded_chars else []
    
    # Initialize feature extraction
    feature_config = FeatureConfig(
        column_map=column_map,
        row_map=row_map,
        finger_map=finger_map,
        engram_position_values=engram_position_values,
        row_position_values=row_position_values,
        angles=angles,
        bigrams=bigrams,
        bigram_frequencies_array=bigram_frequencies_array
    )
    feature_extractor = FeatureExtractor(feature_config)
    
    # Precompute features for all possible bigrams
    all_bigrams, all_bigram_features = feature_extractor.precompute_all_features(
        config.data.layout['chars']
    )
    
    # Get feature names from first computed features
    feature_names = list(next(iter(all_bigram_features.values())).keys())
    
    # Load dataset with precomputed features
    dataset = PreferenceDataset(
        Path(config.data.input_file),
        feature_extractor=feature_extractor,
        config=config,
        precomputed_features={
            'all_bigrams': all_bigrams,
            'all_bigram_features': all_bigram_features,
            'feature_names': feature_names
        }
    )
    
    # Load the model
    model_path = args.model_path or config.feature_selection.model_file
    model = PreferenceModel.load(Path(model_path))
    
    # Initialize recommender with excluded characters
    recommender = BigramRecommender(
        dataset, 
        model, 
        config,
        excluded_chars=excluded_chars
    )
    
    # Visualize initial state
    recommender.visualize_feature_space()
    recommender.visualize_feature_distributions()

    # Generate recommendations using our custom function
    print(f"Generating {args.count} comprehensive recommendations...")
    if args.include_existing:
        print("Including existing data points in distance calculations")
    else:
        print("Ignoring existing data points for more uniform coverage")
        
    recommended_pairs = recommend_comprehensive_pairs(recommender, include_existing=args.include_existing)
    
    # Visualize with recommendations
    recommender.visualize_feature_space_with_recommendations(recommended_pairs)
    
    # Save recommendations
    output_path = args.output
    pd.DataFrame(recommended_pairs, columns=['bigram1', 'bigram2']).to_csv(
        output_path, index=False
    )
    
    print(f"Generated {len(recommended_pairs)} comprehensive bigram pairs for data collection")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()