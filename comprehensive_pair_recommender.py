# comprehensive_pair_recommender.py
# python comprehensive_pair_recommender.py --config config.yaml --count 100 --output comprehensive_pairs.csv

import argparse
import yaml
import pandas as pd
from pathlib import Path

from bigram_typing_preferences_to_comfort_scores.utils.config import Config
from bigram_typing_preferences_to_comfort_scores.data import PreferenceDataset
from bigram_typing_preferences_to_comfort_scores.model import PreferenceModel
from bigram_typing_preferences_to_comfort_scores.recommendations import BigramRecommender
from bigram_typing_preferences_to_comfort_scores.features.feature_extraction import FeatureExtractor, FeatureConfig
from bigram_typing_preferences_to_comfort_scores.features.features import angles
from bigram_typing_preferences_to_comfort_scores.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from bigram_typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive bigram recommendations')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--count', type=int, default=200, help='Number of recommendations to generate')
    parser.add_argument('--excluded_chars', type=str, default='', help='Characters to exclude (no spaces)')
    parser.add_argument('--model_path', type=str, help='Path to trained feature selection model')
    parser.add_argument('--output', type=str, default='comprehensive_bigram_pairs.csv', help='Output file path')
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
    
    # Visualize
    recommender.visualize_feature_space()
    recommender.visualize_feature_distributions()

    # Generate recommendations
    recommended_pairs = recommender.recommend_pairs()
    
    # Visualize with recommendations
    recommender.visualize_feature_space_with_recommendations(recommended_pairs)
    
    # Save recommendations
    pd.DataFrame(recommended_pairs, columns=['bigram1', 'bigram2']).to_csv(
        args.output, index=False
    )
    
    print(f"Generated {len(recommended_pairs)} comprehensive bigram pairs for data collection")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()