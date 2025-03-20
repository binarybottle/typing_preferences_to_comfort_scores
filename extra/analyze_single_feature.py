#!/usr/bin/env python3
"""
Analyze a single feature to build feature_metrics.csv incrementally.
"""
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Import from your existing modules
from bigram_typing_preferences_to_comfort_scores.utils.config import Config
from bigram_typing_preferences_to_comfort_scores.data import PreferenceDataset
from bigram_typing_preferences_to_comfort_scores.model import PreferenceModel
from bigram_typing_preferences_to_comfort_scores.features.feature_extraction import FeatureExtractor, FeatureConfig
from bigram_typing_preferences_to_comfort_scores.features.features import angles
from bigram_typing_preferences_to_comfort_scores.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from bigram_typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array
from bigram_typing_preferences_to_comfort_scores.utils.logging import LoggingManager

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_single_feature.py config.yaml feature_name")
        sys.exit(1)
    
    config_path = sys.argv[1]
    feature_name = sys.argv[2]
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    
    # Set model settings
    config.model.chains = 4
    config.model.warmup = 3000
    config.model.n_samples = 7000
    config.model.max_treedepth = 15
    config.model.adapt_delta = 0.95
    
    # Setup logging
    LoggingManager(config).setup_logging()
    logger = LoggingManager.getLogger(__name__)
    
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
    
    # Create model
    model = PreferenceModel(config=config)
    
    # Analyze just this feature
    control_features = config.features.control_features
    metrics = model._calculate_feature_importance(
        feature=feature_name,
        dataset=dataset,
        current_features=control_features
    )
    
    # Add feature name and metadata
    metrics['feature_name'] = feature_name
    metrics['selected'] = 0  # Not selected yet
    
    # Save to metrics file
    metrics_file = Path(config.feature_selection.metrics_file)
    metrics_df = pd.DataFrame([metrics])
    
    if metrics_file.exists():
        # Append to existing file
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        # Create new file
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Feature: {feature_name}")
    print(f"Effect magnitude: {metrics['effect_magnitude']:.6f}")
    print(f"Effect std dev: {metrics['effect_std']:.6f}")
    print(f"Importance score: {metrics['selected_importance']:.6f}")
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
