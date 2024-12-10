import logging

# Configure logging at the start
logging.basicConfig(
    level=logging.INFO,  # Set console output to INFO/DEBUG level
    format='%(message)s'  # Simple format for console
)

import argparse
import yaml
import json
import numpy as np
from typing import Dict, Any
from pathlib import Path

from engram3.data import PreferenceDataset
from engram3.utils import validate_config
from engram3.models.bayesian import BayesianPreferenceModel
from engram3.features.selection import FeatureEvaluator
from engram3.features.recommendations import BigramRecommender
from engram3.features.extraction import precompute_all_bigram_features
from engram3.features.bigram_frequencies import bigrams, bigram_frequencies_array
from engram3.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_directories(config: Dict[str, Any]) -> None:
    """Create necessary output directories."""
    dirs_to_create = [
        Path(config['data']['output_dir']),
        Path(config['logging']['output_file']).parent,
        Path(config['feature_evaluation']['output_dir'])
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    # Parse command line arguments    
    parser = argparse.ArgumentParser(description='Preference Learning Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['select_features', 'train_model'], required=True,
                       help='Pipeline mode: feature selection or model training')
    args = parser.parse_args()

    try:
        # Load configuration and setup
        config = load_config(args.config)
        validate_config(config)
        create_output_directories(config)
        
        # Use model's CV random seed instead of feature evaluation seed
        np.random.seed(config['model']['cross_validation']['random_seed'])        
        # Precompute bigram features
        logger.info("Precomputing bigram features...")
        all_bigrams, all_bigram_features, feature_names = precompute_all_bigram_features(
            layout_chars=config['data']['layout']['chars'],
            column_map=column_map,
            row_map=row_map,
            finger_map=finger_map,
            engram_position_values=engram_position_values,
            row_position_values=row_position_values,
            config=config
        )

        # Load dataset with precomputed features
        dataset = PreferenceDataset(
            config['data']['input_file'],
            column_map=column_map,
            row_map=row_map,
            finger_map=finger_map,
            engram_position_values=engram_position_values,
            row_position_values=row_position_values,
            precomputed_features={
                'all_bigrams': all_bigrams,
                'all_bigram_features': all_bigram_features,
                'feature_names': feature_names
            }
        )
            
        #-------------------------------------------------------------------
        # FEATURE SELECTION
        #-------------------------------------------------------------------
        if args.mode == 'select_features':
            logger.info("Starting feature selection phase...")
            
            # Initialize evaluator with config settings
            evaluator = FeatureEvaluator(config)

            # Run feature selection
            selected_features, diagnostics = evaluator.run_feature_selection(
                dataset,
                output_dir=Path(config['feature_evaluation']['output_dir']),
                feature_set_config=config['features']  # Pass entire features section
            )
            
            logger.info(f"\nFeature selection completed:")
            logger.info(f"- Selected {len(selected_features)} features")
            
            # Get importance metrics with fallback
            importance_metrics = diagnostics.get('importance', {})
            
            if selected_features:
                logger.info("\nSelected features:")
                for feature in selected_features:
                    importance = importance_metrics.get(feature, {}).get('combined_score', 0.0)
                    logger.info(f"  - {feature} (importance: {importance:.3f})")
            
            # Always show non-selected features
            non_selected = set(importance_metrics.keys()) - set(selected_features)
            if non_selected:
                logger.info("\nNon-selected features:")
                for feature in sorted(non_selected):
                    importance = importance_metrics.get(feature, {}).get('combined_score', 0.0)
                    logger.info(f"  - {feature} (importance: {importance:.3f})")
            
            if not importance_metrics:
                logger.warning("No importance metrics available")

        #-------------------------------------------------------------------
        # BIGRAM PAIR DATA COLLECTION RECOMMENDATIONS
        #-------------------------------------------------------------------
        elif args.mode == "recommend_bigram_pairs":

            logger.info("Generating bigram pair recommendations...")
            
            # Load selected features from CSV
            features_file = Path(config['feature_evaluation']['output_dir']) / 'selected_features.csv'
            if not features_file.exists():
                raise FileNotFoundError("Selected features file not found. Run feature selection first.")
            
            with open(features_file, 'r') as f:
                # Skip comment lines and header
                selected_features = [line.strip() for line in f 
                                  if line.strip() and not line.startswith('#') 
                                  and line.strip() != 'feature_name']
            
            # Initialize recommender
            recommender = BigramRecommender(dataset, config, selected_features)

            # Generate recommendations
            uncertainty_pairs = recommender.get_uncertainty_pairs()
            interaction_pairs = recommender.get_interaction_pairs()
            transitivity_pairs = recommender.get_transitivity_pairs()
            
            # Create output directory
            output_dir = Path(config['feature_evaluation']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save recommendations to CSV
            csv_file = output_dir / 'recommended_pairs.csv'
            with open(csv_file, 'w') as f:
                f.write("rank,type,bigram1,bigram2\n")
                for name, pairs in [
                    ('uncertainty', uncertainty_pairs),
                    ('interaction', interaction_pairs),
                    ('transitivity', transitivity_pairs)
                ]:
                    for i, (b1, b2) in enumerate(pairs, 1):
                        f.write(f"{i},{name},{b1},{b2}\n")
            
            logger.info(f"Generated recommendations saved to {csv_file}")

        #-------------------------------------------------------------------
        # MODEL TRAINING
        #-------------------------------------------------------------------
        elif args.mode == 'train_model':
            logger.info("Starting model training phase...")
            
            # Load selected features
            features_file = Path(config['feature_evaluation']['output_dir']) / 'selected_features.json'
            if not features_file.exists():
                raise FileNotFoundError("Selected features file not found. Run feature selection first.")
                
            with open(features_file, 'r') as f:
                selected_features_data = json.load(f)
                selected_features = selected_features_data['features']
            
            logger.info(f"Using {len(selected_features)} selected features for training.")
            
            # Create train/test split
            train_data, test_data = dataset.split_by_participants(
                test_fraction=config['data']['splits']['test_ratio']
            )
            
            # Initialize and train model with config
            model = BayesianPreferenceModel(config=config)
            model.fit(train_data)
            
            logger.info(f"Model training completed.")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()