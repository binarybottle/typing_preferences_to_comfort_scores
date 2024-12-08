import logging

# Configure logging at the start
logging.basicConfig(
    level=logging.INFO,  # Set console output to INFO level
    format='%(message)s'  # Simple format for console
)

import argparse
import logging
from pathlib import Path
import yaml
import json
import numpy as np
from typing import Dict, Any

from engram3.data import PreferenceDataset
from engram3.analysis import analyze_feature_importance, find_sparse_regions
from engram3.utils import setup_logging
from engram3.models.bayesian import BayesianPreferenceModel
from engram3.features.selection import FeatureEvaluator
from engram3.features.recommendations import BigramRecommender

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_output_directories(config: Dict[str, Any]) -> None:
    """Create necessary output directories."""
    dirs_to_create = [
        Path(config['paths']['base']),
        Path(config['paths']['analysis']),
        Path(config['logging']['file']).parent,
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
    parser.add_argument('--n_repetitions', type=int, default=10,
                        help='Number of feature selection repetitions')
    parser.add_argument('--generate_recommendations', action='store_true',
                        help='Generate recommended bigram pairs')
    args = parser.parse_args()

    try:
        # Load configuration and setup
        config = load_config(args.config)
        create_output_directories(config)
        
        # Use feature evaluation random seed instead of data splits seed
        np.random.seed(config['feature_evaluation']['random_seed'])
        
        # Load dataset
        logger.info(f"Loading data from {config['data']['file']}")
        dataset = PreferenceDataset(config['data']['file'])
        logger.info(f"Loaded {len(dataset.preferences)} preferences from "
                   f"{len(dataset.participants)} participants")
        
        #-------------------------------------------------------------------
        # FEATURE SELECTION
        #-------------------------------------------------------------------
        if args.mode == 'select_features':
            logger.info("Starting feature selection phase...")
            
            # Run basic analyses first
            if config['analysis']['check_transitivity']:
                transitivity_results = dataset.check_transitivity()

            if config['analysis']['analyze_features']:
                importance = analyze_feature_importance(dataset)
                logger.info("Initial feature importance scores:")
                for feature, score in sorted(importance['correlations'].items(), 
                                          key=lambda x: abs(x[1]), 
                                          reverse=True):
                    logger.info(f"  {feature}: {score:.3f}")

            if config['analysis']['find_sparse_regions']:
                sparse_points = find_sparse_regions(dataset)
                logger.info(f"Found {len(sparse_points)} points in sparse regions")
            
            # Initialize evaluator with config settings
            evaluator = FeatureEvaluator(
                importance_threshold=config['feature_evaluation']['thresholds']['importance'],
                stability_threshold=config['feature_evaluation']['thresholds']['stability'],
                correlation_threshold=config['feature_evaluation']['thresholds']['correlation']
            )
            
            # Run feature selection with all config settings
            feature_set_config = config['feature_evaluation']['feature_sets'][0]
            
            # Validate minimum sample sizes
            if len(dataset.preferences) < config['feature_evaluation']['validation']['min_training_samples']:
                logger.warning("Dataset smaller than minimum recommended size")
            
            selected_features, diagnostics = evaluator.run_feature_selection(
                dataset,
                n_repetitions=args.n_repetitions,
                output_dir=Path(config['feature_evaluation']['output_dir']),
                feature_set_config=feature_set_config
            )
            
            # Save results based on config settings
            if config['feature_evaluation']['reporting']['save_fold_details']:
                details_file = Path(config['feature_evaluation']['output_dir']) / 'fold_details.json'
                if 'fold_details' in diagnostics:  # Add this check
                    with open(details_file, 'w') as f:
                        json.dump(diagnostics['fold_details'], f, indent=2)
                else:
                    logger.warning("No fold details available in diagnostics")
            
            # Save selected features to CSV
            features_file = Path(config['feature_evaluation']['output_dir']) / 'selected_features.csv'
            with open(features_file, 'w') as f:
                # Write header with metadata as comments
                f.write(f"# Feature selection configuration:\n")
                f.write(f"# n_samples: {config['feature_evaluation']['n_samples']}\n")
                f.write(f"# n_splits: {config['feature_evaluation']['n_splits']}\n")
                f.write(f"# random_seed: {config['feature_evaluation']['random_seed']}\n")
                f.write(f"\n# Selected features:\n")
                f.write("feature_name\n")  # CSV header
                for feature in selected_features:
                    f.write(f"{feature}\n")
            
            logger.info(f"Feature selection completed. Selected {len(selected_features)} features.")

        #-------------------------------------------------------------------
        # BIGRAM PAIR DATA COLLECTION RECOMMENDATIONS
        #-------------------------------------------------------------------
        elif args.mode == "generate_recommendations":

            logger.info("Generating bigram pair recommendations...")
            
            # Load selected features from CSV instead of JSON
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
            output_dir = Path(config['recommendations']['output_dir'])
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