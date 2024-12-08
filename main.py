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
from engram3.features.feature_selection import FeatureEvaluator

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
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        create_output_directories(config)

        # Setup logging with both file and console handlers
        log_file = Path(config['logging']['file'])
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)
        
        # Console handler - show only INFO and above
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler - show all DEBUG and above
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Set random seed
        np.random.seed(config['data']['splits']['random_seed'])

        # Load dataset
        logger.info(f"Loading data from {config['data']['file']}")
        dataset = PreferenceDataset(config['data']['file'])
        logger.info(f"Loaded {len(dataset.preferences)} preferences from "
                   f"{len(dataset.participants)} participants")

        if args.mode == 'select_features':
            # Set up detailed logging first
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            # Force debug level for all loggers
            logging.getLogger().setLevel(logging.DEBUG)
            for handler in logging.getLogger().handlers:
                handler.setLevel(logging.DEBUG)
                
            logger.info("Starting feature selection phase...")
            
            # Basic analyses
            if config['analysis']['check_transitivity']:
                logger.info("Checking transitivity...")
                transitivity_results = dataset.check_transitivity()
                logger.info(f"Transitivity results: {transitivity_results}")

            if config['analysis']['analyze_features']:
                logger.info("Analyzing feature importance...")
                importance = analyze_feature_importance(dataset)
                logger.info("Feature importance scores:")
                for feature, score in sorted(importance['correlations'].items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True):
                    logger.info(f"  {feature}: {score:.3f}")

            if config['analysis']['find_sparse_regions']:
                logger.info("Finding sparse regions...")
                sparse_points = find_sparse_regions(dataset)
                logger.info(f"Found {len(sparse_points)} points in sparse regions")

            # Comprehensive feature evaluation
            if config['analysis']['evaluate_features']:
                logger.info("Starting comprehensive feature evaluation...")
                
                # Add debug logging setup here
                logging.getLogger().setLevel(logging.DEBUG)
                print("\n" + "="*50)
                print("DEBUG: SETTING UP FEATURE EVALUATION")
                print("="*50)
                
                # Initialize model and evaluator
                model = BayesianPreferenceModel()
                evaluator = FeatureEvaluator(
                    importance_threshold=config['feature_evaluation']['thresholds']['importance'],
                    stability_threshold=config['feature_evaluation']['thresholds']['stability'],
                    correlation_threshold=config['feature_evaluation']['thresholds']['correlation']
                )
                print("DEBUG: Created evaluator")       
                logger.debug("DEBUG: Created FeatureEvaluator")
                
                # Run feature selection
                selected_features, diagnostics = evaluator.run_feature_selection(
                    dataset,
                    n_repetitions=args.n_repetitions,
                    output_dir=Path(config['feature_evaluation']['output_dir'])
                )
                
                # Save selected features
                features_file = Path(config['feature_evaluation']['output_dir']) / 'selected_features.json'
                with open(features_file, 'w') as f:
                    json.dump(selected_features, f, indent=2)
                
                logger.info(f"Feature selection completed. Selected {len(selected_features)} features.")
                                    
        elif args.mode == 'train_model':
            logger.info("Starting model training phase...")
            
            # Load selected features
            features_file = Path(config['feature_evaluation']['output_dir']) / 'selected_features.json'
            if not features_file.exists():
                raise FileNotFoundError("Selected features file not found. Run feature selection first.")
                
            with open(features_file, 'r') as f:
                selected_features = json.load(f)
            
            logger.info(f"Using {len(selected_features)} selected features for training.")
            
            # Create train/test split
            train_data, test_data = dataset.split_by_participants(
                test_fraction=config['data']['splits']['test_ratio']
            )
            
            # Train model using selected features
            model = BayesianPreferenceModel()
            model.fit(train_data)  # You might need to add feature filtering here
            
            logger.info(f"Model training completed.")

        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Set logging to DEBUG level for detailed output
    logging.getLogger().setLevel(logging.DEBUG)
    main()