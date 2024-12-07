import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, Any

from engram3.data import PreferenceDataset
from engram3.analysis import analyze_feature_importance, find_sparse_regions
from engram3.utils import setup_logging
from engram3.models.simple import MockPreferenceModel

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
        Path(config['logging']['file']).parent
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preference Learning Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directories
        create_output_directories(config)
        
        # Setup logging
        setup_logging(Path(config['logging']['file']))
        logger = logging.getLogger(__name__)
        logger.info("Starting preference learning pipeline")

        # Set random seed
        np.random.seed(config['data']['splits']['random_seed'])

        # Load dataset
        logger.info(f"Loading data from {config['data']['file']}")
        dataset = PreferenceDataset(config['data']['file'])
        logger.info(f"Loaded {len(dataset.preferences)} preferences from "
                   f"{len(dataset.participants)} participants")

        # Perform analyses if configured
        if config['analysis']['check_transitivity']:
            logger.info("Checking transitivity...")
            transitivity_results = dataset.check_transitivity()
            logger.info(f"Transitivity results: {transitivity_results}")

        if config['analysis']['analyze_features']:
            logger.info("Analyzing feature importance...")
            importance = analyze_feature_importance(dataset)
            logger.info("Feature importance scores:")
            # Sort by absolute correlation values
            for feature, score in sorted(importance['correlations'].items(), 
                                        key=lambda x: abs(x[1]), 
                                        reverse=True):
                logger.info(f"  {feature}: {score:.3f}")

        if config['analysis']['find_sparse_regions']:
            logger.info("Finding sparse regions...")
            sparse_points = find_sparse_regions(dataset)
            logger.info(f"Found {len(sparse_points)} points in sparse regions")

        # Create train/test split
        logger.info("Creating train/test split...")
        train_data, test_data = dataset.split_by_participants(
            test_fraction=config['data']['splits']['test_ratio']
        )
        logger.info(f"Split dataset into {len(train_data.preferences)} train and "
                   f"{len(test_data.preferences)} test preferences")

        if config['analysis']['train_model']:
            logger.info("Training preference model...")
            model = MockPreferenceModel()
            model.fit(train_data)
            logger.info("Model training completed")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()