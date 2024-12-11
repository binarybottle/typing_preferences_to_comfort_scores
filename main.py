import argparse
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import logging

from engram3.data import PreferenceDataset
from engram3.utils import validate_config
from engram3.model import PreferenceModel
from engram3.features.recommendations import BigramRecommender
from engram3.features.extraction import precompute_all_bigram_features
#from engram3.features.bigram_frequencies import bigrams, bigram_frequencies_array
from engram3.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)

logger = logging.getLogger(__name__)

# In main.py, modify the setup_logging function:

def setup_logging(config):
    # Create log directory if it doesn't exist
    log_file = Path(config['logging']['output_file'])
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Configure console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    console_handler.setFormatter(console_formatter)
    
    # Configure file handler with DEBUG level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    
    # Set root logger to DEBUG to capture all messages
    root_logger.setLevel(logging.DEBUG)
    
    # Add both handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_or_create_split(dataset: PreferenceDataset, config: Dict) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """Load existing split or create and save new one."""
    split_file = Path(config['data']['splits']['split_data_file'])
    
    if split_file.exists():
        logger.info("Loading existing train/test split...")
        split_data = np.load(split_file)
        train_indices = split_data['train_indices']
        test_indices = split_data['test_indices']
    else:
        logger.info("Creating new train/test split...")
        
        # Get participant IDs
        participant_ids = list(dataset.participants)
        n_participants = len(participant_ids)
        n_test = int(n_participants * config['data']['splits']['test_ratio'])
        
        # Randomly select test participants
        test_participants = set(np.random.choice(participant_ids, n_test, replace=False))
        
        # Get indices for train/test split
        train_indices = []
        test_indices = []
        for i, pref in enumerate(dataset.preferences):
            if pref.participant_id in test_participants:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        # Save split
        split_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_file, 
                 train_indices=train_indices, 
                 test_indices=test_indices)
    
    # Create datasets
    train_data = dataset._create_subset_dataset(train_indices)
    test_data = dataset._create_subset_dataset(test_indices)
    
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description='Preference Learning Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['select_features', 'train_model', 'recommend_bigram_pairs'], 
                       required=True,
                       help='Pipeline mode: feature selection, model training, or bigram recommendations')
    args = parser.parse_args()

    try:
        # Load configuration and setup
        config = load_config(args.config)
        validate_config(config)
        setup_logging(config)

        # Set random seed for all operations
        np.random.seed(config['data']['splits']['random_seed'])
        
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
        logger.info("Loading dataset...")
        dataset = PreferenceDataset(
            Path(config['data']['input_file']),
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

        #---------------------------------
        # Select features
        #---------------------------------
        if args.mode == 'select_features':

            # Get train/test split
            train_data, test_data = load_or_create_split(dataset, config)
            logger.info(f"Split dataset: {len(train_data.preferences)} train, "
                       f"{len(test_data.preferences)} test preferences")
            
            # Run feature selection on training data
            logger.info("Running feature selection on training data...")
            model = PreferenceModel(config=config)
            results = model.cross_validate(train_data)

        #---------------------------------
        # Recommend bigram pairs
        #---------------------------------
        elif args.mode == 'recommend_bigram_pairs':

            # Load feature metrics from previous feature selection
            metrics_file = Path(config['feature_evaluation']['metrics_file'])
            if not metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")
            
            df = pd.read_csv(metrics_file)
            selected_features = df[df['selected'] == 1]['feature_name'].tolist()
            
            if not selected_features:
                raise ValueError("No features were selected in feature selection phase")
            
            # Load trained model
            logger.info("Loading trained model...")
            model = PreferenceModel(config=config)
            model.fit(dataset, features=selected_features)
            
            # Initialize recommender
            logger.info("Generating bigram pair recommendations...")
            recommender = BigramRecommender(dataset, model, config)
            recommended_pairs = recommender.get_recommended_pairs()
            
            # Visualize recommendations
            logger.info("Visualizing recommendations...")
            recommender.visualize_recommendations(recommended_pairs)
            
            # Save recommendations
            output_file = Path(config['recommendations']['recommendations_file'])
            pd.DataFrame(recommended_pairs, columns=['bigram1', 'bigram2']).to_csv(
                output_file, index=False)
            logger.info(f"Saved recommendations to {output_file}")
            
            # Print recommendations
            logger.info("\nRecommended bigram pairs:")
            for b1, b2 in recommended_pairs:
                logger.info(f"{b1} - {b2}")

        #---------------------------------
        # Train model
        #---------------------------------
        elif args.mode == 'train_model':
            # Load train/test split
            train_data, test_data = load_or_create_split(dataset, config)
            
            # Load selected features
            metrics_file = config['feature_evaluation']['metrics_file']
            if not metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")
            
            df = pd.read_csv(metrics_file)
            selected_features = df[df['selected'] == 1]['feature_name'].tolist()
            
            if not selected_features:
                raise ValueError("No features were selected in feature selection phase")
            
            logger.info(f"Training model using {len(selected_features)} selected features...")
            
            # Train final model
            model = PreferenceModel(config=config)
            model.fit(train_data, features=selected_features)
            
            # Evaluate on test set
            logger.info("Evaluating model on test set...")
            test_metrics = model.evaluate(test_data)
            
            logger.info("\nTest set metrics:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.3f}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
