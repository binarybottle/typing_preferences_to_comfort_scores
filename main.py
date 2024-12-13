# main.py
"""
Main entry point for the Engram3 keyboard layout optimization system.
Handles command-line interface and orchestrates the three main workflows:
1. Feature selection - identifies important typing comfort features
2. Bigram recommendations - suggests new bigram pairs for preference collection
3. Model training - trains the final preference model using selected features
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.recommendations import BigramRecommender
from engram3.features.extraction import precompute_all_bigram_features
from engram3.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)

logger = logging.getLogger(__name__)

def setup_logging(config):
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log directory if it doesn't exist
    log_base = Path(config['logging']['output_file']).parent
    log_base.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename
    log_filename = f"debug_{timestamp}.log"
    log_file = log_base / log_filename

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

def load_or_create_split(dataset: PreferenceDataset, 
                            config: Dict) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """Load existing split or create and save new one."""
    split_file = Path(config['data']['splits']['split_data_file'])
    
    try:
        if split_file.exists():
            logger.info("Loading existing train/test split...")
            split_data = np.load(split_file)
            train_indices = split_data['train_indices']
            test_indices = split_data['test_indices']
            
            # Add validation
            if len(train_indices) + len(test_indices) > len(dataset.preferences):
                raise ValueError(f"Split indices ({len(train_indices)} train + {len(test_indices)} test) "
                            f"exceed total preferences ({len(dataset.preferences)})")
                            
            logger.info(f"Loaded split: {len(train_indices)} train, {len(test_indices)} test indices")
            
        else:
            logger.info("Creating new train/test split...")
            
            # Get all indices
            indices = np.arange(len(dataset.preferences))
            
            # Get test size from config
            test_ratio = config['data']['splits']['test_ratio']
            test_size = int(len(indices) * test_ratio)
            
            # Set random seed for reproducibility
            np.random.seed(config['data']['splits']['random_seed'])
            
            # Randomly select test indices
            test_indices = np.random.choice(indices, size=test_size, replace=False)
            train_indices = np.array([i for i in indices if i not in test_indices])
            
            # Save split
            split_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(split_file, train_indices=train_indices, test_indices=test_indices)
            
            logger.info(f"Created and saved new split: {len(train_indices)} train, "
                    f"{len(test_indices)} test indices")
        
        # Create datasets with validation
        train_data = dataset._create_subset_dataset(train_indices)
        test_data = dataset._create_subset_dataset(test_indices)
        
        logger.info(f"Created datasets: {len(train_data.preferences)} train, "
                f"{len(test_data.preferences)} test preferences")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in split creation/loading: {str(e)}")
        logger.error(f"Dataset preferences: {len(dataset.preferences)}")
        if split_file.exists():
            logger.error(f"Split file exists at: {split_file}")
        raise
    
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
            logger.info("Starting feature selection...")

            # Get train/test split
            train_data, holdout_data = load_or_create_split(dataset, config)
            logger.info(f"Split dataset: {len(train_data.preferences)} train, "
                    f"{len(holdout_data.preferences)} holdout preferences")
            
            # Initialize model
            model = PreferenceModel(config=config)
            
            # First select features
            selected_features = model.select_features(train_data)
            
            # Then fit model with selected features
            model.fit_model(train_data, selected_features)
            
            # Now get feature weights
            feature_weights = model.get_feature_weights()
            
            # Generate visualizations
            if model.visualizer:
                fig = model.visualizer.plot_feature_space(model, train_data, "Feature Space")
                fig.savefig(Path(config['data']['output_dir']) / 'final_feature_space.png')
                plt.close()
            
            # Create results DataFrame
            results = []
            for feature, (weight, std) in feature_weights.items():
                results.append({
                    'feature_name': feature,
                    'selected': 1 if feature in selected_features else 0,
                    'weight': weight,
                    'weight_std': std
                })
            
            # Save feature selection results
            metrics_file = Path(config['feature_evaluation']['metrics_file'])
            pd.DataFrame(results).to_csv(metrics_file, index=False)
            logger.info(f"Saved feature metrics to {metrics_file}")
            
            # Print summary
            logger.info(f"Selected features: {selected_features}")
            for feature in selected_features:
                weight, std = feature_weights[feature]
                logger.info(f"{feature}: {weight:.3f} ± {std:.3f}")
                
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
        
        #---------------------------------
        # Predict bigram scores
        #---------------------------------
        elif args.mode == 'predict_bigram_scores':
            # Load feature metrics and selected features
            metrics_file = Path(config['feature_evaluation']['metrics_file'])
            if not metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")
                    
            df = pd.read_csv(metrics_file)
            selected_features = df[df['selected'] == 1]['feature_name'].tolist()

            # Ensure model is trained
            logger.info(f"Loading trained model with {len(selected_features)} features...")
            model = PreferenceModel(config=config)
            model.fit(dataset, features=selected_features)

            # Generate all possible bigrams
            layout_chars = config['data']['layout']['chars']
            all_bigrams = []
            for char1 in layout_chars:
                for char2 in layout_chars:
                    all_bigrams.append(char1 + char2)

            # Calculate comfort scores for all bigrams
            results = []
            for bigram in all_bigrams:
                comfort_mean, comfort_std = model.get_bigram_comfort_scores(bigram)
                results.append({
                    'bigram': bigram,
                    'comfort_score': comfort_mean,
                    'uncertainty': comfort_std,
                    'first_char': bigram[0],
                    'second_char': bigram[1]
                })

            # Save results
            output_file = Path(config['data']['output_dir']) / 'bigram_comfort_scores.csv'
            pd.DataFrame(results).to_csv(output_file, index=False)
            logger.info(f"Saved comfort scores for {len(all_bigrams)} bigrams to {output_file}")

            # Generate summary statistics and visualizations
            df = pd.DataFrame(results)
            logger.info("\nComfort Score Summary:")
            logger.info(f"Mean comfort score: {df['comfort_score'].mean():.3f}")
            logger.info(f"Score range: {df['comfort_score'].min():.3f} to {df['comfort_score'].max():.3f}")
            logger.info(f"Mean uncertainty: {df['uncertainty'].mean():.3f}")

            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
