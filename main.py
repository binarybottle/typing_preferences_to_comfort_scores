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

from engram3.utils.config import Config
from engram3.data import PreferenceDataset
from engram3.model import PreferenceModel
from engram3.recommendations import BigramRecommender
from engram3.utils.visualization import PlottingUtils
from engram3.features.feature_extraction import FeatureExtractor, FeatureConfig
from engram3.features.features import key_metrics
from engram3.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_or_create_split(dataset: PreferenceDataset, config: Dict) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """
    Load existing train/test split or create and save a new one.
    """
    split_file = Path(config.data.splits['split_data_file'])
    
    # If split file exists, delete it to force new split creation
    if split_file.exists():
        logger.info("Removing existing split file to create new split...")
        split_file.unlink()
    
    try:
        logger.info("Creating new train/test split...")
        
        # Get test size from config
        test_ratio = config.data.splits['test_ratio']
        
        # Set random seed for reproducibility
        np.random.seed(config.data.splits['random_seed'])
        
        # Get unique participant IDs and their corresponding preference indices
        participant_to_indices = {}
        for i, pref in enumerate(dataset.preferences):
            if pref.participant_id not in participant_to_indices:
                participant_to_indices[pref.participant_id] = []
            participant_to_indices[pref.participant_id].append(i)
        
        # Randomly select participants for test set
        all_participants = list(participant_to_indices.keys())
        n_test = int(len(all_participants) * test_ratio)
        test_participants = set(np.random.choice(all_participants, n_test, replace=False))
        train_participants = set(all_participants) - test_participants
        
        logger.info(f"Split participants: {len(train_participants)} train, {len(test_participants)} test")
        
        # Split indices based on participants
        train_indices = []
        test_indices = []
        for participant, indices in participant_to_indices.items():
            if participant in test_participants:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        logger.info(f"Split preferences: {len(train_indices)} train, {len(test_indices)} test")
        
        # Save split
        split_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_file, train_indices=train_indices, test_indices=test_indices)
        
        # Create datasets
        train_data = dataset._create_subset_dataset(train_indices)
        test_data = dataset._create_subset_dataset(test_indices)
        
        # Verify no overlap in participants
        train_participants_actual = set(p.participant_id for p in train_data.preferences)
        test_participants_actual = set(p.participant_id for p in test_data.preferences)
        
        if train_participants_actual & test_participants_actual:
            overlap = train_participants_actual & test_participants_actual
            logger.error(f"Train participants: {len(train_participants_actual)}")
            logger.error(f"Test participants: {len(test_participants_actual)}")
            logger.error(f"Overlap: {len(overlap)}")
            logger.error(f"Sample overlapping IDs: {list(overlap)[:5]}")
            raise ValueError("Train and test sets contain overlapping participants")
            
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in split creation: {str(e)}")
        logger.error(f"Dataset preferences: {len(dataset.preferences)}")
        logger.error(f"Total participants: {len(participant_to_indices)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Preference Learning Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['select_features', 'train_model', 'recommend_bigram_pairs'], 
                       required=True,
                       help='Pipeline mode: feature selection, model training, or bigram recommendations')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_dict = load_config(args.config)
        
        # Convert to Pydantic model
        config = Config(**config_dict)
        
        # Setup logging using LoggingManager
        LoggingManager(config).setup_logging()        
        
        # Add plotting utils
        plotting_utils = PlottingUtils(config.paths.plots_dir)
        
        # Set random seed for all operations
        np.random.seed(config.data.splits['random_seed'])
        
        # Initialize feature extraction
        logger.info("Initializing feature extraction...")
        feature_config = FeatureConfig(
            column_map=column_map,
            row_map=row_map,
            finger_map=finger_map,
            engram_position_values=engram_position_values,
            row_position_values=row_position_values,
            key_metrics=key_metrics
        )
        feature_extractor = FeatureExtractor(feature_config)
        
        # Precompute features for all possible bigrams
        logger.info("Precomputing bigram features...")
        all_bigrams, all_bigram_features = feature_extractor.precompute_all_features(
            config.data.layout['chars']
        )
        
        # Get feature names from first computed features
        feature_names = list(next(iter(all_bigram_features.values())).keys())
        
        # Load dataset with precomputed features
        logger.info("Loading dataset...")
        dataset = PreferenceDataset(
            Path(config.data.input_file),
            feature_extractor=feature_extractor,  # Make sure this is passed
            config=config,
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
            model.fit(train_data, selected_features)
            
            # Now get feature weights
            feature_weights = model.get_feature_weights()
            
            # Generate visualizations
            if model.feature_visualizer:
                fig = model.feature_visualizer.plot_feature_space(model, train_data, "Feature Space")
                fig.savefig(Path(config.paths.plots_dir) / 'final_feature_space.png')
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
            metrics_file = Path(config.feature_selection.metrics_file)
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
            metrics_file = Path(config.feature_selection.metrics_file)
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
            output_file = Path(config.recommendations.recommendations_file)
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
            metrics_file = config.feature_selection.metrics_file
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
            metrics_file = Path(config.feature_selection.metrics_file)
            if not metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")
                    
            df = pd.read_csv(metrics_file)
            selected_features = df[df['selected'] == 1]['feature_name'].tolist()

            # Ensure model is trained
            logger.info(f"Loading trained model with {len(selected_features)} features...")
            model = PreferenceModel(config=config)
            model.fit(dataset, features=selected_features)

            # Generate all possible bigrams
            layout_chars = config.data.layout['chars']
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
            output_file = Path(config.paths.metrics_dir) / 'bigram_comfort_scores.csv'
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
