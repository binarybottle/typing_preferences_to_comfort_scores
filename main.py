# main.py
"""
Command-line pipeline for estimating both bigram and single-key typing comfort scores through preference learning.

The input is bigram typing preference data (which bigram is easier to type?),
and the output includes:
1. Latent bigram typing comfort scores
2. Individual key comfort scores derived from the bigram model

The goal is to use these scores to optimize keyboard layouts.
Core features:

1. Data Management:
- Participant-aware train/test splitting
- Additional dataset handling
- Feature extraction setup
- Precomputed feature validation

2. Mode Implementation:
- analyze_features: Calculate feature importance metrics
- select_features: Iterative feature selection with CV
- recommend_bigram_pairs: Generate diverse pair recommendations
- train_model: Train on selected features with splits
- predict_bigram_scores: Generate bigram comfort predictions
- predict_key_scores: Generate individual key comfort predictions

3. Resource Handling:
- Memory monitoring
- Logging configuration
- Result persistence
- Error recovery

Usage: python main.py --config config.yaml --mode MODE
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

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
from bigram_typing_preferences_to_comfort_scores.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

# I renamed the repository/module, so need to symlink the module name in the pickle files
# Create an alias from old to new package name
import sys
import importlib
sys.modules['engram3'] = importlib.import_module('bigram_typing_preferences_to_comfort_scores')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_or_create_split(dataset: PreferenceDataset, config: Dict) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """
    Load/create splits for both original and additional data if it exists.
    """
    split_file = Path(config.data.splits['split_data_file'])
    
    try:
        # First handle original dataset
        if split_file.exists():
            logger.info("Loading existing split for original dataset...")
            split_data1 = np.load(split_file)
            train_indices1 = split_data1['train_indices']
            test_indices1 = split_data1['test_indices']
        else:
            logger.info("Creating new split for original dataset...")
            train_indices1, test_indices1 = create_participant_split(
                dataset, 
                test_ratio=config.data.splits['test_ratio'],
                random_seed=config.data.splits['random_seed']
            )
            # Save split
            split_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(split_file, train_indices=train_indices1, test_indices=test_indices1)

        # Handle additional data if it exists
        if hasattr(config.data, 'input_file2') and config.data.input_file2:
            file2 = Path(config.data.input_file2)
            split_file2 = Path(config.data.splits['split_data_file2'])
            
            logger.info("Processing additional dataset...")
            # Pass all necessary precomputed features from original dataset
            dataset2 = PreferenceDataset(
                file2, 
                feature_extractor=dataset.feature_extractor, 
                config=config,
                precomputed_features={
                    'all_bigrams': dataset.all_bigrams,
                    'all_bigram_features': dataset.all_bigram_features,
                    'feature_names': dataset.feature_names
                }
            )
            
            if split_file2.exists():
                logger.info("Loading existing split for additional dataset...")
                split_data2 = np.load(split_file2)
                train_indices2 = split_data2['train_indices']
                test_indices2 = split_data2['test_indices']
            else:
                logger.info("Creating new split for additional dataset...")
                train_indices2, test_indices2 = create_participant_split(
                    dataset2, 
                    test_ratio=config.data.splits['test_ratio'],
                    random_seed=config.data.splits['random_seed']
                )
                # Save split
                split_file2.parent.mkdir(parents=True, exist_ok=True)
                np.savez(split_file2, train_indices=train_indices2, test_indices=test_indices2)

            # Combine the splits
            train_data = dataset._create_subset_dataset(np.concatenate([train_indices1, train_indices2]))
            test_data = dataset._create_subset_dataset(np.concatenate([test_indices1, test_indices2]))
        else:
            # Just use original dataset split
            train_data = dataset._create_subset_dataset(train_indices1)
            test_data = dataset._create_subset_dataset(test_indices1)
        
        logger.info(f"Final splits:")
        logger.info(f"Training set: {len(train_data.preferences)} preferences, {len(train_data.participants)} participants")
        logger.info(f"Test set: {len(test_data.preferences)} preferences, {len(test_data.participants)} participants")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error creating/loading splits: {str(e)}")
        raise

def create_participant_split(dataset: PreferenceDataset, test_ratio: float, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a participant-aware train/test split."""
    # Set random seed
    np.random.seed(random_seed)
    
    # Get participant to indices mapping
    participant_to_indices = {}
    for i, pref in enumerate(dataset.preferences):
        if pref.participant_id not in participant_to_indices:
            participant_to_indices[pref.participant_id] = []
        participant_to_indices[pref.participant_id].append(i)
    
    # Split participants
    all_participants = list(participant_to_indices.keys())
    n_test = int(len(all_participants) * test_ratio)
    test_participants = set(np.random.choice(all_participants, n_test, replace=False))
    train_participants = set(all_participants) - test_participants
    
    # Get indices for each split
    train_indices = []
    test_indices = []
    for participant, indices in participant_to_indices.items():
        if participant in test_participants:
            test_indices.extend(indices)
        else:
            train_indices.extend(indices)
            
    return np.array(train_indices), np.array(test_indices)

def main():
    parser = argparse.ArgumentParser(description='Preference Learning Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['analyze_features', 'select_features', 
                                           'recommend_bigram_pairs', 'train_model',
                                           'predict_bigram_scores', 'predict_key_scores'],  # Added new mode
                        required=True,
                        help='Pipeline mode: feature selection, model training, 1-/2-gram comfort predictions')
    args = parser.parse_args()
    
    print("\n=== Program Start ===")
    print(f"Mode argument: {args.mode}")

    try:
        # Load configuration and convert to Pydantic model
        config_dict = load_config(args.config)
        print(f"\nLoaded config: {config_dict}")
        config = Config(**config_dict)
        print(f"\nConfig features: {config.features}")

        # Setup logging using LoggingManager
        LoggingManager(config).setup_logging()        
        
        # Set random seed for all operations
        np.random.seed(config.data.splits['random_seed'])
        
        # Initialize feature extraction
        logger.info("Initializing feature extraction...")
        # Debug to check the values are imported
        logger.debug(f"Loaded bigrams: {len(bigrams)} items")
        logger.debug(f"Loaded bigram frequencies: {bigram_frequencies_array.shape}")
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
        logger.info("Precomputing bigram features...")
        all_bigrams, all_bigram_features = feature_extractor.precompute_all_features(
            config.data.layout['chars']
        )
        # Debug:
        logger.debug("First bigram features:")
        first_bigram = next(iter(all_bigram_features))
        logger.debug(f"  bigram: {first_bigram}")
        logger.debug(f"  features: {all_bigram_features[first_bigram]}")

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
        # Analyze features
        #---------------------------------
        if args.mode == 'analyze_features':
            try:
                logger.info("\n=== ANALYZE FEATURES MODE ===")
                logger.info(f"Config base features: {config.features.base_features}")
                logger.info(f"Config interactions: {config.features.interactions}")
                logger.info(f"Config control features: {config.features.control_features}")

                # Load/create dataset split
                train_data, test_data = load_or_create_split(dataset, config)
                logger.info(f"Split dataset:\n  Train: {len(train_data.preferences)} preferences, {len(train_data.participants)} participants")

                # Initialize model
                model = PreferenceModel(config=config)

                # Get all features (base + interactions)
                base_features = config.features.base_features
                control_features = config.features.control_features
                interaction_features = config.features.get_all_interaction_names()
                all_features = base_features + interaction_features
                
                logger.info("\nFeatures prepared:")
                logger.info(f"  Base features: {base_features}")
                logger.info(f"  Interaction features: {interaction_features}")
                logger.info(f"  Control features: {control_features}")
                logger.info(f"  All features: {all_features}")

                # Check for existing results
                feature_metrics = []
                metrics_file = Path(config.feature_selection.metrics_file)
                completed_features = set()
                if metrics_file.exists():
                    try:
                        existing_df = pd.read_csv(metrics_file)
                        feature_metrics.extend(existing_df.to_dict('records'))
                        completed_features = set(existing_df['feature_name'])
                        logger.info(f"Loaded {len(completed_features)} existing feature metrics")
                    except Exception as e:
                        logger.error(f"Error loading existing metrics: {e}")

                # Initialize progress tracking
                start_time = datetime.now()
                total_features = len(all_features)
                features_done = len(completed_features)
                current_features = control_features.copy()  # Start with just control features

                for feature in all_features:
                    if feature in completed_features:
                        logger.info(f"Skipping already evaluated feature: {feature}")
                        continue

                    # Print progress stats
                    stats = model._get_progress_stats(features_done, total_features, start_time, feature)
                    model._print_progress(stats)

                    logger.info(f"\nEvaluating feature: {feature}")
                    try:
                        # Calculate importance metrics for this feature alone
                        metrics = model._calculate_feature_importance(
                            feature=feature,
                            dataset=train_data,
                            current_features=current_features
                        )
                        
                        # Add feature name and selected status
                        metrics['feature_name'] = feature
                        metrics['selected'] = 0  # Not selected yet
                        
                        feature_metrics.append(metrics)
                        features_done += 1  # Increment counter after successful evaluation
                        
                        logger.info(f"\nImportance metrics for '{feature}':")
                        logger.info(f"  Effect magnitude: {metrics['effect_magnitude']:.6f}")
                        logger.info(f"  Effect std dev: {metrics['effect_std']:.6f}")
                        logger.info(f"  Importance score: {metrics['selected_importance']:.6f}")
                        
                        # Save metrics after each feature
                        metrics_df = pd.DataFrame(feature_metrics)
                        metrics_df.to_csv(metrics_file, index=False)
                        logger.info(f"Saved metrics through feature '{feature}' to {metrics_file}")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating feature {feature}: {str(e)}")
                        continue

                # Basic statistics
                if feature_metrics:  # Only if we have any metrics
                    try:
                        logger.info("\nFeature evaluation summary:")
                        metrics_df = pd.DataFrame(feature_metrics)
                        logger.info(f"Total features evaluated: {len(metrics_df)}")
                        logger.info(f"Mean importance score: {metrics_df['selected_importance'].mean():.6f}")
                        logger.info(f"Max importance score: {metrics_df['selected_importance'].max():.6f}")
                    except Exception as e:
                        logger.error(f"Error generating summary statistics: {e}")
            
            except Exception as e:
                logger.error(f"Error in analyze_features mode: {e}")

        #---------------------------------
        # Select features
        #---------------------------------
        elif args.mode == 'select_features':
            logger.info("\n=== SELECT FEATURES MODE ===")
            logger.info(f"Config base features: {config.features.base_features}")
            logger.info(f"Config interactions: {config.features.interactions}")
            logger.info(f"Config control features: {config.features.control_features}")
            
            # Define all features first
            base_features = config.features.base_features
            interaction_features = config.features.get_all_interaction_names()
            control_features = config.features.control_features
            all_features = base_features + interaction_features + control_features
            
            # Split data for feature selection
            train_data, holdout_data = load_or_create_split(dataset, config)
            feature_select_train, feature_select_val = train_data.split_by_participants(test_fraction=0.2)

            # Preprocess feature selection training data
            logger.info("Preprocessing feature selection training data...")
            valid_prefs = []
            for pref in feature_select_train.preferences:
                valid = True
                for feature in all_features:
                    if (pref.features1.get(feature) is None or 
                        pref.features2.get(feature) is None):
                        valid = False
                        break
                if valid:
                    valid_prefs.append(pref)

            # Create processed dataset for feature selection
            processed_train = PreferenceDataset.__new__(PreferenceDataset)
            processed_train.preferences = valid_prefs
            processed_train.participants = {p.participant_id for p in valid_prefs}
            processed_train.file_path = feature_select_train.file_path
            processed_train.config = feature_select_train.config
            processed_train.control_features = feature_select_train.control_features
            processed_train.feature_extractor = feature_select_train.feature_extractor
            processed_train.feature_names = feature_select_train.feature_names
            processed_train.all_bigrams = feature_select_train.all_bigrams
            processed_train.all_bigram_features = feature_select_train.all_bigram_features  

            # Verify feature_extractor is set
            if processed_train.feature_extractor is None:
                raise ValueError("Feature extractor not set in processed dataset")

            logger.info(f"Preprocessed feature selection training data:")
            logger.info(f"  Original size: {len(feature_select_train.preferences)} preferences")
            logger.info(f"  After filtering: {len(processed_train.preferences)} preferences")
            logger.info(f"  Participants: {len(processed_train.participants)}")

            logger.info("Features prepared:")
            logger.info(f"  Base features: {base_features}")
            logger.info(f"  Interaction features: {interaction_features}")
            logger.info(f"  Control features: {control_features}")
            logger.info(f"  All features: {all_features}")

            model = PreferenceModel(config=config)
            
            try:
                # Select features using training subset
                logger.info("Calling model.select_features()...")
                selected_features = model.select_features(processed_train, all_features)
                logger.info(f"Feature selection completed. Selected features: {selected_features}")

                # Add check to ensure all base components of interactions are included
                selected_interactions = [f for f in selected_features if '_x_' in f]
                required_base_features = set()

                # Collect all base features required by interactions
                for interaction in selected_interactions:
                    components = interaction.split('_x_')
                    required_base_features.update(components)

                # Add any missing base features
                missing_base_features = []
                for base_feature in required_base_features:
                    if base_feature not in selected_features:
                        logger.warning(f"Adding base feature '{base_feature}' required by selected interactions")
                        missing_base_features.append(base_feature)

                # Update selected features list with missing base features
                if missing_base_features:
                    logger.info(f"Added {len(missing_base_features)} base features required by interactions: {missing_base_features}")
                    selected_features.extend(missing_base_features)

                # Train model on training set and validate on held-out validation set
                model.fit(feature_select_train, selected_features)
                val_metrics = model.evaluate(feature_select_val)
                logger.info(f"Validation metrics - Accuracy: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
                
                # Final model fit on full training data
                model.fit(train_data, selected_features, fit_purpose="Final model fit with selected features")
                model_save_path = Path(config.feature_selection.model_file)
                model.save(model_save_path)

                # Get feature weights once and log both weights and stored importance metrics
                feature_weights = model.get_feature_weights(include_control=True)
                for feature in selected_features:
                    weight, std = feature_weights.get(feature, (0.0, 0.0))
                    logger.info(f"{feature}:")
                    logger.info(f"  Weight: {weight:.3f} ± {std:.3f}")
                    
                    # Use stored importance metrics if available and feature is not a control feature
                    if feature not in control_features and feature in model.feature_importance_metrics:
                        importance = model.feature_importance_metrics[feature]
                        logger.info(f"  Importance metrics:")
                        logger.info(f"    Effect magnitude: {importance['effect_magnitude']:.3f}")
                        logger.info(f"    Effect consistency: {importance['selected_consistency']:.3f}")                
                        logger.info(f"    Importance: {importance['selected_importance']:.3f}")

            except Exception as e:
                logger.error(f"Error in feature selection: {str(e)}")
                raise
                                        
        #---------------------------------
        # Recommend bigram pairs
        #---------------------------------
        elif args.mode == 'recommend_bigram_pairs':
            logger.info("\n=== RECOMMEND BIGRAM PAIRS MODE ===")

            # Load feature selection trained model
            logger.info("Loading feature selection trained model...")
            selection_model_save_path = Path(config.feature_selection.model_file)
            feature_selection_model = PreferenceModel.load(selection_model_save_path)
            
            # Specify characters to exclude (example)
            excluded_chars = [] #['t', 'g', 'b']  # Add excluded characters here
            
            # Initialize recommender with excluded characters
            logger.info("Generating bigram pair recommendations...")
            recommender = BigramRecommender(
                dataset, 
                feature_selection_model, 
                config,
                excluded_chars=excluded_chars
            )
    
            recommender.visualize_feature_space()
            recommender.visualize_feature_distributions()

            logger.debug(f"Using features (including control): {feature_selection_model.get_feature_weights(include_control=True).keys()}")
            recommended_pairs = recommender.recommend_pairs()
            
            # Visualize recommendations
            logger.info("Visualizing recommendations...")
            recommender.visualize_feature_space_with_recommendations(recommended_pairs)
            
            # Save recommendations
            recommendations_file = Path(config.recommendations.recommendations_file)
            pd.DataFrame(recommended_pairs, columns=['bigram1', 'bigram2']).to_csv(
                         recommendations_file, index=False)
            logger.info(f"Saved recommendations to {recommendations_file}")
            
            # Print recommendations
            logger.info("\nRecommended bigram pairs:")
            for b1, b2 in recommended_pairs:
                logger.info(f"{b1} - {b2}")

        #---------------------------------
        # Train model
        #---------------------------------
        elif args.mode == 'train_model':
            logger.info("\n=== TRAIN MODEL MODE ===")

            # Load train/test split
            train_data, test_data = load_or_create_split(dataset, config)
            
            # Load selected features including control features
            feature_metrics_file = Path(config.feature_selection.metrics_file)
            if not feature_metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")      
            feature_metrics_df = pd.read_csv(feature_metrics_file)
            selected_features = (feature_metrics_df[feature_metrics_df['selected'] == 1]['feature_name'].tolist() + 
                                 list(config.features.control_features))
            if not selected_features:
                raise ValueError("No features were selected in feature selection phase")
            
            # Train prediction model
            logger.info(f"Training model on training data using {len(selected_features)} selected features...")
            model = PreferenceModel(config=config)

            try:
                model.fit(train_data, features=selected_features,
                          fit_purpose="Training model on training data")

                # Save the trained model
                model_save_path = Path(config.model.model_file)
                model.save(model_save_path)  # Save the model we just trained

                # Evaluate on test data (model.evaluate will use all features including control)
                logger.info("Evaluating model on test data...")
                test_metrics = model.evaluate(test_data)
                
                logger.info("\nTest data metrics:")
                for metric, value in test_metrics.items():
                    logger.info(f"{metric}: {value:.3f}")

            except Exception as e:
                logger.error(f"Error generating visualizations: {str(e)}")
                raise

        #---------------------------------
        # Predict bigram scores
        #---------------------------------
        elif args.mode == 'predict_bigram_scores':
            logger.info("\n=== PREDICT BIGRAM SCORES MODE ===")

            # Load selected features
            feature_metrics_file = Path(config.feature_selection.metrics_file)
            if not feature_metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")      
            feature_metrics_df = pd.read_csv(feature_metrics_file)
            # Change to handle control features
            selected_features = (feature_metrics_df[feature_metrics_df['selected'] == 1]['feature_name'].tolist() + 
                                 list(config.features.control_features))
            if not selected_features:
                raise ValueError("No features were selected in feature selection phase")

            # Load trained model
            logger.info("Loading trained model...")
            model_save_path = Path(config.model.model_file)
            trained_model = PreferenceModel.load(model_save_path)

            # Generate all possible bigrams
            layout_chars = config.data.layout['chars']
            all_bigrams = []
            for char1 in layout_chars:
                for char2 in layout_chars:
                    all_bigrams.append(char1 + char2)

            # Calculate comfort scores for all bigrams
            results = []
            for bigram in all_bigrams:
                comfort_mean, comfort_std = trained_model.get_bigram_comfort_scores(bigram)
                results.append({
                    'bigram': bigram,
                    'comfort_score': comfort_mean,
                    'uncertainty': comfort_std,
                    'first_char': bigram[0],
                    'second_char': bigram[1]
                })

            # Save results
            bigram_scores_file = Path(config.model.bigram_comfort_predictions_file)
            pd.DataFrame(results).to_csv(bigram_scores_file, index=False)
            logger.info(f"Saved comfort scores for {len(all_bigrams)} bigrams to {bigram_scores_file}")

            # Generate summary statistics and visualizations
            df = pd.DataFrame(results)
            logger.info("\nComfort Score Summary:")
            logger.info(f"Mean comfort score: {df['comfort_score'].mean():.3f}")
            logger.info(f"Score range: {df['comfort_score'].min():.3f} to {df['comfort_score'].max():.3f}")
            logger.info(f"Mean uncertainty: {df['uncertainty'].mean():.3f}")

        #---------------------------------
        # Predict single-key scores
        #---------------------------------
        elif args.mode == 'predict_key_scores':
            logger.info("\n=== PREDICT KEY SCORES MODE ===")
            # Load selected features
            feature_metrics_file = Path(config.feature_selection.metrics_file)
            if not feature_metrics_file.exists():
                raise FileNotFoundError("Feature metrics file not found. Run feature selection first.")      
            feature_metrics_df = pd.read_csv(feature_metrics_file)
            
            # Load trained model
            logger.info("Loading trained model...")
            model_save_path = Path(config.model.model_file)
            trained_model = PreferenceModel.load(model_save_path)
            
            # Calculate comfort scores for all keys
            logger.info("Calculating key comfort scores...")
            results = trained_model.predict_key_scores()
            
            # Save results
            key_scores_file = Path(config.model.key_comfort_predictions_file)
            results.to_csv(key_scores_file, index=False)
            logger.info(f"Saved comfort scores for {len(results)} keys to {key_scores_file}")
            
            # Generate summary statistics
            logger.info("\nComfort Score Summary:")
            logger.info(f"Mean comfort score: {results['comfort_score'].mean():.3f}")
            logger.info(f"Score range: {results['comfort_score'].min():.3f} to {results['comfort_score'].max():.3f}")
            logger.info(f"Mean uncertainty: {results['uncertainty'].mean():.3f}")

        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
