# main.py
"""
Command-line pipeline for estimating both bigram and single-key typing comfort scores through preference learning.

The input is bigram typing preference data (which bigram is easier to type?),
and the output includes:
1. Latent bigram typing comfort scores
2. Individual key comfort scores derived from the bigram model
3. Direct key comfort scores derived from same-key bigram preferences

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
- predict_key_scores: Generate individual key comfort predictions from bigram data
- compute_key_scores: Generate key comfort scores from same-key bigram preferences

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

from typing_preferences_to_comfort_scores.utils.config import Config
from typing_preferences_to_comfort_scores.data import PreferenceDataset
from typing_preferences_to_comfort_scores.model import PreferenceModel
from typing_preferences_to_comfort_scores.recommendations import BigramRecommender
from typing_preferences_to_comfort_scores.features.feature_extraction import FeatureExtractor, FeatureConfig
from typing_preferences_to_comfort_scores.features.features import angles
from typing_preferences_to_comfort_scores.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array
from typing_preferences_to_comfort_scores.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

# I renamed the repository/module, so need to symlink the module name in the pickle files
# Create an alias from old to new package name
import sys
import importlib
sys.modules['engram3'] = importlib.import_module('typing_preferences_to_comfort_scores')


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_or_create_split(dataset: PreferenceDataset, config: Dict, no_split: bool = False) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """
    Load/create splits for both original and additional data if it exists.
    If no_split is True, returns the full dataset for both train and test.
    """
    # Handle the no-split case first
    if no_split:
        logger.info("No-split mode: Using full dataset for both training and testing")
        all_indices = np.arange(len(dataset.preferences))
        train_data = dataset._create_subset_dataset(all_indices)
        test_data = dataset._create_subset_dataset(all_indices)
        logger.info(f"Full dataset: {len(train_data.preferences)} preferences, {len(train_data.participants)} participants")
        return train_data, test_data
        
    # Original splitting code follows below
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

        # Use original dataset split
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
                                           'predict_bigram_scores', 'predict_key_scores',
                                           'compute_key_scores'],
                        required=True,
                        help='Pipeline mode: feature selection, model training, 1-/2-gram comfort predictions')
    parser.add_argument('--no-split', action='store_true', help='Use all data for training (no test split)')
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
                train_data, test_data = load_or_create_split(dataset, config, args.no_split)
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
            train_data, holdout_data = load_or_create_split(dataset, config, args.no_split)
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
            train_data, test_data = load_or_create_split(dataset, config, args.no_split)
            
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

        #-----------------------------------------
        # Compute key scores from same-key bigrams
        #-----------------------------------------
        elif args.mode == 'compute_key_scores':
            """
            - Identify and analyze all same-key bigram comparisons in your dataset
            - Compute both win ratio and Bradley-Terry model scores for keys with same-key data
            - Load your existing model-based scores for other keys
            - Create  a hybrid scoring approach that prioritizes direct comparison data where available
            - Produce  detailed rankings and statistics
            - Save  everything to a CSV file for further analysis or use in keyboard layout optimization
            """
            logger.info("\n=== COMPUTE KEY SCORES MODE ===")
            
            # Load the dataset
            logger.info("Loading dataset...")
            
            # Extract same-key bigram preferences
            logger.info("Extracting same-key bigram preferences...")
            same_key_prefs = []
            for pref in dataset.preferences:
                # Check if both bigrams are same-key bigrams
                if (len(pref.bigram1) >= 2 and pref.bigram1[0] == pref.bigram1[1] and 
                    len(pref.bigram2) >= 2 and pref.bigram2[0] == pref.bigram2[1]):
                    same_key_prefs.append(pref)
            
            logger.info(f"Found {len(same_key_prefs)} same-key bigram comparisons")
            
            # Organize preferences by key pairs
            key_pair_prefs = {}
            for pref in same_key_prefs:
                key1 = pref.bigram1[0]  # First char of first bigram
                key2 = pref.bigram2[0]  # First char of second bigram
                key_pair = tuple(sorted([key1, key2]))  # Sort for consistent lookups
                
                if key_pair not in key_pair_prefs:
                    key_pair_prefs[key_pair] = {'total': 0, f'{key1}_wins': 0, f'{key2}_wins': 0}
                
                key_pair_prefs[key_pair]['total'] += 1
                
                # Use the 'preferred' attribute to determine which bigram was chosen
                if pref.preferred:  # If bigram1 was preferred
                    key_pair_prefs[key_pair][f'{key1}_wins'] += 1
                else:  # If bigram2 was preferred
                    key_pair_prefs[key_pair][f'{key2}_wins'] += 1
            
            # Print detailed comparison data for each key pair
            logger.info("\nDetailed key pair comparisons:")
            for pair, stats in key_pair_prefs.items():
                key1, key2 = pair
                key1_wins = stats[f'{key1}_wins']
                key2_wins = stats[f'{key2}_wins']
                total = stats['total']
                
                logger.info(f"Key pair {key1}-{key2}: {key1} wins {key1_wins}/{total} ({key1_wins/total:.2f}), {key2} wins {key2_wins}/{total} ({key2_wins/total:.2f})")
            
            # Get set of keys that have same-key bigram data
            keys_with_data = set()
            for pair in key_pair_prefs:
                keys_with_data.update(pair)
            
            logger.info(f"\nKeys with same-key bigram comparison data: {len(keys_with_data)}")
            logger.info(f"Keys: {', '.join(sorted(keys_with_data))}")
            
            # Define Bradley-Terry function
            def compute_bradley_terry_scores(key_pair_prefs, all_keys):
                """
                Compute key scores using Bradley-Terry model.
                This is a statistical model for pairwise comparison data.
                """
                import numpy as np
                from scipy.optimize import minimize
                
                # Get keys with data (those that appear in any comparison)
                keys_with_data = set()
                for pair in key_pair_prefs:
                    keys_with_data.update(pair)
                
                # Only include keys with data in the model
                model_keys = list(keys_with_data)
                
                # Create wins matrix
                n = len(model_keys)
                key_to_idx = {k: i for i, k in enumerate(model_keys)}
                
                # Initialize wins matrix
                wins = np.zeros((n, n))
                
                # Fill with win counts
                for pair, stats in key_pair_prefs.items():
                    key1, key2 = pair
                    idx1, idx2 = key_to_idx[key1], key_to_idx[key2]
                    wins[idx1, idx2] = stats[f'{key1}_wins']
                    wins[idx2, idx1] = stats[f'{key2}_wins']
                
                # Define negative log-likelihood function
                def neg_log_likelihood(params):
                    strengths = np.exp(params)
                    nll = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if wins[i, j] + wins[j, i] > 0:  # If these keys were compared
                                w_ij = wins[i, j]
                                w_ji = wins[j, i]
                                p_ij = strengths[i] / (strengths[i] + strengths[j])
                                
                                # Add small epsilon to avoid log(0)
                                epsilon = 1e-10
                                nll -= w_ij * np.log(p_ij + epsilon) + w_ji * np.log(1 - p_ij + epsilon)
                    
                    return nll
                
                # Initial parameters (all keys equal strength)
                initial_params = np.zeros(n)
                
                # Minimize negative log-likelihood
                result = minimize(neg_log_likelihood, initial_params, method='BFGS')
                
                # Convert optimized parameters to scores
                strengths = np.exp(result.x)
                
                # Normalize to sum to 1
                strengths = strengths / np.sum(strengths)
                
                # Create scores dict with all keys
                scores = {}
                for k in all_keys:
                    if k in key_to_idx:
                        scores[k] = strengths[key_to_idx[k]]
                    else:
                        scores[k] = None
                
                return scores
            
            # Compute Bradley-Terry scores
            bt_scores = compute_bradley_terry_scores(key_pair_prefs, config.data.layout['chars'])

            # Define bootstrap function
            def bootstrap_bt_scores(key_pair_prefs, all_keys, n_bootstrap=1000):
                """
                Compute bootstrap confidence intervals for Bradley-Terry scores.

                This code adds bootstrap confidence intervals to your analysis by:
                - Resampling your pairwise comparison data with replacement
                - Recalculating Bradley-Terry scores for each bootstrap sample
                - Providing mean scores, standard deviations, and 95% confidence intervals for each key

                When looking at the results, if the confidence intervals of two keys overlap substantially, 
                you shouldn't consider them statistically significantly different. 
                The standard deviation values will also give you a sense of the uncertainty in each score.                
                """
                import numpy as np
                import random
                
                # Get all preferences as a list for resampling
                all_prefs = []
                for pair, stats in key_pair_prefs.items():
                    key1, key2 = pair
                    # Add wins for key1
                    all_prefs.extend([pair + (True,)] * stats[f'{key1}_wins'])
                    # Add wins for key2
                    all_prefs.extend([pair + (False,)] * stats[f'{key2}_wins'])
                
                # Store bootstrap results
                bootstrap_results = []
                
                for _ in range(n_bootstrap):
                    # Resample preferences with replacement
                    resampled_prefs = random.choices(all_prefs, k=len(all_prefs))
                    
                    # Reconstruct key_pair_prefs
                    resampled_key_pair_prefs = {}
                    for key1, key2, key1_won in resampled_prefs:
                        pair = (key1, key2)
                        if pair not in resampled_key_pair_prefs:
                            resampled_key_pair_prefs[pair] = {'total': 0, f'{key1}_wins': 0, f'{key2}_wins': 0}
                        
                        resampled_key_pair_prefs[pair]['total'] += 1
                        if key1_won:
                            resampled_key_pair_prefs[pair][f'{key1}_wins'] += 1
                        else:
                            resampled_key_pair_prefs[pair][f'{key2}_wins'] += 1
                    
                    # Compute BT scores for this bootstrap sample
                    bootstrap_bt_scores = compute_bradley_terry_scores(resampled_key_pair_prefs, all_keys)
                    bootstrap_results.append(bootstrap_bt_scores)
                
                # Calculate mean and confidence intervals
                means = {}
                lower_cis = {}
                upper_cis = {}
                std_devs = {}
                
                for key in all_keys:
                    if key in bootstrap_results[0]:
                        scores = [res[key] for res in bootstrap_results if res[key] is not None]
                        if scores:
                            means[key] = np.mean(scores)
                            std_devs[key] = np.std(scores)
                            lower_cis[key] = np.percentile(scores, 2.5)  # 95% CI lower bound
                            upper_cis[key] = np.percentile(scores, 97.5)  # 95% CI upper bound
                
                return {
                    'means': means,
                    'std_devs': std_devs,
                    'lower_cis': lower_cis,
                    'upper_cis': upper_cis
                }

            # Run bootstrap analysis
            logger.info("Running bootstrap analysis to calculate confidence intervals...")
            bootstrap_results = bootstrap_bt_scores(key_pair_prefs, config.data.layout['chars'])

            # Print rankings with confidence intervals
            logger.info("\nKey comfort rankings with 95% confidence intervals:")
            keys_with_bt_scores = [k for k in config.data.layout['chars'] if k in bootstrap_results['means']]
            sorted_keys = sorted(keys_with_bt_scores, key=lambda k: bootstrap_results['means'][k], reverse=True)

            for i, key in enumerate(sorted_keys, 1):
                mean = bootstrap_results['means'][key]
                lower = bootstrap_results['lower_cis'][key]
                upper = bootstrap_results['upper_cis'][key]
                std = bootstrap_results['std_devs'][key]
                logger.info(f"{i}. {key} - Score: {mean:.4f} (95% CI: {lower:.4f}-{upper:.4f}, SD: {std:.4f})")

            # Load existing model-based scores for keys without comparison data
            model_predictions_file = Path(config.model.key_comfort_predictions_file)
            model_scores = {}

            if model_predictions_file.exists():
                try:
                    model_df = pd.read_csv(model_predictions_file)
                    for _, row in model_df.iterrows():
                        # Check if key is in the layout before adding
                        if row['key'] in config.data.layout['chars']:
                            model_scores[row['key']] = row['comfort_score']
                    logger.info(f"Loaded model-based scores for {len(model_scores)} keys")
                except Exception as e:
                    logger.error(f"Error loading model scores: {e}")
            else:
                logger.warning(f"Model predictions file not found: {model_predictions_file}")

            # Compile results
            key_scores = []
            for key in config.data.layout['chars']:
                entry = {
                    'key': key,
                    'has_same_key_data': key in keys_with_data,
                }
                
                # Add Bradley-Terry score if available
                if key in keys_with_data:
                    entry['same_key_score'] = bt_scores[key]
                    
                    # Also store the rank for keys with same-key data
                    entry['same_key_rank'] = sum(1 for k in keys_with_data if bt_scores[k] > bt_scores[key]) + 1
                    
                    # Add bootstrap statistics if available
                    if key in bootstrap_results['means']:
                        entry['same_key_score_mean'] = bootstrap_results['means'][key]
                        entry['same_key_score_std'] = bootstrap_results['std_devs'][key]
                        entry['same_key_score_ci_low'] = bootstrap_results['lower_cis'][key]
                        entry['same_key_score_ci_high'] = bootstrap_results['upper_cis'][key]
                
                # Add model score if available
                if key in model_scores:
                    entry['model_score'] = model_scores[key]
                
                key_scores.append(entry)

            logger.info(f"Bootstrap results keys: {sorted(bootstrap_results['means'].keys())}")
            logger.info(f"Keys with data: {sorted(keys_with_data)}")

            # During key_scores creation
            if key in bootstrap_results['means']:
                logger.info(f"Adding bootstrap data for key {key}")
                entry['same_key_score_mean'] = bootstrap_results['means'][key]
                # ... rest of the code
            else:
                logger.info(f"Key {key} not found in bootstrap results")

            # Convert to DataFrame
            df = pd.DataFrame(key_scores)

            # Calculate model ranks for all keys with model scores
            if 'model_score' in df.columns:
                # This works by counting how many keys have a higher score and adding 1
                df['model_rank'] = df['model_score'].apply(
                    lambda x: sum(1 for s in df['model_score'].dropna() if s > x) + 1 if pd.notnull(x) else None
                )
                        
            # Save results
            key_scores_file = Path(config.model.key_comfort_predictions_file.replace('.csv', '_integrated.csv'))
            df.to_csv(key_scores_file, index=False)
            logger.info(f"Save key comfort scores to {key_scores_file}")
            
            # Print summary statistics and rankings
            logger.info("\nKey comfort rankings based on same-key bigram data (Bradley-Terry model):")
            bt_ranking_df = df[df['has_same_key_data']].sort_values('same_key_score', ascending=False)
            for i, (_, row) in enumerate(bt_ranking_df.iterrows(), 1):
                logger.info(f"{i}. {row['key']} - Score: {row['same_key_score']:.4f}")
            
            # Print model-based rankings for all keys
            if 'model_score' in df.columns:
                logger.info("\nModel-based rankings for all keys:")
                model_df = df[df['model_score'].notnull()].sort_values('model_score', ascending=False)
                for i, (_, row) in enumerate(model_df.iterrows(), 1):
                    has_data = "✓" if row['has_same_key_data'] else " "
                    logger.info(f"{i}. {row['key']} - Score: {row['model_score']:.4f} {has_data}")
            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
