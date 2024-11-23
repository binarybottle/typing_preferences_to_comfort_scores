"""
Main script for keyboard layout analysis pipeline.

This script orchestrates the entire keyboard layout analysis process:
1. Data preprocessing and train/test splitting
2. Feature extraction and evaluation
3. Timing and frequency analysis
4. Feature space analysis
5. Bayesian model training and evaluation
"""

import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any
import pandas as pd

from data_processing import DataPreprocessor, generate_train_test_splits, manage_data_splits
from bigram_frequency_timing import (plot_frequency_timing_relationship, plot_timing_by_frequency_groups,
                                     save_timing_analysis)
from bigram_feature_definitions import (column_map, row_map, finger_map, engram_position_values,
                                        row_position_values, bigrams, bigram_frequencies_array)
from bigram_feature_extraction import (precompute_all_bigram_features, precompute_bigram_feature_differences)
from bigram_pair_feature_evaluation import evaluate_feature_sets
from bigram_pair_recommendations import (analyze_feature_space, save_feature_space_analysis_results)
from bayesian_modeling import (train_bayesian_glmm, calculate_bigram_comfort_scores, 
                               save_model_results, plot_model_diagnostics, evaluate_model_performance, 
                               validate_comfort_scores, plot_sensitivity_analysis)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    required_sections = [
        'data', 'splits', 'layout', 'features', 'feature_evaluation',
        'model', 'output', 'logging'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model settings
    if 'model' in config:
        model_config = config['model']
        if 'train' not in model_config:
            raise ValueError("Missing 'train' setting in model config")
        if model_config['train']:
            required_model_settings = ['inference_method', 'num_samples', 'chains']
            for setting in required_model_settings:
                if setting not in model_config:
                    raise ValueError(f"Missing required model setting: {setting}")

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config

def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Create all necessary output directories.
    
    Args:
        config: Configuration dictionary containing output paths
    """
    dirs_to_create = [
        # Base directories
        Path(config['output']['base_dir']),
        Path(config['output']['logs']),
        
        # Feature evaluation
        Path(config['output']['feature_evaluation']['dir']),
        
        # Feature space
        Path(config['output']['feature_space']['dir']),
        Path(config['output']['feature_space']['analysis']).parent,
        Path(config['output']['feature_space']['recommendations']).parent,
        Path(config['output']['feature_space']['pca']).parent,
        Path(config['output']['feature_space']['bigram_graph']).parent,
        Path(config['output']['feature_space']['underrepresented']).parent,
        
        # Frequency timing
        Path(config['output']['frequency_timing']['dir']),
        Path(config['output']['frequency_timing']['analysis']).parent,
        Path(config['output']['frequency_timing']['relationship']).parent,
        
        # Model outputs
        Path(config['output']['model']['dir']),
        Path(config['output']['model']['results']).parent,
        Path(config['output']['model']['diagnostics']).parent,
        Path(config['output']['model']['forest']).parent,
        Path(config['output']['model']['posterior']).parent,
        Path(config['output']['model']['participant_effects']).parent,
        Path(config['output']['model']['sensitivity']).parent,
        
        # Scores
        Path(config['output']['scores']).parent
    ]

    for directory in dirs_to_create:
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Keyboard Layout Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load and validate configuration
    config = load_config(args.config)
    
    # Create logs directory and setup logging
    Path(config['logging']['file']).parent.mkdir(parents=True, exist_ok=True)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting keyboard layout analysis pipeline")

    try:
        # Create all output directories
        create_output_directories(config)

        # Initialize preprocessor and load data
        logger.info("Loading and preprocessing data")
        preprocessor = DataPreprocessor(config['data']['input_file'])
        preprocessor.load_data()

        # Prepare features
        logger.info("Computing bigram features")
        all_bigrams, all_bigram_features, feature_names, samekey_bigrams, \
            samekey_bigram_features, samekey_feature_names = precompute_all_bigram_features(
                layout_chars=config['layout']['left_chars'],
                column_map=column_map, 
                row_map=row_map, 
                finger_map=finger_map,
                engram_position_values=engram_position_values,
                row_position_values=row_position_values,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                config=config
            )

        # Compute feature differences
        all_feature_differences = precompute_bigram_feature_differences(all_bigram_features)

        # Process data
        logger.info("Processing data")
        preprocessor.prepare_bigram_pairs()
        preprocessor.extract_target_vector()
        preprocessor.process_participants()
        preprocessor.create_feature_matrix(
            all_feature_differences=all_feature_differences,
            feature_names=feature_names,
            config=config
        )

        # Get processed data and create train/test split
        processed_data = preprocessor.get_processed_data()
        generate_train_test_splits(processed_data, config)
        train_data, test_data = manage_data_splits(processed_data, config)

        if config['feature_evaluation']['enabled']:            
            # Run feature evaluation
            feature_eval_results = evaluate_feature_sets(
                feature_matrix=test_data.feature_matrix,
                target_vector=test_data.target_vector,
                participants=test_data.participants,
                candidate_features=config['feature_evaluation']['combinations'],
                feature_names=feature_names,
                output_dir=Path(config['output']['feature_evaluation']['dir']),
                config=config,
                n_splits=config['feature_evaluation']['n_splits'],
                n_samples=config['feature_evaluation']['n_samples']
            )
            
            logger.info("Feature evaluation completed")

        # Analyze bigram frequency/timing relationship using all of the data
        if config['output']['frequency_timing']['enabled']:
            logger.info("Analyzing timing-frequency relationship")
            timing_results = plot_frequency_timing_relationship(
                bigram_data=processed_data,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                output_path=config['output']['frequency_timing']['relationship']
            )
            # Create timing by frequency groups analysis using all of the data
            group_comparison_results = plot_timing_by_frequency_groups(
                processed_data,
                bigrams,
                bigram_frequencies_array,
                output_base_path=config['output']['frequency_timing']['group_directory'],
                n_groups=config['output']['frequency_timing']['n_groups']
            )
            save_timing_analysis(timing_results, 
                               group_comparison_results,
                               config['output']['frequency_timing']['analysis'])

        # Evaluate feature space using all of the data and generate recommendations
        if config['output']['feature_space']['enabled']:
            logger.info("Analyzing feature space")
            feature_space_results = analyze_feature_space(
                feature_matrix=processed_data.feature_matrix,
                output_paths=config['output']['feature_space'],
                all_feature_differences=all_feature_differences,
                config=config,
                recommend_bigrams=config['output']['feature_space']['recommend_bigrams'],
                num_recommendations=config['output']['feature_space']['num_recommendations']
            )

            # Save analysis results
            save_feature_space_analysis_results(feature_space_results, 
                                              config['output']['feature_space']['analysis'])

            # Log recommendations and feature space metrics
            logger.info(f"Feature space analysis completed:")
            logger.info(f"  Hull area: {feature_space_results['feature_space_metrics']['hull_area']:.3f}")
            logger.info(f"  Point density: {feature_space_results['feature_space_metrics']['point_density']:.3f}")
            if 'recommendations' in feature_space_results:
                logger.info(f"  Generated {len(feature_space_results['recommendations'])} bigram pair recommendations")

        # Train model if requested
        if config['model']['train']:
            logger.info("Training Bayesian GLMM")
            
            # Train model
            trace, model, priors = train_bayesian_glmm(
                feature_matrix=train_data.feature_matrix,
                target_vector=train_data.target_vector,
                participants=train_data.participants,
                design_features=config['features']['groups']['design'],
                control_features=config['features']['groups']['control'],
                inference_method=config['model']['inference_method'],
                num_samples=config['model']['num_samples'],
                chains=config['model']['chains']
            )

            # Save model results
            model_prefix = f"{config['output']['model']['results']}_{config['model']['inference_method']}"
            save_model_results(trace, model, model_prefix)

            # Plot diagnostics
            if config['model'].get('visualization', {}).get('enabled', False):
                logger.info("Generating model diagnostics")
                plot_model_diagnostics(
                    trace=trace,
                    output_base_path=config['output']['model']['diagnostics'],
                    inference_method=config['model']['inference_method']
                )

                plot_sensitivity_analysis(
                    parameter_estimates=trace,
                    design_features=config['features']['groups']['design'],
                    control_features=config['features']['groups']['control'],
                    output_path=config['output']['model']['sensitivity'].format(
                        inference_method=config['model']['inference_method']
                    )
                )

            # Calculate comfort scores
            if config['model'].get('scoring', {}).get('enabled', False):
                logger.info("Calculating comfort scores")
                comfort_scores = calculate_bigram_comfort_scores(
                    trace,
                    train_data.feature_matrix,
                    features_for_design=config['features']['groups']['design'],
                    mirror_left_right_scores=config['model']['scoring'].get('mirror_left_right_scores', True)
                )

                # Validate scores against test data
                validate_comfort_scores(comfort_scores, test_data)

                # Save comfort scores
                output_path = Path(config['output']['scores'])
                pd.DataFrame.from_dict(comfort_scores, orient='index',
                                     columns=['comfort_score']).to_csv(output_path)

            # Evaluate model on test data
            if config['model'].get('evaluate_test', True):  # Make evaluation optional
                logger.info("Evaluating model on test data")
                test_score = evaluate_model_performance(
                    trace=trace,
                    feature_matrix=test_data.feature_matrix,
                    target_vector=test_data.target_vector,
                    participants=test_data.participants,
                    design_features=config['features']['groups']['design'],
                    control_features=config['features']['groups']['control']
                )
                logger.info(f"Test set performance metrics:")
                for metric, value in test_score.items():
                    logger.info(f"  {metric}: {value:.3f}")


        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
