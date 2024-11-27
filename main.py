"""
Main script for keyboard layout analysis pipeline.

This script orchestrates the entire keyboard layout analysis process:
1. Data preprocessing
2. Feature extraction
3. Full dataset analyses:
   - Timing and frequency analysis
   - Feature space analysis
4. Train/test splitting
5. Split dataset analyses:
   - Feature evaluation (using test_data)
   - Bayesian model training (using train_data)
   - Bayesian model testing (using test_data)
"""
import os
import argparse
from pathlib import Path
import yaml
from typing import Dict, List, Any
import pandas as pd
import logging

from data_processing import DataPreprocessor, generate_train_test_splits, manage_data_splits
from bigram_frequency_timing import (plot_frequency_timing_relationship, plot_timing_by_frequency_groups,
                                     save_timing_analysis)
from bigram_feature_definitions import (column_map, row_map, finger_map, engram_position_values,
                                        row_position_values, bigrams, bigram_frequencies_array)
from bigram_feature_extraction import (precompute_all_bigram_features, precompute_bigram_feature_differences)
from bigram_pair_feature_evaluation import evaluate_feature_sets, analyze_and_recommend
from bigram_pair_recommendations import (analyze_feature_space, save_feature_space_analysis_results)
from bayesian_modeling import (train_bayesian_glmm, calculate_bigram_comfort_scores, 
                               save_model_results, plot_model_diagnostics, evaluate_model_performance, 
                               validate_comfort_scores)

logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    required_sections = [
        'data',           # Data settings including file, splits, layout
        'paths',          # All file/directory paths
        'analysis',       # Analysis settings
        'model',          # Model settings
        'logging'         # Logging settings
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data section
    required_data_settings = ['file', 'splits', 'layout']
    for setting in required_data_settings:
        if setting not in config['data']:
            raise ValueError(f"Missing required data setting: {setting}")
            
    # Validate splits settings
    if not all(key in config['data']['splits'] for key in ['generate', 'train_ratio']):
        raise ValueError("Missing required splits settings")
    
    # Validate layout settings
    if 'left_chars' not in config['data']['layout']:
        raise ValueError("Missing left_chars in layout settings")
    
    # Validate features are under model section
    if 'features' not in config['model']:
        raise ValueError("Missing required 'features' section under 'model'")
        
    # Validate model settings if training enabled
    if config['model'].get('train', False):
        required_model_settings = ['inference_method', 'n_samples', 'chains']
        for setting in required_model_settings:
            if setting not in config['model']:
                raise ValueError(f"Missing required model setting: {setting}")

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration with proper stream and file handlers."""
    # Get log level from config
    log_level = getattr(logging, config['logging']['level'].upper())
    
    # Create formatters
    file_formatter = logging.Formatter(config['logging']['format'])
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(config['paths']['logs'])
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(stream_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
    logger.info(f"Log file: {config['paths']['logs']}")
    logger.info(f"Log level: {config['logging']['level']}")

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
        Path(config['paths']['base']),
        Path(config['paths']['logs']).parent,
        Path(config['paths']['feature_evaluation']),
        Path(config['paths']['feature_space']['dir']),
        Path(config['paths']['frequency_timing']['dir']),
        Path(config['paths']['frequency_timing']['group_directory']),
        Path(config['paths']['model']['dir']),
        Path(config['paths']['model']['results']).parent,
        Path(config['paths']['scores']).parent
    ]

    for directory in dirs_to_create:
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)

def main():
    # Initialize basic logging before config load
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Keyboard Layout Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = load_config(args.config)
        
        # Create logs directory and setup proper logging
        Path(config['paths']['logs']).parent.mkdir(parents=True, exist_ok=True)
        setup_logging(config)

        logger.info("Starting keyboard layout analysis pipeline")

        # Create all output directories
        create_output_directories(config)

        # Initialize preprocessor and load data
        logger.info("Loading and preprocessing data")
        preprocessor = DataPreprocessor(config['data']['file'])
        preprocessor.load_data()

        # Prepare features with conditional computation
        logger.info("Computing bigram features")
        should_compute_interactions = (
            config['model']['train'] and 
            config['model'].get('features', {}).get('interactions', {}).get('enabled', False)
        )
        
        if should_compute_interactions:
            logger.info("Computing full feature set including interactions")
        else:
            logger.info("Computing base features only - skipping interactions")
        
        all_bigrams, all_bigram_features, feature_names, samekey_bigrams, \
            samekey_bigram_features, samekey_feature_names = precompute_all_bigram_features(
                layout_chars=config['data']['layout']['left_chars'],
                column_map=column_map, 
                row_map=row_map, 
                finger_map=finger_map,
                engram_position_values=engram_position_values,
                row_position_values=row_position_values,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                config=config
            )

        # Compute feature differences conditionally
        if should_compute_interactions:
            all_feature_differences = precompute_bigram_feature_differences(all_bigram_features)
        else:
            # Filter out interaction features from the values (not the keys)
            filtered_features = {}
            for bigram, features in all_bigram_features.items():
                filtered_features[bigram] = {
                    k: v for k, v in features.items() 
                    if not k.startswith('interaction_')
                }
            all_feature_differences = precompute_bigram_feature_differences(filtered_features)

        # Process data
        logger.info("Processing data")
        preprocessor.prepare_bigram_pairs()
        preprocessor.extract_target_vector()
        preprocessor.process_participants()
        preprocessor.extract_typing_times()
        preprocessor.create_feature_matrix(
            all_feature_differences=all_feature_differences,
            feature_names=feature_names,
            config=config
        )

        # Get processed data BEFORE splitting
        processed_data = preprocessor.get_processed_data()

        # === FULL DATASET ANALYSES === #

        # Analyze bigram frequency/timing relationship if enabled
        if config['analysis']['frequency_timing']['enabled']:
            logger.info("Starting frequency/timing analysis")
            
            try:
                timing_results = plot_frequency_timing_relationship(
                    bigram_data=processed_data,
                    bigrams=bigrams,
                    bigram_frequencies_array=bigram_frequencies_array,
                    output_path=config['paths']['frequency_timing']['relationship'],
                    n_groups=config['analysis']['frequency_timing']['n_groups']
                )
                
                if 'error' in timing_results:
                    logger.error(f"Frequency-timing analysis failed: {timing_results['error']}")
                else:
                    # Create group visualizations
                    if 'group_analysis' in timing_results:
                        plot_timing_by_frequency_groups(
                            bigram_data=processed_data,
                            bigram_frequencies_array=bigram_frequencies_array,
                            group_results=timing_results['group_analysis'],
                            output_dir=config['paths']['frequency_timing']['group_directory']
                        )
                    
                    # Save analysis results
                    save_timing_analysis(
                        timing_results=timing_results,
                        group_comparison_results=timing_results.get('group_analysis'),
                        output_path=config['paths']['frequency_timing']['analysis']
                    )
                    
                    logger.info("Frequency/timing analysis completed")
                    logger.info(f"Results:")
                    logger.info(f"  Median correlation: {timing_results['correlation_median']:.3f} "
                                f"(p = {timing_results['correlation_median_p']:.3e})")
                    logger.info(f"  Mean correlation: {timing_results['correlation_mean']:.3f} "
                                f"(p = {timing_results['correlation_mean_p']:.3e})")
                    logger.info(f"  R-squared: {timing_results['r2']:.3f}")
                    logger.info(f"  ANOVA F-stat: {timing_results['anova_f_stat']:.3f} "
                                f"(p = {timing_results['anova_p_value']:.3e})")
                    logger.info(f"  Number of unique bigrams: {timing_results['n_unique_bigrams']}")
                    logger.info(f"  Total timing instances: {timing_results['total_occurrences']}")

            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
                raise

        # Evaluate feature space
        if config['analysis']['feature_space']['enabled']:
            logger.info("Analyzing feature space")
            try:
                base_dir = Path(config['paths']['feature_space']['dir'])
                output_paths = {
                    'pca': str(base_dir / 'pca.png'),
                    'underrepresented': str(base_dir / 'underrepresented.png'),
                    'bigram_graph': str(base_dir / 'bigram_graph.png'),
                    'recommendations': str(base_dir / 'recommended_bigram_pairs_scores.txt'),
                    'analysis': str(base_dir / 'analysis.txt')
                }
                
                feature_space_results = analyze_feature_space(
                    feature_matrix=processed_data.feature_matrix,
                    output_paths=output_paths,
                    all_feature_differences=all_feature_differences,
                    config=config,
                    recommend_bigrams=config['analysis']['feature_space']['recommend_bigrams'],
                    num_recommendations=config['analysis']['feature_space']['num_recommendations']
                )
                
                if feature_space_results is None:
                    logger.error("Feature space analysis returned no results")
                else:
                    save_feature_space_analysis_results(
                        feature_space_results, 
                        output_paths['analysis']
                    )

                    logger.info(f"Feature space analysis completed:")
                    if 'feature_space_metrics' in feature_space_results:
                        metrics = feature_space_results['feature_space_metrics']
                        logger.info(f"  Hull area: {metrics['hull_area']:.3f}")
                        logger.info(f"  Point density: {metrics['point_density']:.3f}")
                    if 'recommendations' in feature_space_results:
                        logger.info(f"  Generated {len(feature_space_results['recommendations'])} recommendations")
                        
            except Exception as e:
                logger.error(f"Error in feature space analysis: {str(e)}")
                if config['model']['train']:
                    logger.warning("Continuing with model training despite feature space analysis failure")

        # === SPLIT DATASET ANALYSES === #
        
        # Create train/test splits
        generate_train_test_splits(processed_data, config)
        train_data, test_data = manage_data_splits(processed_data, config)

        # Run feature evaluation if enabled
        if config['analysis']['feature_evaluation']['enabled']:            
            logger.info("Starting feature evaluation on test data")
            evaluation_results = evaluate_feature_sets(
                feature_matrix=test_data.feature_matrix,  # Make sure this contains all features
                target_vector=test_data.target_vector,
                participants=test_data.participants,
                feature_sets=config['analysis']['feature_evaluation']['combinations'],
                output_dir=Path(config['paths']['feature_evaluation']),
                n_splits=config['analysis']['feature_evaluation']['n_splits'],
                n_samples=config['analysis']['feature_evaluation']['n_samples'],
                chains=config['analysis']['feature_evaluation']['chains'],
                target_accept=config['analysis']['feature_evaluation']['target_accept']
            )
            logger.info("Feature evaluation completed")

            # Run analysis only if specifically enabled
            if config['analysis']['feature_evaluation']['analysis']['enabled']:
                logger.info("Running feature analysis and recommendations")
                analyze_and_recommend(
                    output_dir=Path(config['paths']['feature_evaluation']),
                    cv_metrics=evaluation_results['cv_metrics'],
                    feature_effects=evaluation_results['feature_effects'],
                    feature_sets=config['analysis']['feature_evaluation']['combinations'],
                    correlation_threshold=config['analysis']['feature_evaluation']['analysis']['thresholds']['correlation'],
                    importance_threshold=config['analysis']['feature_evaluation']['analysis']['thresholds']['importance'],
                    variability_threshold=config['analysis']['feature_evaluation']['analysis']['thresholds']['variability']
                )

        # Train model if requested
        if config['model']['train']:
            logger.info("Training Bayesian GLMM on training data")
            
            # Train model
            trace, model, priors = train_bayesian_glmm(
                feature_matrix=train_data.feature_matrix,
                target_vector=train_data.target_vector,
                participants=train_data.participants,
                design_features=config['model']['features']['groups']['design'],
                control_features=config['model']['features']['groups']['control'],
                inference_method=config['model']['inference_method'],
                n_samples=config['model']['n_samples'],
                chains=config['model']['chains'],
                target_accept=config['model']['target_accept']
            )

            # Save model results
            model_prefix = f"{config['paths']['model']['results']}_{config['model']['inference_method']}"
            save_model_results(trace, model, model_prefix)

            # Plot diagnostics
            if config['model'].get('visualization', {}).get('enabled', False):
                logger.info("Generating model diagnostics")
                plot_model_diagnostics(
                    trace=trace,
                    output_base_path=config['paths']['model']['diagnostics'],
                    inference_method=config['model']['inference_method']
                )

            # Calculate comfort scores using the trained model
            if config['model'].get('scoring', {}).get('enabled', False):
                logger.info("Calculating comfort scores")
                comfort_scores = calculate_bigram_comfort_scores(
                    trace,
                    train_data.feature_matrix,
                    features_for_design=config['model']['features']['groups']['design'],
                    mirror_left_right_scores=config['model']['scoring'].get('mirror_left_right_scores', True)
                )

                # Validate scores against test data
                validate_comfort_scores(comfort_scores, test_data)

                # Save comfort scores
                output_path = Path(config['paths']['scores'])
                pd.DataFrame.from_dict(comfort_scores, orient='index',
                                     columns=['comfort_score']).to_csv(output_path)

            # Evaluate model on test data
            if config['model'].get('evaluate_test', True):
                logger.info("Evaluating model on test data")
                test_score = evaluate_model_performance(
                    trace=trace,
                    feature_matrix=test_data.feature_matrix,
                    target_vector=test_data.target_vector,
                    participants=test_data.participants,
                    design_features=config['model']['features']['groups']['design'],
                    control_features=config['model']['features']['groups']['control']
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