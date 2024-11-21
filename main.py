"""
Main script for keyboard layout analysis pipeline.
"""

import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import DataPreprocessor, manage_data_splits
from bigram_feature_extraction import precompute_all_bigram_features, precompute_bigram_feature_differences
from bayesian_modeling import train_bayesian_glmm, calculate_all_bigram_comfort_scores, save_model_results
from analysis_visualization import (plot_timing_frequency_relationship, plot_timing_by_frequency_groups,
                                  save_timing_analysis, analyze_feature_space, 
                                  save_feature_space_analysis_results, plot_model_diagnostics)
from bigram_features import (column_map, row_map, finger_map, engram_position_values,
                           row_position_values, bigrams, bigram_frequencies_array)
from feature_evaluation import evaluate_feature_sets

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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Keyboard Layout Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Create logs directory first
    Path(config['logging']['file']).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting keyboard layout analysis pipeline")

    try:
        # Create necessary directories
        dirs_to_create = [
            config['output']['base_dir'],
            config['output']['logs'],
            config['output']['model']['dir'],
            config['output']['feature_space']['dir'],
            config['output']['timing_frequency']['dir'],
            Path(config['output']['comfort_scores']).parent
        ]

        # Create directories
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Initialize data preprocessor
        preprocessor = DataPreprocessor(config['data']['input_file'])
        preprocessor.load_data()

        # Prepare features
        logger.info("Computing bigram features")
        all_bigrams, all_bigram_features, feature_names, samekey_bigrams, \
            samekey_bigram_features, samekey_feature_names = precompute_all_bigram_features(
                config['layout']['left_chars'],
                column_map, row_map, finger_map,
                engram_position_values,
                row_position_values,
                bigrams,
                bigram_frequencies_array
            )

        # Compute feature differences
        all_feature_differences = precompute_bigram_feature_differences(all_bigram_features)

        # Process data before any analysis
        logger.info("Processing data")
        preprocessor.prepare_bigram_pairs()
        preprocessor.extract_target_vector()
        preprocessor.process_participants()
        preprocessor.extract_typing_times()
        preprocessor.create_feature_matrix(all_feature_differences, feature_names)

        # Get processed data
        processed_data = preprocessor.get_processed_data()

        # If feature evaluation is enabled, do it before any model training
        if config['feature_evaluation']['enabled']:
            logger.info("Starting feature evaluation")
            
            # Get consistent train/test split
            train_data, test_data = manage_data_splits(processed_data, config)
            
            # Create required directories
            Path(config['feature_evaluation']['analysis_dir']).mkdir(parents=True, exist_ok=True)
            Path(config['feature_evaluation']['plots_dir']).mkdir(parents=True, exist_ok=True)
            
            # Run feature evaluation
            feature_eval_results = evaluate_feature_sets(
                feature_matrix=train_data.feature_matrix,
                target_vector=train_data.target_vector,
                participants=train_data.participants,
                candidate_features=config['feature_evaluation']['candidate_features'],
                feature_names=feature_names,
                output_dir=Path(config['feature_evaluation']['dir']),
                n_splits=config['feature_evaluation']['n_splits'],
                n_samples=config['feature_evaluation']['n_samples']
            )
            
            logger.info("Feature evaluation completed")
            
            # Use test_data for final model evaluation
            evaluation_data = test_data
        else:
            # If not doing feature evaluation, use all data for model
            evaluation_data = processed_data

        # Continue with existing pipeline using evaluation_data instead of processed_data
        if config['output']['timing_frequency']['enabled']:
            logger.info("Analyzing timing-frequency relationship")
            timing_results = plot_timing_frequency_relationship(
                bigram_data=preprocessor.data,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                output_path=config['output']['timing_frequency']['relationship']
            )

            # Create timing by frequency groups analysis
            group_comparison_results = plot_timing_by_frequency_groups(
                preprocessor.data,
                bigrams,
                bigram_frequencies_array,
                n_groups=config['output']['timing_frequency']['n_groups'],
                output_base_path=config['output']['timing_frequency']['groups']
            )

            save_timing_analysis(timing_results, 
                               group_comparison_results,
                               config['output']['timing_frequency']['analysis'])

        # Evaluate feature space and generate recommendations
        if config['output']['feature_space']['enabled']:
            logger.info("Analyzing feature space")
            feature_space_results = analyze_feature_space(
                feature_matrix=processed_data.feature_matrix,
                output_paths=config['output']['feature_space'],
                all_feature_differences=all_feature_differences,
                check_multicollinearity=config['output']['feature_space']['check_multicollinearity'],
                recommend_bigrams=config['output']['feature_space']['recommend_bigrams'],
                num_recommendations=config['output']['feature_space']['num_recommendations']
            )

            # Save analysis results
            save_feature_space_analysis_results(feature_space_results, 
                                              config['output']['feature_space']['analysis'])

            if feature_space_results['multicollinearity']['high_correlations']:
                logger.warning("Found high correlations between features:")
                for corr in feature_space_results['multicollinearity']['high_correlations']:
                    logger.warning(f"{corr['Feature1']} - {corr['Feature2']}: {corr['Correlation']:.3f}")

        # Train model if requested
        if config['model']['train']:
            logger.info("Training Bayesian GLMM")

            trace, model, priors = train_bayesian_glmm(
                feature_matrix=processed_data.feature_matrix,
                target_vector=processed_data.target_vector,
                participants=processed_data.participants,
                design_features=config['features']['design'],
                control_features=config['features']['control'],
                inference_method=config['model']['inference_method'],
                num_samples=config['model']['num_samples'],
                chains=config['model']['chains']
            )

            # Save model results
            save_model_results(trace, model, 
                             f"{config['output']['model']['results']}_{config['model']['inference_method']}")

            # Plot model diagnostics if visualization is enabled
            if config['visualization']['enabled']:
                plot_model_diagnostics(
                    trace=trace,
                    output_base_path=config['output']['model']['diagnostics'],
                    inference_method=config['model']['inference_method']
                )

            # Calculate comfort scores if requested
            if config['scoring']['enabled']:
                logger.info("Calculating comfort scores")
                comfort_scores = calculate_all_bigram_comfort_scores(
                    trace,
                    all_bigram_features,
                    features_for_design=config['features']['design'],
                    mirror_scores=config['scoring']['mirror_scores']
                )

                # Save comfort scores
                output_path = Path(config['output']['comfort_scores'])
                pd.DataFrame.from_dict(comfort_scores, orient='index',
                                     columns=['comfort_score']).to_csv(output_path)
 
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()