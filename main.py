"""
Main script for keyboard layout analysis pipeline.
"""

import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any
import pandas as pd

from data_preprocessing import DataPreprocessor
from feature_extraction import precompute_all_bigram_features, precompute_bigram_feature_differences
from bayesian_modeling import train_bayesian_glmm, calculate_all_bigram_comfort_scores, save_model_results
from visualization import (plot_timing_frequency_relationship, plot_timing_by_frequency_groups,
                         analyze_feature_space, plot_model_diagnostics)
from bigram_features import (column_map, row_map, finger_map, engram_position_values,
                           row_position_values, bigrams, bigram_frequencies_array)

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
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting keyboard layout analysis pipeline")

    try:
        # Create all necessary directories
        dirs_to_create = [
            config['output']['model']['dir'],
            config['output']['model']['visualizations']['mcmc'],
            config['output']['model']['visualizations']['vi'],
            config['output']['feature_space']['dir'],
            config['output']['feature_space']['visualizations'],
            config['output']['feature_space']['analysis'],
            config['output']['timing_frequency']['dir'],
            config['output']['timing_frequency']['visualizations'],
            config['output']['timing_frequency']['analysis'],
            Path(config['output']['comfort_scores']).parent,
            Path(config['output']['logs']).parent
        ]
        
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

        # Process data
        logger.info("Processing data")
        preprocessor.prepare_bigram_pairs()
        preprocessor.extract_target_vector()
        preprocessor.process_participants()
        preprocessor.extract_typing_times()
        preprocessor.create_feature_matrix(all_feature_differences, feature_names)

        if config['features']['add_interactions']:
            preprocessor.add_feature_interactions(config['features']['interaction_pairs'])

        # Get processed data
        processed_data = preprocessor.get_processed_data()

        # Run timing-frequency analysis if enabled
        if config['analysis']['timing_frequency']['enabled']:
            logger.info("Analyzing timing-frequency relationship")
            timing_results = plot_timing_frequency_relationship(
                bigram_data=preprocessor.data,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                output_path=config['output']['timing_frequency']['relationship_plot']
            )
            
            # Create timing by frequency groups analysis
            group_comparison_results = plot_timing_by_frequency_groups(
                preprocessor.data,
                bigrams,
                bigram_frequencies_array,
                n_groups=config['analysis']['timing_frequency']['n_groups'],
                output_base_path=config['output']['timing_frequency']['groups_plot']
            )

        # Evaluate feature space and generate recommendations
        if config['analysis']['feature_space']['enabled']:
            logger.info("Analyzing feature space")
            feature_space_results = analyze_feature_space(
                feature_matrix=processed_data.feature_matrix,
                output_paths=config['output']['feature_space'],
                all_feature_differences=all_feature_differences,
                check_multicollinearity=config['analysis']['feature_space']['check_multicollinearity'],
                recommend_bigrams=config['analysis']['feature_space']['recommend_bigrams'],
                num_recommendations=config['analysis']['feature_space']['num_recommendations']
            )

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
            save_model_results(trace, model, config['output']['model']['results'])

            # Plot model diagnostics if visualization is enabled
            if config['visualization']['enabled']:
                diagnostics_path = (config['output']['model']['visualizations']['mcmc']
                                  if config['model']['inference_method'] == 'mcmc'
                                  else config['output']['model']['visualizations']['vi'])
                plot_model_diagnostics(trace, 
                                     diagnostics_path,
                                     inference_method=config['model']['inference_method'])

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
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame.from_dict(comfort_scores, orient='index',
                                     columns=['comfort_score']).to_csv(output_path)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
