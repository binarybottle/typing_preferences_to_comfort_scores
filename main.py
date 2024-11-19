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
                           plot_feature_space, plot_model_diagnostics, plot_bigram_graph)
from bigram_features import (column_map, row_map, finger_map, engram_position_values,
                             row_position_values, bigrams, bigram_frequencies_array)
from visualization import analyze_feature_space

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
                output_path=config['analysis']['timing_frequency']['plot_path']
            )
            
            # Create timing by frequency groups analysis
            group_comparison_results = plot_timing_by_frequency_groups(
                preprocessor.data,
                bigrams,
                bigram_frequencies_array,
                n_groups=config['analysis']['timing_frequency']['n_groups'],
                output_base_path=config['analysis']['timing_frequency']['groups_plot_path']
            )

        # Evaluate feature space and generate recommendations for new bigram pairs to test
        if config['analysis']['feature_space']['enabled']:
            logger.info("Analyzing feature space")
            feature_space_results = analyze_feature_space(
                feature_matrix=processed_data.feature_matrix,
                output_dir=config['analysis']['feature_space']['output_dir'],
                all_feature_differences=all_feature_differences,  # Pass this through
                check_multicollinearity=config['analysis']['feature_space']['check_multicollinearity'],
                recommend_bigrams=config['analysis']['feature_space']['recommend_bigrams'],
                num_recommendations=config['analysis']['feature_space']['num_recommendations']
            )
            
            if feature_space_results['multicollinearity']['high_correlations']:
                logger.warning("Found high correlations between features:")
                for corr in feature_space_results['multicollinearity']['high_correlations']:
                    logger.warning(f"{corr['Feature1']} - {corr['Feature2']}: {corr['Correlation']:.3f}")

        # Create feature space visualizations if requested
        if config['visualization']['enabled']:
            logger.info("Creating feature space visualizations")
            plot_feature_space(processed_data.feature_matrix,
                             config['visualization']['feature_space_plot'])
            plot_bigram_graph(processed_data.bigram_pairs,
                            config['visualization']['bigram_graph'])

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
            save_model_results(trace, model, config['output']['model_results'])

            # Plot model diagnostics
            if config['visualization']['enabled']:
                diagnostics_path = (config['visualization']['model_diagnostics_mcmc'] 
                                  if config['model']['inference_method'] == 'mcmc'
                                  else config['visualization']['model_diagnostics_vi'])
                # Create directory if it doesn't exist
                Path(diagnostics_path).parent.mkdir(parents=True, exist_ok=True)
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