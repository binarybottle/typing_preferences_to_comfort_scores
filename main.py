"""
Main script for keyboard layout analysis pipeline.
"""

import logging
import argparse
from pathlib import Path
import yaml
from typing import Dict, Any
import pandas as pd

from data_processing import DataPreprocessor, manage_data_splits
from bigram_frequency_timing import (plot_frequency_timing_relationship, plot_timing_by_frequency_groups,
                                     save_timing_analysis)
from bigram_feature_definitions import (column_map, row_map, finger_map, engram_position_values,
                                        row_position_values, bigrams, bigram_frequencies_array)
from bigram_feature_extraction import (precompute_all_bigram_features, precompute_bigram_feature_differences,
                                       get_feature_combinations, get_feature_groups)
from bigram_pair_feature_evaluation import evaluate_feature_sets
from bigram_pair_recommendations import (analyze_feature_space, save_feature_space_analysis_results)
from bayesian_modeling import train_bayesian_glmm, calculate_all_bigram_comfort_scores, save_model_results, plot_model_diagnostics

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
        # Define necessary directories based on your structure
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
            
            # Scores
            Path(config['output']['scores']).parent
        ]

        # Create all the directories if they do not exist
        for directory in dirs_to_create:
            directory_path = Path(directory)  # Ensure it's a Path object
            if not directory_path.exists():
                logger.info(f"Creating directory: {directory_path}")
                directory_path.mkdir(parents=True, exist_ok=True)

        # Initialize data preprocessor
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

        # Get processed data and create train/test split if feature evaluation is enabled
        processed_data = preprocessor.get_processed_data()
        
        if config['feature_evaluation']['enabled']:
            train_data, test_data = manage_data_splits(
                processed_data, 
                config
            )
            
            # Get feature combinations and groups from config
            feature_combinations = get_feature_combinations(config)
            feature_groups = get_feature_groups(config)
            
            # Run feature evaluation
            feature_eval_results = evaluate_feature_sets(
                feature_matrix=train_data.feature_matrix,
                target_vector=train_data.target_vector,
                participants=train_data.participants,
                candidate_features=feature_combinations,
                feature_names=feature_names,
                output_dir=Path(config['output']['feature_evaluation']['dir']),  # Updated path
                config=config,
                n_splits=config['feature_evaluation']['n_splits'],
                n_samples=config['feature_evaluation']['n_samples']
            )
            
            logger.info("Feature evaluation completed")
            evaluation_data = test_data
        else:
            evaluation_data = processed_data

        # Continue with existing pipeline using evaluation_data instead of processed_data
        if config['output']['frequency_timing']['enabled']:
            logger.info("Analyzing timing-frequency relationship")
            timing_results = plot_frequency_timing_relationship(
                bigram_data=preprocessor.data,
                bigrams=bigrams,
                bigram_frequencies_array=bigram_frequencies_array,
                output_path=config['output']['frequency_timing']['relationship']
            )

            # Create timing by frequency groups analysis
            group_comparison_results = plot_timing_by_frequency_groups(
                preprocessor.data,
                bigrams,
                bigram_frequencies_array,
                n_groups=config['output']['frequency_timing']['n_groups'],
                output_base_path=config['output']['frequency_timing']['group_directory']  # Updated key
            )

            save_timing_analysis(timing_results, 
                               group_comparison_results,
                               config['output']['frequency_timing']['analysis'])

        # Evaluate feature space and generate recommendations
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

            trace, model, priors = train_bayesian_glmm(
                feature_matrix=processed_data.feature_matrix,
                target_vector=processed_data.target_vector,
                participants=processed_data.participants,
                design_features=config['features']['groups']['design'],
                control_features=config['features']['groups']['control'],
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
                    features_for_design = config['features']['groups']['design'],
                    mirror_scores=config['scoring']['mirror_scores']
                )

                # Save comfort scores
                output_path = Path(config['output']['scores'])
                pd.DataFrame.from_dict(comfort_scores, orient='index',
                                       columns=['comfort_score']).to_csv(output_path)
 
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()