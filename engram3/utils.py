from pathlib import Path
from typing import Optional, List, Dict
import yaml
import logging

logger = logging.getLogger(__name__)

def validate_config(config: Dict) -> None:
    """Validate required configuration settings exist."""
    required = [
        ('data', 'splits', 'random_seed'),
        ('data', 'splits', 'test_ratio'),
        ('data', 'splits', 'split_data_file'),
        ('data', 'input_file'),
        ('data', 'output_dir'),
        ('data', 'layout', 'chars'),
        ('feature_evaluation', 'thresholds', 'importance'),
        ('feature_evaluation', 'thresholds', 'stability'),
        ('feature_evaluation', 'importance_weights', 'model_effect'),
        ('feature_evaluation', 'importance_weights', 'correlation'),
        ('feature_evaluation', 'importance_weights', 'mutual_info'),
        ('feature_evaluation', 'metrics_file'),
        ('features', 'base_features'),
        ('features', 'interactions_file'),
        ('model', 'chains'),
        ('model', 'target_accept'),
        ('model', 'n_samples'),
        ('model', 'cross_validation', 'n_splits'),
        ('model', 'cross_validation', 'n_repetitions')
    ]
    
    for path in required:
        value = config
        for key in path:
            if key not in value:
                raise ValueError(f"Missing required config: {' -> '.join(path)}")
            value = value[key]

def load_interactions(filepath: str) -> List[List[str]]:
    """
    Load feature interactions from file.
    
    Args:
        filepath: Path to YAML file containing interaction definitions
        
    Returns:
        List of lists, where each inner list contains feature names to interact
        
    Example:
        interactions:
        - ['same_finger', 'sum_finger_values']
        - ['same_finger', 'rows_apart']
        - ['sum_finger_values', 'adj_finger_diff_row']
    """
    logger.debug(f"Loading interactions from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        if not data or 'interactions' not in data:
            logger.warning(f"No interactions found in {filepath}")
            return []
            
        interactions = data['interactions']
        
        if not isinstance(interactions, list):
            logger.error("Interactions must be a list")
            return []
            
        # Validate each interaction
        valid_interactions = []
        for interaction in interactions:
            if isinstance(interaction, list) and all(isinstance(f, str) for f in interaction):
                valid_interactions.append(interaction)
            else:
                logger.warning(f"Skipping invalid interaction format: {interaction}")
                
        logger.info(f"Loaded {len(valid_interactions)} valid interactions")
        return valid_interactions
        
    except Exception as e:
        logger.error(f"Error loading interactions file: {str(e)}")
        return []
