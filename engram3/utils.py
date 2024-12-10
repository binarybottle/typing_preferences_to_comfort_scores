from pathlib import Path
from typing import Optional, List, Dict
import yaml
import logging

logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None) -> None:
    """Setup basic logging configuration.
    
    Args:
        log_file: Optional path to log file. If None, logs to console only.
    """
    # Basic config
    config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    # Add file handler if log_file specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        config['filename'] = str(log_file)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Apply configuration
    logging.basicConfig(**config)

def validate_config(config: Dict) -> None:
    """Validate required configuration settings exist."""
    required = [
        ('model', 'n_samples'),
        ('model', 'chains'),
        ('model', 'target_accept'),
        ('model', 'cross_validation', 'n_splits'),
        ('model', 'cross_validation', 'n_repetitions'),
        ('model', 'cross_validation', 'random_seed')
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
