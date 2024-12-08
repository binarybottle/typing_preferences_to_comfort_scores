import logging
from pathlib import Path
from typing import Optional, List

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

def load_interactions(filepath: str) -> List[List[str]]:
    """Load feature interactions from file."""
    import yaml
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
        
    if not data or 'interactions' not in data:
        return []
        
    return data['interactions']

