# engram3/utils/logging.py
"""
Logging configuration module for preference learning system.

Provides standardized logging setup with:
  - Console and file output handlers
  - Timestamp-based log files
  - Configurable log levels and formats
  - Centralized error handling
"""
from pathlib import Path
import logging
from typing import Union, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import traceback

class LoggingManager:
    """Centralized logging configuration and error handling."""

    _instance = None  # For singleton pattern
    
    def __init__(self, config: Union[Dict, BaseModel]):
        self.config = config

    @classmethod
    def getLogger(cls, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def setup_logging(self) -> None:
        """Initialize and configure logging system."""
        try:
            # Handle both dict and Pydantic config
            if hasattr(self.config, 'logging'):
                logging_config = self.config.logging
            else:
                logging_config = getattr(self.config, 'logging', {})
            # Create log directory
            log_dir = Path(logging_config.output_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=logging_config.file_level,
                format=logging_config.format,
                handlers=[
                    logging.FileHandler(logging_config.output_file),
                    logging.StreamHandler()  # Console output
                ]
            )
            
            # Set console level separately
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_config.console_level)
            
            # Get root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(min(
                logging.getLevelName(logging_config.file_level),
                logging.getLevelName(logging_config.console_level)
            ))
            
        except Exception as e:
            # Fallback to basic configuration if there's an error
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.error(f"Error setting up logging: {str(e)}")

    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with standard formatting."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.logging.console_level)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        return console_handler
        
    def _create_file_handler(self, log_dir: Path, timestamp: str) -> logging.Handler:
        """Create file handler with detailed formatting."""
        log_file = log_dir / f"debug_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config.logging.file_level)
        file_handler.setFormatter(
            logging.Formatter(self.config.logging.format)
        )
        return file_handler
        
    @staticmethod
    def handle_error(e: Exception, 
                     context: str = "",
                     logger: Optional[logging.Logger] = None) -> None:
        """Standardized error handling with logging."""
        if logger is None:
            logger = logging.getLogger(__name__)
            
        error_msg = f"{context}: {str(e)}" if context else str(e)
        logger.error(error_msg)
        logger.debug(traceback.format_exc())