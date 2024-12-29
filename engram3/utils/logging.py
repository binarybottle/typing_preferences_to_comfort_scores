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
from typing import Union, Dict, Optional
from pydantic import BaseModel
import traceback
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

from engram3.utils.config import Config

class LoggingManager:
    def __init__(self, config: Union[Dict, Config]):
        self.config = config

    def setup_logging(self) -> None:
        """Initialize and configure logging system."""
        try:
            # Get logging config
            if hasattr(self.config, 'logging'):
                logging_config = self.config.logging
            else:
                logging_config = self.config.get('logging', {})
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create log directory and file path
            log_dir = self.config.paths.logs_dir
            log_file = log_dir / f"debug_{timestamp}.log"
            
            # Create directory if it doesn't exist
            log_dir.mkdir(parents=True, exist_ok=True)

            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
            
            # Clear any existing handlers
            root_logger.handlers.clear()

            # File handler - captures everything with rotation
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB per file
                backupCount=5,          # Keep 5 backup files
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, logging_config.file_level))
            file_handler.setFormatter(logging.Formatter(logging_config.format))
            root_logger.addHandler(file_handler)

            # Console handler - only shows INFO and above by default
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, logging_config.console_level))
            console_handler.setFormatter(logging.Formatter(logging_config.format))
            root_logger.addHandler(console_handler)

            logger = logging.getLogger(__name__)
            logger.info(f"Logging initialized - writing to {log_file}")
            logger.debug("Debug logging enabled")

        except Exception as e:
            # Fallback to basic configuration if there's an error
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.error(f"Error setting up logging: {str(e)}")

    @classmethod
    def getLogger(cls, name: str) -> logging.Logger:
        return logging.getLogger(name)
                
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with standard formatting."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.logging.console_level)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        return console_handler
        
    def _create_file_handler(self, log_dir: Path, timestamp: str) -> logging.Handler:
        """Create file handler with detailed formatting."""
        log_file = log_dir / f"debug_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5,          # Keep 5 backup files
            encoding='utf-8'
        )
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

