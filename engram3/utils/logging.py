# engram3/utils/logging.py
"""
Logging configuration module for preference learning system.

Provides standardized logging setup with:
  - Console and file output handlers
  - Timestamp-based log files
  - Configurable log levels and formats
  - Centralized error handling
"""
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import traceback

class LoggingManager:
    """Centralized logging configuration and error handling."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @classmethod
    def getLogger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the given name.
        
        Args:
            name: Name for the logger (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)

    def setup_logging(self) -> None:
        """Configure logging with consistent formatting and handlers."""
        # Create timestamp string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup log directory
        log_dir = Path(self.config['logging']['output_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = LoggingManager.getLogger(__name__)

        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Console handler
        console_handler = self._create_console_handler()
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = self._create_file_handler(log_dir, timestamp)
        root_logger.addHandler(file_handler)
        
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with standard formatting."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config['logging']['console_level'])
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        return console_handler
        
    def _create_file_handler(self, log_dir: Path, timestamp: str) -> logging.Handler:
        """Create file handler with detailed formatting."""
        log_file = log_dir / f"debug_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config['logging']['file_level'])
        file_handler.setFormatter(
            logging.Formatter(self.config['logging']['format'])
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