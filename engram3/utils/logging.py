# engram3/utils/logging.py
"""
Logging configuration manager for keyboard layout optimization system.
Implements a configurable logging system with:
  - Rotating file logs with size limits (10MB, 5 backups)
  - Console output with customizable levels
  - Timestamp-based log files
  - UTF-8 encoding support

Core functionality:
  1. Log Configuration:
    - Separate console and file handlers
    - Configurable log levels for each handler
    - Automatic log directory creation
    - Timestamp-based file naming
    - Log rotation to manage disk space

  2. Error Handling:
    - Fallback to basic config on setup failure
    - Standard error logging with context
    - Debug traceback capture
    - UTF-8 encoding support

  3. Logger Management:
    - Centralized logger access
    - Configurable formatting
    - Root logger configuration
    - Handler cleanup

Configuration options via config:
    logging:
        console_level: Minimum level for console output
        file_level: Minimum level for file output
        format: Log message format string
    paths:
        logs_dir: Directory for log files

Usage:
    manager = LoggingManager(config)
    manager.setup_logging()
    logger = LoggingManager.getLogger(__name__)
    logger.info("Message")

Methods:
    setup_logging(): Initialize logging system
    getLogger(): Get logger instance
    handle_error(): Standard error logging
"""
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Optional
import traceback

from engram3.utils.config import Config

class LoggingManager:
    def __init__(self, config: Union[Dict, Config]):
        """Initialize LoggingManager with config."""
        self.config = config if isinstance(config, Config) else Config(**config)

    @classmethod
    def getLogger(cls, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)

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

    def setup_logging(self) -> None:
        """Initialize and configure logging system with rotation."""
        try:
            # Get logging config
            logging_config = getattr(self.config, 'logging', {})
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create log directory and file path
            log_dir = Path(self.config.paths.logs_dir)
            log_file = log_dir / f"debug_{timestamp}.log"
            
            # Create directory if it doesn't exist
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)  # Set to lowest level
            
            # Clear any existing handlers
            root_logger.handlers.clear()
            
            # File handler with rotation
            file_handler = self._create_file_handler(log_dir, timestamp)
            root_logger.addHandler(file_handler)
            
            # Console handler
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
            
            # Test logging
            logger = logging.getLogger(__name__)
            logger.info(f"Logging initialized - writing to {log_file}")
            logger.debug("Debug logging enabled")
            
        except Exception as e:
            # Fallback to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            logger.error(f"Error setting up logging: {str(e)}")
            logger.debug(traceback.format_exc())

    def _create_file_handler(self, log_dir: Path, timestamp: str) -> logging.Handler:
        """Create rotating file handler with detailed formatting."""
        log_file = log_dir / f"debug_{timestamp}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5,          # Keep 5 backup files
            encoding='utf-8'
        )
        file_level = getattr(self.config.logging, 'file_level', 'DEBUG')
        file_handler.setLevel(getattr(logging, file_level))
        
        log_format = getattr(self.config.logging, 'format', 
                            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(logging.Formatter(log_format))
        return file_handler

    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with standard formatting."""
        console_handler = logging.StreamHandler()
        console_level = getattr(self.config.logging, 'console_level', 'INFO')
        console_handler.setLevel(getattr(logging, console_level))
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        return console_handler

