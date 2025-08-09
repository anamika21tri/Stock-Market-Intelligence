"""
Logging configuration module for Market Intelligence System
"""
import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional

from .config import get_config


class LoggingManager:
    """Manages logging configuration and setup"""
    
    def __init__(self):
        self._initialized = False
        self.config = get_config().logging
        self.paths = get_config().paths
    
    def setup_logging(self, 
                     log_level: Optional[str] = None,
                     log_file: Optional[str] = None) -> None:
        """Setup logging configuration"""
        
        if self._initialized:
            return
        
        # Remove default logger
        logger.remove()
        
        # Get configuration
        level = log_level or self.config.level
        log_format = self.config.format
        
        # Console logging
        logger.add(
            sys.stdout,
            format=log_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File logging
        if log_file is None:
            log_file = os.path.join(self.paths.logs_dir, "market_intelligence.log")
        
        # Ensure log directory exists
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=self.config.file_rotation,
            retention=self.config.retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # Error file logging
        error_log_file = os.path.join(self.paths.logs_dir, "errors.log")
        logger.add(
            error_log_file,
            format=log_format,
            level="ERROR",
            rotation=self.config.file_rotation,
            retention=self.config.retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        self._initialized = True
        logger.info("Logging system initialized")
    
    def get_logger(self, name: str = None):
        """Get a logger instance"""
        if not self._initialized:
            self.setup_logging()
        
        if name:
            return logger.bind(name=name)
        return logger


# Global logging manager
logging_manager = LoggingManager()

# Convenience function
def get_logger(name: str = None):
    """Get a logger instance"""
    return logging_manager.get_logger(name)

# Setup logging immediately
logging_manager.setup_logging()

# Export logger for direct use
__all__ = ['get_logger', 'logging_manager', 'logger']
