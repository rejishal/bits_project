# src/utils/logger.py
"""Logging configuration for the Banking ML Pipeline"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class Logger:
    """Custom logger class for the project"""
    
    def __init__(self, name: str, log_dir: str = 'logs', level: str = 'INFO'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.level = getattr(logging, level.upper())
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            self.log_dir / f'{self.name}_{timestamp}.log'
        )
        file_handler.setLevel(self.level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self) -> logging.Logger:
        """Get the logger instance"""
        return self.logger

# Create loggers for different modules
def get_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """Get a logger instance for a module"""
    logger = Logger(name, level=level)
    return logger.get_logger()

# Pre-configured loggers
data_logger = get_logger('data_processing')
model_logger = get_logger('model_training')
api_logger = get_logger('api_server')
monitor_logger = get_logger('monitoring')