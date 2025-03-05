"""Logging configuration for voice agent"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with proper encoding and formatting"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level))
    
    # Create file handler
    log_file = log_dir / "voice_agent.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger
