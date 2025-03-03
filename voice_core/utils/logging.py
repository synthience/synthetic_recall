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
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / "voice_agent.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Log initialization
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_dir.absolute()}")
    
    return logger
