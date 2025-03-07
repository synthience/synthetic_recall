"""
LUCID RECALL PROJECT
Logging Configuration

Standardized logging setup for consistent logging across all components.
"""

import logging
import sys
from pathlib import Path

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(name: str, level: str = "info", log_file: Path = None) -> logging.Logger:
    """
    Setup a logger with standardized formatting.

    Args:
        name (str): Name of the logger (usually the module name)
        level (str): Logging level as a string (debug, info, warning, error, critical)
        log_file (Path, optional): File path to write logs to.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))

    # Create log formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if a log file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Example: Initialize a logger for general use
logger = setup_logger("Lucidia", level="debug", log_file=Path("logs/lucidia.log"))
logger.info("Logging system initialized.")