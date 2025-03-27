# synthians_memory_core/custom_logger.py

import logging
import os
import time
from typing import Dict, Any, Optional

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, log_level.upper(), None)
if not isinstance(numeric_level, int):
    numeric_level = logging.INFO

logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class Logger:
    """A simplified logger compatible with the original interface"""

    def __init__(self, name="SynthiansMemory"):
        self.logger = logging.getLogger(name)

    def info(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.info(log_msg)

    def warning(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.warning(log_msg)

    def error(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log error message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.error(log_msg, exc_info=True if isinstance(data, Exception) else False)

    def debug(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.debug(log_msg)

# Create a singleton logger instance
logger = Logger()