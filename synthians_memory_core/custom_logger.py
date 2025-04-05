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
    """
    A simplified logger compatible with both the original interface
    (context, message, data) and standard logging calls (message, *args, **kwargs).
    """

    def __init__(self, name="SynthiansMemory"):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, context_or_msg: str, msg: Optional[str] = None, data: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal log handler."""
        exc_info = kwargs.pop('exc_info', None) # Extract standard exc_info kwarg

        # Determine how the method was called
        if msg is not None:
            # Called likely with (context, message, data)
            log_message = f"[{context_or_msg}] {msg}"
            if data:
                 log_message += f" | Data: {data}"
        else:
            # Called likely with standard (message, *args) or (message, data={})
            # Treat the first argument as the main message
            log_message = context_or_msg
            if data: # If data was passed as the third positional arg (legacy)
                 log_message += f" | Data: {data}"
            elif kwargs: # Or if data was passed as kwargs (more standard)
                 # Filter out standard logging kwargs if any snuck in
                 log_kwargs = {k: v for k, v in kwargs.items() if k not in ['level', 'name', 'pathname', 'lineno', 'funcName', 'exc_text', 'stack_info']}
                 if log_kwargs:
                      log_message += f" | Data: {log_kwargs}"

        self.logger.log(level, log_message, exc_info=exc_info)

    def debug(self, context_or_msg, msg=None, data=None, **kwargs):
        self._log(logging.DEBUG, context_or_msg, msg, data, **kwargs)

    def info(self, context_or_msg, msg=None, data=None, **kwargs):
        self._log(logging.INFO, context_or_msg, msg, data, **kwargs)

    def warning(self, context_or_msg, msg=None, data=None, **kwargs):
        self._log(logging.WARNING, context_or_msg, msg, data, **kwargs)

    def error(self, context_or_msg, msg=None, data=None, **kwargs):
        self._log(logging.ERROR, context_or_msg, msg, data, **kwargs)

# Create a singleton logger instance
logger = Logger()

def get_logger(name="SynthiansMemory"):
    """
    Factory function to create a logger instance with the given name.
    This function is used by the explainability and metrics modules.
    
    Args:
        name: Name for the logger instance
        
    Returns:
        Logger instance with the specified name
    """
    return Logger(name)