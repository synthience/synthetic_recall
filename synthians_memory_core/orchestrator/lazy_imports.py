# Lazy importer for NumPy and TensorFlow
# Based on the approach described in the memory about NumPy version incompatibility

import importlib
import logging
import sys
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global references to lazily loaded modules
_np = None
_tf = None

# The specific NumPy version that is compatible with FAISS and TensorFlow
COMPATIBLE_NUMPY_VERSION = "1.25.2"

def _fix_numpy_version():
    """Ensure NumPy is at the compatible version before any TensorFlow imports."""
    try:
        # First try to import NumPy to check its version
        import numpy as np
        current_version = np.__version__
        
        if current_version != COMPATIBLE_NUMPY_VERSION:
            logger.warning(f"Current NumPy version {current_version} is not compatible. Downgrading to {COMPATIBLE_NUMPY_VERSION}")
            
            try:
                # Uninstall current NumPy
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Install the compatible version
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"numpy=={COMPATIBLE_NUMPY_VERSION}"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Force reload numpy
                if 'numpy' in sys.modules:
                    del sys.modules['numpy']
                import numpy as np
                logger.info(f"NumPy downgraded and reloaded, version: {np.__version__}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error fixing NumPy version: {e}")
                return False
        else:
            logger.info(f"NumPy version {current_version} is already compatible")
            return True
    except ImportError:
        logger.warning("NumPy not found. Installing compatible version...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"numpy=={COMPATIBLE_NUMPY_VERSION}"],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing NumPy: {e}")
            return False

def get_numpy() -> Any:
    """Lazily import NumPy only when needed."""
    global _np
    if _np is None:
        logger.info("Lazily importing NumPy...")
        try:
            # Ensure we have the right version first
            _fix_numpy_version()
            _np = importlib.import_module("numpy")
            logger.info(f"NumPy imported successfully, version: {_np.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import NumPy: {e}")
            raise
    return _np

def get_tensorflow() -> Optional[Any]:
    """Lazily import TensorFlow only when needed."""
    global _tf
    if _tf is None:
        logger.info("Lazily importing TensorFlow...")
        try:
            # Ensure NumPy is at the right version before importing TensorFlow
            if not _fix_numpy_version():
                logger.error("Failed to fix NumPy version. TensorFlow import may fail.")
            
            # Import TensorFlow after NumPy version is fixed
            _tf = importlib.import_module("tensorflow")
            logger.info(f"TensorFlow imported successfully, version: {_tf.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow: {e}")
            _tf = None  # Ensure it's None on failure
    return _tf
