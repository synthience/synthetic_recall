#!/usr/bin/env python

import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tf_installer")

# The specific NumPy version that is compatible with FAISS
COMPATIBLE_NUMPY_VERSION = "1.25.2"

def fix_numpy():
    """Ensure NumPy is properly downgraded to a compatible version before TensorFlow is imported."""
    try:
        import numpy as np
        current_version = np.__version__
        
        if current_version != COMPATIBLE_NUMPY_VERSION:
            logger.warning(f"Current NumPy version {current_version} may not be compatible. Downgrading to {COMPATIBLE_NUMPY_VERSION}")
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"numpy=={COMPATIBLE_NUMPY_VERSION}", "--force-reinstall"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"NumPy downgrade completed with output: {result.stdout}")
                
                # Force reload numpy
                if 'numpy' in sys.modules:
                    del sys.modules['numpy']
                import numpy as np
                logger.info(f"NumPy reloaded, version: {np.__version__}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error downgrading NumPy: {e.stderr}")
                return False
        else:
            logger.info(f"NumPy version {current_version} is already compatible")
            return True
    except ImportError:
        logger.warning("NumPy not found. Installing compatible version...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", f"numpy=={COMPATIBLE_NUMPY_VERSION}"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"NumPy installation completed with output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing NumPy: {e.stderr}")
            return False

def ensure_tensorflow_installed():
    """Ensures that TensorFlow is installed for the Titans variants."""
    # First, ensure NumPy is at the right version
    if not fix_numpy():
        logger.error("Failed to fix NumPy version. TensorFlow installation may fail.")
        return False
    
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow already installed, version: {tf.__version__}")
        return True
    except ImportError:
        logger.warning("TensorFlow not found. Attempting to install...")
        
        try:
            logger.info("Installing TensorFlow...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tensorflow"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"TensorFlow installation completed with output: {result.stdout}")
            
            # Verify installation was successful
            try:
                import tensorflow as tf
                logger.info(f"TensorFlow successfully installed, version: {tf.__version__}")
                return True
            except ImportError:
                logger.error("TensorFlow import still failing after installation!")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing TensorFlow: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing TensorFlow: {str(e)}")
            return False

if __name__ == "__main__":
    fix_numpy()
    ensure_tensorflow_installed()
