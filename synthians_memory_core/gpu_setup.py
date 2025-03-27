#!/usr/bin/env python
# synthians_memory_core/gpu_setup.py

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPU-Setup")


def check_gpu_available():
    """Check if CUDA is available."""
    try:
        # Try to import torch and check CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"Found {device_count} CUDA device(s). Using: {device_name}")
            return True
        else:
            # Try nvidia-smi as a backup check
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logger.info("nvidia-smi detected GPU, but PyTorch CUDA not available.")
                # Still return True as FAISS might be able to use it
                return True
            else:
                logger.info("No CUDA devices detected through nvidia-smi")
                return False
    except (ImportError, FileNotFoundError):
        logger.warning("Could not check CUDA availability through PyTorch or nvidia-smi")
        return False


def install_faiss_gpu():
    """Install FAISS with GPU support."""
    try:
        # Try to import faiss-gpu first to see if it's already installed
        try:
            import faiss
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                logger.info(f"FAISS-GPU already installed. Available GPUs: {faiss.get_num_gpus()}")
                return True
            else:
                logger.info("FAISS is installed but no GPUs detected by FAISS")
        except ImportError:
            logger.info("FAISS not installed yet, proceeding with installation")
        
        # First uninstall faiss-cpu if it exists
        logger.info("Uninstalling faiss-cpu if present...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Install faiss-gpu
        logger.info("Installing faiss-gpu...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "faiss-gpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install faiss-gpu: {result.stderr.decode()}")
            return False
        
        # Verify installation
        try:
            import faiss
            logger.info(f"FAISS version: {faiss.__version__}")
            if hasattr(faiss, 'get_num_gpus'):
                gpu_count = faiss.get_num_gpus()
                logger.info(f"FAISS detected {gpu_count} GPUs")
                return gpu_count > 0
            else:
                logger.warning("FAISS installed but get_num_gpus not available")
                return False
        except ImportError:
            logger.error("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        logger.error(f"Error during FAISS-GPU installation: {str(e)}")
        return False


def install_faiss_cpu():
    """Install FAISS CPU version as fallback."""
    try:
        # Check if faiss is already installed
        try:
            import faiss
            logger.info(f"FAISS already installed (CPU version). Version: {faiss.__version__}")
            return True
        except ImportError:
            logger.info("FAISS not installed yet, proceeding with CPU installation")
        
        # Install faiss-cpu
        logger.info("Installing faiss-cpu...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "faiss-cpu>=1.7.4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install faiss-cpu: {result.stderr.decode()}")
            return False
        
        # Verify installation
        try:
            import faiss
            logger.info(f"FAISS CPU version: {faiss.__version__}")
            return True
        except ImportError:
            logger.error("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        logger.error(f"Error during FAISS-CPU installation: {str(e)}")
        return False


def setup_faiss():
    """Set up FAISS with GPU support if available, otherwise use CPU version."""
    logger.info("Checking for GPU availability...")
    if check_gpu_available():
        logger.info("GPU detected, installing FAISS with GPU support")
        if install_faiss_gpu():
            logger.info("Successfully installed FAISS with GPU support")
            return True
        else:
            logger.warning("Failed to install FAISS with GPU support, falling back to CPU version")
            return install_faiss_cpu()
    else:
        logger.info("No GPU detected, installing FAISS CPU version")
        return install_faiss_cpu()


if __name__ == "__main__":
    logger.info("=== FAISS GPU Setup Script ===")
    success = setup_faiss()
    if success:
        logger.info("FAISS setup completed successfully")
        sys.exit(0)
    else:
        logger.error("FAISS setup failed")
        sys.exit(1)
