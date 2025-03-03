#!/usr/bin/env python
"""
Script to install minimal dependencies for the enhanced STT service.
This script installs only the essential packages needed for basic STT functionality.
"""

import os
import sys
import subprocess
import argparse
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define minimal dependencies
MINIMAL_DEPS = [
    "numpy>=1.20.0",
    "torch>=1.13.0",
    "torchaudio>=0.13.0",
    "faster-whisper>=0.6.0",
    "webrtcvad>=2.0.10",
    "soundfile>=0.12.1",
    "df-nightly",  # DeepFilterNet for noise reduction
    "tokenizers==0.21.0",  # Fixed version for compatibility
]

# Optional dependencies
OPTIONAL_DEPS = {
    "diarization": ["pyannote.audio>=2.1.1"],
    "vad": ["silero-vad"],
}

def install_packages(packages: List[str], upgrade: bool = False) -> bool:
    """
    Install packages using pip.
    
    Args:
        packages: List of packages to install
        upgrade: Whether to upgrade existing packages
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install minimal dependencies for STT service")
    parser.add_argument("--all", action="store_true", help="Install all dependencies including optional ones")
    parser.add_argument("--diarization", action="store_true", help="Install speaker diarization dependencies")
    parser.add_argument("--vad", action="store_true", help="Install neural VAD dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    args = parser.parse_args()
    
    # Install minimal dependencies
    logger.info("Installing minimal dependencies...")
    success = install_packages(MINIMAL_DEPS, args.upgrade)
    if not success:
        logger.error("Failed to install minimal dependencies")
        sys.exit(1)
    
    # Install optional dependencies
    if args.all or args.diarization:
        logger.info("Installing speaker diarization dependencies...")
        install_packages(OPTIONAL_DEPS["diarization"], args.upgrade)
    
    if args.all or args.vad:
        logger.info("Installing neural VAD dependencies...")
        install_packages(OPTIONAL_DEPS["vad"], args.upgrade)
    
    logger.info("Installation completed successfully")

if __name__ == "__main__":
    main()
