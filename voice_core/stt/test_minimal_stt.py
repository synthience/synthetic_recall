#!/usr/bin/env python
"""
Test script to verify that the enhanced STT service can be initialized with minimal dependencies.
This script attempts to initialize the STT service with only the essential dependencies.
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import voice_core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from voice_core.stt.enhanced_stt_service import EnhancedSTTService, DIARIZATION_AVAILABLE
    from voice_core.state.voice_state_manager import VoiceStateManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def test_stt_initialization():
    """Test initializing the STT service with minimal dependencies."""
    logger.info("Testing STT service initialization with minimal dependencies")
    
    # Create a dummy state manager
    class DummyStateManager:
        def __init__(self):
            self.processing_lock = DummyLock()
            self.voice_state = None
        
        async def update_voice_state(self, *args, **kwargs):
            pass
            
    class DummyLock:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args):
            pass
    
    try:
        # Initialize the STT service
        stt_service = EnhancedSTTService(
            state_manager=DummyStateManager(),
            whisper_model="tiny.en",  # Use the smallest model for testing
            device="cpu"
        )
        
        # Check if optional components are available
        logger.info(f"Diarization available: {DIARIZATION_AVAILABLE}")
        logger.info(f"Silero VAD available: {getattr(stt_service, 'silero_vad_available', False)}")
        
        # Log initialization success
        logger.info("Successfully initialized STT service with minimal dependencies")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize STT service: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test STT service initialization with minimal dependencies")
    args = parser.parse_args()
    
    success = test_stt_initialization()
    sys.exit(0 if success else 1)
