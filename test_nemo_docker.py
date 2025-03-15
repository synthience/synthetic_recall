#!/usr/bin/env python
"""Test script to verify the NemoSTT Docker integration with proper message handling."""

import os
import asyncio
import logging
import time
import sys
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NemoSTT
from voice_core.stt.nemo_stt import NemoSTT

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("nemo_docker_test")

# Get Docker endpoint from environment or use default
NEMO_DOCKER_ENDPOINT = os.environ.get(
    "NEMO_DOCKER_ENDPOINT", 
    "ws://localhost:5002/ws/transcribe"
)
logger.info(f"Using NEMO Docker endpoint: {NEMO_DOCKER_ENDPOINT}")

# Callback functions for monitoring transcriptions
async def on_transcription(data: Dict[str, Any]):
    logger.info(f"NemoSTT transcription received")
    logger.info(f"Text: '{data.get('text', 'N/A')}'")
    logger.info(f"Confidence: {data.get('confidence', 'N/A')}")
    logger.info(f"Processing time: {data.get('processing_time', 'N/A')}s")
    logger.info(f"Source: {data.get('source', 'N/A')}")

async def on_error(data: Dict[str, Any]):
    logger.error(f"Error from NEMO STT: {data}")

async def on_semantic(data: Dict[str, Any]):
    logger.info(f"Semantic processing result: {data}")

# Create a mock audio if no file is available
def create_mock_audio(duration=5, sample_rate=16000):
    import numpy as np
    # Create a multi-tone waveform to simulate speech
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Combine several frequencies to make it more speech-like
    audio_np = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # Base frequency
        0.2 * np.sin(2 * np.pi * 880 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 1320 * t) +  # Second harmonic
        0.05 * np.sin(2 * np.pi * 220 * t)    # Sub-harmonic
    )
    # Add some noise to simulate real-world conditions
    audio_np += 0.05 * np.random.normal(0, 1, len(audio_np))
    # Normalize to [-1, 1]
    audio_np = np.clip(audio_np / 1.1, -1.0, 1.0)
    
    logger.info("Created mock audio with multiple tones to simulate speech")
    return audio_np, sample_rate

# Function to load a test WAV file
def load_test_audio(audio_path):
    try:
        import wave
        import numpy as np
        
        with wave.open(audio_path, 'rb') as wav_file:
            # Get basic info
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read frames
            audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit audio
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit audio
                audio_np = np.frombuffer(audio_data, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to float32 and normalize to [-1, 1]
            audio_np = audio_np.astype(np.float32) / np.iinfo(audio_np.dtype).max
            
            # If stereo, convert to mono by averaging channels
            if n_channels == 2:
                audio_np = np.mean(audio_np.reshape(-1, 2), axis=1)
            
            logger.info(f"Loaded audio file: {audio_path}")
            logger.info(f"Sample rate: {sample_rate}, Duration: {n_frames/sample_rate:.2f}s")
            return audio_np, sample_rate
            
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return None, None

# Main function to test the NemoSTT Docker integration
async def test_nemo_docker():
    logger.info("Starting NemoSTT Docker integration test")
    
    try:
        # Initialize NemoSTT with Docker endpoint
        nemo_config = {
            "docker_endpoint": NEMO_DOCKER_ENDPOINT,
            "enable_streaming": False,
            "sample_rate": 16000
        }
        
        nemo_stt = NemoSTT(nemo_config)
        
        # Register callbacks
        nemo_stt.register_callback("on_transcription", on_transcription)
        nemo_stt.register_callback("on_error", on_error)
        nemo_stt.register_callback("on_semantic", on_semantic)
        
        # Initialize NemoSTT
        logger.info("Initializing NemoSTT...")
        await nemo_stt.initialize()
        logger.info("NemoSTT initialized successfully")
        
        # Test connection to Docker endpoint
        logger.info("Testing direct connection to Docker endpoint...")
        connection_success, message = await nemo_stt.test_docker_connection()
        logger.info(f"Connection test result: {message}")
        
        if not connection_success:
            logger.error("Failed to connect to Docker endpoint, exiting test")
            return
        
        # Load or create test audio
        audio_np = None
        sample_rate = 16000
        
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            audio_np, sample_rate = load_test_audio(sys.argv[1])
        
        # If no file specified or loading failed, create mock audio
        if audio_np is None:
            audio_np, sample_rate = create_mock_audio()
        
        # Transcribe audio
        logger.info("Sending audio for transcription...")
        transcription_result = await nemo_stt.transcribe(audio_np)
        
        # Print transcription result
        logger.info(f"Transcription result: {transcription_result}")
        
        # Wait for any remaining callbacks
        logger.info("Waiting for callbacks to complete...")
        await asyncio.sleep(3)
        
        logger.info("Test completed, cleaning up...")
        
    except Exception as e:
        logger.error(f"Error in NemoSTT Docker test: {e}", exc_info=True)
    finally:
        # Clean up
        if 'nemo_stt' in locals() and hasattr(nemo_stt, 'docker_client') and nemo_stt.docker_client:
            logger.info("Disconnecting from Docker endpoint...")
            await nemo_stt.docker_client.disconnect()

# Entry point
if __name__ == "__main__":
    asyncio.run(test_nemo_docker())
