#!/usr/bin/env python
"""Test script to verify connection to NEMO STT Docker endpoint."""

import os
import asyncio
import logging
import time
import numpy as np
import wave
import sys

# Import the NemoSTT class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voice_core.stt.nemo_stt import NemoSTT

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("nemo_test")

# Get Docker endpoint from environment or use default
NEMO_DOCKER_ENDPOINT = os.environ.get(
    "NEMO_DOCKER_ENDPOINT", 
    "ws://localhost:5002/ws/transcribe"
)
logger.info(f"Using NEMO Docker endpoint: {NEMO_DOCKER_ENDPOINT}")

# Function to load a test WAV file
def load_test_audio(audio_path):
    try:
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

# Create a mock audio if no file is available
def create_mock_audio():
    # Create a 5-second sine wave at 440 Hz
    sample_rate = 16000
    duration = 5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_np = 0.5 * np.sin(2 * np.pi * 440 * t)
    logger.info("Created mock audio: 5-second 440 Hz tone")
    return audio_np, sample_rate

# Callback functions for NemoSTT
async def on_transcription(data):
    logger.info(f"Received transcription: {data}")
    if "text" in data:
        logger.info(f"Transcribed text: '{data['text']}'")
        logger.info(f"Confidence: {data.get('confidence', 'N/A')}")
    else:
        logger.warning("No text in transcription response")

async def on_error(data):
    logger.error(f"Error from NEMO STT: {data}")

async def on_semantic(data):
    logger.info(f"Semantic processing result: {data}")

# Main function to test connection
async def test_nemo_docker_connection():
    logger.info("Starting NEMO Docker connection test")
    
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
    
    try:
        # Initialize NemoSTT (which will try to connect to Docker endpoint)
        logger.info("Initializing NemoSTT...")
        await nemo_stt.initialize()
        logger.info("NemoSTT initialized successfully")
        
        # First test the direct connection to Docker endpoint
        logger.info("Testing direct connection to Docker endpoint...")
        connection_success, message = await nemo_stt.test_docker_connection()
        logger.info(f"Connection test result: {message}")
        
        if not connection_success:
            logger.error("Failed to connect to Docker endpoint, exiting test")
            return
        
        # First, try to load a test file if specified as argument
        audio_np = None
        sample_rate = 16000
        
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            audio_np, sample_rate = load_test_audio(sys.argv[1])
        
        # If no file specified or loading failed, create mock audio
        if audio_np is None:
            audio_np, sample_rate = create_mock_audio()
        
        # Process the audio through NemoSTT
        logger.info("Sending audio to NEMO Docker endpoint...")
        start_time = time.time()
        
        # Call the transcribe method - note we don't pass sample_rate here
        # as the NemoSTT class already has this in its config
        result = await nemo_stt.transcribe(audio_np)
        
        process_time = time.time() - start_time
        logger.info(f"Processing completed in {process_time:.2f} seconds")
        logger.info(f"Result: {result}")
        
        if result and "text" in result:
            logger.info(f"SUCCESS: Audio was transcribed to: '{result['text']}'")
            logger.info(f"Connection to NEMO Docker endpoint is working!")
        else:
            logger.error("FAILURE: Did not get expected transcription result")
        
    except Exception as e:
        logger.error(f"Error testing NEMO Docker connection: {e}", exc_info=True)
    finally:
        # Clean up
        try:
            if hasattr(nemo_stt, 'docker_client') and nemo_stt.docker_client:
                logger.info("Disconnecting from Docker endpoint...")
                await nemo_stt.docker_client.disconnect()
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

# Entry point
if __name__ == "__main__":
    asyncio.run(test_nemo_docker_connection())
