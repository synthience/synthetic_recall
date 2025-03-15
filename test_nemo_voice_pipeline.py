#!/usr/bin/env python
"""Test script to verify the full voice pipeline from EnhancedSTTService to NemoSTT Docker endpoint."""

import os
import asyncio
import logging
import time
import numpy as np
import wave
import sys
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components
from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.stt.nemo_stt import NemoSTT

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("voice_pipeline_test")

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

# Callback functions for monitoring transcriptions
async def on_preliminary_transcript(text):
    logger.info(f"Preliminary transcript: '{text}'")

async def on_nemo_transcription(data):
    logger.info(f"NemoSTT final transcription: {data}")
    if "text" in data:
        logger.info(f"Final text: '{data['text']}'")
        logger.info(f"Confidence: {data.get('confidence', 'N/A')}")
    else:
        logger.warning("No text in transcription response")

async def on_nemo_error(data):
    logger.error(f"Error from NEMO STT: {data}")

async def on_nemo_semantic(data):
    logger.info(f"Semantic processing result: {data}")

# Function to simulate voice frames being processed
async def simulate_audio_frames(enhanced_stt, audio_np, sample_rate, frame_size_ms=20):
    # Calculate frame size in samples
    frame_size = int(sample_rate * frame_size_ms / 1000)
    
    # Split audio into frames
    total_frames = len(audio_np) // frame_size
    
    logger.info(f"Simulating {total_frames} audio frames of {frame_size_ms}ms each")
    
    # Call VAD engine's process_frame method directly
    audio_level_db = -30  # Simulate a reasonable audio level
    
    for i in range(total_frames):
        frame_start = i * frame_size
        frame_end = (i + 1) * frame_size
        frame = audio_np[frame_start:frame_end]
        
        # Process with VAD engine through enhanced STT
        try:
            # Use the VAD engine directly since we're not in a LiveKit environment
            vad_result = enhanced_stt.vad_engine.process_frame(frame, audio_level_db)
            
            # Add to buffer if speaking
            if vad_result["is_speaking"]:
                enhanced_stt.buffer.append(frame)
                frame_duration = len(frame) / sample_rate
                enhanced_stt.buffer_duration += frame_duration
                logger.debug(f"Added frame {i+1}/{total_frames} to buffer, duration: {enhanced_stt.buffer_duration:.2f}s")
            
            # Check for a completed speech segment
            if vad_result["speech_segment_complete"] and vad_result["valid_speech_segment"]:
                logger.info(f"Speech segment complete: {vad_result['speech_duration']:.2f}s")
                
                # Let EnhancedSTTService handle the complete segment
                # This should trigger sending to NemoSTT
                if enhanced_stt.buffer:
                    # Combine buffer into a single array
                    full_audio = np.concatenate(enhanced_stt.buffer)
                    
                    # Get preliminary transcription
                    transcription_result = await enhanced_stt.transcriber.transcribe(full_audio, enhanced_stt.sample_rate)
                    
                    if transcription_result["success"] and transcription_result["text"]:
                        transcript = transcription_result["text"]
                        logger.info(f"Preliminary transcription: '{transcript}'")
                    
                    # This should trigger the NemoSTT processing via the modified EnhancedSTTService
                    logger.info("Waiting for NemoSTT processing response...")
                    
                    # Clear buffer for next segment
                    enhanced_stt.buffer = []
                    enhanced_stt.buffer_duration = 0.0
        
        except Exception as e:
            logger.error(f"Error processing frame {i}: {e}")
        
        # Add small delay to simulate real-time processing
        await asyncio.sleep(0.01)

# Main function to test the integrated pipeline
async def test_voice_pipeline():
    logger.info("Starting voice pipeline test")
    
    try:
        # Create VoiceStateManager
        state_manager = VoiceStateManager()
        
        # Initialize NemoSTT with Docker endpoint
        nemo_config = {
            "docker_endpoint": NEMO_DOCKER_ENDPOINT,
            "enable_streaming": False,
            "sample_rate": 16000
        }
        
        nemo_stt = NemoSTT(nemo_config)
        
        # Register callbacks for NemoSTT
        nemo_stt.register_callback("on_transcription", on_nemo_transcription)
        nemo_stt.register_callback("on_error", on_nemo_error)
        nemo_stt.register_callback("on_semantic", on_nemo_semantic)
        
        # Initialize NemoSTT
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
        
        # Initialize EnhancedSTTService with NemoSTT
        enhanced_stt = EnhancedSTTService(
            state_manager=state_manager,
            whisper_model="small.en",
            device="cpu",
            min_speech_duration=0.3,
            max_speech_duration=30.0,
            energy_threshold=0.05,
            on_transcript=on_preliminary_transcript,
            nemo_stt=nemo_stt
        )
        
        # Initialize EnhancedSTTService
        logger.info("Initializing EnhancedSTTService...")
        await enhanced_stt.initialize()
        logger.info("EnhancedSTTService initialized successfully")
        
        # Load or create test audio
        audio_np = None
        sample_rate = 16000
        
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            audio_np, sample_rate = load_test_audio(sys.argv[1])
        
        # If no file specified or loading failed, create mock audio
        if audio_np is None:
            audio_np, sample_rate = create_mock_audio()
        
        # Simulate processing audio frames
        logger.info("Starting audio frame simulation...")
        await simulate_audio_frames(enhanced_stt, audio_np, sample_rate)
        
        # Wait for NemoSTT to finish processing
        logger.info("Waiting for final results...")
        await asyncio.sleep(3)  # Allow time for Docker service to respond
        
        logger.info("Test completed, cleaning up...")
        
    except Exception as e:
        logger.error(f"Error in voice pipeline test: {e}", exc_info=True)
    finally:
        # Clean up
        try:
            if 'nemo_stt' in locals() and hasattr(nemo_stt, 'docker_client') and nemo_stt.docker_client:
                logger.info("Disconnecting from Docker endpoint...")
                await nemo_stt.docker_client.disconnect()
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

# Entry point
if __name__ == "__main__":
    asyncio.run(test_voice_pipeline())
