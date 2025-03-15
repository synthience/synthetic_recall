#!/usr/bin/env python3
"""
Test script for STT service inside Docker
"""

import asyncio
import json
import numpy as np
import websockets
import wave
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_stt(audio_file_path):
    # Read the WAV file
    logger.info(f"Reading audio file: {audio_file_path}")
    with wave.open(audio_file_path, 'rb') as wav_file:
        # Get audio parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        
        # Read all frames
        frames = wav_file.readframes(num_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:  # 32-bit audio
            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert stereo to mono if needed
        if channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        logger.info(f"Audio data loaded: shape={audio_data.shape}, sample_rate={sample_rate}")
    
    # Try both the service directly and through the FastAPI endpoint
    uris = ["ws://localhost:5002/ws/transcribe", "ws://127.0.0.1:5002/ws/transcribe"]
    
    for uri in uris:
        try:
            logger.info(f"Connecting to STT service at {uri}")
            # Set longer timeout for connection and operations
            async with websockets.connect(uri, close_timeout=60) as websocket:
                # Send audio data - split into smaller chunks to avoid large message errors
                chunk_size = 8000  # About 0.5 seconds at 16kHz
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    message = {
                        "type": "audio",
                        "audio": chunk.tolist()
                    }
                    logger.info(f"Sending audio chunk {i//chunk_size + 1}/{(len(audio_data) + chunk_size - 1)//chunk_size}...")
                    await websocket.send(json.dumps(message))
                    logger.info("Chunk sent, waiting for result...")
                    
                    # Wait for response with timeout
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30)
                        result = json.loads(response)
                        logger.info(f"Got response for chunk {i//chunk_size + 1}: {result['type']}")
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for transcription result")
                        break
                
                # Wait for final response
                logger.info("All chunks sent, getting final result...")
                try:
                    # Send a ping to check if connection is still alive
                    await websocket.send(json.dumps({"type": "ping"}))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    logger.info(f"Ping response: {response}")
                except Exception as e:
                    logger.error(f"Error sending ping: {e}")
                
                # Get history
                try:
                    await websocket.send(json.dumps({"type": "get_history"}))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    history = json.loads(response)
                    logger.info("\nTranscription history:")
                    logger.info(json.dumps(history, indent=2))
                except Exception as e:
                    logger.error(f"Error getting history: {e}")
                    
                # Success with this URI
                return
                
        except Exception as e:
            logger.error(f"Error with {uri}: {e}")
            continue
    
    logger.error("Failed to connect to STT service with all URIs")

if __name__ == "__main__":
    asyncio.run(test_stt("/workspace/001-intro.wav"))
