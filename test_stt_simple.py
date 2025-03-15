#!/usr/bin/env python3
"""
Simple test script for STT service using WebSocket
"""

import asyncio
import json
import wave
import websockets
import logging
import base64
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_stt_service(audio_file_path):
    """Test the STT service with an audio file"""
    logger.info(f"Reading audio file: {audio_file_path}")
    
    # Read the WAV file as binary data
    with open(audio_file_path, 'rb') as f:
        audio_binary = f.read()
    
    # Encode binary data to base64 string for JSON compatibility
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    logger.info(f"Audio file loaded, size: {len(audio_binary)} bytes")
    
    # Connect to the STT service
    # Get STT host and port from environment variables or use default
    stt_host = os.getenv("STT_HOST", "localhost")
    stt_port = os.getenv("STT_SERVER_PORT", "5002")
    uri = f"ws://{stt_host}:{stt_port}/ws/transcribe"
    logger.info(f"Connecting to STT service at {uri}")
    
    try:
        async with websockets.connect(uri, close_timeout=60) as websocket:
            # Send audio data
            message = {
                "type": "audio",
                "audio_data": audio_base64,
                "format": "wav"  # Assuming WAV format
            }
            
            logger.info("Sending audio data...")
            await websocket.send(json.dumps(message))
            
            # Wait for responses
            logger.info("Waiting for messages from STT service...")
            
            # Keep receiving messages until we get a transcription result or an error
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30)
                    result = json.loads(response)
                    
                    if result.get("type") == "transcription":
                        logger.info("\nTranscription result:")
                        logger.info(json.dumps(result, indent=2))
                        # We got what we came for, so break the loop
                        break
                    elif result.get("type") == "status":
                        # Just a status update, keep waiting
                        logger.info(f"Status update: {result.get('message')}")
                    elif result.get("type") == "semantic":
                        logger.info(f"Semantic results: Significance = {result.get('significance')}")
                    elif result.get("type") == "error":
                        logger.error(f"Error from STT service: {result.get('message')}")
                        break
                    else:
                        logger.info(f"Received message: {json.dumps(result)}")
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for transcription result")
                    break
            
            # Try to send a ping to keep connection alive
            try:
                await websocket.send(json.dumps({"type": "ping"}))
                pong = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info(f"Ping response: {pong}")
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
    
    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}")

if __name__ == "__main__":
    import sys
    audio_file = "/workspace/001-intro.wav" if len(sys.argv) < 2 else sys.argv[1]
    asyncio.run(test_stt_service(audio_file))
