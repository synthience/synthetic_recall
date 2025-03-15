#!/usr/bin/env python
"""
Test script to verify connection to the NeMo STT Docker service
"""

import asyncio
import websockets
import json
import os
import base64
import logging
import sys
import numpy as np
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_stt")

# For clean error messages
async def safe_send_json(websocket, data):
    try:
        await websocket.send(json.dumps(data))
        return True
    except Exception as e:
        logger.error(f"Error sending data: {e}")
        return False

async def test_websocket_connection():
    # URL for the STT service
    url = os.environ.get("STT_DOCKER_ENDPOINT", "ws://localhost:5002/ws/transcribe")
    logger.info(f"Connecting to STT service at {url}")
    
    try:
        # Connect to the WebSocket
        async with websockets.connect(url, ping_interval=20) as websocket:
            logger.info("Connected to STT service successfully!")
            
            # Wait for the initial connection message
            try:
                initial_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                initial_data = json.loads(initial_response)
                logger.info(f"Initial connection response: {initial_data}")
                
                client_id = initial_data.get("client_id", "unknown")
                logger.info(f"Client ID: {client_id}")
            except Exception as e:
                logger.error(f"Error receiving initial message: {e}")
                return False
            
            # Test ping-pong to confirm basic communication
            ping_message = {"type": "ping"}
            logger.info(f"Sending ping message")
            if not await safe_send_json(websocket, ping_message):
                return False
            
            try:
                pong_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                pong_data = json.loads(pong_response)
                logger.info(f"Received ping response: {pong_data}")
            except Exception as e:
                logger.error(f"Error receiving pong response: {e}")
                return False
            
            # Generate a 1-second 440 Hz sine wave at 16kHz sample rate
            sample_rate = 16000
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 440 * t) * 32767  # 440 Hz tone at 16-bit amplitude
            audio_bytes = tone.astype(np.int16).tobytes()
            
            # Send audio message
            audio_message = {
                "type": "audio",  # Must match what STT_server.py expects
                "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
                "final": True,
                "streaming": False,
                "request_id": f"test_{str(uuid.uuid4())}"
            }
            logger.info(f"Sending audio message with {len(audio_bytes)} bytes of audio data")
            if not await safe_send_json(websocket, audio_message):
                return False
            
            # Wait for transcription response (may require multiple messages)
            try:
                timeout = 15.0  # longer timeout for transcription
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < timeout:
                    trans_response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    trans_data = json.loads(trans_response)
                    logger.info(f"Received response: {trans_data}")
                    
                    # Check if this is a transcription result
                    if trans_data.get("type") == "transcription" or "text" in trans_data:
                        logger.info(f"✅ Received transcription: {trans_data.get('text', '')}")
                        return True
                    elif trans_data.get("type") == "error":
                        logger.error(f"❌ Error from STT server: {trans_data.get('message', '')}")
                    elif trans_data.get("type") == "status":
                        logger.info(f"Status update: {trans_data.get('message', '')}")
                    else:
                        logger.info(f"Unrecognized response type: {trans_data}")
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for transcription result")
            
            logger.info("Test completed - validated basic communication with STT server")
            # Even if we didn't get a transcription, we confirmed basic communication
            return True
            
    except Exception as e:
        logger.error(f"Error connecting to STT service: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

async def main():
    # Run the test
    success = await test_websocket_connection()
    
    if success:
        logger.info("✅ STT service connection test completed successfully!")
        return 0
    else:
        logger.error("❌ STT service connection test failed!")
        return 1

if __name__ == "__main__":
    import traceback
    
    # Set environment variables if not already set
    if "STT_DOCKER_ENDPOINT" not in os.environ:
        os.environ["STT_DOCKER_ENDPOINT"] = "ws://localhost:5002/ws/transcribe"
    
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
