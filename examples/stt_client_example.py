#!/usr/bin/env python
"""
Example client for the STT Transcription Service

This script demonstrates how to connect to the STT transcription service
and process real-time transcriptions from audio data.
"""

import asyncio
import websockets
import json
import numpy as np
import argparse
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class STTClient:
    """Client for the STT Transcription Service."""
    
    def __init__(self, url: str = "ws://stt_transcription:5000/ws/transcribe"):
        """
        Initialize the STT client.
        
        Args:
            url: WebSocket URL for the STT transcription service
        """
        self.url = url
        self.websocket = None
        self.connected = False
        self.transcription_callback = None
        
    async def connect(self) -> bool:
        """
        Connect to the STT transcription service.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to STT service at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to STT service: {e}")
            self.connected = False
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from the STT transcription service."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from STT service")
            
    async def send_audio(self, audio_data: np.ndarray) -> None:
        """
        Send audio data to the STT service for transcription.
        
        Args:
            audio_data: Audio data as numpy array (float32, 16kHz mono)
        """
        if not self.connected:
            await self.connect()
            
        if not self.connected:
            logger.error("Cannot send audio: not connected to STT service")
            return
            
        try:
            # Convert numpy array to bytes
            audio_bytes = audio_data.tobytes()
            
            # Send audio data
            await self.websocket.send(audio_bytes)
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            self.connected = False
            
    async def receive_transcription(self) -> Dict[str, Any]:
        """
        Receive transcription result from the STT service.
        
        Returns:
            Dict containing the transcription result
        """
        if not self.connected:
            logger.error("Cannot receive transcription: not connected to STT service")
            return {"error": "Not connected"}
            
        try:
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error receiving transcription: {e}")
            self.connected = False
            return {"error": str(e)}
            
    async def process_continuous_audio(self, audio_generator, callback=None):
        """
        Process continuous audio from a generator and receive transcriptions.
        
        Args:
            audio_generator: Generator yielding audio chunks (numpy arrays)
            callback: Optional callback function to process transcriptions
        """
        if not await self.connect():
            return
            
        self.transcription_callback = callback
        
        # Start a background task to receive transcriptions
        receive_task = asyncio.create_task(self.receive_transcriptions())
        
        try:
            # Process audio chunks from the generator
            async for audio_chunk in audio_generator:
                await self.send_audio(audio_chunk)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
        finally:
            # Cancel the receive task
            receive_task.cancel()
            await self.disconnect()
            
    async def receive_transcriptions(self):
        """Continuously receive and process transcriptions."""
        try:
            while True:
                result = await self.receive_transcription()
                
                if "error" in result:
                    logger.error(f"Transcription error: {result['error']}")
                    continue
                    
                logger.info(f"Transcription: {result.get('text', '')}")
                
                # Call the callback if provided
                if self.transcription_callback:
                    self.transcription_callback(result)
                    
        except asyncio.CancelledError:
            logger.info("Transcription receiving task cancelled")
        except Exception as e:
            logger.error(f"Error in transcription receiving loop: {e}")

# Example audio generator (simulates audio chunks)
async def dummy_audio_generator():
    """Generate dummy audio chunks for testing."""
    # Generate 10 chunks of random audio data (2 seconds each at 16kHz)
    for _ in range(10):
        # Create random audio data (2 seconds at 16kHz = 32000 samples)
        audio_chunk = np.random.uniform(-0.1, 0.1, 32000).astype(np.float32)
        yield audio_chunk
        await asyncio.sleep(2)  # Simulate 2 seconds of audio

# Example callback function
def process_transcription(result):
    """Process transcription results."""
    if "text" in result:
        text = result["text"]
        timestamp = result.get("timestamp", 0)
        print(f"[{timestamp}] Transcription: {text}")

async def main():
    """Main function to demonstrate the STT client."""
    parser = argparse.ArgumentParser(description="STT Client Example")
    parser.add_argument("--url", default="ws://localhost:5500/ws/transcribe", 
                        help="WebSocket URL for the STT service")
    args = parser.parse_args()
    
    # Create STT client
    client = STTClient(url=args.url)
    
    # Process dummy audio
    print("Sending dummy audio data to STT service...")
    await client.process_continuous_audio(
        dummy_audio_generator(),
        callback=process_transcription
    )
    
    print("STT client example completed")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())