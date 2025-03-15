#!/usr/bin/env python3
"""
Test script for STT service
"""

import asyncio
import json
import sys
import numpy as np
import websockets
import wave

async def test_stt(audio_file_path):
    # Read the WAV file
    print(f"Reading audio file: {audio_file_path}")
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
        
        print(f"Audio data loaded: {audio_data.shape}, sample rate: {sample_rate}")
    
    # Connect to the STT service
    uri = "ws://localhost:5002/ws/transcribe"
    print(f"Connecting to STT service at {uri}")
    
    async with websockets.connect(uri) as websocket:
        # Send audio data
        message = {
            "type": "audio",
            "audio": audio_data.tolist()
        }
        print("Sending audio data...")
        await websocket.send(json.dumps(message))
        
        # Wait for response
        print("Waiting for transcription result...")
        response = await websocket.recv()
        result = json.loads(response)
        
        print("\nTranscription result:")
        print(json.dumps(result, indent=2))
        
        # Close the connection
        print("\nClosing connection")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    asyncio.run(test_stt(audio_file))
