import asyncio
import logging
from typing import Optional
from livekit import rtc

logger = logging.getLogger(__name__)

class LiveKitHandler:
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.room: Optional[rtc.Room] = None
        self._setup_room()

    def _setup_room(self):
        """Setup LiveKit room with default configuration"""
        self.room = rtc.Room()
        
    async def connect(self, url: str, token: str):
        """Connect to LiveKit room"""
        try:
            await self.room.connect(url, token)
            logger.info(f"Connected to room: {self.room_name}")
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            raise

    async def disconnect(self):
        """Disconnect from LiveKit room"""
        if self.room:
            await self.room.disconnect()
            logger.info("Disconnected from room")

    async def send_audio_chunk(self, chunk_data: bytes):
        """Send audio chunk through LiveKit with proper format conversion"""
        if not self.room:
            logger.error("Not connected to room")
            return False

        try:
            # Convert to proper format for LiveKit (48kHz stereo)
            import numpy as np
            from voice_core.utils.audio_utils import resample_audio

            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(chunk_data, dtype=np.int16)
            
            # Resample to 48kHz if needed (assuming input is 16kHz)
            resampled = resample_audio(audio_array, 16000, 48000)
            
            # Convert mono to stereo
            stereo = np.column_stack((resampled, resampled))
            
            # Create audio frame for LiveKit
            frame = rtc.AudioFrame(
                data=stereo.tobytes(),
                sample_rate=48000,  # LiveKit requires 48kHz
                channels=2,  # Stereo required for compatibility
                samples_per_channel=len(stereo)
            )
            
            # Get local participant and publish
            local_participant = self.room.local_participant
            if not local_participant:
                logger.error("No local participant available")
                return False
                
            # Publish the frame
            local_participant.publish_audio_frame(frame)
            return True

        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}", exc_info=True)
            return False
