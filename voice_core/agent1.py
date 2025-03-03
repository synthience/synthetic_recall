from __future__ import annotations

import os
import sys
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any, Callable, Coroutine, AsyncIterator
from datetime import datetime

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero

# --------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Prevent duplicate logs from livekit
logging.getLogger("livekit").propagate = False
logging.getLogger("livekit.agents").propagate = False

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Environment Variables
# --------------------------------------------------------------------------------
load_dotenv()

LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')

# Optional overrides for TTS voice and STT model
DEFAULT_TTS_VOICE = os.getenv('EDGE_TTS_VOICE', 'en-US-AndrewNeural')
DEFAULT_STT_MODEL = os.getenv('OPENAI_STT_MODEL', 'whisper-1')

if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    raise ValueError("Missing required environment variables: LIVEKIT_API_KEY and LIVEKIT_API_SECRET")


# --------------------------------------------------------------------------------
# Prewarm Function
# --------------------------------------------------------------------------------
def prewarm(proc: JobContext) -> None:
    """
    Preload or "warm up" any large or time-intensive resources to improve 
    performance at runtime.
    """
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD prewarmed successfully.")


# --------------------------------------------------------------------------------
# Audio Utilities
# --------------------------------------------------------------------------------
class AudioFrame:
    """
    Wrapper for audio data that provides the frame interface expected by livekit.
    Assumes 16-bit mono audio at 24kHz by default.
    """

    def __init__(self, data: bytes):
        # Warn if data length is not aligned properly for 16-bit audio frames
        if len(data) % 2 != 0:
            logging.warning("Audio data length is not aligned with 16-bit frames")

        from livekit.rtc import AudioFrame as LiveKitAudioFrame
        # Get audio settings from environment
        sample_rate = int(os.getenv('LIVEKIT_SAMPLE_RATE', '48000'))
        num_channels = int(os.getenv('LIVEKIT_CHANNELS', '2'))
        
        self._lk_frame = LiveKitAudioFrame(
            data=data,
            samples_per_channel=len(data) // 2,
            sample_rate=sample_rate,
            num_channels=num_channels
        )
        # Store these for reference
        self.data = data
        self.samples_per_channel = len(data) // 2
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    def __bytes__(self) -> bytes:
        return self.data

    def frame(self) -> Any:
        """
        Return the underlying LiveKit audio frame object.
        """
        return self._lk_frame


# --------------------------------------------------------------------------------
# Cleanup Helpers
# --------------------------------------------------------------------------------
async def cleanup_connection(assistant: Optional[VoicePipelineAgent], ctx: JobContext) -> None:
    """
    Gracefully cleanup the connection and resources.
    """
    try:
        if assistant:
            logger.info("Stopping the voice assistant...")
            # Attempt to call either stop() or close() if available
            if hasattr(assistant, 'stop'):
                assistant.stop()
            elif hasattr(assistant, 'close'):
                await assistant.close()

        if ctx and ctx.room:
            logger.info("Disconnecting from room...")
            if ctx.room.connection_state != 0:  # 0 = Disconnected
                try:
                    await ctx.room.disconnect()
                    # Wait briefly for the disconnection to complete
                    for _ in range(5):  # Up to 5 seconds
                        if ctx.room.connection_state == 0:
                            break
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Error during room disconnect: {str(e)}")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        # Swallow exceptions in cleanup so they don't mask real errors


async def force_room_cleanup(ctx: JobContext) -> None:
    """Force cleanup of room state with extended wait times"""
    try:
        logger.info("Starting forced room cleanup...")
        
        if hasattr(ctx, 'room') and ctx.room:
            try:
                # Log current state
                if hasattr(ctx.room, '_participants'):
                    participants = list(ctx.room._participants.values())
                    logger.info(f"Current participants in room: {[p.identity for p in participants]}")
                
                # Close WebSocket first
                if hasattr(ctx.room, '_ws') and ctx.room._ws:
                    logger.info("Closing existing WebSocket connection...")
                    try:
                        await ctx.room._ws.close()
                        await asyncio.sleep(2)  # Wait for WebSocket to close
                    except Exception as e:
                        logger.warning(f"Error closing WebSocket: {e}")
                
                # Force disconnect and wait
                logger.info("Forcing room disconnect...")
                await ctx.room.disconnect()
                await wait_for_disconnect(ctx, timeout=15)  # Extended wait
                
            except Exception as e:
                logger.warning(f"Error during graceful cleanup: {e}")
        
        # Clear mutable state
        if hasattr(ctx, 'room'):
            try:
                if hasattr(ctx.room, '_participants'):
                    ctx.room._participants.clear()
                if hasattr(ctx.room, '_ws'):
                    ctx.room._ws = None
                if hasattr(ctx.room, '_remote_tracks'):
                    ctx.room._remote_tracks.clear()
                
            except Exception as e:
                logger.warning(f"Error clearing room state: {e}")
        
        # Clear any cached token
        if hasattr(ctx, '_access_token'):
            delattr(ctx, '_access_token')
        
        # Extended wait for server cleanup
        await asyncio.sleep(5)
        logger.info("Room cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during room cleanup: {e}")
        raise


async def wait_for_disconnect(ctx, timeout: int = 15) -> bool:
    """
    Wait for room to fully disconnect with extended timeout.
    Returns True if disconnect confirmed, False if timeout.
    """
    logger.info(f"Waiting up to {timeout} seconds for room disconnect...")
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if not hasattr(ctx, 'room') or not ctx.room or ctx.room.connection_state == 0:
            logger.info("Room disconnect confirmed")
            return True
        await asyncio.sleep(1)
        
    logger.warning(f"Room disconnect timeout after {timeout} seconds")
    return False


async def verify_room_state(ctx: JobContext) -> bool:
    """
    Verify that the room is truly clean and ready for a new connection
    """
    try:
        if not hasattr(ctx, 'room'):
            return True
            
        # Check WebSocket state
        if hasattr(ctx.room, '_ws') and ctx.room._ws and ctx.room._ws.connected:
            logger.warning("Room still has connected WebSocket")
            return False
            
        # Check participants
        if hasattr(ctx.room, '_participants') and ctx.room._participants:
            logger.warning(f"Room still has participants: {[p.identity for p in ctx.room._participants.values()]}")
            return False
            
        # Check connection state without trying to modify it
        if hasattr(ctx.room, 'connection_state') and ctx.room.connection_state != 0:
            logger.warning(f"Room connection state is not disconnected: {ctx.room.connection_state}")
            # Don't return False here since we can't control this property
            
        return True
    except Exception as e:
        logger.error(f"Error verifying room state: {e}")
        return False


# --------------------------------------------------------------------------------
# Main Application Entrypoint
# --------------------------------------------------------------------------------
async def connect_with_retry(ctx: JobContext, room_name: str, assistant: VoicePipelineAgent) -> None:
    """Connect to room with retry logic and IPC error handling"""
    logger = logging.getLogger(__name__)
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connection attempt {attempt}/{max_retries}")
            logger.info(f"Connecting to room {room_name}...")

            # Connect using the JobContext
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

            # Set up assistant with the room
            assistant.start(ctx.room)
            return

        except Exception as e:
            logger.error(f"Connection attempt {attempt} failed: {str(e)}")
            
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                logger.error("Max retries reached, giving up")
                raise


async def entrypoint(ctx: JobContext) -> None:
    """
    The primary async entrypoint for running the voice assistant
    via a LiveKit connection.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting entrypoint for the voice assistant...")

    # The system prompt or "initial context" for your conversation
    initial_ctx = llm.ChatContext(messages=[
        llm.ChatMessage(role="system", content="You are a helpful voice assistant.")
    ])

    try:
        logger.info("Initializing speech-to-text...")

        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        participant = await ctx.wait_for_participant()

        # Initialize the voice assistant
        agent = VoicePipelineAgent(
            vad=silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=0.6,
            ),
            stt=openai.STT(),
            llm=openai.LLM(model="gpt-4"),
            tts=openai.TTS(voice="alloy"),
            chat_ctx=initial_ctx,
        )

        # Start the agent
        agent.start(ctx.room, participant)
        await agent.say("Hi there, how can I help you today?", allow_interruptions=True)
        
        # Keep the agent running
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Fatal error in entrypoint: {str(e)}")
        raise


def generate_unique_identity() -> str:
    """Generate a guaranteed unique identity using full UUID4"""
    return f"agent_{uuid.uuid4().hex}"


# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="nova",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
