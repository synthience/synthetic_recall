# __init__.py

```py


```

# .gitignore

```
# Python cache files
__pycache__/
*.py[cod]

# Environment files
.env

# Temporary audio chunks
temp_chunks/

# Virtual environment
.venv/

```

# .pytest_cache\.gitignore

```
# Created by pytest automatically.
*

```

# .pytest_cache\CACHEDIR.TAG

```TAG
Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#	https://bford.info/cachedir/spec.html

```

# .pytest_cache\README.md

```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

# .pytest_cache\v\cache\lastfailed

```
{
  "tests/test_stt_pipeline.py::test_stt_pipeline_integration": true,
  "tests/test_stt_pipeline.py::test_vad_detection": true
}
```

# .pytest_cache\v\cache\nodeids

```
[
  "tests/test_stt_pipeline.py::test_stt_pipeline_integration",
  "tests/test_stt_pipeline.py::test_vad_detection"
]
```

# .pytest_cache\v\cache\stepwise

```
[]
```

# agent1.py

```py
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

```

# agent2.py

```py
"""Enhanced voice agent implementation with reliable UI updates and interruption handling."""

from __future__ import annotations

import os
import sys
import uuid
import logging
import asyncio
import time
import json
import traceback
from typing import Optional, Dict, Any, Callable, Coroutine, AsyncIterator, List
from datetime import datetime

from dotenv import load_dotenv
import torch
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)

# Import enhanced components
from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.llm.llm_pipeline import LocalLLMPipeline
from voice_core.config.config import LucidiaConfig, LLMConfig, WhisperConfig, TTSConfig, StateConfig, RoomConfig
from voice_core.connection_utils import force_room_cleanup, cleanup_connection

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

LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', 'devkey')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', 'secret')

# Optional overrides
DEFAULT_TTS_VOICE = os.getenv('EDGE_TTS_VOICE', 'en-US-AvaMultilingualNeural')
DEFAULT_STT_MODEL = os.getenv('OPENAI_STT_MODEL', 'whisper-1')
INITIAL_GREETING = os.getenv('INITIAL_GREETING', "Hello! I'm Lucidia, your voice assistant. How can I help you today?")

class LucidiaVoiceAgent:
    """Enhanced voice agent with proper UI integration and state management."""
    
    def __init__(self, 
                 job_context: JobContext,
                 initial_greeting: Optional[str] = None,
                 config: Optional[LucidiaConfig] = None):
        """Initialize the voice agent."""
        self.job_context = job_context
        self.room = None
        self.initial_greeting = initial_greeting or INITIAL_GREETING
        self.session_id = str(uuid.uuid4())
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Create or use provided configuration
        self.config = config or LucidiaConfig()
        
        # Create state manager
        self.state_manager = VoiceStateManager(
            processing_timeout=self.config.state.processing_timeout,
            speaking_timeout=self.config.state.speaking_timeout,
            debug=self.config.state.debug
        )
        
        # Create core services
        self.stt_service = None
        self.tts_service = None
        self.llm_service = None
        
        # Connection state
        self._initialized = False
        self._running = False
        self._shutdown_requested = False
        self._heartbeat_task = None
        self._connection_retry_count = 0
        self._max_connection_retries = 5
        
        # Conversation state
        self.conversation_history = []
        self.max_history = 10
        
        # Metrics tracking
        self.metrics = {
            "start_time": time.time(),
            "conversations": 0,
            "successful_responses": 0,
            "errors": 0,
            "interruptions": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        }
        
        # Task tracking for better cleanup
        self._tasks = {}
        
    async def initialize(self) -> None:
        """Initialize services and connect to room."""
        try:
            logger.info(f"Initializing Lucidia voice agent (session: {self.session_id})")
            
            # Connection handling first
            if not self.job_context or not self.job_context.room:
                logger.error("No room provided in job context")
                raise RuntimeError("No room provided in job context")
                
            # Store room reference
            self.room = self.job_context.room
            
            # Start heartbeat for connection monitoring
            self._heartbeat_task = asyncio.create_task(self._connection_heartbeat())
            self._tasks["heartbeat"] = self._heartbeat_task
            
            # Initialize state manager with room
            await self.state_manager.set_room(self.room)
            
            # Initialize core services
            self.stt_service = EnhancedSTTService(
                state_manager=self.state_manager,
                whisper_model=self.config.whisper.model_name,
                device=self.config.whisper.device,
                min_speech_duration=self.config.whisper.min_speech_duration,
                max_speech_duration=self.config.whisper.max_audio_length,
                energy_threshold=self.config.whisper.speech_confidence_threshold,
                on_transcript=self._handle_transcript
            )
            
            self.tts_service = InterruptibleTTSService(
                state_manager=self.state_manager,
                voice=self.config.tts.voice,
                sample_rate=self.config.tts.sample_rate,
                num_channels=self.config.tts.channels,
                on_interrupt=self._handle_tts_interrupt,
                on_complete=self._handle_tts_complete
            )
            
            self.llm_service = LocalLLMPipeline(self.config.llm)
            
            # Register transcript handler with state manager
            self.state_manager.register_transcript_handler(self._handle_transcript)
            
            # Initialize services
            logger.info("Initializing STT service...")
            await self.stt_service.initialize()
            
            logger.info("Initializing TTS service...")
            await self.tts_service.initialize()
            await self.tts_service.set_room(self.room)
            
            logger.info("Initializing LLM service...")
            await self.llm_service.initialize()
            
            # Register room with services
            self.stt_service.set_room(self.room)
            
            # Publish initialization status to UI
            await self._publish_ui_update({
                "type": "agent_initialized",
                "timestamp": time.time(),
                "agent_id": self.session_id,
                "config": {
                    "whisper_model": self.config.whisper.model_name,
                    "tts_voice": self.config.tts.voice,
                    "llm_model": self.config.llm.model
                }
            })
            
            # Mark as initialized
            self._initialized = True
            logger.info("Lucidia voice agent initialized")
        
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            await self.state_manager.register_error(e, "initialization")
            raise
            
    async def _connection_heartbeat(self) -> None:
        """Monitor connection health with heartbeat."""
        try:
            logger.info("Starting connection heartbeat")
            heartbeat_interval = 5  # seconds
            max_retries = 5
            backoff_factor = 2
            attempt = 0
            
            while not self._shutdown_requested:
                try:
                    # Check room connection
                    if not self.room:
                        logger.warning("No room available for heartbeat")
                        await asyncio.sleep(heartbeat_interval)
                        continue
                        
                    # Check connection state
                    if self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                        if attempt < max_retries:
                            self._connection_retry_count += 1
                            logger.info(f"Attempting reconnection ({self._connection_retry_count}/{max_retries})")
                            await self.job_context.reconnect()
                            attempt += 1
                            await asyncio.sleep(heartbeat_interval * (backoff_factor ** attempt))
                        else:
                            logger.error("Max reconnection attempts reached, shutting down")
                            self._shutdown_requested = True
                            await self.cleanup()
                            break
                    else:
                        # Reset retry count on successful connection
                        attempt = 0  # Reset on successful connection
                        
                    # Send heartbeat data for debugging
                    if self.room and self.room.local_participant and attempt % 12 == 0:
                        await self._publish_ui_update({
                            "type": "heartbeat",
                            "count": self._connection_retry_count,
                            "timestamp": time.time(),
                            "agent_id": self.session_id,
                            "state": self.state_manager.current_state.name,
                            "metrics": self._get_metrics()
                        })
                        
                    # Wait for next interval
                    await asyncio.sleep(heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    await asyncio.sleep(heartbeat_interval)
                
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in heartbeat: {e}", exc_info=True)
            
    async def start(self) -> None:
        """Start the voice agent with greeting."""
        try:
            if not self._initialized:
                logger.error("Agent not initialized")
                raise RuntimeError("Agent not initialized")
                
            # Mark as running
            self._running = True
            
            # Transition to idle state
            await self.state_manager.transition_to(VoiceState.IDLE)
            
            # Send greeting
            logger.info("Sending greeting")
            await self._send_greeting()
            
            # Reset STT buffer
            await self.stt_service.clear_buffer()
            
            # Start listening
            await self.state_manager.transition_to(VoiceState.LISTENING)
            logger.info("Listening for speech input")
            
        except Exception as e:
            logger.error(f"Error starting agent: {e}", exc_info=True)
            await self.state_manager.transition_to(VoiceState.ERROR, {"error": str(e)})
            
    async def _publish_ui_update(self, data: Dict[str, Any]) -> bool:
        """Publish UI update with automatic error handling."""
        if not self.room or not self.room.local_participant:
            return False
            
        try:
            # Prepare data as bytes
            message_bytes = json.dumps(data).encode()
            
            # Publish with retries
            return await self.state_manager._publish_with_retry(message_bytes, description=data.get("type", "data"))
            
        except Exception as e:
            logger.error(f"Failed to publish UI update: {e}")
            return False
            
    async def _handle_transcript(self, text: str) -> None:
        """Handle transcript from STT service with coordinated state transitions."""
        start_time = time.time()
        try:
            logger.info(f"Processing transcript: '{text[:50]}...'")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Use state manager to publish transcription with explicit user role
            # This ensures it's displayed correctly in the frontend
            await self.state_manager.publish_transcription(text, "user", True)
            
            # Trim conversation history if needed
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Process with LLM
            llm_task = asyncio.create_task(self.llm_service.generate_response(text))
            self.state_manager._in_progress_task = llm_task
            self._tasks["llm_processing"] = llm_task

            try:
                # Process with timeout
                response = await asyncio.wait_for(
                    llm_task, 
                    timeout=self.config.llm.timeout
                )
                
                if response:
                    # Update conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Track metrics
                    self.metrics["conversations"] += 1
                    self.metrics["successful_responses"] += 1
                    response_time = time.time() - start_time
                    self.metrics["total_response_time"] += response_time
                    self.metrics["avg_response_time"] = (
                        self.metrics["total_response_time"] / 
                        self.metrics["successful_responses"]
                    )
                    
                    # Clean up task reference
                    self._tasks.pop("llm_processing", None)
                    
                    # Create and await TTS task explicitly
                    tts_task = asyncio.create_task(self.tts_service.speak(response, self._get_local_identity()))
                    self._tasks["tts_speaking"] = tts_task
                    
                    await self.state_manager.start_speaking(tts_task)
                    
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        logger.info("TTS task was cancelled")
                    finally:
                        self._tasks.pop("tts_speaking", None)
                    
                    await self.stt_service.clear_buffer()
                else:
                    # Handle empty response
                    logger.warning("LLM returned empty response")
                    error_message = "I'm sorry, I couldn't generate a response."
                    tts_task = asyncio.create_task(self.tts_service.speak(error_message, self._get_local_identity()))
                    self._tasks["tts_error"] = tts_task
                    
                    await self.state_manager.start_speaking(tts_task)
                    
                    try:
                        await tts_task
                    except asyncio.CancelledError:
                        logger.info("TTS error task was cancelled")
                    finally:
                        self._tasks.pop("tts_error", None)
                    
                    # Update metrics
                    self.metrics["errors"] += 1
                    await self.stt_service.clear_buffer()

            except asyncio.TimeoutError:
                logger.warning(f"LLM processing timed out after {self.config.llm.timeout}s")
                
                # Cancel the LLM task
                llm_task.cancel()
                try:
                    await asyncio.wait_for(llm_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info("LLM task cancelled after timeout")
                
                # Clean up task reference
                self._tasks.pop("llm_processing", None)
                
                # Let the user know about timeout
                error_message = "Sorry, processing took too long."
                tts_task = asyncio.create_task(self.tts_service.speak(error_message, self._get_local_identity()))
                self._tasks["tts_timeout"] = tts_task
                
                await self.state_manager.start_speaking(tts_task)
                
                try:
                    await tts_task
                except asyncio.CancelledError:
                    logger.info("TTS timeout task was cancelled")
                finally:
                    self._tasks.pop("tts_timeout", None)
                
                # Update metrics
                self.metrics["errors"] += 1
                await self.stt_service.clear_buffer()

            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                
                # Clean up task reference
                self._tasks.pop("llm_processing", None)
                
                # Let the user know about the error
                error_message = "Sorry, an error occurred."
                tts_task = asyncio.create_task(self.tts_service.speak(error_message, self._get_local_identity()))
                self._tasks["tts_error"] = tts_task
                
                await self.state_manager.start_speaking(tts_task)
                
                try:
                    await tts_task
                except asyncio.CancelledError:
                    logger.info("TTS error task was cancelled")
                finally:
                    self._tasks.pop("tts_error", None)
                
                # Update metrics
                self.metrics["errors"] += 1
                await self.stt_service.clear_buffer()

            finally:
                # Ensure we're listening unless in ERROR or INTERRUPTED state
                current_state = self.state_manager.current_state
                if current_state not in [VoiceState.ERROR, VoiceState.INTERRUPTED]:
                    await self.state_manager.transition_to(VoiceState.LISTENING)
                
                self.logger.debug(f"Transcript processing completed in {time.time() - start_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error handling transcript: {e}", exc_info=True)
            await self.state_manager.transition_to(VoiceState.ERROR, {"error": str(e)})
            
    def _get_context(self) -> Dict[str, Any]:
        """Get conversation context for LLM."""
        return {
            "conversation_history": self.conversation_history,
            "system": self.config.llm.system_prompt
        }
    
    async def _handle_tts_interrupt(self) -> None:
        """Handle TTS interruption."""
        logger.info("TTS interrupted")
        
        # Track interruption
        self.metrics["interruptions"] += 1
        
        # Ensure we handle the interruption properly
        if self.state_manager.current_state == VoiceState.INTERRUPTED:
            # Wait briefly to ensure any new speech is processed
            await asyncio.sleep(0.1)
            
            # Transition back to LISTENING if there's no new processing
            if self.state_manager.current_state == VoiceState.INTERRUPTED:
                await self.state_manager.transition_to(VoiceState.LISTENING, {"reason": "interrupt_handled"})
        
    async def _handle_tts_complete(self, text: str) -> None:
        """Handle TTS completion with state transition."""
        logger.info(f"TTS completed: '{text[:50]}...'")
        
        # Transition back to listening
        if self.state_manager.current_state == VoiceState.SPEAKING:
            await self.state_manager.transition_to(VoiceState.LISTENING, {"reason": "tts_complete"})
        
    async def _send_greeting(self) -> None:
        """Send initial greeting."""
        try:
            logger.info(f"Sending greeting: '{self.initial_greeting}'")
            
            # Create TTS task
            tts_task = asyncio.create_task(self.tts_service.speak(self.initial_greeting, self._get_local_identity()))
            self._tasks["tts_greeting"] = tts_task
            
            # Register with state manager
            await self.state_manager.start_speaking(tts_task)
            
            try:
                await tts_task
            except asyncio.CancelledError:
                logger.info("Greeting TTS task was cancelled")
            finally:
                self._tasks.pop("tts_greeting", None)
            
        except Exception as e:
            logger.error(f"Error sending greeting: {e}", exc_info=True)
            
    async def process_audio(self, track: rtc.AudioTrack) -> None:
        """Process audio from track."""
        try:
            if self.stt_service and self._running:
                process_task = asyncio.create_task(self.stt_service.process_audio(track))
                self._tasks["process_audio"] = process_task
                
                try:
                    await process_task
                except asyncio.CancelledError:
                    logger.info("Audio processing task was cancelled")
                finally:
                    self._tasks.pop("process_audio", None)
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            
    def _get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        uptime = time.time() - self.metrics["start_time"]
        
        return {
            "uptime": uptime,
            "conversations": self.metrics["conversations"],
            "successful_responses": self.metrics["successful_responses"],
            "errors": self.metrics["errors"],
            "interruptions": self.metrics["interruptions"],
            "avg_response_time": self.metrics["avg_response_time"],
            "stt_stats": self.stt_service.get_stats() if self.stt_service else {},
            "tts_stats": self.tts_service.get_stats() if self.tts_service else {},
            "state_metrics": self.state_manager.get_analytics()
        }
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up agent resources")
        
        self._shutdown_requested = True
        self._running = False
        
        # Publish final metrics
        try:
            await self._publish_ui_update({
                "type": "agent_metrics",
                "metrics": self._get_metrics(),
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Error publishing final metrics: {e}")
        
        # Cancel all tracked tasks
        for name, task in list(self._tasks.items()):
            if not task.done():
                logger.info(f"Cancelling task: {name}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        # Clear task dictionary
        self._tasks.clear()
                
        # Clean up services with buffer clearing
        services = []
        
        if self.stt_service:
            services.append(self.stt_service.cleanup())
            await self.stt_service.clear_buffer()
            
        if self.tts_service:
            services.append(self.tts_service.cleanup())
            
        if self.llm_service:
            services.append(self.llm_service.cleanup())
            
        # Clean up state manager
        if self.state_manager:
            services.append(self.state_manager.cleanup())
            
        # Wait for all services to clean up
        if services:
            await asyncio.gather(*services, return_exceptions=True)
            
        # Publish final cleanup
        await self._publish_ui_update({
            "type": "agent_cleanup",
            "timestamp": time.time(),
            "agent_id": self.session_id
        })

    def _get_local_identity(self) -> str:
        """Get the local participant identity or a default value if not available."""
        if self.room and self.room.local_participant:
            return self.room.local_participant.identity
        return "assistant"  # Default fallback

# --------------------------------------------------------------------------------
# Debug Utilities
# --------------------------------------------------------------------------------
def debug_inspect_object(obj, max_depth=3, prefix='', current_depth=0):
    """Recursively inspect an object's attributes for debugging."""
    if current_depth > max_depth:
        return f"{prefix}[MAX DEPTH REACHED]"
    
    result = []
    
    if obj is None:
        return f"{prefix}None"
    
    try:
        # Try to get attributes
        attrs = dir(obj)
        for attr in attrs:
            # Skip private attributes and methods
            if attr.startswith('_') or callable(getattr(obj, attr)):
                continue
            
            try:
                value = getattr(obj, attr)
                if isinstance(value, (str, int, float, bool, type(None))):
                    result.append(f"{prefix}{attr}: {value}")
                else:
                    result.append(f"{prefix}{attr}: {type(value).__name__}")
                    result.append(debug_inspect_object(value, max_depth, prefix + '  ', current_depth + 1))
            except Exception as e:
                result.append(f"{prefix}{attr}: [ERROR: {str(e)}]")
    except Exception as e:
        return f"{prefix}[ERROR inspecting object: {str(e)}]"
    
    return '\n'.join(result)

# --------------------------------------------------------------------------------
# Main Application Entrypoint
# --------------------------------------------------------------------------------
async def connect_with_retry(ctx: JobContext, room_name: str, participant_identity: Optional[str] = None) -> rtc.Room:
    """Connect to a LiveKit room with retry logic.
    
    Args:
        ctx: The JobContext object from LiveKit
        room_name: The name of the room to connect to
        participant_identity: The identity to use for the participant
        
    Returns:
        The connected Room object
        
    Raises:
        Exception: If connection fails after max retries
    """
    max_retries = 5
    retry_delay = 2  # seconds
    
    logger.info(f"Connecting to room '{room_name}' with participant identity '{participant_identity}'")
    
    # Log LiveKit URL for debugging
    livekit_url = os.environ.get("LIVEKIT_URL", "Not set in environment")
    logger.info(f"LiveKit URL: {livekit_url}")
    
    # Set environment variables for room name and participant identity
    if room_name:
        os.environ["LIVEKIT_ROOM"] = room_name
    
    if participant_identity:
        os.environ["LIVEKIT_PARTICIPANT_IDENTITY"] = participant_identity
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connection attempt {attempt}/{max_retries}...")
            
            # Connect to the room using ctx.connect() with the correct parameters
            # The room name and participant identity should be set via environment variables
            await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
            
            # Get the room from ctx.room after connecting
            room = ctx.room
            
            # Get room properties safely
            try:
                # Simple logging that doesn't try to access properties that might be coroutines
                logger.info(f"Successfully connected to room")
                logger.info(f"Local participant connected")
                
                # Log other participants count only
                other_participants = list(room.participants.values())
                logger.info(f"Other participants in room: {len(other_participants)}")
            except Exception as e:
                logger.error(f"Error accessing room properties: {e}")
                # Continue anyway since we're connected
            
            return room
            
        except Exception as e:
            logger.error(f"Connection attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                # Increase delay for next attempt (exponential backoff)
                retry_delay = min(retry_delay * 2, 30)  # Cap at 30 seconds
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                logger.error(f"Last error: {e}")
                logger.error(f"Room name: {room_name}")
                logger.error(f"Participant identity: {participant_identity}")
                logger.error(f"LiveKit URL: {livekit_url}")
                raise RuntimeError(f"Failed to connect to LiveKit room after {max_retries} attempts: {e}")

async def entrypoint(ctx: JobContext, cli_room_name: Optional[str] = None, cli_participant_identity: Optional[str] = None) -> None:
    """Main entrypoint function for the voice agent.
    
    Args:
        ctx: The JobContext object from LiveKit
        cli_room_name: Room name from CLI arguments (highest priority)
        cli_participant_identity: Participant identity from CLI arguments (highest priority)
    """
    logger.info("Lucidia Voice Agent starting...")
    
    # Parameter resolution with clear precedence:
    # 1. CLI arguments (passed to this function)
    # 2. JobContext attributes
    # 3. Environment variables
    # 4. Default values
    
    # Resolve room name
    room_name = None
    room_name_source = "default"
    
    # 1. CLI arguments (highest priority)
    if cli_room_name:
        room_name = cli_room_name
        room_name_source = "CLI arguments"
    
    # 2. JobContext attributes
    elif hasattr(ctx, "info") and ctx.info and hasattr(ctx.info, "job") and ctx.info.job:
        if hasattr(ctx.info.job, "room") and ctx.info.job.room and hasattr(ctx.info.job.room, "name"):
            room_name = ctx.info.job.room.name
            room_name_source = "JobContext"
    
    # 3. Environment variables
    if not room_name:
        # Try LIVEKIT_ROOM first (standard LiveKit variable)
        room_name = os.environ.get("LIVEKIT_ROOM")
        if room_name:
            room_name_source = "LIVEKIT_ROOM environment variable"
        else:
            # Try ROOM_NAME as fallback
            room_name = os.environ.get("ROOM_NAME")
            if room_name:
                room_name_source = "ROOM_NAME environment variable"
    
    # 4. Default value
    if not room_name:
        room_name = "lucidia_room"
        room_name_source = "default hardcoded value"
    
    # Resolve participant identity
    participant_identity = None
    identity_source = "default"
    
    # 1. CLI arguments (highest priority)
    if cli_participant_identity:
        participant_identity = cli_participant_identity
        identity_source = "CLI arguments"
    
    # 2. JobContext attributes
    elif hasattr(ctx, "info") and ctx.info:
        accept_args = getattr(ctx.info, "accept_arguments", None)
        if accept_args and hasattr(accept_args, "identity") and accept_args.identity:
            participant_identity = accept_args.identity
            identity_source = "JobContext"
    
    # 3. Environment variables
    if not participant_identity:
        participant_identity = os.environ.get("LIVEKIT_PARTICIPANT_IDENTITY")
        if participant_identity:
            identity_source = "LIVEKIT_PARTICIPANT_IDENTITY environment variable"
    
    # 4. Default value (generate a unique ID)
    if not participant_identity:
        participant_identity = f"lucidia-{uuid.uuid4()}"
        identity_source = "generated unique ID"
    
    # Log the resolved parameters
    logger.info(f"Room name: '{room_name}' (source: {room_name_source})")
    logger.info(f"Participant identity: '{participant_identity}' (source: {identity_source})")
    
    # Connect to the room with retry
    room = await connect_with_retry(ctx, room_name, participant_identity)
    
    agent = None

    try:
        # Create configuration from environment
        config = LucidiaConfig()
        
        # Update config with effective values
        config.room.room_name = room_name
        config.room.participant_identity = participant_identity
        
        # Create and initialize agent
        agent = LucidiaVoiceAgent(ctx, INITIAL_GREETING, config)
        await agent.initialize()
        
        # Start agent
        await agent.start()
        
        # Find a suitable participant with audio tracks
        async def find_audio_track():
            for _ in range(30):  # Try for 30 seconds
                for participant in ctx.room.remote_participants.values():
                    for pub in participant.track_publications.values():
                        if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                            return pub.track
                await asyncio.sleep(1)
            return None
            
        # Look for audio track
        audio_track = await find_audio_track()
        if audio_track:
            # Process audio
            logger.info(f"Found audio track, processing...")
            await agent.process_audio(audio_track)
        else:
            # Just keep agent alive waiting for participants
            logger.info("No audio track found, waiting...")
            
        # Keep running until disconnected
        while ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            
        logger.info("Room disconnected, ending agent")

    except Exception as e:
        logger.error(f"Fatal error in entrypoint: {e}", exc_info=True)
    finally:
        # Clean up resources
        if agent:
            await agent.cleanup()
        await cleanup_connection(agent, ctx)

def generate_unique_identity() -> str:
    """Generate a guaranteed unique identity using full UUID4"""
    return f"agent_{uuid.uuid4().hex}"

# Simplified direct entrypoint for the CLI to call
# This should be picklable since it's a regular named function
async def entrypoint_cli(ctx: JobContext) -> None:
    """Entrypoint function for the CLI interface."""
    try:
        # Log detailed information about the JobContext for debugging
        logger.info("JobContext structure:")
        logger.info(debug_inspect_object(ctx))
        
        # Extract room name from context with proper fallbacks
        room_name = None
        participant_identity = None
        
        # First try to get from CLI args (highest priority)
        if hasattr(ctx, 'args') and ctx.args:
            logger.info(f"CLI args found in JobContext: {ctx.args}")
            if hasattr(ctx.args, 'room'):
                room_name = ctx.args.room
                logger.info(f"Room name from CLI args: {room_name}")
            if hasattr(ctx.args, 'participant_identity'):
                participant_identity = ctx.args.participant_identity
                logger.info(f"Participant identity from CLI args: {participant_identity}")
        
        # Log environment variables for debugging
        env_vars = {
            'LIVEKIT_ROOM': os.environ.get('LIVEKIT_ROOM'),
            'ROOM_NAME': os.environ.get('ROOM_NAME'),
            'LIVEKIT_PARTICIPANT_IDENTITY': os.environ.get('LIVEKIT_PARTICIPANT_IDENTITY')
        }
        logger.info(f"Environment variables: {env_vars}")
        
        # Call the main entrypoint directly without asyncio.run() since we're already in an async context
        await entrypoint(ctx, room_name, participant_identity)
    except Exception as e:
        logger.error(f"Error in entrypoint_cli: {e}")
        logger.error(traceback.format_exc())
        raise

# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Use the CLI-compatible entrypoint function
    logger.info("Starting Lucidia Voice Agent")
    logger.info(f"Command line arguments: {sys.argv}")
    
    # Parse command line arguments to check what's being passed
    import argparse
    parser = argparse.ArgumentParser(description='Lucidia Voice Agent')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # These arguments will be parsed by the LiveKit CLI, but we parse them here to log them
    parser.add_argument('--room', help='Room name to connect to')
    parser.add_argument('--participant-identity', help='Participant identity')
    
    try:
        args, unknown = parser.parse_known_args()
        logger.info(f"Parsed arguments: {args}")
        logger.info(f"Unknown arguments: {unknown}")
        
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
    
    # Run the agent with LiveKit CLI
    cli.run_app(
        WorkerOptions(
            agent_name="lucidia",
            entrypoint_fnc=entrypoint_cli,
            worker_type=WorkerType.ROOM,
        )
    )
```

# agents\worker_factory.py

```py
"""Factory for creating and managing LiveKit workers and agents with proper token permissions."""

import os
import time
import logging
import asyncio
import jwt
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from functools import partial

from livekit.agents import (
    JobContext, 
    WorkerOptions, 
    JobExecutorType,
    JobRequest
)

from voice_core.config.config import LucidiaConfig, LLMConfig
from voice_core.agents.livekit_voice_agent import LiveKitVoiceAgent
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.llm.llm_pipeline import LocalLLMPipeline
from voice_core.utils.pipeline_logger import PipelineLogger
from voice_core.state.voice_state_manager import VoiceStateManager

logger = logging.getLogger(__name__)

@dataclass
class WorkerConfig:
    """Configuration for worker initialization."""
    ws_url: str
    api_key: str
    api_secret: str
    executor_type: JobExecutorType = JobExecutorType.PROCESS
    dev_mode: bool = False
    initial_greeting: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        """Create config from environment variables."""
        return cls(
            ws_url=os.getenv('LIVEKIT_URL', 'ws://localhost:7880'),
            api_key=os.getenv('LIVEKIT_API_KEY', ''),
            api_secret=os.getenv('LIVEKIT_API_SECRET', ''),
            dev_mode=os.getenv('DEV_MODE', '').lower() == 'true',
            initial_greeting=os.getenv('INITIAL_GREETING')
        )

class VoiceWorkerFactory:
    """Factory for creating LiveKit workers with voice agents."""
    
    def __init__(self, worker_config: WorkerConfig):
        """Initialize the worker factory."""
        self.worker_config = worker_config
        self.pipeline_logger = PipelineLogger("worker_factory")
        self._token_cache = {}
        self._refresh_tasks = {}
        
    def _generate_token(self, room_name: str, identity: str) -> str:
        """Generate a new LiveKit token with proper permissions for UI updates."""
        try:
            # Set token expiration to 12 hours from now for testing reliability
            # In production, use a more reasonable expiration time (e.g., 1 hour)
            exp_time = int(time.time()) + (12 * 60 * 60)
            
            # Token claims
            # CRITICAL: Include canPublishData: true for UI updates to work
            claims = {
                "iss": self.worker_config.api_key,  # Issuer
                "sub": identity,  # Subject (participant identity)
                "exp": exp_time,  # Expiration time
                "nbf": int(time.time()) - 300,  # Not before time (allow 5 min clock skew)
                "video": {
                    "room": room_name,  # Room name
                    "roomJoin": True,  # Allow joining room
                    "canPublish": True,  # Allow publishing audio/video
                    "canSubscribe": True,  # Allow subscribing
                    "canPublishData": True,  # CRITICAL for UI updates
                    "roomAdmin": True,  # Useful for troubleshooting
                    "roomCreate": True   # Allow creating room if it doesn't exist
                },
                "name": identity,  # Participant name
                "metadata": json.dumps({  # Useful metadata for debugging
                    "type": "voice_assistant",
                    "created": time.time(),
                    "version": "1.0"
                })
            }
            
            # Generate token
            token = jwt.encode(
                claims,
                self.worker_config.api_secret,
                algorithm="HS256"
            )
            
            # Cache token with expiration
            self._token_cache[f"{room_name}:{identity}"] = {
                "token": token,
                "expires": exp_time
            }
            
            # Schedule token refresh
            self._schedule_token_refresh(room_name, identity, exp_time)
            
            # Log token details (only in debug mode, redacting sensitive parts)
            if self.worker_config.dev_mode:
                redacted_token = token[:20] + "..." + token[-20:] if token else None
                logger.debug(f"Generated token: {redacted_token} for room: {room_name}, identity: {identity}")
                logger.debug(f"Token permissions: canPublish: True, canSubscribe: True, canPublishData: True")
                
            return token
            
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "token_generation",
                "room": room_name,
                "identity": identity
            })
            raise
            
    async def _refresh_token_task(self, room_name: str, identity: str):
        """Background task to refresh token before expiration."""
        try:
            while True:
                cache_key = f"{room_name}:{identity}"
                cached = self._token_cache.get(cache_key)
                
                if not cached:
                    # Token no longer in cache, stop refresh task
                    break
                    
                # Get time until expiration
                time_to_exp = cached["expires"] - time.time()
                
                # Refresh when token is within 30 minutes of expiring
                if time_to_exp <= 1800:  # 30 minutes before expiration
                    logger.info(f"Refreshing token for {identity} in room {room_name}")
                    # Generate new token
                    self._generate_token(room_name, identity)
                    
                # Check every 15 minutes
                await asyncio.sleep(900)
                
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "token_refresh",
                "room": room_name,
                "identity": identity
            })
        finally:
            # Remove refresh task
            task_key = f"{room_name}:{identity}"
            if task_key in self._refresh_tasks:
                del self._refresh_tasks[task_key]
                
    def _schedule_token_refresh(self, room_name: str, identity: str, exp_time: int):
        """Schedule a token refresh task."""
        task_key = f"{room_name}:{identity}"
        
        # Cancel existing refresh task if any
        existing_task = self._refresh_tasks.get(task_key)
        if existing_task:
            existing_task.cancel()
            
        # Create new refresh task
        refresh_task = asyncio.create_task(
            self._refresh_token_task(room_name, identity)
        )
        self._refresh_tasks[task_key] = refresh_task
            
    def _get_valid_token(self, room_name: str, identity: str) -> str:
        """Get a valid token, generating new one if needed."""
        cache_key = f"{room_name}:{identity}"
        cached = self._token_cache.get(cache_key)
        
        # Check if we have a valid cached token
        if cached:
            # Add 30 minute buffer before expiration
            if cached["expires"] > (time.time() + 1800):
                return cached["token"]
                
        # Generate new token
        return self._generate_token(room_name, identity)
            
    def _create_agent_services(self, job_context: JobContext) -> Dict[str, Any]:
        """Create agent services (STT, TTS, LLM) for a job."""
        try:
            # Create configs
            config = LucidiaConfig()
            llm_config = LLMConfig()
            
            # Create state manager first (centralized state handling)
            state_manager = VoiceStateManager()
            
            # Initialize services with state manager
            stt_service = EnhancedSTTService(
                state_manager=state_manager,
                vosk_model='small',
                whisper_model='small.en',
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            tts_service = InterruptibleTTSService(
                state_manager=state_manager,
                voice=config.tts.get('voice', 'en-US-AvaMultilingualNeural')
            )
            
            llm_service = LocalLLMPipeline(llm_config)
            
            return {
                "config": config,
                "state_manager": state_manager,
                "stt_service": stt_service,
                "tts_service": tts_service,
                "llm_service": llm_service
            }
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "service_initialization",
                "job_id": job_context.job_id
            })
            raise
            
    def _job_request_handler(self, request: JobRequest) -> bool:
        """Handle incoming job requests."""
        try:
            # Log request
            self.pipeline_logger._log(
                logging.INFO,
                "JobRequest",
                f"Received job request: {request.job_id}",
                job_id=request.job_id,
                room_name=request.room_name
            )
            
            # Validate request
            if not request.room_name:
                logger.warning("Rejecting job: missing room name")
                return False
                
            # Generate fresh token for the job
            try:
                identity = f"agent-{request.job_id}"
                self._get_valid_token(request.room_name, identity)
            except Exception as e:
                logger.error(f"Failed to generate token: {e}")
                return False
                
            # Accept request
            return True
            
        except Exception as e:
            logger.error(f"Error handling job request: {e}")
            return False
            
    async def _agent_entrypoint(self, job_context: JobContext) -> None:
        """Entrypoint function for agent jobs."""
        try:
            # Get fresh token
            identity = f"agent-{job_context.job_id}"
            token = self._get_valid_token(job_context.room_name, identity)
            
            # Update job context with fresh token
            job_context.token = token
            
            # Create services
            services = self._create_agent_services(job_context)
            
            # Set room in state manager first
            await job_context.connect(auto_subscribe=True)
            await services["state_manager"].set_room(job_context.room)
            
            # Create and start agent
            agent = LiveKitVoiceAgent(
                job_context=job_context,
                config=services["config"],
                state_manager=services["state_manager"],
                stt_service=services["stt_service"],
                tts_service=services["tts_service"],
                llm_service=services["llm_service"],
                initial_greeting=self.worker_config.initial_greeting
            )
            
            try:
                # Initialize agent (connects services to room)
                await agent.initialize()
                
                # Start agent with greeting
                await agent.start()
                
                # Keep agent running until disconnected
                while job_context.room and job_context.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    await asyncio.sleep(5)
                    
            finally:
                # Cleanup token refresh task
                task_key = f"{job_context.room_name}:{identity}"
                refresh_task = self._refresh_tasks.get(task_key)
                if refresh_task:
                    refresh_task.cancel()
                    del self._refresh_tasks[task_key]
                    
                # Cleanup agent
                await agent.cleanup()
            
        except Exception as e:
            logger.error(f"Error in agent entrypoint: {e}")
            await job_context.fail(str(e))
            raise
            
    def create_worker_options(self) -> WorkerOptions:
        """Create worker options for running agents."""
        return WorkerOptions(
            ws_url=self.worker_config.ws_url,
            api_key=self.worker_config.api_key,
            api_secret=self.worker_config.api_secret,
            executor_type=self.worker_config.executor_type,
            entrypoint_fnc=self._agent_entrypoint,
            request_fnc=self._job_request_handler,
            dev_mode=self.worker_config.dev_mode
        )
        
    async def prewarm_services(self) -> None:
        """Prewarm services for faster startup."""
        try:
            logger.info("Prewarming services")
            
            # Create dummy context for initialization
            dummy_context = JobContext(
                job_id="prewarm",
                room_name="prewarm",
                url=self.worker_config.ws_url,
                token="",
                identity="prewarm"
            )
            
            # Initialize services
            services = self._create_agent_services(dummy_context)
            
            # Prewarm STT
            await services["stt_service"].initialize()
            
            # Prewarm TTS
            await services["tts_service"].initialize()
            
            # Prewarm LLM
            await services["llm_service"].initialize()
            
            logger.info("Services prewarmed successfully")
            
        except Exception as e:
            logger.error(f"Error prewarming services: {e}")
            raise
            
def create_worker(config: Optional[WorkerConfig] = None) -> WorkerOptions:
    """Create a worker with the given config or from environment."""
    if config is None:
        config = WorkerConfig.from_env()
        
    factory = VoiceWorkerFactory(config)
    
    # Prewarm services if not in dev mode
    if not config.dev_mode:
        asyncio.create_task(factory.prewarm_services())
        
    return factory.create_worker_options()
```

# audio\__init__.py

```py
"""Audio processing utilities."""

from .audio_utils import (
    AudioFrame,
    AudioBuffer as AudioStream,
    normalize_audio,
    resample_audio,
    split_audio_chunks,
    convert_audio_format
)

__all__ = [
    'AudioFrame',
    'AudioStream',
    'normalize_audio',
    'resample_audio',
    'split_audio_chunks',
    'convert_audio_format'
]
```

# audio\audio_frame.py

```py
import numpy as np
import time

class AudioFrame:
    """Represents a frame of audio data with associated metadata."""
    
    def __init__(self, data: np.ndarray, sample_rate: int, num_channels: int, samples_per_channel: int, timestamp: float = None):
        """Initialize an audio frame.
        
        Args:
            data: Audio samples as a numpy array
            sample_rate: Sample rate in Hz
            num_channels: Number of audio channels
            samples_per_channel: Number of samples per channel
            timestamp: Optional timestamp in seconds. If None, current time is used.
        """
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
        self._energy = None
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    @property
    def energy(self) -> float:
        """Calculate and cache the energy of the frame."""
        if self._energy is None:
            self._energy = float(np.mean(np.abs(self.data)))
        return self._energy
    
    @energy.setter
    def energy(self, value: float):
        """Set the energy value directly."""
        self._energy = float(value)
    
    @property
    def duration(self) -> float:
        """Get the duration of the frame in seconds."""
        return self.samples_per_channel / self.sample_rate
    
    def resample(self, new_sample_rate: int) -> 'AudioFrame':
        """Create a new frame with resampled data.
        
        Args:
            new_sample_rate: Target sample rate in Hz
            
        Returns:
            A new AudioFrame with resampled data
        """
        if new_sample_rate == self.sample_rate:
            return self
        
        from scipy import signal
        resampled_length = int(len(self.data) * new_sample_rate / self.sample_rate)
        resampled_data = signal.resample(self.data, resampled_length)
        
        return AudioFrame(
            data=resampled_data,
            sample_rate=new_sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=len(resampled_data) // self.num_channels
        )
    
    def to_mono(self) -> 'AudioFrame':
        """Convert the frame to mono by averaging channels if necessary."""
        if self.num_channels == 1:
            return self
            
        mono_data = np.mean(
            self.data.reshape(-1, self.num_channels),
            axis=1
        )
        
        return AudioFrame(
            data=mono_data,
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_data)
        )

```

# audio\audio_pipeline.py

```py
"""
Enhanced audio pipeline for voice processing with LiveKit integration.
Handles sample rate conversion, normalization, and frame management.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import livekit.rtc as rtc
from scipy import signal
from collections import deque

logger = logging.getLogger(__name__)

class AudioPipeline:
    """
    Audio processing pipeline with sample rate conversion and normalization.
    """
    
    def __init__(self, 
                 input_sample_rate: int = 16000,
                 output_sample_rate: int = 48000,
                 normalize_audio: bool = True,
                 frame_size: int = 960):  # Match LiveKit's preferred frame size
        """
        Initialize audio pipeline.
        
        Args:
            input_sample_rate: Expected input sample rate (Hz)
            output_sample_rate: Target output sample rate (Hz)
            normalize_audio: Whether to normalize audio to [-1, 1]
            frame_size: Size of audio frames to process
        """
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.normalize_audio = normalize_audio
        self.frame_size = frame_size
        
        # Use deque for efficient buffer management
        self._buffer = deque(maxlen=frame_size * 4)  # 4x frame size for safety
        
        # Resampling state
        self._resampler = signal.resample_poly
        
        # Performance tracking
        self._processed_frames = 0
        self._total_samples = 0
        self._last_process_time = 0
        
        logger.info(f"Initialized AudioPipeline: {input_sample_rate}Hz → {output_sample_rate}Hz")
        
    def process_frame(self, frame_data: bytes, sample_rate: int) -> Optional[np.ndarray]:
        """
        Process a single audio frame.
        
        Args:
            frame_data: Raw audio frame data
            sample_rate: Sample rate of input data
            
        Returns:
            Processed audio data as float32 numpy array, or None if not enough data
        """
        try:
            # Convert bytes to numpy array
            data = np.frombuffer(frame_data, dtype=np.int16)
            
            # Convert to float32 and normalize if needed
            if self.normalize_audio:
                data = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Resample if needed using high-quality polyphase resampling
            if sample_rate != self.output_sample_rate:
                up = self.output_sample_rate
                down = sample_rate
                data = self._resampler(data, up, down)
            
            # Add to buffer efficiently
            self._buffer.extend(data)
            
            # Extract frames if we have enough data
            if len(self._buffer) >= self.frame_size:
                frames = []
                while len(self._buffer) >= self.frame_size:
                    # Convert deque slice to numpy array
                    frame = np.array([self._buffer.popleft() for _ in range(self.frame_size)])
                    frames.append(frame)
                    
                self._processed_frames += len(frames)
                self._total_samples += len(frames) * self.frame_size
                
                return np.concatenate(frames)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}", exc_info=True)
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "processed_frames": self._processed_frames,
            "total_samples": self._total_samples,
            "buffer_samples": len(self._buffer),
            "input_rate": self.input_sample_rate,
            "output_rate": self.output_sample_rate
        }
        
    def reset(self) -> None:
        """Reset internal buffer and counters."""
        self._buffer.clear()
        self._processed_frames = 0
        self._total_samples = 0
        logger.info("Audio pipeline reset")

```

# audio\audio_playback.py

```py
import pygame
import threading
import queue
from voice_core.shared_state import should_interrupt

# Initialize pygame
print("Initializing pygame")
pygame.mixer.init()
print("Pygame initialized")


def play_audio(audio_data):
    """Play audio_data using pygame and allow interruption."""
    print("Playing audio")
    try:
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not should_interrupt.is_set():
            pygame.time.Clock().tick(10)
        if should_interrupt.is_set():
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print(f"Error stopping audio playback: {e}")
            print("Audio playback interrupted")
            should_interrupt.clear()
        else:
            print("Audio playback complete")
    except Exception as e:
        print(f"Error playing audio: {e}")


def playback_worker(playback_queue):
    """Thread worker for playing back audio."""
    audio_buffer = []
    buffer_size = 3  # Adjust how many audio chunks to buffer

    while True:
        if should_interrupt.is_set():
            # Clear current playback, flush buffer
            try:
                pygame.mixer.music.stop()
            except Exception as e:
                print(f"Error stopping audio playback: {e}")
            audio_buffer.clear()
            should_interrupt.clear()
            continue

        # Attempt to fetch next audio_data
        if len(audio_buffer) < buffer_size:
            try:
                audio_data = playback_queue.get(timeout=0.05)
                if audio_data is None:
                    # End signal
                    if audio_buffer:
                        # Play remaining buffer
                        pass
                    else:
                        return
                else:
                    audio_buffer.append(audio_data)
                playback_queue.task_done()
            except queue.Empty:
                # Nothing else to enqueue, continue to process buffer
                pass

        # Process the buffer
        if audio_buffer and not should_interrupt.is_set():
            current_audio = audio_buffer.pop(0)
            play_audio(current_audio)


def cleanup_audio():
    """Clean up pygame audio resources."""
    try:
        pygame.mixer.music.stop()  # Stop any playing music
    except:
        pass  # Ignore errors if no music is playing
    try:
        pygame.mixer.quit()  # Clean up mixer
    except:
        pass  # Ignore cleanup errors

```

# audio\audio_utils.py

```py
"""Audio utilities for voice pipeline."""

from __future__ import annotations

import io
import logging
import numpy as np
from typing import Union, Optional
from scipy import signal
import soundfile as sf
from dataclasses import dataclass
from livekit import rtc

logger = logging.getLogger(__name__)

@dataclass
class AudioFrame:
    """Audio frame container with metadata."""
    data: np.ndarray
    sample_rate: int
    num_channels: int
    samples_per_channel: int

    def to_bytes(self) -> bytes:
        """Convert frame data to bytes."""
        return self.data.tobytes()

    def to_pcm(self) -> np.ndarray:
        """Ensure data is in PCM format."""
        if self.data.dtype != np.float32:
            return self.data.astype(np.float32)
        return self.data

    def to_rtc(self) -> rtc.AudioFrame:
        """Convert to LiveKit audio frame."""
        # Ensure data is in PCM format
        pcm_data = self.to_pcm()
        
        # Resample to 48kHz if needed (LiveKit default)
        if self.sample_rate != 48000:
            pcm_data = resample_audio(pcm_data, self.sample_rate, 48000)
            
        # Ensure contiguous memory layout
        if not pcm_data.flags['C_CONTIGUOUS']:
            pcm_data = np.ascontiguousarray(pcm_data)
            
        return rtc.AudioFrame(
            data=pcm_data.tobytes(),
            samples_per_channel=len(pcm_data),
            sample_rate=48000,  # LiveKit default
            num_channels=self.num_channels
        )

def normalize_audio(data: np.ndarray, target_range: float = 1.0) -> np.ndarray:
    """Normalize audio data to target range [-target_range, target_range]."""
    if data.size == 0:
        return data
        
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Handle int16 conversion
    if np.abs(data).max() > 1.0:
        data = data / 32768.0
        
    # Normalize to target range
    max_val = np.abs(data).max()
    if max_val > 0:
        data = data * (target_range / max_val)
        
    return data

def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int, 
                  preserve_shape: bool = True) -> np.ndarray:
    """Resample audio data to target sample rate."""
    if src_rate == dst_rate:
        return data
        
    # Ensure data is float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    # Calculate new length
    new_length = int(len(data) * dst_rate / src_rate)
    
    # Resample using scipy
    resampled = signal.resample(data, new_length)
    
    # Preserve original shape if needed
    if preserve_shape and len(resampled) != new_length:
        if len(resampled) < new_length:
            resampled = np.pad(resampled, (0, new_length - len(resampled)))
        else:
            resampled = resampled[:new_length]
            
    return resampled

def split_audio_chunks(data: np.ndarray, chunk_size: int, 
                      overlap: int = 0) -> np.ndarray:
    """Split audio data into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
        
    # Calculate step size
    step = chunk_size - overlap
    
    # Calculate number of chunks
    num_chunks = (len(data) - overlap) // step
    
    # Create output array
    chunks = np.zeros((num_chunks, chunk_size), dtype=data.dtype)
    
    # Fill chunks
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        if end <= len(data):
            chunks[i] = data[start:end]
        else:
            # Pad last chunk if needed
            remaining = len(data) - start
            chunks[i, :remaining] = data[start:]
            
    return chunks

def convert_audio_format(data: Union[bytes, np.ndarray], 
                        src_format: str,
                        dst_format: str,
                        sample_rate: Optional[int] = None) -> bytes:
    """Convert audio between different formats."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
        
    # Create in-memory buffer
    with io.BytesIO(data) as buf:
        # Read audio data
        audio_data, sr = sf.read(buf, format=src_format)
        
        # Resample if needed
        if sample_rate and sr != sample_rate:
            audio_data = resample_audio(audio_data, sr, sample_rate)
            sr = sample_rate
            
        # Write to output buffer
        out_buf = io.BytesIO()
        sf.write(out_buf, audio_data, sr, format=dst_format)
        return out_buf.getvalue()

class EdgeAudioFrame:
    """Wrapper for Edge TTS audio data that provides LiveKit frame interface."""
    def __init__(self, pcm_data: bytes, sample_rate: int = 48000, num_channels: int = 1):
        # Convert bytes to int16 numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Create AudioFrame
        self._frame = AudioFrame(
            data=audio_array,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(audio_array)
        )
    
    @property
    def frame(self) -> rtc.AudioFrame:
        """Get LiveKit audio frame."""
        return self._frame.to_rtc()

class AudioBuffer:
    """Buffer for collecting audio frames for VAD and STT."""
    def __init__(self, max_size: int = 48000 * 5):  # 5 seconds at 48kHz
        self.buffer = io.BytesIO()
        self.max_size = max_size
        self.last_speech = False
        self.speech_start = None
        self.silence_duration = 0
        self.is_speaking = False

    def add_frame(self, frame_data: bytes) -> None:
        """Add a frame to the buffer, maintaining max size."""
        current_size = self.buffer.tell()
        if current_size + len(frame_data) > self.max_size:
            # Keep the last 2 seconds of audio
            keep_size = 48000 * 2
            self.buffer.seek(max(0, current_size - keep_size))
            remaining_data = self.buffer.read()
            self.buffer = io.BytesIO()
            self.buffer.write(remaining_data)
        self.buffer.write(frame_data)

    def get_data(self) -> bytes:
        """Get all buffered audio data."""
        current_pos = self.buffer.tell()
        self.buffer.seek(0)
        data = self.buffer.read()
        self.buffer.seek(current_pos)
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = io.BytesIO()
        self.last_speech = False
        self.speech_start = None
        self.silence_duration = 0
        self.is_speaking = False

```

# audio\mic_utils.py

```py
import logging
import pyaudio
import numpy as np
import time
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_microphone(device_index: Optional[int] = None, duration: float = 3.0) -> bool:
    """
    Test if the selected microphone is working by measuring audio levels.
    
    Args:
        device_index: Index of the device to test, None for default
        duration: Duration in seconds to test the microphone
    
    Returns:
        bool: True if microphone is working and detecting audio
    """
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 0.01  # Adjust this for sensitivity

    p = pyaudio.PyAudio()
    
    try:
        logger.info("Testing microphone...")
        logger.info("Please speak into your microphone for %d seconds...", int(duration))
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_index,
                       frames_per_buffer=CHUNK)
        
        max_amplitude = 0.0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
                amplitude = np.abs(data).mean()
                max_amplitude = max(max_amplitude, amplitude)
                
                # Only log significant changes in amplitude
                if amplitude > 0.01:
                    bar_length = int(amplitude * 50)
                    bar = '█' * min(bar_length, 50)
                    logger.debug(f"Level: {bar:<50} [{amplitude:.4f}]")
                
            except Exception as e:
                logger.error("Error reading from microphone: %s", e)
                return False
        
        stream.stop_stream()
        stream.close()
        
        if max_amplitude > THRESHOLD:
            logger.info("✓ Microphone test successful! Max amplitude: %.4f", max_amplitude)
            return True
        else:
            logger.warning("✗ No significant audio detected. Please check your microphone.")
            return False
            
    except Exception as e:
        logger.error("Failed to test microphone: %s", e)
        return False
    finally:
        p.terminate()

def get_device_details(p: pyaudio.PyAudio, index: int) -> Dict[str, Any]:
    """Get detailed information about an audio device."""
    try:
        info = p.get_device_info_by_index(index)
        return {
            "index": index,
            "name": info["name"],
            "channels": info["maxInputChannels"],
            "sample_rate": int(info["defaultSampleRate"]),
            "latency": info["defaultLowInputLatency"] * 1000,
            "is_default": p.get_default_input_device_info()["index"] == index
        }
    except Exception as e:
        logger.error(f"Error getting device info for index {index}: {e}")
        return {}

def list_microphones() -> List[Dict[str, Any]]:
    """
    List all available microphones and return their details.
    
    Returns:
        List[Dict[str, Any]]: List of microphone details
    """
    p = pyaudio.PyAudio()
    devices = []
    
    try:
        logger.info("\n=== Available Audio Input Devices ===")
        
        # Get default device first
        try:
            default_info = p.get_default_input_device_info()
            logger.info("\n▶ Default Input Device:")
            logger.info("  Index: %d", default_info["index"])
            logger.info("  Name: %s", default_info["name"])
            logger.info("  Channels: %d", default_info["maxInputChannels"])
            logger.info("  Sample Rate: %d Hz", int(default_info["defaultSampleRate"]))
            logger.info("  Latency: %.2f ms", default_info["defaultLowInputLatency"] * 1000)
        except Exception as e:
            logger.warning("Could not get default device info: %s", e)
        
        logger.info("\n=== Other Input Devices ===")
        for i in range(p.get_device_count()):
            device_info = get_device_details(p, i)
            if device_info and device_info["channels"] > 0:  # Only input devices
                devices.append(device_info)
                logger.info("\nDevice %d%s:", i, " (Default)" if device_info["is_default"] else "")
                logger.info("  Name: %s", device_info["name"])
                logger.info("  Channels: %d", device_info["channels"])
                logger.info("  Sample Rate: %d Hz", device_info["sample_rate"])
                logger.info("  Input Latency: %.2f ms", device_info["latency"])
    except Exception as e:
        logger.error("Error listing audio devices: %s", e)
    finally:
        p.terminate()
    
    return devices

def select_microphone(auto_select_default: bool = True) -> Optional[int]:
    """
    Select a microphone device interactively.
    Returns the device index of the selected microphone.
    """
    p = pyaudio.PyAudio()
    
    try:
        # Get input devices
        devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                devices.append((i, device_info))
        
        if not devices:
            logger.error("No input devices found")
            return None
            
        # Print device information
        for i, (idx, info) in enumerate(devices):
            logger.info(f"\nDevice {idx}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Channels: {info['maxInputChannels']}")
            logger.info(f"  Sample Rate: {int(info['defaultSampleRate'])} Hz")
            logger.info(f"  Input Latency: {info['defaultLowInputLatency']*1000:.2f} ms")
        
        # Always use device 1 for this system
        if test_microphone(1):
            logger.info("Using device 1")
            return 1
            
        # Fallback to default device if device 1 fails
        if auto_select_default:
            default_idx = p.get_default_input_device_info()['index']
            logger.info(f"Device 1 failed, using default device: {default_idx}")
            if test_microphone(default_idx):
                return default_idx
            
        # Manual selection as last resort
        while True:
            try:
                selection = input("\nSelect input device by number (or press Enter for default): ").strip()
                if not selection:
                    default_idx = p.get_default_input_device_info()['index']
                    logger.info(f"Using default device: {default_idx}")
                    if test_microphone(default_idx):
                        return default_idx
                    continue
                    
                device_idx = int(selection)
                if device_idx < 0 or device_idx >= len(devices):
                    logger.error("Invalid device number")
                    continue
                    
                if test_microphone(device_idx):
                    return device_idx
                    
            except ValueError:
                logger.error("Please enter a valid number")
            except KeyboardInterrupt:
                logger.info("\nMicrophone selection cancelled")
                return None
                
    finally:
        p.terminate()

if __name__ == "__main__":
    selected_index = select_microphone()
    if selected_index is None:
        logger.info("Using default microphone")
    else:
        logger.info("Selected microphone index: %d", selected_index)

```

# cache\tiny.en.pt

This is a binary file of the type: Binary

# clock_sync.py

```py
import asyncio
import logging
import ntplib
import time
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ClockDrift:
    start_time: float
    end_time: float
    drift_ms: float
    samples: int
    rtp_clock_rate: float

@dataclass
class RTPStats:
    """Statistics for RTP stream synchronization."""
    ssrc: int
    last_rtp_ts: int = 0
    last_ntp_ts: float = 0
    last_sr_ntp: Optional[float] = None
    last_sr_rtp: Optional[int] = None
    drift_samples: List[Dict] = field(default_factory=list)
    rtt_samples: List[float] = field(default_factory=list)
    jitter: float = 0.0
    packets_received: int = 0
    packets_lost: int = 0

class ClockSync:
    """Enhanced clock synchronization with RTP drift compensation."""
    
    def __init__(self, ntp_servers=None, sync_interval=30, sample_rate: int = 48000):
        self.ntp_client = ntplib.NTPClient()
        self.ntp_servers = ntp_servers or ['pool.ntp.org', 'time.google.com', 'time.windows.com']
        self.clock_offset = 0.0
        self.rtp_stats: Dict[int, RTPStats] = {}
        self.last_sync = 0
        self.sync_interval = sync_interval
        self.max_history = 20
        self.logger = logging.getLogger(__name__)
        self._consecutive_failures = 0
        self.offset_history = []
        self.sample_rate = sample_rate
        self.last_system_sync = 0
        self.system_offset = 0.0
        self.max_drift_samples = 50
        self.drift_correction_threshold = 10
        
    async def check_system_time(self) -> bool:
        """Check if system time is properly synchronized."""
        try:
            # Get Windows Time service status
            process = await asyncio.create_subprocess_shell(
                'w32tm /query /status',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"❌ Failed to query time status: {stderr.decode()}")
                return False
                
            output = stdout.decode()
            
            # Check if time is synced
            if "Last Successful Sync Time" in output:
                sync_time_str = output.split("Last Successful Sync Time: ")[1].split("\n")[0].strip()
                try:
                    sync_time = datetime.strptime(sync_time_str, "%d/%m/%Y %H:%M:%S")
                    time_diff = abs((datetime.now() - sync_time).total_seconds())
                    
                    if time_diff > 3600:  # More than 1 hour since last sync
                        logger.warning(f"⚠️ System time sync is old: {time_diff:.1f} seconds ago")
                        return False
                        
                    logger.info(f"✅ System time in sync (last sync: {time_diff:.1f} seconds ago)")
                    return True
                except ValueError:
                    logger.error(f"❌ Failed to parse sync time: {sync_time_str}")
                    return False
            else:
                logger.error("❌ Could not find last sync time in status output")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking system time: {e}")
            return False
            
    async def sync(self):
        if time.time() - self.last_sync < self.sync_interval:
            return

        failed_servers = []
        for server in self.ntp_servers:
            try:
                response = await asyncio.to_thread(self.ntp_client.request, server, timeout=2)
                self.offset_history.append({'timestamp': time.time(), 'offset': response.offset, 'delay': response.delay, 'server': server})

                if len(self.offset_history) > self.max_history:
                    self.offset_history.pop(0)

                self.last_sync = time.time()
                self.logger.debug(f"Clock sync: offset={response.offset:.3f}ms, delay={response.delay:.3f}ms from {server}")
                self._consecutive_failures = 0
                break
            except Exception as e:
                failed_servers.append(f"{server}: {e}")

        if len(failed_servers) == len(self.ntp_servers):
            self.logger.error(f"All NTP sync attempts failed: {failed_servers}")
            self._consecutive_failures += 1
            
        if await self.check_system_time():
            self.last_system_sync = time.time()
            logger.info("✅ Clock sync verified")
        else:
            logger.warning("⚠️ System time may be out of sync")
            
    async def force_sync(self):
        self.logger.info("Forcing manual clock sync...")
        await self.sync()

    async def sync_loop(self):
        while True:
            try:
                jitter = self.sync_interval * 0.1 * (2 * np.random.random() - 1)
                await asyncio.sleep(self.sync_interval + jitter)
                await self.sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in clock sync loop: {e}")

    @property
    def offset(self):
        return self.get_current_offset()

    @property
    def is_synced(self):
        return (time.time() - self.last_sync) <= self.sync_interval

    async def sync_clock(self) -> float:
        """Synchronize with NTP servers and return clock offset."""
        now = time.time()
        if now - self.last_sync < self.sync_interval:
            return self.clock_offset
            
        offsets = []
        for server in self.ntp_servers:
            try:
                response = self.ntp_client.request(server, timeout=2)
                offset = response.offset
                offsets.append(offset)
                logger.debug(f"🕒 NTP offset from {server}: {offset:.6f}s")
            except Exception as e:
                logger.warning(f"⚠️ Failed to sync with NTP server {server}: {e}")
                continue
                
        if offsets:
            # Use median to filter out outliers
            self.clock_offset = statistics.median(offsets)
            self.last_sync = now
            logger.info(f"✅ Clock synchronized, offset: {self.clock_offset:.6f}s")
        else:
            logger.error("❌ Failed to sync with any NTP servers")
            
        return self.clock_offset
        
    def get_corrected_time(self) -> float:
        """Get current time corrected by NTP offset."""
        return time.time() + self.clock_offset
        
    def update_rtp_stats(self, ssrc: int, rtp_ts: int, ntp_ts: float, sr_ntp: Optional[float] = None, sr_rtp: Optional[int] = None, rtt: Optional[float] = None):
        """Update RTP statistics with enhanced drift tracking."""
        if ssrc not in self.rtp_stats:
            self.rtp_stats[ssrc] = RTPStats(ssrc=ssrc)
            
        stats = self.rtp_stats[ssrc]
        stats.packets_received += 1
        
        # Update sender report info
        if sr_ntp is not None and sr_rtp is not None:
            stats.last_sr_ntp = sr_ntp
            stats.last_sr_rtp = sr_rtp
            
        # Calculate drift if we have previous samples
        if stats.last_rtp_ts > 0 and stats.last_ntp_ts > 0:
            rtp_diff = rtp_ts - stats.last_rtp_ts
            ntp_diff = (ntp_ts - stats.last_ntp_ts) * self.sample_rate
            
            if rtp_diff > 0 and ntp_diff > 0:
                # Calculate drift in samples
                drift = abs(rtp_diff - ntp_diff)
                # Calculate jitter (RFC 3550)
                stats.jitter = stats.jitter + (drift - stats.jitter) / 16
                
                stats.drift_samples.append({
                    'drift': drift,
                    'timestamp': ntp_ts,
                    'rtp_diff': rtp_diff,
                    'ntp_diff': ntp_diff
                })
                
                # Keep drift samples bounded
                while len(stats.drift_samples) > self.max_drift_samples:
                    stats.drift_samples.pop(0)
                    
        # Update RTT samples if provided
        if rtt is not None:
            stats.rtt_samples.append(rtt)
            while len(stats.rtt_samples) > self.max_rtt_samples:
                stats.rtt_samples.pop(0)
                
        # Update last seen timestamps
        stats.last_rtp_ts = rtp_ts
        stats.last_ntp_ts = ntp_ts
        
    def compensate_rtp_timestamp(self, rtp_timestamp: int, ssrc: int) -> int:
        """Compensate for RTP drift using weighted average of recent samples."""
        if ssrc not in self.rtp_stats:
            return rtp_timestamp
            
        stats = self.rtp_stats[ssrc]
        recent_drifts = stats.drift_samples[-5:]  # Use last 5 samples
        
        if not recent_drifts:
            return rtp_timestamp
            
        # Calculate weighted average (more recent samples have higher weight)
        total_weight = sum(range(1, len(recent_drifts) + 1))
        weighted_drift = sum(
            s['drift'] * (i + 1) for i, s in enumerate(recent_drifts)
        ) / total_weight
        
        # Apply compensation with jitter consideration
        compensation = int(weighted_drift + stats.jitter)
        compensated_ts = rtp_timestamp - compensation
        
        if compensation > 0:
            logger.debug(f"🔄 RTP compensation: {compensation} samples (drift: {weighted_drift:.1f}, jitter: {stats.jitter:.1f})")
            
        return compensated_ts
        
    def get_rtt_stats(self, ssrc: int) -> Optional[Dict[str, float]]:
        """Get RTT statistics for a stream."""
        if ssrc not in self.rtp_stats:
            return None
            
        stats = self.rtp_stats[ssrc]
        if not stats.rtt_samples:
            return None
            
        return {
            'min_rtt': min(stats.rtt_samples),
            'max_rtt': max(stats.rtt_samples),
            'avg_rtt': statistics.mean(stats.rtt_samples),
            'median_rtt': statistics.median(stats.rtt_samples)
        }
        
    def get_stream_stats(self, ssrc: int) -> Optional[Dict]:
        """Get comprehensive statistics for a stream."""
        if ssrc not in self.rtp_stats:
            return None
            
        stats = self.rtp_stats[ssrc]
        return {
            'packets_received': stats.packets_received,
            'packets_lost': stats.packets_lost,
            'jitter': stats.jitter,
            'rtt_stats': self.get_rtt_stats(ssrc),
            'last_drift': stats.drift_samples[-1] if stats.drift_samples else None
        }
        
    def clear_stats(self, ssrc: int):
        """Clear statistics for a stream."""
        if ssrc in self.rtp_stats:
            del self.rtp_stats[ssrc]
            logger.debug(f"🧹 Cleared RTP stats for SSRC {ssrc}")
            
    def clear_all_stats(self):
        """Clear all stream statistics."""
        self.rtp_stats.clear()
        logger.info("🧹 Cleared all RTP stats")


@dataclass
class SenderReport:
    ntp_timestamp: float
    rtp_timestamp: int
    packet_count: int
    octet_count: int
    received_at: float

@dataclass
class StreamStats:
    ssrc: str
    last_seq: int
    highest_seq: int
    cycle_count: int
    packets_received: int
    packets_lost: int
    last_sr_time: float
    last_sr_ntp: float
    last_sr_rtp: int
    jitter: float
    rtt: float

class EnhancedClockSync:
    """Enhanced clock synchronization with improved NTP and RTT handling."""
    
    def __init__(self):
        self._streams: Dict[str, StreamStats] = {}
        self._sender_reports: Dict[str, SenderReport] = {}
        self._ntp_offset: float = 0
        self._last_ntp_sync = 0
        self._sync_interval = 30  # Reduced from 60s to 30s
        self._max_rtt = 0.5      # Reduced from 1.0s to 500ms
        self._min_sr_interval = 0.1  # Reduced from 0.5s to 100ms
        self._last_sr_sent: Dict[str, float] = {}
        self._buffer_target = 50  # Target buffer size in milliseconds
        self._max_buffer = 100    # Maximum buffer size in milliseconds
        self._min_buffer = 20     # Minimum buffer size in milliseconds
        self._adaptive_buffer = True
        
    def _ntp_to_unix(self, ntp_timestamp: int) -> float:
        """Convert NTP timestamp to UNIX timestamp with microsecond precision."""
        ntp_era = (ntp_timestamp >> 32) & 0xFFFFFFFF
        ntp_secs = ntp_timestamp & 0xFFFFFFFF
        
        # NTP epoch starts at 1900, Unix at 1970
        ntp_to_unix_offset = 2208988800
        
        unix_secs = ntp_era - ntp_to_unix_offset
        unix_fraction = (ntp_secs * 1000000) >> 32  # Convert to microseconds
        
        return unix_secs + (unix_fraction / 1000000.0)
        
    def _unix_to_ntp(self, unix_timestamp: float) -> int:
        """Convert UNIX timestamp to NTP timestamp."""
        ntp_to_unix_offset = 2208988800
        
        ntp_secs = int(unix_timestamp) + ntp_to_unix_offset
        ntp_frac = int((unix_timestamp % 1.0) * 0x100000000)
        
        return (ntp_secs << 32) | ntp_frac
        
    def update_ntp_offset(self):
        """Update NTP offset with system time."""
        now = time.time()
        if now - self._last_ntp_sync >= self._sync_interval:
            try:
                # Get current UTC time
                utc_now = datetime.now(timezone.utc).timestamp()
                self._ntp_offset = utc_now - now
                self._last_ntp_sync = now
                logger.debug(f"Updated NTP offset: {self._ntp_offset*1000:.2f}ms")
            except Exception as e:
                logger.error(f"Failed to update NTP offset: {e}")
                
    def process_sender_report(self, ssrc: str, sr_data: Dict) -> Optional[float]:
        """Process RTCP sender report with improved timing accuracy."""
        try:
            now = time.time() + self._ntp_offset
            ntp_ts = sr_data.get('NtpTimestamp', 0)
            rtp_ts = sr_data.get('RtpTimestamp', 0)
            
            if not ntp_ts or not rtp_ts:
                return None
                
            # Convert NTP timestamp string to float
            if isinstance(ntp_ts, str):
                try:
                    ntp_dt = datetime.fromisoformat(ntp_ts.replace('Z', '+00:00'))
                    ntp_ts = ntp_dt.timestamp()
                except ValueError:
                    logger.error(f"Invalid NTP timestamp format: {ntp_ts}")
                    return None
                    
            # Check if we should process this SR
            if ssrc in self._last_sr_sent:
                time_since_last = now - self._last_sr_sent[ssrc]
                if time_since_last < self._min_sr_interval:
                    logger.debug(f"Skipping SR, too soon after last: {time_since_last*1000:.1f}ms")
                    return None
                    
            # Store sender report
            self._sender_reports[ssrc] = SenderReport(
                ntp_timestamp=ntp_ts,
                rtp_timestamp=rtp_ts,
                packet_count=sr_data.get('Packets', 0),
                octet_count=sr_data.get('Octets', 0),
                received_at=now
            )
            
            self._last_sr_sent[ssrc] = now
            
            # Calculate RTT if we have stream stats
            if ssrc in self._streams:
                stats = self._streams[ssrc]
                delay_since_last_sr = now - stats.last_sr_time
                
                if delay_since_last_sr > 0 and delay_since_last_sr < self._max_rtt:
                    rtt = delay_since_last_sr * 2  # Round trip is twice the one-way delay
                    stats.rtt = rtt
                    logger.debug(f"Updated RTT for stream {ssrc}: {rtt*1000:.1f}ms")
                    return rtt
                    
            return None
            
        except Exception as e:
            logger.error(f"Error processing sender report: {e}")
            return None
            
    def update_stream_stats(self, ssrc: str, seq: int, timestamp: int, received_at: float):
        """Update stream statistics with enhanced sequence tracking."""
        if ssrc not in self._streams:
            self._streams[ssrc] = StreamStats(
                ssrc=ssrc,
                last_seq=seq,
                highest_seq=seq,
                cycle_count=0,
                packets_received=1,
                packets_lost=0,
                last_sr_time=0,
                last_sr_ntp=0,
                last_sr_rtp=0,
                jitter=0,
                rtt=0
            )
            return
            
        stats = self._streams[ssrc]
        
        # Handle sequence number wrapping
        if seq < stats.last_seq and stats.last_seq - seq > 0x8000:
            stats.cycle_count += 1
            
        # Update highest sequence
        extended_seq = stats.cycle_count << 16 | seq
        if extended_seq > stats.highest_seq:
            stats.highest_seq = extended_seq
            
        # Calculate packets lost
        expected = stats.highest_seq - stats.last_seq + 1
        if expected > 0:
            stats.packets_lost = expected - stats.packets_received
            
        stats.last_seq = seq
        stats.packets_received += 1
        
        # Update jitter
        if ssrc in self._sender_reports:
            sr = self._sender_reports[ssrc]
            transit_time = (received_at - sr.received_at) - (
                (timestamp - sr.rtp_timestamp) / 48000  # Assuming 48kHz audio
            )
            delta = abs(transit_time - stats.jitter)
            stats.jitter = stats.jitter + (delta - stats.jitter) / 16
            
    def get_stream_stats(self, ssrc: str) -> Optional[Dict]:
        """Get stream statistics with enhanced metrics."""
        if ssrc not in self._streams:
            return None
            
        stats = self._streams[ssrc]
        return {
            'ssrc': stats.ssrc,
            'packets_received': stats.packets_received,
            'packets_lost': stats.packets_lost,
            'loss_rate': stats.packets_lost / max(stats.packets_received, 1),
            'jitter': stats.jitter * 1000,  # Convert to ms
            'rtt': stats.rtt * 1000,  # Convert to ms
            'last_seq': stats.last_seq,
            'highest_seq': stats.highest_seq
        }
        
    def estimate_rtp_timestamp(self, ssrc: str, ntp_time: float) -> Optional[int]:
        """Estimate RTP timestamp for a given NTP time with improved accuracy."""
        if ssrc not in self._sender_reports:
            return None
            
        sr = self._sender_reports[ssrc]
        time_diff = ntp_time - sr.ntp_timestamp
        
        # Convert time difference to RTP timestamp units (assuming 48kHz audio)
        timestamp_diff = int(time_diff * 48000)
        
        # Apply timestamp wraparound handling
        estimated_ts = (sr.rtp_timestamp + timestamp_diff) & 0xFFFFFFFF
        
        return estimated_ts

```

# config\__init__.py

```py
"""Configuration classes for voice agent."""

from .config import LucidiaConfig, WhisperConfig, LLMConfig

__all__ = ['LucidiaConfig', 'WhisperConfig', 'LLMConfig']

```

# config\config.py

```py
# voice_core/config/config.py
"""Configuration classes for the voice assistant."""

import os
import torch
from dotenv import load_dotenv
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class WhisperConfig:
    """Configuration for Whisper STT"""
    model_name: str = field(default_factory=lambda: os.getenv('WHISPER_MODEL', 'base'))
    language: str = field(default_factory=lambda: os.getenv('WHISPER_LANGUAGE', 'en'))
    sample_rate: int = 16000  # Fixed at 16kHz for Whisper
    num_channels: int = 1
    vad_threshold: float = 0.25
    chunk_duration_ms: int = 1000
    silence_duration_ms: int = 500
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    min_audio_length: float = 0.5
    max_audio_length: float = 30.0
    initial_silence_threshold: float = -45.0
    silence_duration: float = 0.5
    noise_floor: float = -65.0
    max_buffer_length: int = 30000
    beam_size: int = 5
    best_of: int = 3
    temperature: float = 0.0
    patience: float = 1.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous: bool = True
    initial_prompt: str = None
    fp16: bool = True
    verbose: bool = False
    bandpass_low: float = 100.0
    bandpass_high: float = 4000.0
    min_speech_duration: float = 0.5
    heartbeat_interval: int = 10
    max_init_retries: int = 3
    speech_confidence_threshold: float = 0.3
    speech_confidence_decay: float = 0.1
    speech_confidence_boost: float = 0.3
    max_low_energy_frames: int = 5
    energy_threshold_end: float = 20.0

    def __post_init__(self):
        cuda_available = torch.cuda.is_available()
        if self.device == "cuda" and not cuda_available:
            logger.warning("CUDA is not available, defaulting to CPU")
            self.device = "cpu"
        
        logger.info(f"🎙️ Whisper config: model={self.model_name}, device={self.device}, sr={self.sample_rate}Hz")

@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    model: str = "qwen2.5-7b-instruct-1m"
    api_endpoint: str = "http://localhost:1234/v1"
    temperature: float = 0.7
    max_tokens: int = 150
    system_prompt: str = "You are Lucidia, a helpful voice assistant. Keep your responses concise and natural for spoken conversation."
    timeout: float = 15.0
    stream: bool = True
    
    def __post_init__(self):
        if os.getenv("LLM_MODEL"):
            self.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_ENDPOINT"):
            self.api_endpoint = os.getenv("LLM_API_ENDPOINT")
        if os.getenv("LLM_TEMPERATURE"):
            self.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        if os.getenv("LLM_SYSTEM_PROMPT"):
            self.system_prompt = os.getenv("LLM_SYSTEM_PROMPT")
        
        logger.info(f"🤖 LLM config: model={self.model}, temp={self.temperature}, max_tokens={self.max_tokens}")

@dataclass
class TTSConfig:
    """Configuration for TTS service."""
    voice: str = "en-US-AvaMultilingualNeural"
    sample_rate: int = 24000
    channels: int = 1
    ssml_enabled: bool = True
    cache_enabled: bool = True
    cache_size: int = 50
    
    def __post_init__(self):
        if os.getenv("EDGE_TTS_VOICE"):
            self.voice = os.getenv("EDGE_TTS_VOICE")
        if os.getenv("TTS_SAMPLE_RATE"):
            self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE"))
        if os.getenv("TTS_CHANNELS"):
            self.channels = int(os.getenv("TTS_CHANNELS"))
        if os.getenv("TTS_SSML_ENABLED"):
            self.ssml_enabled = os.getenv("TTS_SSML_ENABLED").lower() in ["true", "1", "yes"]
        if os.getenv("TTS_CACHE_ENABLED"):
            self.cache_enabled = os.getenv("TTS_CACHE_ENABLED").lower() in ["true", "1", "yes"]
        if os.getenv("TTS_CACHE_SIZE"):
            self.cache_size = int(os.getenv("TTS_CACHE_SIZE"))

@dataclass
class StateConfig:
    """Configuration for state management."""
    processing_timeout: float = 15.0
    speaking_timeout: float = 60.0
    debug: bool = False
    
    def __post_init__(self):
        if os.getenv("PROCESSING_TIMEOUT"):
            self.processing_timeout = float(os.getenv("PROCESSING_TIMEOUT"))
        if os.getenv("SPEAKING_TIMEOUT"):
            self.speaking_timeout = float(os.getenv("SPEAKING_TIMEOUT"))
        if os.getenv("STATE_DEBUG"):
            self.debug = os.getenv("STATE_DEBUG").lower() in ["true", "1", "yes"]

@dataclass
class RoomConfig:
    """Configuration for LiveKit room."""
    url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "secret"
    room_name: str = "playground"
    sample_rate: int = 48000  # LiveKit standard
    channels: int = 1
    chunk_size: int = 480  # 10ms at 48kHz
    buffer_size: int = 4800  # 100ms buffer
    
    def __post_init__(self):
        if os.getenv("LIVEKIT_URL"):
            self.url = os.getenv("LIVEKIT_URL")
        if os.getenv("LIVEKIT_API_KEY"):
            self.api_key = os.getenv("LIVEKIT_API_KEY")
        if os.getenv("LIVEKIT_API_SECRET"):
            self.api_secret = os.getenv("LIVEKIT_API_SECRET")
        if os.getenv("ROOM_NAME"):
            self.room_name = os.getenv("ROOM_NAME")

@dataclass
class LucidiaConfig:
    """Main configuration for the voice assistant."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    state: StateConfig = field(default_factory=StateConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    initial_greeting: str = "Hello! I'm Lucidia, your voice assistant. How can I help you today?"
    
    def __post_init__(self):
        if os.getenv("INITIAL_GREETING"):
            self.initial_greeting = os.getenv("INITIAL_GREETING")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "model": self.llm.model,
                "api_endpoint": self.llm.api_endpoint,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "system_prompt": self.llm.system_prompt,
                "timeout": self.llm.timeout,
                "stream": self.llm.stream
            },
            "whisper": {
                "model_name": self.whisper.model_name,
                "device": self.whisper.device,
                "language": self.whisper.language,
                "sample_rate": self.whisper.sample_rate,
                "vad_threshold": self.whisper.vad_threshold,
                "min_speech_duration": self.whisper.min_speech_duration,
                "max_audio_length": self.whisper.max_audio_length,
                "speech_confidence_threshold": self.whisper.speech_confidence_threshold
            },
            "tts": {
                "voice": self.tts.voice,
                "sample_rate": self.tts.sample_rate,
                "channels": self.tts.channels,
                "ssml_enabled": self.tts.ssml_enabled,
                "cache_enabled": self.tts.cache_enabled,
                "cache_size": self.tts.cache_size
            },
            "state": {
                "processing_timeout": self.state.processing_timeout,
                "speaking_timeout": self.state.speaking_timeout,
                "debug": self.state.debug
            },
            "room": {
                "url": self.room.url,
                "api_key": self.room.api_key,
                "api_secret": "[REDACTED]",
                "room_name": self.room.room_name,
                "sample_rate": self.room.sample_rate,
                "channels": self.room.channels,
                "chunk_size": self.room.chunk_size,
                "buffer_size": self.room.buffer_size
            },
            "initial_greeting": self.initial_greeting
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LucidiaConfig':
        """Create configuration from dictionary."""
        llm_config = LLMConfig(
            model=config_dict.get("llm", {}).get("model", "qwen2.5-7b-instruct-1m"),
            api_endpoint=config_dict.get("llm", {}).get("api_endpoint", "http://localhost:1234/v1"),
            temperature=config_dict.get("llm", {}).get("temperature", 0.7),
            max_tokens=config_dict.get("llm", {}).get("max_tokens", 150),
            system_prompt=config_dict.get("llm", {}).get("system_prompt", "You are Lucidia, a helpful voice assistant."),
            timeout=config_dict.get("llm", {}).get("timeout", 15.0),
            stream=config_dict.get("llm", {}).get("stream", True)
        )
        
        whisper_config = WhisperConfig(
            model_name=config_dict.get("whisper", {}).get("model_name", "base"),
            device=config_dict.get("whisper", {}).get("device", "cpu"),
            language=config_dict.get("whisper", {}).get("language", "en"),
            sample_rate=config_dict.get("whisper", {}).get("sample_rate", 16000),
            vad_threshold=config_dict.get("whisper", {}).get("vad_threshold", 0.25),
            min_speech_duration=config_dict.get("whisper", {}).get("min_speech_duration", 0.5),
            max_audio_length=config_dict.get("whisper", {}).get("max_audio_length", 30.0),
            speech_confidence_threshold=config_dict.get("whisper", {}).get("speech_confidence_threshold", 0.3)
        )
        
        tts_config = TTSConfig(
            voice=config_dict.get("tts", {}).get("voice", "en-US-AvaMultilingualNeural"),
            sample_rate=config_dict.get("tts", {}).get("sample_rate", 24000),
            channels=config_dict.get("tts", {}).get("channels", 1),
            ssml_enabled=config_dict.get("tts", {}).get("ssml_enabled", True),
            cache_enabled=config_dict.get("tts", {}).get("cache_enabled", True),
            cache_size=config_dict.get("tts", {}).get("cache_size", 50)
        )
        
        state_config = StateConfig(
            processing_timeout=config_dict.get("state", {}).get("processing_timeout", 15.0),
            speaking_timeout=config_dict.get("state", {}).get("speaking_timeout", 60.0),
            debug=config_dict.get("state", {}).get("debug", False)
        )
        
        room_config = RoomConfig(
            url=config_dict.get("room", {}).get("url", "ws://localhost:7880"),
            api_key=config_dict.get("room", {}).get("api_key", "devkey"),
            api_secret=config_dict.get("room", {}).get("api_secret", "secret"),
            room_name=config_dict.get("room", {}).get("room_name", "lucidia_room"),
            sample_rate=config_dict.get("room", {}).get("sample_rate", 48000),
            channels=config_dict.get("room", {}).get("channels", 1),
            chunk_size=config_dict.get("room", {}).get("chunk_size", 480),
            buffer_size=config_dict.get("room", {}).get("buffer_size", 4800)
        )
        
        return cls(
            llm=llm_config,
            whisper=whisper_config,
            tts=tts_config,
            state=state_config,
            room=room_config,
            initial_greeting=config_dict.get("initial_greeting", "Hello! I'm Lucidia, your voice assistant. How can I help you today?")
        )
```

# connection_utils.py

```py
"""Connection utilities for LiveKit room management."""
import asyncio
import logging
from typing import Optional, Any
from livekit import rtc
from livekit.agents import JobContext
from livekit.rtc import Room

logger = logging.getLogger(__name__)

async def cleanup_connection(assistant: Optional[Any], ctx: JobContext) -> None:
    """Gracefully cleanup the connection and resources with enhanced state management."""
    if not ctx or not hasattr(ctx, 'room'):
        return

    try:
        if assistant:
            await assistant.cleanup()

        if ctx.room:
            # First attempt graceful disconnect
            try:
                async with asyncio.timeout(5.0):
                    await ctx.room.disconnect()
            except asyncio.TimeoutError:
                logger.warning("Room disconnect timed out, forcing cleanup")
            except Exception as e:
                logger.error(f"Error during room disconnect: {e}")

            # Force cleanup of internal state
            await force_room_cleanup(ctx.room)

    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")

async def force_room_cleanup(room: Room) -> None:
    """Force cleanup of room resources."""
    if not room or not room.local_participant:
        return
        
    try:
        # Unpublish all tracks
        if hasattr(room.local_participant, 'published_tracks'):
            for track_pub in room.local_participant.published_tracks:
                try:
                    await room.local_participant.unpublish_track(track_pub)
                except Exception as e:
                    logger.error(f"Error unpublishing track {track_pub.sid}: {e}")
                
        # Close room connection
        await room.disconnect()
    except Exception as e:
        logger.error(f"Error during room cleanup: {e}")

async def wait_for_disconnect(ctx: JobContext, timeout: int = 15) -> bool:
    """
    Wait for room to fully disconnect with extended timeout.
    Returns True if disconnect confirmed, False if timeout.
    """
    try:
        async with asyncio.timeout(timeout):
            while ctx.room and (
                ctx.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED or
                (hasattr(ctx.room, '_ws') and ctx.room._ws and ctx.room._ws.connected)
            ):
                await asyncio.sleep(0.1)
            return True
    except asyncio.TimeoutError:
        return False

async def verify_room_state(ctx: JobContext, check_connection_state: bool = True) -> bool:
    """
    Verify that the room is truly clean and ready for a new connection.
    Returns True if room is clean, False otherwise.
    """
    if not ctx or not hasattr(ctx, 'room'):
        return True

    state = {
        'ws_connected': bool(ctx.room._ws and ctx.room._ws.connected) if hasattr(ctx.room, '_ws') else False,
        'participants': len(ctx.room._participants) if hasattr(ctx.room, '_participants') else 0,
        'connection_state': ctx.room.connection_state if hasattr(ctx.room, 'connection_state') else None
    }

    if not check_connection_state:
        is_clean = (not state['ws_connected'] and state['participants'] == 0)
    else:
        is_clean = (
            not state['ws_connected'] and
            state['participants'] == 0 and
            (state['connection_state'] is None or state['connection_state'] == rtc.ConnectionState.CONN_DISCONNECTED)
        )

    if not is_clean:
        logger.warning(f"Room not fully cleaned: {state}")

    return is_clean
```

# conversation\__init__.py

```py

```

# conversation\conversation_manager.py

```py
import asyncio
import websockets
import json
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, tensor_server_url: str = "ws://localhost:5001"):
        """Initialize conversation manager with tensor server connection"""
        self.tensor_server_url = tensor_server_url
        self.conversation_history: List[Dict] = []
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.memory_context: List[str] = []
        
    async def connect(self):
        """Connect to the tensor server for memory operations"""
        try:
            self.websocket = await websockets.connect(
                self.tensor_server_url,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            )
            logger.info("Connected to tensor server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to tensor server: {str(e)}")
            return False
            
    async def close(self):
        """Close the websocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
    async def add_to_memory(self, text: str, role: str):
        """Add a conversation turn to memory"""
        self.conversation_history.append({
            "role": role,
            "content": text
        })
        
        if self.websocket:
            try:
                # Match tensor server's expected message type
                message = {
                    "type": "embed",  
                    "text": text,
                    "metadata": {"role": role}
                }
                await self.websocket.send(json.dumps(message))
                response = await self.websocket.recv()
                data = json.loads(response)
                if data.get('type') != 'embeddings':
                    logger.error(f"Failed to add memory: Unexpected response type {data.get('type')}")
            except Exception as e:
                logger.error(f"Error adding to memory: {str(e)}")
                
    async def search_relevant_context(self, query: str, k: int = 3):
        """Search for relevant past context given a query"""
        if self.websocket:
            try:
                # Match tensor server's expected message type
                message = {
                    "type": "search",  
                    "text": query,
                    "limit": k
                }
                await self.websocket.send(json.dumps(message))
                response = await self.websocket.recv()
                data = json.loads(response)
                if data.get('type') == 'search_results':
                    self.memory_context = [r['text'] for r in data.get('results', [])]
                else:
                    logger.error(f"Failed to search context: Unexpected response type {data.get('type')}")
            except Exception as e:
                logger.error(f"Error searching context: {str(e)}")
                
    def get_context_for_llm(self, current_query: str) -> str:
        """Format conversation context for the LLM"""
        context = []
        
        # Add memory context if available
        if self.memory_context:
            context.append("Relevant past context:")
            context.extend(self.memory_context)
            context.append("")
            
        # Add recent conversation history
        context.append("Recent conversation:")
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            role_prefix = "User:" if turn["role"] == "user" else "Assistant:"
            context.append(f"{role_prefix} {turn['content']}")
            
        # Add current query
        context.append(f"User: {current_query}")
        context.append("Assistant:")
        
        return "\n".join(context)

```

# custom\__init__.py

```py

```

# custom\custom_speech_recognition.py

```py
import speech_recognition as sr
import threading
import queue
import logging
from typing import Optional, AsyncGenerator, Callable, Dict, Any, Awaitable
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class StreamingRecognizer:
    """A class for streaming audio recognition with interruption support and testing features."""

    def __init__(self, 
                device_index: Optional[int] = None,
                on_energy_update: Optional[Callable[[float], Awaitable[None]]] = None,
                on_speech_start: Optional[Callable[[], Awaitable[None]]] = None,
                on_speech_end: Optional[Callable[[], Awaitable[None]]] = None,
                on_recognition: Optional[Callable[[str, float], Awaitable[None]]] = None) -> None:
        """
        Initialize the recognizer with the specified device.
        
        Args:
            device_index (Optional[int]): The index of the microphone device to use.
        """
        # Device-related attributes
        self.device_index: Optional[int] = device_index
        self.recognizer: sr.Recognizer = sr.Recognizer()

        # Callback functions
        self._on_energy_update = on_energy_update
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_recognition = on_recognition

        # Performance monitoring and timing
        self._recognition_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None
        
        # Thread control attributes
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.processing_speech: bool = False
        
        # Queue management
        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=100)  # Bounded queue
        self.audio_buffer: queue.Queue[bytes] = queue.Queue(maxsize=1000)  # Audio buffer queue
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # Thread pool for concurrent processing
        
        # Locks for thread safety
        self._processing_lock = threading.Lock()
        self._queue_lock = threading.Lock()

        # Apply recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    async def process_file(self, file_path: str) -> str:
        """
        Process an audio file and return the recognized text.
        
        Args:
            file_path (str): Path to the audio file to process
            
        Returns:
            str: The recognized text from the audio file
            
        Raises:
            Exception: If there's an error processing the file
        """
        try:
            logger.debug(f"Processing file: {file_path}")
            
            async with asyncio.Lock():  # Ensure thread-safe file processing
                # Create a new recognizer instance for file processing
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = self.recognizer.energy_threshold
                recognizer.dynamic_energy_threshold = self.recognizer.dynamic_energy_threshold
                recognizer.dynamic_energy_adjustment_damping = self.recognizer.dynamic_energy_adjustment_damping
                recognizer.dynamic_energy_ratio = self.recognizer.dynamic_energy_ratio
                recognizer.pause_threshold = self.recognizer.pause_threshold
                recognizer.phrase_threshold = self.recognizer.phrase_threshold
                recognizer.non_speaking_duration = self.recognizer.non_speaking_duration
                
                # Read the audio file
                logger.debug("Opening audio file")
                with sr.AudioFile(file_path) as source:
                    # Record the audio file data
                    logger.debug("Recording audio data")
                    audio = recognizer.record(source)
                    
                    # Process recognition with retries and backoff
                    result = await self._process_recognition_with_retries(recognizer, audio)
                    if result:
                        return result
                    
                    return ""  # Return empty string if all attempts fail
                    
        except Exception as e:
            logger.error(f"Error processing audio file: {e}", exc_info=True)
            raise Exception(f"Failed to process audio file: {str(e)}")

    async def _process_recognition_with_retries(self, recognizer: sr.Recognizer, audio: sr.AudioData, max_retries: int = 3) -> Optional[str]:
        """Process recognition with retries and exponential backoff"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"Starting recognition attempt {attempt + 1}")
                
                # Adjust recognition parameters based on attempt
                if attempt == 1:
                    recognizer.energy_threshold = 150
                    recognizer.dynamic_energy_threshold = False
                elif attempt == 2:
                    recognizer.pause_threshold = 1.0
                    recognizer.phrase_threshold = 0.5
                
                # Use thread pool for recognition to prevent blocking
                text = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    recognizer.recognize_google,
                    audio
                )
                
                if text:
                    logger.debug(f"Recognition complete: {text}")
                    return text
                    
            except sr.UnknownValueError:
                if attempt == max_retries - 1:
                    logger.warning("Speech not recognized in any attempt")
                    return None
                    
                # Exponential backoff between retries
                await asyncio.sleep(0.5 * (2 ** attempt))
                logger.debug(f"Recognition attempt {attempt + 1} failed, trying different settings")
                continue
                
            except sr.RequestError as e:
                logger.error(f"Could not request results from speech recognition service: {e}")
                raise
        
        return None

    def start(self) -> None:
        """Start the recognition process."""
        with self._processing_lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._recognition_thread)
                self.thread.daemon = True
                self.thread.start()
                logger.info("Started recognition process.")

    def stop(self) -> None:
        """Stop the audio processing and wait for the thread to terminate."""
        with self._processing_lock:
            if self.running:
                self.running = False
                logger.info("Stopping audio processing...")
                
                # Clear queues
                with self._queue_lock:
                    while not self.text_queue.empty():
                        try:
                            self.text_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    while not self.audio_buffer.empty():
                        try:
                            self.audio_buffer.get_nowait()
                        except queue.Empty:
                            break
                
                if self.thread:
                    self.thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
                    if self.thread.is_alive():
                        logger.warning("Audio processing thread did not terminate cleanly")
                    else:
                        logger.info("Audio processing thread stopped.")
                
                # Shutdown thread pool
                self.thread_pool.shutdown(wait=False)

    def get_text(self) -> str:
        """
        Retrieve recognized text from the queue.
        
        Returns:
            str: The next recognized text, or an empty string if the queue is empty.
        """
        try:
            with self._queue_lock:
                return self.text_queue.get_nowait()
        except queue.Empty:
            return ""

    def _recognition_thread(self) -> None:
        """Main recognition thread that processes audio data"""
        while self.running:
            try:
                # Process audio from buffer
                if not self.audio_buffer.empty():
                    with self._queue_lock:
                        audio_data = self.audio_buffer.get_nowait()
                        
                    if audio_data:
                        self.processing_speech = True
                        try:
                            # Process audio chunk
                            text = self.recognizer.recognize_google(audio_data)
                            if text:
                                with self._queue_lock:
                                    if self.text_queue.full():
                                        # Remove oldest item if queue is full
                                        try:
                                            self.text_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    self.text_queue.put(text)
                        finally:
                            self.processing_speech = False
                            
            except Exception as e:
                logger.error(f"Error in recognition thread: {e}")
                
            # Small sleep to prevent tight loop
            time.sleep(0.1)

```

# event_emitter.py

```py
from typing import Any, Callable, Dict, List

class EventEmitter:
    """
    A simple event emitter class that allows registering event handlers,
    emitting events with arbitrary arguments, and removing handlers.
    
    Usage:
    
        emitter = EventEmitter()
        
        # Register a handler normally
        def on_event(data):
            print("Event received:", data)
        emitter.on("data", on_event)
        
        # Or register using decorator syntax
        @emitter.on("data")
        def handle_data(data):
            print("Decorator handler received:", data)
        
        # Emit an event:
        emitter.emit("data", {"key": "value"})
        
        # Remove a handler:
        emitter.off("data", on_event)
    """
    
    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler for the specified event.
        If used as a decorator (i.e. without providing a handler),
        the function will be automatically registered.
        
        Args:
            event_name: The name of the event.
            handler: Optional callable to handle the event.
            
        Returns:
            If used as a decorator, returns the wrapped function.
            Otherwise, returns the handler.
        """
        if handler is None:
            # Return a decorator if no handler is passed.
            def decorator(fn: Callable) -> Callable:
                self.on(event_name, fn)
                return fn
            return decorator
        
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)
        return handler

    def off(self, event_name: str, handler: Callable) -> None:
        """
        Remove a registered event handler.
        
        Args:
            event_name: The name of the event.
            handler: The handler to remove.
        """
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
                if not self._handlers[event_name]:
                    del self._handlers[event_name]
            except ValueError:
                # Handler was not found
                pass

    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event_name: The name of the event.
            *args: Positional arguments passed to the handler.
            **kwargs: Keyword arguments passed to the handler.
        """
        if event_name in self._handlers:
            for handler in self._handlers[event_name]:
                handler(*args, **kwargs)

```

# handlers\__init__.py

```py

```

# handlers\voice_handler.py

```py
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
import os
from voice_core.llm_communication import get_llm_response
from voice_core.response_processor import process_response
from voice_core.tts_utils import select_voice
from voice_core.conversation_manager import ConversationManager
from voice_core.custom_speech_recognition import StreamingRecognizer
from voice_core.mic_utils import select_microphone
from voice_core.shared_state import interrupt_handler, should_interrupt
from voice_core.livekit_stt_service import LiveKitSTTService
from voice_core.livekit_tts_service import LiveKitTTSService
from voice_core.config.config import LucidiaConfig
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-selected voice
VOICE_NAME = "en-US-AvaMultilingualNeural"

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Tensor server configuration
TENSOR_SERVER_URL = os.getenv("TENSOR_SERVER_URL", "ws://localhost:5001")

@dataclass
class VoiceSession:
    conversation_manager: ConversationManager
    voice: str
    active: bool = True
    recognizer: Optional[StreamingRecognizer] = None
    mic_device: Optional[str] = None
    stt_service: Optional[LiveKitSTTService] = None
    tts_service: Optional[LiveKitTTSService] = None
    room_name: Optional[str] = None
    tensor_ws: Optional[websockets.WebSocketClientProtocol] = None

class VoiceHandler:
    def __init__(self):
        self.sessions: Dict[str, VoiceSession] = {}
        self.config = LucidiaConfig()
        
    async def initialize_session(self, client_id: str) -> VoiceSession:
        """Initialize a new voice session for a client"""
        if client_id in self.sessions:
            await self.cleanup_session(client_id)
            
        # Initialize conversation manager
        conversation_manager = ConversationManager()
        
        # Select voice
        voice = await select_voice(VOICE_NAME)
        
        # Create session
        session = VoiceSession(
            conversation_manager=conversation_manager,
            voice=voice
        )
        
        # Initialize LiveKit services
        session.stt_service = LiveKitSTTService(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            on_transcript=lambda text: self.handle_transcript(client_id, text)
        )
        
        session.tts_service = LiveKitTTSService(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        
        # Connect to tensor server
        try:
            session.tensor_ws = await websockets.connect(TENSOR_SERVER_URL)
            logger.info(f"Connected to tensor server for client {client_id}")
        except Exception as e:
            logger.error(f"Failed to connect to tensor server: {e}")
            
        self.sessions[client_id] = session
        return session

    async def handle_transcript(self, client_id: str, text: str):
        """Handle transcribed text by sending it to tensor server"""
        session = self.sessions.get(client_id)
        if not session or not session.tensor_ws:
            return
            
        try:
            # Send transcript to tensor server
            await session.tensor_ws.send(json.dumps({
                'type': 'transcript',
                'text': text,
                'client_id': client_id
            }))
            
            # Wait for processed response
            response = await session.tensor_ws.recv()
            response_data = json.loads(response)
            
            if response_data['type'] == 'response':
                # Send to TTS service
                await session.tts_service.synthesize_speech(
                    response_data['text'],
                    session.voice
                )
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")

    async def handle_voice_message(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle incoming voice messages"""
        try:
            # Get or create session
            session = await self.initialize_session(client_id)
            
            message_type = message.get('type', '')
            if message_type == 'voice_input':
                return await self.handle_voice_input(message, session)
            elif message_type == 'session_control':
                return await self.handle_session_control(message, session)
            elif message_type == 'start_listening':
                return await self.handle_start_listening(session)
            elif message_type == 'stop_listening':
                return await self.handle_stop_listening(session)
            elif message_type == 'livekit_connect':
                return await self.handle_livekit_connect(message, session)
            else:
                return {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            logger.error(f"Error handling voice message: {str(e)}")
            return {
                "type": "error",
                "error": str(e)
            }

    async def handle_livekit_connect(self, message: Dict[str, Any], session: VoiceSession) -> Dict[str, Any]:
        """Handle LiveKit connection request"""
        try:
            if not session.stt_service or not session.tts_service:
                return {"type": "error", "error": "LiveKit services not initialized"}
                
            token = message.get('token')
            if not token:
                return {"type": "error", "error": "Token not provided"}
                
            # Connect both services
            await session.stt_service.connect(LIVEKIT_URL, token, session.room_name)
            await session.tts_service.connect(LIVEKIT_URL, token, session.room_name)
            
            return {
                "type": "livekit_connected",
                "room": session.room_name,
                "stt_room": f"{session.room_name}_stt",
                "tts_room": f"{session.room_name}_tts"
            }
        except Exception as e:
            return {"type": "error", "error": str(e)}

    async def handle_start_listening(self, session: VoiceSession) -> Dict[str, Any]:
        """Start listening for voice input"""
        if session.stt_service:
            # Using LiveKit for audio processing
            return {
                "type": "listening_started",
                "mode": "livekit"
            }
        elif session.recognizer:
            # Fallback to local recognition
            try:
                async def text_callback(text: str):
                    await self.process_voice_input(text, session)
                    
                session.recognizer.set_text_callback(text_callback)
                session.recognizer.start()
                return {
                    "type": "listening_started",
                    "mode": "local"
                }
            except Exception as e:
                return {"type": "error", "error": str(e)}
        else:
            return {
                "type": "error",
                "error": "No audio processing service available"
            }

    async def handle_stop_listening(self, session: VoiceSession) -> Dict[str, Any]:
        """Stop listening for voice input"""
        if session.stt_service:
            await session.stt_service.disconnect()
            
        if session.tts_service:
            await session.tts_service.disconnect()
            
        if session.recognizer:
            session.recognizer.stop()
            
        return {"type": "listening_stopped"}

    async def process_voice_input(self, text: str, session: VoiceSession):
        """Process transcribed voice input"""
        try:
            # Get LLM response
            response = await get_llm_response(text, session.conversation_manager)
            
            # Process response
            processed_response = await process_response(response)
            
            if session.tts_service:
                # Send response through LiveKit TTS
                await session.tts_service.synthesize_speech(processed_response)
            else:
                # Handle response through existing pipeline
                # (implement your existing TTS handling here)
                pass
                
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            
    async def cleanup_session(self, client_id: str):
        """Cleanup session resources"""
        session = self.sessions.get(client_id)
        if session:
            if session.tensor_ws:
                await session.tensor_ws.close()
            if session.stt_service:
                await session.stt_service.cleanup()
            if session.tts_service:
                await session.tts_service.cleanup()
            self.sessions.pop(client_id)

```

# livekit_integration\__init__.py

```py
"""LiveKit integration module for voice_core."""

import livekit.rtc as rtc
from .livekit_service import LiveKitService, LiveKitTransport

__all__ = [
    'rtc',
    'LiveKitService',
    'LiveKitTransport'
]
```

# livekit_integration\agent_forwarder.py

```py

```

# livekit_integration\agents.py

```py
"""Temporary stub implementation of LiveKit agents functionality"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict, List, AsyncIterator
from .rtc_stub import Room
import asyncio
import numpy as np
import logging
import argparse
import jwt
import time
import json
from voice_core.config.config import LiveKitConfig

logger = logging.getLogger(__name__)

class AutoSubscribe(Enum):
    NONE = "none"
    AUDIO_ONLY = "audio_only"
    VIDEO_ONLY = "video_only"
    ALL = "all"

class SpeechEventType(Enum):
    START = "start"
    TRANSCRIPT = "transcript"
    END = "end"

@dataclass
class SpeechEvent:
    type: SpeechEventType
    text: Optional[str] = None
    is_final: bool = False
    language: Optional[str] = None

@dataclass
class STTCapabilities:
    streaming: bool = False
    interim_results: bool = False
    punctuation: bool = False
    profanity_filter: bool = False

@dataclass
class APIConnectOptions:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    host: Optional[str] = None

DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()

class AudioBuffer:
    def __init__(self):
        self.data = np.array([], dtype=np.float32)
        self.sample_rate = 16000

    def append(self, data: np.ndarray):
        self.data = np.concatenate([self.data, data])

class RecognizeStream:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._running = True

    async def write(self, data: np.ndarray):
        if self._running:
            await self._queue.put(data)

    async def stop(self):
        self._running = False

    async def read(self) -> AsyncIterator[SpeechEvent]:
        while self._running or not self._queue.empty():
            try:
                data = await self._queue.get()
                yield SpeechEvent(type=SpeechEventType.TRANSCRIPT, text="", is_final=False)
            except asyncio.CancelledError:
                break

class STT:
    def __init__(self, capabilities: STTCapabilities):
        self.capabilities = capabilities

    async def stream(self, language: Optional[str] = None, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> RecognizeStream:
        return RecognizeStream()

    async def recognize(self, buffer: AudioBuffer, language: Optional[str] = None, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> str:
        return ""

class TTSSegmentsForwarder:
    def __init__(self, room: Room):
        self.room = room
        self._running = True

    async def forward_segments(self, segments: List[bytes]):
        """Forward TTS audio segments to LiveKit"""
        for segment in segments:
            if not self._running:
                break
            # In real implementation, this would publish audio data
            pass

    async def stop(self):
        """Stop forwarding segments"""
        self._running = False

@dataclass
class WorkerOptions:
    agent_name: str
    entrypoint_fnc: Callable
    prewarm_fnc: Optional[Callable] = None

class JobContext:
    def __init__(self, room_name: str = None):
        """Initialize JobContext with a Room instance"""
        try:
            self.config = LiveKitConfig()
            self.room = Room()
            self._initialized = False
            self.room_name = room_name
        except Exception as e:
            logger.error(f"Failed to initialize JobContext: {e}")
            self.room = None
            self._initialized = False

    def _generate_token(self) -> str:
        """Generate LiveKit token"""
        try:
            if not self.room_name:
                raise ValueError("Room name not provided")
                
            # Token claims
            claims = {
                "room": {
                    "room": self.room_name,
                    "roomJoin": True,
                    "canPublish": True,
                    "canSubscribe": True
                },
                "name": "lucidia-bot",  # Participant identity
                "metadata": json.dumps({"type": "bot"}),  # Optional metadata
                "iss": self.config.api_key,  # Use API key as issuer
                "sub": "lucidia-bot",  # Must match name
                "exp": int(time.time()) + 3600,  # 1 hour expiry
                "nbf": int(time.time()) - 300,  # Valid from 5 mins ago (clock skew)
                "iat": int(time.time())
            }
            
            # Generate token
            if not self.config.api_secret:
                raise ValueError("LiveKit API secret not configured")
                
            token = jwt.encode(
                claims,
                self.config.api_secret,
                algorithm="HS256"
            )
            
            logger.debug(f"Generated LiveKit token for room {self.room_name}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            raise

    async def connect(self, auto_subscribe: AutoSubscribe = AutoSubscribe.NONE):
        """Connect to LiveKit room"""
        try:
            if not self.room:
                self.room = Room()
            
            if not self._initialized:
                # Generate token and connect
                token = self._generate_token()
                await self.room.connect(
                    url=self.config.url,
                    token=token
                )
                self._initialized = True
                logger.info(f"Connected to room {self.room_name} with state: {self.room.connection_state}")
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            raise

    async def reconnect(self):
        """Reconnect to LiveKit room"""
        try:
            if self.room:
                await self.room.disconnect()
            self._initialized = False
            await self.connect()
        except Exception as e:
            logger.error(f"Failed to reconnect to room: {e}")
            raise

class cli:
    @staticmethod
    def run_app(options: WorkerOptions):
        """Run the voice agent application"""
        try:
            # Parse command line arguments
            parser = argparse.ArgumentParser(description="Voice Agent CLI")
            parser.add_argument("command", choices=["connect"], help="Command to execute")
            parser.add_argument("--room", required=True, help="Room name to connect to")
            args = parser.parse_args()
            
            # Create context with room name
            ctx = JobContext(room_name=args.room)
            
            # Run prewarm if provided
            if options.prewarm_fnc:
                logger.info("Prewarming resources...")
                options.prewarm_fnc(ctx)
                logger.info("Prewarm completed successfully")
            
            # Run entrypoint
            asyncio.run(options.entrypoint_fnc(ctx))
            
        except Exception as e:
            logger.error(f"Failed to run application: {e}")
            raise

```

# livekit_integration\livekit_handler.py

```py
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

```

# livekit_integration\livekit_service.py

```py
"""LiveKit service implementation."""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
import numpy as np
import livekit.rtc as rtc
import jwt

from voice_core.shared_state import should_interrupt
from voice_core.audio import AudioFrame, normalize_audio, resample_audio
from voice_core.stt import EnhancedSTTPipeline, WhisperConfig
from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState

logger = logging.getLogger(__name__)

# LiveKit server configuration
LIVEKIT_URL = "ws://localhost:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"

def generate_token(room_name: str, identity: str = "bot") -> str:
    """Generate a LiveKit access token."""
    claims = {
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
            "roomAdmin": False,
            "roomCreate": True
        },
        "iss": LIVEKIT_API_KEY,
        "sub": identity,
        "exp": 4070908800,  # Some time in 2099
        "jti": room_name + "_" + identity,
    }
    return jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")

class LiveKitAudioTrack:
    """Enhanced LiveKit audio track with proper PCM handling."""
    
    def __init__(self, track: rtc.LocalTrack):
        self.track = track
        self.input_sample_rate = 16000  # Edge TTS native rate
        self.output_sample_rate = 48000  # LiveKit required rate
        self.channels = 1
        self.frame_duration = 20  # ms
        self.samples_per_frame = int(self.output_sample_rate * self.frame_duration / 1000)
        self.buffer = np.array([], dtype=np.float32)
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def write_frame(self, frame: AudioFrame):
        """Write audio frame with proper resampling and buffering."""
        async with self._lock:
            try:
                # Ensure data is float32 and normalized
                frame_data = frame.data
                if frame_data.dtype != np.float32:
                    frame_data = frame_data.astype(np.float32)
                frame_data = normalize_audio(frame_data)
                
                # Resample if needed
                if frame.sample_rate != self.output_sample_rate:
                    frame_data = resample_audio(
                        frame_data,
                        frame.sample_rate,
                        self.output_sample_rate
                    )
                
                # Add to buffer
                self.buffer = np.append(self.buffer, frame_data)
                
                # Process complete frames
                while len(self.buffer) >= self.samples_per_frame:
                    frame_samples = self.buffer[:self.samples_per_frame]
                    self.buffer = self.buffer[self.samples_per_frame:]
                    
                    # Convert to int16 for LiveKit
                    int16_data = (frame_samples * 32767).astype(np.int16)
                    
                    # Create LiveKit audio frame
                    rtc_frame = rtc.AudioFrame(
                        data=int16_data.tobytes(),
                        samples_per_channel=self.samples_per_frame,
                        sample_rate=self.output_sample_rate
                    )
                    
                    # Write to track
                    await self.track.write_frame(rtc_frame)
                    
            except Exception as e:
                self.logger.error(f"Error writing audio frame: {e}")
                raise

    async def cleanup(self):
        """Clean up resources and flush buffer."""
        async with self._lock:
            if len(self.buffer) > 0:
                # Pad last frame if needed
                remaining_samples = len(self.buffer)
                if remaining_samples < self.samples_per_frame:
                    padding = np.zeros(self.samples_per_frame - remaining_samples, dtype=np.float32)
                    self.buffer = np.append(self.buffer, padding)
                
                # Convert to int16 for LiveKit
                int16_data = (self.buffer * 32767).astype(np.int16)
                
                # Send final frame
                rtc_frame = rtc.AudioFrame(
                    data=int16_data.tobytes(),
                    samples_per_channel=len(self.buffer),
                    sample_rate=self.output_sample_rate
                )
                await self.track.write_frame(rtc_frame)
            
            self.buffer = np.array([], dtype=np.float32)

class LiveKitTransport:
    """LiveKit transport layer for voice pipeline."""
    
    def __init__(self):
        self.room = None
        self.logger = logging.getLogger(__name__)
        self._event_handlers: Dict[str, Callable] = {}
        
    async def connect_to_room(self, room_name: str) -> rtc.Room:
        """Connect to a LiveKit room."""
        try:
            # Create room if needed
            if not self.room:
                self.room = rtc.Room()
                
            # Connect to room
            token = generate_token(room_name)
            await self.room.connect(LIVEKIT_URL, token)
            
            self.logger.info(f"Connected to LiveKit room: {room_name}")
            return self.room
            
        except Exception as e:
            self.logger.error(f"Failed to connect to room: {e}")
            raise
        
    def on(self, event: str, callback: Optional[Callable] = None):
        """Register event handlers."""
        def decorator(func: Callable):
            self._event_handlers[event] = func
            return func
            
        if callback:
            self._event_handlers[event] = callback
            return callback
            
        return decorator
        
    def _emit(self, event: str, data: Any = None):
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            try:
                self._event_handlers[event](data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event}: {e}")

class LiveKitService:
    """LiveKit service for managing room connections and audio streaming."""
    
    def __init__(self, config: WhisperConfig, room: rtc.Room, state_manager: Optional[VoiceStateManager] = None):
        self.config = config
        self.room = room
        self.stt_service = EnhancedSTTPipeline(config)
        self.state_manager = state_manager or VoiceStateManager()
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._audio_tasks: Dict[str, asyncio.Task] = {}
        self._state = {
            'is_publishing': False,
            'active_tracks': set(),
            'error': None
        }
        
    async def publish_track(self, track_name: str, source: rtc.AudioSource) -> rtc.LocalAudioTrack:
        """Publish an audio track with state management and event emission."""
        try:
            if self._state['is_publishing']:
                raise RuntimeError("Already publishing a track")
                
            self._state['is_publishing'] = True
            
            # Notify state change
            await self.state_manager.transition_to(
                VoiceState.SPEAKING,
                {"track_name": track_name}
            )
            
            local_track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
            await self.room.local_participant.publish_track(local_track)
            
            self._state['active_tracks'].add(track_name)
            self.logger.info(f"Published track: {track_name}")
            
            return local_track
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to publish track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            raise
        finally:
            self._state['is_publishing'] = False
            
    async def subscribe_to_track(self, track: rtc.AudioTrack, participant_id: str):
        """Subscribe to a remote audio track with state coordination."""
        try:
            if participant_id in self._audio_tasks:
                return
                
            # Update state for new track
            await self.state_manager.transition_to(
                VoiceState.LISTENING,
                {"participant_id": participant_id}
            )
            
            task = asyncio.create_task(self._process_audio_track(track, participant_id))
            self._audio_tasks[participant_id] = task
            self._state['active_tracks'].add(participant_id)
            
            self.logger.info(f"Subscribed to track from participant: {participant_id}")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to subscribe to track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            raise
            
    def get_state(self) -> Dict[str, Any]:
        """Get current service state including voice state."""
        return {
            'is_publishing': self._state['is_publishing'],
            'active_tracks': list(self._state['active_tracks']),
            'error': self._state['error'],
            'running': self._running,
            'voice_state': self.state_manager.current_state.value
        }
        
    async def stop_track(self, track_id: str):
        """Stop processing a track with state cleanup."""
        try:
            if track_id in self._audio_tasks:
                task = self._audio_tasks.pop(track_id)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
            self._state['active_tracks'].discard(track_id)
            
            # Reset state if no active tracks
            if not self._state['active_tracks']:
                await self.state_manager.transition_to(VoiceState.IDLE)
                
            self.logger.info(f"Stopped track: {track_id}")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to stop track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            
    async def stop(self):
        """Stop all audio processing with state cleanup."""
        try:
            self._running = False
            tasks = list(self._audio_tasks.values())
            self._audio_tasks.clear()
            
            for task in tasks:
                task.cancel()
                
            await asyncio.gather(*tasks, return_exceptions=True)
            self._state['active_tracks'].clear()
            
            # Reset to idle state
            await self.state_manager.transition_to(VoiceState.IDLE)
            await self.cleanup()
            
            self.logger.info("Stopped all audio processing")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Error during shutdown: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )

    async def _process_audio_track(self, track: rtc.AudioTrack, participant_id: str) -> None:
        """Process audio from a single track and publish transcripts."""
        if not self.room:
            self.logger.error("No LiveKit room provided for recognition.")
            return

        self.logger.info(f"Starting recognition for participant {participant_id}")
        try:
            audio_stream = rtc.AudioStream(track)
            async for event in audio_stream:
                if not self._running:
                    break
                try:
                    # Convert to float32 normalized audio
                    audio_np = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32)
                    audio_np /= 32767.0

                    transcript = await self.stt_service.process_audio(audio_np)
                    if transcript and transcript.strip():
                        self.logger.info(f"Final transcript for {participant_id}: {transcript}")
                        data = {
                            "type": "transcript",
                            "text": transcript,
                            "is_final": True,
                            "participant_id": participant_id,
                            "timestamp": time.time()
                        }
                        try:
                            await self.room.local_participant.publish_data(
                                json.dumps(data).encode("utf-8"),
                                reliable=True
                            )
                            self.logger.info("Published transcript to LiveKit.")
                        except Exception as e:
                            self.logger.error(f"Error publishing transcript: {e}")

                except Exception as e:
                    self.logger.error(f"Failed to process audio frame: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in recognition loop for {participant_id}: {e}", exc_info=True)
        finally:
            self.logger.info(f"Stopped recognition for participant {participant_id}")

    async def initialize(self) -> None:
        await self.stt_service.initialize()
        self._running = True
        self.logger.info("LiveKit service initialized with enhanced STT pipeline.")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop()
        await self.stt_service.cleanup()
        self.logger.info("LiveKit service cleanup complete.")

```

# livekit_integration\run_forwarded_agent.py

```py

```

# livekit_integration\run_voice_agent.py

```py

```

# livekit_integration\voice_agent.py

```py

```

# llm\__init__.py

```py
"""Local LLM service for voice agent responses"""

import logging
import asyncio
import json
import aiohttp
from typing import Optional, Dict, List

from .llm_pipeline import LocalLLMPipeline
from ..config.config import LLMConfig

logger = logging.getLogger(__name__)

__all__ = ['LocalLLMPipeline', 'LLMConfig', 'LocalLLMService']

class LocalLLMService:
    """Local LLM service using Qwen 2.5 7B model"""
    
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.session = None
        self.initialized = False
        self.system_prompt = """You are Lucidia, a helpful voice assistant. Keep your responses natural, concise, and conversational. 
        Avoid long explanations unless asked. Respond in a way that sounds natural when spoken."""
        
    async def initialize(self):
        """Initialize the LLM service"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Test connection to API
            async with self.session.get("http://localhost:1234/v1/models") as response:
                if response.status != 200:
                    raise ConnectionError(f"Failed to connect to LLM API: {response.status}")
                    
            self.initialized = True
            logger.info("✅ LLM service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            if self.session:
                await self.session.close()
            raise
            
    async def generate_response(self, text: str) -> Optional[str]:
        """Generate a response to user input"""
        try:
            if not self.initialized or not self.session:
                logger.warning("⚠️ LLM not initialized")
                return None
                
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Make API request
            async with self.session.post(
                self.api_url,
                json={
                    "messages": messages,
                    "model": "qwen2.5-7b",
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ LLM API error: {response.status}")
                    return None
                    
                data = await response.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return None
            
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False

```

# llm\llm_communication.py

```py
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Optional, Union, Generator, Any
from voice_core.config.config import LucidiaConfig

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
config = LucidiaConfig()
LLM_SERVER_URL = config.llm_server_url
MAX_HISTORY_LENGTH = 10
DEFAULT_TIMEOUT = 60

class LLMCommunicationError(Exception):
    """Custom exception for LLM communication errors."""
    pass

async def verify_llm_server() -> bool:
    """Verify LM Studio server connection."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLM_SERVER_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("LM Studio server connection verified")
                    return True
                logger.warning(f"LM Studio health check failed: {response.status}")
                return False
    except Exception as e:
        logger.warning(f"Could not connect to LM Studio: {e}")
        return False

class LocalLLM:
    """LM Studio LLM wrapper with OpenAI-compatible interface."""
    def __init__(self):
        self.chat_history = [
            {"role": "system", "content": "You are Lucidia, a helpful and engaging voice assistant."}
        ]
        self.capabilities = type("Capabilities", (), {"streaming": False})()

    async def generate(self, text: str) -> str:
        """Generate a response using LM Studio with OpenAI-compatible payload."""
        try:
            self.chat_history.append({"role": "user", "content": text})
            if len(self.chat_history) > MAX_HISTORY_LENGTH + 1:
                self.chat_history = [self.chat_history[0]] + self.chat_history[-MAX_HISTORY_LENGTH:]

            payload = {
                "model": config.llm.model_name,
                "messages": self.chat_history,
                "temperature": config.llm.temperature,
                "max_tokens": 150,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LLM_SERVER_URL,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise LLMCommunicationError(f"LM Studio error: {error_msg}")
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    self.chat_history.append({"role": "assistant", "content": content})
                    return content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I couldn’t generate a response."
```

# llm\llm_pipeline.py

```py
from __future__ import annotations
import aiohttp
import asyncio
import logging
from typing import Optional
from voice_core.config.config import LLMConfig

logger = logging.getLogger(__name__)

class LocalLLMPipeline:
    """Pipeline for local LLM integration using LM Studio."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "http://127.0.0.1:1234/v1"  # LM Studio default
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the aiohttp session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.debug("Initialized aiohttp ClientSession")
        try:
            # Test connectivity to LM Studio
            async with self.session.get(f"{self.base_url}/models") as resp:
                resp.raise_for_status()
                self.logger.info("Successfully connected to LM Studio")
        except Exception as e:
            self.logger.warning(f"LM Studio connectivity test failed: {e}")

    async def generate_response(self, prompt: str) -> str:
        """Generate a complete response from LM Studio."""
        if not prompt:
            self.logger.warning("Empty prompt provided, returning empty response")
            return ""

        if not self.session or self.session.closed:
            await self.initialize()

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are Lucidia, a helpful voice assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response = data["choices"][0]["message"]["content"].strip()
                self.logger.info(f"Generated response: {response[:50]}...")
                return response
        except aiohttp.ClientError as e:
            self.logger.error(f"LM Studio API error: {e}")
            return "Sorry, I couldn’t process that right now."
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_response: {e}", exc_info=True)
            return "An error occurred, please try again."

    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        if hasattr(self, 'model'):
            # Clean up any model resources
            self.model = None

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed aiohttp ClientSession")
        self.session = None
```

# main.py

```py
import asyncio
from voice_core.config.config import LucidiaConfig, PHRASE_TIMEOUT, MIN_PHRASE_TIMEOUT
from voice_core.mic_utils import select_microphone
from voice_core.custom_speech_recognition import StreamingRecognizer
from voice_core.tts_utils import select_voice
from voice_core.llm_communication import get_llm_response, verify_llm_server
from voice_core.response_processor import process_response
from voice_core.audio_playback import cleanup_audio
import queue
import voice_core.shared_state as shared_state

# Create default configuration instance
config = LucidiaConfig()
conversation_manager = None

async def get_audio_input():
    """Get audio input using streaming recognition (minimal delay)."""
    print("\nStarting audio processing pipeline")
    recognizer = StreamingRecognizer(shared_state.selected_microphone)
    print("Initializing speech recognition...")
    recognizer.start()
    print("Speech recognition started")

    full_text = []
    last_text_time = asyncio.get_event_loop().time()
    collection_started = False

    try:
        while True:
            try:
                if not collection_started:
                    print("Waiting for speech input...")

                text = recognizer.text_queue.get(timeout=PHRASE_TIMEOUT)
                current_time = asyncio.get_event_loop().time()

                if not collection_started:
                    print("Speech detected, collecting input...")
                    collection_started = True

                print(f"Received text: '{text}'")
                full_text.append(text)
                last_text_time = current_time

                if text.strip().endswith((".", "!", "?")):
                    print("Sentence break detected, checking for more input...")
                    try:
                        next_text = recognizer.text_queue.get(
                            timeout=MIN_PHRASE_TIMEOUT
                        )
                        print(f"Additional text received: '{next_text}'")
                        full_text.append(next_text)
                        last_text_time = asyncio.get_event_loop().time()
                    except queue.Empty:
                        pass

                # Check if we've been silent for long enough to conclude
                current_time = asyncio.get_event_loop().time()
                silence_duration = current_time - last_text_time
                # If we've been silent long enough, conclude
                if silence_duration > MIN_PHRASE_TIMEOUT:
                    print("Speech input complete")
                    if full_text:  # Only break if we have text to process
                        break

            except queue.Empty:
                if collection_started:
                    print("No more input detected")
                    if full_text:  # Only break if we have text to process
                        break
                continue

    except Exception as e:
        print(f"Error in audio input: {e}")
        return None

    finally:
        recognizer.stop()

    return " ".join(full_text)


async def main():
    """Main entry point."""
    global conversation_manager
    
    print("Entering main function")
    print("Conversational AI with Local LLM, Edge TTS, and Speech Recognition")
    print("You can say 'stop', 'wait', 'pause', or 'quiet' to interrupt the AI's speech.\n")

    # Initialize conversation manager
    conversation_manager = voice_core.conversation_manager.ConversationManager()
    if not await conversation_manager.connect():
        print("Warning: Could not connect to memory system. Continuing without memory.")

    # Verify LLM server connection
    if not verify_llm_server():
        print("Error: Unable to connect to LLM server. Please ensure it is running.")
        return

    # Select and calibrate microphone at startup
    print("\nSetting up microphone...")
    shared_state.selected_microphone = select_microphone()
    if shared_state.selected_microphone is None:
        print("Using default microphone")
    else:
        print(f"Selected microphone index: {shared_state.selected_microphone}")

    # Initialize recognizer to perform one-time calibration
    print("\nPerforming initial microphone calibration...")
    initial_recognizer = StreamingRecognizer(shared_state.selected_microphone)
    initial_recognizer.start()
    initial_recognizer.stop()
    print("Microphone setup complete\n")

    voice = await select_voice()
    if not voice:
        print("No voice selected. Exiting.")
        return

    print(f"Selected voice: {voice}")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Or press Enter (empty) to use voice input.\n")

    while True:
        try:
            # Get user input
            user_input = input("You (type or press Enter to speak): ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            # Handle speech input if no text input
            if not user_input:
                print("\nStarting speech recognition...")
                try:
                    user_input = await get_audio_input()
                    if user_input:
                        print(f"\nSpeech recognized successfully: {user_input}")
                    else:
                        print("No speech input detected")
                        continue
                except Exception as e:
                    print(f"\nError during speech recognition: {e}")
                    continue

            # Process input through LLM if we have valid input
            if user_input:
                # Store user input in memory
                if conversation_manager:
                    await conversation_manager.add_to_memory(user_input, "user")
                    # Get relevant context
                    await conversation_manager.search_relevant_context(user_input)
                
                print("\nSubmitting to LLM for processing...")
                print("AI: ", end="", flush=True)

                try:
                    # Get context-enhanced prompt
                    context_prompt = conversation_manager.get_context_for_llm(user_input) if conversation_manager else user_input
                    
                    # Get LLM response
                    llm_response = get_llm_response(context_prompt, stream=True)
                    if not llm_response:
                        print("Error: Failed to get response from LLM")
                        continue

                    # Store AI response in memory
                    if conversation_manager:
                        await conversation_manager.add_to_memory(llm_response, "assistant")

                    # Process the response
                    print("Processing LLM response...")
                    await process_response(llm_response, voice, streaming=True)
                    print("Response processing complete")
                except Exception as e:
                    print(f"\nError during LLM processing: {e}")
                    continue

        except Exception as e:
            print(f"\nUnexpected error: {e}")
            continue

    print("Thank you for the conversation!")
    if conversation_manager:
        await conversation_manager.close()
    cleanup_audio()


if __name__ == "__main__":
    print("Starting main function")
    asyncio.run(main())
    print("Script completed")

```

# memory_client.py

```py
"""
Memory system client for connecting to the tensor and HPC servers.
Handles embedding generation and memory operations.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
import websockets

logger = logging.getLogger(__name__)

class MemoryClient:
    """Client for interacting with the memory system servers."""
    
    def __init__(self, 
                 tensor_url: str = "ws://localhost:5001",
                 hpc_url: str = "ws://localhost:5005",
                 session_id: str = None):
        """
        Initialize memory client.
        
        Args:
            tensor_url: WebSocket URL for tensor server
            hpc_url: WebSocket URL for HPC server
            session_id: Unique session identifier
        """
        self.tensor_url = tensor_url
        self.hpc_url = hpc_url
        self.session_id = session_id or str(time.time())
        
        # Connection state
        self._tensor_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._hpc_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Local cache
        self._conversation_history: List[Dict[str, Any]] = []
        self._embeddings_cache: Dict[str, List[float]] = {}
        
    async def initialize(self) -> bool:
        """Initialize connections to memory servers."""
        try:
            # Start connection handler
            self._reconnect_task = asyncio.create_task(self._maintain_connections())
            
            # Wait for tensor server connection first
            for _ in range(3):  # Try for 3 seconds
                if self._tensor_ws:
                    break
                await asyncio.sleep(1)
                
            if not self._tensor_ws:
                logger.error("Failed to connect to tensor server")
                return False
                
            # Consider initialization successful if tensor server is connected
            # HPC connection will be maintained in background
            logger.info("Memory client initialized with tensor server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory client: {e}")
            return False
            
    async def _maintain_connections(self) -> None:
        """Maintain WebSocket connections with reconnection."""
        while True:
            try:
                # Connect to tensor server if needed
                if not self._tensor_ws:
                    try:
                        async with websockets.connect(self.tensor_url) as ws:
                            self._tensor_ws = ws
                            logger.info("Connected to tensor server")
                            
                            # Handle tensor server messages
                            async for message in ws:
                                try:
                                    data = json.loads(message)
                                    await self._handle_tensor_message(data)
                                except json.JSONDecodeError:
                                    logger.error("Invalid JSON from tensor server")
                                except Exception as e:
                                    logger.error(f"Error handling tensor message: {e}")
                    except Exception as e:
                        logger.error(f"Error connecting to tensor server: {e}")
                        self._tensor_ws = None
                        await asyncio.sleep(1)  # Delay before retry
                
                # Connect to HPC server if needed
                if not self._hpc_ws:
                    try:
                        async with websockets.connect(self.hpc_url) as ws:
                            self._hpc_ws = ws
                            logger.info("Connected to HPC server")
                            self._connected = True
                            
                            # Handle HPC server messages
                            async for message in ws:
                                try:
                                    data = json.loads(message)
                                    await self._handle_hpc_message(data)
                                except json.JSONDecodeError:
                                    logger.error("Invalid JSON from HPC server")
                                except Exception as e:
                                    logger.error(f"Error handling HPC message: {e}")
                    except Exception as e:
                        logger.error(f"Error connecting to HPC server: {e}")
                        self._hpc_ws = None
                        await asyncio.sleep(1)  # Delay before retry
                        
            except Exception as e:
                logger.error(f"Error in connection maintenance: {e}")
                
            await asyncio.sleep(1)  # Main loop delay
            
    async def _handle_tensor_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming tensor server message."""
        msg_type = data.get("type")
        
        if msg_type == "embeddings":
            # Cache embeddings
            timestamp = data.get("timestamp")
            embeddings = data.get("embeddings")
            if timestamp and embeddings:
                self._embeddings_cache[timestamp] = embeddings
                
                # Forward to HPC server
                if self._hpc_ws:
                    await self._hpc_ws.send(json.dumps({
                        "type": "process_embeddings",
                        "session_id": self.session_id,
                        "timestamp": timestamp,
                        "embeddings": embeddings
                    }))
                    
    async def _handle_hpc_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming HPC server message."""
        msg_type = data.get("type")
        
        if msg_type == "memory_processed":
            # Update local cache with processed memory
            memory_id = data.get("memory_id")
            if memory_id:
                logger.info(f"Memory processed: {memory_id}")
                
    async def store_transcript(self, text: str, sender: str) -> bool:
        """
        Store a conversation transcript.
        
        Args:
            text: The transcript text
            sender: Who sent the message ("user" or "assistant")
            
        Returns:
            True if stored successfully
        """
        try:
            # Add to local cache
            entry = {
                "text": text,
                "sender": sender,
                "timestamp": time.time()
            }
            self._conversation_history.append(entry)
            
            # Request embeddings from tensor server
            if self._tensor_ws:
                await self._tensor_ws.send(json.dumps({
                    "type": "embed",
                    "session_id": self.session_id,
                    "text": text,
                    "timestamp": entry["timestamp"]
                }))
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return False
            
    async def store_conversation(self, text: str, role: str = "assistant") -> bool:
        """Store conversation entry with embeddings."""
        try:
            # Add to conversation history
            entry = {
                "text": text,
                "role": role,
                "timestamp": time.time()
            }
            self._conversation_history.append(entry)
            
            # Request embeddings from tensor server
            if self._tensor_ws:
                await self._tensor_ws.send(json.dumps({
                    "type": "embed",
                    "session_id": self.session_id,
                    "text": text,
                    "timestamp": entry["timestamp"]
                }))
                
                # Wait for response
                response = await self._tensor_ws.recv()
                result = json.loads(response)
                
                if result.get('type') == 'embeddings':
                    # Cache the embeddings
                    self._embeddings_cache[text] = result['embeddings']
                    logger.info(f"Stored conversation with embeddings: {text[:50]}...")
                    return True
                else:
                    logger.error(f"Unexpected response type: {result.get('type')}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return False
            
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get cached conversation history."""
        return self._conversation_history.copy()
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            
        if self._tensor_ws:
            await self._tensor_ws.close()
            
        if self._hpc_ws:
            await self._hpc_ws.close()
            
        self._connected = False
        self._tensor_ws = None
        self._hpc_ws = None
        logger.info("Memory client cleaned up")

```

# pipeline\__init__.py

```py

```

# pipeline\voice_pipeline_metrics.py

```py
"""Voice pipeline metrics tracking."""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class TimingMetric:
    """Timing metric for a pipeline stage."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    count: int = 0
    error_count: int = 0
    cancel_count: int = 0

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def end(self, error: bool = False, cancelled: bool = False):
        """End timing and update metrics."""
        self.end_time = time.time()
        self.duration += self.end_time - self.start_time
        self.count += 1
        if error:
            self.error_count += 1
        if cancelled:
            self.cancel_count += 1

class VoicePipelineMetrics:
    """Tracks metrics for the voice pipeline."""
    
    def __init__(self):
        """Initialize metrics."""
        self.logger = logging.getLogger(__name__)
        self._metrics: Dict[str, TimingMetric] = {
            'speech': TimingMetric(),
            'stt': TimingMetric(),
            'llm': TimingMetric(),
            'tts': TimingMetric()
        }
        self._start_time = time.time()

    def speech_start(self):
        """Mark start of speech."""
        self._metrics['speech'].start()
        self.logger.debug("Speech started")

    def speech_end(self):
        """Mark end of speech."""
        self._metrics['speech'].end()
        self.logger.debug("Speech ended")

    def start_stt(self):
        """Mark start of STT processing."""
        self._metrics['stt'].start()
        self.logger.debug("STT started")

    def end_stt(self, error: bool = False, cancelled: bool = False):
        """Mark end of STT processing."""
        self._metrics['stt'].end(error, cancelled)
        self.logger.debug("STT ended")

    def start_llm(self):
        """Mark start of LLM processing."""
        self._metrics['llm'].start()
        self.logger.debug("LLM started")

    def end_llm(self, error: bool = False, cancelled: bool = False):
        """Mark end of LLM processing."""
        self._metrics['llm'].end(error, cancelled)
        self.logger.debug("LLM ended")

    def start_tts(self):
        """Mark start of TTS processing."""
        self._metrics['tts'].start()
        self.logger.debug("TTS started")

    def end_tts(self, error: bool = False, cancelled: bool = False):
        """Mark end of TTS processing."""
        self._metrics['tts'].end(error, cancelled)
        self.logger.debug("TTS ended")

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current metrics."""
        metrics = {}
        for name, metric in self._metrics.items():
            if metric.count > 0:
                avg_duration = metric.duration / metric.count
                metrics[name] = {
                    'avg_duration': avg_duration,
                    'count': metric.count,
                    'error_rate': metric.error_count / metric.count if metric.count > 0 else 0,
                    'cancel_rate': metric.cancel_count / metric.count if metric.count > 0 else 0
                }
        return metrics

    def log_metrics(self):
        """Log current metrics."""
        metrics = self.get_metrics()
        for name, stats in metrics.items():
            self.logger.info(
                f"{name.upper()}: avg={stats['avg_duration']:.3f}s, " +
                f"count={stats['count']}, errors={stats['error_rate']:.1%}, " +
                f"cancels={stats['cancel_rate']:.1%}"
            )

```

# plugins\__init__.py

```py
"""Voice core plugins for enhanced functionality."""

from .tts_segments_forwarder import TTSSegment, TTSSegmentsForwarder
from .turn_detector import TurnConfig, TurnDetector

__all__ = [
    'TTSSegment',
    'TTSSegmentsForwarder',
    'TurnConfig',
    'TurnDetector',
]

```

# plugins\tts_segments_forwarder.py

```py
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

from livekit import rtc


@dataclass
class TTSSegment:
    text: str
    id: str = ""
    final: bool = True
    language: str = "en-US"


class TTSSegmentsForwarder:
    """
    Forwards TTS transcription segments to a LiveKit room.

    This component maintains an internal asynchronous queue of TTSSegment
    objects. As segments are added, they are packaged into a LiveKit
    Transcription object and published using the room's local participant.
    """

    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.Participant | str,
        language: str = "en-US",
        speed: float = 1.0,
    ):
        self.room = room
        # If a participant object is provided, use its identity; otherwise assume a string
        self.participant = participant if isinstance(participant, str) else participant.identity
        self.language = language
        self.speed = speed
        self._queue = asyncio.Queue[Optional[TTSSegment]]()
        self._task = asyncio.create_task(self._run())

        # Retrieve the audio track SID from the participant's publications if available.
        if not isinstance(participant, str):
            audio_tracks = [
                track for track in participant.track_publications.values()
                if track.kind == rtc.TrackKind.KIND_AUDIO
            ]
            if audio_tracks:
                self.track_sid = audio_tracks[0].sid
            else:
                self.track_sid = None
        else:
            self.track_sid = None

    async def _run(self):
        """Process segments from the queue and forward them to LiveKit."""
        try:
            while True:
                segment = await self._queue.get()
                if segment is None:
                    break

                # If no audio track is available, skip processing.
                if not self.track_sid:
                    print("No audio track available for transcription")
                    continue

                # Create a transcription object using the segment data.
                transcription = rtc.Transcription(
                    participant_identity=self.participant,
                    track_sid=self.track_sid,
                    segments=[
                        rtc.TranscriptionSegment(
                            id=segment.id,
                            text=segment.text,
                            start_time=0,
                            end_time=0,
                            final=segment.final,
                            language=segment.language
                        )
                    ]
                )

                # Only publish if the room is connected.
                if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    try:
                        await self.room.local_participant.publish_transcription(transcription)
                    except Exception as e:
                        print(f"Error publishing transcription: {e}")
        except Exception as e:
            print(f"Error in TTS forwarder: {e}")

    async def add_text(self, text: str, final: bool = True):
        """Add a text segment to be forwarded as a transcription."""
        segment = TTSSegment(
            text=text,
            id=str(uuid.uuid4()),
            final=final,
            language=self.language
        )
        await self._queue.put(segment)

    async def close(self):
        """Close the forwarder and wait for the processing task to complete."""
        await self._queue.put(None)
        if self._task:
            await self._task


# Example usage for testing purposes
if __name__ == "__main__":
    async def main():
        # Create dummy LiveKit objects for testing.
        class DummyParticipant:
            identity = "dummy_participant"
            async def publish_transcription(self, transcription):
                print("Publishing transcription:")
                print(transcription)
        class DummyRoom:
            connection_state = rtc.ConnectionState.CONN_CONNECTED
            local_participant = DummyParticipant()
            # Dummy implementations for compatibility.
            def __init__(self):
                self.track_publications = {}
        dummy_room = DummyRoom()

        # Instantiate the forwarder with a dummy participant.
        forwarder = TTSSegmentsForwarder(room=dummy_room, participant="dummy_participant", language="en-US")
        await forwarder.add_text("Hello, this is a test transcription.", final=True)
        await forwarder.close()

    asyncio.run(main())

```

# plugins\turn_detector.py

```py
from __future__ import annotations

import asyncio
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import logging

# Configure logging for demonstration purposes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class TurnConfig:
    """Configuration for turn detection."""
    min_silence_duration: float = 1.0  # Minimum silence duration to trigger end of turn
    max_turn_duration: float = 30.0      # Maximum duration of a single turn
    initial_buffer_duration: float = 0.5 # Initial buffer before starting turn detection
    energy_threshold: float = -40        # Energy threshold in dB for silence detection


class TurnDetector:
    """Detect conversation turns based on audio energy and timing."""
    
    def __init__(
        self,
        config: Optional[TurnConfig] = None,
        on_turn_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_turn_end: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.config = config or TurnConfig()
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        
        self._turn_active = False
        self._last_audio_time = 0.0
        self._turn_start_time = 0.0
        self._last_energy = -60.0
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the turn detection process."""
        self.logger.info("Starting turn detector")
        self._running = True
        # Launch the monitoring loop as a background task.
        self._monitor_task = asyncio.create_task(self._monitor_turns())

    async def stop(self):
        """Stop the turn detection process."""
        self.logger.info("Stopping turn detector")
        self._running = False
        if self._monitor_task:
            await self._monitor_task
        # Allow time for any pending tasks to finish.
        await asyncio.sleep(0.1)
    
    def is_active(self) -> bool:
        """Return whether a turn is currently active."""
        return self._turn_active

    def update_audio_level(self, energy: float):
        """
        Update the current audio energy level.
        
        Args:
            energy: Raw energy value (e.g. mean squared amplitude)
        """
        # Convert energy to decibels, with a floor at -60 dB.
        energy_db = max(-60, 10 * np.log10(energy + 1e-10))
        self._last_energy = energy_db
        self._last_audio_time = time.time()
        self.logger.debug(f"Updated energy level: {energy_db:.2f} dB")

    async def _monitor_turns(self):
        """Monitor audio levels and detect turn boundaries."""
        try:
            while self._running:
                now = time.time()
                
                if self._turn_active:
                    # Calculate silence duration and turn duration.
                    silence_duration = now - self._last_audio_time
                    turn_duration = now - self._turn_start_time
                    
                    if (silence_duration >= self.config.min_silence_duration or 
                        turn_duration >= self.config.max_turn_duration):
                        self.logger.info("Turn ended (silence or max duration reached)")
                        self._turn_active = False
                        if self.on_turn_end:
                            await self.on_turn_end()
                
                else:
                    # Check if conditions to start a turn are met.
                    if (self._last_energy > self.config.energy_threshold and
                        now - self._last_audio_time <= self.config.initial_buffer_duration):
                        self.logger.info("Turn started")
                        self._turn_active = True
                        self._turn_start_time = now
                        if self.on_turn_start:
                            await self.on_turn_start()
                
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in turn detection loop: {e}")
            raise


# ------------------------- DEMONSTRATION USAGE -------------------------

async def demo_turn_detector():
    # Define asynchronous callbacks for turn events.
    async def on_turn_start():
        print(">>> Turn started!")
        
    async def on_turn_end():
        print("<<< Turn ended!")

    # Create a turn detector instance with default configuration and callbacks.
    detector = TurnDetector(on_turn_start=on_turn_start, on_turn_end=on_turn_end)
    detector.start()

    # Simulate audio level updates.
    # We'll simulate a "speech turn" by providing high energy values,
    # then simulate silence with very low energy values.
    try:
        print("Simulating speech turn (2 seconds)...")
        # Simulate speech (high energy values) for 2 seconds.
        for _ in range(20):
            # An energy value that converts to a high dB level (above threshold).
            detector.update_audio_level(energy=0.01)
            await asyncio.sleep(0.1)
        
        print("Simulating silence (1.5 seconds)...")
        # Simulate silence by using very low energy values.
        for _ in range(15):
            detector.update_audio_level(energy=1e-12)
            await asyncio.sleep(0.1)
        
        print("Simulating another speech turn (1 second)...")
        # Simulate another speech turn.
        for _ in range(10):
            detector.update_audio_level(energy=0.01)
            await asyncio.sleep(0.1)
        
        # Let the detector run a bit longer.
        await asyncio.sleep(2)
        
    finally:
        await detector.stop()
        print("Turn detector stopped.")

if __name__ == "__main__":
    asyncio.run(demo_turn_detector())

```

# README.md

```md
# Voice Core System Fixes

This document outlines the fixes implemented to address critical issues in the voice assistant pipeline.

## Issues Fixed

### 1. Transcription.__init__() Error

**Problem**: The voice assistant was using an incorrect format for the Transcription API, causing errors when trying to publish transcriptions.

**Fix**: Updated all components to use the correct Transcription API structure with segments, track_sid, and participant identity.

**Files Modified**:
- `voice_state_manager.py`: Updated `handle_stt_transcript` method
- `enhanced_stt_service.py`: Updated `_publish_transcript_to_ui` method
- `interruptible_tts_service.py`: Updated `_publish_transcription` method

### 2. Multi-Turn Flow Hanging

**Problem**: The voice assistant would hang during multi-turn conversations, failing to transition back to the LISTENING state.

**Fix**: Improved state transition logic, added timeout handling for LLM processing, and ensured proper buffer clearing after each turn.

**Files Modified**:
- `agent2.py`: Added timeout handling in `_handle_transcript` method
- `voice_state_manager.py`: Enhanced `start_speaking` method to ensure transitions to LISTENING
- `interruptible_tts_service.py`: Added explicit state transitions after TTS completion

### 3. "Task was destroyed but it is pending!" Error

**Problem**: Asyncio tasks were not being properly managed, leading to "Task was destroyed but it is pending!" errors.

**Fix**: Implemented proper task cancellation and cleanup with appropriate exception handling.

**Files Modified**:
- `voice_state_manager.py`: Improved task management in `start_speaking` method
- `agent2.py`: Added proper task handling in `_handle_transcript` method

## Testing the Fixes

### Automated Tests

Run the automated tests to verify the fixes:

\`\`\`bash
python -m tests.test_voice_fixes
\`\`\`

These tests verify:
- Correct Transcription API usage with segments
- Multi-turn conversation flow without hanging
- LLM timeout handling
- TTS interruption handling

### Demo Script

Run the demo script to see the fixes in action:

\`\`\`bash
python run_voice_assistant_demo.py
\`\`\`

Optional arguments:
- `--room`: LiveKit room name (default: "demo-room")
- `--url`: LiveKit server URL (default: "ws://localhost:7880")
- `--api-key`: LiveKit API key (default: "devkey")
- `--api-secret`: LiveKit API secret (default: "secret")
- `--identity`: Participant identity (default: auto-generated)
- `--debug`: Enable debug logging

### Manual Testing

To manually test the fixes:

1. Start the LiveKit server:
   \`\`\`bash
   docker-compose up -d livekit
   \`\`\`

2. Run the voice assistant:
   \`\`\`bash
   python server/run_voice_server.py
   \`\`\`

3. Connect to the LiveKit Agents Playground:
   - Open `http://localhost:3000` in your browser
   - Connect to the same room as the voice assistant

4. Test multi-turn conversations:
   - Speak a phrase and wait for the response
   - Speak another phrase immediately after the response
   - Verify that the assistant responds to each input without hanging

5. Test interruptions:
   - Start speaking while the assistant is responding
   - Verify that the assistant stops speaking and processes your new input

## Implementation Details

### Sequence Tracking for Transcripts

Added sequence numbers to ensure transcripts are displayed in the correct order in the UI:

\`\`\`python
self._transcript_sequence += 1
sequence = self._transcript_sequence
\`\`\`

### LLM Timeout Handling

Added timeout handling to prevent hanging during LLM processing:

\`\`\`python
try:
    response = await asyncio.wait_for(llm_task, timeout=15.0)
    # Process response
except asyncio.TimeoutError:
    # Handle timeout
\`\`\`

### State Transition Guarantees

Added finally blocks to ensure state transitions happen even after errors:

\`\`\`python
finally:
    # Ensure transition to LISTENING after each turn
    if self.state_manager.current_state not in [VoiceState.ERROR, VoiceState.INTERRUPTED]:
        await self.state_manager.transition_to(VoiceState.LISTENING)
\`\`\`

### Proper Task Cancellation

Implemented proper task cancellation with exception handling:

\`\`\`python
if self._current_tts_task and not self._current_tts_task.done():
    self._current_tts_task.cancel()
    try:
        await self._current_tts_task
    except asyncio.CancelledError:
        pass
```

# requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
edge-tts==6.1.9
SpeechRecognition==3.10.0
numpy==1.26.2
python-dotenv==1.0.0
livekit>=0.20.0
livekit-agents>=0.12.0
livekit-plugins-openai>=0.0.4
livekit-plugins-silero>=0.0.4
playsound==1.3.0
aiohttp>=3.10.0
aiofiles==23.2.1
websockets>=12.0
sentence-transformers>=2.2.2
torch>=2.0.0
sounddevice>=0.4.6
beautifulsoup4>=4.12.0
markdown>=3.5.1
pydub>=0.25.1
```

# response_processor.py

```py
import threading
import queue
import re
import asyncio
import logging
from typing import Iterable, Tuple, Union

from voice_core.shared_state import should_interrupt
from voice_core.tts_utils import text_to_speech, markdown_to_text
from voice_core.llm_communication import update_conversation_history
from voice_core.audio_playback import playback_worker

# Configure logging for detailed traceability.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum word threshold for processing a text chunk.
THRESHOLD_WORDS: int = 20

# TTS timeout settings
MIN_TTS_TIMEOUT: float = 2.0  # Minimum timeout for any TTS conversion
BASE_TTS_TIMEOUT: float = 5.0  # Base timeout for longer text
CHARS_PER_SECOND: float = 15.0  # Expected TTS processing speed


async def async_process_word(text: str, voice: str, playback_queue: queue.Queue) -> bool:
    """
    Asynchronously convert a text chunk to speech and enqueue the resulting audio data.
    
    Args:
        text (str): The text chunk to convert.
        voice (str): The desired TTS voice.
        playback_queue (queue.Queue): Queue where resulting audio data is enqueued.
    
    Returns:
        bool: True if conversion succeeded and audio data was enqueued; False otherwise.
    """
    if should_interrupt.is_set():
        logger.debug("Interrupt set before processing; skipping TTS conversion.")
        return False

    if len(text.strip()) < 2:
        logger.debug("Text chunk is too short; skipping TTS conversion.")
        return False

    logger.info(f"Starting TTS conversion for chunk: '{text}'")
    try:
        # Calculate timeout based on text length with a minimum threshold
        # For short phrases, use MIN_TTS_TIMEOUT
        # For longer text, scale based on character count but cap at 10 seconds
        char_count = len(text.strip())
        if char_count < 20:  # Very short phrases
            chunk_timeout = MIN_TTS_TIMEOUT
        else:
            # Calculate expected time based on character count
            expected_time = char_count / CHARS_PER_SECOND
            chunk_timeout = min(BASE_TTS_TIMEOUT + expected_time, 10.0)
        
        logger.debug(f"Using {chunk_timeout:.1f}s timeout for {char_count} characters")
        audio_data = await asyncio.wait_for(text_to_speech(text, voice), timeout=chunk_timeout)
        
        if audio_data and not should_interrupt.is_set():
            logger.info("TTS conversion succeeded; enqueuing audio data for playback.")
            playback_queue.put(audio_data)
            return True
        else:
            logger.info("TTS conversion produced no audio data or was interrupted post-conversion.")
            return False
            
    except asyncio.TimeoutError:
        logger.warning(f"TTS conversion timed out after {chunk_timeout:.1f}s for text: '{text}'")
        return False
    except Exception as e:
        logger.error(f"Error during TTS conversion: {e}")
        return False


class WorkerManager:
    """
    Manages TTS and playback worker threads.
    
    This class encapsulates a TTS worker (which uses its own asyncio event loop)
    and a playback worker. It exposes helper methods to enqueue text for TTS conversion
    and cleanly shuts down the worker threads.
    """
    def __init__(self, voice: str) -> None:
        self.voice: str = voice
        self.tts_queue: queue.Queue[Tuple[Union[str, None], Union[str, None]]] = queue.Queue()
        self.playback_queue: queue.Queue = queue.Queue()
        self.tts_thread: threading.Thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.playback_thread: threading.Thread = threading.Thread(target=playback_worker, args=(self.playback_queue,), daemon=True)

    def start(self) -> None:
        """
        Start both TTS and playback worker threads.
        """
        self.tts_thread.start()
        self.playback_thread.start()
        logger.info("WorkerManager: Started TTS and playback threads.")

    def _tts_worker(self) -> None:
        """
        Worker thread function that processes TTS tasks.
        
        Runs an asyncio event loop to process text chunks asynchronously.
        Terminates on receiving a sentinel or if an interrupt is detected.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("TTS worker: Event loop created.")
        try:
            while not should_interrupt.is_set():
                try:
                    item = self.tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Sentinel check: (None, None) signals termination.
                if item[0] is None:
                    logger.info("TTS worker: Received termination sentinel.")
                    self.tts_queue.task_done()
                    break

                text, voice = item
                logger.info(f"TTS worker: Processing text '{text}' with voice '{voice}'.")
                try:
                    loop.run_until_complete(async_process_word(text, voice, self.playback_queue))
                except Exception as e:
                    logger.error(f"TTS worker: Error processing text '{text}': {e}")
                finally:
                    self.tts_queue.task_done()
        finally:
            loop.close()
            logger.info("TTS worker: Event loop closed.")

    def enqueue(self, text: str) -> None:
        """
        Enqueue a text chunk for TTS processing.
        
        Args:
            text (str): The text chunk to be processed.
        """
        logger.info(f"WorkerManager: Enqueuing text chunk: '{text}'")
        self.tts_queue.put((text, self.voice))

    def join(self) -> None:
        """
        Wait for all enqueued tasks to complete and then cleanly terminate worker threads.
        """
        self.tts_queue.join()
        self.playback_queue.join()
        # Send sentinel values to signal termination.
        self.tts_queue.put((None, None))
        self.playback_queue.put(None)
        self.tts_thread.join()
        self.playback_thread.join()
        logger.info("WorkerManager: Worker threads have terminated.")


def enqueue_chunks_for_streaming(response: Iterable[str], manager: WorkerManager) -> str:
    """
    Process a streaming response by accumulating text chunks, splitting on sentence
    boundaries (or by word threshold), and enqueuing each chunk for TTS.
    
    Args:
        response (Iterable[str]): Generator or iterable yielding text chunks.
        manager (WorkerManager): Manager instance for enqueuing TTS tasks.
    
    Returns:
        str: The complete response text accumulated from the stream.
    """
    buffer = ""
    complete_response = ""
    for chunk in response:
        if should_interrupt.is_set():
            logger.info("Interrupt detected; aborting streaming chunk enqueuing.")
            break
        if not chunk:
            continue

        buffer += chunk
        complete_response += chunk
        logger.debug(f"Streaming: Accumulated buffer length is {len(buffer)} characters.")

        # Split text on sentence boundaries using regex with lookbehind.
        sentences = re.split(r'(?<=[.!?])\s+', buffer)
        if len(sentences) > 1 or len(buffer.split()) >= THRESHOLD_WORDS:
            if len(sentences) > 1:
                *complete_sentences, buffer = sentences
                to_process = " ".join(complete_sentences)
                logger.info(f"Streaming: Extracted {len(complete_sentences)} complete sentence(s) for TTS.")
            else:
                to_process = buffer
                buffer = ""
                logger.info("Streaming: Buffer reached word threshold for TTS processing.")

            if to_process.strip():
                manager.enqueue(to_process.strip())

    if buffer.strip() and not should_interrupt.is_set():
        logger.info(f"Streaming: Enqueuing final TTS chunk from buffer: '{buffer.strip()}'")
        manager.enqueue(buffer.strip())

    return complete_response


def enqueue_chunks_for_non_streaming(response: str, manager: WorkerManager) -> None:
    """
    Process a non-streaming response by converting Markdown to plain text,
    splitting it into natural sentence chunks, and enqueuing each for TTS.
    
    Args:
        response (str): The complete response text in Markdown.
        manager (WorkerManager): Manager instance for enqueuing TTS tasks.
    """
    plain_text = markdown_to_text(response)
    sentences = re.split(r'(?<=[.!?])\s+', plain_text)
    current_chunk = ""
    for sentence in sentences:
        current_chunk += sentence + " "
        if len(current_chunk.split()) >= THRESHOLD_WORDS:
            if current_chunk.strip():
                manager.enqueue(current_chunk.strip())
            current_chunk = ""
    if current_chunk.strip():
        manager.enqueue(current_chunk.strip())
    if plain_text.strip():
        logger.info("Non-streaming: Updating conversation history with assistant response.")
        update_conversation_history("assistant", plain_text.strip())


async def process_response(
    response: Union[Iterable[str], str],
    voice: str,
    streaming: bool = False
) -> None:
    """
    Process a TTS response (streaming or non-streaming) and coordinate playback.
    
    This function creates a WorkerManager to handle TTS and playback tasks. For a
    streaming response, it accumulates text chunks, splits them appropriately, and
    enqueues each for TTS conversion. For non-streaming responses, it converts Markdown
    to plain text, splits the text into natural chunks, and enqueues them.
    
    After enqueuing, it waits for all tasks to complete, updates conversation history,
    and terminates the worker threads.
    
    Args:
        response (Union[Iterable[str], str]): The response text or generator of text chunks.
        voice (str): The TTS voice parameter.
        streaming (bool): Flag indicating whether the response is streaming.
    """
    logger.info("Processing response for TTS and playback.")
    manager = WorkerManager(voice)
    manager.start()

    if streaming:
        logger.info("Processing streaming response.")
        complete_response = enqueue_chunks_for_streaming(response, manager)
        if complete_response.strip():
            logger.info("Streaming: Updating conversation history with full response.")
            update_conversation_history("assistant", complete_response.strip())
    else:
        logger.info("Processing non-streaming response.")
        enqueue_chunks_for_non_streaming(response, manager)

    logger.info("Waiting for all TTS and playback tasks to complete.")
    manager.join()
    logger.info("Response processing complete; all tasks finished.")

```

# state\voice_state_enum.py

```py
# voice_state_enum.py

from enum import Enum

class VoiceState(Enum):
    """Possible states for the voice assistant pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    INTERRUPTED = "interrupted"
    ERROR = "error"

```

# state\voice_state_manager.py

```py
# voice_state_manager.py
# Enhanced version with proper transcript attribution and sequencing

import asyncio
import json
import logging
import time
import uuid
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Awaitable

import livekit.rtc as rtc

from enum import Enum, auto

class VoiceState(Enum):
    """Possible states for the voice assistant."""
    IDLE = auto()          # Initial state, not yet listening
    LISTENING = auto()     # Actively listening for speech
    PROCESSING = auto()    # Processing speech (LLM)
    SPEAKING = auto()      # Speaking (TTS)
    INTERRUPTED = auto()   # Interrupted by user
    ERROR = auto()         # Error state

class VoiceStateManager:
    """
    Enhanced state manager with improved UI integration for LiveKit.
    Manages state transitions and coordination between voice pipeline components.
    """

    def __init__(
        self,
        processing_timeout: float = 15.0,
        speaking_timeout: float = 60.0,
        debug: bool = False
    ) -> None:
        """Initialize the voice state manager.
        
        Args:
            processing_timeout: Maximum time to wait for LLM processing (seconds)
            speaking_timeout: Maximum time to wait for TTS completion (seconds)
            debug: Enable debug logging
        """
        self.logger = logging.getLogger("VoiceStateManager")
        self._debug = debug
        
        # State management
        self._state = VoiceState.IDLE
        self._state_lock = asyncio.Lock()
        self._state_history: List[Dict[str, Any]] = []
        self._state_monitor_task: Optional[asyncio.Task] = None
        self._processing_timeout = processing_timeout
        self._speaking_timeout = speaking_timeout
        
        # Event handling
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Task management - improved tracking
        self._tasks: Dict[str, asyncio.Task] = {}
        self._in_progress_task: Optional[asyncio.Task] = None
        self._current_tts_task: Optional[asyncio.Task] = None
        self._last_tts_text: Optional[str] = None
        
        # Transcript handling - improved deduplication and sequencing
        self._transcript_handler: Optional[Callable[[str], Awaitable[None]]] = None
        self._transcript_sequence: int = 0
        self._recent_processed_transcripts: List[Tuple[str, float]] = []  # (hash, timestamp)
        self._transcript_hash_memory_seconds = 5.0  # Remember transcripts for this many seconds
        self._min_transcript_interval = 0.5  # Seconds
        self._last_transcript_time = 0.0
        
        # Interrupt handling - enhanced
        self._interrupt_requested = False
        self._interrupt_handled = asyncio.Event()
        self._interrupt_handled.set()  # Start in "handled" state
        
        # LiveKit integration
        self._room: Optional[rtc.Room] = None
        self._tts_track: Optional[rtc.LocalAudioTrack] = None
        self._tts_source: Optional[rtc.AudioSource] = None
        
        # Retry settings for data publishing
        self._max_retries = 3
        self._retry_delay = 0.5  # seconds
        
        # Publishing stats
        self._publish_stats = {
            "attempts": 0,
            "failures": 0,
            "retries": 0,
            "successes": 0,
            "last_error": None,
            "last_successful_publish": 0
        }
        
        # Last error for debugging
        self._last_error = None
        
        # Start background state monitor
        self._start_state_monitor()
        
        # Initialize context for tts_session
        self._tts_context_manager_initialized = False

    def _start_state_monitor(self) -> None:
        """Start a background task to monitor state transitions."""
        if self._state_monitor_task is None or self._state_monitor_task.done():
            self._state_monitor_task = asyncio.create_task(self._monitor_state())
            self._tasks["state_monitor"] = self._state_monitor_task

    async def _monitor_state(self) -> None:
        """Periodically check if stuck in a non-listening state for too long."""
        try:
            while True:
                current_state = self._state
                last_transition = next((t for t in reversed(self._state_history) if t["to"] == current_state.name), None)
                
                if last_transition:
                    time_in_state = time.time() - last_transition["timestamp"]
                    
                    # Determine timeout based on state
                    if current_state == VoiceState.PROCESSING:
                        timeout = self._processing_timeout
                    elif current_state == VoiceState.SPEAKING:
                        timeout = self._speaking_timeout
                    elif current_state == VoiceState.ERROR:
                        timeout = 10.0  # Short timeout for ERROR state
                    else:
                        timeout = None  # No timeout for other states
                    
                    if timeout and time_in_state > timeout:
                        self.logger.warning(f"Stuck in {current_state.name} for {time_in_state:.1f}s, forcing reset to LISTENING")
                        
                        # Forcefully cancel any in-progress tasks
                        await self._cancel_active_tasks()
                        
                        # Transition to LISTENING state with timeout reason
                        await self.transition_to(
                            VoiceState.LISTENING,
                            {"reason": f"{current_state.name.lower()}_timeout", "prev_state": current_state.name}
                        )
                
                # Check more frequently to be more responsive
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.logger.info("State monitor task cancelled")
        except Exception as e:
            self.logger.error(f"Error in state monitor: {e}", exc_info=True)
            self._last_error = str(e)

    async def _cancel_active_tasks(self):
        """Cancel all active tasks with proper cleanup."""
        # Cancel in-progress task
        if self._in_progress_task and not self._in_progress_task.done():
            self.logger.info(f"Cancelling in-progress task from state {self._state.name}")
            self._in_progress_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._in_progress_task), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.warning("Task cancellation timed out or was cancelled")
            except Exception as e:
                self.logger.error(f"Error during task cancellation: {e}", exc_info=True)
            finally:
                self._in_progress_task = None
                
        # Cancel TTS task
        if self._current_tts_task and not self._current_tts_task.done():
            self.logger.info("Cancelling stuck TTS task")
            self._current_tts_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.warning("TTS task cancellation timed out or was cancelled")
            except Exception as e:
                self.logger.error(f"Error during TTS task cancellation: {e}", exc_info=True)
            finally:
                self._current_tts_task = None
                self._last_tts_text = None
                
        # Cancel other tracked tasks
        for name, task in list(self._tasks.items()):
            if not task.done() and task != self._state_monitor_task:
                self.logger.info(f"Cancelling tracked task: {name}")
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    self.logger.error(f"Error cancelling task {name}: {e}")
                finally:
                    self._tasks.pop(name, None)

    @property
    def current_state(self) -> VoiceState:
        return self._state

    async def set_room(self, room: rtc.Room) -> None:
        """Set LiveKit room for UI updates and track publishing."""
        self._room = room
        await self.setup_tts_track(room)

    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler, with optional decorator usage.
        """
        def decorator(func):
            self._event_handlers.setdefault(event_name, []).append(func)
            return func

        if handler is None:
            return decorator
        else:
            return decorator(handler)

    async def emit(self, event_name: str, data: Any = None) -> None:
        """Emit an event to all registered handlers."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    result = handler(data)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_name}: {e}")

    def _normalize_text(self, text: str) -> str:
        """Lowercases, strips punctuation, and merges whitespace for better dedup checks."""
        if not text:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two strings (0.0 to 1.0)."""
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # Simple case: exact match
        if text1 == text2:
            return 1.0
            
        # Simple case: one is substring of the other
        if text1 in text2 or text2 in text1:
            return 0.8
            
        # Compute word-level similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        common_words = words1.intersection(words2)
        union_size = len(words1.union(words2))
        
        return len(common_words) / union_size if union_size > 0 else 0.0

    def _is_duplicate_transcript(self, text: str) -> bool:
        """Check if this transcript is a duplicate of a recent one."""
        if not text or not text.strip():
            return True
        
        # Clean up transcript history - remove old entries
        now = time.time()
        self._recent_processed_transcripts = [
            (h, t) for h, t in self._recent_processed_transcripts 
            if now - t < self._transcript_hash_memory_seconds
        ]
        
        # Get normalized hash of current text
        norm_text = self._normalize_text(text)
        
        # Check against recent transcript hashes
        for hash_text, timestamp in self._recent_processed_transcripts:
            if self._compute_similarity(norm_text, hash_text) > 0.8:
                return True
        
        # Also check if too soon since last transcript
        if now - self._last_transcript_time < self._min_transcript_interval:
            return True
            
        return False

    def _get_status_for_ui(self, state: VoiceState) -> dict:
        """
        Get status information for UI updates.
        
        Args:
            state: Current voice state
            
        Returns:
            dict: Status information including state name and metadata
        """
        status = {
            "state": state.name,
            "timestamp": time.time(),
            "metrics": {
                "interruptions": 0,
                "errors": 0,
                "transcripts": len(self._recent_processed_transcripts)
            }
        }
        
        # Add state-specific metadata
        if state == VoiceState.LISTENING:
            status.update({
                "listening": True,
                "speaking": False,
                "processing": False
            })
        elif state == VoiceState.SPEAKING:
            status.update({
                "listening": False,
                "speaking": True,
                "processing": False,
                "tts_active": True if self._current_tts_task and not self._current_tts_task.done() else False
            })
        elif state == VoiceState.PROCESSING:
            status.update({
                "listening": False,
                "speaking": False,
                "processing": True
            })
        elif state == VoiceState.ERROR:
            status.update({
                "listening": False,
                "speaking": False,
                "processing": False,
                "error": self._last_error if self._last_error else "Unknown error"
            })
            
        return status

    async def transition_to(self, new_state: VoiceState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Change state with concurrency protection and UI updates."""
        if metadata is None:
            metadata = {}
            
        # Add timestamp to metadata
        metadata["timestamp"] = time.time()
        
        async with self._state_lock:
            old_state = self._state
            
            # Skip if same state with special handling for PROCESSING
            if new_state == self._state and new_state != VoiceState.PROCESSING:
                return
                
            # Special handling for specific state transitions
            if new_state == VoiceState.INTERRUPTED:
                # When interrupting, set interrupt flag
                self._interrupt_requested = True
                self._interrupt_handled.clear()
                
                # Cancel the current TTS task if exists
                if self._current_tts_task and not self._current_tts_task.done():
                    self.logger.info("Cancelling TTS task due to interruption")
                    self._current_tts_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    self._current_tts_task = None
            
            elif new_state == VoiceState.ERROR:
                # When transitioning to ERROR, cancel any in-progress tasks
                await self._cancel_active_tasks()
                    
            elif new_state == VoiceState.LISTENING:
                # When transitioning to LISTENING, clear interrupt flags
                self._interrupt_requested = False
                self._interrupt_handled.set()
                
            # Update state
            self._state = new_state
            
            # Record state transition in history
            transition = {
                "from": old_state.name,
                "to": new_state.name,
                "timestamp": metadata.get("timestamp", time.time()),
                "metadata": metadata
            }
            self._state_history.append(transition)
            
            # Log the transition
            self.logger.info(f"State transition: {old_state.name} -> {new_state.name} {metadata}")
            
            # Emit state change event
            await self.emit("state_change", {
                "old_state": old_state,
                "new_state": new_state,
                "metadata": metadata
            })
            
            # Publish state update to LiveKit if room is available
            if self._room and self._room.local_participant:
                try:
                    await self._publish_with_retry(
                        json.dumps({
                            "type": "state_update",
                            "from": old_state.name,
                            "to": new_state.name,
                            "timestamp": metadata.get("timestamp", time.time()),
                            "metadata": metadata,
                            "status": self._get_status_for_ui(new_state)
                        }).encode(),
                        "state update"
                    )
                    
                    # Also publish agent-status for UI compatibility
                    await self._publish_with_retry(
                        json.dumps({
                            "type": "agent-status",
                            "status": self._get_status_for_ui(new_state),
                            "timestamp": time.time()
                        }).encode(),
                        "agent status"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to publish state update: {e}", exc_info=True)

    async def handle_user_speech_detected(self, text: Optional[str] = None) -> None:
        """
        Handle detection of user speech with improved interruption handling.
        
        Args:
            text: Optional transcript text if available
        """
        # Only interrupt if we're currently speaking
        if self._state == VoiceState.SPEAKING:
            self.logger.info("User speech detected while speaking, interrupting TTS")
            
            # Set interrupt flags immediately
            self._interrupt_requested = True
            self._interrupt_handled.clear()
            
            # Cancel the current TTS task if exists
            if self._current_tts_task and not self._current_tts_task.done():
                self.logger.info("Cancelling TTS task due to interruption")
                self._current_tts_task.cancel()
                try:
                    # Use a shorter timeout for more responsive cancellation
                    await asyncio.wait_for(self._current_tts_task, timeout=0.2)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self.logger.info("TTS task cancelled due to interruption")
                self._current_tts_task = None
                    
            # Wait briefly for interrupt to be handled
            try:
                # Reduced timeout for faster transitions
                interrupt_handled = await asyncio.wait_for(
                    self._interrupt_handled.wait(), 
                    timeout=0.3  # Very short timeout for responsiveness
                )
                
                if not interrupt_handled:
                    self.logger.warning("Interrupt not handled within timeout")
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for interrupt to be handled")
            
            # If we have text, transition to PROCESSING
            if text:
                await self.transition_to(VoiceState.PROCESSING, {"text": text})
            else:
                # Otherwise go back to LISTENING
                await self.transition_to(VoiceState.LISTENING, {"reason": "interrupted"})

    async def handle_stt_transcript(self, text: str) -> bool:
        """
        Handle a final STT transcript from the STT service.
        
        Returns:
            bool: True if transcript was processed, False if ignored
        """
        if not text or not text.strip():
            self.logger.warning("Ignoring empty transcript")
            return False
            
        # Check for duplicate or too recent
        now = time.time()
        normalized_text = self._normalize_text(text)
        if self._is_duplicate_transcript(text):
            self.logger.warning(f"Ignoring duplicate transcript: '{text[:30]}...'")
            if self._state != VoiceState.LISTENING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "duplicate_ignored"})
            return False
            
        # Record this as processed
        self._last_transcript_time = now
        self._recent_processed_transcripts.append((normalized_text, now))
        
        # Log the accepted transcript
        self.logger.info(f"Processing transcript: '{text[:30]}...'")
        
        # Publish transcript to UI with clear user attribution
        await self.publish_transcription(text, "user", True)
        
        # Handle based on current state
        if self._state == VoiceState.SPEAKING:
            # If speaking, handle as interruption
            await self.handle_user_speech_detected(text)
            return True
            
        # If already processing, cancel the current task
        if self._state == VoiceState.PROCESSING and self._in_progress_task and not self._in_progress_task.done():
            self.logger.info("Already processing, cancelling current task")
            self._in_progress_task.cancel()
            try:
                await asyncio.wait_for(self._in_progress_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._in_progress_task = None
        
        # Transition to PROCESSING state
        await self.transition_to(VoiceState.PROCESSING, {"text": text})
        
        # Create a task for the transcript handler
        if self._transcript_handler:
            task_name = f"transcript_{time.time()}"
            task = asyncio.create_task(self._transcript_handler(text))
            self._tasks[task_name] = task
            self._in_progress_task = task
            
            try:
                # Set a timeout for processing
                await asyncio.wait_for(task, timeout=self._processing_timeout)
                # Clean up task reference
                self._tasks.pop(task_name, None)
                self._in_progress_task = None
                return True
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Transcript processing timed out after {self._processing_timeout}s")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                self._tasks.pop(task_name, None)
                self._in_progress_task = None
                await self.transition_to(VoiceState.LISTENING, {"reason": "processing_timeout"})
                
            except asyncio.CancelledError:
                self.logger.info("Transcript processing was cancelled")
                self._tasks.pop(task_name, None)
                self._in_progress_task = None
                
            except Exception as e:
                self.logger.error(f"Error processing transcript: {e}", exc_info=True)
                self._tasks.pop(task_name, None)
                self._in_progress_task = None
                await self.transition_to(VoiceState.ERROR, {"error": str(e)})
                
        else:
            self.logger.warning("No transcript handler registered")
            await self.transition_to(VoiceState.LISTENING, {"reason": "no_handler"})
            
        return True

    async def setup_tts_track(self, room: rtc.Room) -> None:
        """
        Initialize a TTS track for the pipeline to send out audio.
        
        Args:
            room: LiveKit room to publish the track to
        """
        self.logger.info("Setting up TTS track")
        self._room = room
        
        # Clean up any existing track first
        await self.cleanup_tts_track()
        
        try:
            # Create audio source (48kHz mono)
            self._tts_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
            
            # Create track from source
            self._tts_track = rtc.LocalAudioTrack.create_audio_track("tts-track", self._tts_source)
            
            # Publish track to room
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE  # Treat as microphone for clients
            
            self.logger.info("Publishing TTS track to room")
            await room.local_participant.publish_track(self._tts_track, options)
            
            # Publish track info to UI
            await self._publish_with_retry(
                json.dumps({
                    "type": "tts_track_published",
                    "track_id": self._tts_track.sid if hasattr(self._tts_track, "sid") else None,
                    "timestamp": time.time()
                }).encode(),
                "TTS track info"
            )
            
            self.logger.info("TTS track setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up TTS track: {e}", exc_info=True)
            await self.register_error(e, "setup_tts_track")
            # Clean up any partial setup
            await self.cleanup_tts_track()

    async def start_speaking(self, tts_task: asyncio.Task) -> None:
        """Begin a TTS task with cancellation of previous tasks if needed."""
        # Cancel existing TTS if it exists
        if self._current_tts_task and not self._current_tts_task.done():
            # Extract the text from the task if possible
            if hasattr(self._current_tts_task, 'text'):
                self._last_tts_text = getattr(self._current_tts_task, 'text')
            self._current_tts_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Store the task
        task_name = f"tts_{time.time()}"
        self._tasks[task_name] = tts_task
        self._current_tts_task = tts_task
        
        # Store the text being spoken to avoid processing it as input
        if hasattr(tts_task, 'text'):
            self._last_tts_text = getattr(tts_task, 'text')
        
        # Clear interrupt flag
        self._interrupt_requested = False
        self._interrupt_handled.set()
        
        # Transition to SPEAKING state
        await self.transition_to(VoiceState.SPEAKING)

        # Wait for task completion or cancellation
        try:
            await tts_task
            self.logger.info("TTS task completed normally")
            if self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_complete"})
                
        except asyncio.CancelledError:
            self.logger.info("TTS task was cancelled")
            if self._state == VoiceState.INTERRUPTED:
                self._interrupt_handled.set()
            elif self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_cancelled"})
                
        except Exception as e:
            self.logger.error(f"Error in TTS task: {e}", exc_info=True)
            await self.register_error(e, "tts")
            
        finally:
            self._tasks.pop(task_name, None)
            self._current_tts_task = None
            
            # Ensure safe fallback if we are still in SPEAKING state
            if self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_completion"})

    def tts_session(self, text: str):
        """Context manager for TTS sessions that ensures proper state transitions."""
        if not self._tts_context_manager_initialized:
            # Initialize the context manager class
            outer_self = self
            
            class TTSSession:
                def __init__(self, text):
                    self.text = text
                    self.state_manager = outer_self
                    
                async def __aenter__(self):
                    await self.state_manager.transition_to(VoiceState.SPEAKING)
                    return self
                    
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is asyncio.CancelledError:
                        await self.state_manager.transition_to(VoiceState.LISTENING, 
                                                             {"reason": "tts_cancelled"})
                    elif exc_type is not None:
                        await self.state_manager.register_error(exc_val, "tts_session")
                    elif self.state_manager.current_state == VoiceState.SPEAKING:
                        await self.state_manager.transition_to(VoiceState.LISTENING, 
                                                             {"reason": "tts_complete"})
                    return False  # Don't suppress exceptions
            
            self._TTSSession = TTSSession
            self._tts_context_manager_initialized = True
            
        return self._TTSSession(text)

    async def register_error(self, error: Exception, source: str) -> None:
        """
        Transition to ERROR state with error details.
        
        Args:
            error: The exception that occurred
            source: Source component where the error occurred
        """
        error_str = str(error)
        self.logger.error(f"Error in {source}: {error_str}", exc_info=True)
        
        self._last_error = error_str
        
        # Transition to ERROR state
        await self.transition_to(
            VoiceState.ERROR, 
            {"error": error_str, "source": source}
        )
        
        # Publish error to UI
        if self._room and self._room.local_participant:
            try:
                await self._publish_with_retry(
                    json.dumps({
                        "type": "error",
                        "error": error_str,
                        "source": source,
                        "timestamp": time.time()
                    }).encode(),
                    "error notification"
                )
            except Exception as e:
                self.logger.error(f"Failed to publish error: {e}")
        
        # Emit error event
        await self.emit("error", {"error": error, "source": source})
        
        # After a brief delay, transition back to LISTENING
        await asyncio.sleep(2.0)
        if self._state == VoiceState.ERROR:
            await self.transition_to(VoiceState.LISTENING, {"reason": "error_recovery"})

    async def finish_processing(self) -> None:
        """
        Finish processing state and go back to listening.
        Called when LLM processing is complete and we're ready to listen again.
        """
        if self._state == VoiceState.PROCESSING:
            self.logger.info("Processing complete, transitioning to LISTENING")
            await self.transition_to(VoiceState.LISTENING, {"reason": "processing_complete"})
        else:
            self.logger.warning(f"Called finish_processing while in {self._state.name} state")

    def interrupt_requested(self) -> bool:
        """Check if interruption is requested."""
        return self._interrupt_requested or self._state == VoiceState.ERROR

    async def wait_for_interrupt(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for an interrupt to occur.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if interrupted, False if timed out
        """
        try:
            if timeout:
                return await asyncio.wait_for(self._interrupt_handled.wait(), timeout)
            else:
                await self._interrupt_handled.wait()
                return True
        except asyncio.TimeoutError:
            return False

    async def publish_transcription(
        self, 
        text: str, 
        sender: str = "user", 
        is_final: bool = True,
        participant_identity: Optional[str] = None
    ) -> bool:
        """
        Publish transcript to LiveKit using both data channel and Transcription API.
        
        Args:
            text: The transcript text to publish
            sender: Either "user" or "assistant"
            is_final: Whether this is a final transcript (vs. interim)
            participant_identity: Optional override for participant identity
            
        Returns:
            bool: True if publishing was successful
        """
        if not self._room or not self._room.local_participant:
            self.logger.warning("No room or participant, skipping transcript publish")
            return False
            
        success = True
        
        # Increment sequence for ordering
        self._transcript_sequence += 1
        seq = self._transcript_sequence
        
        # Determine the participant identity to use
        # Use provided identity or fallback to local participant identity
        identity_to_use = participant_identity or self._room.local_participant.identity
        
        # Debug logging for identity resolution
        self.logger.debug(
            f"Transcript identity resolution: sender='{sender}', "
            f"participant_identity={participant_identity}, "
            f"resolved_identity='{identity_to_use}'"
        )
        
        try:
            # 1) Publish custom data via data channel
            data_success = await self._publish_with_retry(
                json.dumps({
                    "type": "transcript",
                    "text": text,
                    "sender": sender,  # Clearly identify sender
                    "participant_identity": identity_to_use,  # Include explicit identity
                    "sequence": seq,   # Include sequence for ordering
                    "timestamp": time.time(),
                    "is_final": is_final
                }).encode(),
                f"{sender} transcript"
            )
            
            if not data_success:
                self.logger.warning(f"Failed to publish {sender} transcript via data channel")
                success = False
                
            # 2) Publish via Transcription API
            try:
                track_sid = None
                if sender == "user":
                    # For user transcripts, find the appropriate remote participant's track
                    for participant in self._room.remote_participants.values():
                        if participant.identity == identity_to_use:
                            for pub in participant.track_publications.values():
                                if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.sid:
                                    track_sid = pub.sid
                                    break
                            if track_sid:
                                break
                else:
                    # For assistant transcripts, use the local participant's track
                    for pub in self._room.local_participant.track_publications.values():
                        if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.sid:
                            track_sid = pub.sid
                            break

                if track_sid:
                    # Important: Use consistent segment format
                    segment_id = str(uuid.uuid4())
                    current_time = int(time.time() * 1000)  # milliseconds
                    
                    trans = rtc.Transcription(
                        participant_identity=identity_to_use,
                        track_sid=track_sid,
                        segments=[
                            rtc.TranscriptionSegment(
                                id=segment_id,
                                text=text,
                                start_time=current_time,
                                end_time=current_time,
                                final=is_final,
                                language="en"
                            )
                        ]
                    )
                    await self._room.local_participant.publish_transcription(trans)
                    self.logger.debug(f"Published transcription with identity '{identity_to_use}'")
                else:
                    self.logger.warning("No audio track SID for Transcription API")
                    success = False
            except Exception as e:
                self.logger.warning(f"Failed transcription API publish: {e}")
                success = False
                
        except Exception as e:
            self.logger.error(f"Failed to publish transcript: {e}", exc_info=True)
            success = False
            
        return success

    def register_transcript_handler(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """
        Register a handler function for transcripts.
        
        Args:
            handler: Async function that takes a transcript string and processes it
        """
        self.logger.info("Registering transcript handler")
        self._transcript_handler = handler

    async def cleanup(self) -> None:
        """Clean up all resources and tasks."""
        self.logger.info("Cleaning up voice state manager resources")
        
        # Cancel all tracked tasks
        await self._cancel_active_tasks()
            
        # Clean up TTS track
        await self.cleanup_tts_track()
            
        # Clear state and collections
        self._state = VoiceState.IDLE
        self._last_tts_text = None
        self._recent_processed_transcripts.clear()
        self._interrupt_requested = False
        self._interrupt_handled.set()
        
        # Clear event handlers
        self._event_handlers.clear()
        
        self.logger.info("Voice state manager cleanup completed")

    async def cleanup_tts_track(self) -> None:
        """
        Clean up TTS track resources.
        """
        try:
            # Unpublish track if it exists and we have a room
            if self._tts_track and self._room and self._room.local_participant:
                try:
                    self.logger.info("Unpublishing TTS track")
                    await self._room.local_participant.unpublish_track(self._tts_track)
                except Exception as e:
                    self.logger.warning(f"Error unpublishing TTS track: {e}")
            
            # Close audio source
            if self._tts_source:
                try:
                    self.logger.info("Closing TTS audio source")
                    if hasattr(self._tts_source, 'aclose'):
                        await self._tts_source.aclose()
                    # Some versions use close() instead of aclose()
                    elif hasattr(self._tts_source, 'close'):
                        self._tts_source.close()
                except Exception as e:
                    self.logger.warning(f"Error closing TTS source: {e}")
        except Exception as e:
            self.logger.error(f"Error during TTS track cleanup: {e}", exc_info=True)
        finally:
            # Always clear references
            self._tts_track = None
            self._tts_source = None
            self.logger.info("TTS track cleanup complete")

    def get_state_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of state transitions.
        
        Returns:
            List of state transition dictionaries with from, to, timestamp, and metadata
        """
        return self._state_history.copy()
        
    def get_current_state(self) -> VoiceState:
        """
        Get the current state.
        
        Returns:
            Current VoiceState
        """
        return self._state

    def get_analytics(self) -> Dict[str, Any]:
        """Gather analytics on state transitions and transcript publishing."""
        if not self._state_history:
            return {}

        # Time in each state
        state_durations: Dict[str, float] = {}
        state_counts: Dict[str, int] = {}

        for i, transition in enumerate(self._state_history):
            s = transition["from"]
            next_time = (self._state_history[i+1]["timestamp"]
                         if i < len(self._state_history) - 1 else time.time())
            duration = next_time - transition["timestamp"]
            state_durations[s] = state_durations.get(s, 0.0) + duration
            state_counts[s] = state_counts.get(s, 0) + 1

        # Interruption stats
        interruption_count = sum(
            1 for t in self._state_history if t["to"] == VoiceState.INTERRUPTED.name
        )
        speaking_count = sum(
            1 for t in self._state_history if t["to"] == VoiceState.SPEAKING.name
        )
        interruption_rate = interruption_count / max(speaking_count, 1)

        # Processing durations
        proc_durations = []
        for i, t in enumerate(self._state_history):
            if t["to"] == VoiceState.PROCESSING.name and i < len(self._state_history) - 1:
                nd = self._state_history[i+1]["timestamp"] - t["timestamp"]
                proc_durations.append(nd)
        avg_proc_time = sum(proc_durations) / max(len(proc_durations), 1)

        # UI stats
        ui_stats = {
            "publish_attempts": self._publish_stats["attempts"],
            "publish_success_rate": self._publish_stats["successes"] / max(self._publish_stats["attempts"], 1),
            "publish_failure_rate": self._publish_stats["failures"] / max(self._publish_stats["attempts"], 1),
            "publish_retries": self._publish_stats["retries"],
            "last_error": self._publish_stats.get("last_error"),
            "last_successful_publish": self._publish_stats.get("last_successful_publish", 0)
        }

        # Transcript stats
        transcript_stats = {
            "transcript_sequence": self._transcript_sequence,
            "recent_processed_count": len(self._recent_processed_transcripts),
            "last_transcript_time": self._last_transcript_time
        }

        return {
            "state_durations": state_durations,
            "state_counts": state_counts,
            "interruption_rate": interruption_rate,
            "avg_processing_time": avg_proc_time,
            "total_interruptions": interruption_count,
            "total_speaking_turns": speaking_count,
            "ui_publishing": ui_stats,
            "transcript_stats": transcript_stats,
            "last_error": self._last_error
        }

    async def _publish_with_retry(self, data: bytes, description: str, max_retries: int = 3) -> bool:
        """
        Publish data to LiveKit room with retries.
        
        Args:
            data: Bytes to publish
            description: Description for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self._room or not self._room.local_participant:
            self.logger.warning(f"Cannot publish {description}: no room or local participant")
            return False
            
        # Update stats
        self._publish_stats["attempts"] += 1
        
        retries = 0
        while retries <= max_retries:
            try:
                await self._room.local_participant.publish_data(data, reliable=True)
                
                # Update success stats
                self._publish_stats["successes"] += 1
                self._publish_stats["last_successful_publish"] = time.time()
                
                if retries > 0:
                    self.logger.debug(f"Successfully published {description} after {retries} retries")
                    self._publish_stats["retries"] += retries
                    
                return True
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    # Update error stats
                    self._publish_stats["failures"] += 1
                    self._publish_stats["last_error"] = str(e)
                    
                    self.logger.error(f"Failed to publish {description} after {max_retries} attempts: {e}")
                    return False
                    
                self.logger.warning(f"Retry {retries}/{max_retries} publishing {description}: {e}")
                await asyncio.sleep(self._retry_delay * retries)  # Exponential backoff
                
        return False

    async def publish_state_update(self, state_data: dict) -> None:
        """
        Publish state update to LiveKit room.
        
        Args:
            state_data: State data to publish
        """
        try:
            if not self._room or not self._room.local_participant:
                self.logger.warning("Cannot publish state update: no room or local participant")
                return
                
            encoded_data = json.dumps({
                "type": "state_update",
                "state": state_data,
                "timestamp": time.time()
            }).encode()
            
            await self._publish_with_retry(encoded_data, "state update")
            
        except Exception as e:
            self.logger.error(f"Failed to publish state update: {e}")

    async def publish_error(self, error: str, error_type: str = "general") -> None:
        """
        Publish error to LiveKit room.
        
        Args:
            error: Error message
            error_type: Type of error
        """
        try:
            if not self._room or not self._room.local_participant:
                self.logger.warning("Cannot publish error: no room or local participant")
                return
                
            encoded_data = json.dumps({
                "type": "error",
                "error": str(error),
                "error_type": error_type,
                "timestamp": time.time()
            }).encode()
            
            await self._publish_with_retry(encoded_data, f"error: {error_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish error: {e}")
```

# stt\__init__.py

```py
"""Speech-to-Text services."""

from __future__ import annotations
from .base import STTService
from .enhanced_stt_service import EnhancedSTTService

__all__ = ['STTService', 'EnhancedSTTService']
```

# stt\base.py

```py
"""Base class for Speech-to-Text services."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from livekit import rtc

class STTService(ABC):
    """Abstract base class for Speech-to-Text services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT service."""
        pass

    @abstractmethod
    async def process_audio(self, track: Optional[rtc.AudioTrack]) -> Optional[str]:
        """Process audio from track and return transcription."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

```

# stt\enhanced_stt_service.py

```py
# enhanced_stt_service.py
# Updated for better interruption detection and speech handling

import asyncio
import logging
import time
import json
import numpy as np
import torch
from typing import Optional, Callable, Any, Dict, List, Set
import uuid
import livekit.rtc as rtc
import collections

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState

class EnhancedSTTService:
    """
    Enhanced STT service that processes audio frames and sends transcripts to the VoiceStateManager.
    """
    def __init__(
        self,
        state_manager: VoiceStateManager,
        vosk_model: str = "small",
        whisper_model: str = "small.en",
        device: str = "cpu",
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 30.0,
        energy_threshold: float = 0.05,
        on_transcript: Optional[Callable[[str], Any]] = None
    ):
        """
        Initialize the STT service.
        
        Args:
            state_manager: Voice state manager for UI updates and state coordination
            vosk_model: Vosk model size ('small', 'medium', etc.)
            whisper_model: Whisper model name ('tiny', 'base', 'small', etc.)
            device: Device to run models on ("cuda" or "cpu")
            min_speech_duration: Minimum duration of speech to process
            max_speech_duration: Maximum duration of speech before forcing processing
            energy_threshold: Threshold for speech detection (0.0-1.0)
            on_transcript: Callback when transcript is available
        """
        self.state_manager = state_manager
        self.vosk_model_name = vosk_model
        self.whisper_model_name = whisper_model
        self.device = device
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.energy_threshold = energy_threshold
        self.on_transcript = on_transcript
        
        # Initialize models to None - will be loaded in initialize()
        self.whisper_model = None
        self.vosk_model = None
        self.vosk_recognizer = None
        
        # Audio processing
        self.sample_rate = 16000  # Standard for speech models
        self.buffer = []  # Audio buffer for speech
        self.buffer_duration = 0.0  # Duration of audio in buffer
        self.is_speaking = False
        
        # Improved speech detection parameters
        self.speech_start_time = 0.0
        self.last_speech_time = 0.0
        self.speech_duration = 0.0
        self.silence_duration = 0.7  # Seconds of silence to end speech
        self.silence_start_time = 0.0
        self.noise_floor = -45.0  # dB threshold for background noise
        self.speech_threshold = -20.0  # dB threshold for speech during TTS (more aggressive)
        self.normal_speech_threshold = -30.0  # dB threshold for normal listening
        self.energy_history = []  # History of audio energy for dynamic thresholding
        self.energy_history_size = 50  # Number of frames to keep for energy history
        
        # Interruption detection parameters
        self.interruption_min_duration = 0.0  # No minimum duration
        self.interruption_max_silence = 0.02  # 20ms max silence
        self.min_speech_frames = 1  # Single frame detection
        self.energy_rise_threshold = 10  # dB rise for instant detection
        
        # Efficient thread management
        self.executor = None  # Will be initialized when needed
        self.room = None  # Store room reference for UI updates
        
        # Audio processing options
        self.vad_enabled = True  # Voice activity detection
        self.auto_threshold = True  # Auto-adjust threshold based on environment
        self.dynamic_silence = True  # Adjust silence duration based on speech context
        
        # Transcript tracking with improved deduplication
        self.last_transcript = ""
        self.transcript_hash_set = set()  # Store normalized transcript hashes
        self.recent_transcripts = []  # List of (text, timestamp) tuples for recent transcripts
        self.max_recent_transcripts = 5
        self.last_partial = ""
        self.last_partial_time = 0.0
        self.partial_cooldown = 0.3  # seconds between UI partial updates
        
        # Recognition stats
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.error_count = 0
        self.avg_processing_time = 0.0
        self.total_processing_time = 0.0
        self.interruptions_detected = 0
        
        # Active processing tracking
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        self.audio_stream = None
        self.active_task = None
        
        # Store participant identity for transcript attribution
        self._participant_identity = None
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize Vosk and Whisper models."""
        try:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=1)
            
            # Initialize Whisper if specified
            if self.whisper_model_name:
                try:
                    import whisper
                    self.logger.info(f"Loading Whisper model '{self.whisper_model_name}' on {self.device}...")
                    self.whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
                    self.logger.info("Whisper model loaded successfully")
                except ImportError:
                    self.logger.warning("Whisper not installed, skipping Whisper initialization")
            
            # Initialize Vosk if specified
            if self.vosk_model_name:
                try:
                    from vosk import Model, KaldiRecognizer, SetLogLevel
                    SetLogLevel(-1)  # Reduce Vosk verbosity
                    
                    # Find Vosk model path - simplified for this example
                    import os
                    model_path = os.path.join('models', 'vosk', 'en-us')
                    if os.path.exists(model_path):
                        self.logger.info(f"Loading Vosk model from: {model_path}")
                        self.vosk_model = Model(model_path)
                        self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                        self.logger.info("Vosk model loaded successfully")
                    else:
                        self.logger.warning(f"Vosk model path not found: {model_path}")
                except ImportError:
                    self.logger.warning("Vosk not installed, skipping Vosk initialization")
            
            self.logger.info("STT service initialized successfully")
            
            # Send initialization message to UI if room available
            if self.room and self.state_manager:
                try:
                    await self.state_manager._publish_with_retry(
                        json.dumps({
                            "type": "stt_initialized",
                            "models": {
                                "vosk": self.vosk_model_name,
                                "whisper": self.whisper_model_name
                            },
                            "device": self.device,
                            "timestamp": time.time()
                        }).encode(),
                        "STT initialization"
                    )
                    
                    # Ensure state manager is in LISTENING state if not already in another state
                    if self.state_manager.current_state not in [VoiceState.SPEAKING, VoiceState.PROCESSING]:
                        await self.state_manager.transition_to(VoiceState.LISTENING)
                    
                    self.logger.debug("Published listening state to UI")
                except Exception as e:
                    self.logger.error(f"Failed to publish initialization message: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize STT service: {e}", exc_info=True)
            raise
            
    def set_room(self, room) -> None:
        """Set the LiveKit room for UI updates."""
        self.room = room

    async def process_audio(self, track) -> Optional[str]:
        """
        Process audio from a LiveKit track and return transcription.
        
        Args:
            track: LiveKit audio track
            
        Returns:
            Transcribed text if speech detected and processed
        """
        # Log track details for debugging
        self.logger.info(f"Processing audio track: sid={track.sid if hasattr(track, 'sid') else 'unknown'}")
        
        if not self.vosk_recognizer and not self.whisper_model:
            self.logger.error("STT not fully initialized")
            return None
        
        # Ensure we're not already processing an audio stream
        async with self.processing_lock:
            if self.is_processing:
                self.logger.warning("Already processing audio, skipping this call")
                return None
            
            self.is_processing = True
        
        # Create cleanup function
        async def cleanup():
            async with self.processing_lock:
                self.is_processing = False
                self.audio_stream = None
                self.active_task = None
                self.potential_interruption = False
        
        try:
            # Determine participant identity with fallback mechanisms
            participant_identity = None
            
            # Method 1: Direct from track.participant (standard approach)
            if hasattr(track, 'participant') and track.participant:
                participant_identity = track.participant.identity
                self.logger.debug(f"Got participant identity directly: {participant_identity}")
            
            # Method 2: Fallback - search in room.remote_participants by track SID
            if not participant_identity and self.room and hasattr(track, 'sid'):
                track_sid = track.sid
                self.logger.warning(f"track.participant is None, attempting fallback via track SID: {track_sid}")
                
                for participant in self.room.remote_participants.values():
                    for pub in participant.track_publications.values():
                        if pub.track and pub.track.sid == track_sid:
                            participant_identity = participant.identity
                            self.logger.info(f"Found participant identity via track SID lookup: {participant_identity}")
                            break
                    if participant_identity:
                        break
            
            # Fallback for debugging/logging
            if not participant_identity:
                self.logger.warning(f"Could not determine participant identity for track {track.sid if hasattr(track, 'sid') else 'unknown'}")
                participant_identity = "unknown_user"
            
            # Store the participant identity for transcript attribution
            self._participant_identity = participant_identity
            
            self.logger.info(f"Processing audio from participant: {participant_identity}")
            
            # Create an AudioStream to receive frames
            self.audio_stream = rtc.AudioStream(track)
            
            # Announce listening to UI only if not already speaking
            if self.room and self.state_manager and self.state_manager.current_state != VoiceState.SPEAKING:
                try:
                    await self.state_manager._publish_with_retry(
                        json.dumps({
                            "type": "listening_state",
                            "active": True,
                            "timestamp": time.time()
                        }).encode(),
                        "listening state"
                    )
                    
                    # Ensure state manager is in LISTENING state if not already in another state
                    if self.state_manager.current_state not in [VoiceState.SPEAKING, VoiceState.PROCESSING]:
                        await self.state_manager.transition_to(VoiceState.LISTENING)
                    
                    self.logger.debug("Published listening state to UI")
                except Exception as e:
                    self.logger.error(f"Failed to publish listening state: {e}")
            
            # Process audio frames
            async for event in self.audio_stream:
                # Check if we should stop processing (error state)
                if self.state_manager.current_state == VoiceState.ERROR:
                    self.logger.info(f"Stopping audio processing due to ERROR state")
                    await cleanup()
                    break
                    
                frame = event.frame
                if frame is None:
                    continue
                    
                # Convert to numpy array and normalize
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                # Resample if needed
                if frame.sample_rate != self.sample_rate:
                    audio_data = self._resample(audio_data, frame.sample_rate, self.sample_rate)
                
                # Calculate audio level - always do this for logging
                audio_level = 20 * np.log10(np.mean(np.abs(audio_data)) + 1e-9)
                
                # Create a safe value for JSON
                audio_level_float = float(audio_level)
                
                # Check if we're currently speaking (TTS active)
                if self.state_manager.current_state == VoiceState.SPEAKING:
                    # When speaking, we use a higher threshold to detect interruptions
                    # This avoids the system responding to its own TTS output
                    interruption_threshold = self.speech_threshold
                    
                    # Check if audio is loud enough to be an interruption
                    if audio_level > interruption_threshold:
                        # Found potential interruption - add to buffer
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.potential_interruption = True
                            self.speech_start_time = time.time()
                            self.buffer = []
                            self.buffer_duration = 0.0
                            
                            self.logger.info(f"Potential interruption detected at {audio_level_float:.1f}dB")
                        
                        # Update speech state
                        self.last_speech_time = time.time()
                        self.silence_start_time = 0
                        
                        # Add to buffer
                        self.buffer.append(audio_data)
                        frame_duration = len(audio_data) / self.sample_rate
                        self.buffer_duration += frame_duration
                        
                        # If we have collected enough speech data to be confident it's an interruption
                        if self.buffer_duration >= self.interruption_min_duration:
                            if self.potential_interruption:
                                self.logger.info(f"Interruption confirmed after {self.buffer_duration:.2f}s")
                                
                                # Signal interruption to state manager
                                self.interruptions_detected += 1
                                await self.state_manager.handle_user_speech_detected()
                                
                                # Attempt to get transcript from buffer
                                interrupt_text = await self._finalize_transcript(participant_identity)
                                if interrupt_text:
                                    self.logger.info(f"Interruption transcript: '{interrupt_text}'")
                                
                                # Mark as no longer just a potential interruption
                                self.potential_interruption = False
                    else:
                        # Not loud enough to be interruption
                        # If we were detecting potential interruption, check if it was just noise
                        if self.is_speaking and self.potential_interruption:
                            # Check if we've been silent for long enough to end potential interruption
                            if self.silence_start_time == 0:
                                self.silence_start_time = time.time()
                            
                            silence_duration = time.time() - self.silence_start_time
                            if silence_duration >= 0.3:  # 300ms of silence to end interruption detection
                                # Reset state - wasn't a real interruption
                                self.is_speaking = False
                                self.potential_interruption = False
                                self.buffer = []
                                self.buffer_duration = 0.0
                                self.silence_start_time = 0
                                self.logger.info("Potential interruption was just noise, ignored")
                else:
                    # Normal audio processing (not during TTS)
                    transcript = await self._process_audio_buffer(audio_data, participant_identity)
                    if transcript:
                        # Publish transcript completion to UI
                        if self.room and self.state_manager:
                            try:
                                await self._publish_transcript_to_ui(transcript, is_final=True)
                            except Exception as e:
                                self.logger.error(f"Failed to publish transcript completion: {e}")
                        
                        # Update statistics
                        self.successful_recognitions += 1
                        
                        # Hash this transcript
                        hash_value = self._get_transcript_hash(transcript)
                        
                        # Add transcript and timestamp to recent list
                        now = time.time()
                        self.recent_transcripts.append((transcript, now))
                        
                        # Trim list if needed
                        if len(self.recent_transcripts) > self.max_recent_transcripts:
                            self.recent_transcripts.pop(0)
                        
                        # Check if this is very recent duplicate (within 2 seconds)
                        is_duplicate = False
                        for text, timestamp in self.recent_transcripts[:-1]:  # Skip the one we just added
                            if now - timestamp < 2.0 and self._similarity(text, transcript) > 0.8:
                                is_duplicate = True
                                break
                        
                        # Only process if not a recent duplicate
                        if is_duplicate:
                            self.logger.warning(f"Detected duplicate transcript within 2 seconds, skipping: '{transcript[:30]}...'")
                        else:
                            # Forward transcript to state manager
                            self.transcript_hash_set.add(hash_value)
                            self.last_transcript = transcript
                            await self.state_manager.handle_stt_transcript(transcript)
            
            # If we exit the loop normally, clean up
            await cleanup()
            return None
                
        except asyncio.CancelledError:
            self.logger.info("Audio processing task cancelled")
            await cleanup()
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}", exc_info=True)
            
            # Record error in statistics
            self.error_count += 1
            
            # Register error with state manager
            if self.state_manager:
                await self.state_manager.register_error(e, "stt_processing")
                
            await cleanup()
            return None

    async def _process_audio_buffer(self, audio_data: np.ndarray, participant_identity: str) -> Optional[str]:
        """
        Process audio buffer with improved speech detection during TTS playback.
        """
        if audio_data is None or len(audio_data) == 0:
            return None
            
        if self.state_manager.current_state == VoiceState.ERROR:
            return None
            
        try:
            current_time = time.time()
            
            # Calculate audio level in dB
            audio_level = 20 * np.log10(np.mean(np.abs(audio_data)) + 1e-10)
            
            # Add to energy history for dynamic thresholding
            self.energy_history.append(audio_level)
            if len(self.energy_history) > self.energy_history_size:
                self.energy_history.pop(0)
                
            # Dynamic threshold adjustment
            if self.auto_threshold and len(self.energy_history) >= 10:
                sorted_energies = sorted(self.energy_history)
                noise_floor = sorted_energies[int(len(sorted_energies) * 0.1)]  # 10th percentile
                self.noise_floor = max(noise_floor, -55)  # Don't go below reasonable floor
                self.speech_threshold = self.noise_floor + 10  # 10 dB above noise floor
            
            # Use lower threshold during TTS to improve interruption detection
            effective_threshold = self.speech_threshold
            if self.state_manager.current_state == VoiceState.SPEAKING:
                effective_threshold = self.speech_threshold - 5  # 5dB lower threshold
                
            # Check for speech using energy threshold
            is_speech = audio_level > effective_threshold
            
            if is_speech:
                # First speech detection
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    self.buffer = []  # Clear buffer
                    self.buffer_duration = 0.0
                    
                    # Publish speech start
                    if self.room and self.room.local_participant:
                        try:
                            await self.room.local_participant.publish_data(
                                json.dumps({
                                    "type": "speech_detected",
                                    "level": float(audio_level),
                                    "timestamp": current_time
                                }).encode(),
                                reliable=True
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to publish speech detection: {e}")
                
                # Update speech timing
                self.last_speech_time = current_time
                self.silence_start_time = 0.0
                
                # Add audio to buffer
                self.buffer.append(audio_data)
                frame_duration = len(audio_data) / self.sample_rate
                self.buffer_duration += frame_duration
                
                # Fast interruption handling during SPEAKING state
                if (self.state_manager.current_state == VoiceState.SPEAKING and 
                    self.buffer_duration >= 0.3):  # 300ms is enough for interruption
                    self.logger.info(f"Potential interruption detected during TTS, level: {audio_level:.1f}dB")
                    
                    # Signal interruption immediately
                    await self.state_manager.handle_user_speech_detected()
                    
                # Check if max duration exceeded
                speech_duration = current_time - self.speech_start_time
                if speech_duration >= self.max_speech_duration:
                    self.logger.info(f"Max speech duration ({self.max_speech_duration}s) reached")
                    return await self._finalize_transcript(participant_identity)
                    
            else:  # Not speech (silence)
                if self.is_speaking:
                    # Start silence timing if needed
                    if self.silence_start_time == 0:
                        self.silence_start_time = current_time
                        
                    # Check if silence duration exceeded
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.silence_duration:
                        self.logger.info(f"Silence timeout ({self.silence_duration}s) reached")
                        return await self._finalize_transcript(participant_identity)
                        
            return None  # No complete transcript yet
            
        except Exception as e:
            self.logger.error(f"Error processing audio buffer: {e}")
            return None

    async def _publish_transcript_to_ui(self, text: str, is_final: bool = False) -> None:
        """
        Publish transcript to LiveKit UI with proper formatting.
        
        Args:
            text: The transcribed text
            is_final: Whether this is a final transcription
        """
        if not self.room or not self.room.local_participant:
            return
            
        try:
            # Use state manager for consistent publishing
            self.total_recognitions += 1
            
            # Pass the participant identity explicitly to ensure proper attribution
            await self.state_manager.publish_transcription(
                text,
                "user",
                is_final, 
                participant_identity=self._participant_identity
            )
            
            # Log success
            self.logger.info(f"Published {'final' if is_final else 'partial'} transcript for {self._participant_identity}: '{text[:50]}...'")
                
        except Exception as e:
            self.logger.error(f"Failed to publish transcript to UI: {e}", exc_info=True)

    async def _finalize_transcript(self, participant_identity: str) -> str:
        """
        Finalize transcript using Whisper for high quality results.
        
        Returns:
            Final transcribed text
        """
        # Skip if buffer is empty
        if not self.buffer or len(self.buffer) == 0:
            self.is_speaking = False
            self.potential_interruption = False
            return None
            
        try:
            # Check speech duration against minimum threshold
            speech_duration = time.time() - self.speech_start_time
            if not self.buffer or speech_duration < self.min_speech_duration:
                self.logger.warning(f"Speech too short: {self.buffer_duration:.2f}s < {self.min_speech_duration:.2f}s")
                # Clear buffer and reset state to avoid repeated processing
                self.buffer = []
                self.buffer_duration = 0.0
                self.is_speaking = False
                self.potential_interruption = False
                return None
                
            # Reset speaking state
            self.is_speaking = False
            self.potential_interruption = False
            speech_duration = time.time() - self.speech_start_time
            self.logger.info(f"Processing speech segment: {speech_duration:.2f}s, {len(self.buffer)} chunks")
            
            # Publish speech end
            if self.room and self.room.local_participant:
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            "type": "speech_end",
                            "duration": float(speech_duration),  # Convert to native Python float
                            "timestamp": time.time()
                        }).encode(),
                        reliable=True
                    )
                except Exception as e:
                    self.logger.error(f"Failed to publish speech end: {e}")
            
            # Update state manager - but only if we're not already in PROCESSING state
            if self.state_manager and self.state_manager.current_state != VoiceState.PROCESSING:
                await self.state_manager.transition_to(
                    VoiceState.PROCESSING,
                    {"duration": float(speech_duration)}  # Convert to native Python float
                )
            
            # Concatenate buffer
            audio_array = np.concatenate(self.buffer)
            
            # Track processing time
            processing_start = time.time()
            
            # Primary transcription via Vosk for faster results if available
            text = ""
            if self.vosk_recognizer:
                try:
                    # Convert to int16 for Vosk
                    audio_int16 = (audio_array * 32768.0).astype(np.int16)
                    
                    # Reset recognizer to clean state
                    self.vosk_recognizer.Reset()
                    
                    # Process the full audio
                    self.vosk_recognizer.AcceptWaveform(audio_int16.tobytes())
                    vosk_result = json.loads(self.vosk_recognizer.FinalResult())
                    
                    # Get text
                    text = vosk_result.get("text", "").strip()
                    self.logger.info(f"Vosk transcription: '{text}'")
                    
                    # If Vosk gave us a good result, use it directly
                    if text and len(text.split()) >= 2:  # At least 2 words
                        processing_time = time.time() - processing_start
                        self.logger.info(f"Using Vosk transcription, processed in {processing_time:.2f}s")
                        
                        # Update processing stats
                        self.total_processing_time += processing_time
                        self.total_recognitions += 1
                        self.avg_processing_time = self.total_processing_time / self.total_recognitions
                        
                        # Clear buffer
                        self.buffer = []
                        self.buffer_duration = 0.0
                        
                        return text
                
                except Exception as e:
                    self.logger.error(f"Error in Vosk transcription: {e}")
            
            # Fall back to Whisper for more accurate results or if Vosk failed
            if not text and self.whisper_model and self.executor:
                loop = asyncio.get_event_loop()
                try:
                    self.logger.info("Using Whisper for transcription")
                    result = await loop.run_in_executor(
                        self.executor,
                        self._whisper_transcribe,
                        audio_array
                    )
                    
                    if result:
                        text = result.get("text", "").strip()
                        self.logger.info(f"Whisper transcription: '{text}'")
                    
                except Exception as e:
                    self.logger.error(f"Error in Whisper transcription: {e}")
            
            # If still no result, use a fallback
            if not text and not self.whisper_model and not self.vosk_recognizer:
                self.logger.warning("No transcription models available")
                text = ""
            
            # Update processing stats
            processing_time = time.time() - processing_start
            self.total_processing_time += processing_time
            self.total_recognitions += 1
            self.avg_processing_time = self.total_processing_time / self.total_recognitions
            
            if text:
                self.logger.info(f"Final transcript: '{text}'")
                self.last_transcript = text
            else:
                self.logger.warning("No transcription result")
            
            # Clear buffer
            self.buffer = []
            self.buffer_duration = 0.0
            
            # Publish transcript to UI
            await self._publish_transcript_to_ui(text, is_final=True)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error finalizing transcript: {e}", exc_info=True)
            
            # Record error
            self.error_count += 1
            
            # Register error with state manager
            if self.state_manager:
                await self.state_manager.register_error(e, "stt_finalization")
                
            self.buffer = []  # Clear buffer
            self.buffer_duration = 0.0
            self.is_speaking = False
            self.potential_interruption = False
            return ""

    async def clear_buffer(self) -> None:
        """Clear audio buffer and reset speech detection state."""
        try:
            self.buffer = []
            self.buffer_duration = 0.0
            self.is_speaking = False
            self.potential_interruption = False
            self.last_partial = ""
            self.logger.debug("STT buffer cleared")
        except Exception as e:
            self.logger.error(f"Error clearing STT buffer: {e}")
            
    def _get_transcript_hash(self, text: str) -> str:
        """Get a hash of the normalized transcript for deduplication."""
        if not text:
            return ""
        # Normalize: lowercase, remove punctuation, extra whitespace
        import re
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Return normalized text as the "hash"
        return normalized
        
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between 0.0 and 1.0."""
        if not text1 or not text2:
            return 0.0
            
        # Convert to sets of words
        words1 = set(self._get_transcript_hash(text1).split())
        words2 = set(self._get_transcript_hash(text2).split())
        
        if not words1 or not words2:
            return 0.0
            
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    async def stop_processing(self) -> None:
        """Stop any ongoing audio processing."""
        # Cancel the active audio processing task
        async with self.processing_lock:
            if self.active_task and not self.active_task.done():
                self.active_task.cancel()
                try:
                    await asyncio.wait_for(self.active_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    self.logger.error(f"Error cancelling active task: {e}")
                finally:
                    self.active_task = None
                    
            # Close audio stream
            if self.audio_stream:
                self.audio_stream = None
                
            self.is_processing = False
            self.potential_interruption = False
            
        # Clear buffer and reset state
        await self.clear_buffer()
        
        self.logger.info("Audio processing stopped")

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up STT service")
        
        # Stop processing first
        await self.stop_processing()
        
        # Publish cleanup to UI
        if self.room and self.state_manager:
            try:
                await self.state_manager._publish_with_retry(
                    json.dumps({
                        "type": "stt_cleanup",
                        "timestamp": time.time()
                    }).encode(),
                    "STT cleanup"
                )
            except Exception as e:
                self.logger.error(f"Failed to publish cleanup: {e}")
                
        # Shut down executor
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        # Release models
        self.whisper_model = None
        self.vosk_model = None
        self.vosk_recognizer = None
        
        # Clear buffer and state
        self.buffer = []
        self.buffer_duration = 0.0
        self.is_speaking = False
        self.potential_interruption = False
        self.transcript_hash_set.clear()
        self.recent_transcripts.clear()
        
        # Clear participant identity
        self._participant_identity = None
        
        # Free CUDA memory if needed
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass
                
        self.logger.info("STT service cleanup complete")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get speech recognition stats."""
        # Convert any numpy types to native Python types
        return {
            "total_recognitions": int(self.total_recognitions),
            "successful_recognitions": int(self.successful_recognitions),
            "error_count": int(self.error_count),
            "avg_processing_time": float(self.avg_processing_time),
            "success_rate": float(self.successful_recognitions) / max(int(self.total_recognitions), 1),
            "current_speech_threshold": float(self.speech_threshold),
            "current_noise_floor": float(self.noise_floor),
            "interruptions_detected": int(self.interruptions_detected)
        }

    def _resample(self, audio_data: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio data from source rate to target rate.
        
        Args:
            audio_data: Input audio data as numpy array
            src_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data as numpy array
        """
        if src_rate == target_rate:
            return audio_data
            
        # Calculate resampling ratio
        ratio = target_rate / src_rate
        
        # Calculate new length
        new_length = int(len(audio_data) * ratio)
        
        try:
            # Use scipy for high quality resampling
            import scipy.signal
            resampled = scipy.signal.resample(audio_data, new_length)
            
        except ImportError:
            # Fallback to linear interpolation if scipy not available
            self.logger.warning("scipy not available, using linear interpolation for resampling")
            
            # Create time arrays for interpolation
            old_time = np.linspace(0, 1, len(audio_data))
            new_time = np.linspace(0, 1, new_length)
            
            # Linear interpolation
            resampled = np.interp(new_time, old_time, audio_data)
        
        # Ensure output is float32
        return resampled.astype(np.float32)

    def _whisper_transcribe(self, audio_array: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio using Whisper model (runs in thread pool).
        
        Args:
            audio_array: Audio samples as numpy array
            
        Returns:
            Dictionary with transcription results
        """
        try:
            import tempfile
            import wave
            import os
            
            # Create temporary WAV file for best compatibility
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write audio to WAV file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_array * 32768.0).astype(np.int16).tobytes())
            
            try:
                # Use Whisper to transcribe
                result = self.whisper_model.transcribe(
                    temp_path,
                    language="en",  # Adjust as needed
                    fp16=False      # Avoid precision issues
                )
                return result
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error in Whisper transcription: {e}", exc_info=True)
            return None
```

# test_transcript_identity.py

```py
import asyncio
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
import livekit.rtc as rtc

class MockTrackPublication:
    def __init__(self, sid="mock-track-sid", kind=rtc.TrackKind.KIND_AUDIO):
        self.sid = sid
        self.kind = kind

class MockLocalParticipant:
    def __init__(self, identity="mock-participant"):
        self.identity = identity
        self.track_publications = {
            "track1": MockTrackPublication()
        }
        self.publish_transcription = AsyncMock()
        self.publish_data = AsyncMock(return_value=True)

class MockRoom:
    def __init__(self, participant_identity="mock-participant"):
        self.local_participant = MockLocalParticipant(identity=participant_identity)
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED

async def test_transcript_identity():
    """Test transcript identity attribution with various scenarios"""
    logger = logging.getLogger("test_transcript_identity")
    logger.info("Creating VoiceStateManager")
    
    # Create the voice state manager
    mgr = VoiceStateManager(debug=True)
    
    # Set up mock room with mock local participant
    mock_room = MockRoom(participant_identity="test-room-participant")
    mgr._room = mock_room
    
    # Test 1: With explicit participant identity
    logger.info("Test 1: Explicit participant identity")
    await mgr.publish_transcription(
        "Test transcript with explicit identity", 
        "user", 
        True, 
        participant_identity="test-user-explicit"
    )
    
    # Verify explicit identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Transcription API - participant_identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-user-explicit", f"Expected 'test-user-explicit', got '{trans.participant_identity}'"
    
    # Verify explicit identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel - participant_identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-user-explicit", f"Expected 'test-user-explicit', got '{data_json['participant_identity']}'"
    
    # Reset mocks
    mock_room.local_participant.publish_transcription.reset_mock()
    mock_room.local_participant.publish_data.reset_mock()
    
    # Test 2: Without explicit participant identity (should use local participant identity)
    logger.info("Test 2: Implicit identity from local participant")
    await mgr.publish_transcription(
        "Test transcript without explicit identity", 
        "user", 
        True
    )
    
    # Verify local participant identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Transcription API - implicit identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-room-participant", f"Expected 'test-room-participant', got '{trans.participant_identity}'"
    
    # Verify local participant identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel - implicit identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-room-participant", f"Expected 'test-room-participant', got '{data_json['participant_identity']}'"
    
    # Reset mocks
    mock_room.local_participant.publish_transcription.reset_mock()
    mock_room.local_participant.publish_data.reset_mock()
    
    # Test 3: Different sender types (assistant vs user)
    logger.info("Test 3: Different sender types (assistant)")
    await mgr.publish_transcription(
        "Test transcript from assistant", 
        "assistant", 
        True, 
        participant_identity="test-assistant"
    )
    
    # Verify assistant identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Assistant transcript - identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-assistant", f"Expected 'test-assistant', got '{trans.participant_identity}'"
    
    # Verify assistant identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel (assistant) - identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-assistant", f"Expected 'test-assistant', got '{data_json['participant_identity']}'"
    assert data_json['sender'] == "assistant", f"Expected sender 'assistant', got '{data_json['sender']}'"
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_transcript_identity())

```

# tests\__init__.py

```py


```

# tests\conftest.py

```py
import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture(autouse=True)
def mock_cuda_available():
    """Force CPU usage for consistent testing"""
    import torch
    torch.cuda.is_available = lambda: False

```

# tests\test_enhanced_stt_pipeline.py

```py
import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from voice_core.stt.enhanced_stt_pipeline import EnhancedSTTPipeline, WhisperConfig
from voice_core.utils.audio_buffer import AudioBuffer

@pytest.fixture
def whisper_config():
    return WhisperConfig(
        model_name="tiny.en",  # Use tiny model for faster tests
        device="cpu",  # Force CPU for consistent testing
        initial_silence_threshold=-35.0,
        min_audio_length=0.7,
        min_speech_duration=0.5,
        silence_duration=1.0
    )

@pytest.fixture
def mock_audio_frame():
    # Create synthetic audio data (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    # Normalize and convert to int16
    audio = (audio * 32767).astype(np.int16)
    
    # Mock LiveKit frame
    frame = Mock()
    frame.data = audio.tobytes()
    frame.sample_rate = sample_rate
    frame.num_channels = 1
    return frame

@pytest.fixture
def mock_audio_track():
    track = Mock()
    return track

@pytest.fixture
def mock_audio_stream(mock_audio_frame):
    class MockAudioStream:
        def __init__(self, track):
            self.track = track
            self.frame = mock_audio_frame
            
        async def __aiter__(self):
            # Yield 3 frames to simulate continuous audio
            for _ in range(3):
                yield Mock(frame=self.frame)
                await asyncio.sleep(0.1)
    
    return MockAudioStream

@pytest.mark.asyncio
async def test_pipeline_initialization(whisper_config):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    assert pipeline.model is not None
    assert pipeline.audio_buffer is not None
    assert isinstance(pipeline.audio_buffer, AudioBuffer)
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_audio_preprocessing(whisper_config, mock_audio_frame):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Convert frame to numpy array
    audio_data = np.frombuffer(mock_audio_frame.data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Test preprocessing
    processed_audio = pipeline.audio_processor.preprocess(audio_data)
    assert processed_audio is not None
    assert isinstance(processed_audio, np.ndarray)
    assert not np.any(np.isnan(processed_audio))
    assert np.max(np.abs(processed_audio)) <= 1.0
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_speech_detection(whisper_config, mock_audio_track, mock_audio_stream):
    with patch('livekit.rtc.AudioStream', mock_audio_stream):
        pipeline = EnhancedSTTPipeline(whisper_config)
        await pipeline.initialize()
        
        transcripts = []
        async for transcript in pipeline.process_audio(mock_audio_track):
            transcripts.append(transcript)
            if len(transcripts) >= 1:  # Get at least one transcript
                break
        
        # Verify speech detection and transcription
        assert pipeline.is_speaking is False  # Should reset after processing
        assert len(pipeline.speech_energy_history) > 0
        assert pipeline.audio_buffer is not None
        
        await pipeline.cleanup()

@pytest.mark.asyncio
async def test_noise_reduction(whisper_config, mock_audio_frame):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Create noisy audio
    audio_data = np.frombuffer(mock_audio_frame.data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    noise = np.random.normal(0, 0.1, len(audio_data))
    noisy_audio = audio_data + noise
    
    # Test noise reduction
    processed_audio = pipeline.audio_processor._apply_noise_reduction(noisy_audio)
    assert processed_audio is not None
    assert np.std(processed_audio) < np.std(noisy_audio)  # Verify noise reduction
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_pipeline_shutdown(whisper_config):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test graceful shutdown
    assert pipeline._shutdown is False
    await pipeline.stop()
    assert pipeline._shutdown is True
    assert pipeline._heartbeat_task is None
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_energy_score_calculation(whisper_config):
    """Test energy score calculation with different audio levels."""
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test cases with different amplitudes and frequencies
    test_cases = [
        # Quiet speech (-20dB)
        {
            'amplitude': 0.1,
            'freq': 200,
            'expected_min': 0.1
        },
        # Normal speech (-10dB)
        {
            'amplitude': 0.3,
            'freq': 1000,
            'expected_min': 0.3
        },
        # Loud speech (-3dB)
        {
            'amplitude': 0.7,
            'freq': 2000,
            'expected_min': 0.5
        }
    ]
    
    for case in test_cases:
        # Generate test signal
        duration = 0.5
        t = np.linspace(0, duration, int(16000 * duration))
        signal = case['amplitude'] * np.sin(2 * np.pi * case['freq'] * t)
        
        # Calculate energy score
        raw_level = 20 * np.log10(np.sqrt(np.mean(signal**2)) + 1e-10)
        energy_score = pipeline._calculate_speech_energy_score(signal, raw_level)
        
        # Verify score is non-zero and scales with amplitude
        assert energy_score > 0, f"Energy score should be > 0 for amplitude {case['amplitude']}"
        assert energy_score >= case['expected_min'], f"Energy score {energy_score} below expected minimum {case['expected_min']}"
        
        print(f"Test case - Amplitude: {case['amplitude']}, Freq: {case['freq']}Hz")
        print(f"Raw Level: {raw_level:.1f}dB, Energy Score: {energy_score:.3f}")
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_voice_detection_thresholds(whisper_config):
    """Test voice detection with various signal levels."""
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test cases with different signal levels
    test_cases = [
        # Very quiet (-30dB) - should not trigger
        {
            'filtered_db': -30,
            'raw_level': -35,
            'expected_detection': False
        },
        # Moderate (-15dB) - should trigger
        {
            'filtered_db': -15,
            'raw_level': -20,
            'expected_detection': True
        },
        # Loud (-5dB) - should definitely trigger
        {
            'filtered_db': -5,
            'raw_level': -10,
            'expected_detection': True
        }
    ]
    
    for case in test_cases:
        # Generate test signal
        duration = 0.5
        t = np.linspace(0, duration, int(16000 * duration))
        amplitude = 10 ** (case['raw_level'] / 20)
        signal = amplitude * np.sin(2 * np.pi * 1000 * t)
        
        # Calculate energy score
        energy_score = pipeline._calculate_speech_energy_score(signal, case['raw_level'])
        
        # Test voice detection
        is_voice = pipeline._detect_voice(
            case['filtered_db'],
            case['raw_level'],
            signal,
            energy_score
        )
        
        assert is_voice == case['expected_detection'], \
            f"Voice detection failed for filtered_db: {case['filtered_db']}, " \
            f"raw_level: {case['raw_level']}, energy_score: {energy_score:.3f}"
        
        print(f"Test case - Filtered: {case['filtered_db']}dB, Raw: {case['raw_level']}dB")
        print(f"Energy Score: {energy_score:.3f}, Detection: {is_voice}")
    
    await pipeline.cleanup()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

```

# TRANSCRIPT_IDENTITY_README.md

```md
## 1. Background: Why Were User Transcriptions Attributed to the Agent?

1. **LiveKit Identifies Speech by Track Owner**  
   When an audio track is published, LiveKit labels the resulting transcriptions with the participant identity of the publisher. If the agent’s code was publishing the user’s microphone track (or always used the agent’s identity for transcripts), the UI naturally labeled all speech as coming from the agent.

2. **STT Code Using the Agent’s Identity**  
   The `VoiceStateManager` and `EnhancedSTTService` were always defaulting to the **agent** participant identity when publishing transcriptions via:
   \`\`\`python
   participant_identity = self._room.local_participant.identity
   \`\`\`
   or simply using `"user"` as a hard-coded label. As a result, the LiveKit Playground saw all transcriptions as belonging to the agent.

---

## 2. High-Level Fix

### **A. Track the Actual User Identity**

- When the STT service subscribes to a user’s audio track, it now attempts to find the **real** participant identity by:
  1. Checking `track.participant.identity` directly, or  
  2. Falling back to searching `room.remote_participants` by track SID.  
- The STT code stores this identity in a new `_participant_identity` field.

### **B. Publish Transcriptions With That Identity**

- When the STT service finishes recognizing speech, it calls `_publish_transcript_to_ui(...)`.
- That method calls `VoiceStateManager.publish_transcription(...)`, now passing the **user’s** identity instead of the agent’s identity:
  \`\`\`python
  await self.state_manager.publish_transcription(
      text,
      "user",
      is_final=True,
      participant_identity=self._participant_identity
  )
  \`\`\`
- `VoiceStateManager.publish_transcription(...)` then uses this identity both in the data channel message (the JSON that says `"type": "transcript", "participant_identity": ...`) and in the Transcription API object (`rtc.Transcription(participant_identity=...)`).

### **C. Result**

- The LiveKit Playground sees a transcription message (or Transcription API call) that says **participant_identity="playground-user"** (or whatever the user identity is). Hence, user speech is labeled as user speech.

---

## 3. What Changed in the Code

**Key modifications** in two files:

1. **`voice_core/stt/enhanced_stt_service.py`**  
   - In `process_audio(...)`, we detect the real participant identity from the remote track or from the fallback.  
   - We store it in `self._participant_identity`.  
   - In `_publish_transcript_to_ui(...)`, we pass `self._participant_identity` to the state manager’s `publish_transcription`.

2. **`voice_core/state/voice_state_manager.py`**  
   - In `publish_transcription(...)`, we add a parameter `participant_identity` and ensure it overrides the default local participant identity.  
   - We then pass that identity to the data channel JSON (`"participant_identity": identity_to_use`) and to the Transcription API (`rtc.Transcription(participant_identity=identity_to_use, ...)`).

The net effect is that **the user’s identity** is always used for user transcriptions, ensuring the UI sees it as user speech.

---

## 4. Summary of the “Breakthrough”

- **We needed to figure out** that the root cause was the agent code was not using the user’s identity for transcriptions.  
- **The fix**: capture the user’s actual identity from the remote audio track, then explicitly pass that identity into both the data channel message and the Transcription API.  
- As soon as we do that, LiveKit labels the user’s speech with the correct identity, and the Playground UI shows “User” (or “playground-user”) instead of the agent.

---

**That’s the essence of the breakthrough:** **publish** user speech with **user** identity rather than always using the agent identity. Once the code tracks and uses `_participant_identity` for each transcript, LiveKit properly attributes speech in the Playground.
```

# tts.py

```py
import asyncio
import edge_tts
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import logging
import livekit
from livekit import Room, RoomOptions, LocalTrack, TrackKind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILE = "/tmp/test.mp3"
app = Flask(__name__)
CORS(app, supports_credentials=True)

class LiveKitTTS:
    def __init__(self, url: str = "ws://localhost:7880", api_key: str = "devkey", api_secret: str = "secret"):
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room = None
        self.participant = None
        self.audio_track = None
        logger.info("Initialized LiveKitTTS")
        
    async def connect(self, room_name: str):
        """Connect to LiveKit room for TTS"""
        try:
            options = RoomOptions(
                auto_subscribe=False,  # We don't need to subscribe to other participants
                adaptive_stream=False,
                dynacast=False
            )
            
            self.room = Room(options=options)
            await self.room.connect(
                self.url,
                token=self.api_key,  # Using api_key as token for development
                participant_name="tts_service",
                room_name=room_name
            )
            
            self.participant = self.room.local_participant
            self.audio_track = await LocalTrack.create_audio_track("tts")
            await self.participant.publish_track(self.audio_track)
            logger.info(f"Connected to LiveKit room {room_name} for TTS")
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            raise
            
    async def synthesize_speech(self, text: str, voice: str = "en-US-AvaMultilingualNeural"):
        """Synthesize speech and stream it through LiveKit"""
        if not self.audio_track:
            logger.error("No audio track available. Make sure to connect first.")
            return
            
        try:
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Convert audio chunk to proper format for LiveKit
                    await self.audio_track.write_frame(chunk["data"])
                    
            logger.info("Speech synthesis completed")
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from LiveKit room"""
        if self.room:
            await self.room.disconnect()
            self.room = None
            self.participant = None
            self.audio_track = None
            logger.info("Disconnected from LiveKit room")

async def get_available_voices():
    """Get list of available voices"""
    try:
        voices = await edge_tts.list_voices()
        return [voice["Name"] for voice in voices]
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return []

livekit_tts = LiveKitTTS()

@app.route('/tts/stream', methods=['POST'])
async def stream_audio_route():
    data = request.get_json()
    text = data['text']
    voice = data.get('voice', 'en-US-AvaMultilingualNeural')
    
    await livekit_tts.connect("tts_room")
    await livekit_tts.synthesize_speech(text, voice)
    await livekit_tts.disconnect()
    
    return jsonify({"message": "TTS streamed successfully"})

@app.route('/voices', methods=['GET'])
async def voices():
    try:
        voices = await get_available_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001)
```

# tts\__init__.py

```py
"""Text-to-Speech services for voice agent."""

from .edge_tts_plugin import EdgeTTSTTS

__all__ = ['EdgeTTSTTS']

```

# tts\edge_tts_plugin.py

```py
import asyncio
import logging
import edge_tts
import io
import numpy as np
import json
from typing import Optional
from pydub import AudioSegment
import livekit.rtc as rtc
from voice_core.utils.audio_utils import AudioFrame, normalize_audio, convert_to_pcm16
from voice_core.state.voice_state_manager import VoiceState, VoiceStateManager
import time

logger = logging.getLogger(__name__)

class EdgeTTSTTS:
    def __init__(self, state_manager: VoiceStateManager, voice: str = "en-US-AvaMultilingualNeural", debug: bool = False):
        self.voice = voice
        self.state_manager = state_manager
        self.room: Optional[rtc.Room] = None
        self._tts_track: Optional[rtc.LocalAudioTrack] = None
        self._tts_source: Optional[rtc.AudioSource] = None
        self.target_rate = 48000
        self.chunk_duration_ms = 20
        self.samples_per_chunk = int(self.target_rate * self.chunk_duration_ms / 1000)
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.frame_count = 0
        self.log_interval = 50
        self._active = False
        self._playback_lock = asyncio.Lock()
        
        # Enhanced interrupt handling
        self._interrupt_check_interval = 0.02  # 20ms - check for interrupts more frequently
        self._cancellable = True              # Flag to indicate if the current TTS can be interrupted
        self._tts_future = None               # Store the future for interrupt handling
        
        # Set up event handlers
        self._setup_state_handlers()
        
    def _setup_state_handlers(self) -> None:
        """Set up handlers for state transitions."""
        @self.state_manager.on("interrupt_requested")
        async def handle_interrupt():
            self.logger.info("Interrupt requested, stopping TTS")
            await self.stop()
            
        @self.state_manager.on("state_changed")
        async def handle_state_change(event_data):
            old_state = event_data.get("old_state")
            new_state = event_data.get("new_state")
            self.logger.debug(f"Voice state changed: {old_state.name} -> {new_state.name}")
            
            # Handle transition to INTERRUPTED state
            if new_state == VoiceState.INTERRUPTED:
                self.logger.info("State changed to INTERRUPTED - stopping speech")
                await self.stop()
            
            # Publish state change to UI
            if self.room and self.room.local_participant:
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            "type": "state_update",
                            "state": new_state.value,
                            "timestamp": time.time()
                        }).encode(),
                        reliable=True
                    )
                    self.logger.debug(f"Published state change to UI: {new_state.value}")
                except Exception as e:
                    self.logger.error(f"Failed to publish state change: {e}")

    async def set_room(self, room: rtc.Room) -> None:
        """Set up TTS track in LiveKit room."""
        if not room:
            self.logger.error("Cannot set room: Room is None")
            return
            
        try:
            room_sid = await room.sid
        except Exception:
            room_sid = "unknown"
        self.logger.debug(f"Setting room in TTS service (room SID: {room_sid})")
        self.room = room
        
        # Initialize TTS track through state manager
        await self.state_manager.setup_tts_track(room)
        self.logger.info("TTS track initialized and published")
        
        # Publish initial state to UI
        if self.room and self.room.local_participant:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "state_update",
                        "state": "ready",
                        "service": "tts",
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
                self.logger.debug("Published TTS ready state to UI")
            except Exception as e:
                self.logger.error(f"Failed to publish initial TTS state: {e}")

    async def _convert_mp3_to_pcm(self, mp3_data: bytes):
        """Convert MP3 data to PCM format for streaming."""
        try:
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio = audio.set_channels(1).set_frame_rate(self.target_rate).set_sample_width(2)
            pcm_data = np.array(audio.get_array_of_samples(), dtype=np.int16)
            if pcm_data.size == 0:
                raise ValueError("Empty PCM data after conversion")
            pcm_data = normalize_audio(pcm_data)
            return pcm_data, self.target_rate
        except Exception as e:
            self.logger.error(f"MP3 to PCM conversion failed: {e}", exc_info=True)
            return np.zeros(self.samples_per_chunk, dtype=np.float32), self.target_rate

    async def _play_silence(self, duration_ms: int = 100) -> None:
        """Play a short silence to ensure clean audio transitions."""
        silence_samples = int(self.target_rate * duration_ms / 1000)
        silence_data = np.zeros(silence_samples, dtype=np.float32)
        
        for start_idx in range(0, silence_samples, self.samples_per_chunk):
            if not self._active:
                break
            end_idx = min(start_idx + self.samples_per_chunk, silence_samples)
            chunk_data = silence_data[start_idx:end_idx]
            if len(chunk_data) < self.samples_per_chunk:
                chunk_data = np.pad(chunk_data, (0, self.samples_per_chunk - len(chunk_data)))
            
            frame = AudioFrame(
                data=convert_to_pcm16(chunk_data),
                sample_rate=self.target_rate,
                num_channels=1,
                samples_per_channel=self.samples_per_chunk
            )
            if self.state_manager._tts_source:
                await self.state_manager._tts_source.capture_frame(frame.to_rtc())
            await asyncio.sleep(0.001)  # Small delay to prevent CPU overload

    async def speak(self, text: str) -> None:
        """Speak text with enhanced interrupt handling."""
        if not self.room or not self.state_manager._tts_track or not self.state_manager._tts_source:
            self.logger.error("TTS not initialized properly")
            raise RuntimeError("TTS track/source not initialized")
        if not text:
            self.logger.warning("Empty text provided to speak")
            return

        # Set up cancellable future for this TTS operation
        self._tts_future = asyncio.Future()
        
        async with self._playback_lock:  # Ensure only one speak operation at a time
            try:
                self._active = True
                self._cancellable = True  # Mark as cancellable
                
                async with self.state_manager.tts_session(text):
                    try:
                        # Publish to UI
                        if self.room and self.room.local_participant:
                            try:
                                await self.room.local_participant.publish_data(
                                    json.dumps({
                                        "type": "tts_start", 
                                        "text": text,
                                        "timestamp": time.time()
                                    }).encode(),
                                    reliable=True
                                )
                                self.logger.debug("Published TTS start event to LiveKit UI")
                            except Exception as e:
                                self.logger.error(f"Failed to publish TTS start event: {e}")
                        
                        self.logger.info(f"Starting TTS for text: {text[:50]}...")
                        
                        # Play a short silence before speech
                        await self._play_silence(50)
                        
                        # Prepare the full MP3 audio first for faster response
                        mp3_buffer = io.BytesIO()
                        communicate = edge_tts.Communicate(text, self.voice)
                        
                        # Load all audio chunks first for faster processing
                        async for chunk in communicate.stream():
                            if not self._active or self._tts_future.cancelled():
                                # Check if we were interrupted
                                self.logger.info("TTS interrupted during audio generation")
                                break
                                
                            if chunk["type"] == "audio":
                                mp3_buffer.write(chunk["data"])

                        # Skip processing if interrupted during audio collection
                        if not self._active or self._tts_future.cancelled():
                            self.logger.info("Skipping TTS playback due to interruption")
                            return

                        # Check if we have audio to play
                        if mp3_buffer.tell() == 0:
                            self.logger.warning("No MP3 data generated")
                            return

                        # Convert to PCM for faster processing
                        mp3_buffer.seek(0)
                        pcm_data, sample_rate = await self._convert_mp3_to_pcm(mp3_buffer.read())
                        total_samples = len(pcm_data)
                        samples_processed = 0
                        chunk_count = 0
                        last_interrupt_check = time.time()

                        # Process audio in chunks with frequent interrupt checks
                        for start_idx in range(0, total_samples, self.samples_per_chunk):
                            # Check if we've been interrupted
                            if not self._active or self._tts_future.cancelled():
                                self.logger.info(f"TTS interrupted after {chunk_count} chunks")
                                break
                                
                            # Frequent checks for interruption
                            current_time = time.time()
                            if current_time - last_interrupt_check > self._interrupt_check_interval:
                                # This allows other tasks to run
                                await asyncio.sleep(0)
                                last_interrupt_check = current_time
                            
                            # Process the next chunk
                            end_idx = min(start_idx + self.samples_per_chunk, total_samples)
                            chunk_data = pcm_data[start_idx:end_idx]
                            
                            # Pad if needed
                            if len(chunk_data) < self.samples_per_chunk:
                                chunk_data = np.pad(chunk_data, (0, self.samples_per_chunk - len(chunk_data)))

                            # Create and send audio frame
                            frame = AudioFrame(
                                data=convert_to_pcm16(chunk_data),
                                sample_rate=self.target_rate,
                                num_channels=1,
                                samples_per_channel=self.samples_per_chunk
                            )
                            
                            await self.state_manager._tts_source.capture_frame(frame.to_rtc())
                            samples_processed += len(chunk_data)
                            chunk_count += 1
                            
                            # Log progress periodically
                            if self.debug and chunk_count % self.log_interval == 0:
                                progress = min(100, int((samples_processed / total_samples) * 100))
                                self.logger.debug(f"TTS progress: {progress}% ({chunk_count} chunks)")
                            
                            self.frame_count += 1
                            
                            # Brief yield to allow interrupts to be processed 
                            if chunk_count % 5 == 0:  # Every 5 chunks
                                await asyncio.sleep(0)
                        
                        # Play a short silence after speech if we weren't interrupted
                        if self._active and not self._tts_future.cancelled():
                            await self._play_silence(50)
                            
                            # Publish completion to UI
                            if self.room and self.room.local_participant:
                                try:
                                    await self.room.local_participant.publish_data(
                                        json.dumps({
                                            "type": "tts_complete", 
                                            "text": text,
                                            "timestamp": time.time()
                                        }).encode(),
                                        reliable=True
                                    )
                                except Exception as e:
                                    self.logger.error(f"Failed to publish TTS complete event: {e}")
                            
                            self.logger.info(f"TTS completed: {samples_processed} samples processed")
                        
                    except asyncio.CancelledError:
                        self.logger.info("TTS task cancelled")
                        # Clean up
                        self._active = False
                        raise
                        
                    except Exception as e:
                        self.logger.error(f"Error during TTS playback: {e}", exc_info=True)
                        raise
                    
            except asyncio.CancelledError:
                self.logger.info("TTS task cancelled (outer)")
                raise
                
            except Exception as e:
                self.logger.error(f"Error in speak method: {e}", exc_info=True)
                raise
                
            finally:
                self._active = False
                self._cancellable = False
                # Complete the future unless it's already done
                if self._tts_future and not self._tts_future.done():
                    self._tts_future.set_result(None)

    async def stop(self) -> None:
        """Stop TTS playback immediately with enhanced cleanup."""
        self.logger.info("Stopping TTS playback")
        self._active = False  # Stop the processing loop immediately
        
        # Cancel the current TTS future if it exists and is cancellable
        if self._tts_future and not self._tts_future.done():
            self._tts_future.cancel()
        
        # Immediate publishing of stop event for UI feedback
        if self.room and self.room.local_participant:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "tts_stopped", 
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
                self.logger.debug("Published TTS stop event to UI")
            except Exception as e:
                self.logger.error(f"Failed to publish TTS stop event: {e}")
        
        # Clean up track resources
        try:
            await self.state_manager.cleanup_tts_track()
            # Recreate track after stopping for next TTS operation
            if self.room:
                await self.state_manager.setup_tts_track(self.room)
        except Exception as e:
            self.logger.error(f"Error during TTS cleanup: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Clean up TTS resources completely."""
        self._active = False
        
        # Cancel any active TTS operation
        if self._tts_future and not self._tts_future.done():
            self._tts_future.cancel()
            
        await self.state_manager.cleanup_tts_track()
        
        # Publish cleanup to UI
        if self.room and self.room.local_participant:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "tts_cleanup", 
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
            except Exception as e:
                self.logger.error(f"Failed to publish TTS cleanup event: {e}")
        
        self.logger.info("TTS cleanup completed")
```

# tts\interruptible_tts_service.py

```py
# interruptible_tts_service.py
# Improved interruption handling with more frequent checks

import asyncio
import logging
import edge_tts
import io
import numpy as np
import json
import time
from typing import Optional, Callable, Any, Dict
import tempfile
from pydub import AudioSegment
import livekit.rtc as rtc

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState

class InterruptibleTTSService:
    """
    Enhanced TTS service with true interruptability and streaming for EdgeTTS.
    Provides immediate interrupt response and efficient stream management.
    """

    def __init__(
        self,
        state_manager: VoiceStateManager,
        voice: str = "en-US-AvaMultilingualNeural",
        sample_rate: int = 48000,
        num_channels: int = 1,
        on_interrupt: Optional[Callable[[], Any]] = None,
        on_complete: Optional[Callable[[str], Any]] = None
    ):
        """
        Initialize enhanced TTS service with interruption capabilities.
        
        Args:
            state_manager: Voice state manager
            voice: EdgeTTS voice to use
            sample_rate: Audio sample rate (default 48kHz for LiveKit)
            num_channels: Number of audio channels (mono=1, stereo=2)
            on_interrupt: Callback when TTS is interrupted
            on_complete: Callback when TTS completes normally
        """
        self.state_manager = state_manager
        self.voice = voice
        self.sample_rate = sample_rate if state_manager._tts_source is None else state_manager._tts_source.sample_rate
        self.num_channels = num_channels if state_manager._tts_source is None else state_manager._tts_source.num_channels
        self.on_interrupt = on_interrupt
        self.on_complete = on_complete
        
        # Internal state
        self.room = None
        self._active = False
        self._cancellable = True
        self._current_task = None
        self._playback_lock = asyncio.Lock()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Playback metrics
        self.start_time = 0
        self.frame_count = 0
        self.samples_per_chunk = 960  # 20ms at 48kHz
        self.log_interval = 10  # Log every 10 chunks
        
        # Interruption handling
        self.interruption_check_interval = 5  # Check every 5 chunks
        self.interruptions_handled = 0
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced TTS service initialized with voice: {voice}")
        
    def _setup_event_handlers(self):
        """Set up handlers for state manager events."""
        # Listen for interrupt requests
        @self.state_manager.on("interrupt_requested")
        async def handle_interrupt(data):
            self.logger.info("Interrupt requested, stopping TTS")
            await self.stop()
        
    async def initialize(self) -> None:
        """
        Initialize TTS service and load necessary resources.
        Should be called before using the service.
        """
        self.logger.info("Initializing TTS service...")
        
        # Verify voices are available
        try:
            # List available voices
            voices = await edge_tts.list_voices()
            voice_names = [v["ShortName"] for v in voices]
            
            # Check if our voice is available
            if self.voice not in voice_names:
                self.logger.warning(f"Voice '{self.voice}' not found, using fallback")
                self.voice = "en-US-GuyNeural"  # Fallback voice
            
            self.logger.info(f"Using TTS voice: {self.voice}")
            
            # Publish initialization status if room available
            if self.room and self.state_manager:
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            "type": "tts_initialized",
                            "voice": self.voice,
                            "timestamp": time.time()
                        }).encode(),
                        reliable=True
                    )
                except Exception as e:
                    self.logger.error(f"Failed to publish TTS initialization: {e}")
                
        except Exception as e:
            self.logger.error(f"Error listing voices: {e}", exc_info=True)
            
        self.logger.info("TTS service initialized")
        
    async def set_room(self, room: rtc.Room) -> None:
        """
        Set LiveKit room for TTS output.
        
        Args:
            room: LiveKit room object
        """
        if not room:
            self.logger.error("Cannot set room: Room is None")
            return
            
        self.room = room
        
        # Publish ready state
        if self.state_manager:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "tts_ready",
                        "voice": self.voice,
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
            except Exception as e:
                self.logger.error(f"Failed to publish TTS ready state: {e}")
            
    async def check_interruption(self) -> bool:
        """
        Check if interruption is requested and handle it immediately.
        
        Returns:
            bool: True if interrupted, False otherwise
        """
        # Quick check for interrupt flag or state
        if (self.state_manager and (
            self.state_manager.interrupt_requested() or 
            self.state_manager.current_state == VoiceState.ERROR or
            not self._active
        )):
            self.logger.info("Interruption detected in TTS stream")
            
            # Stop active streaming immediately
            self._active = False
            self._cancellable = False
            
            # Cancel current task if exists
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            
            # Clear audio source buffer if needed
            if self.state_manager and self.state_manager._tts_source:
                try:
                    silence_frame = rtc.AudioFrame(
                        data=b'\x00' * 960,  # 10ms of silence
                        samples_per_channel=480,
                        sample_rate=48000,
                        num_channels=1
                    )
                    await self.state_manager._tts_source.capture_frame(silence_frame)
                except Exception as e:
                    self.logger.error(f"Error flushing audio buffer: {e}")
            
            # Call interrupt callback if provided
            if self.on_interrupt:
                if asyncio.iscoroutinefunction(self.on_interrupt):
                    await self.on_interrupt()
                else:
                    self.on_interrupt()
                    
            # Track metrics
            self.interruptions_handled += 1
            
            # Signal state manager that interrupt was handled
            if self.state_manager:
                self.state_manager._interrupt_handled.set()
                await self.state_manager.transition_to(VoiceState.LISTENING, {
                    "reason": "tts_interrupted",
                    "timestamp": time.time()
                })
                
            return True
            
        return False

    async def speak(self, text: str, assistant_identity: str = "assistant") -> Optional[str]:
        """
        Speak text with TTS and stream to LiveKit room.
        
        Args:
            text: Text to speak
            assistant_identity: The identity to use for assistant transcripts (default: "assistant")
            
        Returns:
            The spoken text or None if interrupted
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided, not speaking")
            return None
            
        self.logger.info(f"Speaking: '{text[:50]}...'")
        
        # Reset active flag
        self._active = True
        
        # Publish transcription to UI - do this ONCE here, not for every frame
        await self._publish_transcription(text, assistant_identity, final=True)
            
        # Create cancellable task - pass assistant_identity but don't publish transcript in _stream_tts
        task = asyncio.create_task(self._stream_tts(text, assistant_identity, should_publish_transcript=False))
        # Store the text in the task for reference
        setattr(task, 'text', text)
        self._current_task = task
        
        try:
            return await self._current_task
        except asyncio.CancelledError:
            self.logger.info("TTS task was cancelled")
            return None

    async def _publish_transcription(self, text: str, assistant_identity: str = "assistant", final: bool = True):
        """
        Publish text as transcription with proper LiveKit format.
        
        Args:
            text: Text to publish as transcription
            assistant_identity: Identity to use for assistant transcripts (default: "assistant")
            final: Whether this is a final transcription
        """
        if not self.room or not self.room.local_participant:
            return
        
        try:
            # Increment frame count for sequence tracking
            self.frame_count += 1
            sequence = self.frame_count
            
            # Try to publish via state manager for consistency
            if self.state_manager:
                try:
                    # Pass the explicit assistant identity to state manager
                    await self.state_manager.publish_transcription(
                        text,
                        "assistant",  # Always use assistant as sender type 
                        final,
                        participant_identity=assistant_identity  # Pass explicit identity
                    )
                    return  # If state manager succeeds, we don't need the fallback methods
                except Exception as e:
                    self.logger.error(f"Failed to publish via state manager: {e}")
            
            # Fallback: Use direct data channel publish if state manager failed or not available
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "transcript",
                        "text": text,
                        "sender": "assistant",
                        "participant_identity": assistant_identity,  # Include identity in JSON
                        "final": final,
                        "sequence": sequence,
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
            except Exception as e:
                self.logger.error(f"Failed to publish transcription: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to publish transcription: {e}")
            
    async def _stream_tts(self, text: str, assistant_identity: str = "assistant", should_publish_transcript: bool = False) -> str:
        """
        Stream TTS audio to LiveKit room with real-time interruption.
        
        Args:
            text: Text to speak
            assistant_identity: The identity to use for assistant transcripts (default: "assistant")
            should_publish_transcript: Whether to publish transcript (default: False, as it's handled in speak)
            
        Returns:
            The spoken text
        """
        async with self._playback_lock:
            try:
                self._active = True
                self._cancellable = True
                self.start_time = time.time()
                
                # Check for interruption before starting
                if await self.check_interruption():
                    return text
                
                # Get full audio in memory first for faster playback
                mp3_buffer = io.BytesIO()
                communicate = edge_tts.Communicate(text, self.voice)
                
                # Collect all audio chunks with constant interrupt checks
                chunks_collected = 0
                async for chunk in communicate.stream():
                    if await self.check_interruption() or not self._active:
                        return text
                        
                    if chunk["type"] == "audio":
                        mp3_buffer.write(chunk["data"])
                        chunks_collected += 1
                        await asyncio.sleep(0)  # Yield every audio chunk
                
                # Reset buffer position
                mp3_buffer.seek(0)
                
                # Convert MP3 to PCM
                pcm_data, sample_rate = await self._convert_mp3_to_pcm(mp3_buffer.getvalue())
                if pcm_data is None:
                    self.logger.error("Failed to convert MP3 to PCM")
                    return text
                
                # Split audio into micro-chunks for real-time interruption
                total_samples = len(pcm_data)
                total_duration = total_samples / self.sample_rate
                self.logger.info(f"Audio duration: {total_duration:.2f}s ({total_samples} samples)")
                
                samples_processed = 0
                chunk_count = 0
                
                # Use micro-chunks (30 samples = ~2ms) for near-instant interruption
                chunk_size = min(30, self.samples_per_chunk)  # 1/32nd of original size
                
                for start_idx in range(0, total_samples, chunk_size):
                    # Check for interruption before EVERY chunk
                    if await self.check_interruption() or not self._active:
                        self.logger.info(f"TTS interrupted after {chunk_count} chunks")
                        # Send silence to flush buffer
                        if self.state_manager._tts_source:
                            try:
                                silence_frame = rtc.AudioFrame(
                                    data=b'\x00' * chunk_size * 2,
                                    samples_per_channel=chunk_size,
                                    sample_rate=self.sample_rate,
                                    num_channels=1
                                )
                                await self.state_manager._tts_source.capture_frame(silence_frame)
                            except Exception as e:
                                self.logger.error(f"Error sending silence frame: {e}")
                        return text
                    
                    # Get chunk
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_data = pcm_data[start_idx:end_idx]
                    
                    # Pad if needed
                    if len(chunk_data) < chunk_size:
                        chunk_data = np.pad(chunk_data, (0, chunk_size - len(chunk_data)))
                    
                    # Create audio frame
                    frame = rtc.AudioFrame(
                        data=(chunk_data * 32767.0).astype(np.int16).tobytes(),
                        samples_per_channel=len(chunk_data),
                        sample_rate=self.sample_rate,
                        num_channels=self.num_channels
                    )
                    
                    # Send to LiveKit through state manager's TTS source
                    if self.state_manager._tts_source:
                        await self.state_manager._tts_source.capture_frame(frame)
                    else:
                        self.logger.warning("TTS source not available")
                        
                    # Publish transcript to UI
                    if should_publish_transcript and self.state_manager and self.room:
                        try:
                            # Publish assistant transcript with explicit identity
                            await self.state_manager.publish_transcription(
                                text, 
                                "assistant",  # Use sender type
                                is_final=True,
                                participant_identity=assistant_identity  # Use provided assistant identity
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to publish TTS transcript: {e}")
                    
                    # Update counters
                    samples_processed += len(chunk_data)
                    chunk_count += 1
                    self.frame_count += 1
                    
                    # Yield after EVERY chunk for real-time interruption
                    await asyncio.sleep(0)
                
                # Call the completion callback if provided
                if self.on_complete and not await self.check_interruption():
                    if asyncio.iscoroutinefunction(self.on_complete):
                        await self.on_complete(text)
                    else:
                        self.on_complete(text)
                
                return text
                
            except asyncio.CancelledError:
                self.logger.info("TTS task cancelled")
                raise
                
            except Exception as e:
                self.logger.error(f"Error in TTS streaming: {e}", exc_info=True)
                return text
                
            finally:
                self._active = False
                self._cancellable = False

    async def stop(self) -> None:
        """Stop TTS playback immediately."""
        self.logger.info("Stopping TTS playback")
        
        # Stop the processing loop immediately
        self._active = False
        self._cancellable = False

        # Cancel current task if exists
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                # Use very short timeout for responsiveness
                await asyncio.wait_for(self._current_task, timeout=0.1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Clear audio source buffer if needed
        if self.state_manager and self.state_manager._tts_source:
            # Add a small silence frame to flush the buffer
            try:
                silence_frame = rtc.AudioFrame(
                    data=b'\x00' * 960,  # 10ms of silence
                    samples_per_channel=480,
                    sample_rate=48000,
                    num_channels=1
                )
                await self.state_manager._tts_source.capture_frame(silence_frame)
            except Exception as e:
                self.logger.error(f"Error flushing audio buffer: {e}")

        # Signal state manager
        if self.state_manager:
            self.state_manager._interrupt_handled.set()
            await self.state_manager.transition_to(VoiceState.LISTENING, {
                "reason": "tts_stopped",
                "timestamp": time.time()
            })

    async def cleanup(self) -> None:
        """
        Clean up TTS resources.
        """
        self.logger.info("Cleaning up TTS service")
        
        # Stop any active playback
        await self.stop()
        
        # Clean up through state manager
        await self.state_manager.cleanup_tts_track()
        
        # Publish cleanup
        if self.room and self.room.local_participant:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "tts_cleanup",
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
            except Exception as e:
                self.logger.error(f"Failed to publish TTS cleanup: {e}")
                
        self.logger.info("TTS cleanup complete")

    def get_stats(self) -> dict:
        """Get current TTS service stats for monitoring."""
        return {
            "active": self._active,
            "frame_count": self.frame_count,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "interruptions_handled": self.interruptions_handled
        }

    async def _convert_mp3_to_pcm(self, mp3_data: bytes) -> tuple:
        """
        Convert MP3 data to PCM for LiveKit streaming.
        
        Args:
            mp3_data: Raw MP3 bytes
            
        Returns:
            Tuple of (pcm_data as numpy array, sample_rate)
        """
        try:
            # Create temp file for MP3 data
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(mp3_data)
            
            try:
                # Load with pydub for reliable conversion
                audio = AudioSegment.from_mp3(tmp_path)
                
                # Convert to our target format
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(self.num_channels)
                
                # Get raw PCM data
                pcm_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1] range
                pcm_data = pcm_data / 32768.0
                
                return pcm_data, self.sample_rate
                
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error converting MP3 to PCM: {e}", exc_info=True)
            
            # Register error with state manager
            if self.state_manager:
                await self.state_manager.register_error(e, "tts_conversion")
                
            return None, None
```

# tts\livekit_tts_service.py

```py
from __future__ import annotations

import asyncio
import logging
import os
import io
import uuid
import time
from typing import Optional, Dict, Any, Callable
from livekit import rtc
from voice_core.config.config import LucidiaConfig
from voice_core.tts_utils import markdown_to_text
import edge_tts
from pydub import AudioSegment
import numpy as np

logger = logging.getLogger(__name__)

class LiveKitTTSService:
    """Service for text-to-speech using Edge TTS and LiveKit publishing."""
    
    CHUNK_SIZE = 960  # 20ms at 48kHz
    SLEEP_DURATION = 0.02  # 20ms to match chunk duration
    
    def __init__(self, config: LucidiaConfig):
        """Initialize the TTS service with config."""
        self.config = config
        self.audio_queue = asyncio.Queue()
        self._shutdown = False
        self._queue_task = None
        self.room = None
        self.audio_source = None
        self.local_track = None
        self._running = False
        self.session_id = str(uuid.uuid4())
        self._event_handlers: Dict[str, Callable] = {}
        self.stats = {
            "chunks_processed": 0,
            "total_bytes_processed": 0,
            "start_time": None,
            "current_text": None,
            "last_energy": 0.0
        }
        
    def on(self, event: str, callback: Callable) -> None:
        """Register event handlers for monitoring TTS progress."""
        self._event_handlers[event] = callback

    def _emit(self, event: str, data: Any = None) -> None:
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            try:
                self._event_handlers[event](data)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")
        
    async def connect(self, room: rtc.Room) -> None:
        """Connect to LiveKit room and set up audio track."""
        if self._running:
            await self.cleanup()
            
        try:
            logger.info(f"[TTS] Initializing service with session: {self.session_id[:8]}")
            self._emit("tts_init", {"session_id": self.session_id})
            
            self.room = room
            
            # Create audio source and track
            sample_rate = self.config.tts.get('sample_rate', 48000)  # Default to LiveKit's preferred 48kHz
            num_channels = self.config.tts.get('num_channels', 1)
            
            self.audio_source = rtc.AudioSource(
                sample_rate=sample_rate,
                num_channels=num_channels
            )
            
            track_id = f"tts_output_{self.session_id[:8]}"
            self.local_track = rtc.LocalAudioTrack.create_audio_track(
                track_id, 
                self.audio_source
            )
            
            # Publish track
            await self.room.local_participant.publish_track(
                self.local_track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE,  # Use MICROPHONE for better audio handling
                    name=track_id
                )
            )
            logger.info("[TTS] Audio track published successfully")
            self._emit("track_published", {"track_id": track_id})
            
            # Start audio processing
            self._running = True
            if not self._queue_task or self._queue_task.done():
                self._queue_task = asyncio.create_task(self._process_audio_queue())
                
        except Exception as e:
            logger.error(f"[TTS] Error connecting service: {e}")
            self._emit("tts_error", {"error": str(e), "phase": "connect"})
            await self.cleanup()
            raise

    async def synthesize_speech(self, text: str, interrupt: bool = True) -> None:
        """Synthesize speech and queue audio chunks."""
        if not self._running:
            logger.error("[TTS] Service not running")
            return
            
        if interrupt and self.is_speaking():
            await self.stop_speaking()
            
        try:
            clean_text = markdown_to_text(text)
            self.stats["current_text"] = clean_text
            self.stats["start_time"] = time.time()
            
            self._emit("tts_start", {
                "text": clean_text,
                "timestamp": self.stats["start_time"]
            })
            
            communicate = edge_tts.Communicate(
                clean_text, 
                self.config.tts.get('voice', 'en-US-AvaMultilingualNeural')
            )
            
            # Collect all audio data first
            full_audio = io.BytesIO()
            bytes_processed = 0
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    full_audio.write(chunk["data"])
                    bytes_processed += len(chunk["data"])
                    self._emit("tts_progress", {
                        "bytes_processed": bytes_processed,
                        "text": clean_text
                    })
            
            if full_audio.tell() > 0:
                full_audio.seek(0)
                # Convert MP3 to PCM with proper resampling
                mp3_audio = AudioSegment.from_mp3(full_audio)
                target_rate = self.config.tts.get('sample_rate', 48000)
                target_channels = self.config.tts.get('num_channels', 1)
                
                pcm_audio = (mp3_audio
                    .set_frame_rate(target_rate)
                    .set_channels(target_channels)
                    .set_sample_width(2))  # 16-bit audio
                
                # Convert to float32 normalized [-1, 1]
                pcm_data = np.frombuffer(pcm_audio.raw_data, dtype=np.int16)
                float_data = pcm_data.astype(np.float32) / 32768.0
                
                # Apply light compression to prevent clipping
                threshold = 0.8
                ratio = 2.0
                float_data = np.where(
                    np.abs(float_data) > threshold,
                    threshold + (np.abs(float_data) - threshold) / ratio * np.sign(float_data),
                    float_data
                )
                
                # Convert back to int16
                processed_data = (float_data * 32767).astype(np.int16).tobytes()
                await self._queue_audio_chunks(processed_data)
                
                logger.info(f"[TTS] Generated audio for text: {clean_text[:50]}... "
                          f"(sample_rate={target_rate}, channels={target_channels})")
                self._emit("tts_complete", {
                    "text": clean_text,
                    "duration": time.time() - self.stats["start_time"],
                    "total_bytes": len(processed_data)
                })
            else:
                logger.warning("[TTS] No audio generated")
                self._emit("tts_error", {"error": "No audio generated", "text": clean_text})
                
        except Exception as e:
            logger.error(f"[TTS] Error synthesizing speech: {e}")
            self._emit("tts_error", {"error": str(e), "text": clean_text})

    async def _queue_audio_chunks(self, audio_data: bytes) -> None:
        """Queue audio data in fixed-size chunks."""
        chunk_size = self.CHUNK_SIZE
        
        # Calculate energy for monitoring
        samples = np.frombuffer(audio_data, dtype=np.int16)
        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        self.stats["last_energy"] = energy
        
        logger.debug(f"[TTS] Audio energy: {energy:.2f}")
        
        # Process full chunks
        for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await self.audio_queue.put(chunk)
            self.stats["chunks_processed"] += 1
            self.stats["total_bytes_processed"] += len(chunk)
            
            if self.stats["chunks_processed"] % 50 == 0:
                logger.debug(f"[TTS] Processed {self.stats['chunks_processed']} chunks, "
                           f"{self.stats['total_bytes_processed']} bytes, "
                           f"energy={self.stats['last_energy']:.2f}")
        
        # Handle final partial chunk if any
        remaining = len(audio_data) % chunk_size
        if remaining > 0:
            final_chunk = audio_data[-remaining:]
            # Pad with silence (zeros) to maintain fixed chunk size
            padding = bytes(chunk_size - remaining)
            padded_chunk = final_chunk + padding
            await self.audio_queue.put(padded_chunk)
            self.stats["chunks_processed"] += 1
            self.stats["total_bytes_processed"] += len(padded_chunk)

    async def _process_audio_queue(self) -> None:
        """Process audio chunks from the queue and publish to LiveKit."""
        logger.info("[TTS] Starting audio queue processing")
        last_log_time = time.time()
        chunks_since_log = 0
        
        while not self._shutdown:
            try:
                if self.audio_queue.empty():
                    await asyncio.sleep(0.001)  # Prevent busy waiting
                    continue
                    
                chunk = await self.audio_queue.get()
                if chunk and self.audio_source:
                    # Create audio frame with proper sample count
                    frame = rtc.AudioFrame(
                        data=chunk,
                        samples_per_channel=self.CHUNK_SIZE // 2,  # 16-bit audio = 2 bytes per sample
                        sample_rate=self.config.tts.get('sample_rate', 48000),
                        num_channels=self.config.tts.get('num_channels', 1)
                    )
                    
                    try:
                        await self.audio_source.capture_frame(frame)
                        chunks_since_log += 1
                    except Exception as e:
                        logger.error(f"[TTS] Error capturing frame: {e}")
                        self._emit("tts_error", {"error": str(e), "phase": "frame_capture"})
                        continue
                    
                    # Log progress every second
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        logger.debug(f"[TTS] Processed {chunks_since_log} chunks in the last second "
                                   f"(energy={self.stats['last_energy']:.2f})")
                        last_log_time = current_time
                        chunks_since_log = 0
                    
                    # Sleep for precise timing
                    await asyncio.sleep(self.SLEEP_DURATION)
                else:
                    logger.warning("[TTS] Received empty audio chunk or audio source not initialized")
                
                self.audio_queue.task_done()
                    
            except Exception as e:
                logger.error(f"[TTS] Error processing audio queue: {e}")
                self._emit("tts_error", {"error": str(e), "phase": "processing"})
                await asyncio.sleep(self.SLEEP_DURATION)

    async def stop_speaking(self) -> None:
        """Stop current speech playback."""
        logger.info("[TTS] Stopping current speech")
        self._emit("tts_stop", {
            "text": self.stats["current_text"],
            "chunks_processed": self.stats["chunks_processed"]
        })
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.info("[TTS] Speech stopped")

    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking."""
        return not self.audio_queue.empty()

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("[TTS] Cleaning up service")
        self._emit("tts_cleanup", {"session_id": self.session_id})
        self._shutdown = True
        self._running = False

        # Cancel audio task first
        if self._queue_task:
            try:
                self._queue_task.cancel()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[TTS] Error canceling audio task: {e}")

        # Stop and cleanup track
        if self.local_track:
            try:
                if self.room and self.room.local_participant:
                    try:
                        await self.room.local_participant.unpublish_track(self.local_track)
                        logger.info("[TTS] Track unpublished")
                    except Exception as e:
                        logger.error(f"[TTS] Error unpublishing track: {e}")
                self.local_track = None
            except Exception as e:
                logger.error(f"[TTS] Error cleaning up local track: {e}")

        # Close audio source
        self.audio_source = None

        # Clear stats
        self.stats = {
            "chunks_processed": 0,
            "total_bytes_processed": 0,
            "start_time": None,
            "current_text": None,
            "last_energy": 0.0
        }

        # Disconnect room last
        if self.room:
            try:
                await self.room.disconnect()
                self.room = None
                logger.info("[TTS] Room disconnected")
            except Exception as e:
                logger.error(f"[TTS] Error disconnecting room: {e}")
                
        self._emit("tts_cleanup_complete", {"session_id": self.session_id})
```

# tts\tts_forwarder.py

```py
import logging
import json
import time
import asyncio
from livekit import rtc

logger = logging.getLogger(__name__)

class EnhancedTTSForwarder:
    """Wrapper for TTSSegmentsForwarder with proper UI synchronization"""
    def __init__(self, room, participant, audio_source=None):
        self.room = room
        self.participant = participant
        self.audio_source = audio_source
        self._active = False
        
        # Initialize the LiveKit TTSSegmentsForwarder
        from livekit.agents.transcription.tts_forwarder import TTSSegmentsForwarder
        self.forwarder = TTSSegmentsForwarder(
            room=room,
            participant=participant,
            language="en",
            speed=1.0
        )
        
        logger.info("Enhanced TTS Forwarder initialized")

    async def _setup_track_sid(self):
        """Set up track_sid for proper UI synchronization"""
        if not self.participant:
            logger.warning("No participant available for track setup")
            return
            
        try:
            # Wait for up to 5 seconds for an audio track to be published
            for _ in range(50):  # 50 * 0.1s = 5s
                try:
                    # Try getting track directly from participant's track_publications
                    if hasattr(self.participant, 'track_publications'):
                        for pub in self.participant.track_publications.values():
                            if pub.kind == rtc.TrackKind.AUDIO:
                                if pub.sid:
                                    self.forwarder.track_sid = pub.sid
                                    logger.info(f"Set track_sid to {pub.sid}")
                                    return
                                    
                    # Try getting track from published_tracks
                    elif hasattr(self.participant, 'published_tracks'):
                        for track in self.participant.published_tracks.values():
                            if isinstance(track, rtc.LocalAudioTrack):
                                if hasattr(track, 'sid') and track.sid:
                                    self.forwarder.track_sid = track.sid
                                    logger.info(f"Set track_sid to {track.sid}")
                                    return
                                    
                except Exception as e:
                    logger.debug(f"Error accessing tracks: {e}")
                    
                await asyncio.sleep(0.1)
                
            logger.warning("No audio track found after timeout")
            
        except Exception as e:
            logger.error(f"Error setting up track_sid: {e}")
            
    async def display_text(self, text, is_user=False):
        """Display text in the UI"""
        if not self.forwarder:
            logger.warning("No TTS forwarder available")
            return False
            
        try:
            self._active = True
            
            # Start a new segment
            self.forwarder.segment_playout_started()
            
            # Push text - this makes it display in the UI
            self.forwarder.push_text(text)
            
            # Mark text segment end
            self.forwarder.mark_text_segment_end()
            
            # Also publish in standard format for compatibility
            if not is_user and self.room and self.participant:
                await self.participant.publish_data(
                    json.dumps({
                        "type": "agent-message",
                        "text": text,
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
                
            return True
        except Exception as e:
            logger.error(f"Error displaying text: {e}")
            return False
    
    async def process_message(self, text, tts_service):
        """Process a full message with text display and audio"""
        try:
            # Display text
            await self.display_text(text)
            
            # Use the speak method directly instead of process_text
            # This is compatible with InterruptibleTTSService
            await tts_service.speak(text)
            
            # Complete segment
            await self.complete_segment()
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.complete_segment()  # Try to complete segment anyway
            return False
    
    async def complete_segment(self):
        """Complete the current segment"""
        if self._active:
            try:
                self.forwarder.segment_playout_finished()
                self._active = False
                return True
            except Exception as e:
                logger.error(f"Error completing segment: {e}")
        return False
            
    async def close(self):
        """Close the forwarder"""
        try:
            if self._active:
                await self.complete_segment()
                
            # Close the forwarder
            if hasattr(self.forwarder, 'aclose'):
                await self.forwarder.aclose()
            elif hasattr(self.forwarder, 'close'):
                await self.forwarder.close()
                
            self.forwarder = None
            return True
        except Exception as e:
            logger.error(f"Error closing forwarder: {e}")
            return False
```

# tts\tts_segments_forwarder.py

```py
"""
Enhanced TTSSegmentsForwarder that combines transcription API and data publishing 
for maximum UI compatibility.
"""

import asyncio
import uuid
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from livekit import rtc

logger = logging.getLogger(__name__)

@dataclass
class TTSSegment:
    """Represents a TTS text segment."""
    text: str
    id: str = ""
    final: bool = True
    language: str = "en-US"


class EnhancedTTSForwarder:
    """
    Enhanced TTS forwarder that ensures UI compatibility across different LiveKit clients.
    Publishes both transcriptions and custom data for maximum compatibility.
    """
    
    def __init__(self, room: rtc.Room, participant_identity: str):
        """
        Initialize the forwarder.
        
        Args:
            room: LiveKit room
            participant_identity: Participant identity
        """
        self.room = room
        self.participant_identity = participant_identity
        
        # Current state
        self.current_text = ""
        self.segment_id = ""
        self.is_playing = False
        
        # Track information
        self.track_sid = None
        self._find_audio_track()
        
        # Task management
        self._data_task = None
        self._publish_task = None
        self._queue = asyncio.Queue()
        self._running = True
        
        # Error handling
        self._last_error = None
        self._publish_failures = 0
        self._max_retries = 3
        
        # Start processing task
        self._publish_task = asyncio.create_task(self._process_queue())
        logger.info("Enhanced TTS forwarder initialized")
        
    def _find_audio_track(self) -> None:
        """Find audio track SID."""
        if not self.room or not self.room.local_participant:
            logger.warning("No room or local participant available")
            return
            
        for pub in self.room.local_participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                self.track_sid = pub.sid
                logger.info(f"Found audio track: {self.track_sid}")
                return
                
        logger.warning("No audio track found")
        
    async def _process_queue(self) -> None:
        """Process segments from the queue."""
        try:
            while self._running:
                try:
                    # Get segment from queue
                    segment = await self._queue.get()
                    
                    # Process segment
                    await self._publish_segment(segment)
                    
                    # Mark as done
                    self._queue.task_done()
                    
                except asyncio.CancelledError:
                    logger.info("Process queue task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing queue: {e}")
                    await asyncio.sleep(0.5)  # Prevent tight loop on error
                    
        except Exception as e:
            logger.error(f"Fatal error in process queue: {e}")
            
    async def _publish_segment(self, segment: TTSSegment) -> None:
        """
        Publish segment with multiple methods for maximum compatibility.
        
        Args:
            segment: TTS segment to publish
        """
        try:
            if not self.room or not self.room.local_participant:
                logger.warning("Cannot publish segment: no room or participant")
                return
                
            # Update state
            self.current_text = segment.text
            self.segment_id = segment.id
            
            # Method 1: Use Transcription API
            if self.track_sid:
                trans = rtc.Transcription(
                    participant_identity=self.participant_identity,
                    track_sid=self.track_sid,
                    segments=[
                        rtc.TranscriptionSegment(
                            id=segment.id,
                            text=segment.text,
                            start_time=0,
                            end_time=0,
                            final=segment.final,
                            language=segment.language
                        )
                    ]
                )
                await self.room.local_participant.publish_transcription(trans)
                logger.debug(f"Published transcription: {segment.text[:30]}...")
                
            # Method 2: Custom data messages for UI compatibility
            await self._publish_data_message(
                "transcript", 
                {"text": segment.text, "sender": "assistant", "timestamp": time.time()}
            )
            
            # Method 3: Additional UI type messages for custom UIs
            await self._publish_data_message(
                "tts_segment",
                {"text": segment.text, "id": segment.id, "timestamp": time.time()}
            )
            
            # Reset error counter on success
            self._publish_failures = 0
            
        except Exception as e:
            self._last_error = str(e)
            self._publish_failures += 1
            logger.error(f"Failed to publish segment: {e}")
            
            # Try fallback method if available and failures are within limit
            if self._publish_failures <= self._max_retries:
                logger.info(f"Trying fallback publishing method (attempt {self._publish_failures})")
                try:
                    # Fallback to simpler data message
                    await self._publish_data_message(
                        "message",
                        {"text": segment.text, "timestamp": time.time()}
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback publishing also failed: {fallback_error}")
            
    async def _publish_data_message(self, type_name: str, payload: Dict[str, Any]) -> None:
        """
        Publish data message with retry logic.
        
        Args:
            type_name: Message type
            payload: Message payload
        """
        if not self.room or not self.room.local_participant:
            logger.warning(f"Cannot publish {type_name}: no room or participant")
            return
            
        # Prepare message
        message = {"type": type_name, **payload}
        message_data = json.dumps(message).encode()
        
        # Publish with retries
        for attempt in range(self._max_retries):
            try:
                await self.room.local_participant.publish_data(
                    message_data,
                    reliable=True
                )
                return
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                logger.warning(f"Publish attempt {attempt+1} failed: {e}, retrying...")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    async def push_text(self, text: str, language: str = "en-US", final: bool = True) -> None:
        """
        Push text to be forwarded.
        
        Args:
            text: Text to forward
            language: Text language
            final: Whether this is a final segment
        """
        if not text:
            return
            
        segment = TTSSegment(
            text=text,
            id=str(uuid.uuid4()),
            final=final,
            language=language
        )
        
        # Add to queue
        await self._queue.put(segment)
        
    def segment_playout_started(self) -> None:
        """Mark segment playout as started."""
        self.is_playing = True
        
        # Start data task if needed
        if not self._data_task or self._data_task.done():
            self._data_task = asyncio.create_task(self._publish_ui_state("speaking"))
            
    def segment_playout_finished(self) -> None:
        """Mark segment playout as finished."""
        self.is_playing = False
        
        # Start data task if needed
        if not self._data_task or self._data_task.done():
            self._data_task = asyncio.create_task(self._publish_ui_state("idle"))
            
    async def _publish_ui_state(self, state: str) -> None:
        """
        Publish UI state update.
        
        Args:
            state: UI state
        """
        try:
            await self._publish_data_message(
                "ui_state",
                {"state": state, "timestamp": time.time()}
            )
            
            # Also publish as agent-status for agent playground
            await self._publish_data_message(
                "agent-status",
                {"status": state, "timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to publish UI state: {e}")
            
    async def close(self) -> None:
        """Close the forwarder and clean up resources."""
        self._running = False
        
        # Cancel tasks
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
                
        if self._data_task:
            self._data_task.cancel()
            try:
                await self._data_task
            except asyncio.CancelledError:
                pass
                
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info("TTS forwarder closed")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get forwarder statistics."""
        return {
            "is_playing": self.is_playing,
            "publish_failures": self._publish_failures,
            "last_error": self._last_error,
            "track_sid": self.track_sid,
            "queue_size": self._queue.qsize() if self._queue else 0
        }
```

# tts\tts_utils.py

```py
import edge_tts
import io
import logging
import markdown
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Union, BinaryIO

# Configure logging
logger = logging.getLogger(__name__)

# Default voice
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"

async def list_voices() -> List[Dict[str, str]]:
    """
    Fetch available voices from Edge TTS and return them.
    
    Returns:
        List[Dict[str, str]]: List of voice dictionaries containing voice metadata.
    """
    logger.info("Fetching Edge TTS voices...")
    try:
        voices = await edge_tts.list_voices()
        logger.debug(f"Found {len(voices)} available voices")
        return voices
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []

async def select_voice(voice_name: Optional[str] = None) -> str:
    """
    Get the voice to use for TTS. If voice_name is provided, validates and returns it.
    Otherwise, returns the default voice.
    """
    if not voice_name:
        voice_name = DEFAULT_VOICE
        
    # Validate the voice exists
    voices = await list_voices()
    voice_names = [v["ShortName"] for v in voices]
    
    if voice_name in voice_names:
        logger.info(f"Using voice: {voice_name}")
        return voice_name
    else:
        logger.warning(f"Voice {voice_name} not found, using default: {DEFAULT_VOICE}")
        return DEFAULT_VOICE


def markdown_to_text(markdown_string):
    """Convert Markdown to plain text."""
    try:
        html = markdown.markdown(markdown_string)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error converting markdown to text: {e}")
        return ""


async def text_to_speech(text: str, voice: str) -> Optional[BinaryIO]:
    """
    Convert text to audio using Edge TTS and return as BytesIO.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice ID to use for conversion
        
    Returns:
        Optional[BinaryIO]: BytesIO containing audio data if successful, None otherwise
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = io.BytesIO()
        
        # Track progress for longer conversions
        total_chunks = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
                total_chunks += 1
                
                # Log progress for longer texts
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} audio chunks")
                    
        audio_data.seek(0)
        logger.info("Text-to-speech conversion complete")
        return audio_data
        
    except ConnectionError as e:
        logger.error(f"Connection error during TTS conversion: {e}")
        return None
    except OSError as e:
        logger.error(f"IO error during text-to-speech conversion: {e}")
        return None
    except RuntimeError as e:
        logger.error(f"Runtime error during text-to-speech conversion: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during text-to-speech conversion: {e}")
        return None

```

# utils\__init__.py

```py

```

# utils\audio_buffer.py

```py
from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
import time

logger = logging.getLogger(__name__)

class EnhancedAudioBuffer:
    """Responsive audio buffer with improved speech boundary detection and interruption handling"""
    
    def __init__(
        self, 
        max_length: int, 
        sample_rate: int = 16000,
        energy_threshold: float = -35.0,  # dB threshold for speech
        min_speech_duration: float = 0.3,  # seconds
        max_speech_duration: float = 20.0,  # seconds
        silence_duration: float = 0.7,  # seconds of silence to end speech
        interrupt_flush_threshold: float = 0.1  # seconds to keep after interrupt
    ):
        """
        Initialize enhanced audio buffer
        
        Args:
            max_length: Maximum number of samples to store
            sample_rate: Audio sample rate in Hz
            energy_threshold: dB threshold to consider as speech
            min_speech_duration: Minimum duration to consider valid speech
            max_speech_duration: Maximum duration for a speech segment
            silence_duration: Duration of silence to consider speech ended
            interrupt_flush_threshold: How much audio to keep after interrupt
        """
        self.max_length = int(max_length)  # Ensure integer
        self.sample_rate = int(sample_rate)  # Ensure integer
        self.buffer = np.zeros(self.max_length, dtype=np.float32)
        self.write_pos = 0
        self.length = 0
        self.is_speaking = False
        
        # Speech detection parameters
        self.energy_threshold = energy_threshold  # dB
        self.min_speech_duration = min_speech_duration  # seconds
        self.max_speech_duration = max_speech_duration  # seconds
        self.silence_duration = silence_duration  # seconds
        self.interrupt_flush_threshold = interrupt_flush_threshold  # seconds
        
        # Dynamic settings
        self.auto_threshold = True  # Auto-adjust threshold based on environment
        self.dynamic_silence = True  # Adjust silence duration based on speech length
        self.noise_floor = -50.0  # dB - will be adjusted dynamically
        self.speech_threshold_offset = 10.0  # dB above noise floor
        
        # State tracking
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.speech_duration = 0
        self.silence_start_time = 0
        self.speech_energy_history = []
        self.energy_history_max_len = 100
        self.background_energy_history = []  # For dynamic threshold adjustment
        self.background_history_max_len = 200
        
        # Performance tracking
        self.speech_segments_detected = 0
        self.interruptions_handled = 0
        self.overflows = 0
        self.total_audio_duration = 0.0
        
        # Lock for thread safety in write operations
        self._buffer_lock = None  # Will be initialized if using asyncio
        
        logger.debug(f"Created EnhancedAudioBuffer: max_length={self.max_length}, sr={self.sample_rate}Hz")
        
    def set_asyncio_lock(self, lock):
        """Set asyncio lock for thread safety in async contexts"""
        self._buffer_lock = lock
        
    def add(self, data: np.ndarray, is_interruption: bool = False) -> bool:
        """
        Add audio data to buffer with interruption handling
        
        Args:
            data: Audio data as numpy array (-1 to 1 float)
            is_interruption: Whether this is an interruption event
            
        Returns:
            bool: True if speech end detected, False otherwise
        """
        if data.size == 0:
            return False
            
        # Ensure float32 and proper range
        data = np.asarray(data, dtype=np.float32)
        if data.max() > 1 or data.min() < -1:
            data = np.clip(data, -1, 1)
            
        # Handle interruption - quickly flush buffer keeping only a small amount
        if is_interruption and self.is_speaking:
            samples_to_keep = int(self.interrupt_flush_threshold * self.sample_rate)
            if self.length > samples_to_keep:
                # Keep only the most recent samples up to the threshold
                recent_data = self.get_recent_audio(self.interrupt_flush_threshold)
                self.clear()
                self._add_to_buffer(recent_data)
                
                self.interruptions_handled += 1
                logger.debug(f"Interruption: flushed buffer, kept {len(recent_data)} samples")
                return True  # Signal end of speech due to interruption
            
        # Calculate energy level
        energy = np.mean(np.abs(data))
        energy_db = 20 * np.log10(energy + 1e-10)  # Convert to dB
        
        # Update total audio duration
        data_duration = len(data) / self.sample_rate
        self.total_audio_duration += data_duration
        
        # Add to appropriate energy history
        if self.is_speaking:
            # If speaking, add to speech energy history
            self.speech_energy_history.append(energy_db)
            if len(self.speech_energy_history) > self.energy_history_max_len:
                self.speech_energy_history.pop(0)
        else:
            # Otherwise, add to background noise history
            self.background_energy_history.append(energy_db)
            if len(self.background_energy_history) > self.background_history_max_len:
                self.background_energy_history.pop(0)
                
        # Dynamic threshold adjustment if enabled
        if self.auto_threshold and len(self.background_energy_history) > 20:
            # Use lower percentile to get stable noise floor estimate
            sorted_bg = sorted(self.background_energy_history)
            self.noise_floor = sorted_bg[int(len(sorted_bg) * 0.2)]  # 20th percentile
            
            # Ensure noise floor isn't too low
            self.noise_floor = max(self.noise_floor, -60.0)
            
            # Set speech threshold above noise floor
            self.energy_threshold = self.noise_floor + self.speech_threshold_offset
        
        # Speech detection based on energy
        now = time.time()
        speech_detected = energy_db > self.energy_threshold
        
        # Speech state tracking
        if speech_detected:
            if not self.is_speaking:
                # Speech start
                self.is_speaking = True
                self.speech_start_time = now
                self.speech_duration = 0
                logger.debug(f"Speech start detected at {energy_db:.1f}dB (threshold: {self.energy_threshold:.1f}dB)")
            
            self.last_speech_time = now
            self.speech_duration = now - self.speech_start_time
            self.silence_start_time = 0
        else:
            if self.is_speaking:
                # Potential speech end - start measuring silence
                if self.silence_start_time == 0:
                    self.silence_start_time = now
                
                # Dynamically adjust silence duration based on speech duration
                adjusted_silence = self.silence_duration
                if self.dynamic_silence:
                    # Longer utterances can have longer pauses
                    if self.speech_duration > 5.0:
                        # Scale silence duration with speech duration
                        adjusted_silence = min(1.2, self.silence_duration + (self.speech_duration - 5.0) / 20.0)
                
                # Check if silence duration threshold reached
                silence_duration = now - self.silence_start_time
                
                # Also check if max speech duration exceeded
                if (silence_duration >= adjusted_silence and 
                    self.speech_duration >= self.min_speech_duration) or \
                   self.speech_duration >= self.max_speech_duration:
                    # End of speech detected
                    self.is_speaking = False
                    self.speech_segments_detected += 1
                    
                    logger.debug(f"Speech end detected - duration: {self.speech_duration:.2f}s, "
                                f"silence: {silence_duration:.2f}s, adjusted silence: {adjusted_silence:.2f}s")
                    
                    # Add the data, then return True to signal process buffer
                    self._add_to_buffer(data)
                    return True
        
        # Add data to buffer
        self._add_to_buffer(data)
        return False
                
    def _add_to_buffer(self, data: np.ndarray) -> None:
        """Internal method to add data to circular buffer"""
        if data.size == 0:
            return
            
        # Handle data longer than buffer
        if len(data) > self.max_length:
            data = data[-self.max_length:]
            self.overflows += 1
            
        # Calculate positions
        data_len = len(data)
        space_left = self.max_length - self.write_pos
        
        if data_len <= space_left:
            # Simple case - just write data
            self.buffer[self.write_pos:self.write_pos + data_len] = data
            self.write_pos += data_len
        else:
            # Split write across buffer boundary
            self.buffer[self.write_pos:] = data[:space_left]
            remaining = data_len - space_left
            self.buffer[:remaining] = data[space_left:]
            self.write_pos = remaining
            
        self.length = min(self.length + data_len, self.max_length)
        
    def get_all(self) -> np.ndarray:
        """Get all buffered audio data"""
        if self.length == 0:
            return np.array([], dtype=np.float32)
            
        if self.write_pos >= self.length:
            # No wrap-around
            return self.buffer[self.write_pos - self.length:self.write_pos].copy()
        else:
            # Handle wrap-around
            end_data = self.buffer[-(self.length - self.write_pos):]
            start_data = self.buffer[:self.write_pos]
            return np.concatenate([end_data, start_data])
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get most recent audio of specified duration in seconds"""
        if duration <= 0 or self.length == 0:
            return np.array([], dtype=np.float32)
            
        samples = int(duration * self.sample_rate)
        samples = min(samples, self.length)
        
        return self.get_all()[-samples:]
            
    def clear(self) -> None:
        """Clear the buffer and reset speech state"""
        self.write_pos = 0
        self.length = 0
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.speech_duration = 0
        self.silence_start_time = 0
        
    def get_duration(self) -> float:
        """Get duration of buffered audio in seconds"""
        return self.length / self.sample_rate
        
    def get_speech_info(self) -> Dict[str, Any]:
        """Get information about current speech state"""
        return {
            "is_speaking": self.is_speaking,
            "speech_duration": self.speech_duration,
            "buffer_duration": self.get_duration(),
            "avg_energy_db": np.mean(self.speech_energy_history) if self.speech_energy_history else -100,
            "energy_threshold": self.energy_threshold,
            "noise_floor": self.noise_floor
        }
        
    def handle_interruption(self) -> np.ndarray:
        """Handle interruption event, return current buffer and clear"""
        data = self.get_all()
        self.clear()
        self.interruptions_handled += 1
        return data
        
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "speech_segments_detected": self.speech_segments_detected,
            "interruptions_handled": self.interruptions_handled,
            "overflows": self.overflows,
            "total_audio_duration": self.total_audio_duration,
            "current_energy_threshold": self.energy_threshold,
            "current_noise_floor": self.noise_floor,
            "buffer_size": self.length,
            "buffer_duration": self.get_duration()
        }
        
    def __len__(self) -> int:
        """Get number of samples in buffer"""
        return self.length
```

# utils\audio_utils.py

```py
import numpy as np
from livekit import rtc

class AudioFrame:
    def __init__(self, data: bytes, sample_rate: int, num_channels: int, samples_per_channel: int):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel

    def to_rtc(self) -> rtc.AudioFrame:
        return rtc.AudioFrame(
            data=self.data,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=self.samples_per_channel
        )

def normalize_audio(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = data / max_val
    return data

def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy import signal
    if src_rate == dst_rate:
        return data
    target_length = int(len(data) * dst_rate / src_rate)
    resampled = signal.resample(data, target_length)
    return resampled

def split_audio_chunks(data: np.ndarray, chunk_size: int, overlap: int = 0) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    step = chunk_size - overlap
    num_chunks = (len(data) - overlap) // step
    chunks = np.zeros((num_chunks, chunk_size), dtype=data.dtype)
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        chunks[i] = data[start:end]
    return chunks

def convert_to_pcm16(data: np.ndarray) -> bytes:
    if data.dtype == np.float32:
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        raise ValueError(f"Unsupported audio data type: {data.dtype}")
    return data.tobytes()

def create_audio_frame(data: np.ndarray, sample_rate: int, num_channels: int = 1) -> AudioFrame:
    if len(data.shape) == 1:
        data = data.reshape(-1, num_channels)
    samples_per_channel = data.shape[0]
    pcm_data = convert_to_pcm16(data)
    return AudioFrame(
        data=pcm_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel
    )
```

# utils\config.py

```py
class LucidiaConfig:
    def __init__(self):
        self.tts = {
            "voice": "en-US-AvaMultilingualNeural",
            "sample_rate": 48000,
            "num_channels": 1,
        }

class LLMConfig:
    def __init__(self):
        self.server_url = "http://localhost:1234/v1/chat/completions"
        self.model_name = "local-model"
        self.temperature = 0.7
        self.max_tokens = 300

class WhisperConfig:
    def __init__(self):
        self.model_name = "small"
        self.language = "en"
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.min_audio_length = 0.5
        self.max_audio_length = 3.0

```

# utils\event_emitter.py

```py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventEmitter:
    """
    Asynchronous event emitter implementation.
    Supports event subscription and emission with async handlers.
    """
    def __init__(self):
        self._events: Dict[str, List[Callable]] = defaultdict(list)
        self._once_events: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler.
        Can be used as a decorator or method call.
        """
        def decorator(func: Callable) -> Callable:
            self._events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def once(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register a one-time event handler.
        Handler will be removed after first execution.
        """
        def decorator(func: Callable) -> Callable:
            self._once_events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def off(self, event_name: str, handler: Callable) -> None:
        """Remove a specific event handler."""
        if event_name in self._events:
            self._events[event_name] = [h for h in self._events[event_name] if h != handler]
        if event_name in self._once_events:
            self._once_events[event_name] = [h for h in self._once_events[event_name] if h != handler]
            
    def remove_all_listeners(self, event_name: Optional[str] = None) -> None:
        """Remove all handlers for an event, or all events if no name given."""
        if event_name:
            self._events[event_name].clear()
            self._once_events[event_name].clear()
        else:
            self._events.clear()
            self._once_events.clear()
            
    async def emit(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event with optional data.
        Executes all handlers asynchronously.
        """
        # Regular handlers
        handlers = self._events.get(event_name, [])
        once_handlers = self._once_events.get(event_name, [])
        
        # Execute handlers
        for handler in handlers + once_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_name}: {e}", exc_info=True)
                
        # Clear once handlers
        if event_name in self._once_events:
            self._once_events[event_name].clear()

```

# utils\livekit-diagnostic.py

```py
"""LiveKit diagnostic tool for troubleshooting UI update issues."""

import asyncio
import logging
import time
import json
import sys
import argparse
import jwt
from typing import Dict, Any, List, Optional
import os
import socket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import LiveKit SDK with fallback options
try:
    import livekit
    import livekit.rtc as rtc
    logger.info(f"LiveKit SDK version: {livekit.__version__}")
except ImportError:
    logger.error("LiveKit SDK not found. Install with: pip install livekit")
    logger.info("Trying alternative import...")
    try:
        from livekit import rtc
        logger.info("LiveKit rtc module imported through alternative path")
    except ImportError:
        logger.error("Could not import LiveKit rtc. Please install the SDK properly.")
        sys.exit(1)

# Default settings
DEFAULT_URL = "ws://localhost:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
DEFAULT_ROOM = "test-room"

class LiveKitDiagnosticTool:
    """
    Diagnostic tool for LiveKit UI publishing issues.
    Tests various publishing methods to help diagnose UI update problems.
    """
    
    def __init__(self, url: str, api_key: str, api_secret: str, room_name: str):
        """Initialize the diagnostic tool."""
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.identity = f"diagnostic_{int(time.time())}"
        
        # Connection state
        self.room = None
        self.log_messages = []
        
        # Publishing success tracking
        self.successful_publishes = {
            "data": 0,
            "transcription": 0
        }
        self.failed_publishes = {
            "data": 0,
            "transcription": 0
        }
        
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log message with timestamp and optional metadata."""
        timestamp = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
        
        entry = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs
        }
        
        self.log_messages.append(entry)
        logger.log(level, message)
        
    def generate_token(self) -> str:
        """Generate JWT token for LiveKit with proper permissions."""
        exp_time = int(time.time()) + 3600  # 1 hour validity
        
        claims = {
            "iss": self.api_key,
            "sub": self.identity,
            "exp": exp_time,
            "nbf": int(time.time()) - 60,  # Valid from 1 minute ago (allow for clock drift)
            "video": {
                "room": self.room_name,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,  # Critical for UI updates
                "roomAdmin": True,       # Helpful for diagnostics
                "roomCreate": True       # Create room if needed
            },
            "metadata": json.dumps({
                "type": "diagnostic", 
                "version": "1.0"
            })
        }
        
        token = jwt.encode(claims, self.api_secret, algorithm="HS256")
        self.log(logging.INFO, "Token generated with permissions", permissions=claims["video"])
        return token
        
    async def network_diagnostics(self) -> bool:
        """Run network diagnostics for LiveKit connection."""
        self.log(logging.INFO, "Running network diagnostics...")
        
        # Parse URL
        if self.url.startswith("ws://"):
            host = self.url[5:].split(":")[0]
            port = int(self.url.split(":")[-1])
            secure = False
        elif self.url.startswith("wss://"):
            host = self.url[6:].split(":")[0]
            port = int(self.url.split(":")[-1]) if ":" in self.url[6:] else 443
            secure = True
        else:
            host = self.url
            port = 7880
            secure = False
            
        # DNS lookup
        try:
            self.log(logging.INFO, f"DNS lookup for {host}")
            ip_address = socket.gethostbyname(host)
            self.log(logging.INFO, f"DNS lookup successful: {ip_address}")
        except socket.gaierror as e:
            self.log(logging.ERROR, f"DNS lookup failed: {e}")
            return False
            
        # Socket connection test
        try:
            self.log(logging.INFO, f"Testing socket connection to {host}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                self.log(logging.INFO, f"Socket connection successful")
            else:
                self.log(logging.ERROR, f"Socket connection failed with error {result}")
                return False
        except Exception as e:
            self.log(logging.ERROR, f"Socket connection test failed: {e}")
            return False
            
        # Basic HTTP(S) connectivity test
        try:
            import aiohttp
            test_url = f"{'https' if secure else 'http'}://{host}:{port}/rtc"
            self.log(logging.INFO, f"Testing HTTP connectivity to {test_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, timeout=5) as response:
                    if response.status != 404:
                        self.log(logging.WARNING, f"Unexpected response from LiveKit server: {response.status}")
                    else:
                        self.log(logging.INFO, f"HTTP connectivity test successful (404 expected)")
        except Exception as e:
            self.log(logging.WARNING, f"HTTP connectivity test failed: {e}")
            # Continue anyway - this is just an extra check
            
        return True
        
    async def test_publish_methods(self) -> Dict[str, Any]:
        """Test various publishing methods and return results."""
        results = {
            "data_publish": False,
            "transcription_publish": False,
            "data_methods_tested": 0,
            "transcription_methods_tested": 0,
            "errors": []
        }
        
        if not self.room or self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            self.log(logging.ERROR, "Room not connected, cannot test publishing")
            results["errors"].append("Room not connected")
            return results
            
        if not self.room.local_participant:
            self.log(logging.ERROR, "No local participant, cannot test publishing")
            results["errors"].append("No local participant")
            return results
            
        # 1. Test basic data publishing
        try:
            results["data_methods_tested"] += 1
            test_data = json.dumps({
                "type": "diagnostic_test",
                "message": "Basic data test",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(test_data, reliable=True)
            self.log(logging.INFO, "Basic data publishing successful")
            results["data_publish"] = True
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Basic data publishing failed: {e}")
            results["errors"].append(f"Basic data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 2. Test publishing UI state message
        try:
            results["data_methods_tested"] += 1
            ui_data = json.dumps({
                "type": "state_update",
                "state": "listening",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(ui_data, reliable=True)
            self.log(logging.INFO, "UI state data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"UI state data publishing failed: {e}")
            results["errors"].append(f"UI state publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 3. Test publishing transcript data
        try:
            results["data_methods_tested"] += 1
            transcript_data = json.dumps({
                "type": "transcript",
                "text": "This is a test transcript",
                "sender": "diagnostic",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(transcript_data, reliable=True)
            self.log(logging.INFO, "Transcript data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcript data publishing failed: {e}")
            results["errors"].append(f"Transcript data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 4. Test publishing transcription API
        # First publish a local audio track to get a track_sid
        local_track = None
        try:
            audio_source = rtc.AudioSource()
            local_track = rtc.LocalAudioTrack.create_audio_track("diagnostic_audio", audio_source)
            await self.room.local_participant.publish_track(local_track)
            self.log(logging.INFO, "Audio track published successfully")
            
            # Now test transcription API
            results["transcription_methods_tested"] += 1
            trans = rtc.Transcription(
                text="Test transcription API",
                participant_identity=self.identity
            )
            await self.room.local_participant.publish_transcription(trans)
            self.log(logging.INFO, "Transcription API publishing successful")
            results["transcription_publish"] = True
            self.successful_publishes["transcription"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcription API publishing failed: {e}")
            results["errors"].append(f"Transcription API publishing: {str(e)}")
            self.failed_publishes["transcription"] += 1
        finally:
            # Clean up track
            if local_track:
                try:
                    await self.room.local_participant.unpublish_track(local_track)
                except:
                    pass
                    
        return results
        
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run full diagnostics and return results."""
        start_time = time.time()
        self.log(logging.INFO, "Starting LiveKit UI diagnostics")
        
        # Run network diagnostics first
        network_ok = await self.network_diagnostics()
        if not network_ok:
            self.log(logging.ERROR, "Network diagnostics failed, cannot continue")
            return {
                "success": False,
                "stage": "network",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
            
        # Generate token
        token = self.generate_token()
        
        # Connect to room
        try:
            self.log(logging.INFO, f"Connecting to room: {self.room_name}")
            self.room = rtc.Room()
            await self.room.connect(self.url, token)
            
            # Wait for connection to stabilize
            for _ in range(5):  # Wait up to 5 seconds
                if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    break
                await asyncio.sleep(1)
                
            if self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                self.log(logging.ERROR, f"Failed to connect to room: {self.room.connection_state}")
                return {
                    "success": False,
                    "stage": "connection",
                    "duration": time.time() - start_time,
                    "logs": self.log_messages
                }
                
            self.log(logging.INFO, "Successfully connected to room")
            
            # Test publish methods
            publish_results = await self.test_publish_methods()
            
            # Final results
            success = publish_results["data_publish"] or publish_results["transcription_publish"]
            diagnostic_results = {
                "success": success,
                "duration": time.time() - start_time,
                "network_ok": network_ok,
                "room_connected": self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED,
                "publish_results": publish_results,
                "publish_stats": {
                    "successful": self.successful_publishes,
                    "failed": self.failed_publishes
                },
                "logs": self.log_messages,
                "recommendations": []
            }
            
            # Add recommendations based on results
            if not success:
                diagnostic_results["recommendations"].append(
                    "Check token permissions, especially 'canPublishData: true'"
                )
                
            if self.failed_publishes["data"] > 0 and self.successful_publishes["data"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check Docker networking configuration (exposing port 7880)"
                )
                
            if self.failed_publishes["transcription"] > 0 and self.successful_publishes["transcription"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check LiveKit version compatibility (Transcription API requires newer versions)"
                )
                
            if success:
                diagnostic_results["recommendations"].append(
                    "UI updates should be working. If still having issues, check client-side subscription."
                )
                
            return diagnostic_results
            
        except Exception as e:
            self.log(logging.ERROR, f"Error during diagnostics: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "unknown",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
        finally:
            # Clean up
            if self.room:
                try:
                    await self.room.disconnect()
                except:
                    pass

async def print_diagnostic_results(results: Dict[str, Any]) -> None:
    """Print diagnostic results in a readable format."""
    print("\n" + "=" * 50)
    print("LIVEKIT UI CONNECTION DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    print(f"\nOverall success: {'✅' if results['success'] else '❌'}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if 'network_ok' in results:
        print(f"\nNetwork connectivity: {'✅' if results['network_ok'] else '❌'}")
        
    if 'room_connected' in results:
        print(f"Room connection: {'✅' if results['room_connected'] else '❌'}")
        
    if 'publish_results' in results:
        pr = results['publish_results']
        print("\nPublishing tests:")
        print(f"  Data publishing: {'✅' if pr['data_publish'] else '❌'} ({pr['data_methods_tested']} methods tested)")
        print(f"  Transcription publishing: {'✅' if pr['transcription_publish'] else '❌'} ({pr['transcription_methods_tested']} methods tested)")
        
        if pr.get('errors'):
            print("\nPublishing errors:")
            for err in pr['errors']:
                print(f"  - {err}")
                
    if 'publish_stats' in results:
        ps = results['publish_stats']
        print("\nPublishing statistics:")
        print(f"  Successful data publishes: {ps['successful']['data']}")
        print(f"  Failed data publishes: {ps['failed']['data']}")
        print(f"  Successful transcription publishes: {ps['successful']['transcription']}")
        print(f"  Failed transcription publishes: {ps['failed']['transcription']}")
        
    if 'recommendations' in results:
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
            
    print("\nDetailed logs:")
    for log in results['logs'][-10:]:  # Show last 10 logs
        level_color = "\033[92m" if log['level'] == "INFO" else "\033[91m" if log['level'] == "ERROR" else "\033[93m"
        reset_color = "\033[0m"
        print(f"  {log['formatted_time']} {level_color}{log['level']}{reset_color}: {log['message']}")
        
    print("\n" + "=" * 50)
    
    if results['success']:
        print("\n✅ DIAGNOSIS: UI updates should be working correctly.")
        print("If issues persist, check your client-side subscription and UI code.")
    else:
        print("\n❌ DIAGNOSIS: UI updates are not working correctly.")
        print("Follow the recommendations above to fix the issues.")
        
    print("=" * 50 + "\n")

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LiveKit UI Connection Diagnostic Tool")
    parser.add_argument("--url", default=DEFAULT_URL, help="LiveKit server URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LiveKit API key")
    parser.add_argument("--api-secret", default=DEFAULT_API_SECRET, help="LiveKit API secret")
    parser.add_argument("--room", default=DEFAULT_ROOM, help="Room name")
    parser.add_argument("--output", choices=["console", "json"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    # Allow environment variable overrides
    url = os.environ.get("LIVEKIT_URL", args.url)
    api_key = os.environ.get("LIVEKIT_API_KEY", args.api_key)
    api_secret = os.environ.get("LIVEKIT_API_SECRET", args.api_secret)
    room = os.environ.get("LIVEKIT_ROOM", args.room)
    
    # Run diagnostics
    tool = LiveKitDiagnosticTool(url, api_key, api_secret, room)
    results = await tool.run_diagnostics()
    
    # Output results
    if args.output == "json":
        print(json.dumps(results, default=str, indent=2))
    else:
        await print_diagnostic_results(results)

if __name__ == "__main__":
    asyncio.run(main())

```

# utils\logger_config.py

```py
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Configure logging with rotating file handler and console output.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (rotating, max 10MB per file, keep 30 days of logs)
    log_file = log_path / f"voice_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log system info
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    
    return logger

# Performance logging
def log_performance_metrics(operation: str, duration: float, **kwargs):
    """Log performance metrics for voice pipeline operations."""
    metrics = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        **kwargs
    }
    logger = logging.getLogger("voice_assistant")
    logger.debug(f"Performance: {metrics}")

# Error logging with context
def log_error_with_context(message: str, error: Exception, context: dict = None):
    """Log errors with additional context information."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    logger = logging.getLogger("voice_assistant")
    logger.error(f"{message}: {error_info}")

# Connection state logging
def log_connection_state(state: str, connection_state: str, details: dict = None):
    """Log connection state changes with details."""
    logger = logging.getLogger("voice_assistant")
    logger.info(f"Connection {state} (state={connection_state}): {details or {}}")

```

# utils\logging.py

```py
"""Logging configuration for voice agent"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with proper encoding and formatting"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / "voice_agent.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Log initialization
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_dir.absolute()}")
    
    return logger

```

# utils\pipeline_logger.py

```py
"""Pipeline logging utilities for voice agents."""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Metrics for voice pipeline performance tracking."""
    
    start_time: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric with timestamp."""
        self.metrics[name] = {
            'value': value,
            'timestamp': time.time() - self.start_time
        }
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a recorded metric."""
        return self.metrics.get(name)
    
    def get_duration(self, start_event: str, end_event: str) -> Optional[float]:
        """Get duration between two events."""
        start = self.get_metric(start_event)
        end = self.get_metric(end_event)
        if start and end:
            return end['timestamp'] - start['timestamp']
        return None

class PipelineLogger:
    """Logger for voice pipeline events and metrics."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.metrics = PipelineMetrics()
        
    def _log(self, level: int, stage: str, message: str, **kwargs) -> None:
        """Internal logging with consistent format."""
        metadata = {
            'session_id': self.session_id,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.log(level, f"[{stage}] {message}", extra={'metadata': metadata})
        
    # STT Events
    def stt_started(self, config: Dict[str, Any]) -> None:
        """Log STT initialization."""
        self.metrics.record_metric('stt_start', config)
        self._log(logging.INFO, 'STT', 'Speech recognition started', config=config)
        
    def stt_partial(self, text: str) -> None:
        """Log partial STT results."""
        self._log(logging.DEBUG, 'STT', f'Partial transcript: {text}', text=text)
        
    def stt_final(self, text: str, confidence: float) -> None:
        """Log final STT results."""
        self.metrics.record_metric('stt_final', {'text': text, 'confidence': confidence})
        self._log(logging.INFO, 'STT', f'Final transcript: {text}', 
                 text=text, confidence=confidence)
        
    def stt_error(self, error: Exception) -> None:
        """Log STT errors."""
        self._log(logging.ERROR, 'STT', f'Recognition error: {str(error)}', 
                 error=str(error))
        
    # LLM Events
    def llm_request(self, prompt: str) -> None:
        """Log LLM request."""
        self.metrics.record_metric('llm_request', prompt)
        self._log(logging.INFO, 'LLM', 'Sending request to LLM', prompt=prompt)
        
    def llm_response(self, response: str, metadata: Dict[str, Any]) -> None:
        """Log LLM response."""
        self.metrics.record_metric('llm_response', response)
        duration = self.metrics.get_duration('llm_request', 'llm_response')
        self._log(logging.INFO, 'LLM', f'Received response in {duration:.2f}s', 
                 response=response, metadata=metadata)
        
    def llm_error(self, error: Exception) -> None:
        """Log LLM errors."""
        self._log(logging.ERROR, 'LLM', f'LLM error: {str(error)}', 
                 error=str(error))
        
    # TTS Events
    def tts_started(self, text: str, config: Dict[str, Any]) -> None:
        """Log TTS initialization."""
        self.metrics.record_metric('tts_start', {'text': text, 'config': config})
        self._log(logging.INFO, 'TTS', 'Speech synthesis started', 
                 text=text, config=config)
        
    def tts_progress(self, bytes_processed: int) -> None:
        """Log TTS progress."""
        self._log(logging.DEBUG, 'TTS', f'Generated {bytes_processed} bytes', 
                 bytes_processed=bytes_processed)
        
    def tts_complete(self, duration: float, total_bytes: int) -> None:
        """Log TTS completion."""
        self.metrics.record_metric('tts_complete', {
            'duration': duration,
            'total_bytes': total_bytes
        })
        self._log(logging.INFO, 'TTS', 
                 f'Speech synthesis completed in {duration:.2f}s ({total_bytes} bytes)',
                 duration=duration, total_bytes=total_bytes)
        
    def tts_error(self, error: Exception) -> None:
        """Log TTS errors."""
        self._log(logging.ERROR, 'TTS', f'Synthesis error: {str(error)}', 
                 error=str(error))
        
    # LiveKit Events
    def livekit_connected(self, room_name: str, participant_id: str) -> None:
        """Log LiveKit connection."""
        self.metrics.record_metric('livekit_connect', {
            'room': room_name,
            'participant_id': participant_id
        })
        self._log(logging.INFO, 'LiveKit', 'Connected to room', 
                 room=room_name, participant_id=participant_id)
        
    def livekit_track_published(self, track_id: str, kind: str) -> None:
        """Log track publication."""
        self._log(logging.INFO, 'LiveKit', f'Published {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_track_subscribed(self, track_id: str, kind: str) -> None:
        """Log track subscription."""
        self._log(logging.INFO, 'LiveKit', f'Subscribed to {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_error(self, error: Exception) -> None:
        """Log LiveKit errors."""
        self._log(logging.ERROR, 'LiveKit', f'LiveKit error: {str(error)}', 
                 error=str(error))
        
    # Pipeline Events
    def pipeline_started(self, config: Dict[str, Any]) -> None:
        """Log pipeline start."""
        self.metrics = PipelineMetrics()  # Reset metrics
        self._log(logging.INFO, 'Pipeline', 'Voice pipeline started', config=config)
        
    def pipeline_stopped(self) -> None:
        """Log pipeline stop with performance metrics."""
        total_duration = time.time() - self.metrics.start_time
        self._log(logging.INFO, 'Pipeline', 
                 f'Voice pipeline stopped after {total_duration:.2f}s',
                 total_duration=total_duration,
                 metrics=self.metrics.metrics)
        
    def pipeline_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log pipeline errors with context."""
        self._log(logging.ERROR, 'Pipeline', 
                 f'Pipeline error: {str(error)}', 
                 error=str(error), context=context)

```

# utils\sentence_buffer.py

```py
from __future__ import annotations
import re
import time
from typing import Optional, List, Dict, Any, Deque
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

class SentenceBuffer:
    """
    Enhanced SentenceBuffer that manages partial transcriptions and sentence chunking 
    for more natural conversation flow with improved text normalization and context awareness.
    """
    
    def __init__(self, 
                 max_buffer_time: float = 5.0,
                 min_words_for_chunk: int = 3,
                 end_of_sentence_timeout: float = 1.0,
                 max_history_size: int = 10,
                 confidence_threshold: float = 0.7):
        """
        Initialize the sentence buffer with configurable parameters.
        
        Args:
            max_buffer_time: Maximum time (in seconds) to buffer text before forcing processing
            min_words_for_chunk: Minimum number of words required to process a chunk
            end_of_sentence_timeout: Time (in seconds) after which to consider a sentence complete
            max_history_size: Maximum number of processed sentences to keep in history
            confidence_threshold: Minimum confidence score for accepting transcripts
        """
        self.buffer = []
        self.last_update_time = 0
        self.max_buffer_time = max_buffer_time
        self.min_words_for_chunk = min_words_for_chunk
        self.end_of_sentence_timeout = end_of_sentence_timeout
        self.confidence_threshold = confidence_threshold
        
        # Enhanced sentence boundary detection
        self.sentence_endings = re.compile(r'[.!?][\s"\')\]]?$|$')
        self.question_pattern = re.compile(r'\b(who|what|when|where|why|how|is|are|was|were|will|do|does|did|can|could|would|should|may|might)\b', re.IGNORECASE)
        
        # Track processed sentences for context
        self.processed_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        
        # Performance metrics
        self.metrics = {
            "chunks_processed": 0,
            "sentences_completed": 0,
            "avg_sentence_length": 0,
            "total_processing_time": 0
        }
        
        # Additional filler words and hesitation sounds
        self.fillers = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'so', 'basically',
            'actually', 'literally', 'well', 'right', 'okay', 'hmm', 'mmm'
        }
        
        # Common speech recognition errors to correct
        self.common_corrections = {
            "i'm gonna": "I'm going to",
            "i gotta": "I've got to",
            "wanna": "want to",
            "kinda": "kind of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "don't know"
        }
        
        self.logger = logging.getLogger(__name__)
        
    def add_transcript(self, text: str, confidence: float = 1.0) -> Optional[str]:
        """
        Add a new transcript chunk and return a complete sentence if available.
        
        Args:
            text: The transcript text to add
            confidence: Confidence score (0-1) for this transcript
            
        Returns:
            Completed sentence if available, None otherwise
        """
        start_time = time.time()
        current_time = start_time
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.logger.debug(f"Transcript below confidence threshold: {confidence:.2f} < {self.confidence_threshold:.2f}")
            return None
        
        # Clean and normalize the text
        text = text.strip().lower()
        if not text:
            return None
            
        # Check if this is a repeat of the last chunk
        if self.buffer and text == self.buffer[-1]['text']:
            self.logger.debug("Duplicate transcript chunk detected, skipping")
            return None
            
        # Add new chunk to buffer
        self.buffer.append({
            'text': text,
            'timestamp': current_time,
            'confidence': confidence
        })
        
        self.metrics["chunks_processed"] += 1
        self.last_update_time = current_time
        
        # Try to form a complete sentence
        result = self._process_buffer(current_time)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.metrics["total_processing_time"] += processing_time
        self.logger.debug(f"Transcript processing time: {processing_time:.3f}s")
        
        return result
    
    def _process_buffer(self, current_time: float) -> Optional[str]:
        """
        Process buffer to find complete sentences with enhanced detection rules.
        
        Args:
            current_time: Current time for timeout calculation
            
        Returns:
            Completed sentence if available, None otherwise
        """
        if not self.buffer:
            return None
            
        # Join all chunks
        full_text = ' '.join(chunk['text'] for chunk in self.buffer)
        words = full_text.split()
        
        # Calculate average confidence
        avg_confidence = sum(chunk.get('confidence', 1.0) for chunk in self.buffer) / len(self.buffer)
        
        # Enhanced conditions for processing the buffer
        should_process = (
            # Natural sentence ending
            bool(self.sentence_endings.search(full_text)) or
            
            # Question detection (more likely to be a complete thought)
            bool(self.question_pattern.search(full_text) and len(words) >= 4) or
            
            # Enough words and time gap
            (len(words) >= self.min_words_for_chunk and 
             current_time - self.buffer[0]['timestamp'] > self.end_of_sentence_timeout) or
            
            # Buffer timeout
            (current_time - self.buffer[0]['timestamp'] > self.max_buffer_time) or
            
            # High confidence and sufficient length
            (avg_confidence > 0.9 and len(words) >= self.min_words_for_chunk * 2)
        )
        
        if should_process:
            # Clean up the text
            result = self._clean_text(full_text)
            
            # Add to processed history
            self.processed_history.append({
                'text': result,
                'timestamp': current_time,
                'word_count': len(result.split()),
                'confidence': avg_confidence,
                'chunks': len(self.buffer)
            })
            
            # Update metrics
            self.metrics["sentences_completed"] += 1
            total_words = sum(len(entry['text'].split()) for entry in self.processed_history)
            if self.metrics["sentences_completed"] > 0:
                self.metrics["avg_sentence_length"] = total_words / self.metrics["sentences_completed"]
            
            # Clear the buffer for next sentence
            self.buffer.clear()
            return result
            
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize the transcribed text with enhanced processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Original text for logging
        original = text
        
        # Remove filler words and hesitation sounds
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Skip filler words
            if word.lower() in self.fillers:
                continue
                
            # Apply common corrections
            corrected = False
            for error, correction in self.common_corrections.items():
                if word.lower() == error or f"{word.lower()} " == error:
                    if not cleaned_words:  # If first word, capitalize correction
                        cleaned_words.append(correction)
                    else:
                        cleaned_words.append(correction.lower())
                    corrected = True
                    break
                    
            if not corrected:
                cleaned_words.append(word)
        
        # Join words and ensure proper spacing around punctuation
        text = ' '.join(cleaned_words)
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        # Add sentence ending if missing
        if not re.search(r'[.!?]$', text):
            # Add question mark if it looks like a question
            if self.question_pattern.search(text):
                text += '?'
            else:
                text += '.'
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        if original != text:
            self.logger.debug(f"Text cleaned: '{original}' → '{text}'")
            
        return text
    
    def clear(self) -> None:
        """Clear the buffer and reset processing state."""
        self.buffer.clear()
        self.last_update_time = 0
        
    def get_partial_transcript(self) -> str:
        """
        Get the current partial transcript without clearing the buffer.
        
        Returns:
            Current partial transcript as a single string
        """
        if not self.buffer:
            return ""
        return ' '.join(chunk['text'] for chunk in self.buffer)
    
    def get_context(self, max_sentences: int = 3) -> str:
        """
        Get recent conversation context from processed history.
        
        Args:
            max_sentences: Maximum number of recent sentences to include
            
        Returns:
            Recent conversation context as a string
        """
        context = [entry['text'] for entry in list(self.processed_history)[-max_sentences:]]
        return " ".join(context)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the sentence buffer.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate average processing time
        if self.metrics["chunks_processed"] > 0:
            avg_processing_time = self.metrics["total_processing_time"] / self.metrics["chunks_processed"]
        else:
            avg_processing_time = 0
            
        return {
            **self.metrics,
            "buffer_size": len(self.buffer),
            "history_size": len(self.processed_history),
            "avg_processing_time": avg_processing_time
        }
    
    def to_json(self) -> str:
        """
        Convert current buffer state to JSON for debugging or UI display.
        
        Returns:
            JSON representation of current buffer state
        """
        state = {
            "buffer": self.buffer,
            "history": list(self.processed_history),
            "metrics": self.get_metrics(),
            "partial": self.get_partial_transcript()
        }
        return json.dumps(state, indent=2)
    
    def __len__(self) -> int:
        """Return the number of chunks in the buffer."""
        return len(self.buffer)
    
    def __bool__(self) -> bool:
        """Return True if the buffer has content."""
        return bool(self.buffer)
```

# utils\shared_state.py

```py
import threading
import asyncio
from typing import Dict, Any

# Global flag to signal interruption.
# This event can be set when, for example, the user wants to cancel ongoing speech recognition.
should_interrupt = asyncio.Event()

# Global microphone settings.
# 'selected_microphone' will store the device index or identifier of the chosen microphone.
selected_microphone = None

# Global recognizer settings.
# These settings control the sensitivity and behavior of the speech recognizer.
# Fine-tune these values based on the environment, microphone quality, and the desired balance between responsiveness and accuracy.
recognizer_settings: Dict[str, Any] = {
    # Base energy threshold for distinguishing speech from background noise.
    # A lower value makes the recognizer more sensitive, but may pick up ambient sounds.
    "energy_threshold": 300,

    # If True, the recognizer will automatically adjust the energy threshold over time
    # based on ambient noise levels. This helps maintain recognition accuracy in variable environments.
    "dynamic_energy_threshold": True,

    # The maximum length of silence (in seconds) allowed within a phrase.
    # A higher value means the recognizer will wait longer before considering a pause as the end of speech.
    "pause_threshold": 0.8,

    # The amount of non-speaking duration (in seconds) required before finalizing the speech input.
    # Setting this to a higher value (e.g., 0.8) causes the recognizer to wait longer for continued speech.
    "operation_timeout": None,

    # Additional granular settings for enhanced control:

    # The sample rate (in Hz) of the audio input.
    # A common value for many applications is 16000 Hz, balancing detail and processing load.
    "sample_rate": 16000,

    # Duration (in milliseconds) of each audio chunk processed.
    # Smaller chunks (e.g., 20 ms) allow near real-time processing but may require more frequent computation.
    "chunk_duration_ms": 20,

    # Maximum number of consecutive silent audio chunks allowed before the recognizer decides the phrase has ended.
    # Higher values (e.g., 10) allow for more natural pauses in speech.
    "max_silence_chunks": 10,

    # Minimum total phrase length (in seconds) required before processing.
    # This avoids triggering on very short utterances or noise.
    "min_phrase_length": 0.3,
    'dynamic_energy_adjustment_damping': 0.15,
    'dynamic_energy_ratio': 1.5,
}

class InterruptHandler:
    def __init__(self):
        self.interrupt_event = asyncio.Event()
        self.keywords = {"stop", "wait", "pause", "cancel"}

    async def check_for_interrupt(self, text: str) -> bool:
        """Check if text contains interrupt keywords"""
        words = text.lower().split()
        if any(kw in words for kw in self.keywords):
            self.interrupt_event.set()
            should_interrupt.set()  # Set global interrupt
            return True
        return False

    async def reset(self):
        """Reset interrupt flags"""
        self.interrupt_event.clear()
        should_interrupt.clear()

# Global interrupt handler
interrupt_handler = InterruptHandler()

```

# vad\__init__.py

```py
from .silero_vad import SileroVAD

__all__ = ['SileroVAD']

```

# vad\silero_vad.py

```py
from __future__ import annotations
import logging
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional
from functools import lru_cache

class SileroVAD:
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self._init_stats()
        self._load_model()

    def _init_stats(self):
        """Initialize statistics for monitoring VAD performance"""
        self.total_calls = 0
        self.speech_detected = 0
        self.avg_confidence = 0.0
        self.last_error = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

    def _load_model(self):
        """Load Silero VAD model with caching and error handling"""
        try:
            self.logger.info(f"Loading Silero VAD model on {self.device}...")
            
            # Check if model is already loaded
            if self.model is not None:
                self.logger.debug("VAD model already loaded")
                return
                
            # Set torch hub directory to ensure proper caching
            torch.hub.set_dir("./.cache/torch/hub")
            
            # Load model with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False,
                        trust_repo=True
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    continue
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Verify model loaded correctly
            if not isinstance(self.model, torch.nn.Module):
                raise RuntimeError("Model loaded but has incorrect type")
            
            # Run test inference
            test_input = torch.zeros(512, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                test_output = self.model(test_input, self.sampling_rate)
            
            if test_output is None or not isinstance(test_output, torch.Tensor):
                raise RuntimeError("Model test inference failed")
            
            self.logger.info("Silero VAD model loaded and verified successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model: {str(e)}")
            self.last_error = str(e)
            raise RuntimeError(f"VAD initialization failed: {str(e)}") from e

    def _preprocess_audio(self, audio_chunk: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess audio chunk for VAD inference"""
        try:
            if audio_chunk.size == 0:
                return None
                
            # Ensure correct dtype and range
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Ensure audio is in [-1, 1] range
            if np.abs(audio_chunk).max() > 1.0:
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # Ensure correct length
            target_length = self._normalize_audio_length(len(audio_chunk))
            if len(audio_chunk) > target_length:
                audio_chunk = audio_chunk[:target_length]
            elif len(audio_chunk) < target_length:
                audio_chunk = np.pad(audio_chunk, (0, target_length - len(audio_chunk)))
            
            # Convert to tensor and move to device
            tensor = torch.from_numpy(audio_chunk).to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {str(e)}")
            return None

    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech in audio chunk using Silero VAD.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        try:
            # Update stats
            self.total_calls += 1
            
            # Preprocess audio
            tensor = self._preprocess_audio(audio_chunk)
            if tensor is None:
                return False, 0.0
            
            # Calculate audio level for debugging
            audio_level = 20 * np.log10(np.abs(audio_chunk).mean() + 1e-10)
            
            # Run inference
            with torch.no_grad():
                confidence = self.model(tensor, self.sampling_rate).item()
            
            # Update running statistics
            is_speech = confidence > self.threshold
            if is_speech:
                self.speech_detected += 1
            self.avg_confidence = (self.avg_confidence * (self.total_calls - 1) + confidence) / self.total_calls
            
            # Log detailed debug info
            self.logger.debug(f"VAD: level={audio_level:.1f}dB conf={confidence:.2f} speech={is_speech}")
            
            # Reset error counter on successful inference
            self.consecutive_errors = 0
            
            return is_speech, confidence
            
        except Exception as e:
            self.consecutive_errors += 1
            self.last_error = str(e)
            
            # Log error with different severity based on consecutive failures
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.logger.error(f"VAD inference failed {self.consecutive_errors} times: {str(e)}")
            else:
                self.logger.warning(f"VAD inference failed: {str(e)}")
            
            # Return conservative estimate
            return False, 0.0

    @lru_cache(maxsize=32)
    def _normalize_audio_length(self, length: int) -> int:
        """Calculate optimal audio length based on sampling rate"""
        return 512 if self.sampling_rate == 16000 else 256

    def reset(self):
        """Reset VAD state and statistics"""
        if self.model:
            self.model.reset_states()
        self._init_stats()

    def __call__(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        return self.is_speech(audio_chunk)
```

# voice_app.py

```py
"""Core voice application implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any

from config.settings import Settings
from interfaces import (
    DatabaseService,
    TTSService,
    STTService,
    KnowledgeService,
    AgentService,
    ObserverService,
    InterpreterService,
    WorkflowService
)
from utils.error_handling import (
    handle_errors,
    ErrorCategory,
    ErrorSeverity,
    ServiceError,
    CommandError,
    KnowledgeError,
    SpeechError
)
from utils.session_manager import SessionManager

logger = logging.getLogger(__name__)


class InterpreterVoiceLucidiaApp:
    """Enhanced Voice App that integrates multiple services through dependency injection."""
    
    def __init__(
        self,
        settings: Settings,
        services: ServiceContainer
    ):
        """Initialize the application with injected services."""
        self.settings = settings
        
        # Store injected services
        self.db = services.db
        self.tts_service = services.tts
        self.stt_service = services.stt
        self.knowledge_service = services.knowledge
        self.agent_service = services.agent
        self.observer_service = services.observer
        self.interpreter_service = services.interpreter
        self.workflow_service = services.workflow
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.session_manager: Optional[SessionManager] = None
        self.running: bool = False
        
        # Voice processing state
        self._current_command: Optional[str] = None
        self._command_start_time: Optional[datetime] = None
        self._silence_duration: float = 0.0
        
    @handle_errors(
        error_category=ErrorCategory.INITIALIZATION,
        reraise=True,
        log_level=ErrorSeverity.CRITICAL
    )
    async def start(self):
        """Start the voice interpreter application."""
        session_data = {
            'name': f'Interpreter Session {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'timestamp': datetime.now().isoformat()
        }
        
        # Initialize session
        try:
            self.current_session_id = self.db.create_game_session(session_data)
            logger.info(f"Created new session: {self.current_session_id}")
            
            # Initialize session manager
            self.session_manager = SessionManager(
                session_id=self.current_session_id,
                max_chunk_duration=self.settings.voice.max_chunk_duration,
                max_chunk_words=self.settings.voice.max_chunk_words,
                max_memory_chunks=self.settings.voice.max_memory_chunks,
                auto_save_threshold=self.settings.voice.auto_save_threshold
            )
        except Exception as e:
            raise ServiceError(
                "Failed to create session",
                service_name="database",
                original_error=e
            )
        
        # Start required services
        await self._start_services()
            
        # Start main voice input loop
        self.running = True
        while self.running:
            try:
                await self._voice_input_loop()
            except Exception as e:
                logger.error(f"Error in voice input loop: {e}")
                await asyncio.sleep(0.1)
                
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.WARNING
    )
    async def _voice_input_loop(self):
        """Handle continuous voice input processing."""
        # Listen for voice input
        success, text, confidence = await self.stt_service.listen_for_command(
            timeout=5.0,
            silence_timeout=1.0
        )
        
        if not success or not text:
            self._handle_silence()
            return
            
        # Reset silence tracking
        self._silence_duration = 0.0
        
        # Add to current session
        if self.session_manager:
            completed_chunk = await self.session_manager.add_voice_input(
                text=text,
                confidence=confidence,
                metadata={
                    "type": "voice_command",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Process completed chunk if available
            if completed_chunk:
                await self.process_voice_command(completed_chunk.text)
                
    def _handle_silence(self):
        """Handle silence in voice input."""
        self._silence_duration += 0.1  # Assuming 100ms processing loop
        
        # If we have accumulated command and hit silence threshold
        if (
            self._current_command and
            self._silence_duration >= self.settings.voice.silence_threshold
        ):
            # Process the accumulated command
            asyncio.create_task(self.process_voice_command(self._current_command))
            self._current_command = None
            self._command_start_time = None
            
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.ERROR
    )
    async def process_voice_command(self, command: str):
        """Process voice command with context from recent history."""
        logger.info(f"Processing voice command: {command}")
        
        # Get current context including recent history
        context = await self._create_context(command)
        
        # Get knowledge insights
        context = await self._enhance_context_with_knowledge(command, context)
        
        # Process command through available handlers
        await self._process_command_handlers(command, context)
            
    async def _create_context(self, command: str) -> ReasoningContext:
        """Create context with recent session history."""
        try:
            vision_context = await self.observer_service.get_current_context()
        except Exception as e:
            logger.warning(f"Failed to get vision context: {e}")
            vision_context = {}
            
        # Get recent voice history
        recent_history = []
        if self.session_manager:
            recent_history = await self.session_manager.get_recent_context(
                num_chunks=5
            )
            
        return ReasoningContext(
            event_history=[
                *recent_history,
                {
                    "type": "voice_command",
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                    "vision_context": vision_context
                }
            ],
            system_state={
                "current_topics": ["voice_command", "automation"],
                "last_command_time": datetime.now().isoformat()
            },
            performance_metrics=self.session_manager.get_stats() if self.session_manager else {},
            timestamp=datetime.now()
        )
        
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def stop(self):
        """Stop the voice interpreter application."""
        self.running = False
        
        # Clean up session
        if self.session_manager:
            await self.session_manager.cleanup()
            
        # Stop all services
        await self._stop_services()
            
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def _start_services(self):
        """Start all required services."""
        try:
            await self.stt_service.start()
        except Exception as e:
            raise ServiceError(
                "Failed to start STT service",
                service_name="stt",
                original_error=e
            )
            
        try:
            await self.observer_service.start()
        except Exception as e:
            raise ServiceError(
                "Failed to start observer service",
                service_name="observer",
                original_error=e
            )
                
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def _stop_services(self):
        """Stop all services."""
        services_to_stop = [
            (self.stt_service.stop(), "stt"),
            (self.interpreter_service.cleanup(), "interpreter"),
            (self.workflow_service.cleanup(), "workflow"),
            (self.observer_service.stop(), "observer")
        ]
        
        for service_task, service_name in services_to_stop:
            try:
                await service_task
            except Exception as e:
                raise ServiceError(
                    f"Failed to stop {service_name} service",
                    service_name=service_name,
                    original_error=e
                )
            
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.WARNING
    )
    async def _process_command_handlers(self, command: str, context: ReasoningContext):
        """Process command through available handlers."""
        # Try processing with agents first
        try:
            agent_response = await self.agent_service.process_command(command, context)
            if agent_response:
                await self._handle_agent_response(agent_response)
                return
        except Exception as e:
            raise CommandError(
                "Agent processing failed",
                command=command,
                original_error=e
            )
            
        # Try workflow suggestions
        try:
            workflow = await self.workflow_service.suggest_automation(command, context)
            if workflow:
                await self._handle_workflow_suggestion(workflow)
                return
        except Exception as e:
            raise CommandError(
                "Workflow suggestion failed",
                command=command,
                original_error=e
            )
            
        # Fall back to interpreter
        try:
            interpreter_response = await self.interpreter_service.execute_command(
                command,
                context.to_dict()
            )
            await self._handle_interpreter_response(interpreter_response)
        except Exception as e:
            raise CommandError(
                "Interpreter execution failed",
                command=command,
                original_error=e
            )
            
    @handle_errors(
        error_category=ErrorCategory.SPEECH,
        log_level=ErrorSeverity.WARNING
    )
    async def speak(self, text: str):
        """Convert text to speech and play it."""
        try:
            await self.tts_service.speak(text)
        except Exception as e:
            raise SpeechError(
                "Failed to speak text",
                operation="speak",
                original_error=e
            )
            
    async def _handle_agent_response(self, response: Dict[str, Any]):
        """Handle response from an agent."""
        if response.get('speak'):
            await self.speak(response['speak'])
            
    async def _handle_workflow_suggestion(self, workflow: Dict[str, Any]):
        """Handle workflow automation suggestion."""
        if workflow.get('description'):
            await self.speak(f"I can help automate that. {workflow['description']}")
            
    async def _handle_interpreter_response(self, response: Dict[str, Any]):
        """Handle interpreter command response."""
        if response.get('message'):
            await self.speak(response['message'])

```

# voice_pipeline.py

```py
# voice_pipeline.py

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any

import livekit.rtc as rtc
import numpy as np

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService

class VoicePipeline:
    """
    Integrated voice pipeline with interrupt capabilities and optimized processing.
    Coordinates STT, LLM, and TTS services with state management.
    """

    def __init__(self, room: Optional[rtc.Room] = None):
        """
        Initialize the voice pipeline.
        
        Args:
            room: LiveKit room (optional)
        """
        self.logger = logging.getLogger(__name__)

        # The core state manager
        self.state_manager = VoiceStateManager(debug=True)

        # Services
        self.stt_service = EnhancedSTTService(
            state_manager=self.state_manager,
            vosk_model="small",
            whisper_model="small.en",
            device="cuda" if hasattr(np, 'cuda') and np.cuda.is_available() else "cpu",
            min_speech_duration=0.5,
            max_speech_duration=30.0,
            energy_threshold=0.05,
            on_transcript=self._handle_transcript
        )
        
        self.tts_service = InterruptibleTTSService(
            state_manager=self.state_manager,
            voice="en-US-AvaMultilingualNeural",
            sample_rate=48000,
            num_channels=1,
            on_interrupt=self._handle_tts_interrupt,
            on_complete=self._handle_tts_complete
        )

        # Keep reference to the LiveKit room if available
        self.room = room
        
        # Metrics
        self.metrics = {
            "stt_latency": [],
            "tts_latency": [],
            "total_latency": [],
            "interrupts": 0
        }

    async def initialize(self) -> None:
        """
        Initialize pipeline: set the room in the state manager, 
        initialize STT/TTS, set up TTS track, etc.
        """
        self.logger.info("VoicePipeline: Initializing...")

        if self.room:
            await self.state_manager.set_room(self.room)
            
            # Set room for services
            self.stt_service.set_room(self.room)
            await self.tts_service.set_room(self.room)

        # Initialize services
        await self.stt_service.initialize()
        await self.tts_service.initialize()
        
        # Setup event handlers
        self.state_manager.on("interrupt_requested", self._handle_interrupt_request)
        self.state_manager.on("state_change", self._handle_state_change)

        # Start in IDLE state
        await self.state_manager.transition_to(VoiceState.IDLE)
        
        self.logger.info("VoicePipeline: Initialization complete")

    async def start(self, greeting: Optional[str] = None) -> None:
        """
        Start the voice pipeline and optionally speak a greeting.
        
        Args:
            greeting: Optional greeting to speak when starting
        """
        try:
            # Ensure we're initialized
            if self.state_manager.current_state == VoiceState.IDLE:
                # Speak greeting if provided
                if greeting:
                    await self.say(greeting)
                    
                # Transition to LISTENING
                await self.state_manager.transition_to(VoiceState.LISTENING)
                
                # Publish started event
                if self.room and self.room.local_participant:
                    try:
                        await self.room.local_participant.publish_data(
                            json.dumps({
                                "type": "pipeline_started",
                                "timestamp": time.time()
                            }).encode(),
                            reliable=True
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to publish startup event: {e}")
                        
                self.logger.info("Voice pipeline started and listening")
                
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}", exc_info=True)
            await self.state_manager.register_error(e, "start")

    async def start_listening(self) -> None:
        """Enter listening mode, waiting for speech."""
        await self.state_manager.transition_to(VoiceState.LISTENING)

    async def handle_remote_audio(self, track: rtc.AudioTrack) -> None:
        """
        Process audio from a remote track.
        
        Args:
            track: LiveKit audio track to process
        """
        self.logger.info("VoicePipeline: Starting STT processing for remote track.")
        await self.stt_service.process_audio(track)

    async def _handle_transcript(self, text: str) -> None:
        """
        Handle transcript from STT service.
        
        Args:
            text: Transcribed text
        """
        try:
            self.logger.info(f"Processing transcript: '{text[:50]}...'")
            
            # The state_manager.handle_stt_transcript method will handle
            # state transitions and interruption logic
            
            # You would typically process the transcript through an LLM here
            # and then speak the response
            
            # For this example, we'll just echo back the text
            response = f"I heard you say: {text}"
            await self.say(response)
            
        except Exception as e:
            self.logger.error(f"Error handling transcript: {e}", exc_info=True)
            await self.state_manager.register_error(e, "transcript_handling")

    async def _handle_tts_interrupt(self) -> None:
        """Handle TTS interruption."""
        self.metrics["interrupts"] += 1
        self.logger.info("TTS interrupted")

    async def _handle_tts_complete(self, text: str) -> None:
        """
        Handle TTS completion.
        
        Args:
            text: Text that was spoken
        """
        self.logger.info(f"TTS complete: '{text[:50]}...'")
        
        # Transition to LISTENING is handled by the state_manager

    async def _handle_interrupt_request(self, data: Dict[str, Any]) -> None:
        """
        Handle interrupt request from state manager.
        
        Args:
            data: Interrupt data including text
        """
        try:
            self.logger.info(f"Interrupt requested: {data.get('text', '')[:50]}")
            
            # Stop TTS
            await self.tts_service.stop()
            
        except Exception as e:
            self.logger.error(f"Error handling interrupt request: {e}", exc_info=True)

    async def _handle_state_change(self, data: Dict[str, Any]) -> None:
        """
        Handle state change event.
        
        Args:
            data: State change data
        """
        try:
            old_state = data.get("old_state")
            new_state = data.get("new_state")
            metadata = data.get("metadata", {})
            
            self.logger.info(f"State change: {old_state} -> {new_state} with metadata: {metadata}")
            
        except Exception as e:
            self.logger.error(f"Error handling state change: {e}", exc_info=True)

    async def say(self, text: str) -> bool:
        """
        Synthesize text and speak it. The TTS service integrates with the state manager 
        to handle interruptions.
        
        Args:
            text: Text to speak
            
        Returns:
            True if completed successfully, False if interrupted or error
        """
        try:
            # Create TTS task
            tts_task = asyncio.create_task(self.tts_service.speak(text))
            
            # Register with state manager - this will transition to SPEAKING state
            await self.state_manager.start_speaking(tts_task)
            
            # Wait for completion
            return await tts_task
            
        except asyncio.CancelledError:
            self.logger.info("TTS task cancelled")
            return False
            
        except Exception as e:
            self.logger.error(f"Error speaking text: {e}", exc_info=True)
            await self.state_manager.register_error(e, "tts")
            return False

    async def reset_state(self) -> None:
        """Reset state between conversation turns to prevent issues."""
        try:
            self.logger.debug("Resetting pipeline state for next conversation turn")
            
            # Reset STT buffer if needed
            if hasattr(self.stt_service, 'clear_buffer'):
                await self.stt_service.clear_buffer()
                
            # Ensure transition to LISTENING
            await self.state_manager.transition_to(VoiceState.LISTENING)
            
        except Exception as e:
            self.logger.error(f"Error resetting pipeline state: {e}")
            # Still try to ensure LISTENING state
            await self.state_manager.transition_to(VoiceState.LISTENING)

    async def stop(self) -> None:
        """Shut down pipeline gracefully."""
        self.logger.info("Stopping voice pipeline")
        
        # Reset state before stopping
        await self.reset_state()
        
        # Stop services
        await self.tts_service.stop()
        await self.stt_service.cleanup()
        await self.tts_service.cleanup()
        
        # Clean up state manager
        await self.state_manager.cleanup()

        # Disconnect from room if needed
        if self.room:
            try:
                await self.room.disconnect()
                self.logger.info("Disconnected from LiveKit room")
            except Exception as e:
                self.logger.error(f"Error disconnecting from room: {e}")
                
        self.logger.info("Voice pipeline stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = {
            "stt_latency_avg": sum(self.metrics["stt_latency"]) / max(len(self.metrics["stt_latency"]), 1),
            "tts_latency_avg": sum(self.metrics["tts_latency"]) / max(len(self.metrics["tts_latency"]), 1),
            "total_latency_avg": sum(self.metrics["total_latency"]) / max(len(self.metrics["total_latency"]), 1),
            "interrupts": self.metrics["interrupts"],
            "state_metrics": self.state_manager.get_analytics()
        }
        
        return metrics
```

