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
from memory_core.enhanced_memory_client import EnhancedMemoryClient as MemoryClient

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

TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://localhost:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://localhost:5005')

# Optional overrides
DEFAULT_TTS_VOICE = os.getenv('EDGE_TTS_VOICE', 'en-US-AvaMultilingualNeural')
DEFAULT_STT_MODEL = os.getenv('OPENAI_STT_MODEL', 'whisper-1')
INITIAL_GREETING = os.getenv('INITIAL_GREETING', "Hello! I'm Lucidia, your voice assistant with persistent memory. I remember our conversations and learn from our interactions. How can I help you today?")

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
        
        # Create memory client
        self.memory_client = MemoryClient(
            tensor_server_url=TENSOR_SERVER_URL,
            hpc_server_url=HPC_SERVER_URL,
            session_id=self.session_id,
            user_id=self._get_local_identity(),
            ping_interval=30.0,  # Increased ping interval for more stable connections
            max_retries=5,       # More retries for better resilience
            retry_delay=1.5,     # Slightly longer delay between retries
            connection_timeout=15.0  # Longer timeout for initial connections
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
            
            # Initialize memory client first
            logger.info("Initializing memory client...")
            try:
                if not await self.memory_client.initialize():
                    logger.error("Failed to initialize memory client")
                    raise RuntimeError("Memory client initialization failed")
            except Exception as e:
                logger.error(f"Error initializing memory client: {e}", exc_info=True)
                raise
            
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
            
            # Connect memory client to LLM service
            logger.info("Connecting memory client to LLM service...")
            self.llm_service.set_memory_client(self.memory_client)
            
            # Register transcript handler with state manager
            self.state_manager.register_transcript_handler(self._handle_transcript)
            
            # Register room with services
            self.stt_service.set_room(self.room)
            
            # Initialize services
            logger.info("Initializing STT service...")
            await self.stt_service.initialize()
            
            logger.info("Initializing TTS service...")
            await self.tts_service.initialize()
            await self.tts_service.set_room(self.room)
            
            logger.info("Initializing LLM service...")
            await self.llm_service.initialize()
            
            # Explicitly initialize memory client to ensure background tasks start
            logger.info("Initializing memory client...")
            await self.memory_client.initialize()
            
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
            
    async def _handle_transcript(self, text: str, is_final: bool = True) -> None:
        """Handle incoming transcript from STT."""
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

            # Store transcript in memory system
            if is_final:
                await self.memory_client.store_transcript(text, sender="user")
                
                # Detect and store personal details from user message
                await self.memory_client.detect_and_store_personal_details(text, role="user")
                
                # Detect emotional context
                emotional_context = await self.memory_client.detect_emotional_context(text)
                if emotional_context:
                    logger.info(f"Detected emotional context: {emotional_context}")
                    await self.memory_client.store_emotional_context(emotional_context)
                
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
                    
                    # Store assistant response in memory
                    if response:
                        await self.memory_client.store_transcript(response, role="assistant")
                        
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
        """
        Clean up resources and ensure memory persistence before shutdown.
        Implements robust error handling and timeout protection to prevent memory loss.
        """
        self.logger.info("Starting agent cleanup process")
        cleanup_start = time.time()
        
        # Track cleanup tasks for better monitoring
        cleanup_tasks = {
            "memory_persistence": False,
            "websocket_connections": False,
            "audio_resources": False,
            "background_tasks": False
        }
        
        try:
            # First, ensure all memories are persisted
            if self.memory_client:
                try:
                    self.logger.info("Forcing final memory persistence before shutdown")
                    
                    # Use a timeout to prevent hanging during shutdown
                    persistence_timeout = 10  # seconds
                    try:
                        # Create a task for memory persistence
                        persistence_task = asyncio.create_task(self.memory_client.force_persistence())
                        
                        # Wait for the task with a timeout
                        await asyncio.wait_for(persistence_task, timeout=persistence_timeout)
                        self.logger.info("Final memory persistence completed successfully")
                        cleanup_tasks["memory_persistence"] = True
                    except asyncio.TimeoutError:
                        self.logger.error(f"Memory persistence timed out after {persistence_timeout} seconds")
                        # Continue with cleanup even if persistence times out
                    except Exception as e:
                        self.logger.error(f"Error during final memory persistence: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to force memory persistence: {e}")
            
            # Clean up WebSocket connections
            try:
                if hasattr(self, 'ws_client') and self.ws_client:
                    await self.ws_client.close()
                    self.logger.info("Closed WebSocket client connection")
                cleanup_tasks["websocket_connections"] = True
            except Exception as e:
                self.logger.error(f"Error closing WebSocket connection: {e}")
            
            # Clean up audio resources
            try:
                if hasattr(self, 'audio_manager') and self.audio_manager:
                    await self.audio_manager.cleanup()
                    self.logger.info("Cleaned up audio manager resources")
                cleanup_tasks["audio_resources"] = True
            except Exception as e:
                self.logger.error(f"Error cleaning up audio resources: {e}")
            
            # Cancel any remaining background tasks
            try:
                if hasattr(self, '_background_tasks') and self._background_tasks:
                    self.logger.info(f"Cancelling {len(self._background_tasks)} background tasks")
                    for task in self._background_tasks:
                        if not task.done() and not task.cancelled():
                            task.cancel()
                    
                    # Wait for tasks to be cancelled with a timeout
                    tasks_timeout = 5  # seconds
                    try:
                        await asyncio.wait(self._background_tasks, timeout=tasks_timeout)
                        self.logger.info("All background tasks cancelled successfully")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Some background tasks did not cancel within {tasks_timeout} seconds")
                    except Exception as e:
                        self.logger.error(f"Error waiting for background tasks to cancel: {e}")
                    
                    self._background_tasks.clear()
                    cleanup_tasks["background_tasks"] = True
            except Exception as e:
                self.logger.error(f"Error cancelling background tasks: {e}")
            
            # Final cleanup of memory client
            if self.memory_client:
                try:
                    # Use a timeout for memory client cleanup
                    memory_cleanup_timeout = 5  # seconds
                    try:
                        memory_cleanup_task = asyncio.create_task(self.memory_client.cleanup())
                        await asyncio.wait_for(memory_cleanup_task, timeout=memory_cleanup_timeout)
                        self.logger.info("Memory client cleanup completed successfully")
                    except asyncio.TimeoutError:
                        self.logger.error(f"Memory client cleanup timed out after {memory_cleanup_timeout} seconds")
                    except Exception as e:
                        self.logger.error(f"Error during memory client cleanup: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to clean up memory client: {e}")
            
            # Log cleanup status
            cleanup_time = time.time() - cleanup_start
            successful_tasks = sum(1 for status in cleanup_tasks.values() if status)
            self.logger.info(f"Agent cleanup completed in {cleanup_time:.2f}s: {successful_tasks}/{len(cleanup_tasks)} tasks successful")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during agent cleanup: {e}")
        finally:
            # Ensure we always log completion even if there are errors
            self.logger.info("Agent cleanup process finished")
        
    def _get_local_identity(self) -> str:
        """Get the local participant identity or a default value if not available."""
        if self.room and self.room.local_participant:
            return self.room.local_participant.identity
        return "assistant"  # Default fallback

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