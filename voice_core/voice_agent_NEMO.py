"""Enhanced voice agent implementation with reliable UI updates, interruption handling, and memory-driven intelligence."""

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
from voice_core.stt.nemo_stt import NemoSTT
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.llm.llm_pipeline import LocalLLMPipeline
from voice_core.config.config import LucidiaConfig, LLMConfig, WhisperConfig, TTSConfig, StateConfig, RoomConfig
from voice_core.connection_utils import force_room_cleanup, cleanup_connection

# Import memory system components
from memory_core.enhanced_memory_client import EnhancedMemoryClient
try:
    from memory.lucidia_memory_system.core.integration.memory_integration import MemoryIntegration
    from memory.lucidia_memory_system.core.memory_prioritization_layer import MemoryPrioritizationLayer
    HIERARCHICAL_MEMORY_AVAILABLE = True
except ImportError:
    HIERARCHICAL_MEMORY_AVAILABLE = False

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("livekit").propagate = False
logging.getLogger("livekit.agents").propagate = False
logger = logging.getLogger(__name__)

# Environment Variables
load_dotenv()
LIVEKIT_URL = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', 'devkey')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', 'secret')
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://localhost:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://localhost:5005')
DEFAULT_TTS_VOICE = os.getenv('EDGE_TTS_VOICE', 'en-US-AvaMultilingualNeural')
DEFAULT_STT_MODEL = os.getenv('OPENAI_STT_MODEL', 'whisper-1')
INITIAL_GREETING = os.getenv('INITIAL_GREETING', "Hello! I'm Lucidia, your voice assistant with persistent memory. How can I assist you today?")
USE_HIERARCHICAL_MEMORY = os.getenv('USE_HIERARCHICAL_MEMORY', 'true').lower() == 'true'

class LucidiaVoiceAgent:
    """Enhanced voice agent with hierarchical memory system and state management."""
    
    def __init__(self, 
                 job_context: JobContext,
                 initial_greeting: Optional[str] = None,
                 config: Optional[LucidiaConfig] = None):
        self.job_context = job_context
        self.room = None
        self.initial_greeting = initial_greeting or INITIAL_GREETING
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        self.config = config or LucidiaConfig()
        self.state_manager = VoiceStateManager(
            processing_timeout=self.config.state.processing_timeout,
            speaking_timeout=self.config.state.speaking_timeout,
            debug=self.config.state.debug
        )
        
        # Initialize memory system
        self._init_memory_system()
        
        # Core services
        self.enhanced_stt_service = None
        self.nemo_stt_service = None
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
        
        # Task tracking
        self._tasks = {}

    def _init_memory_system(self):
        use_hierarchical = USE_HIERARCHICAL_MEMORY and HIERARCHICAL_MEMORY_AVAILABLE
        
        if use_hierarchical:
            try:
                self.logger.info("Initializing Lucidia's Hierarchical Memory System")
                memory_config = {
                    'embedding_dim': 384,
                    'max_memories': 10000,
                    'stm_max_size': 10,
                    'min_similarity': 0.3,
                    'quickrecal_score_threshold': 0.3,
                    'enable_persistence': True,
                    'decay_rate': 0.05,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                }
                self.memory_integration = MemoryIntegration(memory_config)
                self.memory_client = EnhancedMemoryClient(
                    tensor_server_url=TENSOR_SERVER_URL,
                    hpc_server_url=HPC_SERVER_URL,
                    session_id=self.session_id,
                    user_id=self._get_local_identity(),
                    ping_interval=30.0,
                    max_retries=5,
                    retry_delay=1.5,
                    connection_timeout=15.0,
                    memory_integration=self.memory_integration
                )
                self.logger.info("Hierarchical Memory System initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing hierarchical memory: {e}")
                self.logger.info("Falling back to standard memory client")
                self.memory_client = EnhancedMemoryClient(
                    tensor_server_url=TENSOR_SERVER_URL,
                    hpc_server_url=HPC_SERVER_URL,
                    session_id=self.session_id,
                    user_id=self._get_local_identity(),
                    ping_interval=30.0,
                    max_retries=5,
                    retry_delay=1.5,
                    connection_timeout=15.0
                )
        else:
            self.logger.info("Using standard memory client")
            self.memory_client = EnhancedMemoryClient(
                tensor_server_url=TENSOR_SERVER_URL,
                hpc_server_url=HPC_SERVER_URL,
                session_id=self.session_id,
                user_id=self._get_local_identity(),
                ping_interval=30.0,
                max_retries=5,
                retry_delay=1.5,
                connection_timeout=15.0
            )
        
    def _init_stt_services(self):
        # Initialize Nemo STT first for final transcription using Docker service
        nemo_config = {
            "model_name": self.config.whisper.model_name or DEFAULT_STT_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hpc_url": HPC_SERVER_URL,
            "enable_streaming": False,
            "max_audio_length": self.config.whisper.max_audio_length,
            "sample_rate": 16000,
            "docker_endpoint": os.environ.get("NEMO_DOCKER_ENDPOINT", "ws://localhost:5002/ws/transcribe")
        }
        
        self.nemo_stt_service = NemoSTT(nemo_config)
        self.nemo_stt_service.register_callback("on_transcription", self._handle_nemo_transcription)
        self.nemo_stt_service.register_callback("on_semantic", self._handle_nemo_semantic)
        self.nemo_stt_service.register_callback("on_error", self._handle_nemo_error)
        
        # Initialize enhanced STT service for audio preprocessing and VAD
        # Pass the NemoSTT instance to allow direct sending of audio
        self.enhanced_stt_service = EnhancedSTTService(
            state_manager=self.state_manager,
            whisper_model=self.config.whisper.model_name,
            device=self.config.whisper.device,
            min_speech_duration=self.config.whisper.min_speech_duration,
            max_speech_duration=self.config.whisper.max_audio_length,
            energy_threshold=self.config.whisper.speech_confidence_threshold,
            on_transcript=self._handle_preliminary_transcript,
            nemo_stt=self.nemo_stt_service  # Connect NemoSTT to EnhancedSTTService
        )
    
    async def initialize(self) -> None:
        try:
            logger.info(f"Initializing Lucidia voice agent (session: {self.session_id})")
            logger.info("Initializing memory client...")
            if not await self.memory_client.initialize():
                logger.error("Failed to initialize memory client")
                raise RuntimeError("Memory client initialization failed")
                
            if not self.job_context or not self.job_context.room:
                logger.error("No room provided in job context")
                raise RuntimeError("No room provided in job context")
                
            self.room = self.job_context.room
            self._heartbeat_task = asyncio.create_task(self._connection_heartbeat())
            self._tasks["heartbeat"] = self._heartbeat_task
            
            await self.state_manager.set_room(self.room)
            
            # Initialize both STT services
            self._init_stt_services()
            
            # Initialize NemoSTT service first
            self.logger.info("Initializing NemoSTT service...")
            await self.nemo_stt_service.initialize()
            self.logger.info("NemoSTT service initialized successfully")
            
            # Initialize Enhanced STT service
            self.logger.info("Initializing Enhanced STT service...")
            await self.enhanced_stt_service.initialize()
            self.logger.info("Enhanced STT service initialized successfully")
            
            self.tts_service = InterruptibleTTSService(
                state_manager=self.state_manager,
                voice=self.config.tts.voice,
                sample_rate=self.config.tts.sample_rate,
                num_channels=self.config.tts.channels,
                on_interrupt=self._handle_tts_interrupt,
                on_complete=self._handle_tts_complete
            )
            self.llm_service = LocalLLMPipeline(self.config.llm)
            
            logger.info("Connecting memory client to LLM service...")
            self.llm_service.set_memory_client(self.memory_client)
            
            self.state_manager.register_transcript_handler(self._handle_transcript)
            self.enhanced_stt_service.set_room(self.room)
            
            logger.info("Initializing TTS service...")
            await self.tts_service.initialize()
            await self.tts_service.set_room(self.room)
            
            logger.info("Initializing LLM service...")
            await self.llm_service.initialize()
            
            await self._publish_ui_update({
                "type": "agent_initialized",
                "timestamp": time.time(),
                "agent_id": self.session_id,
                "config": {
                    "whisper_model": self.config.whisper.model_name,
                    "nemo_model": self.nemo_stt_service.config["model_name"],
                    "tts_voice": self.config.tts.voice,
                    "llm_model": self.config.llm.model,
                    "memory_system": "hierarchical" if hasattr(self, "memory_integration") else "standard"
                }
            })
            
            self._initialized = True
            logger.info("Lucidia voice agent initialized")
        
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            await self.state_manager.register_error(e, "initialization")
            raise

    async def start(self) -> None:
        if not self._initialized:
            logger.error("Agent not initialized")
            raise RuntimeError("Agent not initialized")
            
        self._running = True
        await self.state_manager.transition_to(VoiceState.IDLE)
        logger.info("Sending greeting")
        await self._send_greeting()
        await self.enhanced_stt_service.clear_buffer()
        await self.state_manager.transition_to(VoiceState.LISTENING)
        logger.info("Listening for speech input")
        
    async def _publish_ui_update(self, data: Dict[str, Any]) -> bool:
        if not self.room or not self.room.local_participant:
            return False
            
        try:
            message_bytes = json.dumps(data).encode()
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
                    
                    await self.enhanced_stt_service.clear_buffer()
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
                    await self.enhanced_stt_service.clear_buffer()

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
                await self.enhanced_stt_service.clear_buffer()

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
                await self.enhanced_stt_service.clear_buffer()

            finally:
                # Ensure we're listening unless in ERROR or INTERRUPTED state
                current_state = self.state_manager.current_state
                if current_state not in [VoiceState.ERROR, VoiceState.INTERRUPTED]:
                    await self.state_manager.transition_to(VoiceState.LISTENING)
                
                self.logger.debug(f"Transcript processing completed in {time.time() - start_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error handling transcript: {e}", exc_info=True)
            await self.state_manager.transition_to(VoiceState.ERROR, {"error": str(e)})

    async def _handle_preliminary_transcript(self, text: str, is_final: bool = False) -> None:
        """Handle preliminary transcript from EnhancedSTTService."""
        if not is_final:
            # For non-final transcripts, just update the UI
            await self.state_manager.publish_transcription(text, "user", False)
            return
            
        # For final transcript from enhanced STT, pass to NemoSTT if available
        # and set up confidence threshold
        if hasattr(self, "nemo_stt_service") and self.nemo_stt_service and self.enhanced_stt_service.buffer:
            try:
                # Get audio buffer from enhanced STT service
                audio_buffer = np.concatenate(self.enhanced_stt_service.buffer)
                
                # Process with NemoSTT for higher quality transcription
                self.logger.info(f"Processing final transcript with NemoSTT: '{text[:50]}...'")
                
                # Use executor to run NemoSTT transcription in a separate thread
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.nemo_stt_service.transcribe(audio_buffer)
                )
                
                # If NemoSTT processing fails, fall back to enhanced STT result
                if not result or not isinstance(result, dict) or "text" not in result:
                    self.logger.warning("NemoSTT processing failed, using enhanced STT result")
                    await self._handle_transcript(text, True)
            except Exception as e:
                self.logger.error(f"Error processing with NemoSTT: {e}")
                # Fall back to enhanced STT result
                await self._handle_transcript(text, True)
        else:
            # If NemoSTT not available, use enhanced STT result directly
            await self._handle_transcript(text, True)

    async def _handle_nemo_transcription(self, data: Dict[str, Any]) -> None:
        """Callback from NemoSTT: final text recognized."""
        if not data or "text" not in data:
            self.logger.warning("Received empty or invalid transcription data")
            return
        
        text = data["text"]
        confidence = data.get("confidence", 0.0)
        
        self.logger.info(f"NemoSTT transcription: '{text}' (confidence: {confidence:.2f})")
        
        # Use confidence threshold to filter out low-quality transcriptions
        if confidence >= self.config.whisper.speech_confidence_threshold:
            self.logger.info(f"Processing NemoSTT transcription: '{text}'")
            await self._handle_transcript(text, is_final=True)
        else:
            self.logger.warning(f"Skipping transcription due to low confidence: '{text}' ({confidence:.2f})")

    async def _handle_nemo_semantic(self, data: Dict[str, Any]) -> None:
        """Callback from NemoSTT: semantic processing result."""
        if not data:
            return
            
        self.logger.info(f"Semantic processing result: '{data.get('text', '')}' "
                    f"(quickrecal_score: {data.get('quickrecal_score', data.get('significance', 0.0)):.2f})")
                    
        if data.get("quickrecal_score", data.get("significance", 0.0)) >= 0.3 and data.get("text"):
            await self.memory_client.store_memory(
                content=data["text"],
                metadata={
                    "type": "semantic_insight",
                    "quickrecal_score": data.get("quickrecal_score", data.get("significance")),
                    "timestamp": time.time()
                }
            )

    async def _handle_nemo_error(self, data: Dict[str, Any]) -> None:
        """Callback from NemoSTT: error info."""
        self.logger.error(f"NemoSTT error: {data.get('error', 'Unknown error')}")
        await self.state_manager.register_error(
            Exception(data.get('error', 'Unknown STT error')), 
            "stt_processing"
        )

    async def _handle_tts_interrupt(self) -> None:
        logger.info("TTS interrupted")
        self.metrics["interruptions"] += 1
        
        if self.state_manager.current_state == VoiceState.INTERRUPTED:
            await asyncio.sleep(0.1)
            if self.state_manager.current_state == VoiceState.INTERRUPTED:
                await self.state_manager.transition_to(VoiceState.LISTENING, {"reason": "interrupt_handled"})
        
    async def _handle_tts_complete(self, text: str) -> None:
        logger.info(f"TTS completed: '{text[:50]}...'")
        if self.state_manager.current_state == VoiceState.SPEAKING:
            await self.state_manager.transition_to(VoiceState.LISTENING, {"reason": "tts_complete"})
        
    async def _send_greeting(self) -> None:
        try:
            logger.info(f"Sending greeting: '{self.initial_greeting}'")
            tts_task = asyncio.create_task(self.tts_service.speak(self.initial_greeting, self._get_local_identity()))
            self._tasks["tts_greeting"] = tts_task
            await self.state_manager.start_speaking(tts_task)
            await tts_task
        except asyncio.CancelledError:
            logger.info("Greeting TTS task was cancelled")
        finally:
            self._tasks.pop("tts_greeting", None)
            
    async def process_audio(self, track: rtc.AudioTrack) -> None:
        try:
            if self.enhanced_stt_service and self._running:
                process_task = asyncio.create_task(self.enhanced_stt_service.process_audio(track))
                self._tasks["process_audio"] = process_task
                await process_task
        except asyncio.CancelledError:
            logger.info("Audio processing task was cancelled")
        finally:
            self._tasks.pop("process_audio", None)
            
    def _get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.metrics["start_time"]
        return {
            "uptime": uptime,
            "conversations": self.metrics["conversations"],
            "successful_responses": self.metrics["successful_responses"],
            "errors": self.metrics["errors"],
            "interruptions": self.metrics["interruptions"],
            "avg_response_time": self.metrics["avg_response_time"],
            "stt_stats": self.enhanced_stt_service.get_stats() if self.enhanced_stt_service else {},
            "tts_stats": self.tts_service.get_stats() if self.tts_service else {},
            "state_metrics": self.state_manager.get_analytics()
        }
            
    async def cleanup(self) -> None:
        self.logger.info("Starting agent cleanup process")
        cleanup_start = time.time()
        
        cleanup_tasks = {
            "memory_persistence": False,
            "websocket_connections": False,
            "audio_resources": False,
            "background_tasks": False
        }
        
        try:
            if self.memory_client:
                self.logger.info("Forcing final memory persistence before shutdown")
                persistence_timeout = 10
                try:
                    persistence_task = asyncio.create_task(self.memory_client.force_persistence())
                    await asyncio.wait_for(persistence_task, timeout=persistence_timeout)
                    self.logger.info("Final memory persistence completed successfully")
                    cleanup_tasks["memory_persistence"] = True
                except asyncio.TimeoutError:
                    self.logger.error(f"Memory persistence timed out after {persistence_timeout} seconds")
                except Exception as e:
                    self.logger.error(f"Error during final memory persistence: {e}")
            
            try:
                if hasattr(self, 'ws_client') and self.ws_client:
                    await self.ws_client.close()
                    self.logger.info("Closed WebSocket client connection")
                cleanup_tasks["websocket_connections"] = True
            except Exception as e:
                self.logger.error(f"Error closing WebSocket connection: {e}")
            
            try:
                if hasattr(self, 'audio_manager') and self.audio_manager:
                    await self.audio_manager.cleanup()
                    self.logger.info("Cleaned up audio manager resources")
                cleanup_tasks["audio_resources"] = True
            except Exception as e:
                self.logger.error(f"Error cleaning up audio resources: {e}")
            
            try:
                if self._tasks:
                    self.logger.info(f"Cancelling {len(self._tasks)} background tasks")
                    for task_name, task in self._tasks.items():
                        if not task.done() and not task.cancelled():
                            task.cancel()
                    tasks_timeout = 5
                    try:
                        await asyncio.wait(self._tasks.values(), timeout=tasks_timeout)
                        self.logger.info("All background tasks cancelled successfully")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Some background tasks did not cancel within {tasks_timeout} seconds")
                    cleanup_tasks["background_tasks"] = True
            except Exception as e:
                self.logger.error(f"Error cancelling background tasks: {e}")
            
            if self.memory_client:
                memory_cleanup_timeout = 5
                try:
                    memory_cleanup_task = asyncio.create_task(self.memory_client.cleanup())
                    await asyncio.wait_for(memory_cleanup_task, timeout=memory_cleanup_timeout)
                    self.logger.info("Memory client cleanup completed successfully")
                except asyncio.TimeoutError:
                    self.logger.error(f"Memory client cleanup timed out after {memory_cleanup_timeout} seconds")
                except Exception as e:
                    self.logger.error(f"Error during memory client cleanup: {e}")
            
            cleanup_time = time.time() - cleanup_start
            successful_tasks = sum(1 for status in cleanup_tasks.values() if status)
            self.logger.info(f"Agent cleanup completed in {cleanup_time:.2f}s: {successful_tasks}/{len(cleanup_tasks)} tasks successful")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during agent cleanup: {e}")
        finally:
            self.logger.info("Agent cleanup process finished")
        
    def _get_local_identity(self) -> str:
        if self.room and self.room.local_participant:
            return self.room.local_participant.identity
        return "assistant"

    async def _connection_heartbeat(self) -> None:
        try:
            logger.info("Starting connection heartbeat")
            heartbeat_interval = 5
            max_retries = 5
            backoff_factor = 2
            attempt = 0
            
            while not self._shutdown_requested:
                try:
                    if not self.room:
                        logger.warning("No room available for heartbeat")
                        await asyncio.sleep(heartbeat_interval)
                        continue
                        
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
                        attempt = 0
                        
                    if self.room and self.room.local_participant and attempt % 12 == 0:
                        await self._publish_ui_update({
                            "type": "heartbeat",
                            "count": self._connection_retry_count,
                            "timestamp": time.time(),
                            "agent_id": self.session_id,
                            "state": self.state_manager.current_state.name,
                            "metrics": self._get_metrics()
                        })
                        
                    await asyncio.sleep(heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    await asyncio.sleep(heartbeat_interval)
                
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in heartbeat: {e}", exc_info=True)

async def connect_with_retry(ctx: JobContext, room_name: str, participant_identity: Optional[str] = None) -> rtc.Room:
    max_retries = 5
    retry_delay = 2
    
    logger.info(f"Connecting to room '{room_name}' with participant identity '{participant_identity}'")
    livekit_url = os.environ.get("LIVEKIT_URL", "Not set in environment")
    logger.info(f"LiveKit URL: {livekit_url}")
    
    if room_name:
        os.environ["LIVEKIT_ROOM"] = room_name
    if participant_identity:
        os.environ["LIVEKIT_PARTICIPANT_IDENTITY"] = participant_identity
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connection attempt {attempt}/{max_retries}...")
            await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
            room = ctx.room
            logger.info(f"Successfully connected to room")
            logger.info(f"Local participant connected")
            
            # Check for participants in a way that's compatible with LiveKit SDK 0.20.1
            try:
                # New SDK version might use get_participants() method or similar
                if hasattr(room, 'get_participants'):
                    other_participants = room.get_participants()
                    logger.info(f"Other participants in room: {len(other_participants)}")
                # Or it might have a participants property but not as a dict
                elif hasattr(room, 'participants'):
                    if isinstance(room.participants, dict):
                        other_participants = list(room.participants.values())
                    else:
                        other_participants = room.participants
                    logger.info(f"Other participants in room: {len(other_participants)}")
                else:
                    # If we can't find participants, just log and continue
                    logger.warning("Could not access participants list - SDK version compatibility issue")
            except Exception as e:
                logger.warning(f"Error accessing participants: {e}. Continuing anyway.")
            
            return room
        except Exception as e:
            logger.error(f"Connection attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                raise RuntimeError(f"Failed to connect to LiveKit room after {max_retries} attempts: {e}")

async def entrypoint(ctx: JobContext, cli_room_name: Optional[str] = None, cli_participant_identity: Optional[str] = None) -> None:
    logger.info("Lucidia Voice Agent starting...")
    
    room_name = cli_room_name or (ctx.info.job.room.name if hasattr(ctx, "info") and ctx.info and hasattr(ctx.info.job, "room") and ctx.info.job.room else None) or os.environ.get("LIVEKIT_ROOM") or os.environ.get("ROOM_NAME") or "lucidia_room"
    participant_identity = cli_participant_identity or (ctx.info.accept_arguments.identity if hasattr(ctx, "info") and ctx.info and hasattr(ctx.info, "accept_arguments") and ctx.info.accept_arguments else None) or os.environ.get("LIVEKIT_PARTICIPANT_IDENTITY") or f"lucidia-{uuid.uuid4()}"
    
    logger.info(f"Room name: '{room_name}'")
    logger.info(f"Participant identity: '{participant_identity}'")
    
    room = await connect_with_retry(ctx, room_name, participant_identity)
    
    agent = None
    try:
        config = LucidiaConfig()
        config.room.room_name = room_name
        config.room.participant_identity = participant_identity
        
        agent = LucidiaVoiceAgent(ctx, INITIAL_GREETING, config)
        await agent.initialize()
        await agent.start()
        
        async def find_audio_track():
            for _ in range(30):
                for participant in ctx.room.remote_participants.values():
                    for pub in participant.track_publications.values():
                        if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                            return pub.track
                await asyncio.sleep(1)
            return None
            
        audio_track = await find_audio_track()
        if audio_track:
            logger.info("Found audio track, processing...")
            await agent.process_audio(audio_track)
        else:
            logger.info("No audio track found, waiting...")
            
        while ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            
        logger.info("Room disconnected, ending agent")
        
    except Exception as e:
        logger.error(f"Fatal error in entrypoint: {e}", exc_info=True)
    finally:
        if agent:
            await agent.cleanup()
        await cleanup_connection(agent, ctx)

async def entrypoint_cli(ctx: JobContext) -> None:
    try:
        room_name = ctx.args.room if hasattr(ctx, 'args') and ctx.args and hasattr(ctx.args, 'room') else None
        participant_identity = ctx.args.participant_identity if hasattr(ctx, 'args') and ctx.args and hasattr(ctx.args, 'participant_identity') else None
        await entrypoint(ctx, room_name, participant_identity)
    except Exception as e:
        logger.error(f"Error in entrypoint_cli: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logger.info("Starting Lucidia Voice Agent")
    logger.info(f"Command line arguments: {sys.argv}")
    
    import argparse
    parser = argparse.ArgumentParser(description='Lucidia Voice Agent')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--room', help='Room name to connect to')
    parser.add_argument('--participant-identity', help='Participant identity')
    
    args, unknown = parser.parse_known_args()
    logger.info(f"Parsed arguments: {args}")
    logger.info(f"Unknown arguments: {unknown}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli.run_app(
        WorkerOptions(
            agent_name="lucidia",
            entrypoint_fnc=entrypoint_cli,
            worker_type=WorkerType.ROOM,
        )
    )