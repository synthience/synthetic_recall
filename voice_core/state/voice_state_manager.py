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
        processing_timeout: float = 30.0,
        speaking_timeout: float = 120.0,
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