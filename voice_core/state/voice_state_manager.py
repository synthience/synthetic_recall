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
    PAUSED = auto()        # Paused state

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
        
        # Task management
        self._tasks: Dict[str, asyncio.Task] = {}
        self._in_progress_task: Optional[asyncio.Task] = None
        self._current_tts_task: Optional[asyncio.Task] = None
        self._last_tts_text: Optional[str] = None
        
        # Transcript handling
        self._transcript_handler: Optional[Callable[[str], Awaitable[None]]] = None
        self._transcript_sequence: int = 0
        self._recent_processed_transcripts: List[Tuple[str, float]] = []  # (norm_text, timestamp)
        self._transcript_hash_memory_seconds = 5.0
        self._min_transcript_interval = 0.5
        self._last_transcript_time = 0.0
        
        # Interruption handling
        self._interrupt_requested_event = asyncio.Event()
        self._interrupt_handled_event = asyncio.Event()
        self._interrupt_handled_event.set()  # Start in "handled" state
        self._last_interrupt_time: Optional[float] = None
        
        # Enhanced interruption tracking
        self._session_interruptions = 0
        self._interruption_timestamps: List[float] = []
        self._current_session_id = str(uuid.uuid4())
        self._session_start_time = time.time()
        
        # LiveKit integration
        self._room: Optional[rtc.Room] = None
        self._tts_track: Optional[rtc.LocalAudioTrack] = None
        self._tts_source: Optional[rtc.AudioSource] = None
        
        # Retry settings
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
        
        # Last error
        self._last_error = None
        
        # Start background state monitor
        self._start_state_monitor()
        
        # Optional TTS reference (for soft/hard-stop calls)
        self._tts_service = None  # set via set_tts_service if needed

        # For tts_session context manager
        self._tts_context_manager_initialized = False
        
        # Additional dedup dictionary for transcripts
        self._recent_transcripts: Dict[str, float] = {}

    def set_tts_service(self, tts_service: Any) -> None:
        """
        Optional setter for a TTS service if we want to call request_soft_stop / stop directly here.
        """
        self._tts_service = tts_service

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
                last_transition = next(
                    (t for t in reversed(self._state_history) if t["to"] == current_state.name), 
                    None
                )
                
                if last_transition:
                    time_in_state = time.time() - last_transition["timestamp"]
                    
                    # Determine timeout based on state
                    if current_state == VoiceState.PROCESSING:
                        timeout = self._processing_timeout
                    elif current_state == VoiceState.SPEAKING:
                        timeout = self._speaking_timeout
                    elif current_state == VoiceState.ERROR:
                        timeout = 10.0
                    elif current_state == VoiceState.PAUSED:
                        timeout = 1.0
                    else:
                        timeout = None
                    
                    if timeout and time_in_state > timeout:
                        self.logger.warning(f"Stuck in {current_state.name} for {time_in_state:.1f}s, forcing reset to LISTENING")
                        
                        # Forcefully cancel tasks
                        await self._cancel_active_tasks()
                        # Transition to LISTENING
                        await self.transition_to(
                            VoiceState.LISTENING,
                            {"reason": f"{current_state.name.lower()}_timeout", "prev_state": current_state.name}
                        )
                
                # Check more frequently for responsiveness
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.logger.info("State monitor task cancelled")
        except Exception as e:
            self.logger.error(f"Error in state monitor: {e}", exc_info=True)
            self._last_error = str(e)

    async def _cancel_active_tasks(self):
        """Cancel all active tasks with proper cleanup."""
        # Cancel in-progress tasks
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
                await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=0.5)
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

    async def setup_tts_track(self, room: rtc.Room) -> None:
        """Set up TTS track for audio output through LiveKit."""
        try:
            # First clean up any existing track to prevent resource leaks
            await self.cleanup_tts_track()
            
            # Create a new audio source and track
            self.logger.info("Setting up TTS audio track")
            # Standard audio parameters for voice communications
            sample_rate = 48000  # 48kHz is common for high-quality voice
            num_channels = 1     # Mono is typically used for voice
            
            self._tts_source = rtc.AudioSource(sample_rate=sample_rate, num_channels=num_channels)
            
            # Create a local audio track from the source
            self._tts_track = rtc.LocalAudioTrack.create_audio_track("tts-output", self._tts_source)
            
            # Publish the track to the room
            if self._room and self._room.local_participant:
                self.logger.info("Publishing TTS track")
                await self._room.local_participant.publish_track(self._tts_track)
                self.logger.info("TTS track published successfully")
        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error setting up TTS track: {e}", exc_info=True)
            # Clean up in case of partial setup
            await self.cleanup_tts_track()
            # Re-raise to allow proper error handling
            raise

    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """Register an event handler, with optional decorator usage."""
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
        """Lowercases, strips punctuation, merges whitespace for dedup checks."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        if text1 == text2:
            return 1.0
        if text1 in text2 or text2 in text1:
            return 0.8
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        
        common_words = words1.intersection(words2)
        union_size = len(words1.union(words2))
        return len(common_words) / union_size if union_size > 0 else 0.0

    def _is_duplicate_transcript(self, text: str) -> bool:
        """Check if this transcript is a duplicate of a recent one."""
        if not text or not text.strip():
            return True
        
        now = time.time()
        # Remove old entries
        self._recent_processed_transcripts = [
            (h, t) for h, t in self._recent_processed_transcripts
            if now - t < self._transcript_hash_memory_seconds
        ]
        
        norm_text = self._normalize_text(text)
        for hash_text, timestamp in self._recent_processed_transcripts:
            if self._compute_similarity(norm_text, hash_text) > 0.8:
                return True
        
        if now - self._last_transcript_time < self._min_transcript_interval:
            return True
            
        return False

    def _get_status_for_ui(self, state: VoiceState) -> dict:
        """Get status info for UI updates."""
        status = {
            "state": state.name,
            "timestamp": time.time(),
            "metrics": {
                "interruptions": self._session_interruptions,
                "errors": 0,
                "transcripts": len(self._recent_processed_transcripts)
            }
        }
        # State-specific metadata
        if state == VoiceState.LISTENING:
            status.update({"listening": True, "speaking": False, "processing": False})
        elif state == VoiceState.SPEAKING:
            status.update({
                "listening": False,
                "speaking": True,
                "processing": False,
                "tts_active": True if (self._current_tts_task and not self._current_tts_task.done()) else False
            })
        elif state == VoiceState.PROCESSING:
            status.update({"listening": False, "speaking": False, "processing": True})
        elif state == VoiceState.ERROR:
            status.update({
                "listening": False,
                "speaking": False,
                "processing": False,
                "error": self._last_error if self._last_error else "Unknown error"
            })
        elif state == VoiceState.PAUSED:
            status.update({
                "listening": False,
                "speaking": False,
                "processing": False,
                "paused": True
            })
        return status

    async def transition_to(self, new_state: VoiceState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Change state with concurrency protection and UI updates."""
        if metadata is None:
            metadata = {}
        metadata["timestamp"] = time.time()
        
        async with self._state_lock:
            old_state = self._state
            if new_state == self._state and new_state != VoiceState.PROCESSING:
                return
            
            # Special handling
            if new_state == VoiceState.INTERRUPTED:
                self._interrupt_requested_event.set()
                self._interrupt_handled_event.clear()
                # Cancel TTS if needed
                if self._current_tts_task and not self._current_tts_task.done():
                    self.logger.info("Cancelling TTS task due to interruption")
                    self._current_tts_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    self._current_tts_task = None
            
            elif new_state == VoiceState.ERROR:
                await self._cancel_active_tasks()
            
            elif new_state == VoiceState.LISTENING:
                self._interrupt_requested_event.clear()
                self._interrupt_handled_event.set()
            
            self._state = new_state
            
            # Record transition
            transition = {
                "from": old_state.name,
                "to": new_state.name,
                "timestamp": metadata.get("timestamp", time.time()),
                "metadata": metadata
            }
            self._state_history.append(transition)
            
            # Log
            self.logger.info(f"State transition: {old_state.name} -> {new_state.name} {metadata}")
            
            # Emit event
            await self.emit("state_change", {
                "old_state": old_state,
                "new_state": new_state,
                "metadata": metadata
            })
            
            # Publish to LiveKit
            if self._room and self._room.local_participant:
                try:
                    await self._publish_with_retry(
                        json.dumps({
                            "type": "state_update",
                            "from": old_state.name,
                            "to": new_state.name,
                            "timestamp": metadata["timestamp"],
                            "metadata": metadata,
                            "status": self._get_status_for_ui(new_state)
                        }).encode(),
                        "state update"
                    )
                    # Use the dedicated method for agent status publishing
                    await self._publish_agent_status()
                except Exception as e:
                    self.logger.error(f"Failed to publish state update: {e}", exc_info=True)

    async def handle_user_speech_detected(
        self, 
        text: Optional[str] = None, 
        duration_ms: float = 0.0
    ) -> None:
        """
        Handle detection of user speech with improved interruption. 
        If user is speaking for > 300ms => hard stop TTS. 
        If user speaks < 300ms => soft stop TTS (fade out).
        """
        # Only interrupt if we're currently in SPEAKING
        if self._state == VoiceState.SPEAKING:
            # Track interruption for memory
            interrupt_time = time.time()
            self._session_interruptions += 1
            self._interruption_timestamps.append(interrupt_time)
            self._last_interrupt_time = interrupt_time
            
            self.logger.info(f"User speech detected while speaking (duration={duration_ms}ms). " +
                          f"Interrupting TTS. Total interruptions: {self._session_interruptions}")
            
            # Hard vs. soft
            if self._tts_service:
                if duration_ms >= 300:
                    self.logger.info("Hard stop triggered (≥300ms).")
                    await self._tts_service.stop()
                else:
                    self.logger.info("Soft stop triggered (<300ms).")
                    await self._tts_service.request_soft_stop()
            
            # Wait for interrupt to be handled
            self._interrupt_requested_event.set()
            self._interrupt_handled_event.clear()

            if self._current_tts_task and not self._current_tts_task.done():
                self.logger.info("Cancelling TTS task due to interruption.")
                self._current_tts_task.cancel()
                try:
                    await asyncio.wait_for(self._current_tts_task, timeout=0.2)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            try:
                handled = await asyncio.wait_for(self._interrupt_handled_event.wait(), timeout=0.5)
                if not handled:
                    self.logger.warning("Interrupt was not handled in time.")
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for TTS to acknowledge interrupt.")
            
            # If text is provided, go to PROCESSING, else back to LISTENING
            if text:
                await self.transition_to(VoiceState.PROCESSING, {"text": text})
            else:
                await self.transition_to(VoiceState.LISTENING, {"reason": "user_speech_detected"})

    async def request_early_interrupt(
        self, 
        energy_level: Optional[float] = None, 
        duration_ms: float = 30.0
    ) -> None:
        """
        Request immediate TTS interrupt when speech is detected but before transcription.
        This provides significantly faster response to user speech than waiting for transcription.
        
        Args:
            energy_level: Energy level in dB (if available) to help determine interrupt type
            duration_ms: Estimated duration of speech detected so far in milliseconds
        """
        if self._state != VoiceState.SPEAKING:
            self.logger.debug(f"Early interrupt ignored - not in SPEAKING state (current={self._state})")
            return  # Only interrupt if we're actually speaking
        
        if energy_level is not None:
            energy_str = f"{energy_level:.1f}dB"
        else:
            energy_str = "N/A"

        duration_str = f"{duration_ms:.1f}ms"

        self.logger.info(
            f"Early speech interrupt triggered (energy={energy_str}, duration={duration_str})"
        )
        
        # Define energy threshold for hard vs soft interrupts
        # Higher energy likely means more intentional speech
        ENERGY_THRESHOLD_HARD = -30.0  # dB
        
        # Determine if this should be a hard or soft interrupt
        is_hard_interrupt = False
        
        # Check for debounce to avoid rapid toggling
        now = time.time()
        if self._last_interrupt_time and (now - self._last_interrupt_time) < 0.5:
            # Multiple interrupts in quick succession - escalate to hard stop
            self.logger.info("Multiple interruptions detected in quick succession - escalating to hard stop")
            is_hard_interrupt = True
        else:
            # Determine based on energy and duration
            is_hard_interrupt = (
                (energy_level is not None and energy_level > ENERGY_THRESHOLD_HARD) or 
                duration_ms >= 300.0
            )
            energy_check = "N/A"
            if energy_level is not None:
                energy_check = f"{energy_level > ENERGY_THRESHOLD_HARD} (level={energy_level:.1f}dB)"
                
            duration_check = f"{duration_ms >= 300.0} (duration={duration_ms:.1f}ms)"
            
            self.logger.debug(
                f"Interrupt type determination: hard={is_hard_interrupt} (energy_check={energy_check}, duration_check={duration_check})"
            )
        
        self._last_interrupt_time = now
        
        # Execute the appropriate interrupt type
        if is_hard_interrupt:
            self.logger.info("Executing hard stop due to early speech detection")
            if self._tts_service:
                await self._tts_service.stop()
                
            # Set the interrupt events
            self._interrupt_requested_event.set()
            self._interrupt_handled_event.clear()
            
            # Wait briefly for interrupt to be handled
            try:
                await asyncio.wait_for(self._interrupt_handled_event.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for interrupt to be handled")
                
            # Transition directly to LISTENING state
            await self.transition_to(VoiceState.LISTENING, {
                "reason": "early_speech_interrupt",
                "interrupt_type": "hard"
            })
        else:
            self.logger.info("Executing soft stop due to early speech detection")
            if self._tts_service:
                await self._tts_service.request_soft_stop()
                
            # Transition to PAUSED state to allow for possible resumption
            await self.transition_to(VoiceState.PAUSED, {
                "reason": "early_speech_interrupt",
                "interrupt_type": "soft",
                "pause_start_time": time.time()
            })
            
            # Start a task to check for continued speech
            self._tasks["pause_monitor"] = asyncio.create_task(
                self._monitor_paused_state(timeout=0.8)  # 800ms pause monitoring
            )

    def interrupt_requested(self) -> bool:
        """Check if interruption is currently requested."""
        return self._interrupt_requested_event.is_set() or (self._state == VoiceState.ERROR)
        
    async def handle_stt_transcript(self, text: str, confidence: float = 1.0) -> bool:
        """
        Handle a final STT transcript from the STT service with optional confidence-based clarification.
        
        Args:
            text: The transcript text
            confidence: Confidence score [0.0 - 1.0]
        
        Returns:
            True if transcript was processed, False if ignored
        """
        if not text or not text.strip():
            self.logger.debug("Empty transcript ignored")
            return False
        
        if self._is_duplicate_transcript(text):
            self.logger.debug(f"Duplicate transcript ignored: {text}")
            return False
        
        # Mark transcript timestamp
        timestamp = time.time()
        self._last_transcript_time = timestamp
        norm_text = self._normalize_text(text)
        self._recent_processed_transcripts.append((norm_text, timestamp))
        self._recent_transcripts[norm_text] = timestamp
        
        # Prepare interruption metadata
        was_interrupted = False
        interruption_context = {}
        
        # Detect if this transcript came after an interruption
        if self._last_interrupt_time and (timestamp - self._last_interrupt_time < 3.0):
            was_interrupted = True
            interruption_context = {
                "was_interrupted": True,
                "user_interruptions": self._session_interruptions,
                "session_id": self._current_session_id
            }
            # Add timestamps if available (but limit to avoid overflow)
            if self._interruption_timestamps:
                # Only include the last 10 timestamps at most
                interruption_context["interruption_timestamps"] = [
                    round(ts - self._session_start_time, 2) # Store as relative time from session start
                    for ts in self._interruption_timestamps[-10:]
                ]
        
        # Track low confidence separately
        if confidence < 0.7:
            self.logger.info(f"Low confidence transcript: {confidence:.2f} - {text}")
            # could add to interruption_context but not needed for now
        
        # Process transcript through registered handler
        transcript_sequence = self._transcript_sequence
        self._transcript_sequence += 1
        
        # Log transcript with interruption details if present
        if was_interrupted:
            self.logger.info(f"Processing transcript [{transcript_sequence}] (interrupted): {text}")
        else:
            self.logger.info(f"Processing transcript [{transcript_sequence}]: {text}")
        
        # If a handler is registered, call it with the transcript and interruption data
        if self._transcript_handler:
            try:
                # Pass transcript and interruption metadata to handler
                await self._transcript_handler(
                    text, 
                    transcript_sequence=transcript_sequence,
                    timestamp=timestamp,
                    confidence=confidence,
                    **interruption_context
                )
            except Exception as e:
                self.logger.error(f"Error in transcript handler: {e}", exc_info=True)
        
        # Publish transcript with updated info
        await self.publish_transcription(
            text=text,
            sender="user",
            is_final=True,
            metadata={
                "confidence": confidence,
                "sequence": transcript_sequence,
                "timestamp": timestamp,
                **interruption_context
            }
        )
        
        return True

    async def start_speaking(self, tts_task: asyncio.Task) -> None:
        """Begin a TTS task with cancellation of previous tasks if needed."""
        if self._current_tts_task and not self._current_tts_task.done():
            if hasattr(self._current_tts_task, 'text'):
                self._last_tts_text = getattr(self._current_tts_task, 'text')
            self._current_tts_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._current_tts_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        task_name = f"tts_{time.time()}"
        self._tasks[task_name] = tts_task
        self._current_tts_task = tts_task
        
        if hasattr(tts_task, 'text'):
            self._last_tts_text = getattr(tts_task, 'text')
        
        self._interrupt_requested_event.clear()
        self._interrupt_handled_event.set()
        
        await self.transition_to(VoiceState.SPEAKING)

        try:
            await tts_task
            self.logger.info("TTS task completed normally")
            if self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_complete"})
        except asyncio.CancelledError:
            self.logger.info("TTS task was cancelled")
            if self._state == VoiceState.INTERRUPTED:
                self._interrupt_handled_event.set()
            elif self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_cancelled"})
        except Exception as e:
            self.logger.error(f"Error in TTS task: {e}", exc_info=True)
            await self.register_error(e, "tts")
        finally:
            self._tasks.pop(task_name, None)
            self._current_tts_task = None
            if self._state == VoiceState.SPEAKING:
                await self.transition_to(VoiceState.LISTENING, {"reason": "tts_completion"})

    def tts_session(self, text: str):
        """Context manager for TTS sessions with minimal intrusion."""
        if not self._tts_context_manager_initialized:
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
                        await self.state_manager.transition_to(
                            VoiceState.LISTENING, 
                            {"reason": "tts_cancelled"}
                        )
                    elif exc_type is not None:
                        await self.state_manager.register_error(exc_val, "tts_session")
                    elif self.state_manager.current_state == VoiceState.SPEAKING:
                        await self.state_manager.transition_to(
                            VoiceState.LISTENING, 
                            {"reason": "tts_complete"}
                        )
                    return False
            
            self._TTSSession = TTSSession
            self._tts_context_manager_initialized = True
        
        return self._TTSSession(text)

    async def register_error(self, error: Exception, source: str) -> None:
        """Transition to ERROR state with error details."""
        error_str = str(error)
        self.logger.error(f"Error in {source}: {error_str}", exc_info=True)
        
        self._last_error = error_str
        await self.transition_to(VoiceState.ERROR, {"error": error_str, "source": source})
        
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
        
        await self.emit("error", {"error": error, "source": source})
        
        await asyncio.sleep(2.0)
        if self._state == VoiceState.ERROR:
            await self.transition_to(VoiceState.LISTENING, {"reason": "error_recovery"})

    async def finish_processing(self) -> None:
        """Finish processing and go back to LISTENING."""
        if self._state == VoiceState.PROCESSING:
            self.logger.info("Processing complete, transitioning to LISTENING")
            await self.transition_to(VoiceState.LISTENING, {"reason": "processing_complete"})
        else:
            self.logger.warning(f"Called finish_processing while in {self._state.name} state")

    async def wait_for_interrupt(self, timeout: Optional[float] = None) -> bool:
        """Wait for an interrupt."""
        try:
            if timeout:
                return await asyncio.wait_for(self._interrupt_handled_event.wait(), timeout)
            else:
                await self._interrupt_handled_event.wait()
                return True
        except asyncio.TimeoutError:
            return False

    async def publish_transcription(
        self, 
        text: str, 
        sender: str = "user", 
        is_final: bool = True,
        participant_identity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Publish transcript to LiveKit using data channel and optional transcription API."""
        if not text or not text.strip():
            return False
        
        dedup_key = f"{sender}:{text[:50]}"
        current_time = time.time()
        
        # Check duplicates
        for past_key, timestamp in list(self._recent_transcripts.items()):
            if current_time - timestamp > 5.0:
                self._recent_transcripts.pop(past_key, None)
            elif past_key == dedup_key and current_time - timestamp < 2.0:
                self.logger.warning(f"Skipping duplicate transcript publish: '{text[:30]}...' from {sender}")
                return False
        
        self._recent_transcripts[dedup_key] = current_time
        
        if not self._room or not self._room.local_participant:
            self.logger.error("No room or participant for transcription publishing.")
            return False
        
        seq = self._transcript_sequence
        self._transcript_sequence += 1
        success = True
        
        identity_to_use = participant_identity or self._room.local_participant.identity
        self.logger.debug(
            f"Transcript identity resolution: sender='{sender}', "
            f"participant_identity={participant_identity}, "
            f"resolved_identity='{identity_to_use}'"
        )
        
        try:
            # 1) Data channel
            transcript_data = {
                "type": "transcript",
                "text": text,
                "sender": sender,
                "participant_identity": identity_to_use,
                "sequence": seq,
                "timestamp": time.time(),
                "is_final": is_final
            }
            if metadata:
                transcript_data.update(metadata)
            self.logger.info(f"Publishing transcript via data channel: '{text[:30]}...' from {sender}")
            data_success = await self._publish_with_retry(
                json.dumps(transcript_data).encode(),
                f"{sender} transcript"
            )
            if not data_success:
                self.logger.warning(f"Failed to publish {sender} transcript via data channel")
                success = False
                
            # 2) Transcription API
            try:
                track_sid = None
                if sender == "user":
                    # Search remote participant track
                    self.logger.info(f"Looking for track SID for user with identity: {identity_to_use}")
                    remote_participants = list(self._room.remote_participants.values())
                    for participant in remote_participants:
                        if participant.identity == identity_to_use:
                            track_pubs = list(participant.track_publications.values())
                            for pub in track_pubs:
                                if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.sid:
                                    track_sid = pub.sid
                                    break
                            if track_sid:
                                break
                else:
                    # Assistant => local track
                    local_track_pubs = list(self._room.local_participant.track_publications.values())
                    for pub in local_track_pubs:
                        if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.sid:
                            track_sid = pub.sid
                            break
                            
                if not track_sid:
                    self.logger.warning(f"No audio track SID found for {sender}, using data channel only")
                    return data_success
                
                segment_id = str(uuid.uuid4())
                current_time_ms = int(time.time() * 1000)
                
                trans = rtc.Transcription(
                    participant_identity=identity_to_use,
                    track_sid=track_sid,
                    segments=[
                        rtc.TranscriptionSegment(
                            id=segment_id,
                            text=text,
                            start_time=current_time_ms,
                            end_time=current_time_ms,
                            final=is_final,
                            language="en"
                        )
                    ]
                )
                await self._room.local_participant.publish_transcription(trans)
            except Exception as e:
                self.logger.warning(f"Failed transcription API publish: {e}")
                success = False
        except Exception as e:
            self.logger.error(f"Failed to publish transcript: {e}", exc_info=True)
            success = False
        
        return success

    def register_transcript_handler(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """Register a transcript handler."""
        self.logger.info("Registering transcript handler")
        self._transcript_handler = handler

    async def cleanup(self) -> None:
        """Clean up all resources and tasks."""
        self.logger.info("Cleaning up voice state manager resources")
        await self._cancel_active_tasks()
        await self.cleanup_tts_track()
        
        self._state = VoiceState.IDLE
        self._last_tts_text = None
        self._recent_processed_transcripts.clear()
        self._interrupt_requested_event.clear()
        self._interrupt_handled_event.set()
        self._event_handlers.clear()
        
        self.logger.info("Voice state manager cleanup completed")

    async def cleanup_tts_track(self) -> None:
        """Clean up TTS track resources."""
        try:
            if self._tts_track and self._room and self._room.local_participant:
                try:
                    self.logger.info("Unpublishing TTS track")
                    await self._room.local_participant.unpublish_track(self._tts_track)
                except Exception as e:
                    self.logger.warning(f"Error unpublishing TTS track: {e}")
            
            if self._tts_source:
                try:
                    self.logger.info("Closing TTS audio source")
                    if hasattr(self._tts_source, 'aclose'):
                        await self._tts_source.aclose()
                    elif hasattr(self._tts_source, 'close'):
                        self._tts_source.close()
                except Exception as e:
                    self.logger.warning(f"Error closing TTS source: {e}")
        except Exception as e:
            self.logger.error(f"Error during TTS track cleanup: {e}", exc_info=True)
        finally:
            self._tts_track = None
            self._tts_source = None
            self.logger.info("TTS track cleanup complete")

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get the history of state transitions."""
        return self._state_history.copy()
        
    def get_current_state(self) -> VoiceState:
        """Get the current state."""
        return self._state

    def get_analytics(self) -> Dict[str, Any]:
        """Gather analytics on state transitions and transcripts."""
        if not self._state_history:
            return {}
        
        state_durations: Dict[str, float] = {}
        state_counts: Dict[str, int] = {}
        for i, transition in enumerate(self._state_history):
            s = transition["from"]
            next_time = (
                self._state_history[i+1]["timestamp"]
                if i < len(self._state_history) - 1 else time.time()
            )
            duration = next_time - transition["timestamp"]
            state_durations[s] = state_durations.get(s, 0.0) + duration
            state_counts[s] = state_counts.get(s, 0) + 1
        
        interruption_count = sum(1 for t in self._state_history if t["to"] == VoiceState.INTERRUPTED.name)
        speaking_count = sum(1 for t in self._state_history if t["to"] == VoiceState.SPEAKING.name)
        interruption_rate = interruption_count / max(speaking_count, 1)
        
        proc_durations = []
        for i, t in enumerate(self._state_history):
            if t["to"] == VoiceState.PROCESSING.name and i < len(self._state_history) - 1:
                nd = self._state_history[i+1]["timestamp"] - t["timestamp"]
                proc_durations.append(nd)
        avg_proc_time = sum(proc_durations) / max(len(proc_durations), 1)
        
        ui_stats = {
            "publish_attempts": self._publish_stats["attempts"],
            "publish_success_rate": self._publish_stats["successes"] / max(self._publish_stats["attempts"], 1),
            "publish_failure_rate": self._publish_stats["failures"] / max(self._publish_stats["attempts"], 1),
            "publish_retries": self._publish_stats["retries"],
            "last_error": self._publish_stats.get("last_error"),
            "last_successful_publish": self._publish_stats.get("last_successful_publish", 0)
        }
        
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
        """Publish data to LiveKit with retries."""
        if not self._room or not self._room.local_participant:
            self.logger.warning(f"Cannot publish {description}: no room or local participant")
            return False
        
        self._publish_stats["attempts"] += 1
        
        retries = 0
        while retries <= max_retries:
            try:
                await self._room.local_participant.publish_data(data, reliable=True)
                self._publish_stats["successes"] += 1
                self._publish_stats["last_successful_publish"] = time.time()
                if retries > 0:
                    self.logger.debug(f"Successfully published {description} after {retries} retries")
                    self._publish_stats["retries"] += retries
                return True
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    self._publish_stats["failures"] += 1
                    self._publish_stats["last_error"] = str(e)
                    self.logger.error(f"Failed to publish {description} after {max_retries} attempts: {e}")
                    return False
                self.logger.warning(f"Retry {retries}/{max_retries} publishing {description}: {e}")
                await asyncio.sleep(self._retry_delay * retries)

    async def publish_state_update(self, state_data: dict) -> None:
        """Publish a state update to LiveKit."""
        try:
            if not self._room or not self._room.local_participant:
                self.logger.warning("Cannot publish state update: no room or participant")
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
        """Publish error to LiveKit."""
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

    async def _publish_agent_status(self) -> None:
        """Publish agent status update to the room."""
        if not self._room or not self._room.local_participant:
            return
        
        try:
            current_state = self._state
            
            # Add state-specific details
            status_details = {}
            if current_state == VoiceState.PAUSED:
                # For PAUSED state, add pause reason and timestamp
                status_details = {
                    "pause_reason": self._state_metadata.get("reason", "unknown"),
                    "interrupt_type": self._state_metadata.get("interrupt_type", "soft"),
                    "pause_start": self._state_metadata.get("pause_start_time", time.time())
                }
            
            # Format the full status message
            status_message = {
                "type": "agent-status",
                "status": self._get_status_for_ui(current_state),
                "timestamp": time.time(),
                **status_details  # Include any state-specific details
            }
            
            await self._publish_with_retry(
                json.dumps(status_message).encode(),
                "agent status"
            )
        except Exception as e:
            self.logger.error(f"Failed to publish agent status: {e}", exc_info=True)

    async def _monitor_paused_state(self, timeout: float = 0.8) -> None:
        """Monitor the PAUSED state for continued speech or resumption.
        
        Args:
            timeout: The amount of time to wait before resuming TTS if no further user speech is detected.
        """
        try:
            self.logger.info(f"Monitoring PAUSED state for {timeout}s")
            await asyncio.sleep(timeout)
            
            # Check if we're still in PAUSED state after the timeout
            if self._state == VoiceState.PAUSED:
                # Get interrupt type from metadata
                interrupt_type = self._state_metadata.get("interrupt_type", "soft")
                if interrupt_type == "soft":
                    # For soft interrupts, we want to resume TTS if no further speech is detected
                    self.logger.info("No further speech detected during pause period - resuming TTS")
                    if self._tts_service and hasattr(self._tts_service, "resume"):
                        try:
                            await self._tts_service.resume()
                            # Stay in SPEAKING state if resume was successful
                            await self.transition_to(VoiceState.SPEAKING, {"reason": "resumed_from_pause"})
                            return
                        except Exception as e:
                            self.logger.error(f"Failed to resume TTS: {e}")
                
                # If we can't resume or it's not a soft interrupt, transition to LISTENING
                self.logger.info("Transitioning to LISTENING after pause timeout")
                await self.transition_to(VoiceState.LISTENING, {"reason": "pause_timeout"})
        except asyncio.CancelledError:
            self.logger.info("Pause monitor task cancelled")
        except Exception as e:
            self.logger.error(f"Error in pause monitor: {e}", exc_info=True)
