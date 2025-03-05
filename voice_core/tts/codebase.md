# __init__.py

```py
"""Text-to-Speech services for voice agent."""

from .edge_tts_plugin import EdgeTTSTTS

__all__ = ['EdgeTTSTTS']

```

# edge_tts_plugin.py

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

# interruptible_tts_service.py

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
        self.interruption_check_interval = 25  # Check every 25 chunks (500ms)
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
                
                # Use larger chunks for better stability while maintaining responsiveness
                chunk_size = min(480, self.samples_per_chunk)  # 10ms chunks
                
                for start_idx in range(0, total_samples, chunk_size):
                    # Check for interruption periodically instead of every chunk
                    if chunk_count % self.interruption_check_interval == 0:
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

# livekit_tts_service.py

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

# tts_forwarder.py

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

# tts_segments_forwarder.py

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

# tts_utils.py

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

