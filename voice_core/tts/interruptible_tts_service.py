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
        # 20ms (960 samples) per chunk for smoother playback
        self.samples_per_chunk = 960  # 20ms at 48kHz
        self.log_interval = 10  # Log every 10 chunks
        
        # Audio buffering parameters for smoother playback
        self.mp3_buffer_size = 8000  # Increased from 4000 to 8000 bytes
        self.frame_yield_interval = 8  # Only yield every 8 chunks (increased from 3)
        self.prebuffer_frames = 3  # Number of frames to prebuffer before starting playback
        
        # Interruption handling
        self.interrupt_check_interval = 0.4  # Increased from 0.25s to 0.4s
        
        # Background processing
        self._conversion_queue = asyncio.Queue(maxsize=100)  # Limit queue size to avoid memory issues
        self._worker_running = True
        self._worker_task = None  # Will be created during initialization
        
        # Pre-allocate common frames to avoid repeated allocations
        self._silence_frame = None
        
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
        """Initialize TTS service and load necessary resources."""
        self.logger.info("Initializing TTS service...")
        
        # Start the background conversion worker
        if self._worker_task is None or self._worker_task.done():
            self._worker_running = True
            self._worker_task = asyncio.create_task(self._conversion_worker())
            self.logger.info("Started background conversion worker")
        
        # Pre-allocate silence frame for quick interrupt response
        self._silence_frame = rtc.AudioFrame(
            data=b'\x00' * self.samples_per_chunk * 2,  # 2 bytes per sample
            samples_per_channel=self.samples_per_chunk,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels
        )
        
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
        Using incremental processing and background conversion worker.
        
        Args:
            text: Text to speak
            assistant_identity: The identity to use for assistant transcripts (default: "assistant")
            should_publish_transcript: Whether to publish transcript
            
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
                
                # Create Edge TTS communicate instance
                communicate = edge_tts.Communicate(text, self.voice)
                
                # Track stats
                chunks_processed = 0
                last_check_time = time.time()
                processing_started = False
                # Buffer to collect small chunks for smoother playback
                mp3_chunk_buffer = bytearray()
                
                # Pre-buffering setup
                prebuffer_count = 0
                prebuffering = True
                pcm_prebuffer = []
                
                # Stream with incremental processing
                async for chunk in communicate.stream():
                    # First chunk means we started processing
                    if not processing_started:
                        processing_started = True
                        self.logger.info("Started processing TTS stream")
                    
                    # Regular interruption check at intervals (less frequent now)
                    current_time = time.time()
                    if current_time - last_check_time > self.interrupt_check_interval:
                        if await self.check_interruption() or not self._active:
                            self.logger.info("Interruption detected during streaming")
                            return text
                        last_check_time = current_time
                    
                    # Process audio chunks with batching for smoother playback
                    if chunk["type"] == "audio":
                        # Accumulate chunks to reduce processing overhead
                        mp3_chunk_buffer.extend(chunk["data"])
                        
                        # Process in larger batches for smoother playback
                        # Edge-TTS typically produces very small chunks, combining them improves efficiency
                        if len(mp3_chunk_buffer) >= self.mp3_buffer_size or (chunks_processed > 0 and chunks_processed % 10 == 0):
                            # Create a future to receive the processed result
                            result_future = asyncio.Future()
                            
                            # Queue the conversion work
                            await self._conversion_queue.put((bytes(mp3_chunk_buffer), result_future))
                            mp3_chunk_buffer = bytearray()  # Reset buffer after sending
                            
                            # Await the result with timeout
                            try:
                                pcm_data = await asyncio.wait_for(result_future, timeout=0.6)
                                
                                # Process the PCM data if valid
                                if pcm_data is not None:
                                    if prebuffering:
                                        # During prebuffering phase, collect frames
                                        pcm_prebuffer.append(pcm_data)
                                        prebuffer_count += 1
                                        
                                        # Once we have enough prebuffered frames, send them all
                                        if prebuffer_count >= self.prebuffer_frames:
                                            prebuffering = False
                                            self.logger.debug(f"Prebuffering complete, sending {len(pcm_prebuffer)} frames")
                                            for buffered_data in pcm_prebuffer:
                                                await self._send_pcm_frames(buffered_data, yield_after=False)
                                            # Clear prebuffer after sending
                                            pcm_prebuffer = []
                                    else:
                                        # Normal processing after prebuffering
                                        await self._send_pcm_frames(pcm_data)
                                    
                                chunks_processed += 1
                                
                                # Yield less frequently to reduce jitter
                                if not prebuffering and chunks_processed % self.frame_yield_interval == 0:
                                    await asyncio.sleep(0.001)  # Very brief yield
                            except asyncio.TimeoutError:
                                # Log but continue
                                self.logger.debug("Worker taking longer than expected")
                    
                # Process any remaining audio in the buffer
                if len(mp3_chunk_buffer) > 0:
                    result_future = asyncio.Future()
                    await self._conversion_queue.put((bytes(mp3_chunk_buffer), result_future))
                    
                    try:
                        pcm_data = await asyncio.wait_for(result_future, timeout=0.6)
                        if pcm_data is not None:
                            if prebuffering:
                                # Send all prebuffered frames first
                                for buffered_data in pcm_prebuffer:
                                    await self._send_pcm_frames(buffered_data, yield_after=False)
                                # Then send final data
                                await self._send_pcm_frames(pcm_data)
                            else:
                                await self._send_pcm_frames(pcm_data)
                    except asyncio.TimeoutError:
                        self.logger.debug("Timeout processing final audio chunk")
                
                self.logger.info(f"TTS streaming complete, processed {chunks_processed} chunks")
                
                # Call completion callback if provided and not interrupted
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

    async def _send_pcm_frames(self, pcm_data: np.ndarray, yield_after: bool = True) -> None:
        """Split and send PCM data in appropriately sized frames for smooth playback.
        
        Args:
            pcm_data: PCM audio data as numpy array
            yield_after: Whether to yield to event loop after sending all frames
        """
        if pcm_data is None or len(pcm_data) == 0:
            return
            
        # Use larger chunk size for smoother playback (20ms chunks)
        chunk_size = self.samples_per_chunk  # 960 samples (20ms at 48kHz)
        
        # Process in properly sized chunks
        for start_idx in range(0, len(pcm_data), chunk_size):
            # Get chunk with padding if needed
            end_idx = min(start_idx + chunk_size, len(pcm_data))
            chunk = pcm_data[start_idx:end_idx]
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Create and send audio frame
            frame = rtc.AudioFrame(
                data=(chunk * 32767.0).astype(np.int16).tobytes(),
                samples_per_channel=len(chunk),
                sample_rate=self.sample_rate,
                num_channels=self.num_channels
            )
            
            # Send frame via state manager
            if self.state_manager._tts_source:
                await self.state_manager._tts_source.capture_frame(frame)
        
        # Optional yield based on parameter
        # This allows controlled yielding for different scenarios
        if yield_after:
            await asyncio.sleep(0.001)  # Brief sleep instead of sleep(0)

    async def _conversion_worker(self) -> None:
        """Background worker to convert MP3 chunks to PCM efficiently."""
        self.logger.info("Conversion worker started")
        
        while self._worker_running:
            try:
                # Get next item from queue with timeout
                try:
                    mp3_data, result_future = await asyncio.wait_for(
                        self._conversion_queue.get(), 
                        timeout=1.0  # Longer wait for new work (1 second)
                    )
                except asyncio.TimeoutError:
                    # No work available, check if we should continue
                    if not self._worker_running:
                        break
                    continue
                
                # Process the MP3 chunk
                try:
                    pcm_result = await self._convert_mp3_chunk_to_pcm(mp3_data)
                    
                    # Set the result if the future wasn't cancelled
                    if not result_future.done():
                        result_future.set_result(pcm_result)
                        
                except Exception as e:
                    # Handle conversion errors
                    self.logger.error(f"Error in conversion worker: {e}")
                    if not result_future.done():
                        result_future.set_exception(e)
                        
                # Mark task as done
                self._conversion_queue.task_done()
                
            except asyncio.CancelledError:
                # Worker cancelled
                self.logger.info("Conversion worker cancelled")
                break
                
            except Exception as e:
                # Unexpected error in worker
                self.logger.error(f"Unexpected error in conversion worker: {e}", exc_info=True)
                # Continue running for robustness
                await asyncio.sleep(0.1)  # Brief pause before continuing
        
        self.logger.info("Conversion worker stopped")

    async def _convert_mp3_chunk_to_pcm(self, mp3_data: bytes) -> np.ndarray:
        """Convert an MP3 chunk to PCM without using temporary files."""
        try:
            # Use BytesIO instead of temp files for in-memory processing
            mp3_buffer = io.BytesIO(mp3_data)
            
            # Load audio data with pydub directly from buffer
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            
            # Convert to target format
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(self.num_channels)
            
            # Get raw PCM data as numpy array
            pcm_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            pcm_data = pcm_data / 32768.0
            
            return pcm_data
                
        except Exception as e:
            self.logger.error(f"Error converting MP3 chunk to PCM: {e}")
            return None

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
        """Clean up TTS resources and stop worker."""
        self.logger.info("Cleaning up TTS service")
        
        # Stop the worker task
        self._worker_running = False
        if self._worker_task and not self._worker_task.done():
            try:
                # Cancel and wait for worker to complete
                self._worker_task.cancel()
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                self.logger.error(f"Error stopping worker: {e}")
        
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