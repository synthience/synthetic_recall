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