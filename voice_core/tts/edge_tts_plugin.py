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