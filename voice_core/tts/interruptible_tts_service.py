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
    Enhanced TTS service with true async interruptibility for EdgeTTS.
    Uses incremental MP3-to-PCM conversion in memory to simulate streaming.
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
        Initialize the TTS service with asyncio.Event-based interruption.
        
        Args:
            state_manager: Voice state manager
            voice: EdgeTTS voice to use
            sample_rate: Audio sample rate (default 48kHz)
            num_channels: Number of audio channels (default mono=1)
            on_interrupt: Callback when TTS is interrupted
            on_complete: Callback when TTS finishes normally
        """
        self.state_manager = state_manager
        self.voice = voice
        self.sample_rate = (
            sample_rate if state_manager._tts_source is None
            else state_manager._tts_source.sample_rate
        )
        self.num_channels = (
            num_channels if state_manager._tts_source is None
            else state_manager._tts_source.num_channels
        )
        self.on_interrupt = on_interrupt
        self.on_complete = on_complete
        
        # LiveKit room
        self.room: Optional[rtc.Room] = None
        
        # Internal playback state
        self._active = False
        self._cancellable = True
        self._current_task: Optional[asyncio.Task] = None
        self._playback_lock = asyncio.Lock()
        
        # Audio / Streaming parameters
        self.samples_per_chunk = 960         
        self.log_interval = 10              
        self.mp3_buffer_size = 8192         
        self.frame_yield_interval = 10      
        self.yield_duration = 0.0005        
        # Reduced interrupt check interval for faster detection
        self.interrupt_check_interval = 0.1  

        # Async interruption mechanism
        self._interrupt_requested_event = asyncio.Event()
        self._interrupt_handled_event = asyncio.Event()
        self._interrupt_handled_event.set()  # Start in 'handled' state

        # Stats
        self.start_time = 0
        self.frame_count = 0
        self.interruptions_handled = 0

        # Conversion worker
        self._conversion_queue = asyncio.Queue(maxsize=100)
        self._worker_running = True
        self._worker_task: Optional[asyncio.Task] = None

        # Silence frame
        self._silence_frame: Optional[rtc.AudioFrame] = None

        # For soft-stop logic
        self._soft_stop_requested_event = asyncio.Event()
        self._fade_out_in_progress = False
        self._resumed_after_soft_stop = False

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"InterruptibleTTSService initialized with voice: {voice}")

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Attach to the state manager's interrupt_requested event, if any."""
        @self.state_manager.on("interrupt_requested")
        async def handle_interrupt(_data):
            self.logger.info("Received 'interrupt_requested' event from state manager. Stopping TTS.")
            await self.stop()

    async def initialize(self) -> None:
        """Initialize TTS resources and start conversion worker."""
        self.logger.info("Initializing TTS service...")

        # Start conversion worker
        if not self._worker_task or self._worker_task.done():
            self._worker_running = True
            self._worker_task = asyncio.create_task(self._conversion_worker())
            self.logger.info("Started background conversion worker")

        # Silence frame
        self._silence_frame = rtc.AudioFrame(
            data=b'\x00' * self.samples_per_chunk * 2,
            samples_per_channel=self.samples_per_chunk,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels
        )

        # Verify EdgeTTS voice
        try:
            voices = await edge_tts.list_voices()
            voice_names = [v["ShortName"] for v in voices]
            if self.voice not in voice_names:
                self.logger.warning(f"Voice '{self.voice}' not found. Using fallback voice.")
                self.voice = "en-US-GuyNeural"
            self.logger.info(f"Using TTS voice: {self.voice}")
            
            # Notify UI
            if self.room and self.state_manager:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "tts_initialized",
                        "voice": self.voice,
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
        except Exception as e:
            self.logger.error(f"Error listing voices from EdgeTTS: {e}", exc_info=True)

        self.logger.info("TTS service initialized")

    async def set_room(self, room: rtc.Room) -> None:
        """Set LiveKit room for TTS output."""
        if not room:
            self.logger.error("set_room called with None")
            return
        self.room = room
        if self.state_manager:
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "tts_ready",
                    "voice": self.voice,
                    "timestamp": time.time()
                }).encode(),
                reliable=True
            )

    async def speak(self, text: str, assistant_identity: str = "assistant") -> Optional[str]:
        """Speak the given text, streaming TTS to LiveKit."""
        if not text.strip():
            self.logger.warning("Empty text, not speaking.")
            return None

        self._active = True
        await self._publish_transcription(text, assistant_identity, final=True)

        self.logger.info(f"Speaking (TTS): '{text[:50]}...'")
        task = asyncio.create_task(self._stream_tts(text, assistant_identity))
        setattr(task, 'text', text)
        self._current_task = task

        try:
            return await task
        except asyncio.CancelledError:
            self.logger.info("speak() task was cancelled.")
            return None

    async def _publish_transcription(self, text: str, assistant_identity: str, final: bool):
        """Helper to publish transcript data via state manager or fallback."""
        if not self.room or not self.room.local_participant:
            return
        self.frame_count += 1
        seq = self.frame_count

        if self.state_manager:
            try:
                await self.state_manager.publish_transcription(
                    text, 
                    "assistant", 
                    final, 
                    participant_identity=assistant_identity
                )
                return
            except Exception as e:
                self.logger.error(f"Failed to publish transcript via state_manager: {e}")

        # Fallback
        try:
            await self.room.local_participant.publish_data(
                json.dumps({
                    "type": "transcript",
                    "text": text,
                    "sender": "assistant",
                    "participant_identity": assistant_identity,
                    "final": final,
                    "sequence": seq,
                    "timestamp": time.time()
                }).encode(),
                reliable=True
            )
        except Exception as e:
            self.logger.error(f"Failed to publish transcription fallback: {e}")

    async def _stream_tts(self, text: str, assistant_identity: str) -> str:
        """Incremental MP3->PCM streaming, checking interrupts more frequently."""
        async with self._playback_lock:
            try:
                self._active = True
                self.start_time = time.time()
                
                # Quick check
                if await self._check_interruption():
                    return text

                # Create EdgeTTS instance
                communicate = edge_tts.Communicate(text, self.voice)
                mp3_buffer = bytearray()
                chunks_processed = 0
                last_check_time = time.time()

                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_buffer.extend(chunk["data"])

                        # If buffer large or at intervals
                        if len(mp3_buffer) >= self.mp3_buffer_size or (chunks_processed > 0 and chunks_processed % 6 == 0):
                            result_future = asyncio.Future()
                            await self._conversion_queue.put((bytes(mp3_buffer), result_future))
                            mp3_buffer = bytearray()

                            # Wait for conversion
                            try:
                                pcm_data = await asyncio.wait_for(result_future, timeout=0.6)
                                if pcm_data is not None:
                                    await self._send_pcm_frames(pcm_data)
                            except asyncio.TimeoutError:
                                self.logger.debug("Conversion worker took too long.")
                            
                            chunks_processed += 1

                            # Check interrupt frequently
                            if time.time() - last_check_time > self.interrupt_check_interval:
                                if await self._check_interruption():
                                    return text
                                last_check_time = time.time()

                # Leftover buffer
                if mp3_buffer:
                    result_future = asyncio.Future()
                    await self._conversion_queue.put((bytes(mp3_buffer), result_future))
                    try:
                        pcm_data = await asyncio.wait_for(result_future, timeout=0.6)
                        if pcm_data is not None:
                            await self._send_pcm_frames(pcm_data)
                    except asyncio.TimeoutError:
                        self.logger.debug("Timeout converting final chunk.")

                self.logger.info("TTS streaming complete.")
                
                # If not interrupted, call on_complete
                if not await self._check_interruption() and self.on_complete:
                    if asyncio.iscoroutinefunction(self.on_complete):
                        await self.on_complete(text)
                    else:
                        self.on_complete(text)
                return text

            except asyncio.CancelledError:
                self.logger.info("TTS streaming was cancelled.")
                raise
            except Exception as e:
                self.logger.error(f"Error in TTS streaming: {e}", exc_info=True)
                return text
            finally:
                self._active = False
                self._cancellable = False

    async def request_soft_stop(self) -> None:
        """Request a soft stop for TTS playback.
        This will create a fade-out effect rather than an immediate hard stop.
        """
        self.logger.info("Soft stop requested for TTS - fading out")
        self._soft_stop_requested_event.set()
        
    async def resume(self) -> None:
        """Resume TTS playback after a soft stop.
        Will only work if the playback was paused and not fully stopped.
        """
        if not self._is_playing and not self._stopped and self._last_audio_chunk is not None:
            self.logger.info("Resuming TTS playback after soft stop")
            # Clear the soft stop flag
            self._soft_stop_requested_event.clear()
            # Set the resumed flag to restore normal volume
            self._resumed_after_soft_stop = True
            # Resume the player if it was paused
            self._is_playing = True
            return True
        else:
            self.logger.warning("Cannot resume TTS - was not in a resumable state")
            return False

    async def _check_interruption(self) -> bool:
        """Check if TTS should be interrupted (hard or soft)."""
        if not self._active:
            return True  # Already inactive
        # Hard interrupt from event
        if (self._interrupt_requested_event.is_set() or
            (self.state_manager and self.state_manager._interrupt_requested_event.is_set())):
            self.logger.info("Hard interrupt detected, stopping TTS immediately.")
            self.interruptions_handled += 1
            self._active = False

            self._interrupt_requested_event.clear()
            if self.state_manager:
                self.state_manager._interrupt_requested_event.clear()

            # Fire on_interrupt callback
            if self.on_interrupt:
                if asyncio.iscoroutinefunction(self.on_interrupt):
                    await self.on_interrupt()
                else:
                    self.on_interrupt()

            self._interrupt_handled_event.set()
            if self.state_manager:
                self.state_manager._interrupt_handled_event.set()
            return True
        
        # Soft stop
        if self._soft_stop_requested_event.is_set() and not self._fade_out_in_progress:
            self.logger.info("Soft stop requested -> initiating fade-out.")
            self._fade_out_in_progress = True
            self.interruptions_handled += 1

            # Fire on_interrupt
            if self.on_interrupt:
                if asyncio.iscoroutinefunction(self.on_interrupt):
                    await self.on_interrupt()
                else:
                    self.on_interrupt()
            
            # We'll do a short fade out in _send_pcm_frames when next chunk is processed
            self._soft_stop_requested_event.clear()
            return False
        return False

    async def _send_pcm_frames(self, pcm_data: np.ndarray) -> None:
        """Send PCM frames to TTS track, with optional fade-out on soft stop."""
        if not self._active:
            return
        if not self.state_manager or not self.state_manager._tts_source:
            self.logger.warning("No TTS source available to send frames.")
            return
        
        chunk_size = self.samples_per_chunk
        local_frame_count = 0
        index = 0
        total_len = len(pcm_data)

        while index < total_len and self._active:
            end_idx = min(index + chunk_size, total_len)
            chunk = pcm_data[index:end_idx]
            index = end_idx

            # If fade-out in progress, fade the chunk
            if self._fade_out_in_progress:
                fade_steps = len(chunk)
                for i in range(fade_steps):
                    # linear fade from 1.0 to 0.0 across this chunk
                    factor = 1.0 - (i / fade_steps)
                    chunk[i] = chunk[i] * factor
                
                self.logger.info("Fade-out chunk applied.")
                # Once a chunk is faded, we can consider TTS inactive
                self._active = False
                self._fade_out_in_progress = False
                self._interrupt_handled_event.set()
                if self.state_manager:
                    self.state_manager._interrupt_handled_event.set()

            # If resumed after soft stop, ensure volume is normal
            if self._resumed_after_soft_stop:
                self.logger.debug("Restored normal volume after resume")
                self._resumed_after_soft_stop = False

            # Convert float32 [-1,1] -> int16
            int16_data = (chunk * 32767.0).astype(np.int16).tobytes()
            frame = rtc.AudioFrame(
                data=int16_data,
                samples_per_channel=len(chunk),
                sample_rate=self.sample_rate,
                num_channels=self.num_channels
            )

            await self.state_manager._tts_source.capture_frame(frame)
            local_frame_count += 1
            self.frame_count += 1

            # Yield occasionally
            if local_frame_count % self.frame_yield_interval == 0:
                await asyncio.sleep(self.yield_duration)
                if await self._check_interruption():
                    return

    async def _conversion_worker(self) -> None:
        """Background worker for MP3->PCM conversion."""
        self.logger.info("Conversion worker started.")
        while self._worker_running:
            try:
                mp3_data, result_future = await asyncio.wait_for(self._conversion_queue.get(), timeout=1.0)
                try:
                    pcm_data = await self._convert_mp3_to_pcm(mp3_data)
                    if not result_future.done():
                        result_future.set_result(pcm_data)
                except Exception as e:
                    if not result_future.done():
                        result_future.set_exception(e)
                finally:
                    self._conversion_queue.task_done()
            except asyncio.TimeoutError:
                if not self._worker_running:
                    break
            except asyncio.CancelledError:
                self.logger.info("Conversion worker cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in conversion worker: {e}", exc_info=True)
                await asyncio.sleep(0.1)
        self.logger.info("Conversion worker stopped.")

    async def _convert_mp3_to_pcm(self, mp3_data: bytes) -> Optional[np.ndarray]:
        """Convert MP3 to PCM via pydub, returning float32 in [-1,1]."""
        try:
            mp3_buffer = io.BytesIO(mp3_data)
            audio = AudioSegment.from_file(mp3_buffer, format="mp3")
            audio = audio.set_frame_rate(self.sample_rate).set_channels(self.num_channels)
            pcm = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            return pcm
        except Exception as e:
            self.logger.error(f"Error converting MP3 to PCM: {e}", exc_info=True)
            return None

    async def stop(self) -> None:
        """Stop TTS playback immediately (hard stop)."""
        self.logger.info("Stopping TTS playback immediately (hard stop).")
        self._active = False
        self._cancellable = False

        self._interrupt_requested_event.set()
        self._interrupt_handled_event.clear()

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # Send silence frame
        if self.state_manager and self.state_manager._tts_source:
            try:
                if not self._silence_frame:
                    self._silence_frame = rtc.AudioFrame(
                        data=b'\x00' * self.samples_per_chunk * 2,
                        samples_per_channel=self.samples_per_chunk,
                        sample_rate=self.sample_rate,
                        num_channels=self.num_channels
                    )
                await self.state_manager._tts_source.capture_frame(self._silence_frame)
            except Exception as e:
                self.logger.error(f"Error flushing TTS buffer: {e}")

        # Transition to LISTENING
        if self.state_manager:
            self.state_manager._interrupt_handled_event.set()
            await self.state_manager.transition_to(VoiceState.LISTENING, {
                "reason": "tts_stopped",
                "timestamp": time.time()
            })

    async def cleanup(self) -> None:
        """Clean up TTS resources and stop worker."""
        self.logger.info("Cleaning up TTS service.")
        self._worker_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                self.logger.error(f"Error stopping conversion worker: {e}")

        await self.stop()
        await self.state_manager.cleanup_tts_track()

        # Publish final cleanup
        if self.room and self.room.local_participant:
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({"type": "tts_cleanup", "timestamp": time.time()}).encode(),
                    reliable=True
                )
            except Exception as e:
                self.logger.error(f"Failed to publish TTS cleanup: {e}")

        self.logger.info("TTS cleanup complete.")

    def get_stats(self) -> dict:
        """Return TTS stats."""
        return {
            "active": self._active,
            "frame_count": self.frame_count,
            "voice": self.voice,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "interruptions_handled": self.interruptions_handled
        }
