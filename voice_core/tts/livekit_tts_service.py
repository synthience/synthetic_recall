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