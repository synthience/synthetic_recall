"""LiveKit service implementation."""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
import numpy as np
import livekit.rtc as rtc
import jwt

from voice_core.shared_state import should_interrupt
from voice_core.audio import AudioFrame, normalize_audio, resample_audio
from voice_core.stt import EnhancedSTTPipeline, WhisperConfig
from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState

logger = logging.getLogger(__name__)

# LiveKit server configuration
LIVEKIT_URL = "ws://localhost:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"

def generate_token(room_name: str, identity: str = "bot") -> str:
    """Generate a LiveKit access token."""
    claims = {
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
            "roomAdmin": False,
            "roomCreate": True
        },
        "iss": LIVEKIT_API_KEY,
        "sub": identity,
        "exp": 4070908800,  # Some time in 2099
        "jti": room_name + "_" + identity,
    }
    return jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")

class LiveKitAudioTrack:
    """Enhanced LiveKit audio track with proper PCM handling."""
    
    def __init__(self, track: rtc.LocalTrack):
        self.track = track
        self.input_sample_rate = 16000  # Edge TTS native rate
        self.output_sample_rate = 48000  # LiveKit required rate
        self.channels = 1
        self.frame_duration = 20  # ms
        self.samples_per_frame = int(self.output_sample_rate * self.frame_duration / 1000)
        self.buffer = np.array([], dtype=np.float32)
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def write_frame(self, frame: AudioFrame):
        """Write audio frame with proper resampling and buffering."""
        async with self._lock:
            try:
                # Ensure data is float32 and normalized
                frame_data = frame.data
                if frame_data.dtype != np.float32:
                    frame_data = frame_data.astype(np.float32)
                frame_data = normalize_audio(frame_data)
                
                # Resample if needed
                if frame.sample_rate != self.output_sample_rate:
                    frame_data = resample_audio(
                        frame_data,
                        frame.sample_rate,
                        self.output_sample_rate
                    )
                
                # Add to buffer
                self.buffer = np.append(self.buffer, frame_data)
                
                # Process complete frames
                while len(self.buffer) >= self.samples_per_frame:
                    frame_samples = self.buffer[:self.samples_per_frame]
                    self.buffer = self.buffer[self.samples_per_frame:]
                    
                    # Convert to int16 for LiveKit
                    int16_data = (frame_samples * 32767).astype(np.int16)
                    
                    # Create LiveKit audio frame
                    rtc_frame = rtc.AudioFrame(
                        data=int16_data.tobytes(),
                        samples_per_channel=self.samples_per_frame,
                        sample_rate=self.output_sample_rate
                    )
                    
                    # Write to track
                    await self.track.write_frame(rtc_frame)
                    
            except Exception as e:
                self.logger.error(f"Error writing audio frame: {e}")
                raise

    async def cleanup(self):
        """Clean up resources and flush buffer."""
        async with self._lock:
            if len(self.buffer) > 0:
                # Pad last frame if needed
                remaining_samples = len(self.buffer)
                if remaining_samples < self.samples_per_frame:
                    padding = np.zeros(self.samples_per_frame - remaining_samples, dtype=np.float32)
                    self.buffer = np.append(self.buffer, padding)
                
                # Convert to int16 for LiveKit
                int16_data = (self.buffer * 32767).astype(np.int16)
                
                # Send final frame
                rtc_frame = rtc.AudioFrame(
                    data=int16_data.tobytes(),
                    samples_per_channel=len(self.buffer),
                    sample_rate=self.output_sample_rate
                )
                await self.track.write_frame(rtc_frame)
            
            self.buffer = np.array([], dtype=np.float32)

class LiveKitTransport:
    """LiveKit transport layer for voice pipeline."""
    
    def __init__(self):
        self.room = None
        self.logger = logging.getLogger(__name__)
        self._event_handlers: Dict[str, Callable] = {}
        
    async def connect_to_room(self, room_name: str) -> rtc.Room:
        """Connect to a LiveKit room."""
        try:
            # Create room if needed
            if not self.room:
                self.room = rtc.Room()
                
            # Connect to room
            token = generate_token(room_name)
            await self.room.connect(LIVEKIT_URL, token)
            
            self.logger.info(f"Connected to LiveKit room: {room_name}")
            return self.room
            
        except Exception as e:
            self.logger.error(f"Failed to connect to room: {e}")
            raise
        
    def on(self, event: str, callback: Optional[Callable] = None):
        """Register event handlers."""
        def decorator(func: Callable):
            self._event_handlers[event] = func
            return func
            
        if callback:
            self._event_handlers[event] = callback
            return callback
            
        return decorator
        
    def _emit(self, event: str, data: Any = None):
        """Emit an event to registered handlers."""
        if event in self._event_handlers:
            try:
                self._event_handlers[event](data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event}: {e}")

class LiveKitService:
    """LiveKit service for managing room connections and audio streaming."""
    
    def __init__(self, config: WhisperConfig, room: rtc.Room, state_manager: Optional[VoiceStateManager] = None):
        self.config = config
        self.room = room
        self.stt_service = EnhancedSTTPipeline(config)
        self.state_manager = state_manager or VoiceStateManager()
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._audio_tasks: Dict[str, asyncio.Task] = {}
        self._state = {
            'is_publishing': False,
            'active_tracks': set(),
            'error': None
        }
        
    async def publish_track(self, track_name: str, source: rtc.AudioSource) -> rtc.LocalAudioTrack:
        """Publish an audio track with state management and event emission."""
        try:
            if self._state['is_publishing']:
                raise RuntimeError("Already publishing a track")
                
            self._state['is_publishing'] = True
            
            # Notify state change
            await self.state_manager.transition_to(
                VoiceState.SPEAKING,
                {"track_name": track_name}
            )
            
            local_track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
            await self.room.local_participant.publish_track(local_track)
            
            self._state['active_tracks'].add(track_name)
            self.logger.info(f"Published track: {track_name}")
            
            return local_track
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to publish track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            raise
        finally:
            self._state['is_publishing'] = False
            
    async def subscribe_to_track(self, track: rtc.AudioTrack, participant_id: str):
        """Subscribe to a remote audio track with state coordination."""
        try:
            if participant_id in self._audio_tasks:
                return
                
            # Update state for new track
            await self.state_manager.transition_to(
                VoiceState.LISTENING,
                {"participant_id": participant_id}
            )
            
            task = asyncio.create_task(self._process_audio_track(track, participant_id))
            self._audio_tasks[participant_id] = task
            self._state['active_tracks'].add(participant_id)
            
            self.logger.info(f"Subscribed to track from participant: {participant_id}")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to subscribe to track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            raise
            
    def get_state(self) -> Dict[str, Any]:
        """Get current service state including voice state."""
        return {
            'is_publishing': self._state['is_publishing'],
            'active_tracks': list(self._state['active_tracks']),
            'error': self._state['error'],
            'running': self._running,
            'voice_state': self.state_manager.current_state.value
        }
        
    async def stop_track(self, track_id: str):
        """Stop processing a track with state cleanup."""
        try:
            if track_id in self._audio_tasks:
                task = self._audio_tasks.pop(track_id)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
            self._state['active_tracks'].discard(track_id)
            
            # Reset state if no active tracks
            if not self._state['active_tracks']:
                await self.state_manager.transition_to(VoiceState.IDLE)
                
            self.logger.info(f"Stopped track: {track_id}")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Failed to stop track: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )
            
    async def stop(self):
        """Stop all audio processing with state cleanup."""
        try:
            self._running = False
            tasks = list(self._audio_tasks.values())
            self._audio_tasks.clear()
            
            for task in tasks:
                task.cancel()
                
            await asyncio.gather(*tasks, return_exceptions=True)
            self._state['active_tracks'].clear()
            
            # Reset to idle state
            await self.state_manager.transition_to(VoiceState.IDLE)
            await self.cleanup()
            
            self.logger.info("Stopped all audio processing")
            
        except Exception as e:
            self._state['error'] = str(e)
            self.logger.error(f"Error during shutdown: {e}")
            await self.state_manager.transition_to(
                VoiceState.ERROR,
                {"error": str(e)}
            )

    async def _process_audio_track(self, track: rtc.AudioTrack, participant_id: str) -> None:
        """Process audio from a single track and publish transcripts."""
        if not self.room:
            self.logger.error("No LiveKit room provided for recognition.")
            return

        self.logger.info(f"Starting recognition for participant {participant_id}")
        try:
            audio_stream = rtc.AudioStream(track)
            async for event in audio_stream:
                if not self._running:
                    break
                try:
                    # Convert to float32 normalized audio
                    audio_np = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32)
                    audio_np /= 32767.0

                    transcript = await self.stt_service.process_audio(audio_np)
                    if transcript and transcript.strip():
                        self.logger.info(f"Final transcript for {participant_id}: {transcript}")
                        data = {
                            "type": "transcript",
                            "text": transcript,
                            "is_final": True,
                            "participant_id": participant_id,
                            "timestamp": time.time()
                        }
                        try:
                            await self.room.local_participant.publish_data(
                                json.dumps(data).encode("utf-8"),
                                reliable=True
                            )
                            self.logger.info("Published transcript to LiveKit.")
                        except Exception as e:
                            self.logger.error(f"Error publishing transcript: {e}")

                except Exception as e:
                    self.logger.error(f"Failed to process audio frame: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in recognition loop for {participant_id}: {e}", exc_info=True)
        finally:
            self.logger.info(f"Stopped recognition for participant {participant_id}")

    async def initialize(self) -> None:
        await self.stt_service.initialize()
        self._running = True
        self.logger.info("LiveKit service initialized with enhanced STT pipeline.")

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop()
        await self.stt_service.cleanup()
        self.logger.info("LiveKit service cleanup complete.")
