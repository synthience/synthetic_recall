# __init__.py

```py

```

# audio_buffer.py

```py
from __future__ import annotations
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
import time

logger = logging.getLogger(__name__)

class EnhancedAudioBuffer:
    """Responsive audio buffer with improved speech boundary detection and interruption handling"""
    
    def __init__(
        self, 
        max_length: int, 
        sample_rate: int = 16000,
        energy_threshold: float = -35.0,  # dB threshold for speech
        min_speech_duration: float = 0.3,  # seconds
        max_speech_duration: float = 20.0,  # seconds
        silence_duration: float = 0.7,  # seconds of silence to end speech
        interrupt_flush_threshold: float = 0.1  # seconds to keep after interrupt
    ):
        """
        Initialize enhanced audio buffer
        
        Args:
            max_length: Maximum number of samples to store
            sample_rate: Audio sample rate in Hz
            energy_threshold: dB threshold to consider as speech
            min_speech_duration: Minimum duration to consider valid speech
            max_speech_duration: Maximum duration for a speech segment
            silence_duration: Duration of silence to consider speech ended
            interrupt_flush_threshold: How much audio to keep after interrupt
        """
        self.max_length = int(max_length)  # Ensure integer
        self.sample_rate = int(sample_rate)  # Ensure integer
        self.buffer = np.zeros(self.max_length, dtype=np.float32)
        self.write_pos = 0
        self.length = 0
        self.is_speaking = False
        
        # Speech detection parameters
        self.energy_threshold = energy_threshold  # dB
        self.min_speech_duration = min_speech_duration  # seconds
        self.max_speech_duration = max_speech_duration  # seconds
        self.silence_duration = silence_duration  # seconds
        self.interrupt_flush_threshold = interrupt_flush_threshold  # seconds
        
        # Dynamic settings
        self.auto_threshold = True  # Auto-adjust threshold based on environment
        self.dynamic_silence = True  # Adjust silence duration based on speech length
        self.noise_floor = -50.0  # dB - will be adjusted dynamically
        self.speech_threshold_offset = 10.0  # dB above noise floor
        
        # State tracking
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.speech_duration = 0
        self.silence_start_time = 0
        self.speech_energy_history = []
        self.energy_history_max_len = 100
        self.background_energy_history = []  # For dynamic threshold adjustment
        self.background_history_max_len = 200
        
        # Performance tracking
        self.speech_segments_detected = 0
        self.interruptions_handled = 0
        self.overflows = 0
        self.total_audio_duration = 0.0
        
        # Lock for thread safety in write operations
        self._buffer_lock = None  # Will be initialized if using asyncio
        
        logger.debug(f"Created EnhancedAudioBuffer: max_length={self.max_length}, sr={self.sample_rate}Hz")
        
    def set_asyncio_lock(self, lock):
        """Set asyncio lock for thread safety in async contexts"""
        self._buffer_lock = lock
        
    def add(self, data: np.ndarray, is_interruption: bool = False) -> bool:
        """
        Add audio data to buffer with interruption handling
        
        Args:
            data: Audio data as numpy array (-1 to 1 float)
            is_interruption: Whether this is an interruption event
            
        Returns:
            bool: True if speech end detected, False otherwise
        """
        if data.size == 0:
            return False
            
        # Ensure float32 and proper range
        data = np.asarray(data, dtype=np.float32)
        if data.max() > 1 or data.min() < -1:
            data = np.clip(data, -1, 1)
            
        # Handle interruption - quickly flush buffer keeping only a small amount
        if is_interruption and self.is_speaking:
            samples_to_keep = int(self.interrupt_flush_threshold * self.sample_rate)
            if self.length > samples_to_keep:
                # Keep only the most recent samples up to the threshold
                recent_data = self.get_recent_audio(self.interrupt_flush_threshold)
                self.clear()
                self._add_to_buffer(recent_data)
                
                self.interruptions_handled += 1
                logger.debug(f"Interruption: flushed buffer, kept {len(recent_data)} samples")
                return True  # Signal end of speech due to interruption
            
        # Calculate energy level
        energy = np.mean(np.abs(data))
        energy_db = 20 * np.log10(energy + 1e-10)  # Convert to dB
        
        # Update total audio duration
        data_duration = len(data) / self.sample_rate
        self.total_audio_duration += data_duration
        
        # Add to appropriate energy history
        if self.is_speaking:
            # If speaking, add to speech energy history
            self.speech_energy_history.append(energy_db)
            if len(self.speech_energy_history) > self.energy_history_max_len:
                self.speech_energy_history.pop(0)
        else:
            # Otherwise, add to background noise history
            self.background_energy_history.append(energy_db)
            if len(self.background_energy_history) > self.background_history_max_len:
                self.background_energy_history.pop(0)
                
        # Dynamic threshold adjustment if enabled
        if self.auto_threshold and len(self.background_energy_history) > 20:
            # Use lower percentile to get stable noise floor estimate
            sorted_bg = sorted(self.background_energy_history)
            self.noise_floor = sorted_bg[int(len(sorted_bg) * 0.2)]  # 20th percentile
            
            # Ensure noise floor isn't too low
            self.noise_floor = max(self.noise_floor, -60.0)
            
            # Set speech threshold above noise floor
            self.energy_threshold = self.noise_floor + self.speech_threshold_offset
        
        # Speech detection based on energy
        now = time.time()
        speech_detected = energy_db > self.energy_threshold
        
        # Speech state tracking
        if speech_detected:
            if not self.is_speaking:
                # Speech start
                self.is_speaking = True
                self.speech_start_time = now
                self.speech_duration = 0
                logger.debug(f"Speech start detected at {energy_db:.1f}dB (threshold: {self.energy_threshold:.1f}dB)")
            
            self.last_speech_time = now
            self.speech_duration = now - self.speech_start_time
            self.silence_start_time = 0
        else:
            if self.is_speaking:
                # Potential speech end - start measuring silence
                if self.silence_start_time == 0:
                    self.silence_start_time = now
                
                # Dynamically adjust silence duration based on speech duration
                adjusted_silence = self.silence_duration
                if self.dynamic_silence:
                    # Longer utterances can have longer pauses
                    if self.speech_duration > 5.0:
                        # Scale silence duration with speech duration
                        adjusted_silence = min(1.2, self.silence_duration + (self.speech_duration - 5.0) / 20.0)
                
                # Check if silence duration threshold reached
                silence_duration = now - self.silence_start_time
                
                # Also check if max speech duration exceeded
                if (silence_duration >= adjusted_silence and 
                    self.speech_duration >= self.min_speech_duration) or \
                   self.speech_duration >= self.max_speech_duration:
                    # End of speech detected
                    self.is_speaking = False
                    self.speech_segments_detected += 1
                    
                    logger.debug(f"Speech end detected - duration: {self.speech_duration:.2f}s, "
                                f"silence: {silence_duration:.2f}s, adjusted silence: {adjusted_silence:.2f}s")
                    
                    # Add the data, then return True to signal process buffer
                    self._add_to_buffer(data)
                    return True
        
        # Add data to buffer
        self._add_to_buffer(data)
        return False
                
    def _add_to_buffer(self, data: np.ndarray) -> None:
        """Internal method to add data to circular buffer"""
        if data.size == 0:
            return
            
        # Handle data longer than buffer
        if len(data) > self.max_length:
            data = data[-self.max_length:]
            self.overflows += 1
            
        # Calculate positions
        data_len = len(data)
        space_left = self.max_length - self.write_pos
        
        if data_len <= space_left:
            # Simple case - just write data
            self.buffer[self.write_pos:self.write_pos + data_len] = data
            self.write_pos += data_len
        else:
            # Split write across buffer boundary
            self.buffer[self.write_pos:] = data[:space_left]
            remaining = data_len - space_left
            self.buffer[:remaining] = data[space_left:]
            self.write_pos = remaining
            
        self.length = min(self.length + data_len, self.max_length)
        
    def get_all(self) -> np.ndarray:
        """Get all buffered audio data"""
        if self.length == 0:
            return np.array([], dtype=np.float32)
            
        if self.write_pos >= self.length:
            # No wrap-around
            return self.buffer[self.write_pos - self.length:self.write_pos].copy()
        else:
            # Handle wrap-around
            end_data = self.buffer[-(self.length - self.write_pos):]
            start_data = self.buffer[:self.write_pos]
            return np.concatenate([end_data, start_data])
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get most recent audio of specified duration in seconds"""
        if duration <= 0 or self.length == 0:
            return np.array([], dtype=np.float32)
            
        samples = int(duration * self.sample_rate)
        samples = min(samples, self.length)
        
        return self.get_all()[-samples:]
            
    def clear(self) -> None:
        """Clear the buffer and reset speech state"""
        self.write_pos = 0
        self.length = 0
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.speech_duration = 0
        self.silence_start_time = 0
        
    def get_duration(self) -> float:
        """Get duration of buffered audio in seconds"""
        return self.length / self.sample_rate
        
    def get_speech_info(self) -> Dict[str, Any]:
        """Get information about current speech state"""
        return {
            "is_speaking": self.is_speaking,
            "speech_duration": self.speech_duration,
            "buffer_duration": self.get_duration(),
            "avg_energy_db": np.mean(self.speech_energy_history) if self.speech_energy_history else -100,
            "energy_threshold": self.energy_threshold,
            "noise_floor": self.noise_floor
        }
        
    def handle_interruption(self) -> np.ndarray:
        """Handle interruption event, return current buffer and clear"""
        data = self.get_all()
        self.clear()
        self.interruptions_handled += 1
        return data
        
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "speech_segments_detected": self.speech_segments_detected,
            "interruptions_handled": self.interruptions_handled,
            "overflows": self.overflows,
            "total_audio_duration": self.total_audio_duration,
            "current_energy_threshold": self.energy_threshold,
            "current_noise_floor": self.noise_floor,
            "buffer_size": self.length,
            "buffer_duration": self.get_duration()
        }
        
    def __len__(self) -> int:
        """Get number of samples in buffer"""
        return self.length
```

# audio_utils.py

```py
import numpy as np
from livekit import rtc

class AudioFrame:
    def __init__(self, data: bytes, sample_rate: int, num_channels: int, samples_per_channel: int):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel

    def to_rtc(self) -> rtc.AudioFrame:
        return rtc.AudioFrame(
            data=self.data,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=self.samples_per_channel
        )

def normalize_audio(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        max_val = np.max(np.abs(data))
        if max_val > 1.0:
            data = data / max_val
    return data

def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    from scipy import signal
    if src_rate == dst_rate:
        return data
    target_length = int(len(data) * dst_rate / src_rate)
    resampled = signal.resample(data, target_length)
    return resampled

def split_audio_chunks(data: np.ndarray, chunk_size: int, overlap: int = 0) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    step = chunk_size - overlap
    num_chunks = (len(data) - overlap) // step
    chunks = np.zeros((num_chunks, chunk_size), dtype=data.dtype)
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        chunks[i] = data[start:end]
    return chunks

def convert_to_pcm16(data: np.ndarray) -> bytes:
    if data.dtype == np.float32:
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        raise ValueError(f"Unsupported audio data type: {data.dtype}")
    return data.tobytes()

def create_audio_frame(data: np.ndarray, sample_rate: int, num_channels: int = 1) -> AudioFrame:
    if len(data.shape) == 1:
        data = data.reshape(-1, num_channels)
    samples_per_channel = data.shape[0]
    pcm_data = convert_to_pcm16(data)
    return AudioFrame(
        data=pcm_data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel
    )
```

# config.py

```py
class LucidiaConfig:
    def __init__(self):
        self.tts = {
            "voice": "en-US-AvaMultilingualNeural",
            "sample_rate": 48000,
            "num_channels": 1,
        }

class LLMConfig:
    def __init__(self):
        self.server_url = "http://localhost:1234/v1/chat/completions"
        self.model_name = "local-model"
        self.temperature = 0.7
        self.max_tokens = 300

class WhisperConfig:
    def __init__(self):
        self.model_name = "small"
        self.language = "en"
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.min_audio_length = 0.5
        self.max_audio_length = 3.0

```

# event_emitter.py

```py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventEmitter:
    """
    Asynchronous event emitter implementation.
    Supports event subscription and emission with async handlers.
    """
    def __init__(self):
        self._events: Dict[str, List[Callable]] = defaultdict(list)
        self._once_events: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler.
        Can be used as a decorator or method call.
        """
        def decorator(func: Callable) -> Callable:
            self._events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def once(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register a one-time event handler.
        Handler will be removed after first execution.
        """
        def decorator(func: Callable) -> Callable:
            self._once_events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def off(self, event_name: str, handler: Callable) -> None:
        """Remove a specific event handler."""
        if event_name in self._events:
            self._events[event_name] = [h for h in self._events[event_name] if h != handler]
        if event_name in self._once_events:
            self._once_events[event_name] = [h for h in self._once_events[event_name] if h != handler]
            
    def remove_all_listeners(self, event_name: Optional[str] = None) -> None:
        """Remove all handlers for an event, or all events if no name given."""
        if event_name:
            self._events[event_name].clear()
            self._once_events[event_name].clear()
        else:
            self._events.clear()
            self._once_events.clear()
            
    async def emit(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event with optional data.
        Executes all handlers asynchronously.
        """
        # Regular handlers
        handlers = self._events.get(event_name, [])
        once_handlers = self._once_events.get(event_name, [])
        
        # Execute handlers
        for handler in handlers + once_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_name}: {e}", exc_info=True)
                
        # Clear once handlers
        if event_name in self._once_events:
            self._once_events[event_name].clear()

```

# livekit-diagnostic.py

```py
"""LiveKit diagnostic tool for troubleshooting UI update issues."""

import asyncio
import logging
import time
import json
import sys
import argparse
import jwt
from typing import Dict, Any, List, Optional
import os
import socket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import LiveKit SDK with fallback options
try:
    import livekit
    import livekit.rtc as rtc
    logger.info(f"LiveKit SDK version: {livekit.__version__}")
except ImportError:
    logger.error("LiveKit SDK not found. Install with: pip install livekit")
    logger.info("Trying alternative import...")
    try:
        from livekit import rtc
        logger.info("LiveKit rtc module imported through alternative path")
    except ImportError:
        logger.error("Could not import LiveKit rtc. Please install the SDK properly.")
        sys.exit(1)

# Default settings
DEFAULT_URL = "ws://localhost:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
DEFAULT_ROOM = "test-room"

class LiveKitDiagnosticTool:
    """
    Diagnostic tool for LiveKit UI publishing issues.
    Tests various publishing methods to help diagnose UI update problems.
    """
    
    def __init__(self, url: str, api_key: str, api_secret: str, room_name: str):
        """Initialize the diagnostic tool."""
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.identity = f"diagnostic_{int(time.time())}"
        
        # Connection state
        self.room = None
        self.log_messages = []
        
        # Publishing success tracking
        self.successful_publishes = {
            "data": 0,
            "transcription": 0
        }
        self.failed_publishes = {
            "data": 0,
            "transcription": 0
        }
        
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log message with timestamp and optional metadata."""
        timestamp = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
        
        entry = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs
        }
        
        self.log_messages.append(entry)
        logger.log(level, message)
        
    def generate_token(self) -> str:
        """Generate JWT token for LiveKit with proper permissions."""
        exp_time = int(time.time()) + 3600  # 1 hour validity
        
        claims = {
            "iss": self.api_key,
            "sub": self.identity,
            "exp": exp_time,
            "nbf": int(time.time()) - 60,  # Valid from 1 minute ago (allow for clock drift)
            "video": {
                "room": self.room_name,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,  # Critical for UI updates
                "roomAdmin": True,       # Helpful for diagnostics
                "roomCreate": True       # Create room if needed
            },
            "metadata": json.dumps({
                "type": "diagnostic", 
                "version": "1.0"
            })
        }
        
        token = jwt.encode(claims, self.api_secret, algorithm="HS256")
        self.log(logging.INFO, "Token generated with permissions", permissions=claims["video"])
        return token
        
    async def network_diagnostics(self) -> bool:
        """Run network diagnostics for LiveKit connection."""
        self.log(logging.INFO, "Running network diagnostics...")
        
        # Parse URL
        if self.url.startswith("ws://"):
            host = self.url[5:].split(":")[0]
            port = int(self.url.split(":")[-1])
            secure = False
        elif self.url.startswith("wss://"):
            host = self.url[6:].split(":")[0]
            port = int(self.url.split(":")[-1]) if ":" in self.url[6:] else 443
            secure = True
        else:
            host = self.url
            port = 7880
            secure = False
            
        # DNS lookup
        try:
            self.log(logging.INFO, f"DNS lookup for {host}")
            ip_address = socket.gethostbyname(host)
            self.log(logging.INFO, f"DNS lookup successful: {ip_address}")
        except socket.gaierror as e:
            self.log(logging.ERROR, f"DNS lookup failed: {e}")
            return False
            
        # Socket connection test
        try:
            self.log(logging.INFO, f"Testing socket connection to {host}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                self.log(logging.INFO, f"Socket connection successful")
            else:
                self.log(logging.ERROR, f"Socket connection failed with error {result}")
                return False
        except Exception as e:
            self.log(logging.ERROR, f"Socket connection test failed: {e}")
            return False
            
        # Basic HTTP(S) connectivity test
        try:
            import aiohttp
            test_url = f"{'https' if secure else 'http'}://{host}:{port}/rtc"
            self.log(logging.INFO, f"Testing HTTP connectivity to {test_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, timeout=5) as response:
                    if response.status != 404:
                        self.log(logging.WARNING, f"Unexpected response from LiveKit server: {response.status}")
                    else:
                        self.log(logging.INFO, f"HTTP connectivity test successful (404 expected)")
        except Exception as e:
            self.log(logging.WARNING, f"HTTP connectivity test failed: {e}")
            # Continue anyway - this is just an extra check
            
        return True
        
    async def test_publish_methods(self) -> Dict[str, Any]:
        """Test various publishing methods and return results."""
        results = {
            "data_publish": False,
            "transcription_publish": False,
            "data_methods_tested": 0,
            "transcription_methods_tested": 0,
            "errors": []
        }
        
        if not self.room or self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            self.log(logging.ERROR, "Room not connected, cannot test publishing")
            results["errors"].append("Room not connected")
            return results
            
        if not self.room.local_participant:
            self.log(logging.ERROR, "No local participant, cannot test publishing")
            results["errors"].append("No local participant")
            return results
            
        # 1. Test basic data publishing
        try:
            results["data_methods_tested"] += 1
            test_data = json.dumps({
                "type": "diagnostic_test",
                "message": "Basic data test",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(test_data, reliable=True)
            self.log(logging.INFO, "Basic data publishing successful")
            results["data_publish"] = True
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Basic data publishing failed: {e}")
            results["errors"].append(f"Basic data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 2. Test publishing UI state message
        try:
            results["data_methods_tested"] += 1
            ui_data = json.dumps({
                "type": "state_update",
                "state": "listening",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(ui_data, reliable=True)
            self.log(logging.INFO, "UI state data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"UI state data publishing failed: {e}")
            results["errors"].append(f"UI state publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 3. Test publishing transcript data
        try:
            results["data_methods_tested"] += 1
            transcript_data = json.dumps({
                "type": "transcript",
                "text": "This is a test transcript",
                "sender": "diagnostic",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(transcript_data, reliable=True)
            self.log(logging.INFO, "Transcript data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcript data publishing failed: {e}")
            results["errors"].append(f"Transcript data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 4. Test publishing transcription API
        # First publish a local audio track to get a track_sid
        local_track = None
        try:
            audio_source = rtc.AudioSource()
            local_track = rtc.LocalAudioTrack.create_audio_track("diagnostic_audio", audio_source)
            await self.room.local_participant.publish_track(local_track)
            self.log(logging.INFO, "Audio track published successfully")
            
            # Now test transcription API
            results["transcription_methods_tested"] += 1
            trans = rtc.Transcription(
                text="Test transcription API",
                participant_identity=self.identity
            )
            await self.room.local_participant.publish_transcription(trans)
            self.log(logging.INFO, "Transcription API publishing successful")
            results["transcription_publish"] = True
            self.successful_publishes["transcription"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcription API publishing failed: {e}")
            results["errors"].append(f"Transcription API publishing: {str(e)}")
            self.failed_publishes["transcription"] += 1
        finally:
            # Clean up track
            if local_track:
                try:
                    await self.room.local_participant.unpublish_track(local_track)
                except:
                    pass
                    
        return results
        
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run full diagnostics and return results."""
        start_time = time.time()
        self.log(logging.INFO, "Starting LiveKit UI diagnostics")
        
        # Run network diagnostics first
        network_ok = await self.network_diagnostics()
        if not network_ok:
            self.log(logging.ERROR, "Network diagnostics failed, cannot continue")
            return {
                "success": False,
                "stage": "network",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
            
        # Generate token
        token = self.generate_token()
        
        # Connect to room
        try:
            self.log(logging.INFO, f"Connecting to room: {self.room_name}")
            self.room = rtc.Room()
            await self.room.connect(self.url, token)
            
            # Wait for connection to stabilize
            for _ in range(5):  # Wait up to 5 seconds
                if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    break
                await asyncio.sleep(1)
                
            if self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                self.log(logging.ERROR, f"Failed to connect to room: {self.room.connection_state}")
                return {
                    "success": False,
                    "stage": "connection",
                    "duration": time.time() - start_time,
                    "logs": self.log_messages
                }
                
            self.log(logging.INFO, "Successfully connected to room")
            
            # Test publish methods
            publish_results = await self.test_publish_methods()
            
            # Final results
            success = publish_results["data_publish"] or publish_results["transcription_publish"]
            diagnostic_results = {
                "success": success,
                "duration": time.time() - start_time,
                "network_ok": network_ok,
                "room_connected": self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED,
                "publish_results": publish_results,
                "publish_stats": {
                    "successful": self.successful_publishes,
                    "failed": self.failed_publishes
                },
                "logs": self.log_messages,
                "recommendations": []
            }
            
            # Add recommendations based on results
            if not success:
                diagnostic_results["recommendations"].append(
                    "Check token permissions, especially 'canPublishData: true'"
                )
                
            if self.failed_publishes["data"] > 0 and self.successful_publishes["data"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check Docker networking configuration (exposing port 7880)"
                )
                
            if self.failed_publishes["transcription"] > 0 and self.successful_publishes["transcription"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check LiveKit version compatibility (Transcription API requires newer versions)"
                )
                
            if success:
                diagnostic_results["recommendations"].append(
                    "UI updates should be working. If still having issues, check client-side subscription."
                )
                
            return diagnostic_results
            
        except Exception as e:
            self.log(logging.ERROR, f"Error during diagnostics: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "unknown",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
        finally:
            # Clean up
            if self.room:
                try:
                    await self.room.disconnect()
                except:
                    pass

async def print_diagnostic_results(results: Dict[str, Any]) -> None:
    """Print diagnostic results in a readable format."""
    print("\n" + "=" * 50)
    print("LIVEKIT UI CONNECTION DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    print(f"\nOverall success: {'✅' if results['success'] else '❌'}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if 'network_ok' in results:
        print(f"\nNetwork connectivity: {'✅' if results['network_ok'] else '❌'}")
        
    if 'room_connected' in results:
        print(f"Room connection: {'✅' if results['room_connected'] else '❌'}")
        
    if 'publish_results' in results:
        pr = results['publish_results']
        print("\nPublishing tests:")
        print(f"  Data publishing: {'✅' if pr['data_publish'] else '❌'} ({pr['data_methods_tested']} methods tested)")
        print(f"  Transcription publishing: {'✅' if pr['transcription_publish'] else '❌'} ({pr['transcription_methods_tested']} methods tested)")
        
        if pr.get('errors'):
            print("\nPublishing errors:")
            for err in pr['errors']:
                print(f"  - {err}")
                
    if 'publish_stats' in results:
        ps = results['publish_stats']
        print("\nPublishing statistics:")
        print(f"  Successful data publishes: {ps['successful']['data']}")
        print(f"  Failed data publishes: {ps['failed']['data']}")
        print(f"  Successful transcription publishes: {ps['successful']['transcription']}")
        print(f"  Failed transcription publishes: {ps['failed']['transcription']}")
        
    if 'recommendations' in results:
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
            
    print("\nDetailed logs:")
    for log in results['logs'][-10:]:  # Show last 10 logs
        level_color = "\033[92m" if log['level'] == "INFO" else "\033[91m" if log['level'] == "ERROR" else "\033[93m"
        reset_color = "\033[0m"
        print(f"  {log['formatted_time']} {level_color}{log['level']}{reset_color}: {log['message']}")
        
    print("\n" + "=" * 50)
    
    if results['success']:
        print("\n✅ DIAGNOSIS: UI updates should be working correctly.")
        print("If issues persist, check your client-side subscription and UI code.")
    else:
        print("\n❌ DIAGNOSIS: UI updates are not working correctly.")
        print("Follow the recommendations above to fix the issues.")
        
    print("=" * 50 + "\n")

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LiveKit UI Connection Diagnostic Tool")
    parser.add_argument("--url", default=DEFAULT_URL, help="LiveKit server URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LiveKit API key")
    parser.add_argument("--api-secret", default=DEFAULT_API_SECRET, help="LiveKit API secret")
    parser.add_argument("--room", default=DEFAULT_ROOM, help="Room name")
    parser.add_argument("--output", choices=["console", "json"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    # Allow environment variable overrides
    url = os.environ.get("LIVEKIT_URL", args.url)
    api_key = os.environ.get("LIVEKIT_API_KEY", args.api_key)
    api_secret = os.environ.get("LIVEKIT_API_SECRET", args.api_secret)
    room = os.environ.get("LIVEKIT_ROOM", args.room)
    
    # Run diagnostics
    tool = LiveKitDiagnosticTool(url, api_key, api_secret, room)
    results = await tool.run_diagnostics()
    
    # Output results
    if args.output == "json":
        print(json.dumps(results, default=str, indent=2))
    else:
        await print_diagnostic_results(results)

if __name__ == "__main__":
    asyncio.run(main())

```

# logger_config.py

```py
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Configure logging with rotating file handler and console output.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (rotating, max 10MB per file, keep 30 days of logs)
    log_file = log_path / f"voice_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log system info
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    
    return logger

# Performance logging
def log_performance_metrics(operation: str, duration: float, **kwargs):
    """Log performance metrics for voice pipeline operations."""
    metrics = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        **kwargs
    }
    logger = logging.getLogger("voice_assistant")
    logger.debug(f"Performance: {metrics}")

# Error logging with context
def log_error_with_context(message: str, error: Exception, context: dict = None):
    """Log errors with additional context information."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    logger = logging.getLogger("voice_assistant")
    logger.error(f"{message}: {error_info}")

# Connection state logging
def log_connection_state(state: str, connection_state: str, details: dict = None):
    """Log connection state changes with details."""
    logger = logging.getLogger("voice_assistant")
    logger.info(f"Connection {state} (state={connection_state}): {details or {}}")

```

# logging.py

```py
"""Logging configuration for voice agent"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging with proper encoding and formatting"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / "voice_agent.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Log initialization
    logger.info("=== Voice Assistant Logger Initialized ===")
    logger.info(f"Log Level: {level}")
    logger.info(f"Log Directory: {log_dir.absolute()}")
    
    return logger

```

# pipeline_logger.py

```py
"""Pipeline logging utilities for voice agents."""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Metrics for voice pipeline performance tracking."""
    
    start_time: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric with timestamp."""
        self.metrics[name] = {
            'value': value,
            'timestamp': time.time() - self.start_time
        }
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a recorded metric."""
        return self.metrics.get(name)
    
    def get_duration(self, start_event: str, end_event: str) -> Optional[float]:
        """Get duration between two events."""
        start = self.get_metric(start_event)
        end = self.get_metric(end_event)
        if start and end:
            return end['timestamp'] - start['timestamp']
        return None

class PipelineLogger:
    """Logger for voice pipeline events and metrics."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.metrics = PipelineMetrics()
        
    def _log(self, level: int, stage: str, message: str, **kwargs) -> None:
        """Internal logging with consistent format."""
        metadata = {
            'session_id': self.session_id,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.log(level, f"[{stage}] {message}", extra={'metadata': metadata})
        
    # STT Events
    def stt_started(self, config: Dict[str, Any]) -> None:
        """Log STT initialization."""
        self.metrics.record_metric('stt_start', config)
        self._log(logging.INFO, 'STT', 'Speech recognition started', config=config)
        
    def stt_partial(self, text: str) -> None:
        """Log partial STT results."""
        self._log(logging.DEBUG, 'STT', f'Partial transcript: {text}', text=text)
        
    def stt_final(self, text: str, confidence: float) -> None:
        """Log final STT results."""
        self.metrics.record_metric('stt_final', {'text': text, 'confidence': confidence})
        self._log(logging.INFO, 'STT', f'Final transcript: {text}', 
                 text=text, confidence=confidence)
        
    def stt_error(self, error: Exception) -> None:
        """Log STT errors."""
        self._log(logging.ERROR, 'STT', f'Recognition error: {str(error)}', 
                 error=str(error))
        
    # LLM Events
    def llm_request(self, prompt: str) -> None:
        """Log LLM request."""
        self.metrics.record_metric('llm_request', prompt)
        self._log(logging.INFO, 'LLM', 'Sending request to LLM', prompt=prompt)
        
    def llm_response(self, response: str, metadata: Dict[str, Any]) -> None:
        """Log LLM response."""
        self.metrics.record_metric('llm_response', response)
        duration = self.metrics.get_duration('llm_request', 'llm_response')
        self._log(logging.INFO, 'LLM', f'Received response in {duration:.2f}s', 
                 response=response, metadata=metadata)
        
    def llm_error(self, error: Exception) -> None:
        """Log LLM errors."""
        self._log(logging.ERROR, 'LLM', f'LLM error: {str(error)}', 
                 error=str(error))
        
    # TTS Events
    def tts_started(self, text: str, config: Dict[str, Any]) -> None:
        """Log TTS initialization."""
        self.metrics.record_metric('tts_start', {'text': text, 'config': config})
        self._log(logging.INFO, 'TTS', 'Speech synthesis started', 
                 text=text, config=config)
        
    def tts_progress(self, bytes_processed: int) -> None:
        """Log TTS progress."""
        self._log(logging.DEBUG, 'TTS', f'Generated {bytes_processed} bytes', 
                 bytes_processed=bytes_processed)
        
    def tts_complete(self, duration: float, total_bytes: int) -> None:
        """Log TTS completion."""
        self.metrics.record_metric('tts_complete', {
            'duration': duration,
            'total_bytes': total_bytes
        })
        self._log(logging.INFO, 'TTS', 
                 f'Speech synthesis completed in {duration:.2f}s ({total_bytes} bytes)',
                 duration=duration, total_bytes=total_bytes)
        
    def tts_error(self, error: Exception) -> None:
        """Log TTS errors."""
        self._log(logging.ERROR, 'TTS', f'Synthesis error: {str(error)}', 
                 error=str(error))
        
    # LiveKit Events
    def livekit_connected(self, room_name: str, participant_id: str) -> None:
        """Log LiveKit connection."""
        self.metrics.record_metric('livekit_connect', {
            'room': room_name,
            'participant_id': participant_id
        })
        self._log(logging.INFO, 'LiveKit', 'Connected to room', 
                 room=room_name, participant_id=participant_id)
        
    def livekit_track_published(self, track_id: str, kind: str) -> None:
        """Log track publication."""
        self._log(logging.INFO, 'LiveKit', f'Published {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_track_subscribed(self, track_id: str, kind: str) -> None:
        """Log track subscription."""
        self._log(logging.INFO, 'LiveKit', f'Subscribed to {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_error(self, error: Exception) -> None:
        """Log LiveKit errors."""
        self._log(logging.ERROR, 'LiveKit', f'LiveKit error: {str(error)}', 
                 error=str(error))
        
    # Pipeline Events
    def pipeline_started(self, config: Dict[str, Any]) -> None:
        """Log pipeline start."""
        self.metrics = PipelineMetrics()  # Reset metrics
        self._log(logging.INFO, 'Pipeline', 'Voice pipeline started', config=config)
        
    def pipeline_stopped(self) -> None:
        """Log pipeline stop with performance metrics."""
        total_duration = time.time() - self.metrics.start_time
        self._log(logging.INFO, 'Pipeline', 
                 f'Voice pipeline stopped after {total_duration:.2f}s',
                 total_duration=total_duration,
                 metrics=self.metrics.metrics)
        
    def pipeline_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log pipeline errors with context."""
        self._log(logging.ERROR, 'Pipeline', 
                 f'Pipeline error: {str(error)}', 
                 error=str(error), context=context)

```

# sentence_buffer.py

```py
from __future__ import annotations
import re
import time
from typing import Optional, List, Dict, Any, Deque
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)

class SentenceBuffer:
    """
    Enhanced SentenceBuffer that manages partial transcriptions and sentence chunking 
    for more natural conversation flow with improved text normalization and context awareness.
    """
    
    def __init__(self, 
                 max_buffer_time: float = 5.0,
                 min_words_for_chunk: int = 3,
                 end_of_sentence_timeout: float = 1.0,
                 max_history_size: int = 10,
                 confidence_threshold: float = 0.7):
        """
        Initialize the sentence buffer with configurable parameters.
        
        Args:
            max_buffer_time: Maximum time (in seconds) to buffer text before forcing processing
            min_words_for_chunk: Minimum number of words required to process a chunk
            end_of_sentence_timeout: Time (in seconds) after which to consider a sentence complete
            max_history_size: Maximum number of processed sentences to keep in history
            confidence_threshold: Minimum confidence score for accepting transcripts
        """
        self.buffer = []
        self.last_update_time = 0
        self.max_buffer_time = max_buffer_time
        self.min_words_for_chunk = min_words_for_chunk
        self.end_of_sentence_timeout = end_of_sentence_timeout
        self.confidence_threshold = confidence_threshold
        
        # Enhanced sentence boundary detection
        self.sentence_endings = re.compile(r'[.!?][\s"\')\]]?$|$')
        self.question_pattern = re.compile(r'\b(who|what|when|where|why|how|is|are|was|were|will|do|does|did|can|could|would|should|may|might)\b', re.IGNORECASE)
        
        # Track processed sentences for context
        self.processed_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        
        # Performance metrics
        self.metrics = {
            "chunks_processed": 0,
            "sentences_completed": 0,
            "avg_sentence_length": 0,
            "total_processing_time": 0
        }
        
        # Additional filler words and hesitation sounds
        self.fillers = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'so', 'basically',
            'actually', 'literally', 'well', 'right', 'okay', 'hmm', 'mmm'
        }
        
        # Common speech recognition errors to correct
        self.common_corrections = {
            "i'm gonna": "I'm going to",
            "i gotta": "I've got to",
            "wanna": "want to",
            "kinda": "kind of",
            "lemme": "let me",
            "gimme": "give me",
            "dunno": "don't know"
        }
        
        self.logger = logging.getLogger(__name__)
        
    def add_transcript(self, text: str, confidence: float = 1.0) -> Optional[str]:
        """
        Add a new transcript chunk and return a complete sentence if available.
        
        Args:
            text: The transcript text to add
            confidence: Confidence score (0-1) for this transcript
            
        Returns:
            Completed sentence if available, None otherwise
        """
        start_time = time.time()
        current_time = start_time
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.logger.debug(f"Transcript below confidence threshold: {confidence:.2f} < {self.confidence_threshold:.2f}")
            return None
        
        # Clean and normalize the text
        text = text.strip().lower()
        if not text:
            return None
            
        # Check if this is a repeat of the last chunk
        if self.buffer and text == self.buffer[-1]['text']:
            self.logger.debug("Duplicate transcript chunk detected, skipping")
            return None
            
        # Add new chunk to buffer
        self.buffer.append({
            'text': text,
            'timestamp': current_time,
            'confidence': confidence
        })
        
        self.metrics["chunks_processed"] += 1
        self.last_update_time = current_time
        
        # Try to form a complete sentence
        result = self._process_buffer(current_time)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.metrics["total_processing_time"] += processing_time
        self.logger.debug(f"Transcript processing time: {processing_time:.3f}s")
        
        return result
    
    def _process_buffer(self, current_time: float) -> Optional[str]:
        """
        Process buffer to find complete sentences with enhanced detection rules.
        
        Args:
            current_time: Current time for timeout calculation
            
        Returns:
            Completed sentence if available, None otherwise
        """
        if not self.buffer:
            return None
            
        # Join all chunks
        full_text = ' '.join(chunk['text'] for chunk in self.buffer)
        words = full_text.split()
        
        # Calculate average confidence
        avg_confidence = sum(chunk.get('confidence', 1.0) for chunk in self.buffer) / len(self.buffer)
        
        # Enhanced conditions for processing the buffer
        should_process = (
            # Natural sentence ending
            bool(self.sentence_endings.search(full_text)) or
            
            # Question detection (more likely to be a complete thought)
            bool(self.question_pattern.search(full_text) and len(words) >= 4) or
            
            # Enough words and time gap
            (len(words) >= self.min_words_for_chunk and 
             current_time - self.buffer[0]['timestamp'] > self.end_of_sentence_timeout) or
            
            # Buffer timeout
            (current_time - self.buffer[0]['timestamp'] > self.max_buffer_time) or
            
            # High confidence and sufficient length
            (avg_confidence > 0.9 and len(words) >= self.min_words_for_chunk * 2)
        )
        
        if should_process:
            # Clean up the text
            result = self._clean_text(full_text)
            
            # Add to processed history
            self.processed_history.append({
                'text': result,
                'timestamp': current_time,
                'word_count': len(result.split()),
                'confidence': avg_confidence,
                'chunks': len(self.buffer)
            })
            
            # Update metrics
            self.metrics["sentences_completed"] += 1
            total_words = sum(len(entry['text'].split()) for entry in self.processed_history)
            if self.metrics["sentences_completed"] > 0:
                self.metrics["avg_sentence_length"] = total_words / self.metrics["sentences_completed"]
            
            # Clear the buffer for next sentence
            self.buffer.clear()
            return result
            
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize the transcribed text with enhanced processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Original text for logging
        original = text
        
        # Remove filler words and hesitation sounds
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Skip filler words
            if word.lower() in self.fillers:
                continue
                
            # Apply common corrections
            corrected = False
            for error, correction in self.common_corrections.items():
                if word.lower() == error or f"{word.lower()} " == error:
                    if not cleaned_words:  # If first word, capitalize correction
                        cleaned_words.append(correction)
                    else:
                        cleaned_words.append(correction.lower())
                    corrected = True
                    break
                    
            if not corrected:
                cleaned_words.append(word)
        
        # Join words and ensure proper spacing around punctuation
        text = ' '.join(cleaned_words)
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        
        # Add sentence ending if missing
        if not re.search(r'[.!?]$', text):
            # Add question mark if it looks like a question
            if self.question_pattern.search(text):
                text += '?'
            else:
                text += '.'
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            
        if original != text:
            self.logger.debug(f"Text cleaned: '{original}' → '{text}'")
            
        return text
    
    def clear(self) -> None:
        """Clear the buffer and reset processing state."""
        self.buffer.clear()
        self.last_update_time = 0
        
    def get_partial_transcript(self) -> str:
        """
        Get the current partial transcript without clearing the buffer.
        
        Returns:
            Current partial transcript as a single string
        """
        if not self.buffer:
            return ""
        return ' '.join(chunk['text'] for chunk in self.buffer)
    
    def get_context(self, max_sentences: int = 3) -> str:
        """
        Get recent conversation context from processed history.
        
        Args:
            max_sentences: Maximum number of recent sentences to include
            
        Returns:
            Recent conversation context as a string
        """
        context = [entry['text'] for entry in list(self.processed_history)[-max_sentences:]]
        return " ".join(context)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the sentence buffer.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate average processing time
        if self.metrics["chunks_processed"] > 0:
            avg_processing_time = self.metrics["total_processing_time"] / self.metrics["chunks_processed"]
        else:
            avg_processing_time = 0
            
        return {
            **self.metrics,
            "buffer_size": len(self.buffer),
            "history_size": len(self.processed_history),
            "avg_processing_time": avg_processing_time
        }
    
    def to_json(self) -> str:
        """
        Convert current buffer state to JSON for debugging or UI display.
        
        Returns:
            JSON representation of current buffer state
        """
        state = {
            "buffer": self.buffer,
            "history": list(self.processed_history),
            "metrics": self.get_metrics(),
            "partial": self.get_partial_transcript()
        }
        return json.dumps(state, indent=2)
    
    def __len__(self) -> int:
        """Return the number of chunks in the buffer."""
        return len(self.buffer)
    
    def __bool__(self) -> bool:
        """Return True if the buffer has content."""
        return bool(self.buffer)
```

# shared_state.py

```py
import threading
import asyncio
from typing import Dict, Any

# Global flag to signal interruption.
# This event can be set when, for example, the user wants to cancel ongoing speech recognition.
should_interrupt = asyncio.Event()

# Global microphone settings.
# 'selected_microphone' will store the device index or identifier of the chosen microphone.
selected_microphone = None

# Global recognizer settings.
# These settings control the sensitivity and behavior of the speech recognizer.
# Fine-tune these values based on the environment, microphone quality, and the desired balance between responsiveness and accuracy.
recognizer_settings: Dict[str, Any] = {
    # Base energy threshold for distinguishing speech from background noise.
    # A lower value makes the recognizer more sensitive, but may pick up ambient sounds.
    "energy_threshold": 300,

    # If True, the recognizer will automatically adjust the energy threshold over time
    # based on ambient noise levels. This helps maintain recognition accuracy in variable environments.
    "dynamic_energy_threshold": True,

    # The maximum length of silence (in seconds) allowed within a phrase.
    # A higher value means the recognizer will wait longer before considering a pause as the end of speech.
    "pause_threshold": 0.8,

    # The amount of non-speaking duration (in seconds) required before finalizing the speech input.
    # Setting this to a higher value (e.g., 0.8) causes the recognizer to wait longer for continued speech.
    "operation_timeout": None,

    # Additional granular settings for enhanced control:

    # The sample rate (in Hz) of the audio input.
    # A common value for many applications is 16000 Hz, balancing detail and processing load.
    "sample_rate": 16000,

    # Duration (in milliseconds) of each audio chunk processed.
    # Smaller chunks (e.g., 20 ms) allow near real-time processing but may require more frequent computation.
    "chunk_duration_ms": 20,

    # Maximum number of consecutive silent audio chunks allowed before the recognizer decides the phrase has ended.
    # Higher values (e.g., 10) allow for more natural pauses in speech.
    "max_silence_chunks": 10,

    # Minimum total phrase length (in seconds) required before processing.
    # This avoids triggering on very short utterances or noise.
    "min_phrase_length": 0.3,
    'dynamic_energy_adjustment_damping': 0.15,
    'dynamic_energy_ratio': 1.5,
}

class InterruptHandler:
    def __init__(self):
        self.interrupt_event = asyncio.Event()
        self.keywords = {"stop", "wait", "pause", "cancel"}

    async def check_for_interrupt(self, text: str) -> bool:
        """Check if text contains interrupt keywords"""
        words = text.lower().split()
        if any(kw in words for kw in self.keywords):
            self.interrupt_event.set()
            should_interrupt.set()  # Set global interrupt
            return True
        return False

    async def reset(self):
        """Reset interrupt flags"""
        self.interrupt_event.clear()
        should_interrupt.clear()

# Global interrupt handler
interrupt_handler = InterruptHandler()

```

