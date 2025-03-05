"""Audio utilities for voice pipeline."""

from __future__ import annotations

import io
import logging
import numpy as np
from typing import Union, Optional
from scipy import signal
import soundfile as sf
from dataclasses import dataclass
from livekit import rtc

logger = logging.getLogger(__name__)

@dataclass
class AudioFrame:
    """Audio frame container with metadata."""
    data: np.ndarray
    sample_rate: int
    num_channels: int
    samples_per_channel: int

    def to_bytes(self) -> bytes:
        """Convert frame data to bytes."""
        return self.data.tobytes()

    def to_pcm(self) -> np.ndarray:
        """Ensure data is in PCM format."""
        if self.data.dtype != np.float32:
            return self.data.astype(np.float32)
        return self.data

    def to_rtc(self) -> rtc.AudioFrame:
        """Convert to LiveKit audio frame."""
        # Ensure data is in PCM format
        pcm_data = self.to_pcm()
        
        # Resample to 48kHz if needed (LiveKit default)
        if self.sample_rate != 48000:
            pcm_data = resample_audio(pcm_data, self.sample_rate, 48000)
            
        # Ensure contiguous memory layout
        if not pcm_data.flags['C_CONTIGUOUS']:
            pcm_data = np.ascontiguousarray(pcm_data)
            
        return rtc.AudioFrame(
            data=pcm_data.tobytes(),
            samples_per_channel=len(pcm_data),
            sample_rate=48000,  # LiveKit default
            num_channels=self.num_channels
        )

def normalize_audio(data: np.ndarray, target_range: float = 1.0) -> np.ndarray:
    """Normalize audio data to target range [-target_range, target_range]."""
    if data.size == 0:
        return data
        
    # Convert to float32 if needed
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Handle int16 conversion
    if np.abs(data).max() > 1.0:
        data = data / 32768.0
        
    # Normalize to target range
    max_val = np.abs(data).max()
    if max_val > 0:
        data = data * (target_range / max_val)
        
    return data

def resample_audio(data: np.ndarray, src_rate: int, dst_rate: int, 
                  preserve_shape: bool = True) -> np.ndarray:
    """Resample audio data to target sample rate."""
    if src_rate == dst_rate:
        return data
        
    # Ensure data is float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        
    # Calculate new length
    new_length = int(len(data) * dst_rate / src_rate)
    
    # Resample using scipy
    resampled = signal.resample(data, new_length)
    
    # Preserve original shape if needed
    if preserve_shape and len(resampled) != new_length:
        if len(resampled) < new_length:
            resampled = np.pad(resampled, (0, new_length - len(resampled)))
        else:
            resampled = resampled[:new_length]
            
    return resampled

def split_audio_chunks(data: np.ndarray, chunk_size: int, 
                      overlap: int = 0) -> np.ndarray:
    """Split audio data into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
        
    # Calculate step size
    step = chunk_size - overlap
    
    # Calculate number of chunks
    num_chunks = (len(data) - overlap) // step
    
    # Create output array
    chunks = np.zeros((num_chunks, chunk_size), dtype=data.dtype)
    
    # Fill chunks
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_size
        if end <= len(data):
            chunks[i] = data[start:end]
        else:
            # Pad last chunk if needed
            remaining = len(data) - start
            chunks[i, :remaining] = data[start:]
            
    return chunks

def convert_audio_format(data: Union[bytes, np.ndarray], 
                        src_format: str,
                        dst_format: str,
                        sample_rate: Optional[int] = None) -> bytes:
    """Convert audio between different formats."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
        
    # Create in-memory buffer
    with io.BytesIO(data) as buf:
        # Read audio data
        audio_data, sr = sf.read(buf, format=src_format)
        
        # Resample if needed
        if sample_rate and sr != sample_rate:
            audio_data = resample_audio(audio_data, sr, sample_rate)
            sr = sample_rate
            
        # Write to output buffer
        out_buf = io.BytesIO()
        sf.write(out_buf, audio_data, sr, format=dst_format)
        return out_buf.getvalue()

class EdgeAudioFrame:
    """Wrapper for Edge TTS audio data that provides LiveKit frame interface."""
    def __init__(self, pcm_data: bytes, sample_rate: int = 48000, num_channels: int = 1):
        # Convert bytes to int16 numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Create AudioFrame
        self._frame = AudioFrame(
            data=audio_array,
            sample_rate=sample_rate,
            num_channels=num_channels,
            samples_per_channel=len(audio_array)
        )
    
    @property
    def frame(self) -> rtc.AudioFrame:
        """Get LiveKit audio frame."""
        return self._frame.to_rtc()

class AudioBuffer:
    """Buffer for collecting audio frames for VAD and STT."""
    def __init__(self, max_size: int = 48000 * 5):  # 5 seconds at 48kHz
        self.buffer = io.BytesIO()
        self.max_size = max_size
        self.last_speech = False
        self.speech_start = None
        self.silence_duration = 0
        self.is_speaking = False

    def add_frame(self, frame_data: bytes) -> None:
        """Add a frame to the buffer, maintaining max size."""
        current_size = self.buffer.tell()
        if current_size + len(frame_data) > self.max_size:
            # Keep the last 2 seconds of audio
            keep_size = 48000 * 2
            self.buffer.seek(max(0, current_size - keep_size))
            remaining_data = self.buffer.read()
            self.buffer = io.BytesIO()
            self.buffer.write(remaining_data)
        self.buffer.write(frame_data)

    def get_data(self) -> bytes:
        """Get all buffered audio data."""
        current_pos = self.buffer.tell()
        self.buffer.seek(0)
        data = self.buffer.read()
        self.buffer.seek(current_pos)
        return data

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = io.BytesIO()
        self.last_speech = False
        self.speech_start = None
        self.silence_duration = 0
        self.is_speaking = False
