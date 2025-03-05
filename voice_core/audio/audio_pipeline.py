"""
Enhanced audio pipeline for voice processing with LiveKit integration.
Handles sample rate conversion, normalization, and frame management.
"""

import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
import livekit.rtc as rtc
from scipy import signal
from collections import deque

logger = logging.getLogger(__name__)

class AudioPipeline:
    """
    Audio processing pipeline with sample rate conversion and normalization.
    """
    
    def __init__(self, 
                 input_sample_rate: int = 16000,
                 output_sample_rate: int = 48000,
                 normalize_audio: bool = True,
                 frame_size: int = 960):  # Match LiveKit's preferred frame size
        """
        Initialize audio pipeline.
        
        Args:
            input_sample_rate: Expected input sample rate (Hz)
            output_sample_rate: Target output sample rate (Hz)
            normalize_audio: Whether to normalize audio to [-1, 1]
            frame_size: Size of audio frames to process
        """
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.normalize_audio = normalize_audio
        self.frame_size = frame_size
        
        # Use deque for efficient buffer management
        self._buffer = deque(maxlen=frame_size * 4)  # 4x frame size for safety
        
        # Resampling state
        self._resampler = signal.resample_poly
        
        # Performance tracking
        self._processed_frames = 0
        self._total_samples = 0
        self._last_process_time = 0
        
        logger.info(f"Initialized AudioPipeline: {input_sample_rate}Hz → {output_sample_rate}Hz")
        
    def process_frame(self, frame_data: bytes, sample_rate: int) -> Optional[np.ndarray]:
        """
        Process a single audio frame.
        
        Args:
            frame_data: Raw audio frame data
            sample_rate: Sample rate of input data
            
        Returns:
            Processed audio data as float32 numpy array, or None if not enough data
        """
        try:
            # Convert bytes to numpy array
            data = np.frombuffer(frame_data, dtype=np.int16)
            
            # Convert to float32 and normalize if needed
            if self.normalize_audio:
                data = data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Resample if needed using high-quality polyphase resampling
            if sample_rate != self.output_sample_rate:
                up = self.output_sample_rate
                down = sample_rate
                data = self._resampler(data, up, down)
            
            # Add to buffer efficiently
            self._buffer.extend(data)
            
            # Extract frames if we have enough data
            if len(self._buffer) >= self.frame_size:
                frames = []
                while len(self._buffer) >= self.frame_size:
                    # Convert deque slice to numpy array
                    frame = np.array([self._buffer.popleft() for _ in range(self.frame_size)])
                    frames.append(frame)
                    
                self._processed_frames += len(frames)
                self._total_samples += len(frames) * self.frame_size
                
                return np.concatenate(frames)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}", exc_info=True)
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "processed_frames": self._processed_frames,
            "total_samples": self._total_samples,
            "buffer_samples": len(self._buffer),
            "input_rate": self.input_sample_rate,
            "output_rate": self.output_sample_rate
        }
        
    def reset(self) -> None:
        """Reset internal buffer and counters."""
        self._buffer.clear()
        self._processed_frames = 0
        self._total_samples = 0
        logger.info("Audio pipeline reset")
