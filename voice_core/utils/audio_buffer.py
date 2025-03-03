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