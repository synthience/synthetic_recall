# voice_core/stt/vad_engine.py
import logging
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)

class VADEngine:
    """
    Enhanced Voice Activity Detection engine that processes audio frames 
    to determine speech presence with adaptive thresholding.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold_db: float = -35.0,
        silence_duration_sec: float = 0.8,
        min_speech_duration_sec: float = 0.5,
        max_speech_duration_sec: float = 30.0,
        energy_threshold_boost: float = 3.0,
        speech_confidence_threshold: float = 0.7
    ):
        """
        Initialize the VAD engine.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
            energy_threshold_db: Initial energy threshold in dB
            silence_duration_sec: Silence duration to end speech detection
            min_speech_duration_sec: Minimum speech duration to consider valid
            max_speech_duration_sec: Maximum speech duration before forced end
            energy_threshold_boost: Boost for threshold during active speech
            speech_confidence_threshold: Confidence threshold for speech detection
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold_db = energy_threshold_db
        self.silence_duration_sec = silence_duration_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_speech_duration_sec = max_speech_duration_sec
        self.energy_threshold_boost = energy_threshold_boost
        self.speech_confidence_threshold = speech_confidence_threshold
        
        # VAD state
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.speech_duration = 0.0
        self.last_speech_time = 0.0
        self.silence_start_time = 0.0
        
        # Dynamic parameters
        self.noise_floor_db = -60.0
        self.speech_energy_history = deque(maxlen=100)
        self.background_energy_history = deque(maxlen=200)
        self.triggered_count = 0
        self.untriggered_count = 0
        
        # Smoothing
        self.energy_smoothing_alpha = 0.3  # For energy level smoothing
        self.smoothed_energy_db = self.energy_threshold_db
        
        # Statistics
        self.speech_segments_detected = 0
        self.false_triggers = 0
        self.total_speech_duration = 0.0
        
        self.logger = logging.getLogger(__name__)
        
    def process_frame(self, audio_frame: np.ndarray, audio_level_db: float) -> Dict[str, Any]:
        """
        Process an audio frame to detect speech.
        
        Args:
            audio_frame: Audio frame as numpy array
            audio_level_db: Audio level in dB
            
        Returns:
            Dict with detection results
        """
        current_time = time.time()
        
        # Update smoothed energy
        self.smoothed_energy_db = (self.smoothed_energy_db * (1 - self.energy_smoothing_alpha) + 
                                 audio_level_db * self.energy_smoothing_alpha)
        
        # Add to appropriate energy history
        if self.is_speaking:
            self.speech_energy_history.append(audio_level_db)
        else:
            self.background_energy_history.append(audio_level_db)
            
        # Update noise floor and energy threshold
        self._update_noise_floor()
        
        # Calculate adaptive threshold
        adaptive_threshold = max(
            self.noise_floor_db + 15.0,  # At least 15dB above noise floor
            self.energy_threshold_db
        )
        
        # Boost threshold during active speech for better sensitivity
        if self.is_speaking:
            adaptive_threshold += self.energy_threshold_boost
            
        # Enhanced speech detection 
        is_speech = self._is_speech_frame(audio_frame, self.smoothed_energy_db, adaptive_threshold)
        
        # Update state based on detection
        result = self._update_vad_state(is_speech, current_time)
        
        # Add debug data
        result.update({
            "audio_level_db": audio_level_db,
            "smoothed_energy_db": self.smoothed_energy_db,
            "adaptive_threshold": adaptive_threshold,
            "noise_floor_db": self.noise_floor_db
        })
        
        return result
        
    def _is_speech_frame(self, audio_frame: np.ndarray, energy_db: float, threshold_db: float) -> bool:
        """
        Determine if a frame contains speech using multiple features.
        
        Args:
            audio_frame: Audio frame as numpy array
            energy_db: Energy level in dB
            threshold_db: Energy threshold in dB
            
        Returns:
            True if frame contains speech, False otherwise
        """
        # Primary check: energy level
        if energy_db < threshold_db:
            self.untriggered_count += 1
            self.triggered_count = max(0, self.triggered_count - 1)
            return False
            
        # Additional checks could include:
        # 1. Zero-crossing rate for fricatives
        # 2. Spectral flatness for tonal sounds vs noise
        # 3. Spectral centroid for speech formants
        
        # For now, use simple energy with hysteresis
        self.triggered_count += 1
        self.untriggered_count = 0
        
        # Require a few consecutive frames above threshold for robustness
        return self.triggered_count >= 2
        
    def _update_vad_state(self, is_speech: bool, current_time: float) -> Dict[str, Any]:
        """
        Update VAD state based on current frame detection.
        
        Args:
            is_speech: Whether current frame contains speech
            current_time: Current timestamp
            
        Returns:
            Dict with state update results
        """
        result = {
            "is_speech": is_speech,
            "is_speaking": self.is_speaking,
            "speech_segment_complete": False,
            "valid_speech_segment": False
        }
        
        if is_speech:
            if not self.is_speaking:
                # Speech start
                self.is_speaking = True
                self.speech_start_time = current_time
                self.speech_duration = 0.0
                self.logger.debug(f"Speech start detected at {self.smoothed_energy_db:.1f}dB")
                
            # Update speech tracking
            self.last_speech_time = current_time
            self.speech_duration = current_time - self.speech_start_time
            self.silence_start_time = 0
            
            # Check for max duration
            if self.speech_duration >= self.max_speech_duration_sec:
                # Force end of speech segment
                self.is_speaking = False
                self.speech_segments_detected += 1
                self.total_speech_duration += self.speech_duration
                
                result["speech_segment_complete"] = True
                result["valid_speech_segment"] = self.speech_duration >= self.min_speech_duration_sec
                result["speech_duration"] = self.speech_duration
                
                self.logger.debug(f"Speech segment force-ended at max duration: {self.speech_duration:.2f}s")
                
        else:  # Not speech
            if self.is_speaking:
                # Potential speech end - track silence
                if self.silence_start_time == 0:
                    self.silence_start_time = current_time
                    
                # Check if silence duration threshold reached
                silence_duration = current_time - self.silence_start_time
                
                if silence_duration >= self.silence_duration_sec:
                    # End of speech segment detected
                    self.is_speaking = False
                    self.speech_segments_detected += 1
                    self.total_speech_duration += self.speech_duration
                    
                    result["speech_segment_complete"] = True
                    result["valid_speech_segment"] = self.speech_duration >= self.min_speech_duration_sec
                    result["speech_duration"] = self.speech_duration
                    
                    self.logger.debug(f"Speech segment ended after {self.speech_duration:.2f}s, " + 
                                      f"silence: {silence_duration:.2f}s")
                    
                    # Check for false trigger
                    if self.speech_duration < self.min_speech_duration_sec:
                        self.false_triggers += 1
        
        # Update result with current state
        result["is_speaking"] = self.is_speaking
        result["speech_duration"] = self.speech_duration if self.is_speaking else 0.0
        
        return result
        
    def _update_noise_floor(self) -> None:
        """Update noise floor estimate using background energy history."""
        if len(self.background_energy_history) >= 20:
            # Use 20th percentile for robust noise floor estimation
            sorted_bg = sorted(self.background_energy_history)
            idx = max(0, int(len(sorted_bg) * 0.2))
            new_floor = sorted_bg[idx]
            
            # Smooth the update
            alpha = 0.1  # Low-pass filter coefficient
            self.noise_floor_db = (1 - alpha) * self.noise_floor_db + alpha * new_floor
            
            # Ensure noise floor doesn't go too low
            self.noise_floor_db = max(self.noise_floor_db, -65.0)
            
    def reset(self) -> None:
        """Reset VAD state for a new session."""
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.speech_duration = 0.0
        self.last_speech_time = 0.0
        self.silence_start_time = 0.0
        self.triggered_count = 0
        self.untriggered_count = 0
        
    def handle_interruption(self) -> None:
        """Handle an interruption event by resetting state."""
        if self.is_speaking:
            self.logger.debug("VAD interrupted while speaking")
            self.is_speaking = False
            self.speech_duration = 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get VAD engine statistics."""
        return {
            "speech_segments_detected": self.speech_segments_detected,
            "false_triggers": self.false_triggers,
            "total_speech_duration": self.total_speech_duration,
            "current_noise_floor": self.noise_floor_db,
            "current_energy_threshold": self.energy_threshold_db
        }