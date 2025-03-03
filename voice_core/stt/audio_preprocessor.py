# voice_core/stt/audio_preprocessor.py
import logging
import numpy as np
from typing import Optional, Tuple
import scipy.signal

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Preprocesses audio for improved speech recognition with minimal conversions.
    Handles normalization, resampling, and noise reduction in an efficient pipeline.
    """
    
    def __init__(self,
                 target_sample_rate: int = 16000,
                 target_channels: int = 1,
                 enable_noise_reduction: bool = True,
                 enable_normalization: bool = True):
        """
        Initialize the audio preprocessor.
        
        Args:
            target_sample_rate: Target sample rate in Hz
            target_channels: Target number of channels (1=mono, 2=stereo)
            enable_noise_reduction: Whether to apply noise reduction
            enable_normalization: Whether to normalize audio levels
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_normalization = enable_normalization
        self.logger = logging.getLogger(__name__)
        
        # Design filters for noise reduction
        self.sos_filters = self._design_filters() if enable_noise_reduction else []
        
        # Noise floor estimation params
        self.noise_floor = -50.0  # dB
        self.noise_adaptation_rate = 0.05
        self.noise_floor_samples = []
        self.max_noise_samples = 100
        
    def _design_filters(self) -> list:
        """Design audio filters for noise reduction."""
        filters = []
        
        try:
            # Bandpass filter to focus on speech frequencies (300-3400 Hz)
            sos_bandpass = scipy.signal.butter(
                2, [300, 3400], btype='bandpass', 
                output='sos', fs=self.target_sample_rate
            )
            if isinstance(sos_bandpass, np.ndarray) and sos_bandpass.ndim == 2 and sos_bandpass.shape[1] == 6:
                self.logger.debug(f"Created bandpass filter with shape {sos_bandpass.shape}")
                filters.append(sos_bandpass)
            else:
                self.logger.warning(f"Skipping invalid bandpass filter: shape {sos_bandpass.shape if isinstance(sos_bandpass, np.ndarray) else 'not numpy array'}")
            
            # Notch filters for common noise frequencies
            for freq in [50, 60, 120, 240]:  # Power line frequencies and harmonics
                try:
                    # Convert iirnotch output to SOS format
                    b, a = scipy.signal.iirnotch(
                        freq, Q=30, fs=self.target_sample_rate
                    )
                    sos = scipy.signal.tf2sos(b, a)
                    
                    if isinstance(sos, np.ndarray) and sos.ndim == 2 and sos.shape[1] == 6:
                        self.logger.debug(f"Created notch filter for {freq}Hz with shape {sos.shape}")
                        filters.append(sos)
                    else:
                        self.logger.warning(f"Skipping invalid notch filter for {freq}Hz: shape {sos.shape if isinstance(sos, np.ndarray) else 'not numpy array'}")
                        
                except Exception as e:
                    self.logger.warning(f"Error creating notch filter for {freq}Hz: {e}")
                
        except Exception as e:
            self.logger.error(f"Error designing audio filters: {e}")
            
        self.logger.info(f"Created {len(filters)} audio filters for noise reduction")
        return filters
        
    def preprocess(self, audio_data: np.ndarray, source_sample_rate: int) -> Tuple[np.ndarray, float]:
        """
        Process audio data for improved recognition.
        
        Args:
            audio_data: Audio data as numpy array
            source_sample_rate: Source sample rate in Hz
            
        Returns:
            Tuple of (processed_audio, audio_level_db)
        """
        if audio_data.size == 0:
            return np.array([], dtype=np.float32), -100.0
            
        # 1. Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # 2. Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # 3. Resample if needed
        if source_sample_rate != self.target_sample_rate:
            audio_data = self._resample(audio_data, source_sample_rate)
            
        # 4. Calculate audio level before processing
        rms = np.sqrt(np.mean(np.square(audio_data)))
        audio_level_db = 20 * np.log10(rms + 1e-10)
        
        # 5. Apply noise reduction if enabled
        if self.enable_noise_reduction and self.sos_filters:
            audio_data = self._apply_noise_reduction(audio_data)
            
        # 6. Normalize if enabled
        if self.enable_normalization:
            audio_data = self._normalize(audio_data)
            
        # 7. Update noise floor estimate
        self._update_noise_floor(audio_data)
            
        return audio_data, audio_level_db
        
    def _resample(self, audio_data: np.ndarray, source_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio_data: Audio data as numpy array
            source_rate: Source sample rate in Hz
            
        Returns:
            Resampled audio data
        """
        try:
            # Calculate new length
            target_length = int(len(audio_data) * self.target_sample_rate / source_rate)
            
            # Use scipy for high-quality resampling
            resampled = scipy.signal.resample(audio_data, target_length)
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling audio: {e}")
            return audio_data
            
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction filters.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Filtered audio data
        """
        filtered_data = audio_data
        for i, sos in enumerate(self.sos_filters):
            try:
                # Validate SOS filter shape
                if not isinstance(sos, np.ndarray) or sos.ndim != 2 or sos.shape[1] != 6:
                    self.logger.warning(f"Skipping invalid SOS filter {i}: shape {sos.shape if isinstance(sos, np.ndarray) else 'not numpy array'}")
                    continue
                    
                filtered_data = scipy.signal.sosfilt(sos, filtered_data)
            except Exception as e:
                self.logger.warning(f"Error applying filter {i}: {e}")
                # Continue with unfiltered data if a filter fails
                
        return filtered_data
        
    def _normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have consistent volume.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Normalized audio data
        """
        # Skip empty arrays
        if audio_data.size == 0:
            return audio_data
            
        # Calculate max amplitude
        max_amp = np.max(np.abs(audio_data))
        
        # Normalize only if significant content
        if max_amp > 0.01:
            # Don't apply full normalization to avoid amplifying noise
            # Use a target level of 0.5 instead of 1.0
            audio_data = audio_data * (0.5 / max_amp)
            
        return audio_data
        
    def _update_noise_floor(self, audio_data: np.ndarray) -> None:
        """
        Update noise floor estimate using non-speech segments.
        
        Args:
            audio_data: Audio data as numpy array
        """
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(np.square(audio_data)))
        level_db = 20 * np.log10(rms + 1e-10)
        
        # Add to samples if it could be background noise (low level)
        if level_db < self.noise_floor + 10:
            self.noise_floor_samples.append(level_db)
            
            # Limit the number of samples
            if len(self.noise_floor_samples) > self.max_noise_samples:
                self.noise_floor_samples.pop(0)
                
            # Update noise floor estimate (using 10th percentile for robustness)
            if len(self.noise_floor_samples) >= 10:
                sorted_samples = sorted(self.noise_floor_samples)
                p10_idx = len(sorted_samples) // 10
                new_floor = sorted_samples[p10_idx]
                
                # Apply exponential smoothing
                self.noise_floor = (1 - self.noise_adaptation_rate) * self.noise_floor + \
                                   self.noise_adaptation_rate * new_floor
                                   
    def get_stats(self) -> dict:
        """Get statistics about audio preprocessing."""
        return {
            "target_sample_rate": self.target_sample_rate,
            "noise_reduction_enabled": self.enable_noise_reduction,
            "normalization_enabled": self.enable_normalization,
            "estimated_noise_floor": self.noise_floor,
            "filter_count": len(self.sos_filters)
        }