import numpy as np
import time

class AudioFrame:
    """Represents a frame of audio data with associated metadata."""
    
    def __init__(self, data: np.ndarray, sample_rate: int, num_channels: int, samples_per_channel: int, timestamp: float = None):
        """Initialize an audio frame.
        
        Args:
            data: Audio samples as a numpy array
            sample_rate: Sample rate in Hz
            num_channels: Number of audio channels
            samples_per_channel: Number of samples per channel
            timestamp: Optional timestamp in seconds. If None, current time is used.
        """
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
        self._energy = None
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    @property
    def energy(self) -> float:
        """Calculate and cache the energy of the frame."""
        if self._energy is None:
            self._energy = float(np.mean(np.abs(self.data)))
        return self._energy
    
    @energy.setter
    def energy(self, value: float):
        """Set the energy value directly."""
        self._energy = float(value)
    
    @property
    def duration(self) -> float:
        """Get the duration of the frame in seconds."""
        return self.samples_per_channel / self.sample_rate
    
    def resample(self, new_sample_rate: int) -> 'AudioFrame':
        """Create a new frame with resampled data.
        
        Args:
            new_sample_rate: Target sample rate in Hz
            
        Returns:
            A new AudioFrame with resampled data
        """
        if new_sample_rate == self.sample_rate:
            return self
        
        from scipy import signal
        resampled_length = int(len(self.data) * new_sample_rate / self.sample_rate)
        resampled_data = signal.resample(self.data, resampled_length)
        
        return AudioFrame(
            data=resampled_data,
            sample_rate=new_sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=len(resampled_data) // self.num_channels
        )
    
    def to_mono(self) -> 'AudioFrame':
        """Convert the frame to mono by averaging channels if necessary."""
        if self.num_channels == 1:
            return self
            
        mono_data = np.mean(
            self.data.reshape(-1, self.num_channels),
            axis=1
        )
        
        return AudioFrame(
            data=mono_data,
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_data)
        )
