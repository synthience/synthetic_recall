# __init__.py

```py
# voice_core/stt/__init__.py
"""Speech-to-Text services."""

from __future__ import annotations
from .base import STTService
from .enhanced_stt_service import EnhancedSTTService
from .livekit_identity_manager import LiveKitIdentityManager
from .audio_preprocessor import AudioPreprocessor
from .vad_engine import VADEngine
from .streaming_stt import StreamingSTT
from .transcription_publisher import TranscriptionPublisher

__all__ = [
    'STTService', 
    'EnhancedSTTService',
    'LiveKitIdentityManager',
    'AudioPreprocessor',
    'VADEngine',
    'StreamingSTT',
    'TranscriptionPublisher'
]
```

# audio_preprocessor.py

```py
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
```

# base.py

```py
"""Base class for Speech-to-Text services."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from livekit import rtc

class STTService(ABC):
    """Abstract base class for Speech-to-Text services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT service."""
        pass

    @abstractmethod
    async def process_audio(self, track: Optional[rtc.AudioTrack]) -> Optional[str]:
        """Process audio from track and return transcription."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

```

# check_python_path.py

```py
"""
Check Python path and installed packages
"""

import sys
import subprocess
import os

def main():
    # Print Python path
    print("Python executable:", sys.executable)
    print("\nPython path:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Check for faster-whisper using pip
    print("\nChecking for faster-whisper using pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "faster-whisper"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("faster-whisper is installed:")
            print(result.stdout)
        else:
            print("faster-whisper is not installed according to pip")
            print(result.stderr)
    except Exception as e:
        print(f"Error checking pip: {e}")
    
    # Try to find the package manually
    print("\nSearching for faster-whisper in site-packages...")
    for path in sys.path:
        if "site-packages" in path:
            try:
                contents = os.listdir(path)
                faster_whisper_items = [item for item in contents if "faster" in item.lower() and "whisper" in item.lower()]
                if faster_whisper_items:
                    print(f"Found in {path}:")
                    for item in faster_whisper_items:
                        print(f"  - {item}")
            except Exception as e:
                print(f"Error checking {path}: {e}")
    
    # Check if we can import the package components
    print("\nTrying to import faster_whisper components...")
    try:
        import faster_whisper
        print("Successfully imported faster_whisper as a module")
        print(f"Module location: {faster_whisper.__file__}")
    except ImportError as e:
        print(f"Failed to import faster_whisper: {e}")
        
        # Try with underscores
        try:
            import faster_whisper
            print("Successfully imported faster_whisper (with underscore)")
        except ImportError as e:
            print(f"Failed to import faster_whisper (with underscore): {e}")

if __name__ == "__main__":
    main()

```

# enhanced_stt_service.py

```py
# voice_core/stt/enhanced_stt_service.py
import asyncio
import logging
import time
import json
import numpy as np
from typing import Optional, Callable, Any, Dict, List, AsyncIterator
import livekit.rtc as rtc

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.livekit_identity_manager import LiveKitIdentityManager
from voice_core.stt.audio_preprocessor import AudioPreprocessor
from voice_core.stt.vad_engine import VADEngine
from voice_core.stt.streaming_stt import StreamingSTT
from voice_core.stt.transcription_publisher import TranscriptionPublisher

logger = logging.getLogger(__name__)

class EnhancedSTTService:
    """
    Enhanced STT service that processes audio frames and sends transcripts to the VoiceStateManager.
    Uses a modular pipeline architecture for improved performance and maintainability.
    """

    def __init__(
        self,
        state_manager: VoiceStateManager,
        whisper_model: str = "small.en",
        device: str = "cpu",
        min_speech_duration: float = 0.3,  # Reduced from 0.5
        max_speech_duration: float = 30.0,
        energy_threshold: float = 0.05,
        on_transcript: Optional[Callable[[str], Any]] = None
    ):
        """
        Initialize the enhanced STT service with modular components.
        
        Args:
            state_manager: Voice state manager for state tracking
            whisper_model: Whisper model name to use
            device: Device to run inference on ("cpu" or "cuda")
            min_speech_duration: Minimum duration to consider valid speech
            max_speech_duration: Maximum duration for a speech segment
            energy_threshold: Initial energy threshold for speech detection
            on_transcript: Optional callback for final transcripts
        """
        self.state_manager = state_manager
        self.on_transcript = on_transcript
        
        # Initialize modular components
        self.identity_manager = LiveKitIdentityManager()
        self.audio_preprocessor = AudioPreprocessor(target_sample_rate=16000)
        self.vad_engine = VADEngine(
            min_speech_duration_sec=min_speech_duration,
            max_speech_duration_sec=max_speech_duration,
            energy_threshold_db=-40.0  # dB threshold corresponding to energy_threshold
        )
        self.transcriber = StreamingSTT(
            model_name=whisper_model,
            device=device,
            language="en"
        )
        self.publisher = TranscriptionPublisher(state_manager)
        
        # Speech buffer for full segment processing
        self.buffer = []
        self.buffer_duration = 0.0
        
        # Processing state
        self.sample_rate = 16000
        self.is_processing = False
        self.processing_lock = asyncio.Lock()
        self.active_task = None
        self.room = None
        self._participant_identity = None
        
        # Performance tracking
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.error_count = 0
        self.interruptions_detected = 0
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize all STT components."""
        try:
            self.logger.info("Initializing Enhanced STT Service")
            
            # Initialize the transcriber
            await self.transcriber.initialize()
            
            self.logger.info("STT service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize STT service: {e}", exc_info=True)
            raise
            
    def set_room(self, room: rtc.Room) -> None:
        """
        Set LiveKit room for STT processing.
        
        Args:
            room: LiveKit room instance
        """
        self.room = room
        self.publisher.set_room(room)
        
        # Publish initialization status
        if self.room and self.state_manager:
            try:
                self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "stt_initialized",
                        "models": {
                            "whisper": self.transcriber.model_name if hasattr(self.transcriber, "model_name") else "unknown"
                        },
                        "device": self.transcriber.device if hasattr(self.transcriber, "device") else "cpu",
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
                
                if self.state_manager.current_state not in [VoiceState.SPEAKING, VoiceState.PROCESSING]:
                    asyncio.create_task(self.state_manager.transition_to(VoiceState.LISTENING))
                
                self.logger.debug("Published initialization status to room")
            except Exception as e:
                self.logger.error(f"Failed to publish initialization status: {e}")

    async def process_audio(self, track: rtc.AudioTrack) -> Optional[str]:
        """
        Process audio from a LiveKit track.
        
        Args:
            track: LiveKit audio track to process
            
        Returns:
            Final transcript or None if processing failed
        """
        self.logger.info(f"Processing audio track: sid={track.sid if hasattr(track, 'sid') else 'unknown'}")
        
        # Check initialization
        if not hasattr(self.transcriber, "model") or self.transcriber.model is None:
            self.logger.error("STT not fully initialized")
            return None
        
        # Acquire processing lock to prevent concurrent processing
        async with self.processing_lock:
            if self.is_processing:
                self.logger.warning("Already processing audio, skipping this call")
                return None
            self.is_processing = True
        
        # Setup cleanup for any case
        async def cleanup():
            async with self.processing_lock:
                self.is_processing = False
                self.active_task = None
                
        try:
            # Get participant identity
            self._participant_identity = self.identity_manager.get_participant_identity(track, self.room)
            self.logger.info(f"Processing audio from participant: {self._participant_identity}")
            
            # Create audio stream
            audio_stream = rtc.AudioStream(track)
            
            # Publish listening state
            if self.room and self.state_manager and self.state_manager.current_state != VoiceState.SPEAKING:
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            "type": "listening_state",
                            "active": True,
                            "timestamp": time.time()
                        }).encode(),
                        reliable=True
                    )
                    
                    if self.state_manager.current_state not in [VoiceState.SPEAKING, VoiceState.PROCESSING]:
                        await self.state_manager.transition_to(VoiceState.LISTENING)
                        
                    self.logger.debug("Published listening state to room")
                except Exception as e:
                    self.logger.error(f"Failed to publish listening state: {e}")
            
            # Process audio frames
            async for event in audio_stream:
                # Check for error state
                if self.state_manager.current_state == VoiceState.ERROR:
                    self.logger.info("Stopping audio processing due to ERROR state")
                    await cleanup()
                    break
                
                # Skip empty frames
                frame = event.frame
                if frame is None:
                    continue
                
                # Convert to numpy array
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                
                # Preprocess audio
                processed_audio, audio_level_db = self.audio_preprocessor.preprocess(
                    audio_data,
                    frame.sample_rate
                )
                
                # Process with VAD engine
                vad_result = self.vad_engine.process_frame(processed_audio, audio_level_db)
                
                # Check for a completed speech segment
                if vad_result["speech_segment_complete"] and vad_result["valid_speech_segment"]:
                    self.logger.info(f"Speech segment complete: {vad_result['speech_duration']:.2f}s")
                    
                    # Process full speech segment
                    if self.buffer:
                        # Combine buffer into a single array
                        full_audio = np.concatenate(self.buffer)
                        
                        # Transcribe the full segment
                        transcription_result = await self.transcriber.transcribe(full_audio, self.sample_rate)
                        
                        if transcription_result["success"] and transcription_result["text"]:
                            transcript = transcription_result["text"]
                            
                            # Publish transcript
                            await self.publisher.publish_transcript(
                                transcript,
                                self._participant_identity,
                                is_final=True
                            )
                            
                            # Update stats
                            self.successful_recognitions += 1
                            
                            # Call transcript handler if provided
                            if self.on_transcript:
                                if asyncio.iscoroutinefunction(self.on_transcript):
                                    await self.on_transcript(transcript)
                                else:
                                    self.on_transcript(transcript)
                                    
                            self.logger.info(f"Published transcript: '{transcript[:50]}...'")
                        
                        # Clear buffer for next segment
                        self.buffer = []
                        self.buffer_duration = 0.0
                        
                # Buffer audio during active speech
                if vad_result["is_speaking"]:
                    self.buffer.append(processed_audio)
                    frame_duration = len(processed_audio) / self.sample_rate
                    self.buffer_duration += frame_duration
            
            await cleanup()
            return None
                
        except asyncio.CancelledError:
            self.logger.info("Audio processing task cancelled")
            await cleanup()
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}", exc_info=True)
            self.error_count += 1
            if self.state_manager:
                await self.state_manager.register_error(e, "stt_processing")
            await cleanup()
            return None

    async def clear_buffer(self) -> None:
        """Clear audio buffer between turns."""
        try:
            self.buffer = []
            self.buffer_duration = 0.0
            self.vad_engine.reset()
            self.logger.debug("STT buffer cleared")
        except Exception as e:
            self.logger.error(f"Error clearing STT buffer: {e}")

    async def stop_processing(self) -> None:
        """Stop any active processing."""
        async with self.processing_lock:
            if self.active_task and not self.active_task.done():
                self.active_task.cancel()
                try:
                    await asyncio.wait_for(self.active_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    self.logger.error(f"Error cancelling active task: {e}")
                finally:
                    self.active_task = None
            self.is_processing = False
            
        await self.clear_buffer()
        self.logger.info("Audio processing stopped")

    async def cleanup(self) -> None:
        """Clean up STT service resources."""
        self.logger.info("Cleaning up STT service")
        
        await self.stop_processing()
        
        # Publish cleanup event
        if self.room and self.state_manager:
            try:
                await self.state_manager._publish_with_retry(
                    json.dumps({
                        "type": "stt_cleanup",
                        "timestamp": time.time()
                    }).encode(),
                    "STT cleanup"
                )
            except Exception as e:
                self.logger.error(f"Failed to publish cleanup: {e}")
        
        # Clean up transcriber
        await self.transcriber.cleanup()
        
        # Clear buffer
        self.buffer = []
        self.buffer_duration = 0.0
        self._participant_identity = None
        
        self.logger.info("STT service cleanup complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get STT service statistics."""
        component_stats = {
            "audio_preprocessor": self.audio_preprocessor.get_stats(),
            "vad_engine": self.vad_engine.get_stats(),
            "transcriber": self.transcriber.get_stats(),
            "publisher": self.publisher.get_stats(),
            "identity_manager": self.identity_manager.get_stats()
        }
        
        return {
            "total_recognitions": self.total_recognitions,
            "successful_recognitions": self.successful_recognitions,
            "error_count": self.error_count,
            "success_rate": float(self.successful_recognitions) / max(int(self.total_recognitions), 1),
            "current_buffer_duration": self.buffer_duration,
            "interruptions_detected": self.interruptions_detected,
            **component_stats
        }
```

# install_dependencies.bat

```bat
@echo off
echo ===================================================
echo Installing STT Dependencies
echo ===================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "..\..\venv\Scripts\activate.bat"
) else (
    echo No virtual environment found. Installing in global Python.
)

echo.
echo Installing packages from requirements.txt...
echo.

REM Install packages from requirements.txt
python -m pip install -r requirements.txt

echo.
echo ===================================================
echo Installation complete!
echo ===================================================
echo.
echo To verify the installation, run: python test_imports.py
echo.

pause

```

# install_dependencies.py

```py
import subprocess
import sys
import time
import os

def run_pip_install(package):
    """Run pip install for a single package with error handling"""
    print(f"\n{'='*80}\nInstalling {package}...\n{'='*80}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully installed {package}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Read requirements from file
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    # Parse requirements, skipping comments and empty lines
    packages = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            packages.append(line)
    
    print(f"Found {len(packages)} packages to install")
    
    # Install packages one by one
    successful = []
    failed = []
    
    for package in packages:
        if run_pip_install(package):
            successful.append(package)
        else:
            failed.append(package)
        # Small delay to avoid overwhelming the console
        time.sleep(0.5)
    
    # Summary
    print("\n\n" + "="*80)
    print(f"Installation Summary:")
    print(f"Successfully installed: {len(successful)}/{len(packages)}")
    if failed:
        print(f"\nFailed packages ({len(failed)}):")
        for pkg in failed:
            print(f"  - {pkg}")
    
    print("\nYou can try installing failed packages manually with:")
    print("pip install <package-name> --verbose")
    
    # Create a file with failed packages for easy retry
    if failed:
        with open("failed_packages.txt", "w") as f:
            for pkg in failed:
                f.write(f"{pkg}\n")
        print("\nFailed packages have been written to 'failed_packages.txt'")

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()

```

# install_minimal_deps.py

```py
#!/usr/bin/env python
"""
Script to install minimal dependencies for the enhanced STT service.
This script installs only the essential packages needed for basic STT functionality.
"""

import os
import sys
import subprocess
import argparse
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define minimal dependencies
MINIMAL_DEPS = [
    "numpy>=1.20.0",
    "torch>=1.13.0",
    "torchaudio>=0.13.0",
    "faster-whisper>=0.6.0",
    "webrtcvad>=2.0.10",
    "soundfile>=0.12.1",
    "df-nightly",  # DeepFilterNet for noise reduction
    "tokenizers==0.21.0",  # Fixed version for compatibility
]

# Optional dependencies
OPTIONAL_DEPS = {
    "diarization": ["pyannote.audio>=2.1.1"],
    "vad": ["silero-vad"],
}

def install_packages(packages: List[str], upgrade: bool = False) -> bool:
    """
    Install packages using pip.
    
    Args:
        packages: List of packages to install
        upgrade: Whether to upgrade existing packages
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install minimal dependencies for STT service")
    parser.add_argument("--all", action="store_true", help="Install all dependencies including optional ones")
    parser.add_argument("--diarization", action="store_true", help="Install speaker diarization dependencies")
    parser.add_argument("--vad", action="store_true", help="Install neural VAD dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    args = parser.parse_args()
    
    # Install minimal dependencies
    logger.info("Installing minimal dependencies...")
    success = install_packages(MINIMAL_DEPS, args.upgrade)
    if not success:
        logger.error("Failed to install minimal dependencies")
        sys.exit(1)
    
    # Install optional dependencies
    if args.all or args.diarization:
        logger.info("Installing speaker diarization dependencies...")
        install_packages(OPTIONAL_DEPS["diarization"], args.upgrade)
    
    if args.all or args.vad:
        logger.info("Installing neural VAD dependencies...")
        install_packages(OPTIONAL_DEPS["vad"], args.upgrade)
    
    logger.info("Installation completed successfully")

if __name__ == "__main__":
    main()

```

# livekit_identity_manager.py

```py
# voice_core/stt/livekit_identity_manager.py
import logging
from typing import Optional
import livekit.rtc as rtc

logger = logging.getLogger(__name__)

class LiveKitIdentityManager:
    """
    Manages LiveKit participant identity tracking for accurate transcript attribution.
    Extracts participant identity from tracks in a consistent manner.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_participant_identity(self, track: rtc.AudioTrack, room: Optional[rtc.Room] = None) -> Optional[str]:
        """
        Extract participant identity from LiveKit track with fallbacks.
        
        Args:
            track: The audio track to identify
            room: Optional room object for additional lookup methods
            
        Returns:
            Participant identity or None if not identifiable
        """
        participant_identity = None
        
        # Method 1: Direct participant identity from track
        if hasattr(track, 'participant') and track.participant:
            if hasattr(track.participant, 'identity'):
                participant_identity = track.participant.identity
                self.logger.debug(f"Got identity from track.participant: {participant_identity}")
                return participant_identity
        
        # Method 2: Look up by track SID in room participants
        if not participant_identity and room and hasattr(track, 'sid'):
            track_sid = track.sid
            for participant in room.remote_participants.values():
                for pub in participant.track_publications.values():
                    if pub.track and pub.track.sid == track_sid:
                        participant_identity = participant.identity
                        self.logger.debug(f"Found identity by track SID lookup: {participant_identity}")
                        return participant_identity
        
        # Method 3: Check stream if available
        if not participant_identity and hasattr(track, 'stream_id'):
            stream_id = track.stream_id
            if stream_id and "-" in stream_id:
                # Sometimes the stream ID contains the participant identity
                parts = stream_id.split("-")
                if len(parts) >= 2:
                    participant_identity = parts[0]
                    self.logger.debug(f"Extracted identity from stream ID: {participant_identity}")
                    return participant_identity
                
        self.logger.warning(f"Could not determine participant identity for track {track.sid if hasattr(track, 'sid') else 'unknown'}")
        return "unknown_user"  # Default fallback
    
    def get_stats(self) -> dict:
        """Get statistics about identity resolution."""
        return {
            "identity_manager_active": True,
        }
```

# requirements.txt

```txt
# Core dependencies
numpy>=1.20.0
torch>=1.13.0
torchaudio>=0.13.0
faster-whisper>=0.6.0
webrtcvad>=2.0.10
soundfile>=0.12.1
df-nightly
tokenizers==0.21.0

# Optional: Speaker diarization
pyannote.audio>=2.1.1

# Optional: Neural VAD
silero-vad

# Optional: VOSK (for alternative STT)
vosk>=0.3.45

# Optional: Whisper & Faster-Whisper (Optimized STT)
openai-whisper>=20231117
faster-whisper @ git+https://github.com/guillaumekln/faster-whisper.git

# Optional: AI-Based Noise Suppression
df-nightly>=0.5.6

# Optional: Voice Activity Detection (VAD)
torchvision>=0.16.0

# Optional: Audio Processing
librosa>=0.10.1
ffmpeg-python>=0.2.0

# Optional: JSON & Async Management
aiohttp>=3.9.0

# Optional: Logging & Debugging
loguru>=0.7.2

```

# streaming_stt.py

```py
# voice_core/stt/streaming_stt.py
import logging
import asyncio
import numpy as np
import time
import tempfile
import os
from typing import Optional, Dict, Any, List, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class StreamingSTT:
    """
    Streaming speech-to-text engine that converts audio to text
    with optimized performance and real-time processing capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        language: str = "en",
        compute_type: str = "float16",
        on_partial_transcript: Optional[Callable[[str, float], None]] = None,
        fine_tuned_model_path: Optional[str] = None,
        use_fine_tuned_model: bool = False
    ):
        """
        Initialize the streaming STT engine.
        
        Args:
            model_name: Whisper model name to use
            device: Device to run inference on ("cpu" or "cuda")
            language: Language code for recognition
            compute_type: Computation type (float16, float32, etc.)
            on_partial_transcript: Optional callback for partial transcripts
            fine_tuned_model_path: Path to fine-tuned model checkpoint
            use_fine_tuned_model: Whether to use the fine-tuned model
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.compute_type = compute_type
        self.on_partial_transcript = on_partial_transcript
        self.fine_tuned_model_path = fine_tuned_model_path
        self.use_fine_tuned_model = use_fine_tuned_model
        
        # Processing state
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._whisper_loaded = False
        self._vad_loaded = False
        
        # Statistics
        self.transcriptions_count = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0
        self.avg_real_time_factor = 0.0
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self) -> None:
        """Initialize the STT engine and load models."""
        try:
            # Import whisper here to avoid early loading
            try:
                import whisper
                self.whisper = whisper
            except ImportError:
                try:
                    from faster_whisper import WhisperModel
                    self.whisper = None  # Using faster_whisper instead
                    self.model = WhisperModel(
                        self.model_name, 
                        device=self.device, 
                        compute_type=self.compute_type
                    )
                    self._whisper_loaded = True
                    self.logger.info(f"Loaded faster-whisper model '{self.model_name}' on {self.device}")
                except ImportError:
                    self.logger.error("Neither whisper nor faster-whisper is installed")
                    return
            
            # Load model if using standard whisper
            if self.whisper and not self._whisper_loaded:
                loop = asyncio.get_event_loop()
                
                # Check if we should use the fine-tuned model
                if self.use_fine_tuned_model and self.fine_tuned_model_path and os.path.exists(self.fine_tuned_model_path):
                    # First load the base model
                    base_model = await loop.run_in_executor(
                        self.executor,
                        lambda: self.whisper.load_model(self.model_name, device=self.device)
                    )
                    
                    # Then load the fine-tuned model weights
                    self.logger.info(f"Loading fine-tuned model from {self.fine_tuned_model_path}")
                    import torch
                    checkpoint = await loop.run_in_executor(
                        self.executor,
                        lambda: torch.load(self.fine_tuned_model_path, map_location=self.device)
                    )
                    
                    # Apply the weights to the base model
                    if "model_state_dict" in checkpoint:
                        await loop.run_in_executor(
                            self.executor,
                            lambda: base_model.load_state_dict(checkpoint["model_state_dict"])
                        )
                    else:
                        await loop.run_in_executor(
                            self.executor,
                            lambda: base_model.load_state_dict(checkpoint)
                        )
                    
                    self.model = base_model
                    self._whisper_loaded = True
                    self.logger.info(f"Successfully loaded fine-tuned whisper model on {self.device}")
                else:
                    # Load the standard model
                    self.model = await loop.run_in_executor(
                        self.executor,
                        lambda: self.whisper.load_model(self.model_name, device=self.device)
                    )
                    self._whisper_loaded = True
                    self.logger.info(f"Loaded whisper model '{self.model_name}' on {self.device}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize STT engine: {e}")
            raise
            
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Dict with transcription results
        """
        if not self._whisper_loaded or self.model is None:
            self.logger.error("STT engine not initialized")
            return {"text": "", "success": False, "error": "Model not loaded"}
            
        if audio_data.size == 0:
            return {"text": "", "success": True}
            
        start_time = time.time()
        self.total_audio_duration += len(audio_data) / sample_rate
        
        try:
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to file
                import scipy.io.wavfile
                scipy.io.wavfile.write(temp_path, sample_rate, audio_data)
            
            try:
                # Transcribe audio using model
                if hasattr(self.model, 'transcribe'):  # Original whisper
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda: self.model.transcribe(
                            temp_path,
                            language=self.language,
                            fp16=(self.device == "cuda")
                        )
                    )
                    text = result["text"].strip()
                else:  # faster-whisper
                    # Run in executor to prevent blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda: self.model.transcribe(
                            temp_path,
                            language=self.language,
                            beam_size=5
                        )
                    )
                    segments, _ = result
                    text = " ".join([segment.text for segment in segments]).strip()
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
            # Calculate processing time and stats
            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sample_rate
            real_time_factor = processing_time / max(audio_duration, 0.1)
            
            # Update statistics
            self.transcriptions_count += 1
            self.total_processing_time += processing_time
            
            # Update average real-time factor with exponential moving average
            if self.avg_real_time_factor == 0:
                self.avg_real_time_factor = real_time_factor
            else:
                alpha = 0.1  # Smoothing factor
                self.avg_real_time_factor = (1 - alpha) * self.avg_real_time_factor + alpha * real_time_factor
                
            self.logger.info(f"Transcription completed in {processing_time:.2f}s " +
                           f"(RTF: {real_time_factor:.2f}x): '{text[:50]}...'")
                           
            return {
                "text": text,
                "success": True,
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "real_time_factor": real_time_factor
            }
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return {"text": "", "success": False, "error": str(e)}
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        self.model = None
        self._whisper_loaded = False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get STT engine statistics."""
        return {
            "transcriptions_count": self.transcriptions_count,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "avg_real_time_factor": self.avg_real_time_factor,
            "model_name": self.model_name,
            "device": self.device,
            "whisper_loaded": self._whisper_loaded
        }
```

# test_deepfilter.py

```py
"""
Test script to check the correct import and usage of deepfilternet
"""

# Try different import approaches
print("Attempting imports...")

try:
    import deepfilternet
    print("✅ Successfully imported deepfilternet")
    print(f"Module location: {deepfilternet.__file__}")
    print(f"Available attributes: {dir(deepfilternet)}")
except ImportError as e:
    print(f"❌ Failed to import deepfilternet: {e}")

try:
    from deepfilternet import DeepFilterNet
    print("✅ Successfully imported DeepFilterNet class")
except ImportError as e:
    print(f"❌ Failed to import DeepFilterNet: {e}")

try:
    import df
    print("✅ Successfully imported df")
    print(f"Module location: {df.__file__}")
    print(f"Available attributes: {dir(df)}")
except ImportError as e:
    print(f"❌ Failed to import df: {e}")

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    print("✅ Successfully imported df.enhance functions")
except ImportError as e:
    print(f"❌ Failed to import df.enhance: {e}")

# Try to find the package in site-packages
import sys
import os

print("\nSearching for deepfilter in site-packages...")
for path in sys.path:
    if "site-packages" in path:
        try:
            contents = os.listdir(path)
            deepfilter_items = [item for item in contents if "deep" in item.lower() and "filter" in item.lower()]
            if deepfilter_items:
                print(f"Found in {path}:")
                for item in deepfilter_items:
                    print(f"  - {item}")
        except Exception as e:
            print(f"Error checking {path}: {e}")

```

# test_imports.py

```py
"""
Test script to verify that all required packages are correctly installed.
This script attempts to import each package and reports success or failure.
"""

import sys
import importlib

def test_import(package_name):
    """Test importing a package and report success or failure."""
    try:
        # For packages with version specifiers, extract just the package name
        if '>=' in package_name:
            package_name = package_name.split('>=')[0]
        elif '==' in package_name:
            package_name = package_name.split('==')[0]
        
        # Special case for packages with @ notation
        if '@' in package_name:
            package_name = package_name.split('@')[0].strip()
            
        # Handle special cases
        if package_name == 'openai-whisper':
            package_name = 'whisper'
        elif package_name == 'ffmpeg-python':
            package_name = 'ffmpeg'
        elif package_name == 'faster-whisper':
            package_name = 'faster_whisper'  # Use underscore instead of hyphen
        elif package_name == 'df-nightly':
            package_name = 'df'  # DeepFilter package
        elif package_name == 'pyannote.audio':
            package_name = 'pyannote.audio'
            
        importlib.import_module(package_name)
        print(f"✅ Successfully imported {package_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {e}")
        return False

def test_df_import():
    """Test df-nightly specific imports and print available modules"""
    try:
        import df
        print("\nTesting df-nightly imports:")
        print(f"df version: {df.__version__ if hasattr(df, '__version__') else 'unknown'}")
        print("\nAvailable modules in df:")
        for item in dir(df):
            if not item.startswith('_'):
                print(f"- {item}")
                
        print("\nContents of df.enhance:")
        import df.enhance
        for item in dir(df.enhance):
            if not item.startswith('_'):
                print(f"- {item}")
    except Exception as e:
        print(f"Error testing df-nightly: {e}")

def main():
    # List of packages to test
    packages = [
        "torch",
        "torchaudio",
        "numpy",
        "soundfile",
        "scipy",
        "openai-whisper",
        "faster-whisper",
        "vosk",
        "webrtcvad",
        "torchvision",
        "librosa",
        "ffmpeg-python",
        "aiohttp",
        "loguru",
        "df-nightly",
        "pyannote.audio"
    ]
    
    print(f"Testing imports for {len(packages)} packages...\n")
    
    successful = 0
    failed = []
    
    for package in packages:
        if test_import(package):
            successful += 1
        else:
            failed.append(package)
    
    # Summary
    print("\n" + "="*80)
    print(f"Import Test Summary:")
    print(f"Successfully imported: {successful}/{len(packages)}")
    
    if failed:
        print(f"\nFailed imports ({len(failed)}):")
        for pkg in failed:
            print(f"  - {pkg}")
    else:
        print("\nAll packages imported successfully!")

if __name__ == "__main__":
    main()
    print("\n=== Testing df-nightly specifically ===")
    test_df_import()

```

# test_minimal_stt.py

```py
#!/usr/bin/env python
"""
Test script to verify that the enhanced STT service can be initialized with minimal dependencies.
This script attempts to initialize the STT service with only the essential dependencies.
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import voice_core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from voice_core.stt.enhanced_stt_service import EnhancedSTTService, DIARIZATION_AVAILABLE
    from voice_core.state.voice_state_manager import VoiceStateManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def test_stt_initialization():
    """Test initializing the STT service with minimal dependencies."""
    logger.info("Testing STT service initialization with minimal dependencies")
    
    # Create a dummy state manager
    class DummyStateManager:
        def __init__(self):
            self.processing_lock = DummyLock()
            self.voice_state = None
        
        async def update_voice_state(self, *args, **kwargs):
            pass
            
    class DummyLock:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, *args):
            pass
    
    try:
        # Initialize the STT service
        stt_service = EnhancedSTTService(
            state_manager=DummyStateManager(),
            whisper_model="tiny.en",  # Use the smallest model for testing
            device="cpu",
            fine_tuned_model_path=None,  # Don't use fine-tuned model for testing
            use_fine_tuned_model=False
        )
        
        # Check if optional components are available
        logger.info(f"Diarization available: {DIARIZATION_AVAILABLE}")
        logger.info(f"Silero VAD available: {getattr(stt_service, 'silero_vad_available', False)}")
        
        # Log initialization success
        logger.info("Successfully initialized STT service with minimal dependencies")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize STT service: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test STT service initialization with minimal dependencies")
    args = parser.parse_args()
    
    success = test_stt_initialization()
    sys.exit(0 if success else 1)

```

# transcription_publisher.py

```py
# voice_core/stt/transcription_publisher.py
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional
import livekit.rtc as rtc

logger = logging.getLogger(__name__)

class TranscriptionPublisher:
    """
    Publishes transcriptions to LiveKit with correct speaker identity.
    Ensures transcripts are properly attributed in both data messages and Transcription API.
    """
    
    def __init__(self, state_manager):
        """
        Initialize the transcription publisher.
        
        Args:
            state_manager: Voice state manager instance
        """
        self.state_manager = state_manager
        self.room = None
        self._transcript_sequence = 0
        self._publish_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0
        }
        self.logger = logging.getLogger(__name__)
        
    def set_room(self, room: rtc.Room) -> None:
        """
        Set the LiveKit room for publishing.
        
        Args:
            room: LiveKit room instance
        """
        self.room = room
        
    async def publish_transcript(
        self, 
        text: str, 
        participant_identity: str, 
        is_final: bool = True,
        confidence: float = 1.0
    ) -> bool:
        """
        Publish transcript with correct identity attribution.
        
        Args:
            text: Transcript text
            participant_identity: Participant identity for attribution
            is_final: Whether this is a final transcript
            confidence: Confidence score for the transcript
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
            
        success = True
        self._transcript_sequence += 1
        
        try:
            # 1. Publish via state manager if available
            if self.state_manager:
                try:
                    await self.state_manager.publish_transcription(
                        text,
                        "user",  # Clearly identify sender type
                        is_final,
                        participant_identity=participant_identity
                    )
                    self.logger.debug(f"Published transcript via state manager: '{text[:30]}...'")
                    self._publish_stats["successes"] += 1
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to publish via state manager: {e}")
                    success = False
                    self._publish_stats["failures"] += 1
                
            # 2. Fallback: Direct data channel publish
            if not success and self.room and self.room.local_participant:
                try:
                    # Prepare message
                    message = {
                        "type": "transcript",
                        "text": text,
                        "sender": "user",
                        "participant_identity": participant_identity,
                        "sequence": self._transcript_sequence,
                        "timestamp": time.time(),
                        "is_final": is_final,
                        "confidence": confidence
                    }
                    
                    # Publish with retry
                    await self._publish_with_retry(json.dumps(message).encode(), "transcript")
                    
                    self.logger.debug(f"Published transcript via data channel: '{text[:30]}...'")
                    self._publish_stats["successes"] += 1
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Failed to publish via data channel: {e}")
                    success = False
                    self._publish_stats["failures"] += 1
                
            # 3. Fallback: Transcription API for LiveKit compatibility
            if not success and self.room and self.room.local_participant:
                try:
                    # Find suitable track_sid
                    track_sid = self._find_track_sid(participant_identity)
                    
                    if track_sid:
                        # Create transcription
                        segment_id = str(uuid.uuid4())
                        current_time = int(time.time() * 1000)  # milliseconds
                        
                        trans = rtc.Transcription(
                            participant_identity=participant_identity,
                            track_sid=track_sid,
                            segments=[
                                rtc.TranscriptionSegment(
                                    id=segment_id,
                                    text=text,
                                    start_time=current_time,
                                    end_time=current_time,
                                    final=is_final,
                                    language="en"
                                )
                            ]
                        )
                        
                        await self.room.local_participant.publish_transcription(trans)
                        self.logger.debug(f"Published via Transcription API for '{participant_identity}'")
                        self._publish_stats["successes"] += 1
                        return True
                    else:
                        self.logger.warning(f"No track_sid found for {participant_identity}")
                        self._publish_stats["failures"] += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to publish via Transcription API: {e}")
                    self._publish_stats["failures"] += 1
                    
            return False
                
        except Exception as e:
            self.logger.error(f"Error in publish_transcript: {e}")
            self._publish_stats["failures"] += 1
            return False
            
    def _find_track_sid(self, participant_identity: str) -> Optional[str]:
        """
        Find the audio track SID for a participant.
        
        Args:
            participant_identity: Participant identity to search for
            
        Returns:
            Track SID if found, None otherwise
        """
        if not self.room:
            return None
            
        # Search for track by participant identity
        for participant in self.room.remote_participants.values():
            if participant.identity == participant_identity:
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        return pub.sid
                        
        # If not found, use any audio track as fallback
        for participant in self.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    return pub.sid
                    
        return None
        
    async def _publish_with_retry(self, data: bytes, description: str, max_retries: int = 3) -> bool:
        """
        Publish data with retry logic.
        
        Args:
            data: Data to publish
            description: Description for logging
            max_retries: Maximum retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        if not self.room or not self.room.local_participant:
            return False
            
        self._publish_stats["attempts"] += 1
        
        for attempt in range(max_retries + 1):
            try:
                await self.room.local_participant.publish_data(data, reliable=True)
                
                if attempt > 0:
                    self._publish_stats["retries"] += attempt
                    
                return True
                
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"Failed to publish {description} after {max_retries} attempts: {e}")
                    return False
                    
                self.logger.warning(f"Publish attempt {attempt+1} failed, retrying...")
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
        return False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        success_rate = 0
        if self._publish_stats["attempts"] > 0:
            success_rate = self._publish_stats["successes"] / self._publish_stats["attempts"]
            
        return {
            "transcript_sequence": self._transcript_sequence,
            "publish_attempts": self._publish_stats["attempts"],
            "publish_successes": self._publish_stats["successes"],
            "publish_failures": self._publish_stats["failures"],
            "publish_retries": self._publish_stats["retries"],
            "success_rate": success_rate
        }
```

# vad_engine.py

```py
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
```

