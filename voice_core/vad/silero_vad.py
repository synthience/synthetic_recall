from __future__ import annotations
import logging
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional
from functools import lru_cache

class SileroVAD:
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self._init_stats()
        self._load_model()

    def _init_stats(self):
        """Initialize statistics for monitoring VAD performance"""
        self.total_calls = 0
        self.speech_detected = 0
        self.avg_confidence = 0.0
        self.last_error = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

    def _load_model(self):
        """Load Silero VAD model with caching and error handling"""
        try:
            self.logger.info(f"Loading Silero VAD model on {self.device}...")
            
            # Check if model is already loaded
            if self.model is not None:
                self.logger.debug("VAD model already loaded")
                return
                
            # Set torch hub directory to ensure proper caching
            torch.hub.set_dir("./.cache/torch/hub")
            
            # Load model with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False,
                        trust_repo=True
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    continue
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Verify model loaded correctly
            if not isinstance(self.model, torch.nn.Module):
                raise RuntimeError("Model loaded but has incorrect type")
            
            # Run test inference
            test_input = torch.zeros(512, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                test_output = self.model(test_input, self.sampling_rate)
            
            if test_output is None or not isinstance(test_output, torch.Tensor):
                raise RuntimeError("Model test inference failed")
            
            self.logger.info("Silero VAD model loaded and verified successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model: {str(e)}")
            self.last_error = str(e)
            raise RuntimeError(f"VAD initialization failed: {str(e)}") from e

    def _preprocess_audio(self, audio_chunk: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess audio chunk for VAD inference"""
        try:
            if audio_chunk.size == 0:
                return None
                
            # Ensure correct dtype and range
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Ensure audio is in [-1, 1] range
            if np.abs(audio_chunk).max() > 1.0:
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # Ensure correct length
            target_length = self._normalize_audio_length(len(audio_chunk))
            if len(audio_chunk) > target_length:
                audio_chunk = audio_chunk[:target_length]
            elif len(audio_chunk) < target_length:
                audio_chunk = np.pad(audio_chunk, (0, target_length - len(audio_chunk)))
            
            # Convert to tensor and move to device
            tensor = torch.from_numpy(audio_chunk).to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {str(e)}")
            return None

    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech in audio chunk using Silero VAD.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        try:
            # Update stats
            self.total_calls += 1
            
            # Preprocess audio
            tensor = self._preprocess_audio(audio_chunk)
            if tensor is None:
                return False, 0.0
            
            # Calculate audio level for debugging
            audio_level = 20 * np.log10(np.abs(audio_chunk).mean() + 1e-10)
            
            # Run inference
            with torch.no_grad():
                confidence = self.model(tensor, self.sampling_rate).item()
            
            # Update running statistics
            is_speech = confidence > self.threshold
            if is_speech:
                self.speech_detected += 1
            self.avg_confidence = (self.avg_confidence * (self.total_calls - 1) + confidence) / self.total_calls
            
            # Log detailed debug info
            self.logger.debug(f"VAD: level={audio_level:.1f}dB conf={confidence:.2f} speech={is_speech}")
            
            # Reset error counter on successful inference
            self.consecutive_errors = 0
            
            return is_speech, confidence
            
        except Exception as e:
            self.consecutive_errors += 1
            self.last_error = str(e)
            
            # Log error with different severity based on consecutive failures
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.logger.error(f"VAD inference failed {self.consecutive_errors} times: {str(e)}")
            else:
                self.logger.warning(f"VAD inference failed: {str(e)}")
            
            # Return conservative estimate
            return False, 0.0

    @lru_cache(maxsize=32)
    def _normalize_audio_length(self, length: int) -> int:
        """Calculate optimal audio length based on sampling rate"""
        return 512 if self.sampling_rate == 16000 else 256

    def reset(self):
        """Reset VAD state and statistics"""
        if self.model:
            self.model.reset_states()
        self._init_stats()

    def __call__(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        return self.is_speech(audio_chunk)