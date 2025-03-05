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