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
        on_transcript: Optional[Callable[[str], Any]] = None,
        nemo_stt: Optional[Any] = None  # Add NemoSTT parameter
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
            nemo_stt: Optional NemoSTT instance for high-quality transcription
        """
        self.state_manager = state_manager
        self.on_transcript = on_transcript
        self.nemo_stt = nemo_stt  # Store NemoSTT instance
        
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
                asyncio.create_task(self.room.local_participant.publish_data(
                    json.dumps({
                        "type": "stt_initialized",
                        "models": {
                            "whisper": self.transcriber.model_name if hasattr(self.transcriber, "model_name") else "unknown"
                        },
                        "device": self.transcriber.device if hasattr(self.transcriber, "device") else "cpu",
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                ))
                
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
                    asyncio.create_task(self.room.local_participant.publish_data(
                        json.dumps({
                            "type": "listening_state",
                            "active": True,
                            "timestamp": time.time()
                        }).encode(),
                        reliable=True
                    ))
                    
                    if self.state_manager.current_state not in [VoiceState.SPEAKING, VoiceState.PROCESSING]:
                        asyncio.create_task(self.state_manager.transition_to(VoiceState.LISTENING))
                        
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
                        
                        # Transcribe the full segment with built-in transcriber
                        transcription_result = await self.transcriber.transcribe(full_audio, self.sample_rate)
                        
                        # Also send to NemoSTT if available for higher quality transcription
                        nemo_task = None
                        if self.nemo_stt:
                            try:
                                self.logger.info("Sending audio to NemoSTT for high-quality transcription")
                                # Create a non-blocking task to process with NemoSTT
                                nemo_task = asyncio.create_task(
                                    self.nemo_stt.transcribe(full_audio)
                                )
                                # We don't await this here - it's handled by callbacks in voice_agent_NEMO
                            except Exception as nemo_e:
                                self.logger.error(f"Error sending to NemoSTT: {nemo_e}", exc_info=True)
                        
                        if transcription_result["success"] and transcription_result["text"]:
                            transcript = transcription_result["text"]
                            
                            # Publish preliminary transcript
                            await self.publisher.publish_transcript(
                                transcript,
                                self._participant_identity,
                                is_final=not bool(self.nemo_stt)  # Mark as preliminary if NemoSTT is processing
                            )
                            
                            # Update stats
                            self.successful_recognitions += 1
                            
                            # Call transcript handler if provided
                            if self.on_transcript:
                                if asyncio.iscoroutinefunction(self.on_transcript):
                                    await self.on_transcript(transcript)
                                else:
                                    self.on_transcript(transcript)
                                    
                            self.logger.info(f"Published {'preliminary' if self.nemo_stt else 'final'} transcript: '{transcript[:50]}...'")
                        
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

    async def set_nemo_stt(self, nemo_stt: Any) -> None:
        """Set the NemoSTT instance after initialization."""
        self.nemo_stt = nemo_stt
        self.logger.info("NemoSTT instance set in EnhancedSTTService")