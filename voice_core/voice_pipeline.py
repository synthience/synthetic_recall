# voice_pipeline.py

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any

import livekit.rtc as rtc
import numpy as np
import torch

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.llm.local_llm_pipeline import LocalLLMPipeline
from voice_core.memory.memory_client import MemoryClient
from voice_core.config.config import LucidiaConfig

class VoicePipeline:
    """
    Integrated voice pipeline with interrupt capabilities and optimized processing.
    Coordinates STT, LLM, and TTS services with state management.
    """

    def __init__(self, room: Optional[rtc.Room] = None):
        """
        Initialize the voice pipeline.
        
        Args:
            room: LiveKit room (optional)
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = LucidiaConfig()

        # The core state manager
        self.state_manager = VoiceStateManager(debug=True)

        # Memory client for RAG
        self.memory_client = None

        # Services
        self.stt_service = EnhancedSTTService(
            state_manager=self.state_manager,
            whisper_model="small",  
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_speech_duration=0.5,
            max_speech_duration=30.0,
            energy_threshold=0.05,
            on_transcript=self._handle_transcript,
            fine_tuned_model_path=self.config.whisper.fine_tuned_model_path,
            use_fine_tuned_model=self.config.whisper.use_fine_tuned_model
        )
        
        self.tts_service = InterruptibleTTSService(
            state_manager=self.state_manager,
            voice="en-US-AvaMultilingualNeural",
            sample_rate=48000,
            num_channels=1,
            on_interrupt=self._handle_tts_interrupt,
            on_complete=self._handle_tts_complete
        )

        # Keep reference to the LiveKit room if available
        self.room = room
        
        # Metrics
        self.metrics = {
            "stt_latency": [],
            "tts_latency": [],
            "total_latency": [],
            "interrupts": 0
        }

        # LLM pipeline
        self.llm_pipeline = None

    async def initialize(self) -> None:
        """
        Initialize pipeline: set the room in the state manager, 
        initialize STT/TTS, set up TTS track, etc.
        """
        try:
            # Initialize memory client
            self.memory_client = MemoryClient()
            await self.memory_client.initialize()
            
            # Initialize LLM pipeline with memory client
            self.llm_pipeline = LocalLLMPipeline(self.config)
            self.llm_pipeline.set_memory_client(self.memory_client)
            
            # Set room in state manager
            if self.room:
                await self.state_manager.set_room(self.room)
                
            # Initialize STT service
            await self.stt_service.initialize()
            
            # Initialize TTS service
            await self.tts_service.initialize()
            
            # Setup event handlers
            self.state_manager.on("interrupt_requested", self._handle_interrupt_request)
            self.state_manager.on("state_change", self._handle_state_change)

            # Start in IDLE state
            await self.state_manager.transition_to(VoiceState.IDLE)
            
            self.logger.info("Voice pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice pipeline: {e}")
            raise

    async def start(self, greeting: Optional[str] = None) -> None:
        """
        Start the voice pipeline and optionally speak a greeting.
        
        Args:
            greeting: Optional greeting to speak when starting
        """
        try:
            # Ensure we're initialized
            if self.state_manager.current_state == VoiceState.IDLE:
                # Speak greeting if provided
                if greeting:
                    await self.say(greeting)
                    
                # Transition to LISTENING
                await self.state_manager.transition_to(VoiceState.LISTENING)
                
                # Publish started event
                if self.room and self.room.local_participant:
                    try:
                        await self.room.local_participant.publish_data(
                            json.dumps({
                                "type": "pipeline_started",
                                "timestamp": time.time()
                            }).encode(),
                            reliable=True
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to publish startup event: {e}")
                        
                self.logger.info("Voice pipeline started and listening")
                
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}", exc_info=True)
            await self.state_manager.register_error(e, "start")

    async def start_listening(self) -> None:
        """Enter listening mode, waiting for speech."""
        await self.state_manager.transition_to(VoiceState.LISTENING)

    async def handle_remote_audio(self, track: rtc.AudioTrack) -> None:
        """
        Process audio from a remote track.
        
        Args:
            track: LiveKit audio track to process
        """
        self.logger.info("VoicePipeline: Starting STT processing for remote track.")
        await self.stt_service.process_audio(track)

    async def _handle_transcript(self, text: str) -> None:
        """
        Handle transcript from STT service.
        
        Args:
            text: Transcribed text
        """
        try:
            self.logger.info(f"Processing transcript: '{text[:50]}...'")
            
            # The state_manager.handle_stt_transcript method will handle
            # state transitions and interruption logic
            
            # You would typically process the transcript through an LLM here
            # and then speak the response
            
            # For this example, we'll just echo back the text
            response = f"I heard you say: {text}"
            await self.say(response)
            
        except Exception as e:
            self.logger.error(f"Error handling transcript: {e}", exc_info=True)
            await self.state_manager.register_error(e, "transcript_handling")

    async def _handle_tts_interrupt(self) -> None:
        """Handle TTS interruption."""
        self.metrics["interrupts"] += 1
        self.logger.info("TTS interrupted")

    async def _handle_tts_complete(self, text: str) -> None:
        """
        Handle TTS completion.
        
        Args:
            text: Text that was spoken
        """
        self.logger.info(f"TTS complete: '{text[:50]}...'")
        
        # Transition to LISTENING is handled by the state_manager

    async def _handle_interrupt_request(self, data: Dict[str, Any]) -> None:
        """
        Handle interrupt request from state manager.
        
        Args:
            data: Interrupt data including text
        """
        try:
            self.logger.info(f"Interrupt requested: {data.get('text', '')[:50]}")
            
            # Stop TTS
            await self.tts_service.stop()
            
        except Exception as e:
            self.logger.error(f"Error handling interrupt request: {e}", exc_info=True)

    async def _handle_state_change(self, data: Dict[str, Any]) -> None:
        """
        Handle state change event.
        
        Args:
            data: State change data
        """
        try:
            old_state = data.get("old_state")
            new_state = data.get("new_state")
            metadata = data.get("metadata", {})
            
            self.logger.info(f"State change: {old_state} -> {new_state} with metadata: {metadata}")
            
        except Exception as e:
            self.logger.error(f"Error handling state change: {e}", exc_info=True)

    async def say(self, text: str) -> bool:
        """
        Synthesize text and speak it. The TTS service integrates with the state manager 
        to handle interruptions.
        
        Args:
            text: Text to speak
            
        Returns:
            True if completed successfully, False if interrupted or error
        """
        try:
            # Create TTS task
            tts_task = asyncio.create_task(self.tts_service.speak(text))
            
            # Register with state manager - this will transition to SPEAKING state
            await self.state_manager.start_speaking(tts_task)
            
            # Wait for completion
            return await tts_task
            
        except asyncio.CancelledError:
            self.logger.info("TTS task cancelled")
            return False
            
        except Exception as e:
            self.logger.error(f"Error speaking text: {e}", exc_info=True)
            await self.state_manager.register_error(e, "tts")
            return False

    async def reset_state(self) -> None:
        """Reset state between conversation turns to prevent issues."""
        try:
            self.logger.debug("Resetting pipeline state for next conversation turn")
            
            # Reset STT buffer if needed
            if hasattr(self.stt_service, 'clear_buffer'):
                await self.stt_service.clear_buffer()
                
            # Ensure transition to LISTENING
            await self.state_manager.transition_to(VoiceState.LISTENING)
            
        except Exception as e:
            self.logger.error(f"Error resetting pipeline state: {e}")
            # Still try to ensure LISTENING state
            await self.state_manager.transition_to(VoiceState.LISTENING)

    async def stop(self) -> None:
        """Shut down pipeline gracefully."""
        self.logger.info("Stopping voice pipeline")
        
        # Reset state before stopping
        await self.reset_state()
        
        # Stop services
        await self.tts_service.stop()
        await self.stt_service.cleanup()
        await self.tts_service.cleanup()
        
        # Clean up state manager
        await self.state_manager.cleanup()

        # Disconnect from room if needed
        if self.room:
            try:
                await self.room.disconnect()
                self.logger.info("Disconnected from LiveKit room")
            except Exception as e:
                self.logger.error(f"Error disconnecting from room: {e}")
                
        self.logger.info("Voice pipeline stopped")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.memory_client:
                await self.memory_client.cleanup()
            if self.llm_pipeline:
                await self.llm_pipeline.cleanup()
            if self.stt_service:
                await self.stt_service.cleanup()
            if self.tts_service:
                await self.tts_service.cleanup()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = {
            "stt_latency_avg": sum(self.metrics["stt_latency"]) / max(len(self.metrics["stt_latency"]), 1),
            "tts_latency_avg": sum(self.metrics["tts_latency"]) / max(len(self.metrics["tts_latency"]), 1),
            "total_latency_avg": sum(self.metrics["total_latency"]) / max(len(self.metrics["total_latency"]), 1),
            "interrupts": self.metrics["interrupts"],
            "state_metrics": self.state_manager.get_analytics()
        }
        
        return metrics