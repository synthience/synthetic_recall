"""
Dolphining Framework Integration Utilities

Provides integration utilities to connect the Dolphining STT correction
framework with existing STT services and memory systems.
"""

import asyncio
import logging
import json
import base64
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable

from voice_core.stt.dolphining_stt_corrector import DolphiningSttCorrector
from voice_core.stt.nemo_stt import NemoSTT

logger = logging.getLogger(__name__)


class DolphiningSTTIntegrator:
    """
    Integrates the Dolphining Framework with existing STT systems.
    Provides hooks for NemoSTT and other STT services to use Dolphining for correction.
    """
    
    def __init__(self, 
                 memory_client,
                 domain_dictionary: Optional[Dict[str, float]] = None,
                 confidence_threshold: float = 0.7,
                 max_candidates: int = 5,
                 min_similarity: float = 0.6):
        """
        Initialize the Dolphining STT Integration.
        
        Args:
            memory_client: EnhancedMemoryClient instance
            domain_dictionary: Dictionary of domain-specific terms and their importance
            confidence_threshold: Threshold for automatic correction
            max_candidates: Maximum number of candidate interpretations to consider
            min_similarity: Minimum similarity threshold for fuzzy matches
        """
        self.memory_client = memory_client
        self.corrector = DolphiningSttCorrector(
            memory_client=memory_client,
            domain_dictionary=domain_dictionary,
            confidence_threshold=confidence_threshold,
            max_candidates=max_candidates,
            min_similarity=min_similarity
        )
        
        self.callbacks = {
            "on_correction": [],  # Called when a correction is made
            "on_clarification_needed": [],  # Called when clarification is needed
            "on_emotion_detected": [],  # Called when emotion is detected during processing
            "on_correction_feedback": []  # Called when feedback is provided on a correction
        }
    
    def register_callback(self, event_type: str, callback: Union[Callable[[Dict[str, Any]], None], 
                                                          Callable[[Dict[str, Any]], Awaitable[None]]]):
        """
        Register a callback for a specific event.
        
        Args:
            event_type: Event type (on_correction, on_clarification_needed, etc.)
            callback: Function to call when the event occurs
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def _trigger_callbacks(self, event_type: str, data: Dict[str, Any]):
        """
        Trigger registered callbacks for an event.
        
        Args:
            event_type: Type of event
            data: Data to pass to callbacks
        """
        if event_type not in self.callbacks:
            return
            
        for callback in self.callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def integrate_with_nemo_stt(self, nemo_stt: NemoSTT):
        """
        Integrate Dolphining correction with NemoSTT instance.
        
        Args:
            nemo_stt: NemoSTT instance to integrate with
        """
        # Register callback to process transcription results
        nemo_stt.register_callback("on_transcription", self._handle_nemo_transcription)
        
        logger.info("Integrated Dolphining Framework with NemoSTT")
    
    async def _handle_nemo_transcription(self, transcription_data: Dict[str, Any]):
        """
        Handle transcription results from NemoSTT using the Dolphining Framework.
        This implements the complete Dolphining correction flow with emotion analysis
        and context awareness.
        
        Args:
            transcription_data: Transcription data from NemoSTT
        """
        # Extract transcription text
        stt_text = transcription_data.get("text", "")
        if not stt_text:
            return
            
        # Apply Dolphining correction - this uses all 7 phases of the framework
        correction_result = await self.corrector.correct_transcript(stt_text)
        
        # Store emotional context if detected
        emotion_data = {}
        try:
            # Try to get emotional context for this utterance
            emotion_data = await self.memory_client.detect_emotional_context(stt_text)
            await self._trigger_callbacks("on_emotion_detected", emotion_data)
        except Exception as e:
            logger.warning(f"Error detecting emotion: {e}")
        
        # Update the transcription data with corrected text
        transcription_data["original_text"] = stt_text
        transcription_data["text"] = correction_result["corrected"]
        transcription_data["dolphining_correction"] = correction_result
        transcription_data["emotion_data"] = emotion_data
        
        # Trigger appropriate callbacks
        if correction_result["changed"]:
            correction_data = {
                "original": stt_text,
                "corrected": correction_result["corrected"],
                "confidence": correction_result["confidence"],
                "reasoning": correction_result.get("reasoning", ""),
                "candidates": correction_result.get("candidates", []),
                "emotion": emotion_data.get("current_emotion", "neutral")
            }
            await self._trigger_callbacks("on_correction", correction_data)
            
        if correction_result["needs_clarification"]:
            clarification_data = {
                "original": stt_text,
                "options": correction_result["clarification_options"],
                "emotion": emotion_data.get("current_emotion", "neutral"),
                "request_id": transcription_data.get("request_id", str(id(correction_result)))
            }
            await self._trigger_callbacks("on_clarification_needed", clarification_data)
    
    async def process_websocket_stt(self, audio_bytes: bytes, stt_url: str = "ws://stt_transcription:5002/ws/transcribe") -> Dict[str, Any]:
        """
        Process audio with WebSocket STT service, then apply Dolphining correction.
        
        This applies all phases of the Dolphining Framework to the transcribed audio.
        
        Args:
            audio_bytes: Audio data as bytes
            stt_url: WebSocket URL for STT service
            
        Returns:
            Correction result with original and corrected transcripts
        """
        correction_result = await self.corrector.process_with_websocket_stt(audio_bytes, stt_url)
        
        # Analyze emotion if correction was made
        if correction_result.get("changed", False):
            try:
                # Get emotional context
                emotion_data = await self.memory_client.detect_emotional_context(correction_result["corrected"])
                correction_result["emotion_data"] = emotion_data
                
                # Trigger emotion callback
                await self._trigger_callbacks("on_emotion_detected", emotion_data)
            except Exception as e:
                logger.warning(f"Error detecting emotion: {e}")
        
        # Trigger appropriate callbacks
        if correction_result.get("changed", False):
            correction_data = {
                "original": correction_result["original"],
                "corrected": correction_result["corrected"],
                "confidence": correction_result.get("confidence", 0.0),
                "reasoning": correction_result.get("reasoning", ""),
                "candidates": correction_result.get("candidates", [])
            }
            await self._trigger_callbacks("on_correction", correction_data)
            
        if correction_result.get("needs_clarification", False):
            clarification_data = {
                "original": correction_result["original"],
                "options": correction_result.get("clarification_options", []),
                "request_id": str(id(correction_result))
            }
            await self._trigger_callbacks("on_clarification_needed", clarification_data)
            
        return correction_result
    
    async def provide_correction_feedback(self, original: str, correction: str, accepted: bool):
        """
        Provide feedback on a correction (user accepted or rejected).
        This implements the Iterative Adaptation phase of Dolphining.
        
        Args:
            original: Original transcript text
            correction: Corrected text
            accepted: Whether the correction was accepted by the user
        """
        # Update the corrector with the feedback
        self.corrector.feedback_correction(original, correction, accepted)
        
        # Trigger the feedback callback
        feedback_data = {
            "original": original,
            "correction": correction,
            "accepted": accepted
        }
        await self._trigger_callbacks("on_correction_feedback", feedback_data)
        
        logger.info(f"User feedback for correction '{original}' -> '{correction}': {'accepted' if accepted else 'rejected'}")
    
    def get_domain_dictionary(self) -> Dict[str, float]:
        """Get the current domain dictionary."""
        return self.corrector.domain_dictionary
    
    def update_domain_dictionary(self, new_terms: Dict[str, float]):
        """
        Update the domain dictionary with new terms.
        
        Args:
            new_terms: Dictionary of new domain-specific terms and their importance
        """
        self.corrector.update_domain_dictionary(new_terms)
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on corrections made so far.
        
        Returns:
            Dictionary with detailed correction statistics
        """
        return self.corrector.get_correction_statistics()


# Example integration with NemoSTT
async def example_nemo_integration():
    # Initialize components
    from memory_core.enhanced_memory_client import EnhancedMemoryClient
    from voice_core.stt.nemo_stt import NemoSTT
    
    # Memory client (placeholder)
    memory_client = EnhancedMemoryClient(
        tensor_server_url="ws://tensor_server:5003",
        hpc_server_url="ws://hpc_server:5004",
        session_id="example-session"
    )
    
    # STT engine
    nemo_stt = NemoSTT(config={
        "docker_endpoint": "ws://stt_transcription:5002/ws/transcribe"
    })
    await nemo_stt.initialize()
    
    # Domain terms for Lucidia
    domain_terms = {
        "Lucidia": 0.9,
        "RAG": 0.8,
        "EnhancedMemoryClient": 0.9,
        "Dolphining": 0.8,
        "WebSocket": 0.7
    }
    
    # Initialize Dolphining integrator
    integrator = DolphiningSTTIntegrator(
        memory_client=memory_client,
        domain_dictionary=domain_terms,
        confidence_threshold=0.75
    )
    
    # Register callbacks
    def on_correction(data):
        print(f"Correction made: {data['original']} -> {data['corrected']} (confidence: {data['confidence']:.2f})")
        print(f"Reasoning: {data['reasoning']}")
    
    def on_clarification(data):
        print(f"Clarification needed for: {data['original']}")
        print(f"Options: {data['options']}")
        
        # In a real implementation, this would ask the user for clarification
        # For this example, just simulate choosing the first option
        asyncio.create_task(
            integrator.provide_correction_feedback(
                data['original'], 
                data['options'][0], 
                True  # Simulate acceptance
            )
        )
    
    def on_emotion(data):
        print(f"Emotion detected: {data.get('current_emotion', 'unknown')}")
        print(f"Sentiment: {data.get('sentiment', 0.0):.2f}")
    
    integrator.register_callback("on_correction", on_correction)
    integrator.register_callback("on_clarification_needed", on_clarification)
    integrator.register_callback("on_emotion_detected", on_emotion)
    
    # Integrate with NemoSTT
    integrator.integrate_with_nemo_stt(nemo_stt)
    
    # At this point, any transcription from nemo_stt will be processed by Dolphining
    # and the callbacks will be triggered when corrections are made
    
    # Example: Directly process audio with WebSocket STT
    # (In a real implementation, this would be actual audio)
    # mock_audio_bytes = b'dummy audio data'
    # result = await integrator.process_websocket_stt(mock_audio_bytes)
    # print(f"WebSocket STT processing result: {result}")
    
    # Example: View statistics after some time
    # (In a real implementation, this would be called after processing real transcripts)
    # stats = integrator.get_correction_statistics()
    # print(f"Correction statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(example_nemo_integration())
