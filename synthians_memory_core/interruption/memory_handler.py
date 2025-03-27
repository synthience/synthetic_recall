# synthians_memory_core/interruption/memory_handler.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
import json
import aiohttp
import numpy as np

class InterruptionAwareMemoryHandler:
    """
    Specialized handler for transcripts that enriches memory entries with interruption metadata.
    This bridges the voice system's interruption tracking with the memory system.
    """

    def __init__(self, 
                 api_url: str = "http://localhost:8000"):
        """
        Initialize the memory handler with API connection details.
        
        Args:
            api_url: Base URL for the memory API
        """
        self.logger = logging.getLogger("InterruptionAwareMemoryHandler")
        self.api_url = api_url.rstrip('/')
        
    async def __call__(self, 
                       text: str, 
                       transcript_sequence: int = 0,
                       timestamp: float = 0,
                       confidence: float = 1.0,
                       **metadata) -> Dict[str, Any]:
        """
        Process a transcript, enriching it with interruption metadata, and send to memory API.
        This method accepts transcripts and additional metadata from voice processing.
        
        Args:
            text: The transcript text to process
            transcript_sequence: Sequence number of this transcript
            timestamp: Unix timestamp when transcript was received
            confidence: STT confidence score
            **metadata: Additional metadata, including interruption data
            
        Returns:
            Response from the memory API as a dictionary
        """
        try:
            self.logger.info(f"Processing transcript {transcript_sequence}: {text[:50]}...")
            
            # Prepare audio metadata from transcript info
            audio_metadata = {
                "timestamp": timestamp,
                "confidence": confidence,
                "sequence": transcript_sequence,
                "source": "voice_interaction"
            }
            
            # Add interruption metadata if available
            if "was_interrupted" in metadata:
                audio_metadata["was_interrupted"] = metadata["was_interrupted"]
                audio_metadata["user_interruptions"] = metadata.get("user_interruptions", 1)
                
                if "interruption_timestamps" in metadata:
                    audio_metadata["interruption_timestamps"] = metadata["interruption_timestamps"]
                    
                if "session_id" in metadata:
                    audio_metadata["session_id"] = metadata["session_id"]
            
            # Prepare request to memory API
            request_data = {
                "text": text,
                "audio_metadata": audio_metadata
            }
            
            # Use the new transcription feature extraction endpoint
            async with aiohttp.ClientSession() as session:
                self.logger.info(f"Sending transcript to memory API: {self.api_url}/process_transcription")
                async with session.post(
                    f"{self.api_url}/process_transcription", 
                    json=request_data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Memory created/updated with ID: {result.get('memory_id')}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Memory API error: {response.status} - {error_text}")
                        return {"success": False, "error": error_text}
                        
        except Exception as e:
            self.logger.error(f"Error processing transcript: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _validate_embedding(self, embedding):
        """
        Validate that an embedding is properly formed without NaN or Inf values.
        Implements the same validation logic as in memory_core/tools.py.
        
        Args:
            embedding: The embedding vector to validate (np.ndarray or list)
            
        Returns:
            bool: True if the embedding is valid, False otherwise
        """
        if embedding is None:
            return False
            
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False
            
        return True

    @staticmethod
    def get_reflection_prompt(interruption_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a reflection prompt based on interruption patterns to help guide memory retrieval.
        
        Args:
            interruption_data: Dictionary containing interruption metadata
            
        Returns:
            Optional reflection prompt string or None if no reflection needed
        """
        was_interrupted = interruption_data.get("was_interrupted", False)
        interruption_count = interruption_data.get("user_interruptions", 0)
        
        # No reflection needed for normal conversation flow
        if not was_interrupted and interruption_count == 0:
            return None
            
        # Generate prompts based on interruption patterns
        if was_interrupted:
            if interruption_count > 5:
                return "You seem to be interrupting frequently. Would you like me to pause more often to let you speak?"
            else:
                return "I noticed you interrupted. Was there something specific you wanted to address?"
        
        # General high interruption pattern but not this specific utterance
        if interruption_count > 3:
            return "I've noticed several interruptions in our conversation. Would you prefer if I spoke in shorter segments?"
            
        return None
