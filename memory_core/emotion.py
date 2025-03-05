# memory_core/emotion.py

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class EmotionMixin:
    """
    Mixin that handles emotion detection and tracking in the memory system.
    Allows for detecting and storing emotional context of conversations.
    """

    def __init__(self):
        # Initialize emotion tracking
        self.emotion_tracking = {
            "current_emotion": "neutral",
            "emotion_history": [],
            "emotional_triggers": {}
        }

    async def detect_emotion(self, text: str) -> str:
        """
        Detect emotion from text. Uses the HPC service for emotion analysis.
        
        Args:
            text: The text to analyze for emotion
            
        Returns:
            Detected emotion as string
        """
        try:
            connection = await self._get_hpc_connection()
            if not connection:
                logger.error("Cannot detect emotion: No HPC connection")
                return "neutral"
                
            # Create request payload
            payload = {
                "type": "emotion",
                "text": text
            }
            
            # Send request
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            if 'emotion' in data:
                emotion = data['emotion']
                
                # Update emotion tracking
                self.emotion_tracking["current_emotion"] = emotion
                self.emotion_tracking["emotion_history"].append({
                    "text": text,
                    "emotion": emotion,
                    "timestamp": self._get_timestamp()
                })
                
                # Keep history at a reasonable size
                if len(self.emotion_tracking["emotion_history"]) > 50:
                    self.emotion_tracking["emotion_history"] = \
                        self.emotion_tracking["emotion_history"][-50:]
                        
                return emotion
            else:
                logger.warning("No emotion data in response")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral"
        
    async def get_emotional_context(self, limit: int = 5) -> Dict[str, Any]:
        """
        Get the emotional context of recent conversations.
        
        Args:
            limit: Number of recent emotions to include
            
        Returns:
            Dict with emotional context information
        """
        recent_emotions = self.emotion_tracking["emotion_history"][-limit:] if \
            self.emotion_tracking["emotion_history"] else []
            
        return {
            "current_emotion": self.emotion_tracking["current_emotion"],
            "recent_emotions": recent_emotions,
            "emotional_triggers": self.emotion_tracking["emotional_triggers"]
        }
        
    async def store_emotional_trigger(self, trigger: str, emotion: str):
        """
        Store a trigger for a specific emotion.
        
        Args:
            trigger: The text/concept that triggered the emotion
            emotion: The emotion that was triggered
        """
        if trigger and emotion:
            # Add or update the trigger
            self.emotion_tracking["emotional_triggers"][trigger] = emotion
            logger.info(f"Stored emotional trigger: {trigger} -> {emotion}")
