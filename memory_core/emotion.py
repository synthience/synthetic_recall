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
        # Initialize emotions collection if it doesn't exist
        if not hasattr(self, "emotions"):
            self.emotions = {}

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
                    self.emotion_tracking["emotion_history"] = self.emotion_tracking["emotion_history"][-50:]
                        
                return emotion
            else:
                logger.warning("No emotion data in response")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral"
    
    async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
        """
        Detect and analyze emotional context from text.
        This is called by the voice agent to process emotions in transcripts.
        
        Args:
            text: The text to analyze for emotional context
            
        Returns:
            Dict with emotional context information
        """
        try:
            # First detect the primary emotion
            emotion = await self.detect_emotion(text)
            
            # Default emotional data
            timestamp = self._get_timestamp()
            emotional_data = {
                "text": text,
                "emotion": emotion,
                "timestamp": timestamp,
                "sentiment": 0.0,  # Neutral by default
                "emotions": {
                    emotion: 0.7  # Default confidence
                }
            }
            
            # Try to get more detailed emotion analysis from HPC if available
            try:
                connection = await self._get_hpc_connection()
                if connection:
                    # Create detailed emotion analysis request
                    payload = {
                        "type": "emotional_analysis",
                        "text": text
                    }
                    
                    # Send request and get response
                    await connection.send(json.dumps(payload))
                    response = await connection.recv()
                    data = json.loads(response)
                    
                    # Update with more detailed information if available
                    if 'emotions' in data:
                        emotional_data["emotions"] = data['emotions']
                    
                    if 'sentiment' in data:
                        emotional_data["sentiment"] = data['sentiment']
            except Exception as e:
                logger.warning(f"Error getting detailed emotional analysis: {e}")
            
            # Store this emotion in our collection
            self.emotions[str(timestamp)] = emotional_data
            
            # Keep emotions collection at a reasonable size
            if len(self.emotions) > 100:
                # Remove oldest entries
                timestamps = sorted([float(ts) for ts in self.emotions.keys()])
                cutoff = timestamps[len(timestamps) - 100]  # Keep only newest 100
                self.emotions = {ts: data for ts, data in self.emotions.items() 
                                if float(ts) >= cutoff}
            
            # Create a complete emotional context response
            context = {
                "current_emotion": emotion,
                "sentiment": emotional_data.get("sentiment", 0.0),
                "emotions": emotional_data.get("emotions", {}),
                "timestamp": timestamp,
                "text_analyzed": text
            }
            
            logger.info(f"Detected emotional context: {emotion} with sentiment {context['sentiment']:.2f}")
            return context
            
        except Exception as e:
            logger.error(f"Error detecting emotional context: {e}", exc_info=True)
            # Return basic neutral context in case of error
            return {
                "current_emotion": "neutral",
                "sentiment": 0.0,
                "emotions": {"neutral": 1.0},
                "timestamp": self._get_timestamp(),
                "text_analyzed": text,
                "error": str(e)
            }
        
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
    
    async def get_emotional_history(self, limit: int = 5) -> str:
        """
        Get a formatted string of emotional history for RAG context.
        
        Args:
            limit: Number of recent emotions to include
            
        Returns:
            Formatted string of emotional history
        """
        if not hasattr(self, "emotions") or not self.emotions:
            return ""
        
        try:
            # Sort emotions by timestamp (newest first)
            sorted_emotions = sorted(
                self.emotions.items(),
                key=lambda x: float(x[0]),
                reverse=True
            )[:limit]
            
            parts = []
            for timestamp, data in sorted_emotions:
                sentiment = data.get("sentiment", 0)
                emotion = data.get("emotion", "unknown")
                emotions_dict = data.get("emotions", {})
                
                # Format timestamp
                date_str = self._format_timestamp(float(timestamp))
                
                # Describe sentiment
                if sentiment > 0.5:
                    sentiment_desc = "very positive"
                elif sentiment > 0.1:
                    sentiment_desc = "positive"
                elif sentiment > -0.1:
                    sentiment_desc = "neutral"
                elif sentiment > -0.5:
                    sentiment_desc = "negative"
                else:
                    sentiment_desc = "very negative"
                
                # Get top emotions
                emotions_list = []
                if emotions_dict:
                    top_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotions_list = [f"{e}" for e, _ in top_emotions]
                else:
                    emotions_list = [emotion]
                
                emotions_str = ", ".join(emotions_list)
                parts.append(f"• {date_str}: {sentiment_desc} ({emotions_str})")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Error formatting emotional history: {e}")
            return ""
        
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
    
    async def store_emotional_context(self, context: Dict[str, Any]):
        """
        Store emotional context data in the memory system.
        
        Args:
            context: Emotional context dictionary containing emotion data
        """
        if not context:
            logger.warning("Cannot store empty emotional context")
            return
            
        try:
            # Store timestamp if not present
            if "timestamp" not in context:
                context["timestamp"] = self._get_timestamp()
                
            # Store in emotions collection
            timestamp = str(context["timestamp"])
            self.emotions[timestamp] = context
            
            # Update current emotion tracking
            if "current_emotion" in context:
                self.emotion_tracking["current_emotion"] = context["current_emotion"]
                
            # Add to emotion history
            history_entry = {
                "emotion": context.get("current_emotion", "neutral"),
                "timestamp": context["timestamp"],
                "text": context.get("text_analyzed", ""),
                "sentiment": context.get("sentiment", 0.0)
            }
            self.emotion_tracking["emotion_history"].append(history_entry)
            
            # Keep history at a reasonable size
            if len(self.emotion_tracking["emotion_history"]) > 50:
                self.emotion_tracking["emotion_history"] = self.emotion_tracking["emotion_history"][-50:]
                    
            logger.info(f"Stored emotional context: {context.get('current_emotion')} with sentiment {context.get('sentiment', 0.0):.2f}")
            
        except Exception as e:
            logger.error(f"Error storing emotional context: {e}", exc_info=True)
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format a timestamp as a human-readable date string"""
        try:
            import datetime
            return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return f"timestamp: {timestamp}"
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return self._get_current_timestamp() if hasattr(self, "_get_current_timestamp") else time.time()