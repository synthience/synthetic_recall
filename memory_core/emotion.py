# memory_core/emotion.py

import logging
import json
import os
import asyncio
import websockets
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Configure emotion analyzer endpoints
EMOTION_ANALYZER_HOST = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
EMOTION_ANALYZER_PORT = os.getenv('EMOTION_ANALYZER_PORT', '5007')
EMOTION_ANALYZER_ENDPOINT = f"ws://{EMOTION_ANALYZER_HOST}:{EMOTION_ANALYZER_PORT}/ws"

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
        
        # Initialize emotion analyzer connection
        self.emotion_analyzer_endpoint = EMOTION_ANALYZER_ENDPOINT
        logger.info(f"Configured emotion analyzer at: {self.emotion_analyzer_endpoint}")

    async def _connect_to_emotion_analyzer(self):
        """
        Establish connection to the emotion analyzer service.
        
        Returns:
            WebSocket connection to emotion analyzer
        """
        try:
            # Make sure we have a valid endpoint
            if not hasattr(self, 'emotion_analyzer_endpoint'):
                # Get endpoint from EnhancedMemoryClient if available
                if hasattr(self, 'emotion_analyzer_endpoint'):
                    endpoint = self.emotion_analyzer_endpoint
                else:
                    # Fallback to environment variable
                    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
                    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
                    endpoint = f"ws://{host}:{port}/ws"
            else:
                endpoint = self.emotion_analyzer_endpoint
                
            # Add timeout to avoid blocking for too long
            return await asyncio.wait_for(
                websockets.connect(endpoint),
                timeout=2.0  # 2 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to emotion analyzer at {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to emotion analyzer: {e}")
            return None

    async def detect_emotion(self, text: str) -> str:
        """
        Detect emotion from text. Uses the emotion analyzer service.
        
        Args:
            text: The text to analyze for emotion
            
        Returns:
            Detected emotion as string
        """
        try:
            # Try emotion analyzer service first
            try:
                connection = await self._connect_to_emotion_analyzer()
                if connection:
                    # Configure endpoint
                    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
                    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
                    endpoint = f"ws://{host}:{port}/ws"
                    
                    logger.info(f"Connecting to emotion analyzer at {endpoint}")
                    
                    # Connect to the WebSocket with a timeout
                    async with asyncio.timeout(5):
                        async with websockets.connect(endpoint) as connection:
                            # Create request payload
                            payload = {
                                "type": "analyze",
                                "text": text
                            }
                            
                            # Send request
                            logger.debug(f"Sending emotion analysis request: {json.dumps(payload)}")
                            await connection.send(json.dumps(payload))
                            
                            # Get response with timeout
                            async with asyncio.timeout(10):  # Longer timeout for processing
                                response = await connection.recv()
                                logger.debug(f"Received response: {response}")
                            
                            # Close connection
                            await connection.close()
                    
                    # Parse response
                    response_data = json.loads(response)
                    
                    # Check for error response
                    if response_data.get("type") == "error":
                        logger.error(f"Emotion analyzer returned error: {response_data.get('message')}")
                        return "neutral"  # Default to neutral on error
                    
                    # Get emotion from dominant emotions
                    if "dominant_detailed" in response_data and "emotion" in response_data["dominant_detailed"]:
                        emotion = response_data["dominant_detailed"]["emotion"]
                    elif "dominant_primary" in response_data and "emotion" in response_data["dominant_primary"]:
                        emotion = response_data["dominant_primary"]["emotion"]
                    else:
                        emotion = "neutral"  # Default if no dominant emotion
                    
                    logger.info(f"Detected emotion: {emotion} for text: {text[:50]}...")
                    
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
                    
            except Exception as e:
                logger.warning(f"Emotion analyzer service failed, falling back to HPC: {e}")
            
            # Fall back to HPC if emotion analyzer fails
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
            # Default emotional data structure
            timestamp = self._get_timestamp()
            emotional_data = {
                "text": text,
                "emotion": "neutral",
                "timestamp": timestamp,
                "sentiment": 0.0,  # Neutral by default
                "emotions": {
                    "neutral": 0.7  # Default confidence
                }
            }
            
            # Try to get detailed emotion analysis from emotion analyzer
            try:
                connection = await self._connect_to_emotion_analyzer()
                if connection:
                    # Configure endpoint
                    host = os.getenv('EMOTION_ANALYZER_HOST', 'localhost')
                    port = os.getenv('EMOTION_ANALYZER_PORT', '5007')
                    endpoint = f"ws://{host}:{port}/ws"
                    
                    logger.info(f"Connecting to emotion analyzer at {endpoint} for emotional context")
                    
                    # Connect to the WebSocket with a timeout
                    async with asyncio.timeout(5):
                        async with websockets.connect(endpoint) as connection:
                            # Create request payload
                            payload = {
                                "type": "analyze",
                                "text": text
                            }
                            
                            # Send request
                            logger.debug(f"Sending emotional context analysis request: {json.dumps(payload)}")
                            await connection.send(json.dumps(payload))
                            
                            # Get response with timeout
                            async with asyncio.timeout(10):  # Longer timeout for processing
                                response = await connection.recv()
                                logger.debug(f"Received emotional context response: {response}")
                            
                            # Close connection
                            await connection.close()
                    
                    # Parse response
                    response_data = json.loads(response)
                    
                    # Check for error response
                    if response_data.get("type") == "error":
                        logger.error(f"Emotion analyzer returned error: {response_data.get('message')}")
                        return self._create_default_emotional_context(text)  # Default on error
                    
                    # Get emotion from dominant emotions
                    if "dominant_detailed" in response_data and "emotion" in response_data["dominant_detailed"]:
                        emotional_state = response_data["dominant_detailed"]["emotion"]
                    elif "dominant_primary" in response_data and "emotion" in response_data["dominant_primary"]:
                        emotional_state = response_data["dominant_primary"]["emotion"]
                    else:
                        emotional_state = "neutral"  # Default if no dominant emotion
                    
                    # Extract detailed emotions data
                    result = {
                        "timestamp": self._get_timestamp(),
                        "text": text,
                        "emotions": response_data.get("detailed_emotions", {"neutral": 1.0}),
                        "sentiment": 0.0,  # Default sentiment as it's not provided
                        "emotional_state": emotional_state
                    }
                    
                    # Calculate sentiment if primary emotions are available
                    if "primary_emotions" in response_data:
                        primary = response_data["primary_emotions"]
                        # Simple sentiment calculation: joy is positive, sadness/anger/fear are negative
                        positive = primary.get("joy", 0) + primary.get("surprise", 0) * 0.5
                        negative = primary.get("sadness", 0) + primary.get("anger", 0) + primary.get("fear", 0)
                        # Calculate sentiment between -1 and 1
                        if positive + negative > 0:  # Avoid division by zero
                            result["sentiment"] = (positive - negative) / (positive + negative)
                    
                    # Update current emotional state
                    self.emotion_tracking["current_emotion"] = emotional_state
                    self.emotion_tracking["emotion_history"].append({
                        "text": text,
                        "emotion": emotional_state,
                        "timestamp": result["timestamp"]
                    })
                    
                    # Keep history at a reasonable size
                    if len(self.emotion_tracking["emotion_history"]) > 50:
                        self.emotion_tracking["emotion_history"] = self.emotion_tracking["emotion_history"][-50:]
                    
                    return result
                    
            except Exception as e:
                logger.warning(f"Error using emotion analyzer, falling back to HPC: {e}")
                
                # If emotion analyzer failed, fall back to HPC
                # First detect the primary emotion
                emotion = await self.detect_emotion(text)
                emotional_data["emotion"] = emotion
                
                # Try to get more detailed analysis from HPC
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
                    logger.warning(f"Error getting detailed emotional analysis from HPC: {e}")
            
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
                "current_emotion": emotional_data["emotion"],
                "sentiment": emotional_data.get("sentiment", 0.0),
                "emotions": emotional_data.get("emotions", {}),
                "timestamp": timestamp,
                "text_analyzed": text
            }
            
            logger.info(f"Detected emotional context: {context['current_emotion']} with sentiment {context['sentiment']:.2f}")
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
    
    def _create_default_emotional_context(self, text: str) -> Dict[str, Any]:
        """Create a default emotional context when analysis fails."""
        timestamp = self._get_timestamp()
        return {
            "timestamp": timestamp,
            "text": text,
            "emotions": {"neutral": 1.0},
            "sentiment": 0.0,
            "emotional_state": "neutral"
        }