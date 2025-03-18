"""
Emotional Identity Enhancer

Enhances the narrative identity system with emotional analysis,
enriching autobiographical memories with emotional context.
"""

import logging
import asyncio
import json
import websockets
from typing import Dict, Any, List, Optional, Tuple, Union
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class EmotionalIdentityEnhancer:
    """Enhances narrative identity with emotional context.
    
    Connects to the emotion analyzer service to analyze autobiographical memories
    and incorporate emotional context into narrative identity and self-model.
    """
    
    def __init__(self, url: str = "ws://localhost:5007", 
                 self_model = None,
                 emotion_threshold: float = 0.3):
        """Initialize the EmotionalIdentityEnhancer.
        
        Args:
            url: WebSocket URL for the emotion analyzer service
            self_model: SelfModel instance
            emotion_threshold: Confidence threshold for emotions
        """
        self.url = url
        self.self_model = self_model
        self.emotion_threshold = emotion_threshold
        self.websocket = None
        self.connected = False
        self.emotion_cache = {}  # Cache for emotion analysis results
        logger.info(f"Initialized EmotionalIdentityEnhancer, will connect to {url}")
        
    async def connect(self):
        """Connect to the emotion analyzer service."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to emotion analyzer service at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to emotion analyzer: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the emotion analyzer service."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from emotion analyzer service")
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for emotional content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        # Check cache first
        if text in self.emotion_cache:
            return self.emotion_cache[text]
            
        if not self.connected:
            success = await self.connect()
            if not success:
                logger.error("Cannot analyze text: not connected to emotion analyzer")
                return {}
                
        try:
            # Send analysis request
            await self.websocket.send(json.dumps({
                "type": "analyze",
                "text": text,
                "threshold": self.emotion_threshold
            }))
            
            # Get response
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "error":
                logger.error(f"Error analyzing text: {result.get('message')}")
                return {}
            
            # Cache result
            self.emotion_cache[text] = result
            return result
        except Exception as e:
            logger.error(f"Error in analyze_text: {e}")
            # Try to reconnect
            self.connected = False
            return {}
    
    async def enhance_autobiographical_memory(self, memory_id: str, 
                                          text: Optional[str] = None) -> bool:
        """Enhance an autobiographical memory with emotional context.
        
        Args:
            memory_id: ID of the memory to enhance
            text: Text to analyze. If None, uses memory content
            
        Returns:
            True if successful, False otherwise
        """
        if not self.self_model or not hasattr(self.self_model, "get_autobiographical_memory"):
            logger.error("No self_model provided or missing autobiographical memory methods")
            return False
            
        # Get memory
        memory = self.self_model.get_autobiographical_memory(memory_id)
        if not memory:
            logger.error(f"Memory {memory_id} not found")
            return False
            
        # If no text provided, use memory content
        if not text:
            memory_content = memory.get("content", "")
            if isinstance(memory_content, dict):
                # Extract text from memory content
                experience = memory_content.get("experience", "")
                reflection = memory_content.get("reflection", "")
                text = f"{experience}. {reflection}".strip()
            else:
                text = str(memory_content)
            
        if not text:
            logger.warning(f"No text available for memory {memory_id}")
            return False
            
        # Analyze text
        emotion_data = await self.analyze_text(text)
        if not emotion_data:
            return False
            
        # Extract emotion data
        primary_emotions = emotion_data.get("primary_emotions", {})
        dominant_emotion = emotion_data.get("dominant_primary", {}).get("emotion")
        dominant_confidence = emotion_data.get("dominant_primary", {}).get("confidence")
        detailed_emotions = emotion_data.get("detailed_emotions", {})
        
        # Prepare emotion metadata
        emotion_metadata = {
            "primary_emotions": primary_emotions,
            "dominant_emotion": dominant_emotion,
            "confidence": dominant_confidence,
            "detailed_emotions": detailed_emotions
        }
        
        # Update memory metadata
        metadata = memory.get("metadata", {})
        metadata["emotions"] = emotion_metadata
        
        # Update memory
        updated_memory = memory.copy()
        updated_memory["metadata"] = metadata
        
        # Save updated memory
        return self.self_model.update_autobiographical_memory(memory_id, updated_memory)
    
    async def enhance_all_autobiographical_memories(self) -> Dict[str, Any]:
        """Enhance all autobiographical memories with emotional context.
        
        Returns:
            Dictionary with enhancement statistics
        """
        if not self.self_model or not hasattr(self.self_model, "get_all_autobiographical_memories"):
            logger.error("No self_model provided or missing autobiographical memory methods")
            return {"success": False, "error": "No self_model provided or missing methods"}
            
        # Get all memories
        memories = self.self_model.get_all_autobiographical_memories()
        
        stats = {
            "memories_enhanced": 0,
            "failed_memories": 0,
            "memory_details": {}
        }
        
        for memory in memories:
            memory_id = memory.get("id")
            success = await self.enhance_autobiographical_memory(memory_id)
            
            if success:
                stats["memories_enhanced"] += 1
                stats["memory_details"][memory_id] = "success"
            else:
                stats["failed_memories"] += 1
                stats["memory_details"][memory_id] = "failed"
                
        stats["success"] = stats["memories_enhanced"] > 0
        return stats
    
    async def record_emotional_experience(self, experience: str, 
                                      reflection: Optional[str] = None) -> Dict[str, Any]:
        """Record an experience with emotional analysis.
        
        Args:
            experience: Experience description
            reflection: Optional reflection on the experience
            
        Returns:
            Dictionary with recorded memory and emotional analysis
        """
        if not self.self_model or not hasattr(self.self_model, "record_autobiographical_memory"):
            logger.error("No self_model provided or missing record_autobiographical_memory method")
            return {"success": False, "error": "No self_model provided or missing methods"}
            
        # Analyze experience text
        text = experience
        if reflection:
            text = f"{experience}. {reflection}"
            
        emotion_data = await self.analyze_text(text)
        if not emotion_data:
            logger.error("Failed to analyze emotions for experience")
            # Still proceed with recording, but without emotional metadata
        
        # Prepare memory content
        memory_content = {
            "experience": experience
        }
        
        if reflection:
            memory_content["reflection"] = reflection
            
        # Prepare metadata
        metadata = {
            "recorded_at": datetime.now().isoformat(),
            "source": "user_input"
        }
        
        if emotion_data:
            # Extract emotion data
            primary_emotions = emotion_data.get("primary_emotions", {})
            dominant_emotion = emotion_data.get("dominant_primary", {}).get("emotion")
            dominant_confidence = emotion_data.get("dominant_primary", {}).get("confidence")
            detailed_emotions = emotion_data.get("detailed_emotions", {})
            
            # Add to metadata
            metadata["emotions"] = {
                "primary_emotions": primary_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": dominant_confidence,
                "detailed_emotions": detailed_emotions
            }
            
            # Set emotional valence based on dominant emotion
            valence_mapping = {
                "joy": "positive",
                "sadness": "negative",
                "anger": "negative",
                "fear": "negative",
                "surprise": "neutral",
                "neutral": "neutral",
                "other": "neutral"
            }
            
            metadata["emotional_valence"] = valence_mapping.get(dominant_emotion, "neutral")
            
        # Record memory
        memory = self.self_model.record_autobiographical_memory(
            content=memory_content,
            metadata=metadata
        )
        
        return {
            "success": True,
            "memory": memory,
            "emotion_analysis": emotion_data
        }
    
    async def generate_emotional_identity_narrative(self, 
                                              narrative_type: str = "general",
                                              emotion_filter: Optional[str] = None) -> Dict[str, Any]:
        """Generate a narrative based on emotional experiences.
        
        Args:
            narrative_type: Type of narrative to generate
            emotion_filter: Optional emotion to filter by
            
        Returns:
            Dictionary with narrative and emotional analysis
        """
        if not self.self_model or not hasattr(self.self_model, "get_all_autobiographical_memories"):
            logger.error("No self_model provided or missing autobiographical memory methods")
            return {"success": False, "error": "No self_model provided or missing methods"}
            
        # Get all memories
        all_memories = self.self_model.get_all_autobiographical_memories()
        
        # Filter by emotion if specified
        memories = []
        for memory in all_memories:
            if not emotion_filter:
                memories.append(memory)
                continue
                
            metadata = memory.get("metadata", {})
            emotions = metadata.get("emotions", {})
            
            if not emotions:
                # If no emotional data, enhance memory first
                memory_id = memory.get("id")
                await self.enhance_autobiographical_memory(memory_id)
                
                # Get updated memory
                updated_memory = self.self_model.get_autobiographical_memory(memory_id)
                if updated_memory:
                    metadata = updated_memory.get("metadata", {})
                    emotions = metadata.get("emotions", {})
            
            dominant_emotion = emotions.get("dominant_emotion")
            if dominant_emotion == emotion_filter:
                memories.append(memory)
        
        if not memories:
            return {
                "success": False,
                "error": f"No memories found with emotion {emotion_filter}" if emotion_filter else "No memories found"
            }
        
        # Generate narrative
        if hasattr(self.self_model, "generate_identity_narrative"):
            narrative = self.self_model.generate_identity_narrative(
                narrative_type=narrative_type,
                memories=memories
            )
            
            # Analyze narrative for emotional content
            if narrative:
                emotion_data = await self.analyze_text(narrative)
                
                return {
                    "success": True,
                    "narrative": narrative,
                    "emotion_analysis": emotion_data,
                    "memory_count": len(memories)
                }
        
        return {"success": False, "error": "Failed to generate narrative"}
    
    async def update_emotional_state(self) -> Dict[str, Any]:
        """Update the self-model's emotional state based on recent memories.
        
        Returns:
            Dictionary with updated emotional state
        """
        if not self.self_model:
            logger.error("No self_model provided")
            return {"success": False, "error": "No self_model provided"}
            
        # Get recent memories (last 10)
        recent_memories = []
        if hasattr(self.self_model, "get_all_autobiographical_memories"):
            all_memories = self.self_model.get_all_autobiographical_memories()
            
            # Sort by timestamp (newest first)
            all_memories.sort(key=lambda x: x.get("metadata", {}).get("recorded_at", ""), reverse=True)
            
            # Take most recent 10
            recent_memories = all_memories[:10]
        
        if not recent_memories:
            logger.warning("No recent memories found for emotional state update")
            return {"success": False, "error": "No recent memories found"}
            
        # Aggregate emotions across memories
        emotion_counts = {
            "joy": 0,
            "sadness": 0,
            "anger": 0,
            "fear": 0,
            "surprise": 0,
            "neutral": 0,
            "other": 0
        }
        
        emotion_scores = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
            "other": 0.0
        }
        
        # Process each memory
        for memory in recent_memories:
            metadata = memory.get("metadata", {})
            emotions = metadata.get("emotions", {})
            
            if not emotions:
                continue
                
            dominant_emotion = emotions.get("dominant_emotion")
            if dominant_emotion in emotion_counts:
                emotion_counts[dominant_emotion] += 1
                
            # Add primary emotion scores
            primary_emotions = emotions.get("primary_emotions", {})
            for emotion, score in primary_emotions.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += score
        
        # Normalize emotion scores
        total_memories = len(recent_memories)
        if total_memories > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_memories
                
        # Update self-model emotional state
        if hasattr(self.self_model, "update_aspect"):
            # Convert to percentages for better readability
            emotional_state = {emotion: round(score * 100) for emotion, score in emotion_scores.items()}
            
            # Keep only significant emotions (>5%)
            emotional_state = {emotion: score for emotion, score in emotional_state.items() if score > 5}
            
            self.self_model.update_aspect("emotional_state", emotional_state)
            
            return {
                "success": True,
                "emotional_state": emotional_state,
                "based_on_memories": total_memories
            }
        
        return {"success": False, "error": "Self-model does not support updating emotional state"}
