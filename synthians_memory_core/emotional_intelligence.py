# synthians_memory_core/emotional_intelligence.py

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from .custom_logger import logger # Use the shared custom logger
from .emotion_analyzer import EmotionAnalyzer as _EmotionAnalyzer  # Import with alias to avoid name conflicts

# Maintain backward compatibility by re-exporting the class
# This prevents import errors in existing code that imports from this module
class EmotionalAnalyzer(_EmotionAnalyzer):
    """Re-export of the EmotionAnalyzer class from emotion_analyzer.py for backward compatibility."""
    pass

# Export EmotionalAnalyzer for backward compatibility
__all__ = ['EmotionalAnalyzer', 'EmotionalGatingService']

# NOTE: The EmotionalAnalyzer class implementation has been moved to emotion_analyzer.py
# This file now only contains the EmotionalGatingService class and a compatibility wrapper

class EmotionalGatingService:
    """Applies emotional gating to memory retrieval."""
    def __init__(self, emotion_analyzer, config: Optional[Dict] = None):
        """Initialize the emotional gating service.
        
        Args:
            emotion_analyzer: An instance of EmotionAnalyzer from emotion_analyzer.py
            config: Configuration parameters for the gating service
        """
        self.emotion_analyzer = emotion_analyzer
        self.config = config or {}
        
        # Configuration with defaults
        self.emotion_weight = self.config.get('emotional_weight', 0.3)
        self.memory_gate_min_factor = self.config.get('gate_min_factor', 0.5)
        self.cognitive_bias = self.config.get('cognitive_bias', 0.2)
        
        logger.info("EmotionalGatingService", "Initialized with config", {
            "emotion_weight": self.emotion_weight,
            "gate_min_factor": self.memory_gate_min_factor,
            "cognitive_bias": self.cognitive_bias,
            "has_analyzer": self.emotion_analyzer is not None
        })

        # Simplified compatibility - similar emotions are compatible
        self.emotion_compatibility = {
            "joy": {"joy", "excitement", "gratitude", "satisfaction", "content"},
            "sadness": {"sadness", "grief", "disappointment", "melancholy"},
            "anger": {"anger", "frustration", "irritation"},
            "fear": {"fear", "anxiety", "nervousness"},
            "surprise": {"surprise", "amazement", "astonishment"},
            "disgust": {"disgust", "displeasure"},
            "trust": {"trust", "respect", "admiration"},
            "neutral": {"neutral", "calm", "focused"}
        }
        # Add reverse compatibility and self-compatibility
        for emotion, compatible_set in list(self.emotion_compatibility.items()):
             compatible_set.add(emotion) # Self-compatible
             for compatible_emotion in compatible_set:
                  if compatible_emotion not in self.emotion_compatibility:
                       self.emotion_compatibility[compatible_emotion] = set()
                  self.emotion_compatibility[compatible_emotion].add(emotion)
        # Ensure neutral is compatible with everything
        all_emotions = set(self.emotion_compatibility.keys())
        self.emotion_compatibility["neutral"] = all_emotions
        for emotion in all_emotions:
             self.emotion_compatibility[emotion].add("neutral")

    async def gate_memories(self,
                           memories: List[Dict[str, Any]],
                           user_emotion: Optional[Dict[str, Any]],
                           cognitive_load: float = 0.5) -> List[Dict[str, Any]]:
        """Filter and re-rank memories based on emotional context."""
        if not memories or user_emotion is None:
            return memories # No gating if no user emotion provided

        user_dominant = user_emotion.get("dominant_emotion", "neutral")
        user_valence = user_emotion.get("sentiment_value", 0.0)
        user_intensity = user_emotion.get("intensity", 0.0)

        gated_memories = []
        for memory in memories:
            mem_emotion_context = memory.get("metadata", {}).get("emotional_context")
            if not mem_emotion_context:
                 # If no emotion data, assign neutral resonance
                 memory["emotional_resonance"] = 0.5
                 gated_memories.append(memory)
                 continue

            memory_dominant = mem_emotion_context.get("dominant_emotion", "neutral")
            memory_valence = mem_emotion_context.get("sentiment_value", 0.0)
            memory_intensity = mem_emotion_context.get("intensity", 0.0)

            # 1. Cognitive Defense (Simplified)
            if self.config.get('cognitive_defense_enabled', True) and user_valence < -0.5 and user_intensity > 0.6:
                 # If user is highly negative, filter out extremely negative memories
                 if memory_valence < -0.7 and memory_intensity > 0.8:
                      logger.debug("EmotionalGatingService", "Cognitive defense filtered out negative memory", {"memory_id": memory.get("id")})
                      continue # Skip this memory

            # 2. Calculate Emotional Resonance
            # Compatibility score (1 if compatible, 0 if not, 0.5 if neutral involved)
            user_compatibles = self.emotion_compatibility.get(user_dominant, set())
            mem_compatibles = self.emotion_compatibility.get(memory_dominant, set())

            if user_dominant == "neutral" or memory_dominant == "neutral":
                 emotion_compatibility = 0.7 # Neutral is somewhat compatible with everything
            elif memory_dominant in user_compatibles:
                 emotion_compatibility = 1.0 # Direct or similar emotion
            elif user_compatibles.intersection(mem_compatibles):
                 emotion_compatibility = 0.6 # Related emotions
            else:
                 emotion_compatibility = 0.1 # Unrelated emotions

            # Valence alignment (1 for same sign, 0 for opposite, 0.5 if one is neutral)
            if (user_valence > 0.1 and memory_valence > 0.1) or \
               (user_valence < -0.1 and memory_valence < -0.1):
                 valence_alignment = 1.0
            elif (user_valence > 0.1 and memory_valence < -0.1) or \
                 (user_valence < -0.1 and memory_valence > 0.1):
                 valence_alignment = 0.0
            else: # One or both are neutral
                 valence_alignment = 0.5

            # Combined resonance score
            emotional_resonance = (emotion_compatibility * 0.6 + valence_alignment * 0.4)
            memory["emotional_resonance"] = emotional_resonance

            # 3. Cognitive Load Adjustment (Simplified)
            # Higher load makes less resonant memories less likely
            if cognitive_load > 0.7 and emotional_resonance < (0.4 + 0.4 * cognitive_load):
                logger.debug("EmotionalGatingService", f"Memory filtered by high cognitive load ({cognitive_load})", {"memory_id": memory.get("id")})
                continue # Skip less resonant memories under high load

            gated_memories.append(memory)

        # 4. Re-rank based on combined score
        weight = self.emotion_weight
        for memory in gated_memories:
             original_score = memory.get("relevance_score", 0.0) # Use relevance_score if available
             resonance = memory.get("emotional_resonance", 0.5)
             memory["final_score"] = (1 - weight) * original_score + weight * resonance

        gated_memories.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info("EmotionalGatingService", f"Gated memories from {len(memories)} to {len(gated_memories)}", {"user_emotion": user_dominant})
        return gated_memories
