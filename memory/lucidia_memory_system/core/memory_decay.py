"""
Memory decay management for the Lucidia memory system.

Provides mechanisms for controlled and stable memory decay over time, ensuring
memories age consistently without anomalous resets.
"""

import asyncio
import math
import time
import logging
from typing import Dict, Any, Optional


class StableMemoryDecayManager:
    """Ensures consistent memory decay without reset anomalies."""
    
    def __init__(self, half_life_days=30, min_weight=0.05, max_weight=1.0):
        """Initialize the decay manager with configurable half-life.
        
        Args:
            half_life_days: Number of days for a memory to decay to half strength
            min_weight: Minimum decay weight, preventing complete forgetting
            max_weight: Maximum decay weight cap
        """
        self.decay_rate = math.log(2) / (half_life_days * 24 * 3600)  # Convert to seconds
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.original_timestamps = {}  # Store immutable creation times
        self.importance_modifiers = {}  # Store importance modifiers per memory
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def register_memory(self, memory_id: str, creation_time=None, 
                             initial_importance=0.5):
        """Register the original creation time for a memory.
        
        Args:
            memory_id: Unique identifier for the memory
            creation_time: Original creation timestamp (defaults to now)
            initial_importance: Base importance value (0.0-1.0)
        """
        async with self.lock:
            if memory_id in self.original_timestamps:
                self.logger.debug(f"Memory {memory_id} already registered, preserving timestamp")
                return  # Already registered, preserve original timestamp
                
            timestamp = creation_time or time.time()
            self.original_timestamps[memory_id] = timestamp
            self.importance_modifiers[memory_id] = initial_importance
            self.logger.info(f"Registered memory {memory_id} with timestamp {timestamp} "
                          f"and importance {initial_importance}")
            
    async def update_importance(self, memory_id: str, importance: float):
        """Update the importance modifier for a memory without affecting timestamp.
        
        Args:
            memory_id: The memory identifier
            importance: New importance value (0.0-1.0)
        """
        async with self.lock:
            if memory_id not in self.original_timestamps:
                self.logger.warning(f"Cannot update importance for unregistered memory {memory_id}")
                return False
                
            # Ensure importance is within valid range
            clamped_importance = max(0.0, min(1.0, importance))
            self.importance_modifiers[memory_id] = clamped_importance
            self.logger.debug(f"Updated importance for memory {memory_id} to {clamped_importance}")
            return True
            
    async def record_access(self, memory_id: str, access_strength=0.1):
        """Record an access to a memory, slightly boosting its importance.
        
        Args:
            memory_id: The memory identifier
            access_strength: How much to boost importance (0.0-1.0)
        """
        async with self.lock:
            if memory_id not in self.importance_modifiers:
                self.logger.warning(f"Cannot record access for unregistered memory {memory_id}")
                return False
                
            current_importance = self.importance_modifiers.get(memory_id, 0.5)
            # Apply diminishing returns on importance boost
            boost = access_strength * (1 - current_importance)
            new_importance = current_importance + boost
            self.importance_modifiers[memory_id] = min(self.max_weight, new_importance)
            self.logger.debug(f"Recorded access to memory {memory_id}, "
                            f"importance {current_importance} -> {new_importance}")
            return True
            
    async def calculate_decay_weight(self, memory_id: str, memory: Optional[Dict[str, Any]] = None):
        """Calculate current decay weight without resetting the clock.
        
        Args:
            memory_id: The memory identifier
            memory: Optional memory object with metadata
            
        Returns:
            Current decay weight (0.0-1.0)
        """
        async with self.lock:
            if memory_id not in self.original_timestamps:
                # If not registered, use memory's creation time or current time
                if memory and "creation_time" in memory:
                    await self.register_memory(memory_id, memory["creation_time"])
                else:
                    await self.register_memory(memory_id)
                    
            original_time = self.original_timestamps[memory_id]
            importance = self.importance_modifiers.get(memory_id, 0.5)
            
        # Calculate decay based on original timestamp, never resetting
        time_elapsed = time.time() - original_time
        base_decay_weight = math.exp(-self.decay_rate * time_elapsed)
        
        # Apply importance as a modifier to the decay rate
        # Higher importance = slower decay
        importance_factor = 0.5 + (importance * 0.5)  # Range: 0.5-1.0
        modified_decay = math.pow(base_decay_weight, 2 - importance_factor)
        
        # Apply additional metadata-based modifiers if available
        if memory:
            # Consider access count from metadata
            access_count = memory.get("metadata", {}).get("access_count", 0)
            access_bonus = min(0.3, 0.05 * math.log(access_count + 1))
            
            # Consider emotional salience if available
            emotional_salience = memory.get("metadata", {}).get("emotional_salience", 0.5)
            emotion_bonus = (emotional_salience - 0.5) * 0.2  # -0.1 to +0.1
            
            # Apply bonuses to modified decay
            final_weight = modified_decay + (access_bonus * importance) + emotion_bonus
        else:
            final_weight = modified_decay
            
        # Ensure weight remains within bounds
        final_weight = max(self.min_weight, min(self.max_weight, final_weight))
        
        return final_weight
        
    async def prioritize_memories(self, memory_ids: list):
        """Sort memories by current importance (decay-adjusted).
        
        Args:
            memory_ids: List of memory identifiers
            
        Returns:
            Sorted list of (memory_id, weight) tuples, highest weight first
        """
        weighted_memories = []
        
        for memory_id in memory_ids:
            weight = await self.calculate_decay_weight(memory_id)
            weighted_memories.append((memory_id, weight))
            
        # Sort by weight descending
        return sorted(weighted_memories, key=lambda x: x[1], reverse=True)
        
    async def clean_expired_memories(self, threshold=0.1):
        """Identify memories that have decayed below the threshold.
        
        Args:
            threshold: Weight threshold for considering a memory expired
            
        Returns:
            List of memory IDs that have fallen below the threshold
        """
        expired_memories = []
        
        async with self.lock:
            for memory_id in self.original_timestamps.keys():
                weight = await self.calculate_decay_weight(memory_id)
                if weight <= threshold:
                    expired_memories.append(memory_id)
                    
        self.logger.info(f"Identified {len(expired_memories)} expired memories")
        return expired_memories
