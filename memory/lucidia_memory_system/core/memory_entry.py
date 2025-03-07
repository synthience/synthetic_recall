"""
LUCID RECALL PROJECT
Memory Entry Data Structure

Provides a structured format for storing and managing memories
with significance tracking and serialization capabilities.
"""

import torch
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

class MemoryTypes(Enum):
    """Defines memory categories for different types of stored knowledge."""
    EPISODIC = "episodic"        # Event-based memory (conversations, interactions)
    SEMANTIC = "semantic"        # Fact-based knowledge (definitions, information)
    PROCEDURAL = "procedural"    # Skills & how-to memories
    WORKING = "working"          # Temporary processing memory
    PERSONAL = "personal"        # User-specific details
    IMPORTANT = "important"      # High-priority memories
    EMOTIONAL = "emotional"      # Emotionally tagged memories
    SYSTEM = "system"            # System-related configurations

@dataclass
class MemoryEntry:
    """
    Standardized representation for a single memory unit.
    
    This class ensures consistency across all memory operations.
    """
    content: str                                      # The actual memory content
    memory_type: MemoryTypes = MemoryTypes.EPISODIC   # Type of memory (default: EPISODIC)
    embedding: Optional[torch.Tensor] = None          # Vector representation (if applicable)
    
    id: str = field(default_factory=lambda: f"mem_{int(time.time()*1000)}")  # Unique memory ID
    timestamp: float = field(default_factory=time.time)  # Memory creation time
    significance: float = 0.5  # Importance level (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    access_count: int = 0  # Number of times accessed
    last_access: float = field(default_factory=time.time)  # Last access timestamp
    
    def __post_init__(self):
        """Ensure memory integrity after initialization."""
        # Normalize significance value
        self.significance = max(0.0, min(1.0, self.significance))
        
        # Ensure memory type is valid
        if isinstance(self.memory_type, str):
            try:
                self.memory_type = MemoryTypes[self.memory_type.upper()]
            except KeyError:
                self.memory_type = MemoryTypes.EPISODIC  # Default fallback
        
        # Ensure metadata is always a dictionary
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def record_access(self):
        """Update memory access count and timestamp."""
        self.access_count += 1
        self.last_access = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to a dictionary format for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding.cpu().tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "significance": self.significance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Restore memory from a dictionary format."""
        embedding = data.get("embedding")
        if embedding is not None and not isinstance(embedding, torch.Tensor):
            try:
                embedding = torch.tensor(embedding, dtype=torch.float32)
            except:
                embedding = None
        
        memory_type = MemoryTypes[data.get("memory_type", "EPISODIC").upper()]
        
        return cls(
            id=data.get("id", f"mem_{int(time.time()*1000)}"),
            content=data.get("content", ""),
            memory_type=memory_type,
            embedding=embedding,
            timestamp=data.get("timestamp", time.time()),
            significance=data.get("significance", 0.5),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_access=data.get("last_access", time.time())
        )
    
    def get_effective_significance(self, decay_rate: float = 0.05) -> float:
        """Calculate effective significance considering time decay."""
        age_days = (time.time() - self.timestamp) / 86400  # Convert seconds to days
        
        if age_days < 1:
            return self.significance  # No decay for fresh memories
        
        importance_factor = 0.5 + (0.5 * self.significance)
        access_factor = 1.0 if (time.time() - self.last_access) < (7 * 86400) else 0.5
        access_bonus = min(3.0, 1.0 + (self.access_count / 10))
        
        effective_decay_rate = decay_rate / (importance_factor * access_factor * access_bonus)
        decay_factor = pow(2.718, -effective_decay_rate * (age_days - 1))
        
        return max(0.0, min(1.0, self.significance * decay_factor))