"""
LUCID RECALL PROJECT
Memory Types

Defines memory categories and data structures for the memory system.
"""

import torch
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List

class MemoryTypes(Enum):
    """Types of memories that can be stored in the system."""
    
    EPISODIC = "episodic"        # Event/experience memories (conversations, interactions)
    SEMANTIC = "semantic"        # Factual/conceptual memories (knowledge, facts)
    PROCEDURAL = "procedural"    # Skill/procedure memories (how to do things)
    WORKING = "working"          # Temporary processing memories (short-term)
    PERSONAL = "personal"        # Personal information about users
    IMPORTANT = "important"      # High-significance memories that should be preserved
    EMOTIONAL = "emotional"      # Memories with emotional context
    SYSTEM = "system"            # System-level memories (configs, settings)

@dataclass
class MemoryEntry:
    """
    Standardized container for a single memory entry.
    
    This structure ensures consistent memory representation across
    all components of the memory system.
    """
    
    # Core memory data
    content: str                                        # The actual memory content (text)
    memory_type: MemoryTypes = MemoryTypes.EPISODIC     # Type of memory
    embedding: Optional[torch.Tensor] = None            # Vector representation of content
    
    # Metadata
    id: str = field(default_factory=lambda: f"mem_{int(time.time()*1000)}")  # Unique identifier
    timestamp: float = field(default_factory=time.time)  # Creation time
    significance: float = 0.5                           # Importance score (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    # Usage tracking
    access_count: int = 0                               # Number of times accessed
    last_access: float = field(default_factory=time.time)  # Last access timestamp
    
    def __post_init__(self):
        """Validate memory entry after initialization."""
        # Ensure significance is within valid range
        self.significance = max(0.0, min(1.0, self.significance))
        
        # Ensure proper memory type
        if isinstance(self.memory_type, str):
            try:
                self.memory_type = MemoryTypes[self.memory_type.upper()]
            except KeyError:
                # Try to find by value
                for mem_type in MemoryTypes:
                    if mem_type.value == self.memory_type.lower():
                        self.memory_type = mem_type
                        break
                else:
                    # Default to EPISODIC if not found
                    self.memory_type = MemoryTypes.EPISODIC
                        
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
    
    def record_access(self) -> None:
        """Record memory access, updating tracking information."""
        self.access_count += 1
        self.last_access = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        # Convert embedding to list if present
        embedding_data = None
        if self.embedding is not None:
            if isinstance(self.embedding, torch.Tensor):
                embedding_data = self.embedding.cpu().tolist()
            elif isinstance(self.embedding, list):
                embedding_data = self.embedding
            else:
                # Try to convert to list
                try:
                    embedding_data = list(self.embedding)
                except:
                    embedding_data = None
        
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": embedding_data,
            "timestamp": self.timestamp,
            "significance": self.significance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary representation."""
        # Handle embedding conversion
        embedding = data.get("embedding")
        if embedding is not None and not isinstance(embedding, torch.Tensor):
            try:
                embedding = torch.tensor(embedding, dtype=torch.float32)
            except:
                embedding = None
        
        # Extract memory type
        memory_type_str = data.get("memory_type", "EPISODIC")
        memory_type = None
        
        # Try to convert string to MemoryTypes enum
        for mem_type in MemoryTypes:
            if mem_type.value == memory_type_str.lower() or mem_type.name == memory_type_str.upper():
                memory_type = mem_type
                break
                
        # Use default if not found
        if memory_type is None:
            memory_type = MemoryTypes.EPISODIC
            
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
        """
        Calculate effective significance with time decay applied.
        
        Args:
            decay_rate: Rate of significance decay per day
            
        Returns:
            Effective significance after decay
        """
        # Get current time
        current_time = time.time()
        
        # Calculate age in days
        age_days = (current_time - self.timestamp) / 86400  # 86400 seconds per day
        
        # Skip recent memories (less than 1 day old)
        if age_days < 1:
            return self.significance
        
        # Calculate importance factor (more important memories decay slower)
        importance_factor = 0.5 + (0.5 * self.significance)
        
        # Calculate usage factor (more used memories decay slower)
        access_recency_days = (current_time - self.last_access) / 86400
        access_factor = 1.0 if access_recency_days < 7 else 0.5  # Boost for recently accessed
        
        # Apply access count bonus (capped at 3x)
        access_bonus = min(3.0, 1.0 + (self.access_count / 10))
        
        # Calculate effective decay rate (decay slower for important, frequently accessed memories)
        effective_decay_rate = decay_rate / (importance_factor * access_factor * access_bonus)
        
        # Apply exponential decay
        decay_factor = pow(2.718, -effective_decay_rate * (age_days - 1))  # e^(-rate*days)
        effective_significance = self.significance * decay_factor
        
        return max(0.0, min(1.0, effective_significance))  # Ensure within range