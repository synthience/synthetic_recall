# memory/lucidia_memory_system/core/memory_entry.py
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List

class MemoryTypes(Enum):
    """Enumeration of memory types for categorization."""
    GENERAL = "general"
    CONVERSATION = "conversation"
    INSIGHT = "insight"
    EXPERIENCE = "experience"
    REFLECTION = "reflection"
    DREAM = "dream"
    FACTUAL = "factual"
    EMOTIONAL = "emotional"

class MemoryEntry:
    """
    Memory entry containing text content and metadata.
    
    This class represents a discrete memory unit that can be stored,
    retrieved, and processed by the memory system.
    """
    
    def __init__(
        self,
        content: str,
        memory_type: str = "general",
        significance: float = 0.5,
        id: Optional[str] = None,
        created_at: Optional[float] = None,
        last_accessed: Optional[float] = None,
        access_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a memory entry.
        
        Args:
            content: Text content of the memory
            memory_type: Type of memory
            significance: Significance score (0-1)
            id: Optional unique identifier (generated if not provided)
            created_at: Optional creation timestamp (current time if not provided)
            last_accessed: Optional last access timestamp
            access_count: Number of times accessed
            metadata: Optional additional metadata
            embedding: Optional vector embedding
        """
        self.content = content
        self.memory_type = memory_type
        self.significance = significance
        self.id = id or f"memory_{str(uuid.uuid4())[:8]}"
        self.created_at = created_at or time.time()
        self.last_access = last_accessed or self.created_at
        self.access_count = access_count
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def record_access(self) -> None:
        """Record an access to this memory."""
        self.access_count += 1
        self.last_access = time.time()
    
    def update_significance(self, new_value: float) -> None:
        """
        Update the significance value of the memory.
        
        Args:
            new_value: New significance value (0-1)
        """
        self.significance = max(0.0, min(1.0, new_value))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the memory entry to a dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "significance": self.significance,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """
        Create a memory entry from a dictionary.
        
        Args:
            data: Dictionary with memory data
            
        Returns:
            New MemoryEntry instance
        """
        return cls(
            content=data.get("content", ""),
            memory_type=data.get("memory_type", "general"),
            significance=data.get("significance", 0.5),
            id=data.get("id"),
            created_at=data.get("created_at"),
            last_accessed=data.get("last_access"),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding")
        )
    
    def __str__(self) -> str:
        """String representation of the memory entry."""
        return f"Memory({self.id}): {self.content[:50]}... [sig={self.significance:.2f}]"