# memory/lucidia_memory_system/core/memory_entry.py
import time
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List
import torch
import base64

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
        # Handle embedding tensor - convert to list or encode as base64 if bytes
        embedding_serialized = None
        if self.embedding is not None:
            if isinstance(self.embedding, torch.Tensor):
                embedding_serialized = self.embedding.cpu().tolist()
            elif isinstance(self.embedding, (list, tuple)):
                embedding_serialized = list(self.embedding)
            elif isinstance(self.embedding, bytes):
                import base64
                embedding_serialized = {
                    "format": "base64",
                    "data": base64.b64encode(self.embedding).decode('ascii')
                }
            elif isinstance(self.embedding, str):
                embedding_serialized = self.embedding
            else:
                # Try to convert to string as fallback
                try:
                    embedding_serialized = str(self.embedding)
                except:
                    embedding_serialized = "[Unserializable embedding]"
        
        # Process content - ensure it's string
        content_str = self.content
        if isinstance(content_str, bytes):
            try:
                content_str = content_str.decode('utf-8')
            except UnicodeDecodeError:
                import base64
                content_str = f"[BASE64_ENCODED_DATA:{base64.b64encode(content_str).decode('ascii')}]"
        
        # Process metadata - ensure all values are serializable
        metadata_serialized = {}
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, bytes):
                    import base64
                    metadata_serialized[key] = {
                        "format": "base64",
                        "data": base64.b64encode(value).decode('ascii')
                    }
                elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    metadata_serialized[key] = value
                else:
                    # Try to convert to string
                    try:
                        metadata_serialized[key] = str(value)
                    except:
                        metadata_serialized[key] = "[Unserializable value]"
                        
        return {
            "id": self.id,
            "content": content_str,
            "memory_type": self.memory_type,
            "significance": self.significance,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": self.access_count,
            "metadata": metadata_serialized,
            "embedding": embedding_serialized
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