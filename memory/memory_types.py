"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:42 PM EST

Memory Types: Definitions for memory system types and structures
"""

from enum import Enum
from dataclasses import dataclass
import torch
from typing import Optional
import time

class MemoryTypes(Enum):
    """Types of memories that can be stored"""
    EPISODIC = "episodic"      # Event/experience memories
    SEMANTIC = "semantic"       # Factual/conceptual memories
    PROCEDURAL = "procedural"   # Skill/procedure memories
    WORKING = "working"         # Temporary processing memories
    
@dataclass
class MemoryEntry:
    """Container for a single memory entry"""
    embedding: torch.Tensor
    memory_type: MemoryTypes
    significance: float = 0.0
    timestamp: float = time.time()
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate memory entry on creation"""
        if not isinstance(self.embedding, torch.Tensor):
            raise ValueError("Embedding must be a torch.Tensor")
            
        if not isinstance(self.memory_type, MemoryTypes):
            raise ValueError("Invalid memory type")
            
        if self.significance < 0.0 or self.significance > 1.0:
            raise ValueError("Significance must be between 0 and 1")