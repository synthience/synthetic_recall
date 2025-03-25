from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class NemoAPIConfig:
    model_path: str
    embedding_dimension: int
    cutlass_config: Dict[str, Any]

@dataclass
class MemoryManagerConfig:
    persistence_path: str
    max_memory_mb: int
    embedding_dimension: int

@dataclass
class Memory:
    id: str
    content: str
    embedding: List[float]
    timestamp: str
    context: Optional[Dict[str, Any]] = None
