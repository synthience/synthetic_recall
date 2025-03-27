# synthians_memory_core/memory_structures.py

import time
import uuid
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field

from .custom_logger import logger # Use the shared custom logger

@dataclass
class MemoryEntry:
    """Standardized container for a single memory entry."""
    content: str
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    quickrecal_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    # Hyperbolic specific
    hyperbolic_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        self.quickrecal_score = max(0.0, min(1.0, self.quickrecal_score))
        # Ensure embedding is numpy array
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            if isinstance(self.embedding, torch.Tensor):
                self.embedding = self.embedding.detach().cpu().numpy()
            elif isinstance(self.embedding, list):
                self.embedding = np.array(self.embedding, dtype=np.float32)
            else:
                logger.warning("MemoryEntry", f"Unsupported embedding type {type(self.embedding)} for ID {self.id}, clearing.")
                self.embedding = None

        if self.hyperbolic_embedding is not None and not isinstance(self.hyperbolic_embedding, np.ndarray):
            if isinstance(self.hyperbolic_embedding, torch.Tensor):
                self.hyperbolic_embedding = self.hyperbolic_embedding.detach().cpu().numpy()
            elif isinstance(self.hyperbolic_embedding, list):
                 self.hyperbolic_embedding = np.array(self.hyperbolic_embedding, dtype=np.float32)
            else:
                logger.warning("MemoryEntry", f"Unsupported hyperbolic embedding type {type(self.hyperbolic_embedding)} for ID {self.id}, clearing.")
                self.hyperbolic_embedding = None

    def record_access(self):
        self.access_count += 1
        self.last_access_time = time.time()

    def get_effective_quickrecal(self, decay_rate: float = 0.05) -> float:
        """Calculate effective QuickRecal score with time decay."""
        age_days = (time.time() - self.timestamp) / 86400
        if age_days < 1: return self.quickrecal_score
        importance_factor = 0.5 + (0.5 * self.quickrecal_score)
        effective_decay_rate = decay_rate / importance_factor
        decay_factor = np.exp(-effective_decay_rate * (age_days - 1))
        return max(0.0, min(1.0, self.quickrecal_score * decay_factor))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "quickrecal_score": self.quickrecal_score,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary."""
        embedding = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") else None
        hyperbolic = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        # Handle legacy 'significance' field
        quickrecal = data.get("quickrecal_score", data.get("significance", 0.5))

        return cls(
            content=data["content"],
            embedding=embedding,
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            quickrecal_score=quickrecal,
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_access_time=data.get("last_access_time"),
            hyperbolic_embedding=hyperbolic
        )

class MemoryAssembly:
    """Represents a group of related memories forming a coherent assembly."""
    def __init__(self,
                 geometry_manager, # Pass GeometryManager for consistency
                 assembly_id: str = None,
                 name: str = None,
                 description: str = None):
        self.geometry_manager = geometry_manager
        self.assembly_id = assembly_id or f"asm_{uuid.uuid4().hex[:12]}"
        self.name = name or f"Assembly-{self.assembly_id[:8]}"
        self.description = description or ""
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0

        self.memories: Set[str] = set()  # IDs of memories in this assembly
        self.composite_embedding: Optional[np.ndarray] = None
        self.hyperbolic_embedding: Optional[np.ndarray] = None
        self.emotion_profile: Dict[str, float] = {}
        self.keywords: Set[str] = set()
        self.activation_level: float = 0.0
        self.activation_decay_rate: float = 0.05

    def add_memory(self, memory: MemoryEntry):
        """Add a memory and update assembly properties."""
        if memory.id in self.memories:
            return False
        self.memories.add(memory.id)

        # --- Update Composite Embedding ---
        if memory.embedding is not None:
            target_dim = self.geometry_manager.config['embedding_dim']
            # Align memory embedding to target dimension
            mem_emb = memory.embedding
            if mem_emb.shape[0] != target_dim:
                 aligned_mem_emb, _ = self.geometry_manager._align_vectors(mem_emb, np.zeros(target_dim))
            else:
                 aligned_mem_emb = mem_emb

            normalized_mem_emb = self.geometry_manager._normalize(aligned_mem_emb)

            if self.composite_embedding is None:
                self.composite_embedding = normalized_mem_emb
            else:
                # Align composite embedding if needed (should already be target_dim)
                if self.composite_embedding.shape[0] != target_dim:
                     aligned_comp_emb, _ = self.geometry_manager._align_vectors(self.composite_embedding, np.zeros(target_dim))
                else:
                     aligned_comp_emb = self.composite_embedding

                normalized_composite = self.geometry_manager._normalize(aligned_comp_emb)

                # Simple averaging (could be weighted later)
                n = len(self.memories)
                self.composite_embedding = ((n - 1) * normalized_composite + normalized_mem_emb) / n
                # Re-normalize
                self.composite_embedding = self.geometry_manager._normalize(self.composite_embedding)

            # Update hyperbolic embedding if enabled
            if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC:
                self.hyperbolic_embedding = self.geometry_manager._to_hyperbolic(self.composite_embedding)

        # --- Update Emotion Profile ---
        mem_emotion = memory.metadata.get("emotional_context", {})
        if mem_emotion:
            self._update_emotion_profile(mem_emotion)

        # --- Update Keywords ---
        # Simple keyword extraction (could use NLP later)
        content_words = set(re.findall(r'\b\w{3,}\b', memory.content.lower()))
        self.keywords.update(content_words)
        # Limit keyword set size if needed
        if len(self.keywords) > 100:
            # Simple strategy: keep most frequent or randomly sample
            pass # Placeholder for keyword pruning logic

        return True

    def _update_emotion_profile(self, mem_emotion: Dict[str, Any]):
        """Update aggregated emotional profile."""
        n = len(self.memories)
        for emotion, score in mem_emotion.get("emotions", {}).items():
            current_score = self.emotion_profile.get(emotion, 0.0)
            # Weighted average (giving slightly more weight to existing profile)
            self.emotion_profile[emotion] = (current_score * (n - 1) * 0.6 + score * 0.4) / max(1, (n - 1) * 0.6 + 0.4)

    def get_similarity(self, query_embedding: np.ndarray) -> float:
        """Calculate similarity between query and assembly embedding."""
        ref_embedding = self.hyperbolic_embedding if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC and self.hyperbolic_embedding is not None else self.composite_embedding

        if ref_embedding is None:
            return 0.0

        return self.geometry_manager.calculate_similarity(query_embedding, ref_embedding)

    def activate(self, level: float):
        self.activation_level = min(1.0, max(0.0, level))
        self.last_access_time = time.time()
        self.access_count += 1

    def decay_activation(self):
        self.activation_level = max(0.0, self.activation_level - self.activation_decay_rate)

    def to_dict(self) -> Dict[str, Any]:
        """Convert assembly to dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "name": self.name,
            "description": self.description,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "memory_ids": list(self.memories),
            "composite_embedding": self.composite_embedding.tolist() if self.composite_embedding is not None else None,
            "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None,
            "emotion_profile": self.emotion_profile,
            "keywords": list(self.keywords),
            "activation_level": self.activation_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], geometry_manager) -> 'MemoryAssembly':
        """Create assembly from dictionary."""
        assembly = cls(
            geometry_manager,
            assembly_id=data["assembly_id"],
            name=data["name"],
            description=data["description"]
        )
        assembly.creation_time = data.get("creation_time")
        assembly.last_access_time = data.get("last_access_time")
        assembly.access_count = data.get("access_count", 0)
        assembly.memories = set(data.get("memory_ids", []))
        assembly.composite_embedding = np.array(data["composite_embedding"], dtype=np.float32) if data.get("composite_embedding") else None
        assembly.hyperbolic_embedding = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        assembly.emotion_profile = data.get("emotion_profile", {})
        assembly.keywords = set(data.get("keywords", []))
        assembly.activation_level = data.get("activation_level", 0.0)
        return assembly
