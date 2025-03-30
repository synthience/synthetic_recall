# synthians_memory_core/memory_structures.py

import time
import uuid
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone  # Add datetime imports

from .custom_logger import logger # Use the shared custom logger

@dataclass
class MemoryEntry:
    """Standardized container for a single memory entry."""
    content: str
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quickrecal_score: float = 0.5
    quickrecal_updated: Optional[datetime] = None  # Add missing field
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Hyperbolic specific
    hyperbolic_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        self.quickrecal_score = max(0.0, min(1.0, self.quickrecal_score))
        
        # Convert timestamp/last_access_time from potential float on init if needed
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = datetime.fromtimestamp(self.timestamp, timezone.utc)
        if isinstance(self.last_access_time, (int, float)):
            self.last_access_time = datetime.fromtimestamp(self.last_access_time, timezone.utc)

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
        self.last_access_time = datetime.now(timezone.utc)

    def get_effective_quickrecal(self, decay_rate: float = 0.05) -> float:
        """Calculate effective QuickRecal score with time decay."""
        # Calculate age using datetime objects
        age_seconds = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        age_days = age_seconds / 86400
        if age_days < 1: return self.quickrecal_score
        importance_factor = 0.5 + (0.5 * self.quickrecal_score)
        effective_decay_rate = decay_rate / max(0.1, importance_factor)  # Avoid division by zero
        decay_factor = np.exp(-effective_decay_rate * (age_days - 1))
        return max(0.0, min(1.0, self.quickrecal_score * decay_factor))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else self.timestamp,
            "quickrecal_score": self.quickrecal_score,
            "quickrecal_updated": self.quickrecal_updated.isoformat() if hasattr(self.quickrecal_updated, 'isoformat') and self.quickrecal_updated else None,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time.isoformat() if hasattr(self.last_access_time, 'isoformat') else self.last_access_time,
            "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary."""
        mem_id = data.get("id", "unknown_id") # Get ID for logging
        logger.debug("MemoryEntry.from_dict", f"Creating entry for ID: {mem_id}")
        
        try:
            embedding = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") else None
            hyperbolic = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        except Exception as e:
            logger.error("MemoryEntry.from_dict", f"Error processing embedding for ID {mem_id}", {"error": str(e)})
            embedding = None
            hyperbolic = None
            
        # Handle legacy 'significance' field
        quickrecal = data.get("quickrecal_score", data.get("significance", 0.5))

        # Helper to parse timestamp (float or ISO string)
        def parse_datetime(ts_data, field_name):
            if ts_data is None: return None
            try:
                if isinstance(ts_data, str):
                    # Handle potential Z suffix for UTC
                    if ts_data.endswith('Z'): ts_data = ts_data[:-1] + '+00:00'
                    return datetime.fromisoformat(ts_data)
                elif isinstance(ts_data, (int, float)):
                    return datetime.fromtimestamp(ts_data, timezone.utc)
                # --- ADDED Logging ---
                logger.warning("MemoryEntry.from_dict", f"Unsupported timestamp type for {field_name} in ID {mem_id}: {type(ts_data)}")
                return None
            except Exception as e:
                 # --- ADDED Logging ---
                 logger.error("MemoryEntry.from_dict", f"Error parsing {field_name} for ID {mem_id}", {"value": ts_data, "error": str(e)})
                 return None

        timestamp = parse_datetime(data.get("timestamp"), "timestamp") or datetime.now(timezone.utc)
        last_access = parse_datetime(data.get("last_access_time"), "last_access_time") or datetime.now(timezone.utc)
        qr_updated = parse_datetime(data.get("quickrecal_updated"), "quickrecal_updated")

        try:
            entry = cls(
                content=data["content"],
                embedding=embedding,
                id=mem_id, # Use pre-fetched ID
                timestamp=timestamp,
                quickrecal_score=quickrecal,
                quickrecal_updated=qr_updated,
                metadata=data.get("metadata", {}),
                access_count=data.get("access_count", 0),
                last_access_time=last_access,
                hyperbolic_embedding=hyperbolic
            )
            logger.debug("MemoryEntry.from_dict", f"Successfully created entry for ID: {mem_id}")
            return entry
        except Exception as e:
             # --- ADDED Logging ---
             logger.error("MemoryEntry.from_dict", f"Error during final object creation for ID {mem_id}", {"error": str(e)}, exc_info=True)
             raise # Re-raise after logging if creation fails fundamentally

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
        self.creation_time = datetime.now(timezone.utc)
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
        self.last_access_time = datetime.now(timezone.utc)
        self.access_count += 1

    def decay_activation(self):
        self.activation_level = max(0.0, self.activation_level - self.activation_decay_rate)

    def to_dict(self) -> Dict[str, Any]:
        """Convert assembly to dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "name": self.name,
            "description": self.description,
            "creation_time": self.creation_time.isoformat(),
            "last_access_time": self.last_access_time.isoformat(),
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
        assembly.creation_time = datetime.fromisoformat(data.get("creation_time"))
        assembly.last_access_time = datetime.fromisoformat(data.get("last_access_time"))
        assembly.access_count = data.get("access_count", 0)
        assembly.memories = set(data.get("memory_ids", []))
        assembly.composite_embedding = np.array(data["composite_embedding"], dtype=np.float32) if data.get("composite_embedding") else None
        assembly.hyperbolic_embedding = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        assembly.emotion_profile = data.get("emotion_profile", {})
        assembly.keywords = set(data.get("keywords", []))
        assembly.activation_level = data.get("activation_level", 0.0)
        return assembly
