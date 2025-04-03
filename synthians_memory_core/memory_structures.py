# synthians_memory_core/memory_structures.py

import time
import uuid
import re
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .custom_logger import logger
from .geometry_manager import GeometryType

def _parse_datetime_helper(ts_data: Union[str, int, float, None],
                           field_name: str,
                           context_id: str) -> Optional[datetime]:
    if ts_data is None:
        return None
    
    # If it's already a datetime object, return it
    if isinstance(ts_data, datetime):
        return ts_data
    
    # Convert string to datetime
    if isinstance(ts_data, str):
        try:
            # Try parsing with various formats
            try:
                return datetime.fromisoformat(ts_data)
            except ValueError:
                pass
            
            try:
                # Fall back to dateutil for more flexible parsing
                from dateutil import parser
                return parser.parse(ts_data)
            except (ImportError, ValueError) as e:
                logger.error(f"Invalid datetime '{ts_data}' for field '{field_name}' in object '{context_id}': {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error parsing datetime '{ts_data}' for field '{field_name}' in object '{context_id}': {e}")
            return None
    
    # Convert numeric timestamp to datetime
    if isinstance(ts_data, (int, float)):
        try:
            # Handle millisecond timestamps vs second timestamps
            if ts_data > 1e10:  # Milliseconds timestamp (13 digits)
                ts_data = ts_data / 1000.0
            return datetime.fromtimestamp(ts_data, tz=timezone.utc)
        except (ValueError, OverflowError) as e:
            logger.error(f"Invalid timestamp value {ts_data} for field '{field_name}' in object '{context_id}': {e}")
            return None
    
    logger.error(f"Unsupported timestamp type {type(ts_data)} for field '{field_name}' in object '{context_id}'")
    return None

@dataclass
class MemoryEntry:
    content: str
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quickrecal_score: float = 0.5
    quickrecal_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hyperbolic_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        self.quickrecal_score = max(0.0, min(1.0, self.quickrecal_score))
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = datetime.fromtimestamp(self.timestamp, timezone.utc)
        if isinstance(self.last_access_time, (int, float)):
            self.last_access_time = datetime.fromtimestamp(self.last_access_time, timezone.utc)

        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            if isinstance(self.embedding, torch.Tensor):
                self.embedding = self.embedding.cpu().numpy()
            elif isinstance(self.embedding, list):
                self.embedding = np.array(self.embedding, dtype=np.float32)
            else:
                logger.warning(
                    "MemoryEntry",
                    f"Unsupported embedding type {type(self.embedding)} for ID {self.id}; clearing."
                )
                self.embedding = None

        if self.hyperbolic_embedding is not None and not isinstance(self.hyperbolic_embedding, np.ndarray):
            if isinstance(self.hyperbolic_embedding, torch.Tensor):
                self.hyperbolic_embedding = self.hyperbolic_embedding.cpu().numpy()
            elif isinstance(self.hyperbolic_embedding, list):
                self.hyperbolic_embedding = np.array(self.hyperbolic_embedding, dtype=np.float32)
            else:
                logger.warning(
                    "MemoryEntry",
                    f"Unsupported hyperbolic_embedding type {type(self.hyperbolic_embedding)} for ID {self.id}; clearing."
                )
                self.hyperbolic_embedding = None

    def record_access(self):
        self.access_count += 1
        self.last_access_time = datetime.now(timezone.utc)

    def get_effective_quickrecal(self, decay_rate: float = 0.05) -> float:
        age_seconds = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        age_days = age_seconds / 86400.0
        if age_days < 1.0:
            return self.quickrecal_score
        importance_factor = 0.5 + (0.5 * self.quickrecal_score)
        effective_decay_rate = decay_rate / max(0.1, importance_factor)
        decay_factor = np.exp(-effective_decay_rate * (age_days - 1.0))
        return max(0.0, min(1.0, self.quickrecal_score * decay_factor))

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary for serialization."""
        try:
            return {
                "id": self.id,
                "content": self.content,
                "embedding": self.embedding.tolist() if self.embedding is not None else None,
                "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else 
                            str(self.timestamp) if self.timestamp is not None else None,
                "quickrecal_score": self.quickrecal_score,
                "quickrecal_updated": self.quickrecal_updated.isoformat() if isinstance(self.quickrecal_updated, datetime) else 
                                      str(self.quickrecal_updated) if self.quickrecal_updated is not None else None,
                "metadata": self.metadata,
                "access_count": self.access_count,
                "last_access_time": self.last_access_time.isoformat() if isinstance(self.last_access_time, datetime) else 
                                     str(self.last_access_time) if self.last_access_time is not None else None,
                "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None
            }
        except Exception as e:
            logger.error(f"Error serializing memory {self.id}: {str(e)}", exc_info=True)
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        mem_id = data.get("id", f"mem_{uuid.uuid4().hex[:8]}")
        embedding = None
        hyperbolic = None
        if data.get("embedding") is not None:
            try:
                embedding = np.array(data["embedding"], dtype=np.float32)
            except Exception as e:
                logger.error(
                    "MemoryEntry.from_dict",
                    f"Error loading embedding for {mem_id}: {str(e)}"
                )
        if data.get("hyperbolic_embedding") is not None:
            try:
                hyperbolic = np.array(data["hyperbolic_embedding"], dtype=np.float32)
            except Exception as e:
                logger.error(
                    "MemoryEntry.from_dict",
                    f"Error loading hyperbolic_embedding for {mem_id}: {str(e)}"
                )

        timestamp = _parse_datetime_helper(data.get("timestamp"), "timestamp", mem_id) or datetime.now(timezone.utc)
        last_access = _parse_datetime_helper(data.get("last_access_time"), "last_access_time", mem_id) or datetime.now(timezone.utc)
        qr_updated = _parse_datetime_helper(data.get("quickrecal_updated"), "quickrecal_updated", mem_id)
        quickrecal = data.get("quickrecal_score", 0.5)

        return cls(
            content=data.get("content", ""),
            embedding=embedding,
            id=mem_id,
            timestamp=timestamp,
            quickrecal_score=quickrecal,
            quickrecal_updated=qr_updated,
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_access_time=last_access,
            hyperbolic_embedding=hyperbolic
        )

class MemoryAssembly:
    assembly_schema_version = "1.8"  # Updated for Phase 5.8

    def __init__(
        self,
        geometry_manager,
        assembly_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.geometry_manager = geometry_manager
        self.assembly_id = assembly_id or f"asm:{uuid.uuid4().hex[:12]}"
        self.name = name or f"Assembly-{self.assembly_id[:8]}"
        self.description = description or ""
        self.creation_time = datetime.now(timezone.utc)
        self.last_access_time = self.creation_time
        self.access_count = 0
        self.activation_count = 0
        self.last_activated = 0.0
        self.last_activation = self.creation_time

        self.memory_manager = None
        self.memories: Set[str] = set()
        self.composite_embedding: Optional[np.ndarray] = None
        self.hyperbolic_embedding: Optional[np.ndarray] = None
        self.emotion_profile: Dict[str, float] = {}
        self.keywords: Set[str] = set()
        self.activation_level: float = 0.0
        self.activation_decay_rate: float = 0.05
        
        # Phase 5.8: Add timestamp for tracking vector index synchronization
        self.vector_index_updated_at: Optional[datetime] = None
        self.tags: Set[str] = set()
        self.topics: List[str] = []
        self.is_active: bool = True  # Lifecycle flag for assembly management
        self.merged_from: List[str] = []  # Track assemblies that were merged into this one

    def add_memory(self, memory: MemoryEntry, validated_embedding: Optional[np.ndarray] = None) -> bool:
        if memory.id in self.memories:
            return False
        self.memories.add(memory.id)

        if validated_embedding is not None:
            mem_emb = validated_embedding
        else:
            if memory.embedding is None:
                logger.debug("MemoryAssembly.add_memory", f"Memory {memory.id} has no embedding; skip updating composite.")
                return True
            mem_emb = self.geometry_manager._validate_vector(memory.embedding, f"Memory {memory.id} Emb")
            if mem_emb is None:
                logger.warning("MemoryAssembly.add_memory",
                               f"Invalid embedding for {memory.id}; skipping embedding update.")
                return True

        mem_emb = self.geometry_manager._normalize(mem_emb)

        if self.composite_embedding is None:
            self.composite_embedding = mem_emb
        else:
            current_comp = self.geometry_manager._validate_vector(
                self.composite_embedding,
                f"Assembly {self.assembly_id} Composite Emb"
            )
            if current_comp is None:
                logger.warning(
                    "MemoryAssembly",
                    f"Composite embedding invalid for {self.assembly_id}; resetting."
                )
                self.composite_embedding = mem_emb
            else:
                n = len(self.memories)
                new_comp = ((n - 1) * current_comp + mem_emb) / float(n)
                self.composite_embedding = self.geometry_manager._normalize(new_comp)

        if self.geometry_manager.config.get('geometry_type') == GeometryType.HYPERBOLIC:
            self.hyperbolic_embedding = self.geometry_manager._to_hyperbolic(self.composite_embedding)

        mem_emotion = memory.metadata.get("emotional_context", {})
        if mem_emotion:
            self._update_emotion_profile(mem_emotion)

        content_words = set(re.findall(r'\b\w{3,}\b', memory.content.lower()))
        self.keywords.update(content_words)
        if len(self.keywords) > 200:
            self.keywords = set(list(self.keywords)[:200])

        return True

    def _update_emotion_profile(self, mem_emotion: Dict[str, Any]):
        n = len(self.memories)
        if "emotions" not in mem_emotion:
            return
        for emotion, score in mem_emotion["emotions"].items():
            current_score = self.emotion_profile.get(emotion, 0.0)
            new_score = (current_score * (n - 1) + score) / float(n)
            self.emotion_profile[emotion] = new_score

    def get_similarity(self, query_embedding: np.ndarray) -> float:
        ref_emb = self.hyperbolic_embedding if (
            self.geometry_manager.config.get('geometry_type') == GeometryType.HYPERBOLIC and
            self.hyperbolic_embedding is not None
        ) else self.composite_embedding

        if ref_emb is None:
            return 0.0
        return self.geometry_manager.calculate_similarity(query_embedding, ref_emb)

    def activate(self, level: float):
        self.activation_level = min(1.0, max(0.0, level))
        self.last_access_time = datetime.now(timezone.utc)
        self.access_count += 1
        self.activation_count += 1
        self.last_activated = time.time()
        self.last_activation = datetime.now(timezone.utc)
        logger.debug(f"Assembly {self.assembly_id} activated at level {self.activation_level:.3f}")

    def decay_activation(self):
        self.activation_level = max(0.0, self.activation_level - self.activation_decay_rate)

    def update_vector_index(self, vector_index) -> bool:
        """Synchronize this assembly's embedding with the vector index.
        
        This method ensures the assembly's composite embedding is properly indexed
        in the vector index for retrieval, and updates the vector_index_updated_at
        timestamp to track synchronization status.
        
        Args:
            vector_index: The MemoryVectorIndex instance to update
            
        Returns:
            bool: True if successfully synchronized, False otherwise
        """
        if self.composite_embedding is None:
            logger.warning(f"Cannot update vector index for assembly {self.assembly_id}: No composite embedding")
            return False
            
        if not self.is_active:
            logger.debug(f"Skipping vector index update for inactive assembly {self.assembly_id}")
            return False
            
        try:
            # Validate embedding before adding to index
            validated_embedding = self.geometry_manager._validate_vector(
                self.composite_embedding, 
                f"Assembly {self.assembly_id} Composite Embedding"
            )
            
            if validated_embedding is None:
                logger.warning(f"Invalid composite embedding for assembly {self.assembly_id}")
                return False
                
            # Use a consistent ID format for assemblies in the vector index
            assembly_vector_id = f"asm:{self.assembly_id}"
            
            # Update the vector in the index
            success = False
            if assembly_vector_id in vector_index.id_to_index:
                # Update existing vector
                success = vector_index.update_entry(assembly_vector_id, validated_embedding)
            else:
                # Add new vector
                success = vector_index.add(assembly_vector_id, validated_embedding)
                
            if success:
                # Update synchronization timestamp to mark successful index update
                self.vector_index_updated_at = datetime.now(timezone.utc)
                logger.debug(f"Assembly {self.assembly_id} synchronized with vector index")
            else:
                logger.error(f"Failed to update vector index for assembly {self.assembly_id}")
                
            return success
        except Exception as e:
            logger.error(f"Error updating vector index for assembly {self.assembly_id}: {str(e)}", exc_info=True)
            return False
            
    async def update_vector_index_async(self, vector_index) -> bool:
        """Asynchronously synchronize this assembly's embedding with the vector index.
        
        Args:
            vector_index: The MemoryVectorIndex instance to update
            
        Returns:
            bool: True if successfully synchronized, False otherwise
        """
        if self.composite_embedding is None:
            logger.warning(f"Cannot update vector index for assembly {self.assembly_id}: No composite embedding")
            return False
            
        if not self.is_active:
            logger.debug(f"Skipping vector index update for inactive assembly {self.assembly_id}")
            return False
            
        try:
            # Validate embedding before adding to index
            validated_embedding = self.geometry_manager._validate_vector(
                self.composite_embedding, 
                f"Assembly {self.assembly_id} Composite Embedding"
            )
            
            if validated_embedding is None:
                logger.warning(f"Invalid composite embedding for assembly {self.assembly_id}")
                return False
                
            # Use a consistent ID format for assemblies in the vector index
            assembly_vector_id = f"asm:{self.assembly_id}"
            
            # Update the vector in the index asynchronously
            success = False
            if assembly_vector_id in vector_index.id_to_index:
                # Update existing vector
                success = await vector_index.update_entry_async(assembly_vector_id, validated_embedding)
            else:
                # Add new vector
                success = await vector_index.add_async(assembly_vector_id, validated_embedding)
                
            if success:
                # Update synchronization timestamp to mark successful index update
                self.vector_index_updated_at = datetime.now(timezone.utc)
                logger.debug(f"Assembly {self.assembly_id} synchronized with vector index")
            else:
                logger.error(f"Failed to update vector index for assembly {self.assembly_id}")
                
            return success
        except Exception as e:
            logger.error(f"Error updating vector index for assembly {self.assembly_id}: {str(e)}", exc_info=True)
            return False

    def is_synchronized(self, max_allowed_drift_seconds: int = 3600) -> bool:
        """Check if this assembly is properly synchronized with the vector index.
        
        An assembly is considered synchronized if its vector_index_updated_at
        timestamp is present and not older than the maximum allowed drift.
        
        Args:
            max_allowed_drift_seconds: Maximum allowed age of the vector_index_updated_at 
                                       timestamp in seconds (default: 1 hour)
                                       
        Returns:
            bool: True if the assembly is synchronized, False otherwise
        """
        if self.vector_index_updated_at is None:
            return False
            
        # Calculate drift in seconds
        now = datetime.now(timezone.utc)
        drift_seconds = (now - self.vector_index_updated_at).total_seconds()
        
        # Check if drift is within acceptable range
        return drift_seconds <= max_allowed_drift_seconds
        
    def boost_memory_score(self, memory_id: str, base_score: float, 
                          boost_mode: str = "linear", boost_factor: float = 0.3,
                          max_allowed_drift_seconds: int = 3600) -> float:
        """Boost a memory's relevance score based on assembly activation, if synchronized.
        
        This method implements the Phase 5.8 boosting logic to enhance memory retrieval
        based on assembly activation. It only applies the boost if the assembly is properly
        synchronized with the vector index (vector_index_updated_at is recent).
        
        Args:
            memory_id: ID of the memory to boost
            base_score: Original similarity score
            boost_mode: "linear" or "sigmoid" boost application
            boost_factor: Multiplier for the activation level (0-1)
            max_allowed_drift_seconds: Maximum allowed index synchronization drift
            
        Returns:
            float: The boosted relevance score (clamped to 0-1)
        """
        # Only boost if memory is part of this assembly
        if memory_id not in self.memories:
            return base_score
            
        # Only boost if assembly is properly synchronized
        if not self.is_synchronized(max_allowed_drift_seconds):
            # Log this event for diagnostic purposes if assembly has high activation
            if self.activation_level > 0.3:
                logger.warning(
                    f"Assembly {self.assembly_id} has activation {self.activation_level:.2f} "
                    f"but is not synchronized (last update: {self.vector_index_updated_at})"
                )
            return base_score
            
        # Only boost if assembly has meaningful activation
        if self.activation_level <= 0.01:
            return base_score
            
        # Calculate boost based on mode
        boost = 0.0
        if boost_mode == "linear":
            boost = self.activation_level * boost_factor
        elif boost_mode == "sigmoid":
            # Sigmoid provides stronger boost for higher activation levels
            import math
            x = (self.activation_level - 0.5) * 10  # Centered sigmoid
            sigmoid = 1.0 / (1.0 + math.exp(-x))
            boost = sigmoid * boost_factor
        else:
            logger.warning(f"Unknown boost mode '{boost_mode}', using no boost")
            
        # Apply boost and clamp to valid range
        boosted_score = base_score + boost
        clamped_score = max(0.0, min(1.0, boosted_score))
        
        # Log if significant boost was applied
        if boosted_score > base_score + 0.05:
            logger.debug(
                f"Assembly {self.assembly_id[:8]} boosted memory {memory_id[:8]} "
                f"from {base_score:.3f} to {clamped_score:.3f} "
                f"(activation: {self.activation_level:.2f})"
            )
            
        return clamped_score

    def get_sync_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about this assembly's synchronization status.
        
        Returns:
            Dict containing synchronization timing and status information
        """
        now = datetime.now(timezone.utc)
        drift_seconds = None
        if self.vector_index_updated_at:
            drift_seconds = (now - self.vector_index_updated_at).total_seconds()
            
        return {
            "assembly_id": self.assembly_id,
            "name": self.name,
            "memories_count": len(self.memories),
            "is_active": self.is_active,
            "activation_level": round(self.activation_level, 3),
            "activation_count": self.activation_count,
            "vector_index_updated_at": self.vector_index_updated_at.isoformat() if self.vector_index_updated_at else None,
            "drift_seconds": round(drift_seconds, 1) if drift_seconds is not None else None,
            "embedding_dimensions": len(self.composite_embedding) if isinstance(self.composite_embedding, np.ndarray) else None,
            "tags": sorted(list(self.tags)),
            "topics": self.topics,
            "last_activation": self.last_activation.isoformat() if isinstance(self.last_activation, datetime) else None,
            "assembly_schema_version": self.assembly_schema_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        try:
            keywords_list = sorted(list(self.keywords))
            memories_list = sorted(list(self.memories))
            tags_list = sorted(list(self.tags))
            return {
                # CRITICAL: add "id" so the JSON has both fields
                "id": self.assembly_id,
                "assembly_id": self.assembly_id,
                "name": self.name,
                "description": self.description,
                "keywords": keywords_list,
                "memories": memories_list,
                "composite_embedding":
                    self.composite_embedding.tolist() if isinstance(self.composite_embedding, np.ndarray) else None,
                "hyperbolic_embedding":
                    self.hyperbolic_embedding.tolist() if isinstance(self.hyperbolic_embedding, np.ndarray) else None,
                "creation_time": self.creation_time.isoformat() if isinstance(self.creation_time, datetime) else 
                                 str(self.creation_time) if self.creation_time is not None else None,
                "last_access_time": self.last_access_time.isoformat() if isinstance(self.last_access_time, datetime) else 
                                    str(self.last_access_time) if self.last_access_time is not None else None,
                "last_activation": self.last_activation.isoformat() if isinstance(self.last_activation, datetime) else 
                                  str(self.last_activation) if self.last_activation is not None else None,
                "last_activated": self.last_activated,
                "activation_count": self.activation_count,
                "activation_level": self.activation_level,
                "assembly_schema_version": self.assembly_schema_version,
                # Phase 5.8 fields for stability and synchronization tracking
                "vector_index_updated_at": self.vector_index_updated_at.isoformat() if isinstance(self.vector_index_updated_at, datetime) else None,
                "tags": tags_list,
                "topics": self.topics,
                "is_active": self.is_active,
                "merged_from": self.merged_from
            }
        except Exception as e:
            logger.error(
                f"Error serializing assembly {self.assembly_id}: {str(e)}",
                exc_info=True
            )
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any], geometry_manager) -> 'MemoryAssembly':
        if not isinstance(data, dict):
            raise ValueError("Assembly data is not a dictionary")

        # If "id" is present, use that as assembly_id
        assembly_id = data.get("id")
        if not assembly_id:
            # fallback to "assembly_id"
            assembly_id = data.get("assembly_id")
        if not assembly_id:
            logger.warning("MemoryAssembly.from_dict", "No 'id' or 'assembly_id' found in assembly data. Generating random ID.")
            assembly_id = f"asm:{uuid.uuid4().hex[:12]}"

        asm = cls(
            geometry_manager=geometry_manager,
            assembly_id=assembly_id,
            name=data.get("name"),
            description=data.get("description")
        )

        schema_version = data.get("assembly_schema_version", "0.0")
        logger.debug("MemoryAssembly.from_dict", f"Loading assembly {assembly_id} (schema v{schema_version})")

        asm.creation_time = _parse_datetime_helper(data.get("creation_time"), "creation_time", assembly_id) or asm.creation_time
        asm.last_access_time = _parse_datetime_helper(data.get("last_access_time"), "last_access_time", assembly_id) or asm.last_access_time
        last_act_dt = _parse_datetime_helper(data.get("last_activation"), "last_activation", assembly_id)
        if last_act_dt:
            asm.last_activation = last_act_dt
        asm.access_count = data.get("access_count", 0)
        asm.activation_count = data.get("activation_count", 0)
        asm.last_activated = data.get("last_activated", 0.0)
        asm.memories = set(data.get("memories", []))
        asm.keywords = set(data.get("keywords", []))
        asm.activation_level = data.get("activation_level", 0.0)
        asm.activation_decay_rate = data.get("activation_decay_rate", 0.05)

        # Handle Phase 5.8 fields with graceful fallbacks for older schema versions
        asm.vector_index_updated_at = _parse_datetime_helper(data.get("vector_index_updated_at"), 
                                                           "vector_index_updated_at", assembly_id)
        asm.tags = set(data.get("tags", []))
        asm.topics = data.get("topics", [])
        asm.is_active = data.get("is_active", True)
        asm.merged_from = data.get("merged_from", [])

        comp_emb_data = data.get("composite_embedding")
        if comp_emb_data is not None:
            try:
                arr = np.array(comp_emb_data, dtype=np.float32)
                asm.composite_embedding = geometry_manager._validate_vector(arr, "Loaded Composite Emb")
            except Exception as e:
                logger.error("MemoryAssembly.from_dict",
                             f"Error processing composite_embedding for {assembly_id}: {str(e)}")

        hyper_emb_data = data.get("hyperbolic_embedding")
        if hyper_emb_data is not None:
            try:
                arr = np.array(hyper_emb_data, dtype=np.float32)
                asm.hyperbolic_embedding = geometry_manager._validate_vector(arr, "Loaded Hyperbolic Emb")
            except Exception as e:
                logger.error("MemoryAssembly.from_dict",
                             f"Error processing hyperbolic_embedding for {assembly_id}: {str(e)}")

        asm.emotion_profile = data.get("emotion_profile", {})
        
        # Apply schema migration logic if needed
        if schema_version != cls.assembly_schema_version:
            logger.info(f"Migrating assembly {assembly_id} from schema v{schema_version} to v{cls.assembly_schema_version}")
            # Enhanced schema migration for Phase 5.8
            
            # Ensure collections are proper Set[str] types (older schemas may have lists)
            if not isinstance(asm.memories, set):
                logger.debug(f"Converting memories to set for assembly {assembly_id}")
                asm.memories = set(asm.memories or [])
                
            if not isinstance(asm.keywords, set):
                logger.debug(f"Converting keywords to set for assembly {assembly_id}")
                asm.keywords = set(asm.keywords or [])
                
            if not isinstance(asm.tags, set):
                logger.debug(f"Converting tags to set for assembly {assembly_id}")
                asm.tags = set(asm.tags or [])
                
            # Handle memory_ids (legacy field) if present but memories missing
            if not asm.memories and "memory_ids" in data:
                logger.debug(f"Migrating legacy memory_ids field for assembly {assembly_id}")
                asm.memories = set(data.get("memory_ids", []))
                
            # Ensure all Phase 5.8 fields are initialized
            if asm.vector_index_updated_at is None and "vector_index_updated_at" in data:
                # Attempt to parse timestamp even if it was initially invalid
                raw_value = data.get("vector_index_updated_at")
                if raw_value:
                    asm.vector_index_updated_at = _parse_datetime_helper(
                        raw_value, "vector_index_updated_at", assembly_id
                    )
                    
            # Add any other field migrations as needed...
            
        return asm
