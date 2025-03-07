"""
LUCID RECALL PROJECT
Long-Term Memory Storage Handler

Handles persistent storage and retrieval of long-term memories,
integrating memory decay and significance weighting.
"""

import json
import os
import time
import logging
import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from memory_types import MemoryEntry, MemoryTypes

logger = logging.getLogger(__name__)

class LongTermMemoryStorage:
    """
    Persistent storage system for long-term memories.
    
    Features:
    - Stores significant memories persistently
    - Implements memory decay over time
    - Supports retrieval by relevance and significance
    - Indexes stored memories for fast search
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the long-term memory storage.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.storage_path = Path(self.config.get("storage_path", "./storage/ltm_storage"))
        self.index_file = self.storage_path / "memory_index.json"
        self.max_size = self.config.get("max_memories", 10000)
        self.significance_threshold = self.config.get("significance_threshold", 0.3)
        self.decay_rate = self.config.get("decay_rate", 0.05)  # Exponential decay factor

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load memory index
        self.memory_index = self._load_index()

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(f"Initialized LongTermMemoryStorage with max_size={self.max_size}")

    def _load_index(self) -> Dict[str, Any]:
        """Load or create the memory index file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Memory index corrupted, creating a new one.")
        
        return {"memories": {}, "last_updated": time.time()}

    def _save_index(self):
        """Persist the memory index to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.memory_index, f, indent=2)
            self.memory_index["last_updated"] = time.time()
        except Exception as e:
            logger.error(f"Error saving memory index: {e}")

    async def store_memory(self, content: str, embedding: Optional[torch.Tensor] = None, 
                           significance: float = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory persistently.
        
        Args:
            content: Memory content text
            embedding: Optional embedding vector
            significance: Significance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        async with self._lock:
            memory_id = f"ltm_{int(time.time() * 1000)}"

            # Ensure significance value is reasonable
            if significance is None:
                significance = 0.5  # Default value
            significance = max(0.0, min(1.0, significance))
            
            # Create memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                content=content,
                memory_type=MemoryTypes.EPISODIC,
                embedding=embedding,
                significance=significance,
                metadata=metadata or {}
            )

            # Save memory to disk
            memory_file = self.storage_path / f"{memory_id}.json"
            try:
                with open(memory_file, "w", encoding="utf-8") as f:
                    json.dump(memory_entry.to_dict(), f, indent=2)
                self.memory_index["memories"][memory_id] = {"significance": significance, "timestamp": time.time()}
                self._save_index()
                return memory_id
            except Exception as e:
                logger.error(f"Error saving memory {memory_id}: {e}")
                return ""

    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant long-term memories based on query.

        Args:
            query: Search query text
            limit: Max number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memory entries
        """
        async with self._lock:
            relevant_memories = []

            # Load all stored memory files (with caching to optimize retrieval)
            memory_files = sorted(self.storage_path.glob("*.json"), key=os.path.getmtime, reverse=True)
            for memory_file in memory_files:
                try:
                    with open(memory_file, "r", encoding="utf-8") as f:
                        memory_data = json.load(f)

                    # Filter by significance
                    significance = memory_data.get("significance", 0.5)
                    if significance < min_significance:
                        continue

                    # Simple text matching for now (can be extended with embeddings)
                    if query.lower() in memory_data["content"].lower():
                        relevant_memories.append(memory_data)

                    if len(relevant_memories) >= limit:
                        break  # Stop if limit is reached

                except Exception as e:
                    logger.error(f"Error retrieving memory from {memory_file}: {e}")

            return relevant_memories

    async def apply_decay(self):
        """Apply significance decay to stored memories."""
        async with self._lock:
            updated_memories = {}
            current_time = time.time()

            for memory_id, metadata in self.memory_index["memories"].items():
                age_days = (current_time - metadata["timestamp"]) / 86400
                new_significance = metadata["significance"] * (1 - self.decay_rate * age_days)

                if new_significance >= self.significance_threshold:
                    updated_memories[memory_id] = {"significance": new_significance, "timestamp": metadata["timestamp"]}
                else:
                    # Delete the memory file
                    memory_file = self.storage_path / f"{memory_id}.json"
                    if memory_file.exists():
                        os.remove(memory_file)

            self.memory_index["memories"] = updated_memories
            self._save_index()
            logger.info(f"Applied memory decay, {len(updated_memories)} memories retained.")

    async def backup(self) -> bool:
        """Backup all stored memories."""
        backup_file = self.storage_path / f"memory_backup_{int(time.time())}.json"
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(self.memory_index, f, indent=2)
            logger.info(f"Backup created: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve storage statistics."""
        return {
            "total_memories": len(self.memory_index["memories"]),
            "storage_path": str(self.storage_path),
            "significance_threshold": self.significance_threshold,
            "decay_rate": self.decay_rate,
            "last_updated": self.memory_index.get("last_updated", 0),
        }
