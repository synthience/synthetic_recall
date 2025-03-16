"""
LUCID RECALL PROJECT
Enhanced Unified Memory Storage Interface

A standardized interface for memory storage operations that integrates
with the HPCQRFlowManager and new memory architecture.
"""

import os
import time
import json
import logging
import numpy as np
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import asyncio
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep the original MemoryType enum for compatibility
class MemoryType(Enum):
    """Types of memories that can be stored."""
    EPISODIC = "episodic"      # Event/experience memories (conversations, interactions)
    SEMANTIC = "semantic"      # Factual/conceptual memories (knowledge, facts)
    PROCEDURAL = "procedural"  # Skill/procedure memories (how to do things)
    WORKING = "working"        # Temporary processing memories (short-term)
    PERSONAL = "personal"      # Personal information about users
    IMPORTANT = "important"    # High-significance memories that should be preserved
    EMOTIONAL = "emotional"    # Memories with emotional context
    SYSTEM = "system"          # System-level memories (configs, settings)

class MemoryEntry:
    """
    Enhanced memory entry structure for unified access across systems.
    
    This class provides a standard structure for memory entries with
    consistent access patterns, serialization, and QuickRecal integration.
    """
    
    def __init__(self, 
                 content: str,
                 embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                 memory_type: Union[MemoryType, str] = MemoryType.EPISODIC,
                 quickrecal_score: float = 0.5,  # Replaced significance with quickrecal_score
                 metadata: Optional[Dict[str, Any]] = None,
                 id: Optional[str] = None,
                 timestamp: Optional[float] = None):
        """
        Initialize a memory entry.
        
        Args:
            content: Primary memory content as text
            embedding: Vector representation of content
            memory_type: Type of memory
            quickrecal_score: Importance score from QuickRecal HPC (0.0-1.0)
            metadata: Additional data about this memory
            id: Unique identifier (generated if not provided)
            timestamp: Creation time (current time if not provided)
        """
        self.content = content
        self._embedding = self._process_embedding(embedding)
        
        # Handle memory_type as string or enum
        if isinstance(memory_type, str):
            try:
                self.memory_type = MemoryType[memory_type.upper()]
            except KeyError:
                for mem_type in MemoryType:
                    if mem_type.value == memory_type.lower():
                        self.memory_type = mem_type
                        break
                else:
                    logger.warning(f"Unknown memory type: {memory_type}, defaulting to EPISODIC")
                    self.memory_type = MemoryType.EPISODIC
        else:
            self.memory_type = memory_type
            
        # Ensure quickrecal_score is in valid range
        self.quickrecal_score = max(0.0, min(1.0, quickrecal_score))
        
        self.metadata = metadata or {}
        self.id = id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        
        # Additional tracking information
        self.access_count = 0
        self.last_access_time = self.timestamp
        self.creation_source = self.metadata.get("source", "unknown")
        
        # QuickRecal-related fields
        self.original_quickrecal = self.quickrecal_score  # Store original score
        self.boost_factor = 0.0  # QuickRecal boost from repeated access
        
    def _process_embedding(self, embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]]) -> Optional[np.ndarray]:
        """Process and normalize embedding input."""
        if embedding is None:
            return None
            
        # Convert to numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Ensure correct dimensionality
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.flatten()
            
        # Normalize if not already normalized
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-5:
            embedding = embedding / norm
            
        return embedding
    
    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Get memory embedding."""
        return self._embedding
    
    @embedding.setter
    def embedding(self, value: Optional[Union[np.ndarray, torch.Tensor, List[float]]]) -> None:
        """Set memory embedding with automatic processing."""
        self._embedding = self._process_embedding(value)
        
    def record_access(self) -> None:
        """Record memory access, updating tracking information and boosting QuickRecal score."""
        current_time = time.time()
        self.access_count += 1
        self.last_access_time = current_time
        
        # Update boost factor based on access frequency
        time_since_creation = current_time - self.timestamp
        if time_since_creation > 0:
            # More recent accesses provide stronger boost
            recency_factor = min(1.0, 30 * 86400 / max(1, time_since_creation))
            # More accesses provide stronger boost
            access_factor = min(1.0, self.access_count / 10)
            self.boost_factor = 0.3 * recency_factor * access_factor
    
    def get_effective_quickrecal(self) -> float:
        """
        Get effective QuickRecal score with temporal decay and access boost applied.
        
        The QuickRecal score naturally decays over time but can be boosted
        by frequent access.
        """
        current_time = time.time()
        days_elapsed = (current_time - self.timestamp) / (24 * 3600)
        
        # Calculate decay factor based on QuickRecal score
        # Higher QuickRecal scores decay more slowly
        importance_factor = 0.5 + 0.5 * self.quickrecal_score
        effective_decay_rate = 0.05 / importance_factor  # Base decay rate of 5% per day, adjusted by importance
        decay_factor = np.exp(-effective_decay_rate * days_elapsed)
        
        # Apply decay and boost
        effective_quickrecal = self.quickrecal_score * decay_factor + self.boost_factor
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, effective_quickrecal))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self._embedding.tolist() if self._embedding is not None else None,
            "memory_type": self.memory_type.value,
            "quickrecal_score": self.quickrecal_score,  # Use QuickRecal score
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "effective_quickrecal": self.get_effective_quickrecal(),  # Include effective QuickRecal
            "boost_factor": self.boost_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary."""
        # Convert embedding back to numpy array if present
        embedding_data = data.get("embedding")
        embedding = np.array(embedding_data, dtype=np.float32) if embedding_data else None
        
        # Handle legacy 'significance' field and convert to 'quickrecal_score'
        quickrecal_score = data.get("quickrecal_score")
        if quickrecal_score is None:
            quickrecal_score = data.get("significance", 0.5)
        
        return cls(
            content=data.get("content", ""),
            embedding=embedding,
            memory_type=data.get("memory_type", MemoryType.EPISODIC),
            quickrecal_score=quickrecal_score,
            metadata=data.get("metadata", {}),
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time())
        )
    
    def __str__(self) -> str:
        """String representation of memory."""
        return f"Memory({self.id[:8]}, type={self.memory_type.value}, quickrecal={self.quickrecal_score:.2f}): {self.content[:50]}..."


class EnhancedMemoryStorage:
    """
    Enhanced unified memory storage with QuickRecal HPC-QR integration.
    
    This class provides a standardized interface for memory storage operations
    that integrates with the HPCQRFlowManager and new memory architecture.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced memory storage.
        
        Args:
            config: Configuration options including:
                - storage_path: Path for memory storage
                - max_memories: Maximum number of memories to store
                - auto_prune: Whether to automatically prune memories
                - prune_threshold: Percentage at which to start pruning
                - min_quickrecal_to_store: Minimum QuickRecal threshold to store
                - persistence_enabled: Whether to persist memories to disk
                - backup_frequency: Seconds between backups
                - case_sensitive_search: Whether to perform case-sensitive search
                - hpc_manager: Optional HPCQRFlowManager instance
                - embedding_dim: Dimension of embeddings
        """
        self.config = {
            'storage_path': os.path.join('data', 'memory'),
            'max_memories': 10000,
            'auto_prune': True,
            'prune_threshold': 0.9,  # Prune when 90% full
            'min_quickrecal_to_store': 0.1,  # Changed from significance to quickrecal
            'persistence_enabled': True,
            'backup_frequency': 3600,  # Seconds between backups
            'case_sensitive_search': False,
            'embedding_dim': 768,  # Default embedding dimension
            **(config or {})
        }
        
        # Initialize storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_types: Dict[MemoryType, List[str]] = {mem_type: [] for mem_type in MemoryType}
        
        # Initialize HPCQRFlowManager if not provided
        if 'hpc_manager' not in self.config or self.config['hpc_manager'] is None:
            from integration.hpc_qr_flow_manager import HPCQRFlowManager
            self.hpc_manager = HPCQRFlowManager({
                'embedding_dim': self.config['embedding_dim'],
                'device': self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            })
        else:
            self.hpc_manager = self.config['hpc_manager']
        
        # Initialize directories
        if self.config['persistence_enabled']:
            os.makedirs(self.config['storage_path'], exist_ok=True)
            
        # Statistics
        self.stats = {
            'memories_stored': 0,
            'memories_retrieved': 0,
            'memories_pruned': 0,
            'memories_purged': 0,
            'memories_updated': 0,
            'last_prune_time': 0,
            'last_backup_time': 0,
            'avg_quickrecal': 0.0,  # Track average QuickRecal score
            'quickrecal_histogram': {  # Track QuickRecal score distribution
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0, 
                '0.6-0.8': 0,
                '0.8-1.0': 0
            }
        }
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._backup_task = None
        
        # Load memories if persistence enabled
        if self.config['persistence_enabled']:
            asyncio.create_task(self._load_memories())
            
        logger.info(f"Initialized EnhancedMemoryStorage with HPC-QR integration")
    
    async def store(self, 
                  content: str, 
                  embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None, 
                  memory_type: Union[MemoryType, str] = MemoryType.EPISODIC,
                  quickrecal_score: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  memory_id: Optional[str] = None) -> str:
        """
        Store a memory with QuickRecal HPC-QR processing.
        
        Args:
            content: Memory content as text
            embedding: Optional pre-computed embedding
            memory_type: Type of memory
            quickrecal_score: Optional pre-computed QuickRecal score
            metadata: Additional metadata
            memory_id: Optional specific ID to use
            
        Returns:
            Memory ID
        """
        async with self._lock:
            start_time = time.time()
            
            # Process the embedding and calculate QuickRecal score with HPC-QR if not provided
            if embedding is None:
                # Generate embedding from text
                embedding_tensor = await self.hpc_manager.get_embedding(content)
                
                # Process through HPC-QR to get QuickRecal score
                processed_embedding, computed_quickrecal = await self.hpc_manager.process_embedding(embedding_tensor)
                
                # Convert to proper format
                if isinstance(processed_embedding, torch.Tensor):
                    embedding = processed_embedding.detach().cpu().numpy()
                else:
                    embedding = processed_embedding
                    
                # Use computed QuickRecal score if not explicitly provided
                if quickrecal_score is None:
                    quickrecal_score = float(computed_quickrecal)
                    
                logger.debug(f"Generated embedding and QuickRecal score ({quickrecal_score:.3f}) in {time.time() - start_time:.3f}s")
            elif quickrecal_score is None:
                # If embedding provided but no QuickRecal score, calculate it
                if isinstance(embedding, list):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                else:
                    embedding_tensor = embedding
                
                # Process through HPC-QR
                _, computed_quickrecal = await self.hpc_manager.process_embedding(embedding_tensor)
                quickrecal_score = float(computed_quickrecal)
                logger.debug(f"Calculated QuickRecal score ({quickrecal_score:.3f}) for provided embedding in {time.time() - start_time:.3f}s")
            
            # Create memory entry
            memory = MemoryEntry(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                quickrecal_score=quickrecal_score,
                metadata=metadata or {},
                id=memory_id,
                timestamp=time.time()
            )
                
            # Skip if below minimum QuickRecal threshold
            if memory.quickrecal_score < self.config['min_quickrecal_to_store']:
                logger.debug(f"Memory below minimum QuickRecal threshold ({memory.quickrecal_score:.2f}), not storing")
                return ""
                
            # Check if we need to prune
            if self.config['auto_prune'] and len(self.memories) >= self.config['max_memories'] * self.config['prune_threshold']:
                await self._prune_memories()
                
            # Check if we're still full after pruning
            if len(self.memories) >= self.config['max_memories']:
                logger.warning(f"Memory storage full ({len(self.memories)}/{self.config['max_memories']}), cannot store new memory")
                return ""
                
            # Store the memory
            self.memories[memory.id] = memory
            
            # Update type index
            self.memory_types[memory.memory_type].append(memory.id)
            
            # Update stats
            self.stats['memories_stored'] += 1
            self._update_quickrecal_stats(memory.quickrecal_score)
            
            # Persist if enabled
            if self.config['persistence_enabled']:
                await self._persist_memory(memory)
                
            # Start backup task if needed
            if self.config['persistence_enabled'] and (
                self._backup_task is None or 
                self._backup_task.done() or 
                time.time() - self.stats['last_backup_time'] > self.config['backup_frequency']):
                self._backup_task = asyncio.create_task(self._backup_memories())
                
            logger.info(f"Stored memory {memory.id} with QuickRecal score {memory.quickrecal_score:.3f}")
            return memory.id
    
    def _update_quickrecal_stats(self, quickrecal_score: float) -> None:
        """Update QuickRecal statistics."""
        # Update average
        old_avg = self.stats['avg_quickrecal']
        old_count = self.stats['memories_stored'] - 1  # Subtract 1 because we already incremented
        if old_count > 0:
            self.stats['avg_quickrecal'] = (old_avg * old_count + quickrecal_score) / (old_count + 1)
        else:
            self.stats['avg_quickrecal'] = quickrecal_score
            
        # Update histogram
        if 0.0 <= quickrecal_score < 0.2:
            self.stats['quickrecal_histogram']['0.0-0.2'] += 1
        elif 0.2 <= quickrecal_score < 0.4:
            self.stats['quickrecal_histogram']['0.2-0.4'] += 1
        elif 0.4 <= quickrecal_score < 0.6:
            self.stats['quickrecal_histogram']['0.4-0.6'] += 1
        elif 0.6 <= quickrecal_score < 0.8:
            self.stats['quickrecal_histogram']['0.6-0.8'] += 1
        else:
            self.stats['quickrecal_histogram']['0.8-1.0'] += 1
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory entry or None if not found
        """
        async with self._lock:
            memory = self.memories.get(memory_id)
            
            if memory:
                # Record access
                memory.record_access()
                self.stats['memories_retrieved'] += 1
                
            return memory
    
    async def search(self, 
                   query_embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                   query_text: Optional[str] = None,
                   memory_type: Optional[Union[MemoryType, str]] = None,
                   min_quickrecal: float = 0.0,
                   max_count: int = 10,
                   time_range: Optional[Tuple[float, float]] = None) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for memories using embeddings or text.
        
        Args:
            query_embedding: Vector query for semantic search
            query_text: Text query for keyword search
            memory_type: Optional type filter
            min_quickrecal: Minimum QuickRecal threshold
            max_count: Maximum number of results
            time_range: Optional (start_time, end_time) tuple
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        async with self._lock:
            # Generate embedding from text if needed
            if query_embedding is None and query_text is not None:
                try:
                    query_embedding_tensor = await self.hpc_manager.get_embedding(query_text)
                    query_embedding = query_embedding_tensor.detach().cpu().numpy()
                except Exception as e:
                    logger.error(f"Error generating embedding for query text: {e}")
                    # Fall back to keyword search if embedding generation fails
                    query_embedding = None
            
            # Prepare candidates
            candidates = []
            
            # Convert memory_type string to enum if needed
            if isinstance(memory_type, str):
                try:
                    memory_type = MemoryType[memory_type.upper()]
                except KeyError:
                    for mem_type in MemoryType:
                        if mem_type.value == memory_type.lower():
                            memory_type = mem_type
                            break
            
            # Get candidate memories
            if memory_type:
                # Get only memories of specified type
                candidate_ids = self.memory_types.get(memory_type, [])
                candidates = [self.memories[mid] for mid in candidate_ids if mid in self.memories]
            else:
                # Get all memories
                candidates = list(self.memories.values())
                
            # Apply QuickRecal filter
            candidates = [mem for mem in candidates if mem.get_effective_quickrecal() >= min_quickrecal]
                
            # Apply time range filter if provided
            if time_range:
                start_time, end_time = time_range
                candidates = [mem for mem in candidates if start_time <= mem.timestamp <= end_time]
                
            # Process different search types
            if query_embedding is not None:
                # Semantic search using embeddings
                return await self._semantic_search(candidates, query_embedding, max_count)
            elif query_text:
                # Keyword search using text
                return await self._keyword_search(candidates, query_text, max_count)
            else:
                # No query, sort by QuickRecal and recency
                return await self._default_search(candidates, max_count)
    
    async def _semantic_search(self, 
                             candidates: List[MemoryEntry],
                             query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
                             max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            candidates: List of candidate memories
            query_embedding: Query embedding
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        # Convert query embedding to numpy array
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        elif isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
        # Ensure correct dimensionality
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding.flatten()
            
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate similarities and effective QuickRecal for each candidate
        results = []
        for memory in candidates:
            # Skip memories without embeddings
            if memory.embedding is None:
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, memory.embedding)
            
            # Combine similarity with QuickRecal
            effective_quickrecal = memory.get_effective_quickrecal()
            # QuickRecal-weighted scoring: higher QuickRecal memories get more weight
            combined_score = 0.6 * similarity + 0.4 * effective_quickrecal
            
            results.append((memory, combined_score))
            
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def _keyword_search(self, 
                            candidates: List[MemoryEntry],
                            query_text: str,
                            max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Perform keyword search using text.
        
        Args:
            candidates: List of candidate memories
            query_text: Query text
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        # Convert query to lowercase if case-insensitive search
        if not self.config['case_sensitive_search']:
            query_text = query_text.lower()
            
        # Extract keywords from query
        keywords = query_text.split()
        
        # Calculate match scores for each candidate
        results = []
        for memory in candidates:
            content = memory.content
            
            # Convert content to lowercase if case-insensitive search
            if not self.config['case_sensitive_search']:
                content = content.lower()
                
            # Calculate number of matching keywords
            matching_keywords = sum(1 for keyword in keywords if keyword in content)
            match_ratio = matching_keywords / len(keywords) if keywords else 0
            
            # Calculate relevance score
            effective_quickrecal = memory.get_effective_quickrecal()
            combined_score = 0.6 * match_ratio + 0.4 * effective_quickrecal
            
            # Only include if at least one keyword matches
            if match_ratio > 0:
                results.append((memory, combined_score))
                
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def _default_search(self, 
                            candidates: List[MemoryEntry],
                            max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Default search when no query is provided.
        
        Args:
            candidates: List of candidate memories
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        results = []
        current_time = time.time()
        
        for memory in candidates:
            # Calculate recency (higher is more recent)
            recency = 1.0 / (1.0 + (current_time - memory.timestamp) / (24 * 3600))  # Normalize to days
            
            # Calculate relevance score
            effective_quickrecal = memory.get_effective_quickrecal()
            combined_score = 0.5 * effective_quickrecal + 0.5 * recency
            
            results.append((memory, combined_score))
            
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID
            updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            memory = self.memories.get(memory_id)
            
            if not memory:
                return False
                
            # Apply updates
            if 'content' in updates:
                memory.content = updates['content']
                
            if 'embedding' in updates:
                memory.embedding = updates['embedding']
                
            if 'memory_type' in updates:
                old_type = memory.memory_type
                
                # Update memory type
                if isinstance(updates['memory_type'], str):
                    try:
                        new_type = MemoryType[updates['memory_type'].upper()]
                    except KeyError:
                        for mem_type in MemoryType:
                            if mem_type.value == updates['memory_type'].lower():
                                new_type = mem_type
                                break
                        else:
                            logger.warning(f"Unknown memory type: {updates['memory_type']}, ignoring update")
                            new_type = old_type
                else:
                    new_type = updates['memory_type']
                    
                # Update type index
                if old_type != new_type:
                    if memory_id in self.memory_types[old_type]:
                        self.memory_types[old_type].remove(memory_id)
                    self.memory_types[new_type].append(memory_id)
                    memory.memory_type = new_type
                
            # Handle both 'significance' (legacy) and 'quickrecal_score' updates
            if 'quickrecal_score' in updates:
                memory.quickrecal_score = max(0.0, min(1.0, updates['quickrecal_score']))
            elif 'significance' in updates:
                memory.quickrecal_score = max(0.0, min(1.0, updates['significance']))
                
            if 'metadata' in updates:
                if isinstance(updates['metadata'], dict):
                    memory.metadata.update(updates['metadata'])
                    
            # Update stats
            self.stats['memories_updated'] += 1
            
            # Persist if enabled
            if self.config['persistence_enabled']:
                await self._persist_memory(memory)
                
            return True
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            memory = self.memories.pop(memory_id, None)
            
            if not memory:
                return False
                
            # Update type index
            if memory_id in self.memory_types[memory.memory_type]:
                self.memory_types[memory.memory_type].remove(memory_id)
                
            # Delete from disk if persistence enabled
            if self.config['persistence_enabled']:
                memory_path = os.path.join(self.config['storage_path'], f"{memory_id}.json")
                if os.path.exists(memory_path):
                    try:
                        os.remove(memory_path)
                    except Exception as e:
                        logger.error(f"Error deleting memory file: {e}")
                
            # Update stats
            self.stats['memories_purged'] += 1
            
            return True
    
    async def _prune_memories(self) -> None:
        """Prune memories with lowest QuickRecal when storage is full."""
        logger.info("Pruning memories...")
        
        # Get memories sorted by effective QuickRecal (lowest first)
        memories_with_quickrecal = [(mid, mem.get_effective_quickrecal()) for mid, mem in self.memories.items()]
        memories_with_quickrecal.sort(key=lambda x: x[1])
        
        # Calculate number to remove (20% of total)
        prune_count = max(1, int(0.2 * len(self.memories)))
        prune_count = min(prune_count, len(memories_with_quickrecal))
        
        # Remove memories
        for i in range(prune_count):
            memory_id, _ = memories_with_quickrecal[i]
            await self.delete(memory_id)
            
        # Update stats
        self.stats['memories_pruned'] += prune_count
        self.stats['last_prune_time'] = time.time()
        
        logger.info(f"Pruned {prune_count} memories")
    
    async def _persist_memory(self, memory: MemoryEntry) -> None:
        """Persist a memory to disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            memory_path = os.path.join(self.config['storage_path'], f"{memory.id}.json")
            
            # Convert to dictionary
            memory_dict = memory.to_dict()
            
            # Write to file
            with open(memory_path, 'w') as f:
                json.dump(memory_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
    
    async def _load_memories(self) -> None:
        """Load memories from disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            # Get all memory files
            memory_files = [f for f in os.listdir(self.config['storage_path']) if f.endswith('.json')]
            
            # Load each memory
            for filename in memory_files:
                try:
                    memory_path = os.path.join(self.config['storage_path'], filename)
                    
                    with open(memory_path, 'r') as f:
                        memory_dict = json.load(f)
                        
                    # Create memory
                    memory = MemoryEntry.from_dict(memory_dict)
                    
                    # Store in memory (bypassing store method to avoid recursion)
                    self.memories[memory.id] = memory
                    
                    # Update type index
                    self.memory_types[memory.memory_type].append(memory.id)
                    
                    # Update QuickRecal stats
                    self._update_quickrecal_stats(memory.quickrecal_score)
                    
                except Exception as e:
                    logger.error(f"Error loading memory {filename}: {e}")
                    
            logger.info(f"Loaded {len(self.memories)} memories from disk")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    async def _backup_memories(self) -> None:
        """Backup all memories to disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            async with self._lock:
                # Create backup directory
                backup_dir = os.path.join(self.config['storage_path'], 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                # Create backup file
                backup_time = time.strftime('%Y%m%d-%H%M%S')
                backup_path = os.path.join(backup_dir, f"memory_backup_{backup_time}.json")
                
                # Convert memories to dictionaries
                memories_dict = {mid: mem.to_dict() for mid, mem in self.memories.items()}
                
                # Write to file
                with open(backup_path, 'w') as f:
                    json.dump(memories_dict, f, indent=2)
                    
                # Update stats
                self.stats['last_backup_time'] = time.time()
                
                logger.info(f"Backed up {len(self.memories)} memories to {backup_path}")
                
                # Clean up old backups (keep last 10)
                backup_files = sorted([f for f in os.listdir(backup_dir) if f.startswith('memory_backup_')])
                if len(backup_files) > 10:
                    for old_backup in backup_files[:-10]:
                        try:
                            os.remove(os.path.join(backup_dir, old_backup))
                        except Exception as e:
                            logger.error(f"Error removing old backup {old_backup}: {e}")
                
        except Exception as e:
            logger.error(f"Error backing up memories: {e}")
    
    async def clear(self) -> None:
        """Clear all memories."""
        async with self._lock:
            self.memories.clear()
            self.memory_types = {mem_type: [] for mem_type in MemoryType}
            
            # Update stats
            self.stats['memories_purged'] += len(self.memories)
            
            logger.info("Memory storage cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        type_counts = {mem_type.value: len(ids) for mem_type, ids in self.memory_types.items()}
        
        return {
            'total_memories': len(self.memories),
            'memory_types': type_counts,
            'memories_stored': self.stats['memories_stored'],
            'memories_retrieved': self.stats['memories_retrieved'],
            'memories_pruned': self.stats['memories_pruned'],
            'memories_purged': self.stats['memories_purged'],
            'memories_updated': self.stats['memories_updated'],
            'storage_utilization': len(self.memories) / self.config['max_memories'],
            'persistence_enabled': self.config['persistence_enabled'],
            'last_prune_time': self.stats['last_prune_time'],
            'last_backup_time': self.stats['last_backup_time'],
            'avg_quickrecal': self.stats['avg_quickrecal'],
            'quickrecal_histogram': self.stats['quickrecal_histogram']
        }
    
    async def process_with_hpc(self, 
                             content: str, 
                             embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None) -> Tuple[np.ndarray, float]:
        """
        Process content through the HPC-QR pipeline to get embedding and QuickRecal score.
        
        Args:
            content: Content text
            embedding: Optional pre-computed embedding
            
        Returns:
            Tuple of (processed_embedding, quickrecal_score)
        """
        try:
            # Get embedding from content if not provided
            if embedding is None:
                embedding_tensor = await self.hpc_manager.get_embedding(content)
            else:
                # Convert to tensor if needed
                if isinstance(embedding, list):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                elif isinstance(embedding, np.ndarray):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                else:
                    embedding_tensor = embedding
            
            # Process through HPC-QR
            processed_embedding, quickrecal_score = await self.hpc_manager.process_embedding(embedding_tensor)
            
            # Convert to numpy array if tensor
            if isinstance(processed_embedding, torch.Tensor):
                processed_embedding = processed_embedding.detach().cpu().numpy()
                
            return processed_embedding, float(quickrecal_score)
            
        except Exception as e:
            logger.error(f"Error processing with HPC-QR: {e}")
            # Return original embedding with default QuickRecal score
            if embedding is not None:
                if isinstance(embedding, torch.Tensor):
                    numpy_embedding = embedding.detach().cpu().numpy()
                elif isinstance(embedding, list):
                    numpy_embedding = np.array(embedding, dtype=np.float32)
                else:
                    numpy_embedding = embedding
                return numpy_embedding, 0.5
            else:
                # Create a random embedding if none provided
                random_embedding = np.random.randn(self.config['embedding_dim'])
                random_embedding = random_embedding / np.linalg.norm(random_embedding)
                return random_embedding, 0.5


# Create a convenient alias for backward compatibility
UnifiedMemoryStorage = EnhancedMemoryStorage