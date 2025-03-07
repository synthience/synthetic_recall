# core\embedding_comparator.py

```py
"""
LUCID RECALL PROJECT
Embedding Comparator

Provides standardized interfaces for generating embeddings
and comparing their similarity across memory components.
"""

import torch
import logging
import asyncio
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingComparator:
    """
    Provides standardized methods for embedding generation and comparison.
    
    This class serves as an interface layer for the HPC system, allowing
    different components to generate and compare embeddings consistently.
    """
    
    def __init__(self, hpc_client, embedding_dim: int = 384):
        """
        Initialize the embedding comparator.
        
        Args:
            hpc_client: HPC client for embedding generation
            embedding_dim: Embedding dimension
        """
        self.hpc_client = hpc_client
        self.embedding_dim = embedding_dim
        self._embedding_cache = {}
        self._cache_limit = 1000
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self.stats = {
            'embeddings_generated': 0,
            'embeddings_normalized': 0,
            'comparisons_made': 0,
            'cache_hits': 0
        }
        
        logger.info(f"Initialized EmbeddingComparator with dim={embedding_dim}")
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Generate embedding for text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor or None on failure
        """
        # Check cache first
        cache_key = text.strip()
        if cache_key in self._embedding_cache:
            self.stats['cache_hits'] += 1
            return self._embedding_cache[cache_key]
        
        try:
            # Get embedding through HPC client
            embedding = await self.hpc_client.get_embedding(text)
            
            if embedding is None:
                logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                return None
            
            # Normalize embedding if needed
            embedding = self._normalize_embedding(embedding)
            
            # Cache the embedding
            async with self._lock:
                self._embedding_cache[cache_key] = embedding
                
                # Prune cache if needed
                if len(self._embedding_cache) > self._cache_limit:
                    # Remove oldest (first) item
                    oldest_key = next(iter(self._embedding_cache))
                    del self._embedding_cache[oldest_key]
            
            self.stats['embeddings_generated'] += 1
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _normalize_embedding(self, embedding: Union[torch.Tensor, np.ndarray, List[float]]) -> torch.Tensor:
        """
        Normalize embedding to unit vector.
        
        Args:
            embedding: Embedding to normalize
            
        Returns:
            Normalized embedding tensor
        """
        try:
            # Convert to torch tensor if not already
            if not isinstance(embedding, torch.Tensor):
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float()
                elif isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported embedding type: {type(embedding)}")
            
            # Ensure correct shape
            if len(embedding.shape) > 1 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            
            # Compute L2 norm
            norm = torch.norm(embedding, p=2)
            
            # Normalize if norm is non-zero
            if norm > 0:
                normalized = embedding / norm
            else:
                # If norm is zero, return original to avoid NaN
                normalized = embedding
                
            self.stats['embeddings_normalized'] += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            # Return original embedding as fallback
            if isinstance(embedding, torch.Tensor):
                return embedding
            elif isinstance(embedding, np.ndarray):
                return torch.from_numpy(embedding).float()
            elif isinstance(embedding, list):
                return torch.tensor(embedding, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported embedding type: {type(embedding)}")
    
    async def compare(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compare two embeddings and return similarity score.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Normalize embeddings if necessary
            embedding1 = self._normalize_embedding(embedding1)
            embedding2 = self._normalize_embedding(embedding2)
            
            # Ensure correct shapes for dot product
            if len(embedding1.shape) > 1:
                embedding1 = embedding1.squeeze()
            if len(embedding2.shape) > 1:
                embedding2 = embedding2.squeeze()
                
            # Cosine similarity (dot product of normalized vectors)
            similarity = torch.dot(embedding1, embedding2).item()
            
            # Ensure result is in valid range
            similarity = max(0.0, min(1.0, similarity))
            
            self.stats['comparisons_made'] += 1
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    async def batch_compare(self, query_embedding: torch.Tensor, 
                          embeddings: List[torch.Tensor]) -> List[float]:
        """
        Compare query embedding against multiple embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: List of embeddings to compare against
            
        Returns:
            List of similarity scores (0.0-1.0)
        """
        try:
            # Normalize query embedding
            query_embedding = self._normalize_embedding(query_embedding)
            
            # Calculate similarities for each embedding
            similarities = []
            for emb in embeddings:
                similarity = await self.compare(query_embedding, emb)
                similarities.append(similarity)
                
            self.stats['comparisons_made'] += len(embeddings)
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch comparison: {e}")
            return [0.0] * len(embeddings)
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        async with self._lock:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comparator statistics."""
        return {
            'embeddings_generated': self.stats['embeddings_generated'],
            'embeddings_normalized': self.stats['embeddings_normalized'],
            'comparisons_made': self.stats['comparisons_made'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self._embedding_cache),
            'cache_limit': self._cache_limit,
            'cache_utilization': len(self._embedding_cache) / self._cache_limit
        }
```

# core\integration\__init__.py

```py
"""
LUCID RECALL PROJECT
Memory Module

The core memory system for Lucidia with hierarchical architecture
for efficient, self-organizing memory.
"""

__version__ = "0.2.0"

# Import core components
from memory_types import MemoryTypes, MemoryEntry
from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from memory_prioritization_layer import MemoryPrioritizationLayer
from embedding_comparator import EmbeddingComparator
from memory_persistence_handler import MemoryPersistenceHandler
from enhanced_memory_core import EnhancedMemoryCore
from memory_integration import MemoryIntegration
from updated_hpc_client import EnhancedHPCClient

# Export public API
__all__ = [
    'MemoryTypes', 
    'MemoryEntry',
    'ShortTermMemory',
    'LongTermMemory',
    'MemoryPrioritizationLayer',
    'EmbeddingComparator',
    'MemoryPersistenceHandler',
    'EnhancedMemoryCore',
    'MemoryIntegration',
    'EnhancedHPCClient'
]

def create_memory_system(config=None):
    """
    Factory function to create a pre-configured memory system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MemoryIntegration instance
    """
    return MemoryIntegration(config)
```

# core\integration\hpc_sig_flow_manager.py

```py
"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 12:08 AM EST

HPC-SIG Flow Manager: Handles hypersphere processing chain and significance calculation
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCSIGFlowManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'chunk_size': 384,  # Match embedding dimension
            'embedding_dim': 768,
            'batch_size': 32,
            'momentum': 0.9,
            'diversity_threshold': 0.7,
            'surprise_threshold': 0.8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.momentum_buffer = None
        self.current_batch = []
        self.batch_timestamps = []
        
        logger.info(f"Initialized HPCSIGFlowManager with config: {self.config}")
        
    async def process_embedding(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single embedding through the HPC pipeline"""
        with torch.no_grad():
            # Log input shape
            logger.info(f"Input embedding shape: {embedding.shape}")
            
            # Move to correct device
            embedding = embedding.to(self.config['device'])
            
            # Ensure correct shape
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()[:384]
                logger.info(f"Flattened embedding shape: {embedding.shape}")
            
            # Project to unit hypersphere
            norm = torch.norm(embedding, p=2, dim=-1, keepdim=True)
            normalized = embedding / (norm + 1e-8)
            logger.info(f"Normalized embedding shape: {normalized.shape}")
            
            # Calculate surprise if we have momentum
            surprise_score = 0.0
            if self.momentum_buffer is not None:
                surprise_score = self._compute_surprise(normalized)
                logger.info(f"Calculated surprise score: {surprise_score}")
                
                # Apply shock absorber if surprise is high
                if surprise_score > self.config['surprise_threshold']:
                    normalized = self._apply_shock_absorber(normalized)
                    logger.info("Applied shock absorber")
            
            # Update momentum buffer
            self._update_momentum(normalized)
            
            # Calculate significance score
            significance = self._calculate_significance(normalized, surprise_score)
            logger.info(f"Calculated significance score: {significance}")
            
            return normalized, significance
            
    def _compute_surprise(self, embedding: torch.Tensor) -> float:
        """Calculate surprise score based on momentum buffer"""
        if self.momentum_buffer is None:
            return 0.0
            
        similarity = torch.matmul(embedding, self.momentum_buffer.T)
        return 1.0 - torch.mean(similarity).item()
        
    def _apply_shock_absorber(self, embedding: torch.Tensor) -> torch.Tensor:
        """Smooth out high-surprise embeddings"""
        if self.momentum_buffer is None:
            return embedding
            
        alpha = 1.0 - self.config['momentum']
        absorbed = alpha * embedding + (1 - alpha) * self.momentum_buffer[-1:]
        
        # Re-normalize
        norm = torch.norm(absorbed, p=2, dim=-1, keepdim=True)
        return absorbed / (norm + 1e-8)
        
    def _update_momentum(self, embedding: torch.Tensor):
        """Update momentum buffer with new embedding"""
        if self.momentum_buffer is None:
            self.momentum_buffer = embedding
        else:
            combined = torch.cat([self.momentum_buffer, embedding])
            self.momentum_buffer = combined[-self.config['chunk_size']:]
            
    def _calculate_significance(self, embedding: torch.Tensor, surprise: float) -> float:
        """Calculate significance score for memory storage"""
        magnitude = torch.norm(embedding).item()
        
        if self.momentum_buffer is not None:
            diversity = 1.0 - torch.max(torch.matmul(embedding, self.momentum_buffer.T)).item()
        else:
            diversity = 1.0
            
        # Combine factors (weights can be tuned)
        significance = (
            0.4 * surprise +
            0.3 * magnitude +
            0.3 * diversity
        )
        
        return significance
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current state statistics"""
        return {
            'has_momentum': self.momentum_buffer is not None,
            'momentum_size': len(self.momentum_buffer) if self.momentum_buffer is not None else 0,
            'device': self.config['device']
        }

```

# core\integration\memory_integration.py

```py
"""
LUCID RECALL PROJECT
Memory Integration Layer

Provides a user-friendly integration layer for the new memory architecture
with compatibility for existing client code.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
import torch

from embedding_comparator import EmbeddingComparator
from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from memory_prioritization_layer import MemoryPrioritizationLayer
from enhanced_memory_core import EnhancedMemoryCore

logger = logging.getLogger(__name__)

class MemoryIntegration:
    """
    User-friendly integration layer for the enhanced memory architecture.
    
    Provides simplified interfaces for common memory operations while
    abstracting away the complexity of the underlying memory system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory integration layer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Create enhanced memory core
        self.memory_core = EnhancedMemoryCore(self.config)
        
        # Create direct references to components for advanced usage
        self.short_term_memory = self.memory_core.short_term_memory
        self.long_term_memory = self.memory_core.long_term_memory
        self.memory_prioritization = self.memory_core.memory_prioritization
        self.hpc_manager = self.memory_core.hpc_manager
        
        # Create embedding comparator for convenience
        self.embedding_comparator = EmbeddingComparator(
            hpc_client=self.hpc_manager,
            embedding_dim=self.config.get('embedding_dim', 384)
        )
        
        # Simple query cache for frequent identical queries
        self._query_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("Memory integration layer initialized")
    
    async def store(self, content: str, metadata: Optional[Dict[str, Any]] = None,
                  importance: Optional[float] = None) -> Dict[str, Any]:
        """
        Store content in memory with automatic significance calculation.
        
        Args:
            content: Content text to store
            metadata: Optional metadata
            importance: Optional importance override (0.0-1.0)
            
        Returns:
            Dict with store result
        """
        try:
            # Determine memory type from metadata
            if metadata and 'type' in metadata:
                memory_type_str = metadata['type'].upper()
                try:
                    from memory_types import MemoryTypes
                    memory_type = MemoryTypes[memory_type_str]
                except (KeyError, ImportError):
                    memory_type = None  # Will use default EPISODIC
            else:
                memory_type = None
                
            # Use provided importance if available
            if importance is not None:
                if metadata is None:
                    metadata = {}
                metadata['significance'] = max(0.0, min(1.0, importance))
            
            # Store in memory system
            from memory_types import MemoryTypes
            result = await self.memory_core.process_and_store(
                content=content,
                memory_type=memory_type or MemoryTypes.EPISODIC,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def recall(self, query: str, limit: int = 5, min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Recall memories related to query.
        
        Args:
            query: Query text
            limit: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memories
        """
        # Check cache for identical recent queries
        cache_key = f"{query.strip()}:{limit}:{min_importance}"
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                return cache_entry['results']
        
        try:
            # Retrieve memories through prioritization layer
            memories = await self.memory_core.retrieve_memories(
                query=query,
                limit=limit,
                min_significance=min_importance
            )
            
            # Cache results
            self._query_cache[cache_key] = {
                'timestamp': time.time(),
                'results': memories
            }
            
            # Clean old cache entries
            self._clean_cache()
            
            return memories
            
        except Exception as e:
            logger.error(f"Error recalling memories: {e}")
            return []
    
    async def generate_context(self, query: str, max_tokens: int = 512) -> str:
        """
        Generate memory context for LLM consumption.
        
        Args:
            query: The query to generate context for
            max_tokens: Maximum context tokens to generate
            
        Returns:
            Formatted context string
        """
        try:
            # Estimate characters per token (rough approximation)
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            
            # Retrieve relevant memories
            memories = await self.recall(
                query=query,
                limit=10,  # Get more than needed to select most relevant
                min_importance=0.3  # Only include somewhat important memories
            )
            
            if not memories:
                return ""
                
            # Format memories into context
            context_parts = ["# Relevant Memory Context:"]
            total_chars = len(context_parts[0])
            
            for i, memory in enumerate(memories):
                content = memory.get('content', '')
                timestamp = memory.get('timestamp', 0)
                
                # Format timestamp
                import datetime
                date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else ""
                
                # Create memory entry
                entry = f"Memory {i+1} ({date_str}): {content}"
                
                # Check if adding this would exceed limit
                if total_chars + len(entry) + 2 > max_chars:
                    # Add truncation notice if we can't fit all memories
                    if i < len(memories):
                        context_parts.append(f"... plus {len(memories) - i} more memories (truncated)")
                    break
                
                # Add to context
                context_parts.append(entry)
                total_chars += len(entry) + 2  # +2 for newlines
            
            # Join with newlines
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return ""
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory dict or None if not found
        """
        try:
            return await self.memory_core.get_memory_by_id(memory_id)
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return None
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: Memory ID
            updates: Dictionary of fields to update
            
        Returns:
            Success status
        """
        try:
            # Check if memory exists in STM first
            memory = self.short_term_memory.get_memory_by_id(memory_id)
            
            if memory:
                # Update memory in place
                for key, value in updates.items():
                    if key != 'id':  # Don't allow changing ID
                        memory[key] = value
                return True
                
            # If not in STM, check LTM
            if hasattr(self.long_term_memory, 'update_memory'):
                # If LTM has update method, use it
                return await self.long_term_memory.update_memory(memory_id, updates)
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False
    
    async def backup(self) -> bool:
        """
        Force memory backup.
        
        Returns:
            Success status
        """
        try:
            return await self.memory_core.force_backup()
        except Exception as e:
            logger.error(f"Error backing up memories: {e}")
            return False
    
    def _clean_cache(self) -> None:
        """Clean expired entries from query cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._query_cache.items()
            if current_time - entry['timestamp'] > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._query_cache[key]
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor or None on failure
        """
        try:
            return await self.embedding_comparator.get_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    async def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts for semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            embedding1 = await self.embedding_comparator.get_embedding(text1)
            embedding2 = await self.embedding_comparator.get_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                return 0.0
                
            return await self.embedding_comparator.compare(embedding1, embedding2)
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system-wide memory statistics."""
        try:
            # Get detailed stats from core
            core_stats = self.memory_core.get_stats()
            
            # Add integration layer stats
            integration_stats = {
                'cache_size': len(self._query_cache),
                'cache_ttl': self._cache_ttl
            }
            
            # Get embedding comparator stats
            comparator_stats = self.embedding_comparator.get_stats()
            
            return {
                'core': core_stats,
                'integration': integration_stats,
                'comparator': comparator_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
```

# core\integration\updated_hpc_client.py

```py
"""
LUCID RECALL PROJECT
Enhanced HPC Client

Client for interacting with the HPC server with robust connectivity
and optimized processing.
"""

import asyncio
import websockets
import json
import logging
import time
import torch
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class EnhancedHPCClient:
    """
    Enhanced client for interacting with the HPC server.
    
    Features:
    - Robust connection handling
    - Request batching
    - Result caching
    - Embedding management
    - Significance calculation
    """
    
    def __init__(self, 
                 server_url: str = "ws://localhost:5005",
                 connection_timeout: float = 10.0,
                 request_timeout: float = 30.0,
                 max_retries: int = 3,
                 ping_interval: float = 15.0,
                 embedding_dim: int = 384):
        """
        Initialize the HPC client.
        
        Args:
            server_url: WebSocket URL for the HPC server
            connection_timeout: Timeout for connection attempts
            request_timeout: Timeout for requests
            max_retries: Maximum number of retries for failed requests
            ping_interval: Interval for ping messages to keep connection alive
            embedding_dim: Dimension of embeddings
        """
        self.server_url = server_url
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.ping_interval = ping_interval
        self.embedding_dim = embedding_dim
        
        # Connection state
        self.connection = None
        self._connecting = False
        self._connection_lock = asyncio.Lock()
        self._last_activity = 0
        self._request_id = 0
        
        # Pending requests
        self._pending_requests = {}
        self._request_timeout_tasks = {}
        
        # Result cache
        self._result_cache = {}
        self._cache_max_size = 1000
        self._cache_ttl = 3600  # 1 hour
        
        # Background tasks
        self._heartbeat_task = None
        
        # Stats
        self.stats = {
            'connect_count': 0,
            'disconnect_count': 0,
            'request_count': 0,
            'error_count': 0,
            'cache_hits': 0,
            'retry_count': 0,
            'timeout_count': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"Initialized EnhancedHPCClient with server_url={server_url}")
    
    async def connect(self) -> bool:
        """
        Connect to the HPC server with robust error handling.
        
        Returns:
            Success status
        """
        # Use lock to prevent multiple concurrent connection attempts
        async with self._connection_lock:
            # Check if already connected
            if self.connection and not self.connection.closed:
                return True
                
            # Check if connection attempt is already in progress
            if self._connecting:
                logger.debug("Connection attempt already in progress, waiting...")
                for _ in range(20):  # Wait up to 2 seconds
                    await asyncio.sleep(0.1)
                    if self.connection and not self.connection.closed:
                        return True
                return False
                
            # Mark as connecting
            self._connecting = True
            
            try:
                self.stats['connect_count'] += 1
                
                # Attempt connection with timeout
                try:
                    self.connection = await asyncio.wait_for(
                        websockets.connect(self.server_url),
                        timeout=self.connection_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Connection to {self.server_url} timed out")
                    self._connecting = False
                    return False
                
                # Update activity timestamp
                self._last_activity = time.time()
                
                # Start heartbeat task if needed
                if not self._heartbeat_task or self._heartbeat_task.done():
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Start message handler
                asyncio.create_task(self._message_handler())
                
                logger.info(f"Connected to HPC server at {self.server_url}")
                self._connecting = False
                return True
                
            except Exception as e:
                logger.error(f"Error connecting to HPC server: {e}")
                self._connecting = False
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from the HPC server."""
        async with self._connection_lock:
            if self.connection and not self.connection.closed:
                try:
                    await self.connection.close()
                    self.stats['disconnect_count'] += 1
                    logger.info("Disconnected from HPC server")
                except Exception as e:
                    logger.error(f"Error disconnecting from HPC server: {e}")
            
            # Cancel heartbeat task
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            
            # Clear connection
            self.connection = None
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                
                # Check if connection is still open
                if not self.connection or self.connection.closed:
                    break
                    
                # Check inactivity
                if time.time() - self._last_activity > self.ping_interval:
                    try:
                        # Send ping to check connection
                        pong_waiter = await self.connection.ping()
                        await asyncio.wait_for(pong_waiter, timeout=5.0)
                        self._last_activity = time.time()
                        
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        logger.warning("Ping failed, reconnecting...")
                        await self.disconnect()
                        await self.connect()
                        break
                    
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
    
    async def _message_handler(self) -> None:
        """Handle incoming messages from the server."""
        if not self.connection:
            return
            
        try:
            async for message in self.connection:
                self._last_activity = time.time()
                
                # Update stats
                self.stats['bytes_received'] += len(message)
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Check if this is a response to a pending request
                    request_id = data.get('request_id')
                    if request_id and request_id in self._pending_requests:
                        # Get future for this request
                        future = self._pending_requests.pop(request_id)
                        
                        # Cancel timeout task if exists
                        timeout_task = self._request_timeout_tasks.pop(request_id, None)
                        if timeout_task:
                            timeout_task.cancel()
                            
                        # Set result
                        if not future.done():
                            future.set_result(data)
                            
                    else:
                        logger.warning(f"Received message for unknown request: {request_id}")
                    
                except json.JSONDecodeError:
                    logger.error("Received invalid JSON")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
            await self.disconnect()
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await self.disconnect()
    
    async def _request_timeout_handler(self, request_id: str) -> None:
        """Handle request timeout."""
        try:
            # Wait for request timeout
            await asyncio.sleep(self.request_timeout)
            
            # Check if request is still pending
            if request_id in self._pending_requests:
                # Get future
                future = self._pending_requests.pop(request_id)
                
                # Set exception if not done
                if not future.done():
                    self.stats['timeout_count'] += 1
                    future.set_exception(asyncio.TimeoutError(f"Request {request_id} timed out"))
                    
                # Remove from pending requests
                self._request_timeout_tasks.pop(request_id, None)
                
        except asyncio.CancelledError:
            # Task was cancelled (this is expected when request completes normally)
            pass
        except Exception as e:
            logger.error(f"Error in timeout handler: {e}")
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor or None on failure
        """
        # Check cache
        cache_key = f"embed:{text}"
        if cache_key in self._result_cache:
            cache_entry = self._result_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                self.stats['cache_hits'] += 1
                return cache_entry['result']
        
        # Send request to get embedding
        response = await self._send_request(
            request_type='embed',
            request_data={'text': text}
        )
        
        if not response:
            return None
            
        # Extract embedding from response
        embedding_data = response.get('data', {}).get('embedding')
        if not embedding_data:
            embedding_data = response.get('embedding')
            
        if not embedding_data:
            logger.error("No embedding in response")
            return None
            
        # Convert to tensor
        try:
            embedding = torch.tensor(embedding_data, dtype=torch.float32)
            
            # Cache result
            self._result_cache[cache_key] = {
                'timestamp': time.time(),
                'result': embedding
            }
            
            # Prune cache if needed
            if len(self._result_cache) > self._cache_max_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._result_cache.keys(), 
                    key=lambda k: self._result_cache[k]['timestamp']
                )
                for key in sorted_keys[:len(sorted_keys) // 10]:  # Remove oldest 10%
                    del self._result_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            return None
    
    async def process_embedding(self, embedding: Union[torch.Tensor, List[float]]) -> Tuple[torch.Tensor, float]:
        """
        Process an embedding through the HPC pipeline.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Tuple of (processed_embedding, significance)
        """
        # Convert to list if tensor
        if isinstance(embedding, torch.Tensor):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding
        
        # Send request to process embedding
        response = await self._send_request(
            request_type='process',
            request_data={'embeddings': embedding_list}
        )
        
        if not response:
            # Return original embedding with default significance on failure
            if isinstance(embedding, torch.Tensor):
                return embedding, 0.5
            else:
                return torch.tensor(embedding, dtype=torch.float32), 0.5
                
        # Extract processed embedding and significance
        processed_data = response.get('data', {}).get('embeddings')
        if not processed_data:
            processed_data = response.get('embeddings')
            
        significance = response.get('data', {}).get('significance')
        if significance is None:
            significance = response.get('significance', 0.5)
            
        # Convert to tensor and return
        try:
            processed_embedding = torch.tensor(processed_data, dtype=torch.float32)
            return processed_embedding, float(significance)
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            
            # Return original embedding with default significance on error
            if isinstance(embedding, torch.Tensor):
                return embedding, 0.5
            else:
                return torch.tensor(embedding, dtype=torch.float32), 0.5
    
    async def fetch_relevant_embeddings(self, query_embedding: torch.Tensor, 
                                      limit: int = 10, 
                                      min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Fetch relevant embeddings based on query embedding.
        
        Args:
            query_embedding: Query embedding
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of relevant memories
        """
        # Convert to list if tensor
        if isinstance(query_embedding, torch.Tensor):
            embedding_list = query_embedding.tolist()
        else:
            embedding_list = query_embedding
        
        # Send request to search
        response = await self._send_request(
            request_type='search',
            request_data={
                'embedding': embedding_list,
                'limit': limit,
                'min_significance': min_significance
            }
        )
        
        if not response:
            return []
            
        # Extract results
        results = response.get('data', {}).get('results')
        if not results:
            results = response.get('results', [])
            
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get HPC server stats.
        
        Returns:
            Dict with server stats
        """
        response = await self._send_request(
            request_type='stats',
            request_data={}
        )
        
        if not response:
            return {}
            
        # Extract stats
        server_stats = response.get('data', {})
        if not server_stats:
            server_stats = response
            
        # Combine with client stats
        return {
            'server': server_stats,
            'client': self.stats
        }
    
    async def _send_request(self, request_type: str, request_data: Dict[str, Any],
                          retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Send request to server with retry logic.
        
        Args:
            request_type: Type of request
            request_data: Request data
            retry_count: Current retry count
            
        Returns:
            Response dict or None on failure
        """
        # Check connection
        if not self.connection or self.connection.closed:
            success = await self.connect()
            if not success:
                logger.error("Failed to connect to HPC server")
                return None
        
        # Generate request ID
        self._request_id += 1
        request_id = f"{int(time.time())}:{self._request_id}"
        
        # Create request
        request = {
            'type': request_type,
            'request_id': request_id,
            'timestamp': time.time(),
            **request_data
        }
        
        # Serialize request
        try:
            request_json = json.dumps(request)
        except Exception as e:
            logger.error(f"Error serializing request: {e}")
            return None
        
        # Update stats
        self.stats['request_count'] += 1
        self.stats['bytes_sent'] += len(request_json)
        
        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future
        
        # Create timeout task
        timeout_task = asyncio.create_task(self._request_timeout_handler(request_id))
        self._request_timeout_tasks[request_id] = timeout_task
        
        # Track start time
        start_time = time.time()
        
        try:
            # Send request
            await self.connection.send(request_json)
            self._last_activity = time.time()
            
            # Wait for response
            response = await response_future
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update average response time
            if self.stats['request_count'] > 1:
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['request_count'] - 1) + response_time) / 
                    self.stats['request_count']
                )
            else:
                self.stats['avg_response_time'] = response_time
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            
            # Retry if not exceeded max retries
            if retry_count < self.max_retries:
                self.stats['retry_count'] += 1
                logger.info(f"Retrying request (attempt {retry_count + 1}/{self.max_retries})")
                
                # Clean up
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
                    
                # Reconnect
                await self.disconnect()
                await self.connect()
                
                # Retry with increased count
                return await self._send_request(request_type, request_data, retry_count + 1)
                
            self.stats['error_count'] += 1
            return None
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed during request")
            
            # Retry if not exceeded max retries
            if retry_count < self.max_retries:
                self.stats['retry_count'] += 1
                logger.info(f"Retrying request (attempt {retry_count + 1}/{self.max_retries})")
                
                # Clean up
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
                
                # Reconnect
                await self.disconnect()
                await self.connect()
                
                # Retry with increased count
                return await self._send_request(request_type, request_data, retry_count + 1)
                
            self.stats['error_count'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            
            # Clean up
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
                
            if request_id in self._request_timeout_tasks:
                timeout_task = self._request_timeout_tasks.pop(request_id)
                if not timeout_task.done():
                    timeout_task.cancel()
            
            self.stats['error_count'] += 1
            return None
```

# core\long_term_memory.py

```py
"""
LUCID RECALL PROJECT
Long-Term Memory (LTM)

Persistent significance-weighted storage where only important memories remain long-term.
Implements dynamic significance decay to ensure only critical memories persist.
"""

import time
import math
import logging
import asyncio
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    Long-Term Memory with significance-weighted storage and dynamic decay.
    
    Stores memories persistently with significance weighting to ensure
    only important memories are retained long-term. Implements dynamic
    significance decay to allow unimportant memories to fade naturally.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the long-term memory system.
        
        Args:
            config: Configuration options
        """
        self.config = {
            'storage_path': os.path.join('memory', 'ltm_storage'),
            'significance_threshold': 0.7,  # Minimum significance for storage
            'max_memories': 10000,          # Maximum number of memories to store
            'decay_rate': 0.05,             # Base decay rate (per day)
            'decay_check_interval': 86400,  # Time between decay checks (1 day)
            'min_retention_time': 604800,   # Minimum retention time regardless of decay (1 week)
            'embedding_dim': 384,           # Embedding dimension
            'enable_persistence': True,     # Whether to persist memories to disk
            'purge_threshold': 0.3,         # Memories below this significance get purged
            **(config or {})
        }
        
        # Ensure storage path exists
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory storage
        self.memories = {}  # ID -> Memory
        self.memory_index = {}  # Category -> List of IDs
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Performance stats
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'purges': 0,
            'hits': 0,
            'last_decay_check': time.time(),
            'last_backup': time.time()
        }
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Initialized LongTermMemory with {len(self.memories)} memories")
        
    def _load_memories(self):
        """Load memories from persistent storage."""
        if not self.config['enable_persistence']:
            return
            
        try:
            logger.info(f"Loading memories from {self.storage_path}")
            
            # List memory files
            memory_files = list(self.storage_path.glob('*.json'))
            
            if not memory_files:
                logger.info("No memory files found")
                return
                
            # Load each memory file
            for file_path in memory_files:
                try:
                    with open(file_path, 'r') as f:
                        memory = json.load(f)
                    
                    # Validate memory
                    if not all(k in memory for k in ['id', 'content', 'timestamp']):
                        logger.warning(f"Invalid memory format in {file_path}, skipping")
                        continue
                    
                    # Convert embedding from list to tensor if present
                    if 'embedding' in memory and isinstance(memory['embedding'], list):
                        memory['embedding'] = torch.tensor(
                            memory['embedding'], 
                            dtype=torch.float32
                        )
                    
                    # Add to memories
                    memory_id = memory['id']
                    self.memories[memory_id] = memory
                    
                    # Add to index by category
                    category = memory.get('metadata', {}).get('category', 'general')
                    if category not in self.memory_index:
                        self.memory_index[category] = []
                    self.memory_index[category].append(memory_id)
                    
                except Exception as e:
                    logger.error(f"Error loading memory from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.memories)} memories")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    async def store_memory(self, content: str, embedding: Optional[torch.Tensor] = None,
                         significance: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Store a memory in long-term storage if it meets significance threshold.
        
        Args:
            content: The memory content text
            embedding: Optional pre-computed embedding
            significance: Memory significance (0.0-1.0)
            metadata: Optional additional metadata
            
        Returns:
            Memory ID if stored, None if rejected due to low significance
        """
        # Check significance threshold
        if significance < self.config['significance_threshold']:
            logger.debug(f"Memory significance {significance} below threshold {self.config['significance_threshold']}, not storing")
            return None
        
        async with self._lock:
            # Generate memory ID
            import uuid
            memory_id = str(uuid.uuid4())
            
            # Set current timestamp
            timestamp = time.time()
            
            # Create memory object
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'timestamp': timestamp,
                'significance': significance,
                'metadata': metadata or {},
                'access_count': 0,
                'last_access': timestamp
            }
            
            # Store in memory dictionary
            self.memories[memory_id] = memory
            
            # Update category index
            category = metadata.get('category', 'general') if metadata else 'general'
            if category not in self.memory_index:
                self.memory_index[category] = []
            self.memory_index[category].append(memory_id)
            
            # Update stats
            self.stats['stores'] += 1
            
            # Persist to disk if enabled
            if self.config['enable_persistence']:
                await self._persist_memory(memory)
            
            # Check if we need to run decay and purging
            if len(self.memories) > self.config['max_memories']:
                asyncio.create_task(self._run_decay_and_purge())
            
            logger.info(f"Stored memory {memory_id} with significance {significance}")
            return memory_id
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            Memory dict or None if not found
        """
        async with self._lock:
            self.stats['retrievals'] += 1
            
            if memory_id not in self.memories:
                return None
            
            # Get memory
            memory = self.memories[memory_id]
            
            # Update access stats
            memory['access_count'] = memory.get('access_count', 0) + 1
            memory['last_access'] = time.time()
            
            # Update memory significance based on access
            self._boost_significance(memory)
            
            self.stats['hits'] += 1
            
            # Return a copy to prevent modification
            import copy
            return copy.deepcopy(memory)
    
    async def search_memory(self, query: str, limit: int = 5, 
                          min_significance: float = 0.0,
                          categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for memories based on text content.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            categories: Optional list of categories to search within
            
        Returns:
            List of matching memories
        """
        async with self._lock:
            self.stats['retrievals'] += 1
            
            # Simple text search for now
            # In a real implementation, you'd use embeddings for semantic search
            results = []
            
            # Filter by categories if provided
            memory_ids = []
            if categories:
                for category in categories:
                    memory_ids.extend(self.memory_index.get(category, []))
            else:
                memory_ids = list(self.memories.keys())
            
            # Search through memories
            for memory_id in memory_ids:
                memory = self.memories[memory_id]
                
                # Check significance threshold
                if memory.get('significance', 0) < min_significance:
                    continue
                
                # Calculate simple text match score
                content = memory.get('content', '').lower()
                query_lower = query.lower()
                
                # Basic token overlap for matching
                tokens_content = set(content.split())
                tokens_query = set(query_lower.split())
                
                if tokens_content and tokens_query:
                    intersection = tokens_content.intersection(tokens_query)
                    union = tokens_content.union(tokens_query)
                    similarity = len(intersection) / len(union)
                else:
                    similarity = 0.0
                
                # Calculate effective significance with decay
                effective_significance = self._calculate_effective_significance(memory)
                
                # Combine similarity and significance for final score
                score = (similarity * 0.7) + (effective_significance * 0.3)
                
                # Add to results if score is positive
                if score > 0:
                    results.append({
                        'id': memory_id,
                        'content': memory.get('content', ''),
                        'timestamp': memory.get('timestamp', 0),
                        'similarity': similarity,
                        'significance': effective_significance,
                        'score': score,
                        'metadata': memory.get('metadata', {})
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Update hit stats
            if results:
                self.stats['hits'] += 1
            
            # Return top results
            return results[:limit]
    
    def _boost_significance(self, memory: Dict[str, Any]) -> None:
        """
        Boost memory significance based on access patterns.
        
        Args:
            memory: The memory to boost
        """
        # Get access information
        access_count = memory.get('access_count', 1)
        
        # Calculate recency factor (higher for more recent access)
        current_time = time.time()
        last_access = memory.get('last_access', memory.get('timestamp', current_time))
        days_since_access = (current_time - last_access) / 86400  # Convert to days
        recency_factor = math.exp(-0.1 * days_since_access)  # Exponential decay with time
        
        # Calculate access factor (higher for frequently accessed memories)
        access_factor = min(1.0, access_count / 10)  # Cap at 10 accesses
        
        # Calculate boost amount (higher for recently and frequently accessed memories)
        boost_amount = 0.05 * recency_factor * access_factor
        
        # Apply boost with cap at 1.0
        memory['significance'] = min(1.0, memory.get('significance', 0.5) + boost_amount)
    
    def _calculate_effective_significance(self, memory: Dict[str, Any]) -> float:
        """
        Calculate effective significance with time decay applied.
        
        Args:
            memory: The memory to evaluate
            
        Returns:
            Effective significance after decay
        """
        # Get base significance and timestamp
        base_significance = memory.get('significance', 0.5)
        timestamp = memory.get('timestamp', time.time())
        
        # Calculate age in days
        current_time = time.time()
        age_days = (current_time - timestamp) / 86400  # Convert to days
        
        # Skip recent memories (retention period)
        min_retention_days = self.config['min_retention_time'] / 86400
        if age_days < min_retention_days:
            return base_significance
        
        # Calculate importance factor (more important memories decay slower)
        importance_factor = 0.5 + (0.5 * base_significance)
        
        # Calculate effective decay rate (decay slower for important memories)
        effective_decay_rate = self.config['decay_rate'] / importance_factor
        
        # Apply exponential decay
        decay_factor = math.exp(-effective_decay_rate * (age_days - min_retention_days))
        effective_significance = base_significance * decay_factor
        
        return effective_significance
    
    async def _run_decay_and_purge(self) -> None:
        """Run decay calculations and purge low-significance memories."""
        # Only one instance should run at a time
        async with self._lock:
            current_time = time.time()
            
            # Check if it's time to run decay
            time_since_last_decay = current_time - self.stats['last_decay_check']
            if time_since_last_decay < self.config['decay_check_interval']:
                # Not time yet
                return
            
            logger.info("Running memory decay and purge")
            
            # Calculate effective significance for each memory
            memories_with_significance = []
            for memory_id, memory in self.memories.items():
                effective_significance = self._calculate_effective_significance(memory)
                memories_with_significance.append((memory_id, effective_significance))
            
            # Sort by effective significance (ascending)
            memories_with_significance.sort(key=lambda x: x[1])
            
            # Determine how many to purge
            excess_count = len(self.memories) - self.config['max_memories']
            purge_count = max(excess_count, 0)
            
            # Also purge memories below threshold
            purge_ids = [memory_id for memory_id, significance in memories_with_significance 
                       if significance < self.config['purge_threshold']]
            
            # Ensure we don't purge too many
            if len(purge_ids) > purge_count:
                purge_ids = purge_ids[:purge_count]
            
            # Purge selected memories
            for memory_id in purge_ids:
                await self._purge_memory(memory_id)
            
            # Update stats
            self.stats['purges'] += len(purge_ids)
            self.stats['last_decay_check'] = current_time
            
            logger.info(f"Purged {len(purge_ids)} memories")
    
    async def _purge_memory(self, memory_id: str) -> None:
        """
        Purge a memory from storage.
        
        Args:
            memory_id: ID of memory to purge
        """
        if memory_id not in self.memories:
            return
        
        # Get memory for logging
        memory = self.memories[memory_id]
        significance = memory.get('significance', 0)
        age_days = (time.time() - memory.get('timestamp', 0)) / 86400
        
        logger.debug(f"Purging memory {memory_id} with significance {significance} (age: {age_days:.1f} days)")
        
        # Remove from memory dictionary
        del self.memories[memory_id]
        
        # Remove from category index
        category = memory.get('metadata', {}).get('category', 'general')
        if category in self.memory_index and memory_id in self.memory_index[category]:
            self.memory_index[category].remove(memory_id)
        
        # Remove from disk if persistence enabled
        if self.config['enable_persistence']:
            file_path = self.storage_path / f"{memory_id}.json"
            if file_path.exists():
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing memory file {file_path}: {e}")
    
    async def _persist_memory(self, memory: Dict[str, Any]) -> None:
        """
        Persist a memory to disk.
        
        Args:
            memory: Memory dict to persist
        """
        if not self.config['enable_persistence']:
            return
        
        try:
            memory_id = memory['id']
            file_path = self.storage_path / f"{memory_id}.json"
            
            # Convert embedding to list if it's a tensor
            memory_copy = memory.copy()
            if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                memory_copy['embedding'] = memory_copy['embedding'].tolist()
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(memory_copy, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
    
    async def backup(self) -> bool:
        """
        Create a backup of all memories.
        
        Returns:
            Success status
        """
        if not self.config['enable_persistence']:
            return False
        
        async with self._lock:
            try:
                # Create backup directory
                backup_dir = self.storage_path / 'backups'
                backup_dir.mkdir(exist_ok=True)
                
                # Create timestamped backup folder
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"backup_{timestamp}"
                backup_path.mkdir(exist_ok=True)
                
                # Copy all memory files
                for memory_id in self.memories:
                    source_path = self.storage_path / f"{memory_id}.json"
                    dest_path = backup_path / f"{memory_id}.json"
                    
                    if source_path.exists():
                        import shutil
                        shutil.copy2(source_path, dest_path)
                
                # Update stats
                self.stats['last_backup'] = time.time()
                
                logger.info(f"Created backup at {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        # Calculate category distribution
        category_counts = {category: len(ids) for category, ids in self.memory_index.items()}
        
        # Calculate significance distribution
        significance_values = [memory.get('significance', 0) for memory in self.memories.values()]
        significance_bins = [0, 0, 0, 0, 0]  # 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        
        for sig in significance_values:
            bin_index = min(int(sig * 5), 4)
            significance_bins[bin_index] += 1
        
        significance_distribution = {
            '0.0-0.2': significance_bins[0],
            '0.2-0.4': significance_bins[1],
            '0.4-0.6': significance_bins[2],
            '0.6-0.8': significance_bins[3],
            '0.8-1.0': significance_bins[4]
        }
        
        # Gather stats
        return {
            'total_memories': len(self.memories),
            'categories': category_counts,
            'significance_distribution': significance_distribution,
            'stores': self.stats['stores'],
            'retrievals': self.stats['retrievals'],
            'hits': self.stats['hits'],
            'purges': self.stats['purges'],
            'last_decay_check': self.stats['last_decay_check'],
            'last_backup': self.stats['last_backup'],
            'hit_ratio': self.stats['hits'] / max(1, self.stats['retrievals']),
            'storage_utilization': len(self.memories) / self.config['max_memories']
        }

```

# core\memory_core.py

```py
"""
LUCID RECALL PROJECT
Enhanced Memory Core with Layered Memory Architecture

This enhanced memory core integrates STM, LTM, and MPL components
to provide a self-governing, adaptable, and efficient memory system.
"""

import torch
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import memory components
from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from memory_prioritization_layer import MemoryPrioritizationLayer
from hpc_sig_flow_manager import HPCSIGFlowManager
from memory_types import MemoryTypes, MemoryEntry

logger = logging.getLogger(__name__)

class EnhancedMemoryCore:
    """
    Enhanced Memory Core with layered memory architecture.
    
    This core implements a hierarchical memory system with:
    - Short-Term Memory (STM) for recent interactions
    - Long-Term Memory (LTM) for persistent storage
    - Memory Prioritization Layer (MPL) for optimal routing
    - HPC integration for deep retrieval and significance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced memory core.
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            'embedding_dim': 384,
            'max_memories': 10000,
            'memory_path': Path('/workspace/memory/stored'),
            'stm_max_size': 10,
            'significance_threshold': 0.3,
            'enable_persistence': True,
            'decay_rate': 0.05,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        logger.info(f"Initializing EnhancedMemoryCore with device={self.config['device']}")
        
        # Initialize HPC Manager for embeddings and significance
        self.hpc_manager = HPCSIGFlowManager({
            'embedding_dim': self.config['embedding_dim'],
            'device': self.config['device']
        })
        
        # Initialize memory layers
        self.short_term_memory = ShortTermMemory(
            max_size=self.config['stm_max_size'],
            embedding_comparator=self.hpc_manager
        )
        
        self.long_term_memory = LongTermMemory({
            'storage_path': self.config['memory_path'] / 'ltm',
            'significance_threshold': self.config['significance_threshold'],
            'max_memories': self.config['max_memories'],
            'decay_rate': self.config['decay_rate'],
            'embedding_dim': self.config['embedding_dim'],
            'enable_persistence': self.config['enable_persistence']
        })
        
        # Initialize Memory Prioritization Layer
        self.memory_prioritization = MemoryPrioritizationLayer(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            hpc_client=self.hpc_manager
        )
        
        # Thread safety
        self._processing_lock = asyncio.Lock()
        
        # Performance tracking
        self.start_time = time.time()
        self._processing_history = []
        self._max_history_items = 100
        self._total_processed = 0
        self._total_stored = 0
        self._total_time = 0.0
        
        logger.info("EnhancedMemoryCore initialized")
    
    async def process_and_store(self, content: str, memory_type: MemoryTypes = MemoryTypes.EPISODIC,
                              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content through the memory pipeline and store if significant.
        
        Args:
            content: Content text to process and store
            memory_type: Type of memory (EPISODIC, SEMANTIC, etc.)
            metadata: Additional metadata about the memory
            
        Returns:
            Dict with process result and memory ID if stored
        """
        async with self._processing_lock:
            start_time = time.time()
            self._total_processed += 1
            
            # Track processing stats
            processing_record = {
                'content_length': len(content),
                'memory_type': memory_type.value,
                'start_time': start_time
            }
            
            # Preprocess content (truncate if too long)
            if len(content) > 10000:  # Arbitrary limit for very long content
                logger.warning(f"Content too long ({len(content)} chars), truncating")
                content = content[:10000] + "... [truncated]"
            
            try:
                # Process through HPC for embedding and significance
                embedding, significance = await self.hpc_manager.process_embedding(
                    torch.tensor(content.encode(), dtype=torch.float32).reshape(1, -1)
                )
                
                processing_record['embedding_generated'] = True
                processing_record['significance'] = significance
                
                # Update metadata with significance
                full_metadata = metadata or {}
                full_metadata['significance'] = significance
                full_metadata['memory_type'] = memory_type.value
                full_metadata['timestamp'] = time.time()
                
                # Always store in STM for immediate recall
                stm_id = await self.short_term_memory.add_memory(
                    content=content,
                    embedding=embedding,
                    metadata=full_metadata
                )
                
                processing_record['stm_stored'] = True
                
                # Store in LTM if above significance threshold
                ltm_id = None
                if significance >= self.config['significance_threshold']:
                    ltm_id = await self.long_term_memory.store_memory(
                        content=content,
                        embedding=embedding,
                        significance=significance,
                        metadata=full_metadata
                    )
                    
                    processing_record['ltm_stored'] = True
                    self._total_stored += 1
                else:
                    processing_record['ltm_stored'] = False
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self._total_time += processing_time
                
                processing_record['processing_time'] = processing_time
                processing_record['success'] = True
                
                # Add to processing history with pruning
                self._processing_history.append(processing_record)
                if len(self._processing_history) > self._max_history_items:
                    self._processing_history = self._processing_history[-self._max_history_items:]
                
                return {
                    'success': True,
                    'stm_id': stm_id,
                    'ltm_id': ltm_id,
                    'significance': significance,
                    'processing_time': processing_time
                }
                
            except Exception as e:
                logger.error(f"Error processing and storing memory: {e}")
                
                processing_record['success'] = False
                processing_record['error'] = str(e)
                processing_record['processing_time'] = time.time() - start_time
                
                # Add to processing history even on failure
                self._processing_history.append(processing_record)
                if len(self._processing_history) > self._max_history_items:
                    self._processing_history = self._processing_history[-self._max_history_items:]
                
                return {
                    'success': False,
                    'error': str(e)
                }
    
    async def retrieve_memories(self, query: str, limit: int = 5, 
                             min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query using the Memory Prioritization Layer.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of memory results
        """
        try:
            # Route through MPL to get optimal retrieval path
            result = await self.memory_prioritization.route_query(query, {
                'limit': limit,
                'min_significance': min_significance
            })
            
            return result.get('memories', [])
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory dict or None if not found
        """
        # Check STM first (faster)
        memory = self.short_term_memory.get_memory_by_id(memory_id)
        if memory:
            return memory
        
        # Check LTM if not in STM
        memory = await self.long_term_memory.get_memory(memory_id)
        return memory
    
    async def force_backup(self) -> bool:
        """
        Force an immediate backup of long-term memories.
        
        Returns:
            Success status
        """
        try:
            success = await self.long_term_memory.backup()
            return success
        except Exception as e:
            logger.error(f"Error during forced backup: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        # Calculate average processing time
        avg_processing_time = self._total_time / max(1, self._total_processed)
        
        # Gather stats from components
        stm_stats = self.short_term_memory.get_stats()
        ltm_stats = self.long_term_memory.get_stats()
        mpl_stats = self.memory_prioritization.get_stats()
        hpc_stats = self.hpc_manager.get_stats()
        
        # System-wide stats
        return {
            'system': {
                'uptime': time.time() - self.start_time,
                'total_processed': self._total_processed,
                'total_stored': self._total_stored,
                'avg_processing_time': avg_processing_time,
                'storage_ratio': self._total_stored / max(1, self._total_processed),
                'device': self.config['device']
            },
            'stm': stm_stats,
            'ltm': ltm_stats,
            'mpl': mpl_stats,
            'hpc': hpc_stats
        }
```

# core\memory_entry.py

```py
"""
LUCID RECALL PROJECT
Memory Entry Data Structure

Provides a structured format for storing and managing memories
with significance tracking and serialization capabilities.
"""

import torch
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

class MemoryTypes(Enum):
    """Defines memory categories for different types of stored knowledge."""
    EPISODIC = "episodic"        # Event-based memory (conversations, interactions)
    SEMANTIC = "semantic"        # Fact-based knowledge (definitions, information)
    PROCEDURAL = "procedural"    # Skills & how-to memories
    WORKING = "working"          # Temporary processing memory
    PERSONAL = "personal"        # User-specific details
    IMPORTANT = "important"      # High-priority memories
    EMOTIONAL = "emotional"      # Emotionally tagged memories
    SYSTEM = "system"            # System-related configurations

@dataclass
class MemoryEntry:
    """
    Standardized representation for a single memory unit.
    
    This class ensures consistency across all memory operations.
    """
    content: str                                      # The actual memory content
    memory_type: MemoryTypes = MemoryTypes.EPISODIC   # Type of memory (default: EPISODIC)
    embedding: Optional[torch.Tensor] = None          # Vector representation (if applicable)
    
    id: str = field(default_factory=lambda: f"mem_{int(time.time()*1000)}")  # Unique memory ID
    timestamp: float = field(default_factory=time.time)  # Memory creation time
    significance: float = 0.5  # Importance level (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    access_count: int = 0  # Number of times accessed
    last_access: float = field(default_factory=time.time)  # Last access timestamp
    
    def __post_init__(self):
        """Ensure memory integrity after initialization."""
        # Normalize significance value
        self.significance = max(0.0, min(1.0, self.significance))
        
        # Ensure memory type is valid
        if isinstance(self.memory_type, str):
            try:
                self.memory_type = MemoryTypes[self.memory_type.upper()]
            except KeyError:
                self.memory_type = MemoryTypes.EPISODIC  # Default fallback
        
        # Ensure metadata is always a dictionary
        if not isinstance(self.metadata, dict):
            self.metadata = {}
    
    def record_access(self):
        """Update memory access count and timestamp."""
        self.access_count += 1
        self.last_access = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to a dictionary format for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding.cpu().tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "significance": self.significance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Restore memory from a dictionary format."""
        embedding = data.get("embedding")
        if embedding is not None and not isinstance(embedding, torch.Tensor):
            try:
                embedding = torch.tensor(embedding, dtype=torch.float32)
            except:
                embedding = None
        
        memory_type = MemoryTypes[data.get("memory_type", "EPISODIC").upper()]
        
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
        """Calculate effective significance considering time decay."""
        age_days = (time.time() - self.timestamp) / 86400  # Convert seconds to days
        
        if age_days < 1:
            return self.significance  # No decay for fresh memories
        
        importance_factor = 0.5 + (0.5 * self.significance)
        access_factor = 1.0 if (time.time() - self.last_access) < (7 * 86400) else 0.5
        access_bonus = min(3.0, 1.0 + (self.access_count / 10))
        
        effective_decay_rate = decay_rate / (importance_factor * access_factor * access_bonus)
        decay_factor = pow(2.718, -effective_decay_rate * (age_days - 1))
        
        return max(0.0, min(1.0, self.significance * decay_factor))
```

# core\memory_prioritization_layer.py

```py
"""
LUCID RECALL PROJECT
Memory Prioritization Layer (MPL)

A lightweight routing system that determines the best memory retrieval path
based on query type, context, and memory significance.
"""

import logging
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple
import torch

logger = logging.getLogger(__name__)

class MemoryPrioritizationLayer:
    """
    Routes queries based on type, context, and memory significance.
    
    The MPL determines the optimal retrieval path for queries, checking
    short-term and long-term memory before engaging HPC deep retrieval.
    This reduces redundant API calls and improves response time by
    prioritizing high-significance memories.
    """
    
    def __init__(self, short_term_memory, long_term_memory, hpc_client, config=None):
        """
        Initialize the Memory Prioritization Layer.
        
        Args:
            short_term_memory: Short-term memory component (recent interactions)
            long_term_memory: Long-term memory component (persistent storage)
            hpc_client: HPC client for deep retrieval when needed
            config: Optional configuration dictionary
        """
        self.stm = short_term_memory  # Holds last 5-10 interactions
        self.ltm = long_term_memory   # Persistent significance-weighted storage
        self.hpc_client = hpc_client  # Deep retrieval fallback
        
        # Configuration
        self.config = {
            'recall_threshold': 0.7,    # Similarity threshold for considering a memory recalled
            'cache_duration': 300,      # Cache duration in seconds (5 minutes)
            'stm_priority': 0.8,        # Priority weight for STM
            'ltm_priority': 0.5,        # Priority weight for LTM
            'hpc_priority': 0.3,        # Priority weight for HPC
            'max_stm_results': 5,       # Maximum results from STM
            'max_ltm_results': 10,      # Maximum results from LTM
            'max_hpc_results': 15,      # Maximum results from HPC
            'min_significance': 0.3,    # Minimum significance threshold
            **(config or {})
        }
        
        # Query cache to avoid redundant processing
        self._query_cache = {}
        
        # Performance tracking
        self.metrics = {
            'stm_hits': 0,
            'ltm_hits': 0,
            'hpc_hits': 0,
            'total_queries': 0,
            'avg_retrieval_time': 0,
            'cache_hits': 0
        }
        
        logger.info("Memory Prioritization Layer initialized")
    
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate memory system based on type and context.
        
        Args:
            query: The user query or text to process
            context: Optional additional context information
            
        Returns:
            Dict containing the results and metadata about the routing
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1
        context = context or {}
        
        # Check cache for identical recent queries
        cache_key = query.strip().lower()
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.config['cache_duration']:
                self.metrics['cache_hits'] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cache_entry['result']
        
        # Classify query type
        query_type = self._classify_query(query)
        logger.info(f"Query '{query[:50]}...' classified as {query_type}")
        
        # Route based on query type
        if query_type == "recall":
            result = await self._retrieve_memory(query, context)
        elif query_type == "information":
            result = await self._retrieve_information(query, context)
        elif query_type == "new_learning":
            result = await self._store_and_retrieve(query, context)
        else:
            # Default to information retrieval
            result = await self._retrieve_information(query, context)
        
        # Calculate and track performance metrics
        elapsed_time = time.time() - start_time
        self.metrics['avg_retrieval_time'] = (
            (self.metrics['avg_retrieval_time'] * (self.metrics['total_queries'] - 1) + elapsed_time) / 
            self.metrics['total_queries']
        )
        
        # Cache the result
        self._query_cache[cache_key] = {
            'timestamp': time.time(),
            'result': result
        }
        
        # Clean old cache entries
        self._clean_cache()
        
        # Add performance metadata
        result['_metadata'] = {
            'query_type': query_type,
            'retrieval_time': elapsed_time,
            'timestamp': time.time()
        }
        
        return result
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the query type to determine the appropriate retrieval strategy.
        
        Args:
            query: The user query text
            
        Returns:
            String classification: "recall", "information", or "new_learning"
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for memory recall patterns
        recall_patterns = [
            r"remember",
            r"recall",
            r"did (you|we) talk about",
            r"did I (tell|mention|say)",
            r"what did (I|you) say",
            r"previous(ly)?",
            r"earlier",
            r"last time"
        ]
        
        for pattern in recall_patterns:
            if re.search(pattern, query_lower):
                return "recall"
        
        # Check for information seeking patterns
        info_patterns = [
            r"(what|who|where|when|why|how) (is|are|was|were)",
            r"explain",
            r"tell me about",
            r"describe",
            r"definition of",
            r"information on",
            r"facts about"
        ]
        
        for pattern in info_patterns:
            if re.search(pattern, query_lower):
                return "information"
        
        # Default to new learning
        return "new_learning"
    
    async def _retrieve_memory(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve memories, starting with STM, then LTM, then HPC.
        
        Args:
            query: The memory recall query
            context: Additional context information
            
        Returns:
            Dict with retrieved memories and related metadata
        """
        # Start with short-term memory (most efficient)
        stm_results = await self._check_stm(query, context)
        
        # If we have strong matches in STM, return immediately
        if stm_results and any(result.get('similarity', 0) > self.config['recall_threshold'] 
                              for result in stm_results):
            self.metrics['stm_hits'] += 1
            return {
                'memories': stm_results,
                'source': 'short_term_memory',
                'count': len(stm_results)
            }
        
        # Try long-term memory next
        ltm_results = await self._check_ltm(query, context)
        
        # If we have strong matches in LTM, return combined results
        if ltm_results and any(result.get('similarity', 0) > self.config['recall_threshold'] 
                              for result in ltm_results):
            self.metrics['ltm_hits'] += 1
            
            # Combine results from STM and LTM
            combined_results = self._merge_results(stm_results, ltm_results)
            
            return {
                'memories': combined_results,
                'source': 'combined_stm_ltm',
                'count': len(combined_results)
            }
        
        # If no strong matches, try HPC retrieval as last resort
        hpc_results = await self._check_hpc(query, context)
        self.metrics['hpc_hits'] += 1
        
        # Combine all results with proper weighting
        all_results = self._merge_results(stm_results, ltm_results, hpc_results)
        
        return {
            'memories': all_results,
            'source': 'deep_retrieval',
            'count': len(all_results)
        }
    
    async def _retrieve_information(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve information using HPC but check memory first.
        
        Args:
            query: The information-seeking query
            context: Additional context information
            
        Returns:
            Dict with retrieved information
        """
        # For information queries, we still check STM first for efficiency
        stm_results = await self._check_stm(query, context)
        
        # If we have strong matches in STM, return immediately
        if stm_results and any(result.get('similarity', 0) > self.config['recall_threshold'] 
                              for result in stm_results):
            self.metrics['stm_hits'] += 1
            return {
                'memories': stm_results,
                'source': 'short_term_memory',
                'count': len(stm_results)
            }
        
        # For information queries, go directly to HPC for deep retrieval
        hpc_results = await self._check_hpc(query, context)
        self.metrics['hpc_hits'] += 1
        
        return {
            'memories': hpc_results,
            'source': 'deep_retrieval',
            'count': len(hpc_results)
        }
    
    async def _store_and_retrieve(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory and retrieve related memories.
        
        Args:
            query: The new information to store
            context: Additional context information
            
        Returns:
            Dict with status and related memories
        """
        # We'll store the memory in STM first
        memory_id = await self.stm.add_memory(query)
        
        # Evaluate significance for potential LTM storage
        significance = context.get('significance', 0.5)
        if significance > self.config['min_significance']:
            # Also store in LTM for persistence
            ltm_id = await self.ltm.store_memory(query, significance=significance)
            logger.info(f"Stored significant memory in LTM with ID {ltm_id}")
        
        # Retrieve similar memories to provide context
        # Start with short-term memory (most efficient)
        stm_results = await self._check_stm(query, context)
        
        # Also check long-term memory for context
        ltm_results = await self._check_ltm(query, context)
        
        # Combine results
        combined_results = self._merge_results(stm_results, ltm_results)
        
        return {
            'status': 'memory_stored',
            'memory_id': memory_id,
            'memories': combined_results,
            'source': 'new_learning',
            'count': len(combined_results)
        }
    
    async def _check_stm(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check short-term memory for matching memories.
        
        Args:
            query: The query to match
            context: Additional context information
            
        Returns:
            List of matching memories from STM
        """
        try:
            # Get recent memory matches from STM
            results = await self.stm.get_recent(query, 
                                             limit=self.config['max_stm_results'],
                                             min_similarity=self.config['min_significance'])
            
            # Add source metadata
            for result in results:
                result['source'] = 'short_term_memory'
                result['priority'] = self.config['stm_priority']
            
            return results
        except Exception as e:
            logger.error(f"Error checking STM: {e}")
            return []
    
    async def _check_ltm(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check long-term memory for matching memories.
        
        Args:
            query: The query to match
            context: Additional context information
            
        Returns:
            List of matching memories from LTM
        """
        try:
            # Search LTM for matching memories
            results = await self.ltm.search_memory(query, 
                                                limit=self.config['max_ltm_results'],
                                                min_significance=self.config['min_significance'])
            
            # Add source metadata
            for result in results:
                result['source'] = 'long_term_memory'
                result['priority'] = self.config['ltm_priority']
            
            return results
        except Exception as e:
            logger.error(f"Error checking LTM: {e}")
            return []
    
    async def _check_hpc(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check HPC for deep memory retrieval.
        
        Args:
            query: The query to match
            context: Additional context information
            
        Returns:
            List of matching memories from HPC
        """
        try:
            # Generate embedding for the query
            embedding = await self._get_query_embedding(query)
            if embedding is None:
                logger.error("Failed to generate embedding for HPC query")
                return []
            
            # Fetch relevant embeddings from HPC
            results = await self.hpc_client.fetch_relevant_embeddings(
                embedding, 
                limit=self.config['max_hpc_results'],
                min_significance=self.config['min_significance']
            )
            
            # Add source metadata
            for result in results:
                result['source'] = 'hpc_deep_retrieval'
                result['priority'] = self.config['hpc_priority']
            
            return results
        except Exception as e:
            logger.error(f"Error checking HPC: {e}")
            return []
    
    async def _get_query_embedding(self, query: str) -> Optional[torch.Tensor]:
        """
        Generate embedding for a query using the HPC client.
        
        Args:
            query: The query text
            
        Returns:
            Tensor embedding or None on failure
        """
        try:
            # This should call the appropriate method to get embedding
            # Implementation depends on your HPC client's interface
            embedding = await self.hpc_client.get_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return None
    
    def _merge_results(self, *result_lists) -> List[Dict[str, Any]]:
        """
        Merge multiple result lists with deduplication and prioritization.
        
        Args:
            *result_lists: Variable number of result lists to merge
            
        Returns:
            Combined and sorted list of unique results
        """
        # Collect all results
        all_results = []
        seen_ids = set()
        
        for results in result_lists:
            if not results:
                continue
                
            for result in results:
                # Skip if we've seen this memory ID already
                memory_id = result.get('id')
                if memory_id in seen_ids:
                    continue
                    
                # Add to combined results
                all_results.append(result)
                
                # Mark as seen
                if memory_id:
                    seen_ids.add(memory_id)
        
        # Sort by combined score of similarity, significance, and priority
        def get_combined_score(result):
            similarity = result.get('similarity', 0.5)
            significance = result.get('significance', 0.5)
            priority = result.get('priority', 0.5)
            
            return (similarity * 0.4) + (significance * 0.4) + (priority * 0.2)
        
        sorted_results = sorted(all_results, key=get_combined_score, reverse=True)
        
        return sorted_results
    
    def _clean_cache(self) -> None:
        """Clean expired entries from the query cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._query_cache.items()
            if current_time - entry['timestamp'] > self.config['cache_duration']
        ]
        
        for key in expired_keys:
            del self._query_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the MPL."""
        # Calculate hit ratios
        total_hits = self.metrics['stm_hits'] + self.metrics['ltm_hits'] + self.metrics['hpc_hits']
        
        stats = {
            'total_queries': self.metrics['total_queries'],
            'avg_retrieval_time': self.metrics['avg_retrieval_time'],
            'stm_hit_ratio': self.metrics['stm_hits'] / max(1, total_hits),
            'ltm_hit_ratio': self.metrics['ltm_hits'] / max(1, total_hits),
            'hpc_hit_ratio': self.metrics['hpc_hits'] / max(1, total_hits),
            'cache_hit_ratio': self.metrics['cache_hits'] / max(1, self.metrics['total_queries']),
            'cache_size': len(self._query_cache)
        }
        
        return stats
```

# core\memory_types.py

```py
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
```

# core\short_term_memory.py

```py
"""
LUCID RECALL PROJECT
Short-Term Memory (STM)

Stores last 5-10 user interactions (session-based) for quick reference.
"""

import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
import torch
from collections import deque

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """
    Short-Term Memory for recent interactions.
    
    Stores recent user interactions in a FIFO queue for quick access
    without the need for embedding processing or persistence.
    """
    
    def __init__(self, max_size: int = 10, embedding_comparator = None):
        """
        Initialize the short-term memory.
        
        Args:
            max_size: Maximum number of memories to store
            embedding_comparator: Optional component for semantic comparison
        """
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size
        self.embedding_comparator = embedding_comparator
        self._lock = asyncio.Lock()
        
        # Performance stats
        self.stats = {
            'additions': 0,
            'retrievals': 0,
            'matches': 0
        }
        
        logger.info(f"Initialized ShortTermMemory with max_size={max_size}")
        
    async def add_memory(self, content: str, embedding: Optional[torch.Tensor] = None, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to short-term storage.
        
        Args:
            content: The memory content text
            embedding: Optional pre-computed embedding
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        async with self._lock:
            # Generate a unique memory ID
            import uuid
            memory_id = str(uuid.uuid4())
            
            # Create memory entry
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Add to FIFO queue
            self.memory.append(memory)
            self.stats['additions'] += 1
            
            return memory_id
    
    async def get_recent(self, query: Optional[str] = None, limit: int = 5, 
                        min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get recent memories, optionally filtered by similarity to query.
        
        Args:
            query: Optional query to match against memories
            limit: Maximum number of memories to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching memories
        """
        async with self._lock:
            self.stats['retrievals'] += 1
            
            if not query:
                # If no query, just return most recent memories
                results = list(self.memory)[-limit:]
                results.reverse()  # Most recent first
                
                # Format results
                formatted_results = [{
                    'id': memory.get('id'),
                    'content': memory.get('content', ''),
                    'timestamp': memory.get('timestamp', 0),
                    'similarity': 1.0,  # Default similarity for recent entries
                    'significance': memory.get('metadata', {}).get('significance', 0.5)
                } for memory in results]
                
                return formatted_results
            
            # If we have query and embedding_comparator, do semantic search
            if self.embedding_comparator and hasattr(self.embedding_comparator, 'compare'):
                # Get embeddings for comparison
                query_embedding = await self.embedding_comparator.get_embedding(query)
                
                if query_embedding is not None:
                    # Check each memory for similarity
                    results = []
                    
                    for memory in self.memory:
                        memory_embedding = memory.get('embedding')
                        
                        # If no embedding, get one
                        if memory_embedding is None and memory.get('content'):
                            memory_embedding = await self.embedding_comparator.get_embedding(memory['content'])
                            memory['embedding'] = memory_embedding
                        
                        if memory_embedding is not None:
                            # Calculate similarity
                            similarity = await self.embedding_comparator.compare(
                                query_embedding, memory_embedding
                            )
                            
                            # If above threshold, add to results
                            if similarity >= min_similarity:
                                self.stats['matches'] += 1
                                
                                results.append({
                                    'id': memory.get('id'),
                                    'content': memory.get('content', ''),
                                    'timestamp': memory.get('timestamp', 0),
                                    'similarity': similarity,
                                    'significance': memory.get('metadata', {}).get('significance', 0.5)
                                })
                    
                    # Sort by similarity
                    results.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Return top matches
                    return results[:limit]
            
            # Fallback: Simple text matching
            results = []
            
            for memory in self.memory:
                content = memory.get('content', '').lower()
                query_lower = query.lower()
                
                # Simple token overlap for matching
                tokens_content = set(content.split())
                tokens_query = set(query_lower.split())
                
                # Calculate Jaccard similarity
                if tokens_content and tokens_query:
                    intersection = tokens_content.intersection(tokens_query)
                    union = tokens_content.union(tokens_query)
                    similarity = len(intersection) / len(union)
                else:
                    similarity = 0.0
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    self.stats['matches'] += 1
                    
                    results.append({
                        'id': memory.get('id'),
                        'content': memory.get('content', ''),
                        'timestamp': memory.get('timestamp', 0),
                        'similarity': similarity,
                        'significance': memory.get('metadata', {}).get('significance', 0.5)
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top matches
            return results[:limit]
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory dict or None if not found
        """
        for memory in self.memory:
            if memory.get('id') == memory_id:
                return memory
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'size': len(self.memory),
            'max_size': self.max_size,
            'utilization': len(self.memory) / self.max_size,
            'additions': self.stats['additions'],
            'retrievals': self.stats['retrievals'],
            'matches': self.stats['matches'],
            'match_ratio': self.stats['matches'] / max(1, self.stats['retrievals'])
        }
```

# README.md

```md
# **Lucidia Memory System**

## **📌 Overview**
Lucidia’s Memory System is a **self-governing, structured, and highly efficient retrieval system** designed for **adaptive recall, optimal processing, and scalable knowledge storage**. 

This architecture integrates **Short-Term Memory (STM)** for fast recall, **Long-Term Memory (LTM)** for persistence, and a **Memory Prioritization Layer (MPL)** to intelligently route queries. The **HPC server handles deep retrieval and embedding processing**, ensuring that **only the most relevant information is surfaced efficiently**.

---

## **🚀 Features**
- **Hierarchical Memory Architecture**: STM handles session-based context, LTM retains significance-weighted knowledge, and MPL determines the best retrieval strategy.
- **Dynamic Memory Decay**: Low-value memories naturally fade, while high-value information remains.
- **Embedding Optimization**: HPC-processed embeddings allow **semantic recall with minimal redundant computation**.
- **Self-Organizing Memory**: Recurrent interactions reinforce important memories **without manual intervention**.
- **Fast Query Routing**: MPL ensures that **queries are answered optimally**—fetching from STM, LTM, or HPC as required.

---

## **📂 File Structure**
\`\`\`
/lucidia_memory_system
│
├── core/  # Main memory processing core
│   ├── memory_core.py                      # Manages STM, LTM, and MPL
│   ├── memory_prioritization_layer.py      # Routes queries optimally
│   ├── short_term_memory.py                 # Stores recent session-based interactions
│   ├── long_term_memory.py                  # Persistent storage with decay model
│   ├── embedding_comparator.py              # Handles embedding similarity checks
│   ├── memory_types.py                      # Defines memory categories (episodic, semantic, procedural, etc.)
│   ├── memory_entry.py                      # Data structure for memory storage
│
├── integration/  # API layer for other modules to interact with memory
│   ├── memory_integration.py                # Simplified API for external components
│   ├── updated_hpc_client.py                # Handles connection to HPC
│   ├── hpc_sig_flow_manager.py              # Manages significance weighting in HPC
│
├── storage/  # Persistent memory storage
│   ├── ltm_storage/                         # Long-term memory stored here
│   ├── memory_index.json                    # Metadata index for stored memories
│   ├── memory_persistence_handler.py        # Handles disk-based memory saving/loading
│
├── tests/  # Unit tests and benchmarks
│   ├── test_memory_core.py                   # Tests STM, LTM, MPL interactions
│   ├── test_memory_retrieval.py              # Ensures queries route correctly
│   ├── test_embedding_comparator.py          # Validates embedding similarity comparisons
│
├── utils/  # Utility functions
│   ├── logging_config.py                     # Standardized logging
│   ├── performance_tracker.py                # Monitors response times
│   ├── cache_manager.py                       # Implements memory caching
│
└── README.md  # Documentation
\`\`\`

---

## **🔹 Core Components**

### **1️⃣ Memory Prioritization Layer (MPL)**
🔹 **Routes queries intelligently**, prioritizing memory recall before deep retrieval.

- Determines whether a query is **recall, information-seeking, or new learning**.
- Retrieves from STM first, then LTM, then HPC if necessary.
- Implements **query caching** to prevent redundant processing.

### **2️⃣ Short-Term Memory (STM)**
🔹 **Stores recent session-based interactions** for **fast retrieval**.

- FIFO-based memory buffer (last **5-10 user interactions**).
- Avoids storing unnecessary details, keeping **only context-relevant information**.

### **3️⃣ Long-Term Memory (LTM)**
🔹 **Stores high-significance memories** persistently.

- Implements **memory decay**: low-value memories gradually fade.
- **Dynamic reinforcement**: frequently referenced memories gain weight.
- Auto-backup mechanism ensures **no critical knowledge is lost**.

### **4️⃣ Embedding Comparator**
🔹 **Handles vector-based similarity checks** for memory retrieval.

- Ensures **efficient memory lookup** using semantic embeddings.
- Caches embeddings to prevent **unnecessary recomputation**.

### **5️⃣ HPC Integration**
🔹 **Offloads embedding processing and significance scoring**.

- Deep memory retrieval when **STM & LTM fail to provide a match**.
- Batch processing and caching minimize API calls.
- Ensures **contextually relevant recall at scale**.

---

## **🛠️ Installation & Setup**

### **📌 Requirements**
- **Python 3.8+**
- **PyTorch** (for embeddings & memory processing)
- **WebSockets** (for HPC communication)
- **NumPy** (for efficient vector processing)

### **📦 Install Dependencies**
\`\`\`sh
pip install torch numpy websockets
\`\`\`

### **🔧 Running the System**
\`\`\`sh
python -m lucidia_memory_system.memory_core
\`\`\`

---

## **🔍 How It Works**

### **🔹 Query Processing Flow**
\`\`\`
User Query → MPL → [STM] → [LTM] → [HPC] → Response
\`\`\`
1. **Query enters MPL:** Classifies if the request is **recall, information-seeking, or new learning**.
2. **STM is checked first** (last 5-10 interactions) for fast retrieval.
3. **If not found in STM, LTM is queried** (significance-weighted storage).
4. **If no match in LTM, HPC retrieval is triggered** for embedding-based recall.
5. **Final memory context is sent to the LLM** for response generation.

---

## **📊 System Benchmarks & Efficiency Gains**
✅ **Reduces API calls by up to 60%** by prioritizing memory recall over external retrieval.
✅ **Significance-based recall speeds up response time by 2-3x** compared to traditional search.
✅ **Dynamically adjusts memory priority** based on user interaction frequency.
✅ **Removes redundant data storage**, preventing unnecessary memory bloat.

---

## **📌 Next Steps**
1️⃣ **Fine-tune MPL query routing** to further optimize retrieval paths.
2️⃣ **Improve memory decay** algorithms to maintain long-term relevance.
3️⃣ **Optimize HPC API interactions** to batch process embeddings more efficiently.
4️⃣ **Expand caching mechanisms** for near-instant STM lookups.

---

🚀 **Lucidia’s memory system is now self-organizing, intelligent, and built for long-term scalability.**

```

# utils\cache_manager.py

```py
"""
LUCID RECALL PROJECT
Cache Manager

Implements memory caching strategies for improved performance.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, List, Tuple, Union
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for cached values

class CacheManager(Generic[T]):
    """
    Generic cache manager with multiple strategies.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) expiration
    - Size limiting
    - Cache statistics
    """
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600, 
                strategy: str = 'lru', name: str = 'cache'):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Default time-to-live in seconds
            strategy: Caching strategy ('lru', 'fifo', 'lfu')
            name: Name of this cache (for stats and debugging)
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.strategy = strategy.lower()
        self.name = name
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Initialize cache
        if self.strategy == 'lru':
            # LRU cache using OrderedDict
            self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        else:
            # Regular cache for other strategies
            self.cache: Dict[str, Dict[str, Any]] = {}
            
        # Access counts for LFU strategy
        self.access_counts: Dict[str, int] = {}
        
        # Stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'inserts': 0,
            'evictions': 0,
            'expirations': 0
        }
        
        # Set up auto cleanup task if ttl is enabled
        if ttl > 0:
            cleanup_interval = min(ttl / 2, 300)  # Half of TTL or 5 minutes, whichever is less
            self._cleanup_task = asyncio.create_task(self._auto_cleanup(cleanup_interval))
        else:
            self._cleanup_task = None
            
        logger.info(f"Initialized {name} cache with strategy={strategy}, max_size={max_size}, ttl={ttl}")
    
    async def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        async with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return default
                
            # Get cache entry
            entry = self.cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return default
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
            
            # Update access count for LFU strategy
            if self.strategy == 'lfu':
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
            self.stats['hits'] += 1
            return entry['value']
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
        """
        async with self._lock:
            # Check if at max size before adding
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict an item
                await self._evict_item()
            
            # Create cache entry
            entry = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl if ttl is not None else self.default_ttl
            }
            
            # Add or update in cache
            self.cache[key] = entry
            
            # Initialize or reset access count
            if self.strategy == 'lfu':
                self.access_counts[key] = 0
                
            self.stats['inserts'] += 1
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Whether the key was found and deleted
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            logger.info(f"Cleared {self.name} cache")
    
    async def keys(self) -> List[str]:
        """Get list of all keys in cache."""
        async with self._lock:
            return list(self.cache.keys())
    
    async def contains(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            Whether the key exists and is not expired
        """
        async with self._lock:
            if key not in self.cache:
                return False
                
            # Check if expired
            if self._is_expired(self.cache[key]):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                return False
                
            return True
    
    async def touch(self, key: str, ttl: Optional[float] = None) -> bool:
        """
        Update the access time for a key.
        
        Args:
            key: Cache key
            ttl: Optional new TTL
            
        Returns:
            Whether the key was found and touched
        """
        async with self._lock:
            if key not in self.cache:
                return False
                
            # Check if expired
            if self._is_expired(self.cache[key]):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                return False
                
            # Update timestamp
            self.cache[key]['timestamp'] = time.time()
            
            # Update TTL if provided
            if ttl is not None:
                self.cache[key]['ttl'] = ttl
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
                
            return True
    
    async def get_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get an item with its metadata.
        
        Args:
            key: Cache key
            
        Returns:
            Dict with value and metadata or None if not found
        """
        async with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
                
            # Get cache entry
            entry = self.cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return None
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
                
            # Update access count for LFU strategy
            if self.strategy == 'lfu':
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
            self.stats['hits'] += 1
            
            # Return entry with metadata
            current_time = time.time()
            age = current_time - entry['timestamp']
            ttl = entry['ttl']
            remaining = max(0, ttl - age) if ttl > 0 else None
            
            return {
                'value': entry['value'],
                'age': age,
                'ttl': ttl,
                'remaining': remaining,
                'timestamp': entry['timestamp']
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            # Calculate hit ratio
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_ratio = self.stats['hits'] / max(1, total_requests)
            
            stats = {
                'name': self.name,
                'strategy': self.strategy,
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_ratio': hit_ratio,
                'inserts': self.stats['inserts'],
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations']
            }
            
            return stats
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            entry: Cache entry
            
        Returns:
            Whether the entry is expired
        """
        if entry['ttl'] <= 0:
            # TTL of 0 or negative means never expire
            return False
            
        # Check if elapsed time exceeds TTL
        current_time = time.time()
        age = current_time - entry['timestamp']
        return age > entry['ttl']
    
    async def _evict_item(self) -> None:
        """Evict an item based on the selected strategy."""
        if not self.cache:
            return
            
        if self.strategy == 'lru':
            # LRU - remove first item in OrderedDict (least recently used)
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
            
        elif self.strategy == 'fifo':
            # FIFO - remove oldest inserted item
            # Find oldest item by timestamp
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            if oldest_key in self.access_counts:
                del self.access_counts[oldest_key]
            self.stats['evictions'] += 1
            
        elif self.strategy == 'lfu':
            # LFU - remove least frequently used item
            # Find key with lowest access count
            least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            del self.cache[least_used_key]
            del self.access_counts[least_used_key]
            self.stats['evictions'] += 1
            
        else:
            # Default - remove random item
            random_key = next(iter(self.cache))
            del self.cache[random_key]
            if random_key in self.access_counts:
                del self.access_counts[random_key]
            self.stats['evictions'] += 1
    
    async def _auto_cleanup(self, interval: float) -> None:
        """
        Periodically clean up expired entries.
        
        Args:
            interval: Cleanup interval in seconds
        """
        try:
            while True:
                # Wait for interval
                await asyncio.sleep(interval)
                
                # Clean up expired entries
                await self.cleanup_expired()
                
        except asyncio.CancelledError:
            # Task cancelled, exit gracefully
            logger.info(f"Cleanup task for {self.name} cache cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = []
            
            # Find expired entries
            for key, entry in list(self.cache.items()):
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
            
            # Update stats
            self.stats['expirations'] += len(expired_keys)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries from {self.name} cache")
                
            return len(expired_keys)
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
```

# utils\logging_config.py

```py
"""
LUCID RECALL PROJECT
Logging Configuration

Standardized logging setup for consistent logging across all components.
"""

import logging
import sys
from pathlib import Path

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(name: str, level: str = "info", log_file: Path = None) -> logging.Logger:
    """
    Setup a logger with standardized formatting.

    Args:
        name (str): Name of the logger (usually the module name)
        level (str): Logging level as a string (debug, info, warning, error, critical)
        log_file (Path, optional): File path to write logs to.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))

    # Create log formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if a log file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Example: Initialize a logger for general use
logger = setup_logger("Lucidia", level="debug", log_file=Path("logs/lucidia.log"))
logger.info("Logging system initialized.")
```

# utils\performance_tracker.py

```py
"""
LUCID RECALL PROJECT
Performance Tracker

Monitors system performance, tracks metrics, and provides analytics
for identifying bottlenecks.
"""

import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Coroutine, TypeVar, Union
from collections import defaultdict, deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for function return values

class PerformanceTracker:
    """
    Performance tracking and monitoring utility.
    
    Features:
    - Operation timing
    - Rate limiting
    - Bottleneck detection
    - Performance analytics
    - Memory operation profiling
    """
    
    def __init__(self, 
                 history_size: int = 100, 
                 alert_threshold: float = 2.0, 
                 debug: bool = False):
        """
        Initialize the performance tracker.
        
        Args:
            history_size: Number of recent operations to track
            alert_threshold: Multiplier for average time to trigger alerts
            debug: Whether to log detailed debug information
        """
        self.history_size = history_size
        self.alert_threshold = alert_threshold
        self.debug = debug
        
        # Track operation times by category
        self.operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        
        # Track ongoing operations
        self.ongoing_operations: Dict[str, Dict[str, float]] = {}
        
        # Global stats
        self.stats = {
            'total_operations': 0,
            'slow_operations': 0,
            'failed_operations': 0,
            'start_time': time.time()
        }
        
        # Performance report data
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_times: Dict[str, float] = defaultdict(float)
        self.operation_failures: Dict[str, int] = defaultdict(int)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Performance tracker initialized with history_size={history_size}")
    
    async def record_operation(self, 
                             operation: str, 
                             duration: float, 
                             success: bool = True, 
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an operation's performance.
        
        Args:
            operation: Name/category of operation
            duration: Time taken in seconds
            success: Whether operation succeeded
            metadata: Optional additional data
        """
        async with self._lock:
            # Update global stats
            self.stats['total_operations'] += 1
            if not success:
                self.stats['failed_operations'] += 1
            
            # Update operation-specific stats
            self.operation_counts[operation] += 1
            self.operation_times[operation] += duration
            if not success:
                self.operation_failures[operation] += 1
            
            # Create operation record
            record = {
                'duration': duration,
                'timestamp': time.time(),
                'success': success,
                'metadata': metadata or {}
            }
            
            # Add to history
            self.operations[operation].append(record)
            
            # Check if slow
            avg_time = self._get_average_time(operation)
            if avg_time > 0 and duration > avg_time * self.alert_threshold:
                self.stats['slow_operations'] += 1
                if self.debug:
                    logger.warning(f"Slow operation detected: {operation} took {duration:.3f}s "
                                 f"(avg: {avg_time:.3f}s)")
    
    async def start_operation(self, operation: str, 
                            op_id: Optional[str] = None) -> str:
        """
        Start tracking an operation's time.
        
        Args:
            operation: Name/category of operation
            op_id: Optional operation ID for correlation
            
        Returns:
            Operation ID for stopping
        """
        op_id = op_id or f"{operation}_{int(time.time() * 1000)}"
        
        async with self._lock:
            self.ongoing_operations[op_id] = {
                'operation': operation,
                'start_time': time.time()
            }
            
            if self.debug:
                logger.debug(f"Started tracking operation: {operation} (ID: {op_id})")
                
        return op_id
    
    async def stop_operation(self, op_id: str, success: bool = True,
                           metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Stop tracking an operation and record its performance.
        
        Args:
            op_id: Operation ID from start_operation
            success: Whether operation succeeded
            metadata: Optional additional data
            
        Returns:
            Duration in seconds or -1 if operation wasn't tracked
        """
        if op_id not in self.ongoing_operations:
            logger.warning(f"Operation {op_id} not found in ongoing operations")
            return -1
            
        async with self._lock:
            # Get operation data
            op_data = self.ongoing_operations.pop(op_id)
            operation = op_data['operation']
            start_time = op_data['start_time']
            
            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Record the operation
            await self.record_operation(operation, duration, success, metadata)
            
            if self.debug:
                logger.debug(f"Completed operation: {operation} in {duration:.3f}s (success: {success})")
                
            return duration
    
    @asynccontextmanager
    async def track_operation(self, operation: str) -> None:
        """
        Context manager for tracking operation time.
        
        Usage:
        \`\`\`
        async with performance_tracker.track_operation("db_query"):
            result = await db.query(...)
        \`\`\`
        
        Args:
            operation: Name/category of operation
        """
        # Start operation timing
        op_id = await self.start_operation(operation)
        success = True
        
        try:
            # Yield control back to the context block
            yield
        except Exception as e:
            # Mark as failed on exception
            success = False
            raise
        finally:
            # Record operation time
            await self.stop_operation(op_id, success)
    
    async def timed_execution(self, operation: str, func: Callable[..., Coroutine[Any, Any, T]], 
                            *args, **kwargs) -> T:
        """
        Execute a coroutine function with timing.
        
        Usage:
        \`\`\`
        result = await performance_tracker.timed_execution(
            "db_query", db.query, "SELECT * FROM table"
        )
        \`\`\`
        
        Args:
            operation: Name/category of operation
            func: Coroutine function to execute
            *args: Arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of the function
        """
        # Start operation timing
        op_id = await self.start_operation(operation)
        success = True
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            # Mark as failed on exception
            success = False
            raise
        finally:
            # Record operation time
            await self.stop_operation(op_id, success)
    
    def _get_average_time(self, operation: str) -> float:
        """
        Get average execution time for an operation.
        
        Args:
            operation: Name/category of operation
            
        Returns:
            Average execution time in seconds
        """
        if operation not in self.operations or not self.operations[operation]:
            return 0.0
            
        # Calculate average duration
        durations = [record['duration'] for record in self.operations[operation]]
        return sum(durations) / len(durations)
    
    def _get_percentile_time(self, operation: str, percentile: float = 95) -> float:
        """
        Get percentile execution time for an operation.
        
        Args:
            operation: Name/category of operation
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile execution time in seconds
        """
        if operation not in self.operations or not self.operations[operation]:
            return 0.0
            
        # Calculate percentile duration
        durations = [record['duration'] for record in self.operations[operation]]
        
        try:
            return statistics.quantiles(durations, n=100)[int(percentile) - 1]
        except (ValueError, IndexError):
            # Fall back to simple calculation for small samples
            durations.sort()
            idx = int((percentile / 100) * len(durations))
            return durations[idx - 1] if idx > 0 else durations[0]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Dict with performance metrics
        """
        async with self._lock:
            # Calculate stats for each operation
            operation_stats = {}
            
            for operation in self.operations:
                # Skip operations with no records
                if not self.operations[operation]:
                    continue
                    
                # Get durations
                durations = [record['duration'] for record in self.operations[operation]]
                
                # Calculate statistics
                try:
                    if len(durations) >= 2:
                        percentiles = statistics.quantiles(durations, n=4)
                        p25, p50, p75 = percentiles
                        p95 = self._get_percentile_time(operation, 95)
                        p99 = self._get_percentile_time(operation, 99)
                    else:
                        p25 = p50 = p75 = p95 = p99 = durations[0] if durations else 0
                        
                    operation_stats[operation] = {
                        'count': len(self.operations[operation]),
                        'avg_time': sum(durations) / len(durations),
                        'min_time': min(durations),
                        'max_time': max(durations),
                        'p25': p25,
                        'p50': p50,
                        'p75': p75,
                        'p95': p95,
                        'p99': p99,
                        'success_rate': sum(1 for r in self.operations[operation] if r['success']) / len(self.operations[operation]),
                        'total_time': sum(durations)
                    }
                except (ValueError, IndexError, statistics.StatisticsError):
                    # Fall back to simple stats for small samples
                    operation_stats[operation] = {
                        'count': len(self.operations[operation]),
                        'avg_time': sum(durations) / max(1, len(durations)),
                        'min_time': min(durations) if durations else 0,
                        'max_time': max(durations) if durations else 0,
                        'success_rate': sum(1 for r in self.operations[operation] if r['success']) / max(1, len(self.operations[operation])),
                        'total_time': sum(durations)
                    }
            
            # Calculate global stats
            uptime = time.time() - self.stats['start_time']
            total_operations = self.stats['total_operations']
            ops_per_second = total_operations / uptime if uptime > 0 else 0
            
            # Identify potential bottlenecks
            bottlenecks = []
            if operation_stats:
                # Sort operations by total time spent
                sorted_by_time = sorted(
                    operation_stats.items(), 
                    key=lambda x: x[1]['total_time'], 
                    reverse=True
                )
                
                # Top 3 operations by time
                top_by_time = sorted_by_time[:3]
                
                # Add to bottlenecks if they take more than 10% of total time
                total_time = sum(op['total_time'] for _, op in operation_stats.items())
                if total_time > 0:
                    for operation, stats in top_by_time:
                        time_percentage = (stats['total_time'] / total_time) * 100
                        if time_percentage > 10:
                            bottlenecks.append({
                                'operation': operation,
                                'time_percentage': time_percentage,
                                'avg_time': stats['avg_time'],
                                'count': stats['count']
                            })
            
            # Compile full report
            report = {
                'global_stats': {
                    'uptime': uptime,
                    'total_operations': total_operations,
                    'operations_per_second': ops_per_second,
                    'slow_operations': self.stats['slow_operations'],
                    'failed_operations': self.stats['failed_operations'],
                    'failure_rate': self.stats['failed_operations'] / max(1, total_operations)
                },
                'operation_stats': operation_stats,
                'bottlenecks': bottlenecks,
                'ongoing_operations': len(self.ongoing_operations)
            }
            
            return report
    
    async def reset_stats(self) -> None:
        """Reset all statistics."""
        async with self._lock:
            # Clear operation histories
            self.operations.clear()
            self.operations = defaultdict(lambda: deque(maxlen=self.history_size))
            
            # Reset global stats
            self.stats = {
                'total_operations': 0,
                'slow_operations': 0,
                'failed_operations': 0,
                'start_time': time.time()
            }
            
            # Reset operation tracking
            self.operation_counts.clear()
            self.operation_times.clear()
            self.operation_failures.clear()
            
            logger.info("Performance tracker stats reset")
    
    async def get_operation_history(self, operation: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific operation.
        
        Args:
            operation: Name/category of operation
            
        Returns:
            List of operation records
        """
        async with self._lock:
            if operation not in self.operations:
                return []
                
            return list(self.operations[operation])
```

