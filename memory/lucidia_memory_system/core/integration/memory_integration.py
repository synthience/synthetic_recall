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

from ..embedding_comparator import EmbeddingComparator
from ..short_term_memory import ShortTermMemory
from ..long_term_memory import LongTermMemory
from ..memory_prioritization_layer import MemoryPrioritizationLayer

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
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logger.info("Initializing Memory Integration Layer")
        
        # Lazy import MemoryCore to avoid circular reference
        from ..memory_core import MemoryCore as EnhancedMemoryCore
        
        # Define memory architecture components:
        # - MemoryCore (STM, LTM, MPL)
        # - Persistence Layer
        # - Optimizers
        
        # Initialize enhanced memory core
        logger.info("Initializing enhanced memory core...")
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
                    from ..memory_types import MemoryTypes
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
            from ..memory_types import MemoryTypes
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