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