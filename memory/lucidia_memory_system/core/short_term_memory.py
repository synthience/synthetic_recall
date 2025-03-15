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
            'cross_session_imports': 0,
            'retrievals': 0,
            'cross_session_retrievals': 0,
            'matches': 0
        }
        
        logger.info(f"Initialized ShortTermMemory with max_size={max_size}")
        
    async def add_memory(self, content: str, embedding: Optional[torch.Tensor] = None, 
                       metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> str:
        """
        Add a memory to short-term storage.
        
        Args:
            content: The memory content text
            embedding: Optional pre-computed embedding
            metadata: Optional metadata
            memory_id: Optional custom memory ID
            
        Returns:
            Memory ID
        """
        async with self._lock:
            # Generate a unique memory ID if not provided
            import uuid
            memory_id = memory_id or str(uuid.uuid4())
            
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

    async def import_from_ltm(self, ltm_memory: Dict[str, Any]) -> str:
        """
        Import a memory from long-term memory into short-term memory.
        This is used to bring cross-session memories into the current session.
        
        Args:
            ltm_memory: Memory from long-term memory
            
        Returns:
            Memory ID
        """
        async with self._lock:
            # Extract needed fields
            memory_id = ltm_memory.get('id')
            content = ltm_memory.get('content', '')
            embedding = ltm_memory.get('embedding')
            
            # Create metadata with cross-session flag
            metadata = ltm_memory.get('metadata', {}).copy()
            metadata['cross_session'] = True
            metadata['original_timestamp'] = ltm_memory.get('timestamp', 0)
            
            # Boost significance for cross-session memories
            if 'significance' in metadata:
                metadata['significance'] = min(1.0, metadata['significance'] * 1.2)
            
            # Add to STM
            memory = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'timestamp': time.time(),  # Current time
                'metadata': metadata
            }
            
            # Add to FIFO queue
            self.memory.append(memory)
            self.stats['cross_session_imports'] += 1
            
            logger.info(f"Imported cross-session memory {memory_id} from LTM to STM")
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
                    'significance': memory.get('metadata', {}).get('significance', 0.5),
                    'cross_session': memory.get('metadata', {}).get('cross_session', False)
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
                                    'significance': memory.get('metadata', {}).get('significance', 0.5),
                                    'cross_session': memory.get('metadata', {}).get('cross_session', False)
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
                        'significance': memory.get('metadata', {}).get('significance', 0.5),
                        'cross_session': memory.get('metadata', {}).get('cross_session', False)
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
    
    async def keyword_search(self, keywords: List[str], limit: int = 5, min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search memories by keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        logger.debug(f"Performing keyword search with keywords: {keywords}")
        
        if not keywords:
            return []
        
        results = []
        
        # Convert keywords to lowercase for case-insensitive matching
        lowercase_keywords = [k.lower() for k in keywords]
        
        for memory in self.memory:
            # Skip memories below significance threshold
            if memory.get('metadata', {}).get('significance', 0.0) < min_significance:
                continue
                
            # Check if any keyword is in the memory content
            content = memory.get('content', '').lower()
            if any(keyword in content for keyword in lowercase_keywords):
                results.append({
                    'id': memory.get('id'),
                    'content': memory.get('content', ''),
                    'timestamp': memory.get('timestamp', 0),
                    'similarity': 1.0,  # Default similarity for keyword matches
                    'significance': memory.get('metadata', {}).get('significance', 0.5),
                    'cross_session': memory.get('metadata', {}).get('cross_session', False)
                })
                
                # Stop once we reach the limit
                if len(results) >= limit:
                    break
        
        logger.info(f"Keyword search found {len(results)} results")
        return results
    
    async def search(self, query: str, limit: int = 5, min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity to a query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        logger.debug(f"Performing semantic search with query: {query}")
        
        # Try to use the get_recent method first, which has semantic search capabilities
        try:
            results = await self.get_recent(query, limit, min_significance)
            if results:
                return results
        except Exception as e:
            logger.warning(f"Error using get_recent for search: {e}")
        
        # As a fallback, treat this as a keyword search by splitting the query into words
        keywords = query.split()
        return await self.keyword_search(keywords, limit, min_significance)
        
    async def recency_biased_search(self, query: str, limit: int = 5, recency_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search memories with a bias toward recency.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            recency_weight: Weight to give to recency vs content relevance (0.0-1.0)
            
        Returns:
            List of matching memories
        """
        logger.debug(f"Performing recency-biased search with query: {query}")
        
        if not self.memory:
            return []
        
        # Get semantic search results
        semantic_results = []
        try:
            if self.embedding_comparator and hasattr(self.embedding_comparator, 'compare'):
                query_embedding = await self.embedding_comparator.get_embedding(query)
                
                if query_embedding is not None:
                    # Get similarity scores
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
                            
                            semantic_results.append({
                                'id': memory.get('id'),
                                'content': memory.get('content', ''),
                                'timestamp': memory.get('timestamp', 0),
                                'similarity': similarity,
                                'significance': memory.get('metadata', {}).get('significance', 0.5),
                                'cross_session': memory.get('metadata', {}).get('cross_session', False)
                            })
        except Exception as e:
            logger.warning(f"Error in semantic search during recency-biased search: {e}")
        
        # If no semantic results, do keyword matching
        if not semantic_results:
            keywords = query.split()
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
                
                semantic_results.append({
                    'id': memory.get('id'),
                    'content': memory.get('content', ''),
                    'timestamp': memory.get('timestamp', 0),
                    'similarity': similarity,
                    'significance': memory.get('metadata', {}).get('significance', 0.5),
                    'cross_session': memory.get('metadata', {}).get('cross_session', False)
                })
        
        # Apply recency bias
        results = []
        current_time = time.time()
        oldest_time = min([memory.get('timestamp', 0) for memory in self.memory])
        time_range = max(current_time - oldest_time, 1)  # Avoid division by zero
        
        for result in semantic_results:
            # Calculate recency score (0-1)
            recency_score = (result['timestamp'] - oldest_time) / time_range
            
            # Combine recency and semantic scores
            combined_score = (recency_weight * recency_score) + ((1 - recency_weight) * result['similarity'])
            
            results.append({
                'id': result['id'],
                'content': result['content'],
                'timestamp': result['timestamp'],
                'similarity': result['similarity'],
                'significance': result['significance'],
                'combined_score': combined_score,
                'cross_session': result.get('cross_session', False)
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top results
        return results[:limit]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'size': len(self.memory),
            'max_size': self.max_size,
            'utilization': len(self.memory) / self.max_size,
            'additions': self.stats['additions'],
            'retrievals': self.stats['retrievals'],
            'matches': self.stats['matches'],
            'match_ratio': self.stats['matches'] / max(1, self.stats['retrievals']),
            'cross_session_imports': self.stats['cross_session_imports'],
            'cross_session_retrievals': self.stats.get('cross_session_retrievals', 0)
        }
        
    async def update_access_timestamp(self, memory_id: str) -> bool:
        """Update the access timestamp for a memory.
        
        Args:
            memory_id: ID of the memory to update
            
        Returns:
            True if memory was found and updated, False otherwise
        """
        for i, memory in enumerate(self.memory):
            if memory.get('id') == memory_id:
                # Update timestamp
                self.memory[i]['last_access'] = time.time()
                self.memory[i]['access_count'] = self.memory[i].get('access_count', 0) + 1
                
                # Make sure these fields are also in metadata
                if 'metadata' not in self.memory[i]:
                    self.memory[i]['metadata'] = {}
                    
                self.memory[i]['metadata']['last_access'] = self.memory[i]['last_access']
                self.memory[i]['metadata']['access_count'] = self.memory[i].get('access_count', 1)
                
                logger.debug(f"Updated access timestamp for memory {memory_id} in STM")
                return True
                
        return False