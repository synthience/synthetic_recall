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
    
    async def personal_info_search(self, query: str, context: Optional[Dict[str, Any]] = None, min_personal_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for personal information in memory.
        
        This method looks for personal details like names, preferences, relationships,
        and important facts about the user stored in memory.
        
        Args:
            query: The search query or information category to look for
            context: Optional additional context for the search
            min_personal_significance: Minimum significance threshold for personal info
            
        Returns:
            List of memories containing personal information
        """
        try:
            logger.info(f"Searching for personal information related to: {query}")
            context = context or {}
            
            # Define regex patterns for different types of personal information
            patterns = {
                'names': r'\b(?:my name is|I am|I\'m|call me)\s+([A-Z][a-z]+)\b',
                'preferences': r'\b(?:I (?:like|love|enjoy|prefer|favorite))\s+(.+?)(?:\.|,|$)',
                'relationships': r'\b(?:my (?:wife|husband|partner|girlfriend|boyfriend|mother|father|sister|brother|daughter|son|friend))\s+([A-Z][a-z]+)\b',
                'personal_traits': r'\b(?:I am|I\'m)\s+(?!(?:\d+|a|an|the)\b)([^.,;!?]+)\b'
            }
            
            # First check STM for recent personal information
            stm_results = await self._check_stm(query, context)
            
            # Then check LTM with potentially more specific search parameters
            ltm_query = f"personal {query}" if not query.startswith("personal") else query
            ltm_results = await self._check_ltm(ltm_query, context)
            
            # Combine results with proper weighting
            combined_results = self._merge_results(stm_results, ltm_results)
            
            # Add metadata about personal information type if we can detect it
            for result in combined_results:
                content = result.get('content', '')
                for info_type, pattern in patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        if 'personal_info' not in result:
                            result['personal_info'] = {}
                        result['personal_info'][info_type] = matches
            
            # Filter results to only include those with personal information
            # and have significance above threshold
            personal_results = [
                r for r in combined_results 
                if 'personal_info' in r and r.get('metadata', {}).get('significance', 0) >= min_personal_significance
            ]
            
            # If we have specific personal results, return those
            if personal_results:
                return personal_results
            
            # Otherwise return all potentially relevant results
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in personal_info_search: {e}")
            return []