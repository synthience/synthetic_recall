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
import re

# Import memory components
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory
from .memory_prioritization_layer import MemoryPrioritizationLayer
from .integration.hpc_sig_flow_manager import HPCSIGFlowManager
from .memory_types import MemoryTypes, MemoryEntry

logger = logging.getLogger(__name__)

class MemoryCore:
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
            'memory_path': Path('/app/memory/stored'),  # Use the consistent Docker path
            'stm_max_size': 10,
            'significance_threshold': 0.3,
            'enable_persistence': True,
            'decay_rate': 0.05,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        logger.info(f"Initializing MemoryCore with device={self.config['device']}")
        
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
        
        logger.info("MemoryCore initialized")
    
    async def process_and_store(self, content: str, memory_type: MemoryTypes = MemoryTypes.EPISODIC,
                              metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None,
                              memory_id: Optional[str] = None, significance: Optional[float] = None,
                              force_ltm: bool = False) -> Dict[str, Any]:
        """
        Process content through the memory pipeline and store if significant.
        
        Args:
            content: Content text to process and store
            memory_type: Type of memory (EPISODIC, SEMANTIC, etc.)
            metadata: Additional metadata about the memory
            embedding: Optional pre-computed embedding vector
            memory_id: Optional specific memory ID to use
            significance: Optional pre-computed significance score
            force_ltm: Force storage in LTM regardless of significance
            
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
                # Use provided embedding and significance if available
                if embedding is None or significance is None:
                    # Process through HPC for embedding and significance
                    embedding, significance = await self.hpc_manager.process_embedding(
                        torch.tensor(content.encode(), dtype=torch.float32).reshape(1, -1)
                    )
                
                processing_record['embedding_generated'] = embedding is not None
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
                    memory_id=memory_id,  # Use provided ID if available
                    metadata=full_metadata
                )
                
                processing_record['stm_stored'] = True
                
                # Store in LTM if above significance threshold or forced
                ltm_id = None
                if force_ltm or significance >= self.config['significance_threshold']:
                    ltm_id = await self.long_term_memory.store_memory(
                        content=content,
                        embedding=embedding,
                        memory_id=memory_id,  # Use provided ID if available
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
        Retrieve memories based on query using multiple parallel search strategies.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of memory results
        """
        try:
            # Implement parallel search with multiple strategies
            results = await self._parallel_memory_search(
                query=query,
                limit=limit,
                min_significance=min_significance
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            # Attempt fallback to basic search on error
            return await self._fallback_memory_search(query, limit, min_significance)
    
    async def _parallel_memory_search(self, query: str, limit: int, min_significance: float) -> List[Dict[str, Any]]:
        """
        Execute multiple search strategies in parallel for optimal retrieval.
        
        Implements different search strategies:
        1. Semantic search via MPL (primary)
        2. Direct keyword search
        3. Personal information prioritized search
        4. Recency-weighted search
        """
        search_tasks = []
        loop = asyncio.get_event_loop()
        
        # Strategy 1: Normal MPL-routed search (semantic)
        mpl_search = self.memory_prioritization.route_query(query, {
            'limit': limit * 2,  # Request more results to filter from
            'min_significance': min_significance
        })
        search_tasks.append(mpl_search)
        
        # Strategy 2: Direct keyword search in both STM and LTM
        # This helps find exact matches even if semantic similarity is low
        # Implementation inside STM and LTM components
        keyword_search = asyncio.gather(
            self.short_term_memory.keyword_search(query, limit),
            self.long_term_memory.keyword_search(query, limit)
        )
        search_tasks.append(keyword_search)
        
        # Strategy 3: Personal information prioritized search
        # Uses regex patterns to identify personal information requests
        personal_info_patterns = [
            r'\bname\b', r'\bemail\b', r'\baddress\b', r'\bphone\b', 
            r'\bage\b', r'\bbirth\b', r'\bfamily\b', r'\bjob\b',
            r'\bwork\b', r'\bprefer\b', r'\blike\b', r'\bdislike\b'
        ]
        
        # Check if query is likely asking for personal information
        personal_info_search = None
        for pattern in personal_info_patterns:
            if re.search(pattern, query.lower()):
                # Boost significance threshold for personal data
                personal_info_search = self.memory_prioritization.personal_info_search(
                    query, limit, min_personal_significance=0.2
                )
                search_tasks.append(personal_info_search)
                break
        
        # Strategy 4: Recency-weighted search
        # Prioritize recent memories with adjusted significance
        recency_search = self.short_term_memory.recency_biased_search(
            query, limit=limit, recency_weight=0.7
        )
        search_tasks.append(recency_search)
        
        # Execute all search strategies in parallel
        search_timeout = 2.0  # 2-second timeout for search operations
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=search_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Memory search timed out after {search_timeout}s")
            # Get whatever results have completed
            done, pending = await asyncio.wait(search_tasks, timeout=0)
            results = [task.result() if not isinstance(task, Exception) and task.done() 
                      else None for task in done]
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
        
        # Process results from different strategies
        all_memories = []
        
        # Process MPL results
        if results[0] and not isinstance(results[0], Exception):
            all_memories.extend(results[0].get('memories', []))
        
        # Process keyword search results from both STM and LTM
        if results[1] and not isinstance(results[1], Exception):
            stm_results, ltm_results = results[1]
            if stm_results:
                all_memories.extend(stm_results)
            if ltm_results:
                all_memories.extend(ltm_results)
        
        # Process personal info results if available
        if personal_info_search is not None and len(results) > 2:
            if results[2] and not isinstance(results[2], Exception):
                # Prioritize personal info results
                personal_memories = results[2]
                if personal_memories:
                    # Boost significance of personal memories
                    for memory in personal_memories:
                        if memory not in all_memories:
                            # Ensure personal memories have high significance
                            if 'metadata' in memory and 'significance' in memory['metadata']:
                                memory['metadata']['significance'] = max(
                                    memory['metadata']['significance'],
                                    0.8  # Ensure high significance for personal info
                                )
                            all_memories.append(memory)
        
        # Process recency search results
        recency_idx = 3 if personal_info_search is not None else 2
        if len(results) > recency_idx and results[recency_idx] and not isinstance(results[recency_idx], Exception):
            recency_memories = results[recency_idx]
            # Add only new memories not already in the list
            for memory in recency_memories:
                if memory not in all_memories:
                    all_memories.append(memory)
        
        # Deduplicate based on memory_id
        unique_memories = {}
        for memory in all_memories:
            memory_id = memory.get('id')
            if memory_id:
                # If duplicate, keep the one with higher significance
                if memory_id in unique_memories:
                    current_sig = unique_memories[memory_id].get('metadata', {}).get('significance', 0)
                    new_sig = memory.get('metadata', {}).get('significance', 0)
                    if new_sig > current_sig:
                        unique_memories[memory_id] = memory
                else:
                    unique_memories[memory_id] = memory
                    
        # Sort by significance and limit results
        sorted_memories = sorted(
            unique_memories.values(),
            key=lambda x: x.get('metadata', {}).get('significance', 0),
            reverse=True
        )[:limit]
        
        # Update access timestamps for retrieved memories to boost future retrievals
        self._update_memory_access_timestamps(sorted_memories)
        
        return sorted_memories
    
    async def _fallback_memory_search(self, query: str, limit: int, min_significance: float) -> List[Dict[str, Any]]:
        """Fallback search method when primary methods fail"""
        try:
            # First try direct STM retrieval (fast, in-memory)
            stm_results = await self.short_term_memory.search(query, limit)
            if stm_results and len(stm_results) > 0:
                return stm_results
                
            # If no STM results, try LTM with lower significance threshold
            ltm_results = await self.long_term_memory.search(
                query, 
                limit=limit,
                min_significance=max(0.0, min_significance - 0.2)  # Lower threshold
            )
            if ltm_results and len(ltm_results) > 0:
                return ltm_results
                
            # Last resort: return most recent memories regardless of query match
            logger.warning("Fallback to most recent memories regardless of query")
            recent_memories = await self.short_term_memory.get_recent_memories(limit)
            if not recent_memories:
                recent_memories = await self.long_term_memory.get_recent_memories(limit)
                
            return recent_memories or []
            
        except Exception as e:
            logger.error(f"Error in fallback memory search: {e}")
            return []  # Return empty list as last resort
            
    async def _update_memory_access_timestamps(self, memories: List[Dict[str, Any]]):
        """Update access timestamps for retrieved memories to boost future relevance"""
        try:
            for memory in memories:
                memory_id = memory.get('id')
                if not memory_id:
                    continue
                    
                # Update in STM first (if present)
                stm_updated = await self.short_term_memory.update_access_timestamp(memory_id)
                
                # If not in STM, update in LTM
                if not stm_updated:
                    await self.long_term_memory.update_access_timestamp(memory_id)
                    
        except Exception as e:
            logger.warning(f"Failed to update memory access timestamps: {e}")
            # Non-critical operation, so we just log and continue
    
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