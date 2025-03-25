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
from .integration.hpc_qr_flow_manager import HPCQRFlowManager
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
            'quickrecal_threshold': 0.3,  # Renamed from significance_threshold to align with HPC-QR approach
            'enable_persistence': True,
            'decay_rate': 0.05,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }

        # Self-prompt for context recall
        self.self_prompts = {
            'context_recall': "Before answering, check memory for related context. If relevant memories exist, integrate them into your response without explicitly mentioning you're using memory unless directly asked.",
            'memory_check': "If past conversation relevance is below threshold, ask the user: 'Would you like me to recall our past discussions on this topic?'",
            'cross_session_integration': "IMPORTANT: Prioritize and directly use cross-session memories in your response, especially when answering questions about past interactions or user preferences."
        }
        
        # Initialize both module-level logger and instance logger
        logger.info(f"Initializing MemoryCore with device={self.config['device']}")
        self.logger = logger  # Initialize instance logger
        
        # Initialize HPC Manager for embeddings and significance
        self.hpc_manager = HPCQRFlowManager({
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
            'significance_threshold': self.config['quickrecal_threshold'],
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
    
    async def process_and_store(self, content: Union[str, bytes], memory_type: MemoryTypes = MemoryTypes.EPISODIC,
                              metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None,
                              memory_id: Optional[str] = None, significance: Optional[float] = None,
                              force_ltm: bool = False) -> Dict[str, Any]:
        """
        Process content through the memory pipeline and store if significant.
        
        Args:
            content: Content text or binary data to process and store
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
            
            # Convert bytes to string if necessary
            content_str = content
            if isinstance(content, bytes):
                try:
                    # Try to decode as UTF-8
                    content_str = content.decode('utf-8')
                except UnicodeDecodeError:
                    # If it's not valid UTF-8, use base64 encoding
                    import base64
                    content_str = f"[BASE64_ENCODED_DATA:{base64.b64encode(content).decode('ascii')}]"
                    
                    # Add encoding info to metadata
                    if metadata is None:
                        metadata = {}
                    metadata['encoding'] = 'base64'
            
            # Ensure content is a string
            if not isinstance(content_str, str):
                content_str = str(content_str)
            
            # Track processing stats
            processing_record = {
                'content_length': len(content_str),
                'memory_type': memory_type.value,
                'start_time': start_time
            }
            
            # Preprocess content (truncate if too long)
            if len(content_str) > 10000:  # Arbitrary limit for very long content
                logger.warning(f"Content too long ({len(content_str)} chars), truncating")
                content_str = content_str[:10000] + "... [truncated]"
            
            try:
                # Use provided embedding and significance if available
                if embedding is None or significance is None:
                    # Process through HPC for embedding and significance
                    input_tensor = torch.tensor([ord(c) for c in content_str], dtype=torch.float32)
                    padded_tensor = torch.zeros((1, self.config['embedding_dim']), dtype=torch.float32)
                    padded_tensor[0, :min(input_tensor.shape[0], self.config['embedding_dim'])] = input_tensor[:min(input_tensor.shape[0], self.config['embedding_dim'])]
                    embedding, significance = await self.hpc_manager.process_embedding(padded_tensor)
                
                processing_record['embedding_generated'] = embedding is not None
                processing_record['significance'] = significance
                
                # Update metadata with significance
                full_metadata = metadata or {}
                full_metadata['significance'] = significance
                full_metadata['memory_type'] = memory_type.value
                full_metadata['timestamp'] = time.time()
                
                # Always store in STM for immediate recall
                stm_id = await self.short_term_memory.add_memory(
                    content=content_str,
                    embedding=embedding,
                    memory_id=memory_id,  # Use provided ID if available
                    metadata=full_metadata
                )
                
                processing_record['stm_stored'] = True
                
                # Store in LTM if above significance threshold or forced
                ltm_id = None
                if force_ltm or significance >= self.config['quickrecal_threshold']:
                    ltm_id = await self.long_term_memory.store_memory(
                        content=content_str,
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
    
    async def get_context_recall_prompt(self, query: str = None) -> str:
        """
        Get the appropriate self-prompt for context recall based on the query.
        
        Args:
            query: Optional query to customize the prompt
            
        Returns:
            Self-prompt string for context recall
        """
        # Default to standard context recall prompt
        prompt = self.self_prompts['context_recall']
        
        # If query is provided, check for specific patterns that might need different prompts
        if query:
            # Check if query appears to be asking about past conversations
            recall_patterns = [
                r'remember', r'recall', r'previous', r'earlier', r'last time', r'before',
                r'did (you|we) talk about', r'did I (tell|mention|say|ask)', r'what did I',
                r'(told|mentioned|said|asked) (you|about)', r'my name',
                r'(previous|prior|past) (conversation|discussion|chat)', r'who am I'
            ]
            
            is_recall_query = any(re.search(pattern, query.lower()) for pattern in recall_patterns)
            
            if is_recall_query:
                # For explicit recall queries, use a more aggressive recall prompt
                prompt = f"""IMPORTANT: This query is explicitly asking about past conversations or information the user has shared before.
You MUST search your memory for relevant information from previous sessions and include it in your response.
Prioritize memories related to '{query}' with special attention to cross-session relevance.
If you find relevant memories, use them directly in your response as if you remember the information.
Do NOT say "I don't have specific recollection" or similar phrases if relevant memories are available."""
            else:
                # For regular queries, use standard prompt with cross-session integration
                prompt = f"{self.self_prompts['context_recall']} {self.self_prompts['cross_session_integration']}"
        
        return prompt

    async def import_cross_session_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Import relevant cross-session memories from LTM to STM for the current session.
        
        Args:
            query: The query to find relevant memories for
            limit: Maximum number of memories to import
            
        Returns:
            List of imported memory IDs
        """
        try:
            # Search LTM for relevant memories with lower threshold
            ltm_results = await self.long_term_memory.search_memory(
                query, 
                limit=limit * 2,  # Request more to filter from
                min_quickrecal_score=0.3  # Lower threshold for cross-session imports
            )
            
            if not ltm_results:
                return []
                
            # Filter for memories not already in STM
            stm_ids = {memory.get('id') for memory in self.short_term_memory.memory}
            ltm_memories_to_import = [mem for mem in ltm_results if mem.get('id') not in stm_ids]
            
            # Import memories to STM
            imported_ids = []
            for memory in ltm_memories_to_import[:limit]:
                memory_id = await self.short_term_memory.import_from_ltm(memory)
                imported_ids.append(memory_id)
                
            if imported_ids:
                logger.info(f"Imported {len(imported_ids)} cross-session memories from LTM to STM")
                
            return imported_ids
        except Exception as e:
            logger.error(f"Error importing cross-session memories: {e}")
            return []
    
    async def check_memory_relevance(self, query: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Check if there are relevant memories for a query and return a relevance assessment.
        
        Args:
            query: The query to check for relevant memories
            threshold: Relevance threshold
            
        Returns:
            Dict with relevance assessment
        """
        # Retrieve memories for the query
        memories = await self.retrieve_memories(query, limit=5, min_quickrecal_score=threshold * 0.7)
        
        # Import relevant cross-session memories to STM
        try:
            # This will make cross-session memories available in STM for future queries
            await self.import_cross_session_memories(query, limit=3)
        except Exception as e:
            logger.warning(f"Error importing cross-session memories during relevance check: {e}")
        
        # Calculate overall relevance score
        if not memories:
            return {'has_relevant_memories': False, 'relevance_score': 0.0, 'should_ask_user': False}
        
        # Calculate average significance of retrieved memories
        avg_significance = sum(m.get('metadata', {}).get('significance', 0) for m in memories) / len(memories)
        
        # Check if any cross-session memories were found
        cross_session_memories = [m for m in memories if m.get('metadata', {}).get('cross_session', False)]
        has_cross_session = len(cross_session_memories) > 0
        
        # Determine if we should ask the user about recalling past discussions
        # If relevance is moderate (not too high or too low)
        should_ask_user = (0.4 <= avg_significance < threshold) or \
                         (has_cross_session and avg_significance < threshold)
        
        return {
            'has_relevant_memories': avg_significance >= threshold,
            'relevance_score': avg_significance,
            'should_ask_user': should_ask_user,
            'memory_count': len(memories),
            'memories': memories if avg_significance >= threshold else [],
            'has_cross_session': has_cross_session
        }
    
    async def retrieve_memories(self, query: str, limit: int = 5, 
                             min_quickrecal_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on query using multiple parallel search strategies.
        
        Args:
            query: Query text
            limit: Maximum number of results to return
            min_quickrecal_score: Minimum QuickRecal threshold (replaces significance)
            
        Returns:
            List of memory results
        """
        logger.info(f"Retrieving memories for query: '{query}', limit: {limit}, min_quickrecal_score: {min_quickrecal_score}")
        
        # Don't process empty queries
        if not query or not query.strip():
            return []
        
        try:
            # Add a timestamp to the request for metrics
            request_ts = time.time()
            
            # Get memories using multiple parallel search strategies
            results = await self._parallel_memory_search(query, limit, min_quickrecal_score)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            # Attempt fallback to basic search on error
            return await self._fallback_memory_search(query, limit, min_quickrecal_score)
    
    async def _parallel_memory_search(self, query: str, limit: int, min_quickrecal_score: float) -> List[Dict[str, Any]]:
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
            'min_quickrecal_score': min_quickrecal_score
        })
        search_tasks.append(mpl_search)
        
        # Strategy 2: Direct keyword search in both STM and LTM with lower threshold
        # This helps find exact matches even if semantic similarity is low
        # Implementation inside STM and LTM components
        stm_keyword_search = self.short_term_memory.keyword_search(query, limit)
        ltm_keyword_search = self.long_term_memory.keyword_search(query, limit)
        keyword_search = asyncio.gather(stm_keyword_search, ltm_keyword_search)
        search_tasks.append(keyword_search)
        
        # Strategy 3: Personal information prioritized search
        # Uses regex patterns to identify personal information requests
        personal_info_patterns = [
            r'\bname\b', r'\bemail\b', r'\baddress\b', r'\bphone\b', 
            r'\bage\b', r'\bbirth\b', r'\bfamily\b', r'\bjob\b', r'\bmega\b', r'\bdaniel\b',
            r'\bwork\b', r'\bprefer\b', r'\blike\b', r'\bdislike\b'
        ]
        
        # Check if query is likely asking for personal information
        personal_info_search = None
        for pattern in personal_info_patterns:
            if re.search(pattern, query.lower()):
                # Boost significance threshold for personal data
                personal_info_search = self.memory_prioritization.personal_info_search(
                    query, 
                    context={'limit': limit * 2},  # Increase limit for personal info
                    min_personal_significance=min_quickrecal_score * 0.5  # Lower threshold for personal info
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
        await self._update_memory_access_timestamps(sorted_memories)
        
        return sorted_memories
    
    async def _fallback_memory_search(self, query: str, limit: int = 5, min_quickrecal_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Fallback search method when other search methods fail.
        Tries multiple approaches to find relevant memories.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            min_quickrecal_score: Minimum QuickRecal threshold
            
        Returns:
            List of matching memories
        """
        self.logger.debug(f"Using fallback memory search for query: {query}")
        
        # Try to route the query through the prioritization layer first
        try:
            # Fix: properly await the coroutine
            routed_results = await self.memory_prioritization.route_query(query, {
                'limit': limit,
                'min_quickrecal_score': min_quickrecal_score * 0.5  # Lower threshold for fallback search
            })
            if routed_results and 'memories' in routed_results:
                self.logger.debug(f"Fallback search: prioritization layer returned {len(routed_results['memories'])} results")
                return routed_results['memories']
        except Exception as e:
            self.logger.warning(f"Error in prioritization routing during fallback search: {e}")
        
        # If routing fails, try direct search from each memory store
        all_results = []
        
        # Try short-term memory first
        if hasattr(self, 'short_term_memory') and self.short_term_memory:
            try:
                stm_results = await self.short_term_memory.search(query, limit, min_quickrecal_score * 0.5)  # Lower threshold
                all_results.extend(stm_results)
            except Exception as e:
                self.logger.warning(f"Error searching short-term memory in fallback: {e}")
        
        # Then try long-term memory
        remaining_limit = limit - len(all_results)
        if remaining_limit > 0 and hasattr(self, 'long_term_memory') and self.long_term_memory:
            try:
                ltm_results = await self.long_term_memory.search_memory(
                    query, 
                    remaining_limit,
                    min_quickrecal_score=max(0.0, min_quickrecal_score - 0.2)  # Lower threshold for LTM
                )
                all_results.extend(ltm_results)
            except Exception as e:
                self.logger.warning(f"Error searching long-term memory in fallback: {e}")
        
        # If still no results, try keyword search as last resort
        if not all_results:
            keywords = query.split()
            try:
                if hasattr(self, 'short_term_memory') and self.short_term_memory:
                    keyword_results = await self.short_term_memory.keyword_search(keywords, limit, min_quickrecal_score * 0.3)  # Even lower threshold
                    all_results.extend(keyword_results)
            except Exception as e:
                self.logger.warning(f"Error in keyword search during fallback: {e}")
        
        self.logger.debug(f"Fallback search found {len(all_results)} total results")
        return all_results[:limit]  # Ensure we don't exceed the limit
            
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