"""
LUCID RECALL PROJECT
Long-Term Memory (LTM) with Asynchronous Batch Persistence

Persistent significance-weighted storage where only important memories remain long-term.
Implements dynamic significance decay to ensure only critical memories persist.
Features fully asynchronous memory persistence with efficient batch processing.
"""

import time
import math
import logging
import asyncio
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pathlib import Path
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Enum for batch operation types."""
    STORE = 1
    UPDATE = 2
    PURGE = 3

class BatchOperation:
    """Represents a single operation in the batch queue."""
    def __init__(self, op_type: OperationType, memory_id: str, data: Optional[Dict[str, Any]] = None):
        self.op_type = op_type
        self.memory_id = memory_id
        self.data = data
        self.timestamp = time.time()

class LongTermMemory:
    """
    Long-Term Memory with significance-weighted storage and dynamic decay.
    
    Stores memories persistently with significance weighting to ensure
    only important memories are retained long-term. Implements dynamic
    significance decay to allow unimportant memories to fade naturally.
    Features fully asynchronous batch persistence for improved performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the long-term memory system.
        
        Args:
            config: Configuration options
        """
        self.config = {
            'storage_path': os.path.join('/app/memory/stored', 'ltm'),  # Use the consistent Docker path
            'significance_threshold': 0.7,  # Minimum significance for storage
            'max_memories': 10000,          # Maximum number of memories to store
            'decay_rate': 0.05,             # Base decay rate (per day)
            'decay_check_interval': 86400,  # Time between decay checks (1 day)
            'min_retention_time': 604800,   # Minimum retention time regardless of decay (1 week)
            'embedding_dim': 384,           # Embedding dimension
            'enable_persistence': True,     # Whether to persist memories to disk
            'purge_threshold': 0.3,         # Memories below this significance get purged
            
            # Batch persistence configuration
            'batch_size': 50,               # Max operations in a batch
            'batch_interval': 5.0,          # Max seconds between batch processing
            'batch_retries': 3,             # Number of retries for failed batch operations
            'batch_retry_delay': 1.0,       # Delay between retries (seconds)
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
        self._batch_lock = asyncio.Lock()
        
        # Batch persistence
        self._batch_queue = deque()
        self._batch_processing = False
        self._batch_event = asyncio.Event()
        self._shutdown = False
        
        # Performance stats
        self.stats = {
            'stores': 0,
            'retrievals': 0,
            'purges': 0,
            'hits': 0,
            'last_decay_check': time.time(),
            'last_backup': time.time(),
            'batch_operations': 0,
            'batch_successes': 0,
            'batch_failures': 0,
            'avg_batch_size': 0,
            'total_batches': 0,
            'largest_batch': 0
        }
        
        # Start background tasks
        self._tasks = []
        
        # Load existing memories
        self._load_memories()
        
        # Start batch processing task if persistence is enabled
        if self.config['enable_persistence']:
            self._tasks.append(asyncio.create_task(self._batch_processor()))
            logger.info("Started batch persistence processor")
        
        logger.info(f"Initialized LongTermMemory with {len(self.memories)} memories")
    
    async def shutdown(self):
        """
        Safely shut down the LongTermMemory system.
        
        Processes any remaining items in the batch queue and stops background tasks.
        """
        logger.info("Shutting down LongTermMemory system")
        
        # Signal shutdown to prevent new batches from being queued
        self._shutdown = True
        
        if self.config['enable_persistence']:
            # Process any remaining items in the batch queue
            if self._batch_queue:
                logger.info(f"Processing {len(self._batch_queue)} remaining items in batch queue")
                self._batch_event.set()
                
                # Wait a reasonable time for batch processing to complete
                for _ in range(10):
                    if not self._batch_queue:
                        break
                    await asyncio.sleep(0.5)
            
            # Forcibly process any remaining items
            if self._batch_queue:
                logger.warning(f"Force processing {len(self._batch_queue)} items in batch queue")
                await self._process_batch(force=True)
        
        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("LongTermMemory shutdown complete")
        
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
            
            # Add to batch queue for persistence
            if self.config['enable_persistence']:
                await self._add_to_batch_queue(OperationType.STORE, memory_id, memory)
            
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
            
            # Add to batch queue for persistence (update operation)
            if self.config['enable_persistence']:
                await self._add_to_batch_queue(OperationType.UPDATE, memory_id, memory)
            
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
        
        # Add to batch queue for persistence (purge operation)
        if self.config['enable_persistence']:
            await self._add_to_batch_queue(OperationType.PURGE, memory_id)
    
    async def _add_to_batch_queue(self, op_type: OperationType, memory_id: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an operation to the batch processing queue.
        
        Args:
            op_type: Type of operation (STORE, UPDATE, PURGE)
            memory_id: ID of the memory
            data: Memory data for STORE and UPDATE operations
        """
        if not self.config['enable_persistence'] or self._shutdown:
            return
            
        async with self._batch_lock:
            # Create batch operation
            operation = BatchOperation(op_type, memory_id, data)
            
            # Add to queue
            self._batch_queue.append(operation)
            
            # Signal the batch processor if queue exceeds batch size
            if len(self._batch_queue) >= self.config['batch_size']:
                self._batch_event.set()
    
    async def _batch_processor(self) -> None:
        """
        Background task to process batches of memory operations.
        """
        logger.info("Starting batch processor task")
        
        last_process_time = time.time()
        
        while not self._shutdown:
            try:
                # Wait for either:
                # 1. Batch size threshold to be reached (signaled by _batch_event)
                # 2. Batch interval timeout
                try:
                    batch_interval = self.config['batch_interval']
                    await asyncio.wait_for(self._batch_event.wait(), timeout=batch_interval)
                except asyncio.TimeoutError:
                    # Timeout occurred, check if we have any operations to process
                    pass
                finally:
                    # Clear the event for next time
                    self._batch_event.clear()
                
                # Check if we should process the batch
                current_time = time.time()
                time_since_last_process = current_time - last_process_time
                
                if (len(self._batch_queue) > 0 and 
                    (len(self._batch_queue) >= self.config['batch_size'] or 
                     time_since_last_process >= self.config['batch_interval'])):
                    
                    # Process the batch
                    await self._process_batch()
                    last_process_time = time.time()
                
            except asyncio.CancelledError:
                logger.info("Batch processor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def _process_batch(self, force: bool = False) -> None:
        """
        Process a batch of memory operations.
        
        Args:
            force: If True, process all operations regardless of batch settings
        """
        if not self.config['enable_persistence']:
            return
            
        # Skip if no operations or already processing (unless forced)
        if (not self._batch_queue) or (self._batch_processing and not force):
            return
            
        async with self._batch_lock:
            self._batch_processing = True
            
            try:
                # Determine batch size
                batch_size = len(self._batch_queue) if force else min(len(self._batch_queue), self.config['batch_size'])
                
                # Update stats
                self.stats['batch_operations'] += batch_size
                self.stats['total_batches'] += 1
                self.stats['avg_batch_size'] = self.stats['batch_operations'] / self.stats['total_batches']
                self.stats['largest_batch'] = max(self.stats['largest_batch'], batch_size)
                
                logger.debug(f"Processing batch of {batch_size} operations")
                
                # Group operations by type for efficient processing
                store_ops = []
                update_ops = []
                purge_ops = []
                
                # Extract batch operations from queue
                operations = []
                for _ in range(batch_size):
                    if not self._batch_queue:
                        break
                    operations.append(self._batch_queue.popleft())
                
                # Group by operation type
                for op in operations:
                    if op.op_type == OperationType.STORE:
                        store_ops.append(op)
                    elif op.op_type == OperationType.UPDATE:
                        update_ops.append(op)
                    elif op.op_type == OperationType.PURGE:
                        purge_ops.append(op)
                
                # Process operations by type
                store_results = await self._process_store_batch(store_ops)
                update_results = await self._process_update_batch(update_ops)
                purge_results = await self._process_purge_batch(purge_ops)
                
                # Combine results
                success_count = store_results + update_results + purge_results
                
                # Update stats
                self.stats['batch_successes'] += success_count
                self.stats['batch_failures'] += batch_size - success_count
                
                logger.debug(f"Batch processing complete: {success_count}/{batch_size} operations successful")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=True)
            finally:
                self._batch_processing = False
    
    async def _process_store_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process a batch of store operations.
        
        Args:
            operations: List of store operations
            
        Returns:
            Number of successful operations
        """
        if not operations:
            return 0
            
        success_count = 0
        
        try:
            # Group memory data by ID
            memories_to_store = {}
            for op in operations:
                if op.data:
                    memories_to_store[op.memory_id] = op.data
            
            # Process each memory
            for memory_id, memory in memories_to_store.items():
                try:
                    memory_copy = memory.copy()
                    
                    # Convert embedding to list if it's a tensor
                    if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                        memory_copy['embedding'] = memory_copy['embedding'].tolist()
                    
                    # Write to file
                    file_path = self.storage_path / f"{memory_id}.json"
                    with open(file_path, 'w') as f:
                        json.dump(memory_copy, f, indent=2)
                        
                    success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error storing memory {memory_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error in batch store operation: {e}", exc_info=True)
            
        return success_count
    
    async def _process_update_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process a batch of update operations.
        
        Args:
            operations: List of update operations
            
        Returns:
            Number of successful operations
        """
        # For now, update operations are the same as store operations
        # We could optimize this in the future to only update changed fields
        return await self._process_store_batch(operations)
    
    async def _process_purge_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process a batch of purge operations.
        
        Args:
            operations: List of purge operations
            
        Returns:
            Number of successful operations
        """
        if not operations:
            return 0
            
        success_count = 0
        
        try:
            # Group by memory ID to avoid duplicate operations
            memory_ids = set(op.memory_id for op in operations)
            
            # Process each memory ID
            for memory_id in memory_ids:
                try:
                    file_path = self.storage_path / f"{memory_id}.json"
                    if file_path.exists():
                        os.remove(file_path)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error purging memory {memory_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error in batch purge operation: {e}", exc_info=True)
            
        return success_count
    
    async def backup(self) -> bool:
        """
        Create a backup of all memories.
        
        Returns:
            Success status
        """
        if not self.config['enable_persistence']:
            return False
        
        # Process any pending operations first
        if self._batch_queue:
            await self._process_batch(force=True)
        
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
        
        # Batch persistence stats
        batch_stats = {
            'batch_operations': self.stats['batch_operations'],
            'batch_successes': self.stats['batch_successes'],
            'batch_failures': self.stats['batch_failures'],
            'avg_batch_size': self.stats['avg_batch_size'],
            'total_batches': self.stats['total_batches'],
            'largest_batch': self.stats['largest_batch'],
            'queued_operations': len(self._batch_queue) if hasattr(self, '_batch_queue') else 0,
            'batch_success_rate': (self.stats['batch_successes'] / max(1, self.stats['batch_operations'])) * 100
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
            'storage_utilization': len(self.memories) / self.config['max_memories'],
            'batch_persistence': batch_stats
        }