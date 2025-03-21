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
    Long-Term Memory (LTM) with QuickRecal-weighted storage and dynamic decay.

    Stores memories persistently with QuickRecal weighting to ensure
    only important memories remain. Implements dynamic QuickRecal decay
    to allow unimportant memories to fade naturally, with both scheduled
    and on-demand triggers.

    Features fully asynchronous batch persistence and retry logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the long-term memory system.

        Args:
            config: Configuration options.
        """
        self.config = {
            'storage_path': os.path.join('/app/memory/stored', 'ltm'),  # Consistent Docker path
            'quickrecal_threshold': 0.7,    # Minimum QuickRecal score for storage
            'max_memories': 10000,          # Maximum number of memories to store
            'decay_rate': 0.05,             # Base decay rate (per day)
            'decay_check_interval': 86400,  # Time between decay checks (seconds) => 1 day
            'min_retention_time': 604800,   # Minimum retention time (seconds) => 1 week
            'embedding_dim': 384,           # Embedding dimension
            'enable_persistence': True,     # Whether to persist memories to disk
            'purge_threshold': 0.3,         # Memories below this QuickRecal score get purged

            # Batch persistence configuration
            'batch_size': 50,               # Max operations in a batch
            'batch_interval': 5.0,          # Seconds between batch processing
            'batch_retries': 3,             # Number of retries for failed batch operations
            'batch_retry_delay': 1.0,       # Delay between retries in seconds

            # (Optional) Whether to enable the scheduled decay process in the background
            'enable_scheduled_decay': True,
            **(config or {}),
        }

        # Ensure storage path exists
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory store of all loaded (and newly created) memories
        self.memories = {}         # memory_id -> memory_dict
        self.memory_index = {}     # category -> [memory_ids]

        # Thread-safety locks
        self._lock = asyncio.Lock()
        self._batch_lock = asyncio.Lock()

        # Batch queue and related flags
        self._batch_queue = deque()
        self._batch_processing = False
        self._batch_event = asyncio.Event()
        self._shutdown = False

        # Performance and operational stats
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

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Load existing memories from disk
        self._load_memories()

        # If persistence is enabled, start batch processor
        if self.config['enable_persistence']:
            self._tasks.append(asyncio.create_task(self._batch_processor()))
            logger.info("Started batch persistence processor")

        # Optionally, start a scheduled decay background task
        if self.config.get('enable_scheduled_decay', True):
            self._tasks.append(asyncio.create_task(self._decay_scheduler()))
            logger.info("Started scheduled decay task")

        logger.info(f"Initialized LongTermMemory with {len(self.memories)} memories")

    async def shutdown(self):
        """
        Safely shut down the LongTermMemory system.

        1. Set the shutdown flag to prevent new ops from being queued.
        2. Process any remaining items in the batch queue.
        3. Cancel all background tasks (batch processor, decayer, etc.).
        """
        logger.info("Shutting down LongTermMemory system")

        # Signal shutdown to prevent new queues
        self._shutdown = True

        # Process remaining items if persistence is enabled
        if self.config['enable_persistence']:
            if self._batch_queue:
                logger.info(f"Processing {len(self._batch_queue)} remaining items in batch queue before shutdown")
                self._batch_event.set()

                # Wait briefly for batch processing
                for _ in range(10):
                    if not self._batch_queue:
                        break
                    await asyncio.sleep(0.5)

            # If still not empty, force processing
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
        """
        Load memories from persistent storage (disk).
        """
        if not self.config['enable_persistence']:
            return

        try:
            logger.info(f"Loading memories from {self.storage_path}")
            memory_files = list(self.storage_path.glob('*.json'))
            if not memory_files:
                logger.info("No memory files found on disk.")
                return

            for file_path in memory_files:
                try:
                    with open(file_path, 'r') as f:
                        memory = json.load(f)

                    # Support 'text' key as an alias for 'content'
                    if 'text' in memory and 'content' not in memory:
                        memory['content'] = memory['text']
                        logger.debug(f"Converted 'text' to 'content' in memory {file_path.name}")

                    # Validate minimal fields
                    if not all(k in memory for k in ['id', 'content', 'timestamp']):
                        logger.warning(f"Invalid memory format in {file_path}, skipping. Missing required keys: {[k for k in ['id', 'content', 'timestamp'] if k not in memory]}")
                        continue

                    # Convert embedding from list to tensor if present
                    if 'embedding' in memory and isinstance(memory['embedding'], list):
                        memory['embedding'] = torch.tensor(memory['embedding'], dtype=torch.float32)

                    memory_id = memory['id']
                    self.memories[memory_id] = memory

                    # Index by category
                    category = memory.get('metadata', {}).get('category', 'general')
                    if category not in self.memory_index:
                        self.memory_index[category] = []
                    self.memory_index[category].append(memory_id)

                except Exception as e:
                    logger.error(f"Error loading memory from {file_path}: {e}")

            logger.info(f"Loaded {len(self.memories)} memories from disk.")

        except Exception as e:
            logger.error(f"Error loading memories: {e}", exc_info=True)

    async def store_memory(
        self,
        content: str,
        embedding: Optional[torch.Tensor] = None,
        quickrecal_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Store a memory in LTM if it meets the QuickRecal threshold.

        Args:
            content: The memory content text
            embedding: Optional pre-computed embedding (torch.Tensor)
            quickrecal_score: Memory QuickRecal score (0.0 - 1.0)
            metadata: Optional dict with additional metadata
            memory_id: Optional custom ID

        Returns:
            The memory_id if stored, or None if below QuickRecal threshold.
        """
        # QuickRecal threshold check
        if quickrecal_score < self.config['quickrecal_threshold']:
            logger.debug(
                f"Memory QuickRecal {quickrecal_score} < threshold {self.config['quickrecal_threshold']}; not storing."
            )
            return None

        async with self._lock:
            import uuid
            _id = memory_id or str(uuid.uuid4())
            timestamp = time.time()

            # Construct memory object
            memory = {
                'id': _id,
                'content': content,
                'embedding': embedding,
                'timestamp': timestamp,
                'quickrecal_score': quickrecal_score,
                'metadata': metadata or {},
                'access_count': 0,
                'last_access': timestamp
            }

            # Insert into in-memory structures
            self.memories[_id] = memory
            cat = memory['metadata'].get('category', 'general')
            if cat not in self.memory_index:
                self.memory_index[cat] = []
            self.memory_index[cat].append(_id)

            self.stats['stores'] += 1

            # Add to batch queue if persistence is on
            if self.config['enable_persistence']:
                await self._add_to_batch_queue(OperationType.STORE, _id, memory)

            # On-demand decay if above capacity
            if len(self.memories) > self.config['max_memories']:
                asyncio.create_task(self._run_decay_and_purge())

            logger.info(f"Stored memory {_id} with QuickRecal {quickrecal_score:.2f}")
            return _id

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID, updating QuickRecal and access metrics.

        Args:
            memory_id: ID of the memory

        Returns:
            A copy of the memory dict, or None if not found.
        """
        async with self._lock:
            self.stats['retrievals'] += 1

            if memory_id not in self.memories:
                return None

            memory = self.memories[memory_id]
            # Update access stats
            memory['access_count'] = memory.get('access_count', 0) + 1
            memory['last_access'] = time.time()

            # Boost QuickRecal based on usage
            self._boost_quickrecal_score(memory)

            # Queue an update operation (for persistence) if enabled
            if self.config['enable_persistence']:
                await self._add_to_batch_queue(OperationType.UPDATE, memory_id, memory)

            self.stats['hits'] += 1

            import copy
            return copy.deepcopy(memory)

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
        min_quickrecal_score: float = 0.0,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for memories by (very basic) text matching.

        Args:
            query: The query text
            limit: Max number of results
            min_quickrecal_score: Minimum QuickRecal score to qualify
            categories: Optional list of categories to restrict search

        Returns:
            List of memory dicts sorted by a simplistic "score".
        """
        async with self._lock:
            self.stats['retrievals'] += 1

            # Collect memory IDs from specified categories (or all)
            memory_ids = []
            if categories:
                for cat in categories:
                    memory_ids.extend(self.memory_index.get(cat, []))
            else:
                memory_ids = list(self.memories.keys())

            query_lower = query.lower()
            results = []
            for mid in memory_ids:
                mem = self.memories[mid]
                if mem.get('quickrecal_score', 0) < min_quickrecal_score:
                    continue

                content = mem.get('content', '').lower()
                # Basic token overlap
                tokens_content = set(content.split())
                tokens_query = set(query_lower.split())
                if tokens_content and tokens_query:
                    intersection = tokens_content.intersection(tokens_query)
                    union = tokens_content.union(tokens_query)
                    similarity = len(intersection) / len(union)
                else:
                    similarity = 0.0

                # Calculate effective QuickRecal
                eff_qr = self._calculate_effective_quickrecal_score(mem)
                # Weighted combination
                combined_score = (similarity * 0.7) + (eff_qr * 0.3)

                if combined_score > 0:
                    results.append({
                        'id': mid,
                        'content': mem.get('content', ''),
                        'timestamp': mem.get('timestamp', 0),
                        'similarity': similarity,
                        'quickrecal_score': eff_qr,
                        'score': combined_score,
                        'metadata': mem.get('metadata', {})
                    })

            # Sort by combined score
            results.sort(key=lambda x: x['score'], reverse=True)
            if results:
                self.stats['hits'] += 1

            return results[:limit]

    async def keyword_search(
        self,
        query: str,
        limit: int = 5,
        min_quickrecal_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search for memories.

        Args:
            query: Space-separated keywords
            limit: Max number of results
            min_quickrecal_score: Minimum QuickRecal threshold

        Returns:
            List of matching memories sorted by relevance.
        """
        async with self._lock:
            # Track usage
            self.stats['keyword_retrievals'] = self.stats.get('keyword_retrievals', 0) + 1

            keywords = set(query.lower().split())
            if not keywords:
                return []

            results = []
            for mid, mem in self.memories.items():
                # Skip if QuickRecal is too low
                if mem.get('quickrecal_score', 0) < min_quickrecal_score:
                    continue

                content = mem.get('content', '').lower()
                content_tokens = set(content.split())

                matching = keywords.intersection(content_tokens)
                if not matching:
                    continue

                keyword_score = len(matching) / len(keywords)
                eff_qr = self._calculate_effective_quickrecal_score(mem)

                combined_score = (keyword_score * 0.7) + (eff_qr * 0.3)
                results.append({
                    'id': mid,
                    'content': mem.get('content', ''),
                    'timestamp': mem.get('timestamp', 0),
                    'matching_keywords': list(matching),
                    'quickrecal_score': eff_qr,
                    'score': combined_score,
                    'metadata': mem.get('metadata', {})
                })

            results.sort(key=lambda x: x['score'], reverse=True)
            if results:
                self.stats['keyword_hits'] = self.stats.get('keyword_hits', 0) + 1

            return results[:limit]

    def _boost_quickrecal_score(self, memory: Dict[str, Any]) -> None:
        """
        Boost memory's QuickRecal score based on access frequency/recency.
        """
        access_count = memory.get('access_count', 1)
        current_time = time.time()
        last_access = memory.get('last_access', memory.get('timestamp', current_time))

        # Recency factor: decays with time since last access
        days_since_access = (current_time - last_access) / 86400
        recency_factor = math.exp(-0.1 * days_since_access)  # exponential

        # Access factor: up to 10 accesses considered
        access_factor = min(1.0, access_count / 10.0)

        # Combined boost
        boost_amount = 0.05 * recency_factor * access_factor
        memory['quickrecal_score'] = min(
            1.0,
            memory.get('quickrecal_score', 0.5) + boost_amount
        )

    def _calculate_effective_quickrecal_score(self, memory: Dict[str, Any]) -> float:
        """
        Compute effective QuickRecal score with time-based decay.
        """
        base_qr = memory.get('quickrecal_score', 0.5)
        timestamp = memory.get('timestamp', time.time())
        current_time = time.time()

        age_days = (current_time - timestamp) / 86400
        min_retention_days = self.config['min_retention_time'] / 86400

        # Within the min retention window => no decay
        if age_days < min_retention_days:
            return base_qr

        # More important memories decay more slowly
        importance_factor = 0.5 + (0.5 * base_qr)
        effective_decay_rate = self.config['decay_rate'] / importance_factor

        # Exponential decay after min_retention_days
        decay_factor = math.exp(-effective_decay_rate * (age_days - min_retention_days))
        return base_qr * decay_factor

    async def _run_decay_and_purge(self) -> None:
        """
        Decay memory QuickRecal scores and purge those below threshold
        or those exceeding max capacity, if needed.
        """
        async with self._lock:
            current_time = time.time()
            time_since_last_decay = current_time - self.stats['last_decay_check']

            # If not enough time has passed, skip
            if time_since_last_decay < self.config['decay_check_interval']:
                return

            logger.info("Running memory decay and purge (scheduled or on-demand).")

            memories_with_scores = []
            for mid, mem in self.memories.items():
                eff_score = self._calculate_effective_quickrecal_score(mem)
                memories_with_scores.append((mid, eff_score))

            # Sort ascending by effective QuickRecal
            memories_with_scores.sort(key=lambda x: x[1])

            # Identify how many to purge to respect max_memories
            excess_count = len(self.memories) - self.config['max_memories']
            purge_count = max(excess_count, 0)

            # Also gather below purge_threshold
            purge_ids = [m_id for (m_id, eff) in memories_with_scores
                         if eff < self.config['purge_threshold']]

            # If the below-threshold list is bigger than needed, cut it
            if len(purge_ids) > purge_count:
                purge_ids = purge_ids[:purge_count]

            # Purge identified memories
            for m_id in purge_ids:
                await self._purge_memory(m_id)

            self.stats['purges'] += len(purge_ids)
            self.stats['last_decay_check'] = current_time
            logger.info(f"Purged {len(purge_ids)} memories during decay step.")

    async def _purge_memory(self, memory_id: str) -> None:
        """
        Remove a memory from LTM and queue a purge operation for disk.
        """
        if memory_id not in self.memories:
            return

        mem = self.memories[memory_id]
        qr_score = mem.get('quickrecal_score', 0)
        age_days = (time.time() - mem.get('timestamp', 0)) / 86400

        logger.debug(f"Purging memory {memory_id} (QR={qr_score:.2f}, age={age_days:.2f} days)")

        del self.memories[memory_id]
        cat = mem.get('metadata', {}).get('category', 'general')
        if cat in self.memory_index and memory_id in self.memory_index[cat]:
            self.memory_index[cat].remove(memory_id)

        # Queue purge for disk
        if self.config['enable_persistence']:
            await self._add_to_batch_queue(OperationType.PURGE, memory_id)

    async def _add_to_batch_queue(
        self,
        op_type: OperationType,
        memory_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Safely add an operation to the batch queue.
        """
        if not self.config['enable_persistence'] or self._shutdown:
            return

        async with self._batch_lock:
            operation = BatchOperation(op_type, memory_id, data)
            self._batch_queue.append(operation)

            # If we hit batch_size, trigger immediate processing
            if len(self._batch_queue) >= self.config['batch_size']:
                self._batch_event.set()

    async def _batch_processor(self) -> None:
        """
        Background task that processes queued operations at intervals or
        when the queue reaches batch_size.
        """
        logger.info("Starting batch processor task.")
        last_process_time = time.time()

        while not self._shutdown:
            try:
                # Wait for either an event or a timeout
                try:
                    await asyncio.wait_for(
                        self._batch_event.wait(), timeout=self.config['batch_interval']
                    )
                except asyncio.TimeoutError:
                    # Interval expired, continue to processing check
                    pass
                finally:
                    self._batch_event.clear()

                current_time = time.time()
                time_since_last = current_time - last_process_time

                # Process the queue if:
                #  - queue is non-empty AND
                #    EITHER queue >= batch_size OR interval has passed
                if (
                    len(self._batch_queue) > 0
                    and (
                        len(self._batch_queue) >= self.config['batch_size']
                        or time_since_last >= self.config['batch_interval']
                    )
                ):
                    await self._process_batch()
                    last_process_time = time.time()

            except asyncio.CancelledError:
                logger.info("Batch processor task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight error loop

    async def _process_batch(self, force: bool = False) -> None:
        """
        Pull a batch of operations from the queue and persist them.

        Args:
            force: If True, process all queued ops regardless of batch size.
        """
        if not self.config['enable_persistence']:
            return

        # If no ops or if already processing a batch (and not forced), skip
        if (not self._batch_queue) or (self._batch_processing and not force):
            return

        async with self._batch_lock:
            self._batch_processing = True

            try:
                # Decide how many ops to process
                if force:
                    batch_size = len(self._batch_queue)
                else:
                    batch_size = min(len(self._batch_queue), self.config['batch_size'])

                # Update batch stats
                self.stats['batch_operations'] += batch_size
                self.stats['total_batches'] += 1
                self.stats['avg_batch_size'] = (
                    self.stats['batch_operations'] / self.stats['total_batches']
                )
                self.stats['largest_batch'] = max(self.stats['largest_batch'], batch_size)

                logger.debug(f"Processing batch of {batch_size} operations (force={force}).")

                # Collect operations
                operations = []
                for _ in range(batch_size):
                    if not self._batch_queue:
                        break
                    operations.append(self._batch_queue.popleft())

                # Separate them by type
                store_ops = [op for op in operations if op.op_type == OperationType.STORE]
                update_ops = [op for op in operations if op.op_type == OperationType.UPDATE]
                purge_ops = [op for op in operations if op.op_type == OperationType.PURGE]

                # Process each group
                store_results = await self._process_store_batch(store_ops)
                update_results = await self._process_update_batch(update_ops)
                purge_results = await self._process_purge_batch(purge_ops)

                success_count = store_results + update_results + purge_results

                # Update success/failure stats
                self.stats['batch_successes'] += success_count
                self.stats['batch_failures'] += (batch_size - success_count)

                logger.debug(
                    f"Batch done. Successes: {success_count}/{batch_size}. "
                    f"Failures: {batch_size - success_count}."
                )
            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=True)
            finally:
                self._batch_processing = False

    async def _process_store_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process STORE operations with retry logic.
        """
        if not operations:
            return 0

        success_count = 0
        # Prepare memory data grouped by memory_id
        to_store = {}
        for op in operations:
            if op.data:
                to_store[op.memory_id] = op.data

        for memory_id, memory in to_store.items():
            stored_successfully = False
            for attempt in range(self.config['batch_retries']):
                try:
                    # Convert any tensor to list
                    memory_copy = memory.copy()
                    if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                        memory_copy['embedding'] = memory_copy['embedding'].tolist()

                    file_path = self.storage_path / f"{memory_id}.json"
                    with open(file_path, 'w') as f:
                        json.dump(memory_copy, f, indent=2)

                    stored_successfully = True
                    break  # break out of retry loop
                except Exception as e:
                    logger.error(
                        f"Error storing memory {memory_id} (attempt {attempt+1}): {e}",
                        exc_info=True
                    )
                    if attempt < self.config['batch_retries'] - 1:
                        await asyncio.sleep(self.config['batch_retry_delay'])

            if stored_successfully:
                success_count += 1

        return success_count

    async def _process_update_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process UPDATE operations with retry logic.

        (Currently the same as storing, as we overwrite the entire JSON.)
        """
        if not operations:
            return 0

        return await self._process_store_batch(operations)

    async def _process_purge_batch(self, operations: List[BatchOperation]) -> int:
        """
        Process PURGE operations with retry logic.
        """
        if not operations:
            return 0

        success_count = 0
        # Avoid duplicates
        purge_ids = set(op.memory_id for op in operations)

        for memory_id in purge_ids:
            purged_successfully = False
            for attempt in range(self.config['batch_retries']):
                try:
                    file_path = self.storage_path / f"{memory_id}.json"
                    if file_path.exists():
                        os.remove(file_path)
                    purged_successfully = True
                    break
                except Exception as e:
                    logger.error(
                        f"Error purging memory {memory_id} (attempt {attempt+1}): {e}",
                        exc_info=True
                    )
                    if attempt < self.config['batch_retries'] - 1:
                        await asyncio.sleep(self.config['batch_retry_delay'])

            if purged_successfully:
                success_count += 1

        return success_count

    async def backup(self) -> bool:
        """
        Force any pending batch operations, then copy all memory files to a backup folder.

        Returns:
            True if backup completed successfully, False otherwise.
        """
        # If no persistence, can't back up
        if not self.config['enable_persistence']:
            return False

        # Process any pending operations
        if self._batch_queue:
            await self._process_batch(force=True)

        async with self._lock:
            try:
                backup_dir = self.storage_path / 'backups'
                backup_dir.mkdir(exist_ok=True)

                timestamp = time.strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"backup_{timestamp}"
                backup_path.mkdir(exist_ok=True)

                # Copy each known memory file
                import shutil
                for mid in self.memories:
                    src = self.storage_path / f"{mid}.json"
                    dst = backup_path / f"{mid}.json"
                    if src.exists():
                        shutil.copy2(src, dst)

                self.stats['last_backup'] = time.time()
                logger.info(f"Created backup at {backup_path}")
                return True
            except Exception as e:
                logger.error(f"Error creating backup: {e}", exc_info=True)
                return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Return overall statistics about the LTM system.
        """
        cat_counts = {cat: len(ids) for cat, ids in self.memory_index.items()}
        qr_scores = [m.get('quickrecal_score', 0) for m in self.memories.values()]
        bins = [0, 0, 0, 0, 0]  # For 5 intervals: [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        for score in qr_scores:
            idx = min(int(score * 5), 4)
            bins[idx] += 1

        qr_distribution = {
            '0.0-0.2': bins[0],
            '0.2-0.4': bins[1],
            '0.4-0.6': bins[2],
            '0.6-0.8': bins[3],
            '0.8-1.0': bins[4]
        }

        queued = len(self._batch_queue) if hasattr(self, '_batch_queue') else 0
        batch_stats = {
            'batch_operations': self.stats['batch_operations'],
            'batch_successes': self.stats['batch_successes'],
            'batch_failures': self.stats['batch_failures'],
            'avg_batch_size': self.stats['avg_batch_size'],
            'total_batches': self.stats['total_batches'],
            'largest_batch': self.stats['largest_batch'],
            'queued_operations': queued,
            'batch_success_rate': (
                self.stats['batch_successes'] / max(1, self.stats['batch_operations'])
            ) * 100
        }

        return {
            'total_memories': len(self.memories),
            'categories': cat_counts,
            'quickrecal_distribution': qr_distribution,
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

    async def update_access_timestamp(self, memory_id: str) -> bool:
        """
        Update the access timestamp for a memory (manual usage).
        """
        async with self._lock:
            if memory_id not in self.memories:
                logger.debug(f"Memory {memory_id} not found in LTM while updating access timestamp")
                return False

            try:
                current_time = time.time()
                mem = self.memories[memory_id]
                mem['last_access'] = current_time
                mem['access_count'] = mem.get('access_count', 0) + 1

                if 'metadata' not in mem:
                    mem['metadata'] = {}
                mem['metadata']['last_access'] = current_time
                mem['metadata']['access_count'] = mem['access_count']

                if self.config['enable_persistence']:
                    await self._add_to_batch_queue(OperationType.UPDATE, memory_id, mem)

                logger.debug(f"Updated access timestamp for memory {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Error updating access timestamp: {e}", exc_info=True)
                return False

    async def _decay_scheduler(self):
        """
        Periodically trigger _run_decay_and_purge according to decay_check_interval.
        Runs in the background until shutdown is requested.
        """
        while not self._shutdown:
            try:
                # Sleep for a fraction of the check interval or a short interval,
                # then let _run_decay_and_purge() decide if it should skip or run.
                await asyncio.sleep(60)  # check every 60s in this example

                # We rely on _run_decay_and_purge to skip if not enough time has passed
                await self._run_decay_and_purge()

            except asyncio.CancelledError:
                logger.info("Decay scheduler task cancelled.")
                break
            except Exception as e:
                logger.error(f"Decay scheduler encountered an error: {e}", exc_info=True)
                await asyncio.sleep(1)  # prevent rapid looping on error
