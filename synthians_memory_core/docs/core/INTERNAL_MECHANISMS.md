# Synthians Memory Core: Internal Mechanisms

This document details the key internal mechanisms of the `SynthiansMemoryCore` class that are responsible for maintaining system stability, consistency, and performance.

## Concurrency Control and Background Tasks

The Memory Core uses several background tasks and queues to ensure thread safety, data consistency, and resilience to failures. These mechanisms were significantly enhanced in Phase 5.8 to improve stability.

### Core Locking Mechanism

```python
# Main lock for operations that modify core data structures
self._lock = asyncio.Lock()
```

The `_lock` is an asyncio lock that provides thread safety for operations that modify core data structures. It's used in methods like `process_memory`, `update_memory`, and `_execute_merge` to ensure that only one operation modifies the memory core at a time.

This locking mechanism is critical for preventing race conditions in the following scenarios:
1. Multiple concurrent memory creations or updates
2. Assembly merging while retrievals are in progress
3. Vector index updates during retrieval operations

### Persistence Loop

```python
# Set to track "dirty" memories that need to be saved
self._dirty_memories = set()
self._dirty_assemblies = set()

# Background task for persisting memory entries
self._persistence_task = asyncio.create_task(self._persistence_loop())
```

The persistence loop operates as follows:
1. It runs periodically (every `persistence_interval_seconds`, typically 5-10 seconds)
2. Each iteration, it acquires the `_lock` to safely access the dirty sets
3. It copies the current dirty sets and clears the originals while holding the lock
4. After releasing the lock, it asynchronously saves each memory and assembly via `persistence.save_memory()` and `persistence.save_assembly()`
5. Any errors during saving are logged, but don't stop the process

This approach allows memory operations to continue without waiting for I/O, providing both performance and reliability:
```python
async def _persistence_loop(self):
    """Background task to periodically persist dirty memories and assemblies."""
    while True:
        try:
            await asyncio.sleep(self.config.PERSISTENCE_INTERVAL_SECONDS)
            
            # Get dirty items (under lock to avoid race conditions)
            async with self._lock:
                memories_to_save = list(self._dirty_memories)
                assemblies_to_save = list(self._dirty_assemblies)
                self._dirty_memories.clear()
                self._dirty_assemblies.clear()
            
            # Save items (without holding the lock)
            for memory_id in memories_to_save:
                if memory_id in self._memories:
                    await self.persistence.save_memory(self._memories[memory_id])
            
            for assembly_id in assemblies_to_save:
                if assembly_id in self._assemblies:
                    await self.persistence.save_assembly(self._assemblies[assembly_id])
                    
        except Exception as e:
            logger.error(f"Error in persistence loop: {e}")
```

### Vector Update Retry Loop (Phase 5.8)

```python
# Queue for tracking pending vector index updates
self._pending_vector_updates = asyncio.Queue()

# Background task for retrying failed vector updates
self._vector_update_task = asyncio.create_task(self._vector_update_retry_loop())
```

The vector update retry loop is a critical mechanism introduced in Phase 5.8 that replaces the previous `AssemblySyncManager`. This mechanism is now implemented directly within the `SynthiansMemoryCore` class and handles failures in FAISS vector index operations:

1. The queue `_pending_vector_updates` stores operations that failed (both memory and assembly vector updates)
2. Failed operations are queued as dictionaries containing:
   - `"operation"`: Either "add" or "remove"
   - `"id"`: The string ID of the memory or assembly
   - `"embedding"`: The vector embedding to add
   - `"is_assembly"`: Boolean flag indicating if this is an assembly (for updating `vector_index_updated_at`)

3. The retry loop runs at a configurable interval (typically every 30-60 seconds):
```python
async def _vector_update_retry_loop(self):
    """Background task to retry failed vector index updates."""
    while True:
        try:
            await asyncio.sleep(self.config.VECTOR_UPDATE_RETRY_INTERVAL_SECONDS)
            
            # Process all current pending updates
            pending_count = self._pending_vector_updates.qsize()
            if pending_count > 0:
                logger.info(f"Processing {pending_count} pending vector updates")
                
                for _ in range(pending_count):
                    try:
                        # Get the next pending update
                        update = await self._pending_vector_updates.get()
                        
                        # Extract update details
                        operation = update.get("operation")
                        id_str = update.get("id")
                        embedding = update.get("embedding")
                        is_assembly = update.get("is_assembly", False)
                        
                        # Process the update
                        if operation == "add" and embedding:
                            await self.vector_index.add_with_ids([id_str], [embedding])
                            
                            # Update timestamp for assemblies
                            if is_assembly and id_str in self._assemblies:
                                self._assemblies[id_str].vector_index_updated_at = datetime.utcnow().isoformat()
                                self._dirty_assemblies.add(id_str)
                                
                        elif operation == "remove":
                            await self.vector_index.remove_ids([id_str])
                            
                        # Mark task as done
                        self._pending_vector_updates.task_done()
                        
                    except Exception as e:
                        # Re-queue the update for the next retry cycle
                        await self._pending_vector_updates.put(update)
                        logger.error(f"Error processing vector update: {e}")
                        
        except Exception as e:
            logger.error(f"Error in vector update retry loop: {e}")
```

4. Critically, the `vector_index_updated_at` timestamp on assemblies is ONLY updated after a successful vector index update
5. This timestamp gates whether an assembly can be used for boosting during retrieval

#### When Updates Are Queued

Vector updates are added to the pending queue in various places throughout the code:

1. In `process_memory` or `update_memory` when adding/updating memory embeddings:
```python
try:
    await self.vector_index.add_with_ids([memory.id], [memory.embedding])
except Exception as e:
    logger.warning(f"Failed to add memory to vector index, queuing for retry: {e}")
    await self._pending_vector_updates.put({
        "operation": "add",
        "id": memory.id,
        "embedding": memory.embedding,
        "is_assembly": False
    })
```

2. In `create_assembly` or `update_assembly` when adding/updating assembly embeddings:
```python
try:
    await self.vector_index.add_with_ids([assembly.id], [assembly.composite_embedding])
    assembly.vector_index_updated_at = datetime.utcnow().isoformat()
except Exception as e:
    logger.warning(f"Failed to add assembly to vector index, queuing for retry: {e}")
    await self._pending_vector_updates.put({
        "operation": "add",
        "id": assembly.id,
        "embedding": assembly.composite_embedding,
        "is_assembly": True
    })
```

3. In `delete_memory` or `delete_assembly` when removing vector entries:
```python
try:
    await self.vector_index.remove_ids([memory_id])
except Exception as e:
    logger.warning(f"Failed to remove memory from vector index, queuing for retry: {e}")
    await self._pending_vector_updates.put({
        "operation": "remove",
        "id": memory_id
    })
```

This mechanism ensures that:
- Failed vector operations don't block the main processing flow
- Updates are eventually applied, maintaining index consistency
- Only assemblies with successful index updates (indicated by `vector_index_updated_at`) are used for boosting

The vector update retry loop is a critical mechanism introduced in Phase 5.8 to handle failures in FAISS vector index operations:

1. When a memory or assembly is created or updated, an update to the vector index is required
2. If this update fails (e.g., due to GPU memory issues or FAISS errors), it's added to the `_pending_vector_updates` queue
3. The `_vector_update_retry_loop` periodically attempts to process items from this queue
4. Each item contains the ID, embedding, and operation type (add/remove)
5. The timestamp `vector_index_updated_at` is only updated after a successful vector index update

This mechanism ensures that:
- Failed vector operations don't block the main processing flow
- Updates are eventually applied, maintaining index consistency
- Only assemblies with successful index updates (indicated by `vector_index_updated_at`) are used for boosting

### Decay and Pruning Loop

```python
# Background task for QuickRecal decay and assembly maintenance
self._decay_task = asyncio.create_task(self._decay_and_pruning_loop())
```

The decay and pruning loop handles several periodic maintenance tasks in a single background process. This unified approach was chosen to minimize the number of background tasks and to ensure these operations don't interfere with each other:

```python
async def _decay_and_pruning_loop(self):
    """Background task for periodic decay of QuickRecal and assembly maintenance."""
    last_decay_time = time.time()
    last_pruning_time = time.time()
    last_merge_check_time = time.time()
    
    while True:
        try:
            # Sleep for a short interval to avoid tight loop
            await asyncio.sleep(self.config.BACKGROUND_TASK_INTERVAL_SECONDS)
            current_time = time.time()
            
            # 1. QuickRecal decay (every DECAY_INTERVAL_SECONDS)
            if current_time - last_decay_time >= self.config.DECAY_INTERVAL_SECONDS:
                await self._perform_quickrecal_decay()
                last_decay_time = current_time
            
            # 2. Assembly pruning (if enabled, every ASSEMBLY_PRUNING_INTERVAL_SECONDS)
            if (self.config.ENABLE_ASSEMBLY_PRUNING and
                current_time - last_pruning_time >= self.config.ASSEMBLY_PRUNING_INTERVAL_SECONDS):
                await self._prune_stale_assemblies()
                last_pruning_time = current_time
            
            # 3. Assembly merge checks (if enabled, every ASSEMBLY_MERGE_CHECK_INTERVAL_SECONDS)
            if (self.config.ENABLE_ASSEMBLY_MERGING and
                current_time - last_merge_check_time >= self.config.ASSEMBLY_MERGE_CHECK_INTERVAL_SECONDS):
                await self._check_and_execute_merges()
                last_merge_check_time = current_time
                
        except Exception as e:
            logger.error(f"Error in decay and pruning loop: {e}")
```

The loop performs these key tasks:

#### 1. QuickRecal Decay

- Gradually reduces the QuickRecal scores of memories over time to model forgetting
- Runs every `DECAY_INTERVAL_SECONDS` (typically 1-4 hours)
- Applies a configurable decay rate to all memories
- Updates the QuickRecal scores in memory and marks memories as dirty for persistence

#### 2. Assembly Pruning

- Removes assemblies that haven't been activated recently
- Only runs if `ENABLE_ASSEMBLY_PRUNING` is True
- Uses configurable age threshold (`ASSEMBLY_MAX_AGE_SECONDS`) to determine which assemblies to prune
- Removes assemblies from memory, vector index, and storage

#### 3. Assembly Merging

- Checks for similar assemblies that should be merged
- Only runs if `ENABLE_ASSEMBLY_MERGING` is True
- Uses configurable similarity threshold (`ASSEMBLY_MERGE_THRESHOLD`) to identify merge candidates
- Creates new assemblies with combined properties and sets `merged_from` field
- Runs asynchronous cleanup to remove source assemblies after merge

All three maintenance tasks operate on different intervals and can be individually enabled or disabled through configuration. This provides flexibility while ensuring the system's memory structures remain optimized and up-to-date.

## Event Flow and Recovery

Understanding the flow of operations and recovery mechanisms is crucial for maintaining system stability:

### Memory Processing Flow

1. `process_memory` receives content and metadata
2. Creates a `MemoryEntry` with embedding
3. Adds to in-memory store and marks as dirty
4. Attempts to add to vector index
5. If vector index update fails, adds to `_pending_vector_updates` queue
6. Returns the memory ID regardless of vector update status

### Assembly Merging Flow

1. `_check_and_execute_merges` identifies candidate assemblies for merging
2. `_execute_merge` creates a new assembly with combined memories
3. The new assembly is added to the in-memory store and marked dirty
4. Vector index update is attempted
5. If update fails, the assembly is added to `_pending_vector_updates` queue
6. The `vector_index_updated_at` timestamp is only set after successful update

### Recovery Mechanisms

The system includes several recovery mechanisms:

1. **Vector Index Integrity Check**: On startup, `_check_index_integrity` verifies consistency between the vector index and memory/assembly stores
2. **Automatic Repair**: `_repair_index_async` can rebuild the vector index from stored memories and assemblies
3. **Manual Repair**: The `/repair_index` API endpoint allows forcing a complete index rebuild

## Phase 5.9 Planned Enhancements (Not Yet Implemented)

In Phase 5.9, these mechanisms will be enhanced with:

1. **Merge Tracking**: The `_execute_merge` method will log merge events via the `MergeTracker`
2. **Activation Statistics**: Assembly activations will be counted and analyzed
3. **Enhanced Diagnostics**: New API endpoints will expose internal statistics and logs

These enhancements will provide greater visibility into the system's internal operations without changing the core stability mechanisms.