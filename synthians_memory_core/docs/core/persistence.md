# Memory Persistence

The `synthians_memory_core.memory_persistence.MemoryPersistence` class handles the saving and loading of memory structures (primarily `MemoryEntry` and `MemoryAssembly` objects) to and from the filesystem.

## Purpose

Persistence ensures that the state of the memory core (memories, assemblies, metadata) survives restarts and shutdowns.

## Key Component: `MemoryPersistence`

*   **Functionality:**
    *   Provides asynchronous methods (`save_memory`, `load_memory`, `delete_memory`, `save_assembly`, `load_assembly`, etc.) to interact with the filesystem.
    *   Typically saves individual `MemoryEntry` objects as separate JSON files within a structured directory (`storage_path/memories/`).
    *   Saves `MemoryAssembly` objects similarly (`storage_path/assemblies/`).
    *   Manages a central index file (`storage_path/memory_index.json`) which maps memory IDs to their file paths and potentially stores lightweight metadata for faster loading or indexing.
    *   Uses `aiofiles` for non-blocking file I/O, crucial for an asynchronous system.
*   **Integration:**
    *   Used by `SynthiansMemoryCore` to save new/updated memories and assemblies.
    *   Used during `SynthiansMemoryCore` initialization to load existing memories and assemblies from disk.
    *   Coordinates with `MemoryVectorIndex` to ensure consistency between saved memories and their vector representations.

## Storage Structure (Example)

```
<storage_path>/
├── memory_index.json        # Maps memory_id -> filepath, metadata
├── memories/
│   ├── <memory_uuid_1>.json # Complete MemoryEntry object
│   ├── <memory_uuid_2>.json
│   └── ...
├── assemblies/
│   ├── <assembly_id_1>.json # Complete MemoryAssembly object
│   ├── <assembly_id_2>.json
│   └── ...
└── vector_index/           # Managed by MemoryVectorIndex
    ├── memory_vectors.faiss # FAISS binary index file
    └── mapping.json        # Backup of string_id -> faiss_id mapping
```

## Memory Index Structure

The `memory_index.json` file maintains a master record of all memories and their metadata:

```json
{
  "memories": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "filepath": "memories/550e8400-e29b-41d4-a716-446655440000.json",
      "created_at": "2025-03-15T14:32:01.123456",
      "updated_at": "2025-03-15T14:45:22.654321",
      "quickrecal_score": 0.85,
      "content_hash": "sha256:a1b2c3..."
    },
    "550e8400-e29b-41d4-a716-446655440001": {
      "filepath": "memories/550e8400-e29b-41d4-a716-446655440001.json",
      "created_at": "2025-03-16T08:12:35.789012",
      "updated_at": "2025-03-16T08:12:35.789012",
      "quickrecal_score": 0.72,
      "content_hash": "sha256:d4e5f6..."
    },
    // Additional memories...
  },
  "assemblies": {
    "assembly_001": {
      "filepath": "assemblies/assembly_001.json",
      "created_at": "2025-03-17T10:24:56.135790",
      "updated_at": "2025-03-18T15:30:42.864209",
      "member_count": 5
    },
    // Additional assemblies...
  },
  "metadata": {
    "version": "2.3.0",
    "last_updated": "2025-03-18T15:30:42.864209",
    "memory_count": 237,
    "assembly_count": 42
  }
}
```

## Implementation Details

### 1. Asynchronous Operations

All file operations are implemented asynchronously using `aiofiles` to prevent blocking the main API service:

```python
async def save_memory(self, memory: MemoryEntry) -> None:
    """Save a memory to the filesystem asynchronously."""
    file_path = os.path.join(self.memories_path, f"{memory.memory_id}.json")
    memory_dict = memory.dict()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Asynchronously write the memory to a file
    async with aiofiles.open(file_path, mode='w') as f:
        await f.write(json.dumps(memory_dict, indent=2))
    
    # Update the memory index
    await self._update_memory_index(memory)
```

### 2. Batch Operations

The persistence layer supports batch operations for improved performance:

```python
async def save_memories_batch(self, memories: List[MemoryEntry]) -> None:
    """Save multiple memories efficiently."""
    # Group operations to reduce disk I/O
    tasks = [self._save_memory_file(memory) for memory in memories]
    await asyncio.gather(*tasks)
    
    # Update the index in one operation
    await self._update_memory_index_batch(memories)
```

### 3. Error Handling and Recovery

The system implements robust error handling to prevent data loss:

* **Transaction-like Approach**: For critical operations, files are first written to temporary locations, then atomically moved to their final destinations
* **Backup Creation**: Periodic backups of the memory index are maintained
* **Consistency Checks**: When loading memories, the system verifies consistency between the memory index and actual files
* **Auto-Recovery**: Can rebuild the memory index from individual memory files if the index becomes corrupted

```python
async def verify_and_repair_consistency(self) -> Dict[str, Any]:
    """Verify consistency between memory index and files, repairing if needed."""
    # Implementation scans files, verifies against index, and repairs inconsistencies
    found_files = await self._scan_memory_files()
    index_entries = await self._load_memory_index()
    
    missing_from_index = [f for f in found_files if f not in index_entries]
    missing_files = [e for e in index_entries if e not in found_files]
    
    # Repair actions
    repair_results = await self._repair_inconsistencies(missing_from_index, missing_files)
    
    return repair_results
```

## Integration with Vector Index

The persistence layer works in coordination with the `MemoryVectorIndex` to ensure consistency:

1. **Memory Creation Flow**:
   * Memory is saved to filesystem via `save_memory`
   * Memory embedding is added to vector index via `add_vector`
   * Memory index is updated with metadata

2. **Memory Deletion Flow**:
   * Memory is marked for deletion in the index
   * Memory is removed from vector index via `remove_vector`
   * Memory file is deleted from filesystem

3. **Startup Consistency**:
   * During initialization, the system verifies that memories in the filesystem have corresponding vectors in the FAISS index
   * Mismatches are resolved either by rebuilding missing vector entries or removing orphaned vectors

## Configuration

*   `storage_path`: The root directory for all persistent memory data (default: `./storage`)
*   `index_backup_count`: Number of backup copies to maintain for the memory index (default: `3`)
*   `auto_repair`: Whether to automatically repair inconsistencies during startup (default: `True`)
*   `backup_interval`: Interval in seconds between automatic backups (default: `3600` - 1 hour)
*   `flush_threshold`: Number of memory changes before forcing a flush to disk (default: `20`)

## Performance Considerations

* **Lazy Loading**: By default, the system loads only the memory index at startup, with individual memories loaded on-demand
* **LRU Cache**: Frequently accessed memories are cached in memory for faster access
* **Chunked Processing**: For large memory stores, batch operations are chunked to manage memory usage
* **Optimistic Locking**: Minimal file locking to maximize concurrency, with conflicts resolved through update timestamps

## Failure Handling

* **Disk Full**: If the disk is full, the system attempts to complete critical operations and logs severe warnings
* **Corrupted Files**: JSON parsing errors are handled gracefully, with attempts to recover partial data
* **Permission Issues**: Clear error messages indicate permission problems with helpful resolution steps
* **Storage Migration**: Built-in utilities for safely migrating memory storage to a new location

## Importance

Reliable persistence is fundamental. Without it, the memory core would be volatile, losing all information upon restart. The asynchronous nature ensures that saving/loading operations don't block the main application thread, while the robust error handling and recovery mechanisms protect against data loss.
