# Memory Persistence

This document describes the persistence layer for the Synthians Memory Core, which is responsible for saving and loading memory entries and assemblies.

## Overview

The `MemoryPersistence` class handles the storage and retrieval of `MemoryEntry` and `MemoryAssembly` objects to and from disk. It uses a structured filesystem approach with asynchronous I/O operations for efficiency.

## Storage Structure

The persistence layer uses a structured directory layout:

```
<storage_path>/
├── memory_index.json           # Index of all memories and assemblies
├── memories/                   # Directory for memory entries
│   ├── mem_<uuid_1>.json       # Individual memory entry files
│   ├── mem_<uuid_2>.json
│   └── ...
├── assemblies/                 # Directory for memory assemblies
│   ├── asm_<uuid_a>.json       # Individual assembly files
│   ├── asm_<uuid_b>.json
│   └── ...
├── vector_index/               # Directory for FAISS index files
│   ├── faiss_index.bin         # Serialized FAISS index
│   └── faiss_index.bin.mapping.json  # ID mapping information
├── stats/                      # Directory for statistics (Phase 5.9)
│   └── assembly_activation_stats.json # Assembly activation counters
└── logs/                       # Directory for persistent logs (Phase 5.9)
    └── merge_log.jsonl         # Append-only log of merge events
```

## Key Components

### Memory Index

The `memory_index.json` file serves as a central registry for all persisted objects. It has this structure:

```json
{
  "memories": {
    "mem_uuid1": "memories/mem_uuid1.json",
    "mem_uuid2": "memories/mem_uuid2.json",
    ...
  },
  "assemblies": {
    "asm_uuid1": "assemblies/asm_uuid1.json",
    "asm_uuid2": "assemblies/asm_uuid2.json",
    ...
  }
}
```

This index helps the system quickly locate files without scanning directories.

### Memory Entry Files

Each `MemoryEntry` is stored as an individual JSON file with the structure derived from the object's `to_dict()` method:

```json
{
  "id": "mem_uuid1",
  "embedding": [0.1, 0.2, ...],  # Vector representation
  "content": "Memory text content",
  "creation_time": "2025-04-01T10:30:00",
  "quickrecal_score": 0.75,
  "emotional_content": {
    "joy": 0.8,
    "sadness": 0.1,
    ...
  },
  "metadata": {
    "source": "user_input",
    "context": "conversation_123",
    ...
  }
}
```

### Memory Assembly Files

Each `MemoryAssembly` is stored as a JSON file with this structure:

```json
{
  "id": "asm_uuid1",
  "composite_embedding": [0.1, 0.2, ...],  # Combined vector
  "memory_ids": ["mem_uuid1", "mem_uuid2", ...],
  "creation_time": "2025-04-01T14:20:00",
  "name": "Conversation about ML",
  "vector_index_updated_at": "2025-04-01T14:25:00",  # Phase 5.8: Timestamp of last successful index update
  "merged_from": ["asm_uuid2", "asm_uuid3"],  # Phase 5.9: Assembly lineage tracking
  "metadata": {
    "context": "user_session_456",
    ...
  }
}
```

### Phase 5.9 Persistence Files

#### Merge Log (append-only)

The `merge_log.jsonl` is an append-only log file where each line is a separate JSON entry. There are two main event types:

1. **Merge Events**:
```json
{
  "event_type": "merge",
  "merge_event_id": "merge_uuid_123",
  "timestamp": "2025-04-01T15:32:45.123Z",
  "source_assembly_ids": ["asm_abc", "asm_def"],
  "target_assembly_id": "asm_merged_123",
  "similarity_at_merge": 0.92,
  "merge_threshold": 0.85,
  "cleanup_status": "pending"
}
```

2. **Cleanup Update Events**:
```json
{
  "event_type": "cleanup_update",
  "target_merge_event_id": "merge_uuid_123",
  "timestamp": "2025-04-01T15:33:10.456Z",
  "cleanup_status": "completed",
  "error": null
}
```

The `MergeTracker` class manages this log, including rotation and reconciliation.

#### Assembly Activation Stats

The `stats/assembly_activation_stats.json` file contains counters for assembly activations:

```json
{
  "activation_counts": {
    "asm_uuid1": 42,
    "asm_uuid2": 17,
    ...
  },
  "last_updated": "2025-04-01T16:45:30"
}
```

This file is periodically updated by the `_persist_activation_stats` method in `SynthiansMemoryCore`.

## Core Operations

### Saving Operations

Saving operations use asynchronous I/O with atomic file writes to prevent data corruption:

```python
async def save_memory(self, memory: MemoryEntry) -> None:
    """Save a memory entry to disk with atomic file writing."""
    # Update index
    memory_path = f"memories/mem_{memory.id}.json"
    self.memory_index["memories"][memory.id] = memory_path
    
    # Serialize memory
    memory_dict = memory.to_dict()
    
    # Ensure directory exists
    os.makedirs(os.path.join(self.storage_path, "memories"), exist_ok=True)
    
    # Write file atomically
    await self.safe_write_json(
        os.path.join(self.storage_path, memory_path),
        memory_dict
    )
    
    # Update index file
    await self.save_memory_index()
```

The `safe_write_json` method writes to a temporary file and then renames it, ensuring atomic updates.

### Loading Operations

Loading operations fetch objects from disk asynchronously:

```python
async def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
    """Load a memory entry from disk."""
    if memory_id not in self.memory_index["memories"]:
        return None
        
    memory_path = os.path.join(
        self.storage_path, 
        self.memory_index["memories"][memory_id]
    )
    
    async with aiofiles.open(memory_path, "r") as f:
        content = await f.read()
        
    memory_dict = json.loads(content)
    return MemoryEntry.from_dict(memory_dict)
```

## Thread Safety

The persistence layer implements proper locking to ensure thread safety during concurrent operations:

```python
def __init__(self, storage_path: str):
    self.storage_path = storage_path
    self.memory_index = {"memories": {}, "assemblies": {}}
    self._lock = asyncio.Lock()  # Async lock for thread safety
    # ...

async def save_memory(self, memory: MemoryEntry) -> None:
    async with self._lock:  # Ensure thread-safe operations
        # Saving logic here
        # ...
```

## Error Handling

The persistence layer implements robust error handling for I/O operations:

```python
async def safe_write_json(self, filepath: str, data: Dict) -> None:
    """Write JSON data to a file atomically using a temporary file."""
    # Create temp file in the same directory
    temp_path = f"{filepath}.tmp"
    
    try:
        # Write to temp file
        async with aiofiles.open(temp_path, "w") as f:
            await f.write(json.dumps(data, indent=2, default=self._default_serializer))
            await f.flush()  # Ensure data is written to disk
        
        # Rename temp file to target (atomic operation)
        os.replace(temp_path, filepath)
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e  # Re-raise the exception
```

## Serialization Helpers

The persistence implementation provides helper methods for handling special data types:

```python
def _default_serializer(self, obj):
    """Handle special types during JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```
