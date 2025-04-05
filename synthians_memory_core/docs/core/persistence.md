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
│   └── assembly_activation_stats.json # Planned for Phase 5.9
└── logs/                       # Directory for persistent logs (Phase 5.9)
    └── merge_log.jsonl         # Planned for Phase 5.9
```

This structure ensures that:
1. Memory entries and assemblies are stored in separate directories for organization.
2. Each object has its own file, minimizing contention during parallel operations.
3. The index file provides a quick way to list all available items without scanning directories.
4. The vector index is stored separately from the memory objects.
5. Statistics and logs have dedicated locations.

## Key Operations

### Initialization

```python
async def initialize(self) -> None:
    """Initialize the persistence layer."""
    # Create necessary directories
    os.makedirs(os.path.join(self.storage_path, "memories"), exist_ok=True)
    os.makedirs(os.path.join(self.storage_path, "assemblies"), exist_ok=True)
    os.makedirs(os.path.join(self.storage_path, "vector_index"), exist_ok=True)
    
    # Load memory index if it exists
    try:
        await self._load_memory_index()
    except FileNotFoundError:
        self.memory_index = {"memories": {}, "assemblies": {}}
```

### Saving Memory Entries

```python
async def save_memory(self, memory: MemoryEntry) -> None:
    """Save a memory entry to disk."""
    # Add to index
    self.memory_index["memories"][memory.id] = {
        "id": memory.id,
        "timestamp": memory.timestamp,
        "file_path": f"memories/mem_{memory.id}.json"
    }
    
    # Save memory to file
    memory_path = os.path.join(self.storage_path, "memories", f"mem_{memory.id}.json")
    async with aiofiles.open(memory_path, "w") as f:
        await f.write(json.dumps(memory.__dict__, cls=NumpyEncoder))
    
    # Save updated index
    await self._save_memory_index()
```

### Saving Assemblies

```python
async def save_assembly(self, assembly: MemoryAssembly) -> None:
    """Save a memory assembly to disk."""
    # Add to index
    self.memory_index["assemblies"][assembly.id] = {
        "id": assembly.id,
        "updated_at": assembly.updated_at,
        "file_path": f"assemblies/asm_{assembly.id}.json"
    }
    
    # Save assembly to file
    assembly_path = os.path.join(self.storage_path, "assemblies", f"asm_{assembly.id}.json")
    async with aiofiles.open(assembly_path, "w") as f:
        await f.write(json.dumps(assembly.__dict__, cls=NumpyEncoder))
    
    # Save updated index
    await self._save_memory_index()
```

### Loading Memory Entries

```python
async def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
    """Load a memory entry from disk."""
    if memory_id not in self.memory_index["memories"]:
        return None
    
    memory_path = os.path.join(self.storage_path, "memories", f"mem_{memory_id}.json")
    try:
        async with aiofiles.open(memory_path, "r") as f:
            content = await f.read()
            memory_dict = json.loads(content)
            memory = MemoryEntry(**memory_dict)
            return memory
    except FileNotFoundError:
        # Remove from index if file not found
        del self.memory_index["memories"][memory_id]
        await self._save_memory_index()
        return None
```

### Loading Assemblies

```python
async def load_assembly(self, assembly_id: str) -> Optional[MemoryAssembly]:
    """Load a memory assembly from disk."""
    if assembly_id not in self.memory_index["assemblies"]:
        return None
    
    assembly_path = os.path.join(self.storage_path, "assemblies", f"asm_{assembly_id}.json")
    try:
        async with aiofiles.open(assembly_path, "r") as f:
            content = await f.read()
            assembly_dict = json.loads(content)
            assembly = MemoryAssembly(**assembly_dict)
            return assembly
    except FileNotFoundError:
        # Remove from index if file not found
        del self.memory_index["assemblies"][assembly_id]
        await self._save_memory_index()
        return None
```

## Robust Index Saving

Phase 5.8 introduced improvements to ensure atomic index saves:

```python
async def _save_memory_index(self) -> None:
    """Save the memory index to disk with atomic guarantees."""
    index_path = os.path.join(self.storage_path, "memory_index.json")
    temp_path = f"{index_path}.tmp"
    
    # First write to temporary file
    async with aiofiles.open(temp_path, "w") as f:
        await f.write(json.dumps(self.memory_index))
        await f.flush()
        os.fsync(f.fileno())  # Ensure data is written to disk
    
    # Then atomically move to final location
    shutil.move(temp_path, index_path)
```

This approach ensures that the index file is never in a partially written state, which could happen if the system crashes during a write operation.

## Batch Operations

For efficiency, the persistence layer supports batch operations:

```python
async def save_memories_batch(self, memories: List[MemoryEntry]) -> None:
    """Save multiple memory entries in a batch."""
    for memory in memories:
        self.memory_index["memories"][memory.id] = {
            "id": memory.id,
            "timestamp": memory.timestamp,
            "file_path": f"memories/mem_{memory.id}.json"
        }
        
        memory_path = os.path.join(self.storage_path, "memories", f"mem_{memory.id}.json")
        async with aiofiles.open(memory_path, "w") as f:
            await f.write(json.dumps(memory.__dict__, cls=NumpyEncoder))
    
    # Save updated index once for all memories
    await self._save_memory_index()
```

## Planned Phase 5.9 Enhancements

In Phase 5.9, the persistence layer will be enhanced to support:

1. **Merge Logs**: A new file `logs/merge_log.jsonl` will store merge events in JSON Lines format.
2. **Activation Statistics**: A new file `stats/assembly_activation_stats.json` will store assembly activation statistics.
3. **Enhanced Error Handling**: Improved recovery from corrupted files.

```python
# Example of planned merge log persistence
async def log_merge_event(self, merge_event: Dict[str, Any]) -> None:
    """Log a merge event to the merge log file."""
    log_dir = os.path.join(self.storage_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "merge_log.jsonl")
    async with aiofiles.open(log_path, "a") as f:
        await f.write(json.dumps(merge_event) + "\n")
```

## Best Practices

1. **Atomicity**: Use the temporary file + move approach for important files.
2. **Error Handling**: Always handle file I/O exceptions and have recovery strategies.
3. **Batch Operations**: Use batch operations for bulk saves/loads.
4. **Directory Structure**: Maintain the established directory structure for compatibility.
5. **Backups**: Regularly back up the entire storage directory.
