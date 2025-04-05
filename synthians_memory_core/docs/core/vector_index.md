# Vector Index (FAISS)

This document explains the vector indexing system in the Synthians Memory Core, which uses FAISS for efficient similarity search.

## Overview

The vector index is a critical component that enables fast similarity-based retrieval of memories and assemblies. It maps unique identifiers to vector embeddings and provides efficient search capabilities.

The primary implementation is in the `MemoryVectorIndex` class, which wraps FAISS's `IndexIDMap` functionality with additional features for persistence, error handling, and GPU acceleration.

## Key Components

### IndexIDMap Structure

The system uses FAISS's `IndexIDMap` as the primary index structure, which allows:

1. Mapping between arbitrary IDs and internal FAISS indices
2. Adding and removing specific entries by ID
3. Efficient k-nearest neighbor search

```python
# Basic structure of the index in memory
self.index = faiss.IndexIDMap(base_index)  # base_index is usually IndexFlatL2 or IndexFlatIP
self.id_to_index = {}  # Maps string IDs to integer IDs used by FAISS
self.index_to_id = {}  # Reverse mapping for lookups
```

### CRITICAL LIMITATION: CPU-Bound ID Operations

**IMPORTANT:** While the underlying base index (e.g., `IndexFlatIP`) might support GPU search operations if `use_gpu=True` is set, FAISS `IndexIDMap` operations (`add_with_ids`, `remove_ids`) **execute on the CPU**. This is a fundamental limitation of the FAISS library architecture, not our implementation.

```python
# Even with GPU enabled, these operations run on CPU
self.index.add_with_ids(vectors_np, int_ids_np)  # CPU operation
self.index.remove_ids(np.array(int_ids).astype('int64'))  # CPU operation
```

This means:
1. Only search operations (`index.search()`) potentially benefit from GPU acceleration
2. All add/remove operations are CPU-bound regardless of GPU acceleration settings
3. Significant GPU acceleration is primarily seen for search operations on very large base indices *without* using `IndexIDMap`

This limitation informs our error handling strategy with the retry queue in `SynthiansMemoryCore`, as add/remove operations can be relatively slow and potentially fail.

### Persistence Layer

The vector index is persisted to disk for recovery after restarts:

```python
# The on-disk structure
storage_path/
├── vector_index/
    ├── faiss_index.bin              # The serialized FAISS index
    ├── faiss_index.bin.mapping.json # ID mapping for recovery
    └── faiss_index.bin.backup       # Optional backup during saves
```

The mapping file is critical for recovery, as it preserves the mapping between string IDs (used by the application) and integer IDs (used internally by FAISS).

## Key Operations

### Adding Entries

```python
async def add_with_ids(self, ids: List[str], vectors: List[List[float]]) -> None:
    # Convert to numpy arrays
    vectors_np = np.array(vectors).astype('float32')
    
    # Generate integer IDs for FAISS
    int_ids = self._generate_int_ids(len(ids))
    int_ids_np = np.array(int_ids).astype('int64')
    
    # Update the index (CPU operation, even with GPU-enabled index)
    self.index.add_with_ids(vectors_np, int_ids_np)
    
    # Update mappings
    for i, id_str in enumerate(ids):
        self.id_to_index[id_str] = int_ids[i]
        self.index_to_id[int_ids[i]] = id_str
```

### Removing Entries

```python
async def remove_ids(self, ids: List[str]) -> None:
    # Convert to FAISS integer IDs
    int_ids = []
    for id_str in ids:
        if id_str in self.id_to_index:
            int_ids.append(self.id_to_index[id_str])
            
    # Remove from index (CPU operation, even with GPU-enabled index)
    if int_ids:
        self.index.remove_ids(np.array(int_ids).astype('int64'))
        
        # Update mappings
        for id_str in ids:
            if id_str in self.id_to_index:
                int_id = self.id_to_index[id_str]
                del self.id_to_index[id_str]
                del self.index_to_id[int_id]
```

### Searching

```python
async def search(self, query_vectors: List[List[float]], k: int = 10) -> List[Dict[str, Any]]:
    query_np = np.array(query_vectors).astype('float32')
    
    # Search operation (GPU-accelerated if GPU enabled)
    distances, indices = self.index.search(query_np, k)
    
    # Convert results to string IDs
    results = []
    for i in range(len(query_vectors)):
        batch_results = []
        for j in range(k):
            if indices[i][j] != -1 and indices[i][j] in self.index_to_id:
                batch_results.append({
                    "id": self.index_to_id[indices[i][j]],
                    "distance": float(distances[i][j]),
                    "similarity": self._distance_to_similarity(distances[i][j])
                })
        results.append(batch_results)
    
    return results
```

## Phase 5.8 Stability Enhancements

Phase 5.8 introduced several critical improvements to the vector index:

### 1. Drift Detection

The system now tracks synchronization between memory objects and their vector representations using timestamps:

```python
# In MemoryAssembly
self.vector_index_updated_at = vector_index_updated_at  # Timestamp of last successful index update

# In retrieval logic
if assembly.vector_index_updated_at is None or (
    now - datetime.fromisoformat(assembly.vector_index_updated_at)).total_seconds() > self.config.ASSEMBLY_MAX_DRIFT_SECONDS:
    # Skip assembly for boosting - not synchronized
    continue
```

This prevents the system from using stale vector representations during retrieval.

### 2. Pending Updates Queue

Failed vector index operations are now queued for retry:

```python
try:
    await self.vector_index.add_with_ids([assembly.id], [assembly.composite_embedding])
    assembly.vector_index_updated_at = datetime.utcnow().isoformat()
except Exception as e:
    # Queue for retry
    await self._pending_vector_updates.put({
        "operation": "add",
        "id": assembly.id,
        "embedding": assembly.composite_embedding,
        "is_assembly": True
    })
```

This improves resilience to temporary FAISS failures.

### 3. Index Integrity Checking

The system now verifies index consistency on startup:

```python
async def check_index_integrity(self) -> Dict[str, Any]:
    issues = []
    
    # Check if index size matches ID mapping count
    if self.index.ntotal != len(self.id_to_index):
        issues.append(f"Index size mismatch: FAISS has {self.index.ntotal} entries, mapping has {len(self.id_to_index)}")
    
    # Check for missing mappings
    # ...
    
    return {
        "status": "healthy" if not issues else "issues_found",
        "issues": issues,
        # ...
    }
```

### 4. Automatic Repair

The system can rebuild the index from stored data:

```python
async def repair_index_async(self) -> Dict[str, Any]:
    # Get all memories and assemblies from persistence
    memories = await self.persistence.list_all_memories()
    assemblies = await self.persistence.list_all_assemblies()
    
    # Clear the index
    await self.vector_index.reset()
    
    # Rebuild from persistence
    # ...
    
    return {
        "status": "completed",
        "memories_added": memories_added,
        "assemblies_added": assemblies_added,
        # ...
    }
```

## GPU Integration

The system optionally supports GPU acceleration via FAISS's GPU indexing:

```python
if use_gpu:
    try:
        # Move index to GPU
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.using_gpu = True
    except Exception as e:
        logger.warning(f"Failed to initialize GPU index, falling back to CPU: {e}")
        self.using_gpu = False
```

**Important Limitations:**

1. Only search operations benefit from GPU acceleration.
2. Add and remove operations still run on CPU due to FAISS limitations with `IndexIDMap`.
3. The index is temporarily moved back to CPU for serialization during saves.

## Best Practices

1. **Batching**: Batch vector operations where possible to reduce overhead.
2. **Error Handling**: Always handle FAISS exceptions, as they can occur due to GPU memory issues.
3. **Index Size Monitoring**: Regularly check index size to ensure it remains manageable for your hardware.
4. **Regular Verification**: Use `check_index_integrity` periodically to detect inconsistencies.
5. **Backup Strategy**: Maintain backups of both the FAISS index and the ID mappings.
