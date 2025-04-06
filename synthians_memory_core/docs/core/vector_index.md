# Vector Index (FAISS)

This document explains the vector indexing system in the Synthians Memory Core, which uses FAISS for efficient similarity search.

## Overview

The vector index is a critical component that enables fast similarity-based retrieval of memories and assemblies. It maps unique identifiers to vector embeddings and provides efficient search capabilities.

The primary implementation is in the `MemoryVectorIndex` class, which wraps FAISS's `IndexIDMap` functionality with additional features for persistence, error handling, asynchronous operations, and robust recovery mechanisms.

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

## Asynchronous Operations

As of Phase 5.8/5.9, the vector index implements comprehensive asynchronous support to prevent blocking the main event loop:

### Core Async Vector Operations

* **`search_knn_async`**: Performs k-nearest neighbor search asynchronously
* **`add_vector_async`**: Adds single vectors without blocking
* **`add_batch_async`**: Efficiently adds multiple vectors in batches asynchronously
* **`remove_vector_async`**: Removes vectors from the index asynchronously
* **`update_vector_async`**: Updates vector embeddings without blocking
* **`update_entry_async`**: Complete entry update that handles embedding updates via remove+add pattern

### Async Index Management

* **`save_async`** and **`load_async`**: Asynchronous file I/O for index persistence
* **`reset_async`**: Non-blocking index reset operation
* **`repair_index_async`**: Asynchronously repairs corrupted indices
* **`check_index_integrity`**: Verifies the integrity of the FAISS index and ID mappings

All asynchronous methods use `asyncio.run_in_executor` to wrap CPU-bound FAISS operations, preventing them from blocking the event loop.

```python
async def add_vector_async(self, id_str, vector):
    """Add a vector to the index asynchronously."""
    # Vector validation and preprocessing
    # ...
    
    # Run the CPU-bound operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: self.add_vector(id_str, vector)
    )
```

## Vector Index Persistence

The index is persisted to disk in two primary files:

1. **`faiss_index.bin`**: The serialized FAISS index (binary format)
2. **`faiss_id_mapping.json`**: The ID mappings (JSON format)

These files are managed via:

* **`save`/`save_async`**: Atomic saving of both index and mappings
* **`load`/`load_async`**: Loading both index and mappings during initialization

Backups are also automatically created during critical operations:

```python
backup_path = f"{path}.bak-{int(time.time())}"
shutil.copy2(path, backup_path)  # Atomic copy
```

## Validation & Integrity Checking

The system implements robust validation to catch potential issues:

* **`check_index_integrity`**: Verifies that the FAISS index and ID mappings are consistent
* **`validate_vector_index_integrity`**: Higher-level validation including dimension checks
* **Dimension Validation**: Ensures all vectors match the expected dimension
* **NaN/Inf Detection**: Identifies problematic vector values

```python
async def check_index_integrity(self, persistence=None) -> Tuple[bool, Dict[str, Any]]:
    """Returns (is_consistent, diagnostics)"""
    diagnostics = {}
    consistent = True
    
    # Check that all ID mappings are consistent
    id_map_size = len(self.id_to_index)
    reverse_map_size = len(self.index_to_id)
    faiss_ntotal = self.index.ntotal if self.index else 0
    
    diagnostics["id_map_size"] = id_map_size
    diagnostics["reverse_map_size"] = reverse_map_size
    diagnostics["faiss_ntotal"] = faiss_ntotal
    
    if id_map_size != reverse_map_size or id_map_size != faiss_ntotal:
        consistent = False
        diagnostics["error"] = "Size mismatch between mappings and FAISS index"
    
    # Additional validation checks
    # ...
    
    return consistent, diagnostics
```

## Repair Mechanisms

When integrity issues are detected, the system can self-heal:

* **`repair_index_async`**: Main entry point for index repair, which attempts multiple strategies:
  1. **Fix ID mappings**: Rebuilds mappings from the FAISS index
  2. **Full rebuild**: If mapping repair fails, rebuilds the entire index from persistence

* **`_rebuild_id_mapping_from_index_async`**: Attempts to extract and rebuild ID mappings from FAISS
* **`_rebuild_index_from_persistence_async`**: Complete rebuild using persisted memory and assembly data

```python
async def repair_index_async(self, persistence=None, geometry_manager=None):
    """Comprehensive index repair function"""
    if not persistence or not geometry_manager:
        raise ValueError("Persistence and GeometryManager required for repair")
    
    # Try to fix just the ID mappings first (less expensive)
    mapping_fixed = await self._rebuild_id_mapping_from_index_async()
    if mapping_fixed:
        return True
        
    # If mapping repair fails, do a full rebuild from persistence
    return await self._rebuild_index_from_persistence_async(persistence, geometry_manager)
```

## Embedding Dimension Handling

The system supports handling embedding dimension mismatches:

* **Input validation**: Ensures all input vectors have the correct dimension
* **`_align_embeddings_async`**: Handles dimension mismatch by resizing vectors when needed
* **Dimension consistency**: Maintains consistent dimensions across the index

## Error Handling

The system uses comprehensive error handling to prevent failures:

* **`RunTimeError`**: Raised for critical index inconsistencies
* **Trace-based logging**: Detailed logs for diagnosis with trace IDs
* **State validation**: Checks for index initialization before operations
* **Robust locks**: Proper async locking to prevent concurrent modification
* **Exception capture**: All index operations are wrapped in try/except blocks

## Integration with Memory Core

The vector index is integrated into the Memory Core through:

* **Synchronized timestamps**: `MemoryAssembly.vector_index_updated_at` tracks when an assembly's embedding was successfully added to the index
* **Retry queue**: Failed index operations are queued in `_pending_vector_updates` for automatic retry
* **`_vector_update_retry_loop`**: Background task that periodically attempts to process the retry queue
* **Drift detection**: `detect_and_repair_index_drift` identifies mismatches between persisted assemblies and the vector index

## Phase 5.8 Stability Enhancements

Phase 5.8/5.9 introduced several critical improvements to the vector index:

### 1. Comprehensive Asynchronous Support
* Implemented complete async versions of all vector operations
* Properly wrapped CPU-bound operations with `run_in_executor`
* Added dimension checking and automatic resizing of vectors when needed

### 2. Drift Detection & Recovery
* Better detection of index-to-persistence mismatches
* `_pending_vector_updates` queue to track failed operations
* Background retry loop for automatic recovery

### 3. Improved Error Handling
* Enhanced trace ID-based logging for better diagnostics
* Comprehensive exception handling with state management
* Atomic file operations for safer persistence

### 4. Batch Processing Optimization
* Configurable batch sizes to avoid memory issues
* More efficient bulk operations

### 5. Repair Mechanisms
* Multi-tiered repair strategy (mapping fix → full rebuild)
* Integration with the API via `/repair_index` endpoint
* Self-healing via periodic integrity checks
