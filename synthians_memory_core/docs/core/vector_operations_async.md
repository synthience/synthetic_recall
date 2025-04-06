# Asynchronous Vector Operations (Phase 5.8-5.9)

**Document Version:** 1.0  
**Phase:** 5.8-5.9

This document describes the asynchronous operation support implemented for the FAISS-based vector index in the Synthians Memory Core, providing better performance and stability for memory operations.

## Overview

In Phase 5.8 and 5.9, comprehensive asynchronous support was implemented for all vector index operations to prevent blocking the main event loop during CPU-bound FAISS operations. This significantly improves the responsiveness of the system, especially during high-load scenarios, and prevents request timeouts when performing intensive vector operations.

All asynchronous methods provide compatibility with their synchronous counterparts, which now serve as fallbacks. The implementation uses `asyncio.run_in_executor` to wrap CPU-bound operations, allowing them to run in a separate thread without blocking the main asyncio event loop.

## Core Asynchronous Operations

### Vector Addition and Removal

```python
async def add_vector_async(self, id_str: str, vector: np.ndarray) -> bool:
    """Add a vector to the index asynchronously.
    
    Args:
        id_str: String identifier for the vector
        vector: The embedding vector to add
        
    Returns:
        bool: True if successful
    """
    # Validate and preprocess vector
    vector = self._validate_and_process_vector(vector)
    if vector is None:
        return False
        
    # Run the CPU-bound operation in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: self.add_vector(id_str, vector)
    )

async def remove_vector_async(self, id_str: str) -> bool:
    """Remove a vector from the index asynchronously.
    
    Args:
        id_str: String identifier for the vector to remove
        
    Returns:
        bool: True if successful
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.remove_vector(id_str)
    )

async def update_vector_async(self, id_str: str, new_vector: np.ndarray) -> bool:
    """Update a vector in the index asynchronously.
    
    Implemented as a remove + add operation with proper error handling.
    
    Args:
        id_str: String identifier for the vector to update
        new_vector: The new embedding vector
        
    Returns:
        bool: True if successful
    """
    # Validate and preprocess vector
    new_vector = self._validate_and_process_vector(new_vector)
    if new_vector is None:
        return False
    
    # Use a two-step remove then add approach
    removed = await self.remove_vector_async(id_str)
    if not removed:
        logger.warning(f"Vector {id_str} not found for update, adding as new")
        
    # Add the vector with new embedding
    return await self.add_vector_async(id_str, new_vector)

async def add_batch_async(self, id_strs: List[str], vectors: List[np.ndarray], 
                          batch_size: int = 100) -> bool:
    """Add multiple vectors in batches asynchronously.
    
    Args:
        id_strs: List of string identifiers
        vectors: List of embedding vectors
        batch_size: Number of vectors to process in each batch
        
    Returns:
        bool: True if all operations were successful
    """
    if len(id_strs) != len(vectors):
        logger.error(f"ID count ({len(id_strs)}) doesn't match vector count ({len(vectors)})")
        return False
        
    # Process in batches to avoid memory issues with large operations
    for i in range(0, len(id_strs), batch_size):
        batch_ids = id_strs[i:i+batch_size]
        batch_vectors = vectors[i:i+batch_size]
        
        # Create tasks for each vector in the batch
        tasks = [self.add_vector_async(id_str, vector) 
                for id_str, vector in zip(batch_ids, batch_vectors)]
        
        # Run all tasks in the batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error adding vector {batch_ids[j]}: {result}")
                return False
            elif not result:
                logger.error(f"Failed to add vector {batch_ids[j]}")
                return False
                
    return True
```

### Search Operations

```python
async def search_knn_async(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
    """Search for k-nearest neighbors asynchronously.
    
    Args:
        query_vector: The query embedding vector
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Tuple[List[str], List[float]]: IDs and distances of nearest neighbors
    """
    # Validate and preprocess vector
    query_vector = self._validate_and_process_vector(query_vector)
    if query_vector is None:
        return [], []
        
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.search_knn(query_vector, k)
    )

async def _align_embeddings_async(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Align embedding dimensions asynchronously.
    
    Args:
        vectors: List of vectors that might need dimension adjustment
        
    Returns:
        List[np.ndarray]: Aligned vectors matching the index dimension
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self._align_embeddings(vectors)
    )
```

## Index Management Operations

```python
async def save_async(self, filepath: str) -> bool:
    """Save the index to disk asynchronously.
    
    Args:
        filepath: Path to save the index file
        
    Returns:
        bool: True if successful
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    loop = asyncio.get_event_loop()
    index_saved = await loop.run_in_executor(
        None,
        lambda: self.save(filepath)
    )
    
    # Also save the ID mapping asynchronously
    if index_saved:
        mapping_path = f"{filepath}.mapping.json"
        try:
            async with aiofiles.open(mapping_path, "w") as f:
                await f.write(json.dumps(self.id_to_index))
            return True
        except Exception as e:
            logger.error(f"Error saving ID mapping: {e}")
            return False
    return False

async def load_async(self, filepath: str) -> bool:
    """Load the index from disk asynchronously.
    
    Args:
        filepath: Path to the index file
        
    Returns:
        bool: True if successful
    """
    if not os.path.exists(filepath):
        logger.error(f"Index file not found: {filepath}")
        return False
        
    mapping_path = f"{filepath}.mapping.json"
    if not os.path.exists(mapping_path):
        logger.error(f"ID mapping file not found: {mapping_path}")
        return False
    
    # Load ID mapping asynchronously
    try:
        async with aiofiles.open(mapping_path, "r") as f:
            mapping_content = await f.read()
        self.id_to_index = json.loads(mapping_content)
        
        # Load the FAISS index in a separate thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.load(filepath)
        )
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return False

async def reset_async(self) -> bool:
    """Reset the index asynchronously, clearing all vectors.
    
    Returns:
        bool: True if successful
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.reset()
    )

async def _initialize_index_async(self, dim: int) -> bool:
    """Initialize a new FAISS index asynchronously.
    
    Args:
        dim: Dimension of the vectors to be indexed
        
    Returns:
        bool: True if successful
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self._initialize_index(dim)
    )
```

## Diagnostic and Recovery Operations

```python
async def repair_index_async(self) -> bool:
    """Repair a potentially corrupted index asynchronously.
    
    Returns:
        bool: True if repair was successful
    """
    try:
        return await self._repair_index_async()
    except Exception as e:
        logger.error(f"Failed to repair index: {e}")
        return False

async def _repair_index_async(self) -> bool:
    """Internal implementation of index repair.
    
    Performs a reset and rebuilds from ID mapping if possible,
    or creates a fresh index if not.
    
    Returns:
        bool: True if successful
    """
    # Step 1: Back up ID mapping
    backup_mapping = await self._backup_id_mapping_async()
    
    # Step 2: Reset the index
    await self.reset_async()
    
    # Step 3: Attempt to rebuild from backup if available
    if backup_mapping:
        logger.info("Rebuilding index from ID mapping backup")
        return await self._rebuild_id_mapping_from_index_async(backup_mapping)
    else:
        logger.warning("No ID mapping backup available, creating fresh index")
        return await self._initialize_index_async(self.dim)

async def _backup_id_mapping_async(self) -> Dict[str, int]:
    """Create a backup of the current ID mapping asynchronously.
    
    Returns:
        Dict[str, int]: Copy of the current ID mapping, or empty dict on failure
    """
    try:
        # Make a deep copy to avoid reference issues
        return copy.deepcopy(self.id_to_index) 
    except Exception as e:
        logger.error(f"Failed to backup ID mapping: {e}")
        return {}

async def _rebuild_id_mapping_from_index_async(self, mapping: Dict[str, int]) -> bool:
    """Rebuild index using a saved mapping asynchronously.
    
    Args:
        mapping: Dictionary mapping ID strings to index positions
        
    Returns:
        bool: True if successful
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self._rebuild_id_mapping_from_index(mapping)
    )

async def test_faiss_index_directly_async(self, vector: np.ndarray) -> bool:
    """Test if the FAISS index is working correctly asynchronously.
    
    Args:
        vector: Test vector to use for basic operations
        
    Returns:
        bool: True if basic operations succeed
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.test_faiss_index_directly(vector)
    )
```

## Integration with MemoryCore

All operations in `MemoryCore` that interact with the vector index now use these asynchronous methods, ensuring that CPU-bound FAISS operations do not block the main event loop. This significantly improves the responsiveness of the API server, especially during high-load scenarios.

Key integration points include:

1. **Memory Creation and Updates**:
   ```python
   # In MemoryCore.create_memory
   await self.vector_index.add_vector_async(memory.id, memory.embedding)
   
   # In MemoryCore.update_memory
   await self.vector_index.update_vector_async(memory.id, memory.embedding)
   ```

2. **Assembly Management**:
   ```python
   # In MemoryCore.create_assembly
   await self.vector_index.add_vector_async(assembly.id, assembly.composite_embedding)
   
   # In MemoryCore.update_assembly_index
   await self.vector_index.update_vector_async(assembly.id, assembly.composite_embedding)
   ```

3. **Vector Search**:
   ```python
   # In MemoryCore.search_similar_assemblies
   assembly_ids, distances = await self.vector_index.search_knn_async(query_embedding, k)
   ```

4. **Index Integrity Checks**:
   ```python
   # In MemoryCore.check_index_integrity
   # Now uses vector_index.check_integrity() which performs validation
   # without blocking the event loop
   consistent, diagnostics = await self.vector_index.check_integrity_async()
   ```

## Error Handling and Recovery

Asynchronous operations include robust error handling, with specific error signatures allowing for more granular detection and resolution of issues. The system can now better recover from transient failures and maintain state consistency.

```python
try:
    success = await self.vector_index.add_vector_async(memory.id, memory.embedding)
    if not success:
        logger.error(f"Failed to add vector for memory {memory.id}")
        # Attempt recovery
        if await self.vector_index.check_integrity_async()[0] is False:
            logger.warning("Index integrity check failed, attempting repair")
            await self.vector_index.repair_index_async()
            # Retry operation
            success = await self.vector_index.add_vector_async(memory.id, memory.embedding)
 except Exception as e:
    logger.error(f"Vector index operation failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

1. **Thread Pool Management**: Uses the default asyncio executor, which creates a reasonable number of worker threads based on system capabilities.

2. **Batch Processing**: Large vector operations (e.g., adding many vectors) use configurable batch sizes to avoid memory issues.

3. **Dimension Validation**: All vector operations include automatic dimension checking and alignment to handle potential embedding dimension mismatches.

4. **Lazy Loading**: The index is loaded only when needed, and operations are designed to minimize memory usage.

## Configuration

The asynchronous operations respect the following configuration options:

```python
VECTOR_INDEX_CONFIG = {
    "index_type": "IDMap,Flat",  # FAISS index type
    "metric_type": "InnerProduct",  # Similarity metric
    "embedding_dim": 768,  # Default embedding dimension
    "max_batch_size": 100,  # Maximum vectors per batch operation
    "repair_on_error": True,  # Auto-repair index on error
    "backup_interval_hours": 24,  # How often to back up index
}
```

## Future Enhancements

1. **Custom Thread Pool**: Implement a dedicated thread pool for vector operations to better control resource allocation.

2. **Persistent Transaction Log**: Maintain a log of vector operations to enable more robust recovery in case of failures.

3. **Progressive Index Loading**: Load only portions of the vector index on demand for improved memory efficiency with large indexes.

4. **Memory-Mapped Indices**: Use memory-mapped FAISS indices for handling very large vector collections.
