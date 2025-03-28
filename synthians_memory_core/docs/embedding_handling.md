# Embedding Handling in Synthians Memory Core

## Overview

The Synthians Memory Core implements robust handling for embeddings throughout the system, addressing several critical challenges:

1. **Dimension Mismatches**: Safely handling vectors of different dimensions (e.g., 384 vs. 768)
2. **Malformed Embeddings**: Detecting and handling NaN/Inf values in embedding vectors
3. **Efficient Retrieval**: Using FAISS for fast similarity search with automatic GPU acceleration
4. **Component Compatibility**: Ensuring consistent behavior across different components through backward compatibility

## Recent Enhancements (March 2025)

Several significant improvements have been made to the embedding handling system:

### 1. Backward Compatibility Methods in GeometryManager

To ensure consistent behavior across all components, backward compatibility methods were added to the GeometryManager class:

```python
def _align_vectors(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward compatibility method that forwards to align_vectors."""
    return self.align_vectors(v1, v2)

def _normalize(self, vector: np.ndarray) -> np.ndarray:
    """Backward compatibility method that forwards to normalize_embedding."""
    # Ensure vector is numpy array before calling
    validated_vector = self._validate_vector(vector, "Vector for _normalize")
    if validated_vector is None:
        # Return zero vector if validation fails
        return np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)
    return self.normalize_embedding(validated_vector)
```

These methods ensure that both naming conventions (`_align_vectors`/`align_vectors` and `_normalize`/`normalize_embedding`) work correctly across all components.

### 2. Robust Vector Alignment

The system now includes comprehensive support for handling dimension mismatches between different embedding models (particularly 384D vs 768D vectors):

```python
def _align_vectors_for_comparison(v1, v2):
    """Safely align two vectors to the same dimension for comparison operations.
    
    If vectors have different dimensions, either pads the smaller one with zeros
    or truncates the larger one to match dimensions.
    
    Args:
        v1: First vector (numpy array)
        v2: Second vector (numpy array)
        
    Returns:
        Tuple of aligned vectors (v1_aligned, v2_aligned)
    """
    if v1.shape != v2.shape:
        # Get dimensions
        dim1 = v1.shape[0]
        dim2 = v2.shape[0]
        target_dim = max(dim1, dim2)
        
        # Align vectors to target dimension
        if dim1 < target_dim:
            v1 = np.pad(v1, (0, target_dim - dim1))
        elif dim1 > target_dim:
            v1 = v1[:target_dim]
            
        if dim2 < target_dim:
            v2 = np.pad(v2, (0, target_dim - dim2))
        elif dim2 > target_dim:
            v2 = v2[:target_dim]
    
    return v1, v2
```

### 3. Enhanced Embedding Validation

All embeddings are now validated more thoroughly with proper error handling:

```python
def _validate_embedding(embedding, allow_zero=True):
    """Validate that an embedding vector contains only valid values.
    
    Args:
        embedding: The embedding vector to validate
        allow_zero: Whether to allow zero vectors
        
    Returns:
        bool: True if the embedding is valid, False otherwise
    """
    if embedding is None:
        return False
        
    # Convert to numpy array if needed
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding, dtype=np.float32)
        
    # Check for NaN or Inf values
    if np.isnan(embedding).any() or np.isinf(embedding).any():
        return False
        
    # Optionally check for zero vectors
    if not allow_zero and np.all(embedding == 0):
        return False
        
    return True
```

## Embedding Validation

All embeddings in the system are validated before use to ensure robustness:

```python
def _validate_embedding(embedding):
    """Validate that an embedding vector contains only valid values.
    
    Args:
        embedding: The embedding vector to validate
        
    Returns:
        bool: True if the embedding is valid, False otherwise
    """
    if embedding is None:
        return False
        
    # Check for NaN or Inf values
    return not (np.isnan(embedding).any() or np.isinf(embedding).any())
```

Invalid embeddings are replaced with zero vectors to prevent crashes:

```python
def process_embedding(embedding):
    """Process and normalize an embedding, handling malformed inputs."""
    if not _validate_embedding(embedding):
        # Replace invalid embedding with zeros
        logger.warning("Invalid embedding detected (NaN/Inf values). Replacing with zeros.")
        return np.zeros(len(embedding), dtype=np.float32)
    
    # Normalize and return valid embedding
    return normalize_embedding(embedding)
```

## Dimension Alignment

The system can handle embeddings of different dimensions (primarily 384 vs. 768) using a vector alignment utility:

```python
def _align_vectors_for_comparison(vec1, vec2):
    """Align two vectors to the same dimension for comparison operations.
    
    If dimensions differ, either pads the smaller vector with zeros or
    truncates the larger vector to match dimensions.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        tuple: (aligned_vec1, aligned_vec2) with matching dimensions
    """
    dim1 = len(vec1)
    dim2 = len(vec2)
    
    if dim1 == dim2:
        return vec1, vec2
    
    if dim1 < dim2:
        # Pad vec1 with zeros to match vec2
        return np.pad(vec1, (0, dim2 - dim1)), vec2
    else:
        # Pad vec2 with zeros to match vec1
        return vec1, np.pad(vec2, (0, dim1 - dim2))
```

This ensures vector operations work correctly even when embeddings have different dimensions.

## Integration with FAISS Vector Index

The FAISS vector index implementation interacts with the embedding handling system:

### Dimension Handling

The `MemoryVectorIndex` is initialized with a specific dimension and validates all inputs:

```python
def add(self, memory_id: str, embedding: np.ndarray) -> None:
    """Add a memory embedding to the index.
    
    Args:
        memory_id: Unique identifier for the memory
        embedding: Embedding vector as numpy array
    """
    # Validate embedding
    if embedding is None:
        logger.warning(f"Attempted to add None embedding for memory {memory_id}")
        return
        
    if len(embedding) != self.dimension:
        logger.warning(
            f"Embedding dimension mismatch for memory {memory_id}: "
            f"Expected {self.dimension}, got {len(embedding)}"
        )
        # Align dimensions by padding or truncating
        embedding = self._align_embedding_dimension(embedding)
```

The `_align_embedding_dimension` method ensures all embeddings match the expected dimension:

```python
def _align_embedding_dimension(self, embedding):
    """Align embedding to the expected dimension.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        numpy.ndarray: Aligned embedding with correct dimension
    """
    current_dim = len(embedding)
    
    if current_dim == self.dimension:
        return embedding
        
    if current_dim < self.dimension:
        # Pad with zeros
        return np.pad(embedding, (0, self.dimension - current_dim))
    else:
        # Truncate
        return embedding[:self.dimension]
```

### Handling Malformed Embeddings

The vector index works with the validation system to safely handle malformed embeddings:

```python
def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar embeddings in the index.
    
    Args:
        query_embedding: Query embedding vector
        k: Number of nearest neighbors to retrieve
        
    Returns:
        List of (memory_id, similarity_score) tuples
    """
    # Validate query embedding
    if not _validate_embedding(query_embedding):
        logger.warning("Invalid query embedding (NaN/Inf values). Replacing with zeros.")
        query_embedding = np.zeros(self.dimension, dtype=np.float32)
    
    # Align dimensions if needed
    if len(query_embedding) != self.dimension:
        query_embedding = self._align_embedding_dimension(query_embedding)
```

## Memory Retrieval Improvements

Memory retrieval has been enhanced with several improvements:

1. **Lowered Pre-filter Threshold**: Reduced from 0.5 to 0.3 for better recall sensitivity
2. **Explicit Threshold Parameter**: Added client and server-side support for explicit threshold control
3. **Enhanced Logging**: Added detailed similarity score logging for debugging

Example from `_get_candidate_memories`:

```python
async def _get_candidate_memories(self, query_embedding: np.ndarray, limit: int, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Get candidate memories using vector similarity search.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results to return
        threshold: Optional similarity threshold (if None, uses default)
        
    Returns:
        List of candidate memories with similarity scores
    """
    # Apply default threshold if not specified
    threshold = threshold if threshold is not None else self.default_threshold
    
    # Validate embedding
    if not _validate_embedding(query_embedding):
        logger.warning("Invalid query embedding detected in _get_candidate_memories")
        query_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
    
    # Perform vector search using the FAISS index
    results = self.vector_index.search(query_embedding, k=limit * 2)  # Get extra results for filtering
    
    # Filter and format results
    candidates = []
    for memory_id, score in results:
        if score < threshold:  # Lower scores are better for L2 distance
            logger.debug(f"Memory {memory_id} filtered out with score {score} (threshold: {threshold})")
            continue
            
        # Fetch the full memory and add to candidates
        memory = await self.get_memory_by_id(memory_id)
        if memory:
            memory['similarity_score'] = float(score)
            candidates.append(memory)
            
        if len(candidates) >= limit:
            break
            
    return candidates
```

## Testing

Comprehensive tests ensure embedding handling is robust:

- **Validation Tests**: Verify detection of NaN/Inf values
- **Alignment Tests**: Confirm vectors of different dimensions are properly aligned
- **Threshold Tests**: Ensure memory retrieval works with various thresholds

## Conclusion

The embedding handling system in Synthians Memory Core provides a robust foundation for vector operations. Combined with the FAISS vector index implementation, it ensures efficient and reliable memory retrieval while gracefully handling edge cases like dimension mismatches and malformed embeddings.
