# Embedding Validation Guide

## Overview

The Synthians Memory Core now includes robust embedding validation utilities that protect against common failure modes when working with vector embeddings. This guide demonstrates how and when to use these utilities in your code.

## Key Functions

### `validate_embedding`

```python
from synthians_memory_core.utils.embedding_validators import validate_embedding

# Basic usage
validated_emb = validate_embedding(embedding, target_dim=768)

# With normalization option
validated_emb = validate_embedding(embedding, target_dim=768, normalize=True, index_type='L2')
```

**When to use**: 
- Before storing embeddings in the vector index
- When receiving embeddings from external sources
- Before any critical computation that requires valid embedding values

**Example use case**:
```python
async def process_new_memory(self, content, embedding):
    # Validate the embedding before storage
    validated_emb = validate_embedding(embedding, target_dim=self.config['embedding_dim'])
    if validated_emb is None:
        logger.warning(f"Failed to validate embedding for content: {content[:50]}...")
        # Create a zero embedding as fallback or reject entirely
        validated_emb = np.zeros(self.config['embedding_dim'], dtype=np.float32)
    
    # Proceed with the validated embedding
    mem_id = f"mem_{uuid.uuid4().hex[:12]}"
    memory = MemoryEntry(
        content=content,
        embedding=validated_emb,
        id=mem_id,
        # ... other fields
    )
```

### `safe_normalize`

```python
from synthians_memory_core.utils.embedding_validators import safe_normalize

# Basic usage
normalized_vector = safe_normalize(vector)
```

**When to use**:
- Before calculating cosine similarity
- When preparing embeddings for vector search
- When vectors need to be normalized but might contain invalid values

**Example use case**:
```python
def _prepare_vector_for_search(self, query_vector):
    # Ensure the vector is valid and normalized
    normalized = safe_normalize(query_vector)
    if np.all(normalized == 0):
        logger.warning("Query vector could not be normalized, using fallback")
        # Consider using a fallback strategy for zero vectors
    
    return normalized
```

### `safe_calculate_similarity`

```python
from synthians_memory_core.utils.embedding_validators import safe_calculate_similarity

# Basic usage - handles all validation internally
similarity = safe_calculate_similarity(vector1, vector2)
```

**When to use**:
- When calculating similarity between two vectors that might:
  - Have different dimensions
  - Contain NaN/Inf values
  - Have near-zero norms

**Example use case**:
```python
def calculate_relevance(self, query_embedding, memory_embedding):
    # Safely calculate similarity with built-in protection
    similarity = safe_calculate_similarity(query_embedding, memory_embedding)
    
    # Apply any additional relevance factors
    recency_factor = self._calculate_recency_factor(memory)
    
    return similarity * recency_factor
```

### `align_vectors_for_comparison`

```python
from synthians_memory_core.utils.embedding_validators import align_vectors_for_comparison

# Basic usage
aligned_vec1, aligned_vec2 = align_vectors_for_comparison(vector1, vector2)
```

**When to use**:
- When comparing vectors of potentially different dimensions
- When implementing custom similarity measures
- When preparing vectors for operations that require matching dimensions

**Example use case**:
```python
def custom_weighted_similarity(self, vec1, vec2, weights=None):
    # First align the vectors to ensure they have same dimensions
    aligned_vec1, aligned_vec2 = align_vectors_for_comparison(vec1, vec2)
    
    if aligned_vec1 is None or aligned_vec2 is None:
        return 0.0  # Fallback for failed alignment
    
    # Apply custom weighting if provided
    if weights is not None:
        # Ensure weights match the aligned dimension
        if len(weights) != len(aligned_vec1):
            weights = np.ones_like(aligned_vec1)  # Fallback to equal weights
        
        # Apply weights to the vectors
        weighted_vec1 = aligned_vec1 * weights
        weighted_vec2 = aligned_vec2 * weights
        
        # Calculate similarity with the weighted vectors
        return safe_calculate_similarity(weighted_vec1, weighted_vec2)
    
    return safe_calculate_similarity(aligned_vec1, aligned_vec2)
```

## Integration with Memory Core

These validation functions are already integrated into key operations in the Memory Core:

1. **Memory Processing**: The `process_new_memory` method automatically validates embeddings
2. **Assembly Update**: The `_update_assemblies` method validates assembly composite embeddings
3. **Retrieval Pipeline**: The vector similarity calculation uses safe similarity measures
4. **Vector Index Operations**: All add/update operations include embedding validation

## Best Practices

1. **Always validate external inputs**: Any embedding coming from outside your system should be validated

2. **Use safe similarity calculation**: Prefer `safe_calculate_similarity` over raw dot products

3. **Handle dimension mismatches**: Be prepared for embeddings with unexpected dimensions

4. **Check validation results**: Always check if validation returned `None` and have a fallback strategy

5. **Log validation failures**: When validation fails, log relevant details for debugging

6. **Test with malformed data**: Explicitly test your code with NaN/Inf values to ensure it handles them gracefully

## Debugging Tips

1. If you encounter unexpected zero results, check if your vectors failed validation and were replaced with zeros

2. Enable DEBUG logging for `synthians_memory_core.utils.embedding_validators` to see detailed validation warnings

3. Check the vector stats before and after validation operations

4. Use `np.isfinite(vector).all()` to manually verify vector validity at key points
