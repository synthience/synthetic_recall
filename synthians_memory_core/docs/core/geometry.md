# Geometry Management

This document details the `GeometryManager` component of the Synthians Memory Core, which handles vector operations, normalization, and alignment across different embedding spaces.

## Overview

The `GeometryManager` is responsible for ensuring consistent handling of embeddings, regardless of their dimensionality or the underlying geometry. It provides methods for:

1. Vector normalization
2. Similarity calculation
3. Embedding alignment (adjusting dimensions)
4. Composite embedding generation

## Key Functions

### Vector Normalization

```python
def normalize_vector(self, vector: List[float]) -> List[float]:
    """Normalize a vector based on the configured geometry."""
    if self.geometry == "euclidean":
        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            return [x / norm for x in vector]
        return vector
    elif self.geometry == "hyperbolic":
        # Hyperbolic normalization (ensures vector remains within unit ball)
        norm = math.sqrt(sum(x * x for x in vector))
        if norm >= 1.0:  # Prevent vectors outside the unit ball
            return [x / (norm + 1e-5) for x in vector]
        return vector
    else:
        # Default to L2 normalization
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            return [x / norm for x in vector]
        return vector
```

### Similarity Calculation

```python
def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
    """Calculate similarity between two vectors based on the configured geometry."""
    # Align dimensions if needed
    vec1, vec2 = self.align_embeddings(vec1, vec2)
    
    if self.geometry == "euclidean":
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    elif self.geometry == "hyperbolic":
        # Hyperbolic distance-based similarity
        # Convert distance to similarity score
        distance = self._hyperbolic_distance(vec1, vec2)
        return 1.0 / (1.0 + distance)
    else:
        # Default to cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
```

### Embedding Alignment

```python
def align_embeddings(self, vec1: List[float], vec2: List[float]) -> Tuple[List[float], List[float]]:
    """Align two embeddings to the same dimensionality."""
    len1, len2 = len(vec1), len(vec2)
    
    if len1 == len2:
        return vec1, vec2
    
    # Determine target dimension (by default, the larger one)
    target_dim = max(len1, len2)
    if self.embedding_dim is not None:
        target_dim = self.embedding_dim
    
    # Align based on strategy
    if self.alignment_strategy == "pad":
        # Zero-pad both vectors to target dimension
        aligned1 = vec1 + [0.0] * (target_dim - len1) if len1 < target_dim else vec1[:target_dim]
        aligned2 = vec2 + [0.0] * (target_dim - len2) if len2 < target_dim else vec2[:target_dim]
    elif self.alignment_strategy == "truncate":
        # Truncate both vectors to minimum dimension
        min_dim = min(target_dim, len1, len2)
        aligned1 = vec1[:min_dim]
        aligned2 = vec2[:min_dim]
    else:
        # Default behavior: hybrid approach - truncate larger vectors, pad smaller ones
        aligned1 = vec1[:target_dim] if len1 > target_dim else vec1 + [0.0] * (target_dim - len1)
        aligned2 = vec2[:target_dim] if len2 > target_dim else vec2 + [0.0] * (target_dim - len2)
    
    return aligned1, aligned2
```

**Important Note on Default Alignment Strategy**: If `alignment_strategy` is not explicitly set to 'pad' or 'truncate', the default behavior is a hybrid approach: vectors larger than `embedding_dim` are truncated, while vectors smaller than `embedding_dim` are zero-padded to match the target dimension. This strikes a balance between preserving information and ensuring consistent dimensionality.

### Composite Embedding Generation

```python
def generate_composite_embedding(self, embeddings: List[List[float]]) -> List[float]:
    """Generate a composite embedding from multiple embeddings."""
    if not embeddings:
        return []
    
    # Align all embeddings to the same dimensionality
    aligned_embeddings = []
    for emb in embeddings:
        aligned = self.align_embedding(emb)
        aligned_embeddings.append(aligned)
    
    # Simple average as starting point
    dim = len(aligned_embeddings[0])
    composite = [0.0] * dim
    for emb in aligned_embeddings:
        for i in range(dim):
            composite[i] += emb[i] / len(aligned_embeddings)
    
    # Normalize the composite embedding
    return self.normalize_vector(composite)
```

## Configuration Options

The `GeometryManager` accepts several configuration options:

```python
def __init__(
    self,
    embedding_dim: int = 768,
    geometry: str = "euclidean",
    alignment_strategy: Optional[str] = None,
    normalization_epsilon: float = 1e-5
):
    """Initialize the GeometryManager.
    
    Args:
        embedding_dim (int): Target embedding dimension.
        geometry (str): Geometry type ("euclidean" or "hyperbolic").
        alignment_strategy (str, optional): Strategy for aligning embeddings
            of different dimensions ("pad", "truncate", or None for hybrid).
        normalization_epsilon (float): Small value to prevent division by zero.
    """
```

The default configuration uses:
- Euclidean geometry with cosine similarity
- 768-dimensional embeddings (common for models like BERT)
- Hybrid alignment strategy (truncate larger, pad smaller)
- Epsilon of 1e-5 for numerical stability

## Hyperbolic Geometry

For hyperbolic geometry, the manager uses the Poincaré ball model:

```python
def _hyperbolic_distance(self, vec1: List[float], vec2: List[float]) -> float:
    """Calculate the hyperbolic distance between two vectors in the Poincaré ball."""
    # Ensure vectors are within the unit ball
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    if norm1 >= 1.0 or norm2 >= 1.0:
        # Project back into the unit ball if needed
        vec1 = [x / (norm1 + 1e-5) for x in vec1] if norm1 >= 1.0 else vec1
        vec2 = [x / (norm2 + 1e-5) for x in vec2] if norm2 >= 1.0 else vec2
    
    # Calculate Euclidean distance
    euclidean_distance_squared = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
    
    # Calculate hyperbolic distance using the Poincaré formula
    norm1_squared = sum(x * x for x in vec1)
    norm2_squared = sum(x * x for x in vec2)
    
    numerator = 2 * euclidean_distance_squared
    denominator = (1 - norm1_squared) * (1 - norm2_squared)
    
    if denominator <= 0:
        return 100.0  # Large value for invalid points
    
    cosh_distance = 1 + numerator / denominator
    
    # Avoid NaN with clipping
    if cosh_distance < 1.0:
        cosh_distance = 1.0
    
    return math.acosh(cosh_distance)
```

## Usage Examples

### Simple Similarity Calculation

```python
# Create a geometry manager with default settings
geometry_manager = GeometryManager()

# Calculate similarity between two embeddings
vec1 = [0.1, 0.2, 0.3, 0.4]
vec2 = [0.2, 0.3, 0.4, 0.5]
similarity = geometry_manager.calculate_similarity(vec1, vec2)
print(f"Similarity: {similarity:.4f}")
```

### Handling Different Dimensions

```python
# Create a geometry manager with explicit alignment strategy
geometry_manager = GeometryManager(embedding_dim=3, alignment_strategy="pad")

# Calculate similarity between embeddings of different dimensions
vec1 = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5-dimensional
vec2 = [0.3, 0.4]                 # 2-dimensional

# Will be aligned to [0.1, 0.2, 0.3] and [0.3, 0.4, 0.0] respectively
similarity = geometry_manager.calculate_similarity(vec1, vec2)
print(f"Similarity after alignment: {similarity:.4f}")
```

### Creating Composite Embeddings

```python
# Create a geometry manager
geometry_manager = GeometryManager()

# Generate a composite embedding from multiple embeddings
embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]
composite = geometry_manager.generate_composite_embedding(embeddings)
print(f"Composite embedding: {composite}")
```

## Best Practices

1. **Consistency**: Use the same `GeometryManager` instance throughout your application to ensure consistent handling of embeddings.
2. **Configuration**: Set `embedding_dim` to match your preferred model's output dimension.
3. **Alignment Strategy**: Choose an alignment strategy based on your needs:
   - "pad" preserves all information but may introduce noise
   - "truncate" loses information but focuses on the most important dimensions
   - Default hybrid approach (truncate larger, pad smaller) balances these concerns
4. **Model Compatibility**: Ensure your choice of geometry is compatible with your embedding model.
