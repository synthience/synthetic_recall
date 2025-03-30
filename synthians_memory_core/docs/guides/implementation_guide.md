# Bi-Hemispheric Cognitive Architecture: Implementation Guide

## Introduction

This technical guide explains how to implement and integrate the components of the Bi-Hemispheric Cognitive Architecture. It covers deployment, configuration, and development patterns to extend the system.

## System Requirements

- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (optional, for accelerated embedding generation)
- 8GB+ RAM 

## Component Deployment

### Using Docker Compose

The easiest way to deploy the full architecture is using the included `docker-compose-bihemispheric.yml` file:

```bash
docker-compose -f docker-compose-bihemispheric.yml up -d
```

This launches all three components (Memory Core, Trainer Server, and Context Cascade Engine) with proper networking and configuration.

### Manual Deployment

To run components individually (useful for development):

1. **Memory Core**
   ```bash
   cd synthians_memory_core
   python -m server.main
   ```

2. **Trainer Server**
   ```bash
   cd synthians_memory_core/synthians_trainer_server
   python -m http_server
   ```

3. **Context Cascade Engine**
   ```bash
   cd synthians_memory_core/orchestrator
   python -m server
   ```

## Configuration

### Environment Variables

The architecture uses the following environment variables (can be set in Docker Compose or locally):

```
# Memory Core
PORT=8000
VECTOR_DB_PATH=./vectordb
MEMORY_STORE_PATH=./memorystore
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=768
GEOMETRY_TYPE=euclidean
ALIGNMENT_STRATEGY=truncate
VECTOR_INDEX_TYPE=L2
RETRIEVAL_THRESHOLD=0.3

# Trainer Server
PORT=8001
MEMORY_CORE_URL=http://memory_core:8000
INPUT_DIM=768
HIDDEN_DIM=256
OUTPUT_DIM=768
MEMORY_DIM=128
LEARNING_RATE=0.001

# Context Cascade Engine
PORT=8002
MEMORY_CORE_URL=http://memory_core:8000
TRAINER_URL=http://trainer:8001
```

## Component Integration

### GeometryManager

The `GeometryManager` is a central utility class shared across components to ensure consistent handling of embeddings:

```python
from synthians_memory_core.geometry_manager import GeometryManager

# Create a shared instance with default configuration
geometry_manager = GeometryManager({
    'embedding_dim': 768,
    'geometry_type': 'euclidean',
    'alignment_strategy': 'truncate'
})

# Use for vector operations
normalized = geometry_manager.normalize_embedding(embedding)
similarity = geometry_manager.calculate_similarity(vec1, vec2)
aligned_a, aligned_b = geometry_manager.align_vectors(vec1, vec2)
```

### Vector Index Management

The `MemoryVectorIndex` handles storage and retrieval of embedding vectors using FAISS:

```python
from synthians_memory_core.vector_index import MemoryVectorIndex

# Initialize with configuration
index = MemoryVectorIndex({
    'embedding_dim': 768,
    'index_type': 'L2',
    'vector_index_path': './storage/vector_index',
    'use_gpu': False  # Set to True for GPU acceleration where available
})

# Add vectors
index.add_vector('memory_123', embedding)

# Search for similar vectors
results = index.search(query_embedding, k=10)

# Save and load
index.save_index()
index.load_index()
```

### Metadata Enrichment

The `MetadataSynthesizer` enriches memory metadata with various properties:

```python
from synthians_memory_core.metadata_synthesizer import MetadataSynthesizer

# Initialize the synthesizer
metadata_synthesizer = MetadataSynthesizer()

# Enrich a memory's metadata
enriched_metadata = metadata_synthesizer.synthesize_metadata(
    content="Sample memory content",
    embedding=embedding,
    existing_metadata={}
)

# The enriched metadata includes:
# - timestamp_iso, time_of_day, day_of_week
# - complexity_estimate, word_count
# - embedding_dim, embedding_norm
# - uuid (memory_id)
# - content_length
```

## Robust Error Handling

### Embedding Validation

All embeddings are validated to detect and handle invalid values:

```python
def _validate_embedding(embedding, allow_zero=True):
    """Validate that an embedding vector contains only valid values."""
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

### Dimension Mismatch Handling

The system automatically handles embeddings of different dimensions (e.g., 384 vs. 768):

```python
# In Memory Core API handlers
async def retrieve_memories(request_data):
    # Extract query embedding
    query_embedding = request_data.get('query_embedding')
    
    # The system will handle dimension mismatches automatically
    # If the query is 384D but the system uses 768D, alignment happens transparently
    memories = await memory_core.retrieve_memories_by_vector(
        query_embedding=query_embedding,
        limit=request_data.get('limit', 10),
        threshold=request_data.get('threshold', 0.3)  # Explicit threshold parameter
    )
    
    return memories
```

## Performance Optimization

### Memory Retrieval Enhancements

The system includes several optimizations for memory retrieval:

1. **Lower Default Threshold**: The default similarity threshold has been reduced from 0.5 to 0.3 for better recall sensitivity
2. **Client-Controlled Thresholds**: API endpoints accept an explicit `threshold` parameter for fine-tuning retrieval sensitivity
3. **Enhanced Logging**: The system provides detailed similarity score logging for debugging
4. **Two-Stage Retrieval**: First uses vector similarity search, then applies additional filters as needed

### Emotion Analysis Optimization

The system performs emotion analysis efficiently:

1. **Respects Provided Emotions**: If emotions are already provided in the input, no redundant analysis is performed
2. **On-Demand Processing**: Emotion analysis only runs when actually needed
3. **Caching**: Results are cached to avoid repeated analysis of the same content

## Deployment Example

Example Docker Compose configuration:

```yaml
services:
  memory_core:
    build:
      context: ./synthians_memory_core
    ports:
      - "8000:8000"
    volumes:
      - ./storage:/app/storage
    environment:
      - PORT=8000
      - VECTOR_DB_PATH=/app/storage/vectordb
      - MEMORY_STORE_PATH=/app/storage/memorystore
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - EMBEDDING_DIM=768
      - GEOMETRY_TYPE=euclidean
      - ALIGNMENT_STRATEGY=truncate
      - RETRIEVAL_THRESHOLD=0.3

  trainer:
    build:
      context: ./synthians_memory_core/synthians_trainer_server
    ports:
      - "8001:8001"
    environment:
      - PORT=8001
      - MEMORY_CORE_URL=http://memory_core:8000
      - INPUT_DIM=768
      - HIDDEN_DIM=256
      - OUTPUT_DIM=768
      - MEMORY_DIM=128

  orchestrator:
    build:
      context: ./synthians_memory_core/orchestrator
    ports:
      - "8002:8002"
    environment:
      - PORT=8002
      - MEMORY_CORE_URL=http://memory_core:8000
      - TRAINER_URL=http://trainer:8001
    depends_on:
      - memory_core
      - trainer
```

## GPU Acceleration Notes

1. **FAISS GPU Support**: The Memory Core can utilize GPU acceleration for vector similarity search
   * Set `USE_GPU=true` in the environment variables
   * Note the limitation with `IndexIDMap` operations: adding vectors with custom IDs doesn't benefit from GPU acceleration, though search operations still do

2. **Embedding Generation**: If using a local embedding model, GPU acceleration can provide significant performance benefits
   * Requires a CUDA-compatible GPU
   * Set `USE_GPU=true` for the embedding service
