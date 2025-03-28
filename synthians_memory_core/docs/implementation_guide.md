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

# Trainer Server
PORT=8001
MEMORY_CORE_URL=http://memory_core:8000
INPUT_DIM=384
HIDDEN_DIM=256
OUTPUT_DIM=384
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

# Create a shared instance
geometry_manager = GeometryManager()

# Use for vector operations
normalized = geometry_manager.normalize_embedding(embedding)
similarity = geometry_manager.calculate_similarity(vec1, vec2)
aligned_vecs = geometry_manager.align_vectors_for_comparison(vec1, vec2)
```

### SurpriseDetector

The `SurpriseDetector` quantifies deviation between predicted and actual embeddings:

```python
from synthians_memory_core.synthians_trainer_server.surprise_detector import SurpriseDetector

# Initialize with GeometryManager
surprise_detector = SurpriseDetector(geometry_manager=geometry_manager)

# Calculate surprise metrics
metrics = surprise_detector.calculate_surprise(
    predicted_embedding=predicted_vec,
    actual_embedding=actual_vec
)

# Calculate quickrecal boost based on surprise
boost = surprise_detector.calculate_quickrecal_boost(metrics)
```

### Context Cascade Engine

The orchestrator coordinates the bidirectional flow between Memory Core and Trainer:

```python
from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine

# Initialize the engine
engine = ContextCascadeEngine(
    memory_core_url="http://localhost:8000",
    trainer_url="http://localhost:8001"
)

# Process a new memory through the cognitive pipeline
result = await engine.process_new_memory(
    content="Memory content text",
    metadata={"user": "user_id", "topic": "conversation"}
)
```

## Handling Embedding Dimensions

One of the key challenges in implementing this architecture is handling dimensional mismatches between embeddings. The system supports two common dimensions:

- **384-dimensional embeddings**: From models like `all-MiniLM-L6-v2`
- **768-dimensional embeddings**: From models like `all-mpnet-base-v2`

The `GeometryManager` handles these mismatches through alignment strategies:

```python
def _align_vectors_for_comparison(self, vec1, vec2):
    """Align vectors for comparison when dimensions don't match.
    
    Strategies:
    1. If dimensions match, return as is
    2. If one is smaller, pad with zeros
    3. If both differ from target dim, truncate or pad as needed
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # If dimensions already match, return as-is
    if vec1.shape == vec2.shape:
        return vec1, vec2
    
    # If one dimension is smaller, pad with zeros
    if vec1.shape[0] < vec2.shape[0]:
        # Pad vec1 to match vec2
        padded = np.zeros(vec2.shape)
        padded[:vec1.shape[0]] = vec1
        return padded, vec2
    elif vec1.shape[0] > vec2.shape[0]:
        # Pad vec2 to match vec1
        padded = np.zeros(vec1.shape)
        padded[:vec2.shape[0]] = vec2
        return vec1, padded
```

## Error Handling & Robustness

The architecture implements comprehensive error handling:

1. **Embedding Validation**: All embeddings are validated for NaN/Inf values

```python
def _validate_embedding(self, embedding):
    """Validate embedding for NaN or Inf values."""
    try:
        embedding_array = np.array(embedding, dtype=np.float32)
        if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
            return False
        return True
    except Exception:
        return False
```

2. **Network Timeouts**: All inter-service communications have timeout handling

```python
try:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=10.0) as response:
            # Process response
except asyncio.TimeoutError:
    logger.error(f"Timeout while connecting to service at {url}")
    return {"error": "Connection timed out"}
```

3. **State Management**: The Trainer's memory state is preserved for continuity

```python
# Store memory state for next prediction
if "memory_state" in result:
    self.current_memory_state = result["memory_state"]

# Include previous memory state in next request
if self.current_memory_state is not None:
    payload["previous_memory_state"] = self.current_memory_state
```

## Extending the Architecture

### Adding New Memory Storage Backends

To implement a new storage backend:

1. Create a class that implements the `MemoryPersistence` interface
2. Register it in the Memory Core's dependency injection system

### Implementing Custom Prediction Models

To create a new prediction model:

1. Extend the `SequenceTrainer` base class
2. Implement the `predict_next` and `update_memory_state` methods
3. Register it in the Trainer Server

### Customizing Surprise Detection

To modify surprise detection logic:

1. Extend or modify the `SurpriseDetector` class
2. Customize the `calculate_surprise` method to use different metrics
3. Update the `calculate_quickrecal_boost` formula as needed

## Testing

The architecture includes several test suites:

```bash
# Run Memory Core tests
python -m pytest synthians_memory_core/tests

# Run Trainer Server tests
python -m pytest synthians_memory_core/synthians_trainer_server/tests

# Run Orchestrator tests
python -m pytest synthians_memory_core/orchestrator/tests
```

Use the included Docker Compose test configuration for integration testing:

```bash
docker-compose -f docker-compose-test.yml up --abort-on-container-exit
```
