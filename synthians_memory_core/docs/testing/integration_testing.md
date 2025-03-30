# Integration Testing Guide for Synthians Cognitive System

**Author:** Lucidia Core Team  
**Date:** 2025-03-30  
**Status:** Implemented

## Overview

Integration testing for the Synthians Cognitive Architecture focuses on verifying that the three main components (Memory Core, Neural Memory Server, and Context Cascade Engine) work together correctly to implement the complete cognitive cycle, including the surprise feedback loop and variant-specific behaviors.

## Components Under Test

1. **Memory Core (`synthians_memory_core`)**: Responsible for stable, indexed storage of memories and their embeddings.
2. **Neural Memory Server (`synthians_trainer_server`)**: Implements test-time learning and associative memory retrieval.
3. **Context Cascade Engine (`orchestrator`)**: Orchestrates the cognitive flow between Memory Core and Neural Memory.

## Key Integration Points

### Memory Core u2194 Neural Memory Server (via CCE)

- **Store u2192 Update u2192 Boost Flow**: Verify that memories stored in Memory Core trigger Neural Memory updates, which generate surprise metrics that correctly boost the original memory's QuickRecal score.
- **Embedding Validation Chain**: Verify that embedding validation (NaN/Inf checks) is consistently applied across service boundaries.
- **Dimension Alignment**: Confirm that embeddings of different dimensions (384D vs 768D) are correctly aligned when passing between services.

### Context Cascade Engine Orchestration

- **Cognitive Cycle Timing**: Verify the correct sequence and timing of the refactored cognitive flow.
- **Variant-Specific Logic**: Test that MAC, MAG, and MAL variants correctly implement their attention mechanisms.
- **History Management**: Confirm that sequence history is properly maintained for attention calculations.

## Test Environment Setup

```python
from synthians.testing import ServiceTestFixture, MockMemoryCore, MockNeuralMemory

def setup_integration_environment(variant="NONE", mock_services=False):
    """Set up an environment for integration testing."""
    if mock_services:
        # Use mocks for isolated testing
        memory_core = MockMemoryCore()
        neural_memory = MockNeuralMemory()
    else:
        # Use actual services
        memory_core = MemoryCoreClient("http://localhost:5010")
        neural_memory = NeuralMemoryClient("http://localhost:5011")
    
    # Set environment variable for Titans variant
    os.environ["TITANS_VARIANT"] = variant
    
    # Create CCE client
    cce = CCEClient(
        memory_core_url="http://localhost:5010",
        neural_memory_url="http://localhost:5011"
    )
    
    return memory_core, neural_memory, cce
```

## Test Scenarios

### Basic Cognitive Cycle

```python
@pytest.mark.integration
def test_basic_cognitive_cycle():
    # 1. Initialize test setup
    memory_core, neural_memory, cce = setup_integration_environment()
    
    # 2. Process a new memory through CCE
    content = "This is a test memory with specific content."
    response = cce.process_memory(content=content)
    memory_id = response.memory_id
    
    # 3. Verify memory was stored in Memory Core
    memory = memory_core.get_memory_by_id(memory_id)
    assert memory is not None
    assert memory.content == content
    
    # 4. Verify surprise metrics were returned
    assert "loss" in response.surprise_metrics
    assert "grad_norm" in response.surprise_metrics
    
    # 5. Verify QuickRecal boost was applied
    assert response.feedback_applied
    
    # 6. Verify retrieval works
    retrieved = memory_core.retrieve_memories(query=content, top_k=1)
    assert len(retrieved) > 0
    assert retrieved[0].id == memory_id
    
    # 7. Verify embedding validation worked
    embedding = memory.embedding
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()
```

### Surprise Feedback Loop

```python
@pytest.mark.integration
def test_surprise_feedback_loop():
    # Setup
    memory_core, _, cce = setup_integration_environment()
    
    # 1. Process a routine memory (low surprise expected)
    routine_content = "This is routine information similar to existing memories."
    routine_response = cce.process_memory(content=routine_content)
    routine_id = routine_response.memory_id
    routine_surprise = routine_response.surprise_metrics["loss"]
    routine_initial_qr = memory_core.get_memory_by_id(routine_id).quickrecal_score
    
    # 2. Process a surprising memory (high surprise expected)
    surprise_content = "This is completely unexpected and novel information with unusual patterns."
    surprise_response = cce.process_memory(content=surprise_content)
    surprise_id = surprise_response.memory_id
    surprise_surprise = surprise_response.surprise_metrics["loss"]
    surprise_initial_qr = memory_core.get_memory_by_id(surprise_id).quickrecal_score
    
    # 3. Process several more routine memories to establish baseline
    for i in range(5):
        cce.process_memory(content=f"Another routine memory {i}")
    
    # 4. Verify surprising memory got larger boost
    routine_memory = memory_core.get_memory_by_id(routine_id)
    surprise_memory = memory_core.get_memory_by_id(surprise_id)
    
    routine_boost = routine_memory.quickrecal_score - routine_initial_qr
    surprise_boost = surprise_memory.quickrecal_score - surprise_initial_qr
    
    assert surprise_boost > routine_boost
    assert surprise_surprise > routine_surprise
    
    # 5. Verify that surprising memory ranks higher in retrieval despite being older
    results = memory_core.retrieve_memories(query="test information", top_k=10)
    surprise_rank = next((i for i, m in enumerate(results) if m.id == surprise_id), None)
    routine_rank = next((i for i, m in enumerate(results) if m.id == routine_id), None)
    
    assert surprise_rank is not None
    assert routine_rank is not None
    assert surprise_rank < routine_rank  # Lower rank = higher position
```

### Embedding Dimension Handling

```python
@pytest.mark.integration
def test_embedding_dimension_handling():
    # Setup
    memory_core, neural_memory, cce = setup_integration_environment()
    
    # 1. Create embeddings of different dimensions
    embedding_384d = np.random.rand(384).astype(np.float32)  # Simulate 384-dimensional embedding
    embedding_768d = np.random.rand(768).astype(np.float32)  # Simulate 768-dimensional embedding
    
    # Normalize embeddings for realistic testing
    embedding_384d = embedding_384d / np.linalg.norm(embedding_384d)
    embedding_768d = embedding_768d / np.linalg.norm(embedding_768d)
    
    # 2. Process memories with these embeddings through CCE
    response_384d = cce.process_memory(
        content="384d test", 
        embedding=embedding_384d.tolist()
    )
    response_768d = cce.process_memory(
        content="768d test", 
        embedding=embedding_768d.tolist()
    )
    
    # 3. Verify both were processed without errors
    assert response_384d.status == "success"
    assert response_768d.status == "success"
    
    # 4. Verify Neural Memory received appropriate embeddings
    # This requires a method to check the projections used
    nm_history = neural_memory.get_processing_history()
    
    # 5. Verify retrieval works with mixed dimensions
    results_384d_query = memory_core.retrieve_memories(
        query_embedding=embedding_384d.tolist(),
        top_k=5
    )
    results_768d_query = memory_core.retrieve_memories(
        query_embedding=embedding_768d.tolist(),
        top_k=5
    )
    
    assert len(results_384d_query) > 0
    assert len(results_768d_query) > 0
    
    # 6. Verify that the 384d embedding retrieves the 384d memory and vice versa
    assert response_384d.memory_id in [m.id for m in results_384d_query]
    assert response_768d.memory_id in [m.id for m in results_768d_query]
```

### Variant-Specific Tests

#### MAC Variant Test

```python
@pytest.mark.integration
def test_mac_variant():
    # Setup with MAC variant enabled
    memory_core, neural_memory, cce = setup_integration_environment(variant="MAC")
    
    # 1. Process a sequence of related memories to build history
    base_content = "The quick brown fox jumps over the lazy dog."
    memories = []
    for i in range(5):
        modified = base_content.replace("fox", f"fox {i}")
        response = cce.process_memory(content=modified)
        memories.append(response.memory_id)
    
    # 2. Process a query memory that should trigger attention
    query_content = "A quick brown animal jumps over a lazy canine."
    query_response = cce.process_memory(
        content=query_content, 
        include_variant_metrics=True
    )
    
    # 3. Verify attention weights are distributed as expected
    assert "attention_weights" in query_response.variant_metrics
    weights = query_response.variant_metrics["attention_weights"]
    
    # Weights should sum to approximately 1.0
    assert abs(sum(weights) - 1.0) < 0.001
    
    # 4. Confirm attended output differs from raw output
    assert "raw_output" in query_response.variant_metrics
    assert "attended_output" in query_response.variant_metrics
    
    raw = np.array(query_response.variant_metrics["raw_output"])
    attended = np.array(query_response.variant_metrics["attended_output"])
    
    # Calculate cosine similarity between raw and attended outputs
    similarity = np.dot(raw, attended) / (np.linalg.norm(raw) * np.linalg.norm(attended))
    
    # Outputs should be similar but not identical
    assert 0.7 < similarity < 0.99
```

#### MAG Variant Test

```python
@pytest.mark.integration
def test_mag_variant():
    # Setup with MAG variant enabled
    memory_core, neural_memory, cce = setup_integration_environment(variant="MAG")
    
    # 1. Process a sequence of memories to build history
    for i in range(5):
        cce.process_memory(content=f"Memory {i} in the sequence.")
    
    # 2. Process a test memory with metrics collection
    response = cce.process_memory(
        content="Test memory for MAG variant.",
        include_variant_metrics=True
    )
    
    # 3. Verify external gate values are calculated
    assert "external_alpha_gate" in response.variant_metrics
    assert "external_theta_gate" in response.variant_metrics
    assert "external_eta_gate" in response.variant_metrics
    
    # 4. Verify gates are within valid ranges
    alpha = response.variant_metrics["external_alpha_gate"]
    theta = response.variant_metrics["external_theta_gate"]
    eta = response.variant_metrics["external_eta_gate"]
    
    assert 0 <= alpha <= 1
    assert theta > 0
    assert 0 <= eta <= 1
    
    # 5. Process a similar memory and check for lower alpha (less forgetting)
    similar_response = cce.process_memory(
        content="Very similar test memory for MAG variant.",
        include_variant_metrics=True
    )
    
    similar_alpha = similar_response.variant_metrics["external_alpha_gate"]
    assert similar_alpha < alpha  # Similar content should trigger less forgetting
```

#### MAL Variant Test

```python
@pytest.mark.integration
def test_mal_variant():
    # Setup with MAL variant enabled
    memory_core, neural_memory, cce = setup_integration_environment(variant="MAL")
    
    # 1. Process a sequence of memories to build history
    for i in range(5):
        cce.process_memory(content=f"MAL test memory {i}.")
    
    # 2. Process a test memory with metrics collection
    response = cce.process_memory(
        content="Final test memory for MAL variant.",
        include_variant_metrics=True
    )
    
    # 3. Verify value projection was modified
    assert "original_value_projection" in response.variant_metrics
    assert "modified_value_projection" in response.variant_metrics
    
    original_v = np.array(response.variant_metrics["original_value_projection"])
    modified_v = np.array(response.variant_metrics["modified_value_projection"])
    
    # 4. Verify the modification is meaningful but not extreme
    # Calculate cosine similarity between original and modified value projections
    similarity = np.dot(original_v, modified_v) / (np.linalg.norm(original_v) * np.linalg.norm(modified_v))
    
    # Should be similar but not identical
    assert 0.7 < similarity < 0.99
    
    # 5. Verify that the attention mechanism is working
    assert "attention_weights" in response.variant_metrics
    weights = response.variant_metrics["attention_weights"]
    assert abs(sum(weights) - 1.0) < 0.001  # Weights should sum to 1.0
```

## Test Fixtures

### Mock Services

For isolated testing, mock implementations of each service can be used:

```python
class MockMemoryCore:
    def __init__(self):
        self.memories = {}
        self.quickrecal_updates = []
    
    async def process_memory(self, content, embedding=None, metadata=None):
        memory_id = str(uuid.uuid4())
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "embedding": embedding or np.random.rand(384).tolist(),
            "metadata": metadata or {},
            "quickrecal_score": 0.5
        }
        return {
            "memory_id": memory_id,
            "status": "success"
        }
    
    async def update_quickrecal_score(self, memory_id, delta):
        if memory_id in self.memories:
            self.memories[memory_id]["quickrecal_score"] += delta
            self.quickrecal_updates.append((memory_id, delta))
            return {"status": "success"}
        return {"status": "error", "message": "Memory not found"}
    
    async def get_memory_by_id(self, memory_id):
        return self.memories.get(memory_id)
    
    async def retrieve_memories(self, query=None, query_embedding=None, top_k=10):
        # Simple mock implementation
        memories = list(self.memories.values())[:top_k]
        return memories
```

### Integration Test Fixture

A fixture that sets up all three services for integration testing:

```python
@pytest.fixture
async def integrated_services(variant="NONE"):
    # Start all three services with test configuration
    memory_core_proc = await start_memory_core_server(test_config)
    neural_memory_proc = await start_neural_memory_server(test_config)
    
    # Set environment variable for Titans variant
    os.environ["TITANS_VARIANT"] = variant
    
    cce_proc = await start_cce_server(test_config)
    
    # Wait for services to be ready
    await wait_for_service("http://localhost:5010/api/health")
    await wait_for_service("http://localhost:5011/api/health")
    await wait_for_service("http://localhost:5012/api/health")
    
    # Yield the clients
    yield {
        "memory_core": MemoryCoreClient("http://localhost:5010"),
        "neural_memory": NeuralMemoryClient("http://localhost:5011"),
        "cce": CCEClient("http://localhost:5012")
    }
    
    # Cleanup
    for proc in [cce_proc, neural_memory_proc, memory_core_proc]:
        proc.terminate()
        await proc.wait()
```

## Test Data

### Controlled Test Sequences

Predefined sequences of inputs with expected outputs for deterministic testing:

```python
test_sequences = [
    # Sequence 1: Routine information
    {
        "name": "routine_sequence",
        "inputs": [
            "The weather today is sunny with a high of 75 degrees.",
            "Traffic was normal on the highway this morning.",
            "The stock market closed with modest gains yesterday."
        ],
        "expected": {
            "avg_surprise": 0.2,  # Low surprise expected
            "max_quickrecal_boost": 0.1  # Small boosts expected
        }
    },
    # Sequence 2: Novel information
    {
        "name": "novel_sequence",
        "inputs": [
            "Scientists discovered a new particle that defies known physics.",
            "An earthquake of magnitude 9.5 struck in the middle of the desert.",
            "A previously unknown species of large mammals was found in the Amazon."
        ],
        "expected": {
            "avg_surprise": 0.6,  # High surprise expected
            "max_quickrecal_boost": 0.4  # Large boosts expected
        }
    }
]
```

## Continuous Integration

Integration tests should be run automatically as part of the CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Integration Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    - name: Start services
      run: |
        python -m synthians.scripts.start_services --test-mode
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
```

## Best Practices

1. **End-to-End Focus**: Integration tests should focus on end-to-end behavior, not implementation details.

2. **Isolation**: Each test should clean up after itself to prevent interference between tests.

3. **Fixtures Over Setup**: Use pytest fixtures to set up and tear down test environments consistently.

4. **Parameterization**: Use pytest's parameterize feature to test multiple configurations and variants.

5. **Logging**: Enable detailed logging during tests to make debugging easier:

```python
@pytest.fixture(autouse=True)
def enable_test_logging():
    # Set up logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield
    # Reset logging after test
```

6. **Timing Sensitivity**: Include timeouts and retries to handle network-related timing issues in distributed services.

7. **Variant Coverage**: Ensure tests cover all variants and their specific behaviors.
