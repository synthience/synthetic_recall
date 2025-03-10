# Testing Strategy

This document outlines the comprehensive testing strategy for the Lucidia system, covering unit tests, integration tests, system tests, and specialized dream API tests.

## Testing Framework

### Unit Testing

Individual components should have comprehensive unit tests to verify their functionality in isolation:

```bash
# Run all unit tests
pytest tests/unit/

# Run specific component tests
pytest tests/unit/test_self_model.py
```

Key unit test modules:
- `test_self_model.py`: Tests for self-awareness and identity
- `test_world_model.py`: Tests for world representation
- `test_knowledge_graph.py`: Tests for semantic network operations
- `test_memory_system.py`: Tests for memory storage and retrieval
- `test_parameter_manager.py`: Tests for parameter system

### Integration Testing

Integration tests verify that components work together correctly:

```bash
# Run all integration tests
pytest tests/integration/

# Run specific integration tests
pytest tests/integration/test_memory_integration.py
```

Key integration test modules:
- `test_memory_integration.py`: Tests interaction between memory system and knowledge graph
- `test_model_switching.py`: Tests dynamic model selection based on system state
- `test_tensor_integration.py`: Tests integration with tensor server
- `test_hpc_integration.py`: Tests integration with HPC server
- `test_lm_studio_integration.py`: Tests integration with LM Studio

### System Testing

End-to-end test scenarios that verify complete system functionality:

```bash
# Run all system tests
pytest tests/system/

# Run specific system tests
pytest tests/system/test_dreaming.py
```

Key system test modules:
- `test_dreaming.py`: Tests complete dreaming workflow
- `test_resource_management.py`: Tests resource monitoring and optimization
- `test_state_transitions.py`: Tests state management and transitions
- `test_api_endpoints.py`: Tests API functionality
- `test_fault_tolerance.py`: Tests system recovery from failures

## Dream API Testing

Test the Dream API endpoints using the following script:

```bash
# Test batch embedding processing
curl -X POST http://localhost:8081/api/dream/test/batch_embedding \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is a test", "Another test"], "use_hypersphere": false}'

# Test similarity search
curl -X POST http://localhost:8081/api/dream/test/similarity_search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 3}'

# Test dream report creation
curl -X POST http://localhost:8081/api/dream/test/create_test_report \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Report", "fragments": [{"content": "Test insight", "type": "insight", "confidence": 0.8}]}'

# Test report refinement
curl -X POST http://localhost:8081/api/dream/test/refine_report \
  -H "Content-Type: application/json" \
  -d '{"report_id": "report:12345"}'

# Test health check
curl http://localhost:8081/api/dream/health
```

## Test Scripts

Lucidia includes several test scripts to validate functionality:

| Script | Purpose | Status |
|--------|---------|--------|
| `test_dream_api.py` | Tests basic Dream API connectivity and endpoints | ✅ Complete |
| `test_dream_reflection.py` | Tests end-to-end dreaming flow with LM Studio integration | ✅ Complete |
| `docker_test_dream_api.py` | Tests Dream API inside Docker containers | ✅ Complete |
| `test_memory_integration.py` | Tests memory system components | ⚠️ In Progress |
| `test_tensor_connectivity.py` | Tests tensor server connections | ✅ Complete |

## Dream Flow Testing

The `test_dream_reflection.py` script validates the complete dreaming flow with the following components:

1. **LM Studio Connection**: Verifies connectivity to LM Studio and model availability
2. **Dream API Connection**: Confirms Dream API server is operational
3. **Memory Processing**: Adds test memories for dream generation
4. **Dream Generation**: Uses LM Studio to process memories and generate dreams
5. **Report Creation**: Creates structured reports from dream content
6. **Report Refinement**: Tests the refinement process to improve report quality
7. **Fragment Categorization**: Validates correct categorization of fragments as insights, questions, hypotheses, or counterfactuals

This script provides colorized output for better readability and includes comprehensive error handling for API timeouts and connection issues.

## Testing Infrastructure

### Automated Testing

Automated testing is implemented through GitHub Actions workflows:

```yaml
name: Lucidia Tests

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.dev.txt
    - name: Run unit tests
      run: |
        pytest tests/unit/
    - name: Run integration tests
      run: |
        pytest tests/integration/
```

### Test Coverage

Code coverage is tracked using pytest-cov:

```bash
# Generate coverage report
pytest --cov=lucidia tests/

# Generate HTML coverage report
pytest --cov=lucidia --cov-report=html tests/
```

## Test Environment

### Docker Test Environment

The Docker-based test environment ensures consistent testing across different systems:

```bash
# Build test environment
docker build -f Dockerfile.test -t lucidia-test .

# Run tests in container
docker run lucidia-test pytest
```

### Mock Services

For testing without external dependencies, mock services are provided:

- `MockTensorServer`: Simulates tensor server responses
- `MockHPCServer`: Simulates HPC server responses
- `MockLMStudioServer`: Simulates LM Studio responses

Example usage:

```python
from tests.mocks.mock_tensor_server import MockTensorServer

def test_embedding_generation(mocker):
    # Set up mock
    mock_server = MockTensorServer()
    mocker.patch('lucidia.clients.tensor_client.TensorClient._connect', 
                 return_value=mock_server)
    
    # Test with mock
    client = TensorClient("ws://mock-server:5001")
    embedding = client.generate_embedding("Test text")
    
    # Assert result from mock
    assert len(embedding) == 768
```

## Performance Testing

Performance tests measure system responsiveness and resource usage:

```bash
# Run performance tests
pytest tests/performance/

# Test under load
pytest tests/performance/test_load.py
```

Key performance metrics tracked:
- Response time for API endpoints
- Memory usage during different operations
- CPU utilization across components
- Model loading and switching times
- Dream processing throughput