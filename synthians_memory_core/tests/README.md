# Synthians Memory Core Test Suite

This comprehensive test suite is designed to validate the functionality, performance, and reliability of the Synthians Memory Core system. The tests are organized into modular, progressive phases to ensure full coverage of all components while allowing for targeted testing of specific subsystems.

## 🧪 Test Structure

The tests are organized into seven progressive phases, each focusing on different aspects of the system:

### 🔹 Phase 1: Core Infrastructure Validation
- `test_api_health.py` - Basic API endpoints, health, and stats tests

### 🔹 Phase 2: Memory Lifecycle Test
- `test_memory_lifecycle.py` - End-to-end memory creation, retrieval, feedback, deletion

### 🔹 Phase 3: Emotional & Cognitive Layer Test
- `test_emotion_and_cognitive.py` - Tests for emotion analysis, metadata enrichment, and cognitive load scoring

### 🔹 Phase 4: Transcription & Voice Pipeline Test
- `test_transcription_voice_flow.py` - Tests for speech transcription, interruption handling, and voice state management

### 🔹 Phase 5: Retrieval Dynamics Test
- `test_retrieval_dynamics.py` - Tests for memory retrieval with various conditions, thresholds, and filters

### 🔹 Phase 6: Tooling Integration Test
- `test_tool_integration.py` - Tests for tool interfaces that call core functions

### 🔹 Phase 7: Stress + Load Test
- `test_stress_load.py` - High-volume and performance tests 

## 📋 Prerequisites

```bash
pip install pytest pytest-asyncio pytest-html aiohttp
```

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py

# Run with more detailed output
python tests/run_tests.py --verbose

# Run smoke tests only
python tests/run_tests.py --markers="smoke"

# Run a specific test module
python tests/run_tests.py --module="test_api_health.py"

# Run a specific test function
python tests/run_tests.py --test="test_health_and_stats"

# Generate HTML and XML reports
python tests/run_tests.py --report

# Run tests in parallel
python tests/run_tests.py --parallel=4

# Test against a different server
python tests/run_tests.py --url="http://test-server:5010"
```

### Using pytest directly

```bash
# Run all tests
pytest -xvs --asyncio-mode=auto

# Run a specific test module
pytest -xvs test_api_health.py --asyncio-mode=auto

# Run tests with a specific marker
pytest -xvs -m smoke --asyncio-mode=auto
```

## 🏷️ Test Markers

Tests are categorized with the following markers:

- `smoke`: Basic functionality tests that should always pass
- `integration`: Tests that verify integration between components
- `slow`: Tests that take longer to run (e.g., stress tests)
- `emotion`: Tests focused on emotion analysis
- `retrieval`: Tests focused on memory retrieval
- `stress`: High-volume load tests

## 📊 Test Reports

When using the `--report` option, the test suite generates:

- HTML reports in `test_reports/report_TIMESTAMP.html`
- XML reports in `test_reports/report_TIMESTAMP.xml` (JUnit format for CI systems)

## 🔧 Configuration

The test suite can be configured using environment variables:

- `SYNTHIANS_TEST_URL`: URL of the test server (default: http://localhost:5010)

## ⚠️ Implementation Notes

1. Tests use a temporary directory for test data by default
2. Some tests expect specific API functionality which may not be implemented yet
3. Stress tests have reduced volumes by default to run faster - adjust constants in code for full stress testing
4. Pay attention to potential race conditions with concurrent tests
5. Some tests may fail if specific components (e.g., emotion analyzer) are not properly initialized

## 🛠 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Dimension mismatch warnings in logs | Expected during testing with different embedding dimensions |
| Empty embeddings | Check if the embedding model is properly loaded |
| HTTP connection errors | Ensure the server is running and accessible at the configured URL |
| File permission errors | Check that the test directory has proper write permissions |
| Test timeouts | Adjust timeout settings or reduce batch sizes in stress tests |

## 🔄 Continuous Integration

This test suite is designed to be integrated with CI/CD pipelines. XML reports in JUnit format can be consumed by most CI systems.
