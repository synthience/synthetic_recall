# Titans Variants Integration Tests

## Overview

This directory contains integration tests for the Titans variants (MAC, MAG, MAL) in the Lucidia cognitive system. Unlike the component tests, these integration tests validate the variants' behavior by interacting with the live Context Cascade Engine (CCE) API.

## Prerequisites

- Docker services must be running with:
  - Memory Core service
  - Neural Memory service (with TensorFlow)
  - Context Cascade Engine service

- Required Python packages:
  - pytest
  - pytest-asyncio
  - aiohttp
  - numpy

## Running the Tests

### Using the Convenience Script

The easiest way to run the tests is using the convenience script:

```powershell
python run_titans_integration_tests.py [--variant VARIANT] [--verbose]
```

Where `VARIANT` is one of:
- `none` - Test baseline (no attention) variant
- `mac` - Test Memory-Attended Computation variant
- `mag` - Test Memory-Attended Gates variant
- `mal` - Test Memory-Augmented Learning variant
- `all` - Test all variants (default)

### Using pytest Directly

Alternatively, you can run the tests using pytest:

```powershell
python -m pytest tests/integration/test_titans_integration.py -v
```

## Test Structure

The integration tests validate each variant with the following approach:

1. **Controlled Input Sequence**: Generate a sequence of test inputs with predictable patterns
2. **API Interaction**: Send inputs to the CCE API with the variant configured
3. **Validation**: Verify variant-specific behaviors and outputs

### What Each Test Validates

#### Base Variant (NONE)
- Basic functionality without attention mechanisms
- Used as a control/baseline for comparison

#### MAC (Memory-Attended Computation)
- Verifies that `attended_embedding` is produced
- Checks that the attended embedding differs from but relates to the raw retrieved embedding
- Validates that attention over historical outputs affects retrieval results

#### MAG (Memory-Attended Gates)
- Verifies that gate values (`alpha`, `theta`, `eta`) are present in the response
- Checks that gate values influence the learning process
- Validates that historical patterns affect gate dynamics

#### MAL (Memory-Augmented Learning)
- Verifies that `v_prime_t` is produced
- Checks that the modified value projection differs from the original
- Validates that attention over historical values affects what gets stored

## Configuration

The tests use the following environment variables (with defaults):

- `CCE_URL`: URL of the Context Cascade Engine (default: `http://localhost:8002`)
- `MEMORY_CORE_URL`: URL of the Memory Core service (default: `http://localhost:8000`)
- `NEURAL_MEMORY_URL`: URL of the Neural Memory service (default: `http://localhost:8001`)

## Logs

Detailed logs are written to `titans_integration_tests.log`, including metrics and cross-variant comparisons.
