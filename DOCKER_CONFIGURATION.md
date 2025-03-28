# Synthians Docker Configuration

## Overview

This document describes the Docker container setup for the Synthians project. We maintain multiple services to handle different aspects of the system while ensuring compatibility and stability.

## Service Architecture

### Main Services

1. **synthians_core** (Port 5010)
   - This is the primary Memory Core service used by the main application
   - Runs on the `nemo_qr_v1:latest` image
   - Has established compatibility with the orchestrator

2. **memory-core** (Port 5020)
   - Secondary/standby Memory Core service with NumPy compatibility fixes
   - Runs on the same `nemo_qr_v1:latest` image with custom initialization
   - Uses the `docker-numpy-fix.sh` script to downgrade NumPy to version 1.26.4
   - Available for testing and as a fallback

3. **trainer-server** (Port 8001)
   - Sequence prediction service that integrates with Memory Core
   - Runs on the `nemo_qr_v1:latest` image with NumPy compatibility fixes
   - Includes TensorFlow and other ML libraries for sequence training and prediction
   - Connects to memory-core service (port 5020) for integration

## NumPy Compatibility Fix

We encountered binary compatibility issues between NumPy 2.x and some dependencies. The fix involves:

1. Installing all required dependencies from requirements.txt
2. Installing FastAPI and TensorFlow explicitly
3. Downgrading NumPy to version 1.26.4 using `--force-reinstall`

This fix is implemented in the `docker-numpy-fix.sh` script, which is used as the entrypoint for the memory-core and trainer-server services.

## Service Dependencies

- The trainer-server depends on memory-core (port 5020) for embedding operations
- Both memory-core and trainer-server use the NumPy compatibility fix
- The orchestrator currently uses the main synthians_core service (port 5010)

## Environment Variables

### memory-core
```
MEMORY_STORAGE_PATH=/app/memory/stored/memory-core
EMBEDDING_DIM=768
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=5020
```

### trainer-server
```
MEMORY_CORE_URL=http://memory-core:5020
PORT=8001
```

## Maintenance Notes

1. **Service Redundancy**: Currently, synthians_core and memory-core provide similar functionality but at different ports. This allows for testing the NumPy fix without disrupting the main service.

2. **Future Improvements**:
   - Consider building a custom Docker image with the correct dependencies pre-installed
   - Standardize on a single memory-core service once compatibility is fully verified
   - Update orchestrator configuration to use the preferred memory core service

3. **TensorFlow Integration**: The trainer-server includes TensorFlow support for sequence prediction, which requires specific NumPy version compatibility.
