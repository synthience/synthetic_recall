#!/bin/bash

echo "===== Running Memory System Validation in Docker ====="

# Run the validation script in the Docker container
docker exec -it synthians_core bash -c 'cd /workspace/project && python validate_memory_system.py'

echo "===== Validation completed ====="
