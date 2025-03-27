#!/bin/bash

# Script to run FAISS tests inside the Docker container

echo "===== Running FAISS Implementation Tests in Docker ====="

# Create a test directory for mounting if it doesn't exist
mkdir -p ./test_results

# Run the test command in the Docker container
docker exec -it synthians_core bash -c 'cd /workspace/project && python -m synthians_memory_core.tests.test_vector_index'

echo "===== Test execution completed ====="
