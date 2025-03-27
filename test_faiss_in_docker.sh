#!/bin/bash

echo "===== Running FAISS Integration Tests in Docker ====="

# Run the test command in the Docker container
docker exec -it synthians_core bash -c 'cd /workspace/project && python -m synthians_memory_core.test_faiss_integration'

echo "===== Test execution completed ====="
