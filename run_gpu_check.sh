#!/bin/bash

echo "===== Running FAISS GPU Check in Docker ====="

# Run the GPU check in the Docker container
docker exec -it synthians_core bash -c 'cd /workspace/project && python faiss_gpu_check.py'

echo "===== Check completed ====="
