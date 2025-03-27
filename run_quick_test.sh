#!/bin/bash

echo "===== Running Quick FAISS Memory Test in Docker ====="

# Run the quick test within the Docker container's network
docker exec -it synthians_core bash -c 'cd /workspace/project && python -c "import sys, asyncio; from quick_test import test_memory_system; sys.exit(0 if asyncio.run(test_memory_system()) else 1)"'

if [ $? -eq 0 ]; then
  echo "\n✓ FAISS Memory System Test: PASSED"
  echo "===== Test successfully completed ====="
  exit 0
else
  echo "\n✗ FAISS Memory System Test: FAILED"
  echo "===== Test failed ====="
  exit 1
fi
