#!/bin/bash
set -e

echo "===== Memory System Test with FAISS GPU ====="

# First, let's check if the API server is running
echo "Checking if Memory API server is running..."
CONTAINER_ID=$(docker ps --filter "name=synthians_core" --format "{{.ID}}")

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Container synthians_core is not running. Start it with 'docker compose up -d'."
    exit 1
fi

# Check if the memory API server is actually running and listening on port 5010
echo "Checking API server status..."
API_LISTENING=$(docker exec $CONTAINER_ID netstat -tuln | grep 5010 || echo "")

if [ -z "$API_LISTENING" ]; then
    echo "⚠️ Memory API server doesn't appear to be listening on port 5010."
    echo "Starting the API server..."
    docker exec -d $CONTAINER_ID python -m synthians_memory_core.api.server &
    echo "Waiting for API server to start..."
    sleep 5
else
    echo "✅ Memory API server is running."
fi

# For the test itself, we need it to use the right host
echo "Updating test script to use correct host..."
docker exec $CONTAINER_ID sed -i 's/base_url = "http:\/\/127.0.0.1:5010"/base_url = "http:\/\/localhost:5010"/' /workspace/project/docker_test.py

# Run the vector index tests
echo "\n===== Running Vector Index Tests ====="
docker exec $CONTAINER_ID python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU support available: {faiss.get_num_gpus() > 0}')"

# Run the test script
echo "\n===== Running Memory Tests ====="
docker exec $CONTAINER_ID python /workspace/project/docker_test.py

echo "\n===== Tests Complete ====="
