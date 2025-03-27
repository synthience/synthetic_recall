#!/bin/bash
# run_api_tests.sh - Run API-based memory system tests inside Docker container

echo "🧪 Running API-based memory system tests inside Docker container..."

# Create the test directory in the container if needed
docker exec synthians_core mkdir -p /app/tests

# Copy the test script to the container
docker cp tests/test_memory_retrieval_api.py synthians_core:/app/tests/

# Make the script executable
docker exec synthians_core chmod +x /app/tests/test_memory_retrieval_api.py

# Run the API tests inside the container
docker exec -it synthians_core python /app/tests/test_memory_retrieval_api.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo "✅ API tests completed successfully!"
else
    echo "❌ API tests failed. See logs above for details."
    exit 1
fi
