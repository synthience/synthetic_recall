#!/bin/bash

# Script to run comprehensive tests inside the Docker container

echo "===== Running Comprehensive Lucidia System Tests in Docker ====="

# Create a test directory for mounting if it doesn't exist
mkdir -p ./test_results

# Check if the --fast flag is provided
FAST_MODE=""
if [ "$1" == "--fast" ]; then
    FAST_MODE="--fast"
    echo "Running in FAST mode (reduced test set)"
fi

# Run the FAISS vector index test
echo "\n1. Testing FAISS Vector Index (dimension alignment and NaN/Inf handling)..."
docker exec -it synthians_memory_core_gpu bash -c 'cd /workspace/project && python docker_test.py'
VECTOR_TEST_RESULT=$?

# Run the Memory Core API test
echo "\n2. Testing Memory Core API and health..."
docker exec -it synthians_memory_core_gpu bash -c 'cd /workspace/project && python -m synthians_memory_core.tests.test_api_health'
API_TEST_RESULT=$?

# Run the trainer server API test
echo "\n3. Testing Neural Memory trainer API..."
docker exec -it trainer-server bash -c 'cd /app && python -m synthians_memory_core.synthians_trainer_server.tests.test_http_server'
TRAINER_TEST_RESULT=$?

# Run orchestrator test if not in fast mode
ORCHESTRATOR_TEST_RESULT=0
if [ "$FAST_MODE" != "--fast" ]; then
    echo "\n4. Testing Context Cascade Orchestrator..."
    docker exec -it context-cascade-orchestrator bash -c 'cd /app && python -m synthians_memory_core.orchestrator.tests.test_context_cascade_engine'
    ORCHESTRATOR_TEST_RESULT=$?
fi

# Check results
if [ $VECTOR_TEST_RESULT -eq 0 ] && [ $API_TEST_RESULT -eq 0 ] && [ $TRAINER_TEST_RESULT -eq 0 ] && [ $ORCHESTRATOR_TEST_RESULT -eq 0 ]; then
    echo "\n===== All tests PASSED! ====="
    exit 0
else
    echo "\n===== Some tests FAILED! ====="
    echo "FAISS Vector Index Test: $([ $VECTOR_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo "Memory Core API Test: $([ $API_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    echo "Trainer API Test: $([ $TRAINER_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    if [ "$FAST_MODE" != "--fast" ]; then
        echo "Orchestrator Test: $([ $ORCHESTRATOR_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
    fi
    exit 1
fi
