#!/bin/bash
set -e

echo "Starting orchestrator entrypoint script..."

# First install the correct NumPy version
echo "Installing NumPy 1.25.2 specifically..."
pip uninstall -y numpy
pip install numpy==1.25.2

# Install other dependencies
echo "Installing Python dependencies..."
pip install fastapi uvicorn aiohttp pydantic

# Install TensorFlow specifically
echo "Installing TensorFlow..."
pip install tensorflow

# Print NumPy version for debugging
echo "Checking NumPy version:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); print(f'NumPy location: {numpy.__file__}')"

# Set the dev mode flag
export CCE_DEV_MODE=true

# Test host.docker.internal resolution
echo "Testing host.docker.internal resolution:"
ping -c 1 host.docker.internal || echo "host.docker.internal resolution failed"

# Log environment for debugging
echo "Environment configuration:"
echo "MEMORY_CORE_URL: $MEMORY_CORE_URL"
echo "NEURAL_MEMORY_URL: $NEURAL_MEMORY_URL"
echo "CCE_DEV_MODE: $CCE_DEV_MODE"
echo "TITANS_VARIANT: $TITANS_VARIANT"
echo "LLM_STUDIO_ENDPOINT: $LLM_STUDIO_ENDPOINT"

# Start the orchestrator service
echo "Starting orchestrator service..."
exec python -m uvicorn synthians_memory_core.orchestrator.server:app --host 0.0.0.0 --port 8002
