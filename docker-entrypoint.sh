#!/bin/bash
set -e

# Setup NVIDIA runtime environment
source /opt/nvidia/nvidia_entrypoint.sh

# Initialize CUDA if available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "Initializing CUDA environment..."
    nvidia-smi
    
    # Set GPU memory fraction
    export CUDA_VISIBLE_DEVICES=all
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
fi

# Check if models directory is mounted
if [ ! -d "/workspace/models" ]; then
    echo "Warning: Models directory not mounted at /workspace/models"
fi

# Check if project directory is mounted
if [ ! -d "/workspace/project" ]; then
    echo "Error: Project directory not mounted at /workspace/project"
    exit 1
fi

# Initialize ephemeral memory system
echo "Initializing ephemeral memory system..."
mkdir -p /workspace/project/memory_store/ephemeral
chmod 777 /workspace/project/memory_store/ephemeral

# Execute command
exec "$@"