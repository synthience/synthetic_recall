#!/bin/bash
set -e

# Setup NVIDIA runtime environment
source /opt/nvidia/nvidia_entrypoint.sh

# Initialize CUDA if available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "Initializing CUDA environment..."
    nvidia-smi
    
    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Set GPU memory fraction
    export CUDA_VISIBLE_DEVICES=all
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Fix PyTorch and TorchVision compatibility issues by reinstalling
    echo "Ensuring PyTorch/TorchVision/CUDA compatibility..."
    pip uninstall -y torch torchvision torchaudio
    
    # Install PyTorch with matching CUDA version
    if [ "$CUDA_VERSION" -ge "12" ]; then
        echo "Installing PyTorch with CUDA 12.x support..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_VERSION" -ge "11" ]; then
        echo "Installing PyTorch with CUDA 11.8 support..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch with CUDA 11.7 support (fallback)..."
        pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
    fi
    
    # Ensure proper CUDA library pathing
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    
    # Install FAISS with GPU support after PyTorch is properly configured
    echo "Installing FAISS with GPU support..."
    pip install --no-cache-dir faiss-gpu
    echo "FAISS with GPU support installed successfully"
else
    # Install CPU-only FAISS
    echo "No GPU detected, installing CPU-only FAISS..."
    pip install --no-cache-dir faiss-cpu
    echo "FAISS CPU version installed successfully"
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

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Run the GPU diagnostic to check our setup
echo "Running GPU diagnostic..."
python /workspace/project/gpu_diagnostic.py

# Print torch configuration to verify compatibility
echo "Verifying PyTorch/CUDA configuration:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Start the requested command
exec "$@"