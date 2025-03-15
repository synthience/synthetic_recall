#!/bin/bash

# Setup script for STT service with NeMo and Canary-1B model
set -e

echo "Starting STT setup..."

# Install dependencies
echo "Installing dependencies..."
pip uninstall -y numpy huggingface_hub
pip install numpy==1.24.3
pip install websockets sentence-transformers scikit-learn==1.3.2 fastapi uvicorn

# Install specific huggingface-hub version that works with NeMo
echo "Installing compatible huggingface-hub version..."
pip install huggingface-hub==0.20.3

# Install NeMo from specific version
echo "Installing NeMo toolkit..."
pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]

# Apply the fix for ModelFilter in NeMo
echo "Applying fix for ModelFilter in NeMo..."
python /workspace/project/fix_nemo.py

# Create models directory if it doesn't exist
mkdir -p /workspace/models

# Instead of downloading the model, we'll use it directly from pretrained source
echo "Setting up to use the model directly from pretrained source"

echo "STT setup completed successfully!"

# Start the STT server
echo "Starting STT server..."
exec python /workspace/project/server/STT_server.py
