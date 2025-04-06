#!/bin/bash
set -e

echo "===== FAISS GPU Setup Script ====="

# Uninstall any existing FAISS installations
echo "Removing any existing FAISS installations..."
pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true

# Install the correct GPU version for CUDA 12.1
echo "Installing FAISS GPU for CUDA 12.1..."
pip install faiss-gpu-cu12==1.8.0.2

# Pin transformers version to avoid dependency resolution conflicts
echo "Installing specific transformers version..."
pip uninstall -y transformers
pip install transformers==4.30.2

# Install other required dependencies
echo "Installing additional dependencies..."
pip install numpy==1.25.2 fastapi uvicorn pydantic httpx

# Fix huggingface_hub and sentence-transformers compatibility issue
echo "Fixing huggingface_hub and sentence-transformers compatibility..."
pip uninstall -y huggingface_hub sentence-transformers || true
pip install huggingface_hub==0.12.0 sentence-transformers==2.2.2

# Run the docker-numpy-fix.sh script if it exists
if [ -f "/app/docker-numpy-fix.sh" ]; then
    echo "Running numpy fix script..."
    bash /app/docker-numpy-fix.sh
fi

# Verify installation
echo "Verifying FAISS installation..."
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU support available: {faiss.get_num_gpus() > 0}')"

echo "===== FAISS Setup Complete ====="

# Execute the command passed to the script
echo "Starting application..."
exec "$@"
