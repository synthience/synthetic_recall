#!/bin/bash
set -e

echo "Starting orchestrator entrypoint script..."

# Uninstall any existing Python packages that might cause conflicts
echo "Removing potentially conflicting packages..."
pip uninstall -y numpy torch torchvision faiss-cpu faiss-gpu transformers huggingface_hub sentence-transformers nltk scikit-learn pandas tensorflow

# First install the correct NumPy version and pin it to prevent upgrades
echo "Installing NumPy 1.25.2 specifically..."
pip install numpy==1.25.2 --no-deps

# Create pip.conf to prevent numpy from being upgraded
echo "Creating pip.conf to prevent NumPy upgrades..."
mkdir -p /root/.config/pip
echo "[global]" > /root/.config/pip/pip.conf
echo "no-build-isolation = false" >> /root/.config/pip/pip.conf
# Add environment variable to prevent upgrades
export PIP_EXISTS_ACTION=i  # ignore if already installed

# Install base CPU FAISS version first (more compatible)
echo "Installing FAISS CPU version..."
pip install faiss-cpu==1.7.4 --no-deps

# Install specific PyTorch version
echo "Installing PyTorch 1.13.1..."
pip install torch==1.13.1 torchvision==0.14.1 --no-deps

# Install huggingface_hub, transformers, and sentence-transformers
echo "Installing compatible versions of huggingface_hub, transformers, and sentence-transformers..."
pip install --no-deps huggingface_hub==0.12.0
pip install --no-deps transformers==4.26.1
pip install --no-deps sentence-transformers==2.2.2

# Install other core dependencies without allowing them to upgrade NumPy
echo "Installing Python dependencies without upgrading NumPy..."
pip install --no-deps fastapi uvicorn==0.22.0 aiohttp pydantic==1.10.8
pip install --upgrade typing-extensions

# Verify installations
echo "Verifying NumPy version:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); print(f'NumPy location: {numpy.__file__}')"

echo "Verifying FAISS installation:"
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU support available: {faiss.get_num_gpus() > 0}')" || echo "FAISS verification failed, but continuing..."

echo "Verifying transformers and sentence-transformers installation:"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')" || echo "transformers verification failed, but continuing..."
python -c "import sentence_transformers; print(f'sentence_transformers version: {sentence_transformers.__version__}')" || echo "sentence_transformers verification failed, but continuing..."

# Test dependencies without importing them
echo "Checking if critical modules exist:"
python -c "import os; print('FAISS module exists: ' + str(os.path.exists('/usr/local/lib/python3.10/dist-packages/faiss')))" || echo "Failed to check FAISS module"
python -c "import os; print('numpy module exists: ' + str(os.path.exists('/usr/local/lib/python3.10/dist-packages/numpy')))" || echo "Failed to check numpy module"

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
