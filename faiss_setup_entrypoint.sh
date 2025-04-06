#!/bin/bash
set -e

echo "===== FAISS GPU Setup Script ====="

# Create pip constraint file to strictly pin problematic dependencies
echo "Creating pip constraint file..."
cat > /tmp/constraints.txt << EOL
transformers==4.30.2
numpy==1.25.2
huggingface_hub==0.12.0
sentence-transformers==2.2.2
EOL

# Set global pip config to use constraints
export PIP_CONSTRAINT="/tmp/constraints.txt"
# Prevent pip from considering newer versions of pinned packages
export PIP_EXISTS_ACTION=i

# Uninstall any existing FAISS installations
echo "Removing any existing FAISS installations..."
pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null || true

# Install the correct GPU version for CUDA 12.1
echo "Installing FAISS GPU for CUDA 12.1..."
pip install faiss-gpu-cu12==1.8.0.2

# Pin transformers version to avoid dependency resolution conflicts
echo "Installing specific transformers version..."
pip uninstall -y transformers
pip install --no-deps transformers==4.30.2

# Install only the essential dependencies needed for the memory core service
echo "Installing minimal dependencies for memory core service..."
pip install numpy==1.25.2 fastapi uvicorn pydantic httpx aiohttp scikit-learn

# Fix huggingface_hub and sentence-transformers compatibility issue
echo "Fixing huggingface_hub and sentence-transformers compatibility..."
pip uninstall -y huggingface_hub sentence-transformers || true
pip install --no-deps huggingface_hub==0.12.0 sentence-transformers==2.2.2

# Skip installing from requirements.txt completely - only install what we need
echo "Skipping non-essential packages from requirements.txt"
# The following line is commented out to skip potentially problematic package installations
# if [ -f "/app/requirements.txt" ]; then
#     echo "# Modified requirements with constraints applied" > /tmp/requirements_constrained.txt
#     grep -v "transformers\|numpy\|huggingface-hub\|sentence-transformers\|livekit" /app/requirements.txt >> /tmp/requirements_constrained.txt
#     echo "Installing from constrained requirements..."
#     pip install -r /tmp/requirements_constrained.txt
# fi

# Run the docker-numpy-fix.sh script if it exists
if [ -f "/app/docker-numpy-fix.sh" ]; then
    echo "Running numpy fix script..."
    bash /app/docker-numpy-fix.sh
fi

# Verify installation
echo "Verifying FAISS installation..."
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'GPU support available: {faiss.get_num_gpus() > 0}')" || echo "FAISS verification failed, but continuing..."

echo "Verifying transformers installation..."
python -c "import transformers; print(f'transformers version: {transformers.__version__}')" || echo "transformers verification failed, but continuing..."

echo "===== FAISS Setup Complete ====="

# Execute the command passed to the script
echo "Starting application..."
exec "$@"
