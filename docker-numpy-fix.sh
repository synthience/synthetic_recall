#!/bin/bash

# First install all required dependencies
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r /workspace/project/requirements.txt

# Install fastapi explicitly in case it's not in requirements.txt
echo "Installing FastAPI..."
pip install fastapi uvicorn[standard]

# Install TensorFlow explicitly
echo "Installing TensorFlow..."
pip install tensorflow>=2.10

# Downgrade NumPy to avoid binary incompatibility
echo "Downgrading NumPy to 1.26.4..."
pip install --force-reinstall numpy==1.26.4

# Execute the original command
echo "Starting the application..."
exec "$@"
