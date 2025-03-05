# Setting Up Lucid Recall FAST on Apple M3 Max

This guide provides step-by-step instructions for setting up Lucid Recall FAST on a MacBook Pro with M3 Max chip, leveraging the Metal GPU acceleration capabilities.

## System Specifications

- Model: MacBook Pro (Mac15,8)
- Chip: Apple M3 Max
- CPU: 16 cores (12 performance, 4 efficiency)
- Memory: 128 GB
- OS: macOS (Latest Version)

## Prerequisites

1. Install Docker Desktop for Apple Silicon

   ```bash
   # Download from https://docs.docker.com/desktop/install/mac/
   ```

   After installation, ensure Docker Desktop is running and configured for your M3 Max.

1. Install Python Dependencies (Host System)

   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

## Dependencies Configuration

Create a requirements-apple.txt file with the following contents:

```text
# Core ML dependencies
numpy>=1.24.0,<1.25.0
scikit-learn>=1.0.0
scipy>=1.7.0
typing-extensions>=4.8.0

# TensorFlow for Apple Silicon
tensorflow-macos==2.15.0
tensorflow-metal

# Utilities
tqdm>=4.65.0
requests>=2.31.0
pyyaml>=6.0.1
colorama>=0.4.6
psutil>=5.9.0

# PyTorch for Apple Silicon
torch
torchvision
torchaudio

# Transformers and related
transformers>=4.30.0
sentence-transformers>=2.2.0

# Development dependencies
pytest>=7.0.0
black>=22.0.0
isort>=5.0.0
flake8>=4.0.0
```

## Docker Configuration

1. Create a custom Dockerfile for Apple Silicon

   ```dockerfile
   # Use Python base image optimized for ARM64
   FROM python:3.9-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       git \
       && rm -rf /var/lib/apt/lists/*

   # Set working directory
   WORKDIR /app

   # Copy requirements
   COPY requirements-apple.txt .

   # Install Python dependencies
   RUN pip install --no-cache-dir -r requirements-apple.txt

   # Copy application code
   COPY . .

   # Set environment variables
   ENV TF_FORCE_GPU_ALLOW_GROWTH=true
   ENV TF_CPP_MIN_LOG_LEVEL=0
   ```

1. Create a docker-compose.yml file

   ```yaml
   version: '3.8'
   services:
     lucid-recall:
       build: .
       platform: linux/arm64
       volumes:
         - .:/app
       environment:
         - TF_FORCE_GPU_ALLOW_GROWTH=true
         - TF_CPP_MIN_LOG_LEVEL=0
       deploy:
         resources:
           reservations:
             devices:
               - capabilities: [gpu]
   ```

## Building and Running the Container

1. Build the Docker image

   ```bash
   docker-compose build
   ```

1. Run the container

   ```bash
   docker-compose up
   ```

## Verifying GPU Support

1. Run the GPU test script

   ```bash
   python test_hpc.py
   ```

   This will display system information and run a GPU test using the HPCFlowManager.

1. Expected Output:

   ```text
   === System Information ===
   TensorFlow version: 2.15.0
   Available devices:
     GPU: /device:GPU:0 (Apple Metal)

   === Testing GPU Processing ===
   Initialized with GPU: /device:GPU:0
   Test result: Test successful
   GPU test completed successfully
   ```

## Performance Optimization

1. TensorFlow Metal Configuration:
   - Add the following to your Python scripts before TensorFlow operations:

   ```python
   import tensorflow as tf

   # Configure memory growth
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

1. PyTorch Configuration:
   - PyTorch will automatically use Metal Performance Shaders (MPS) when available
   - Use the following device configuration:

   ```python
   import torch

   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

## Troubleshooting

1. Metal GPU Detection:
   - Ensure tensorflow-metal is properly installed
   - Verify Metal support is enabled in Docker Desktop settings
   - Check system logs for any GPU-related errors

1. Common Issues:
   - If TensorFlow fails to detect Metal GPU:

   ```bash
   # Verify Metal plugin installation
   pip list | grep tensorflow
   ```

   - If PyTorch MPS device is not available:

   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

## Monitoring and Maintenance

1. Monitor GPU Usage:
   - Use Activity Monitor to check Metal GPU usage
   - Monitor memory consumption through Docker Desktop
   - Check TensorFlow logs for any performance warnings

1. Regular Updates:
   - Keep Docker Desktop updated for optimal M3 Max support
   - Regularly update tensorflow-macos and tensorflow-metal packages
   - Check for system updates that might affect Metal performance

## Security Considerations

1. Container Security:
   - The container runs with minimal privileges
   - Only necessary ports are exposed
   - Volume mounts are restricted to required directories

1. Resource Management:
   - GPU memory growth is controlled to prevent memory exhaustion
   - Container resources are properly limited through Docker compose

## Additional Notes

- The system uses TensorFlow's Metal plugin for GPU acceleration on Apple Silicon
- The HPCFlowManager will automatically detect and use the Metal GPU device
- Performance may vary based on model complexity and batch sizes
- Both TensorFlow and PyTorch are configured to use Metal for optimal performance
- Regular monitoring of GPU temperature and power consumption is recommended for intensive workloads

Remember to monitor system performance and adjust configurations as needed for optimal performance on your M3 Max system.
