# NGC TensorFlow Environment Setup Guide

## Overview

This guide details the process of setting up a GPU-accelerated TensorFlow environment using NVIDIA NGC containers and Docker. This configuration has been tested and verified with the NVIDIA GeForce RTX 4090.

## Prerequisites

- NVIDIA GPU (Tested with RTX 4090)
- NVIDIA drivers installed and functional
- Docker with NVIDIA Container Toolkit
- NGC CLI tool
- PowerShell or compatible terminal

## Detailed Setup Process

### 1. NGC Configuration

#### Install NGC CLI

1. Download NGC CLI from NVIDIA's website
2. Add to system PATH
3. Verify installation:

```bash
ngc --version
```

#### Configure NGC Credentials

```bash
# Set up NGC API key and organization
ngc config set
```

#### Verify Configuration

```bash
ngc config verify
```

### 2. Container Management

#### List Available Containers

```bash
# View all TensorFlow containers
ngc registry image list nvidia/tensorflow

# View container details
ngc registry image info nvidia/tensorflow:22.12-tf2-py3
```

#### Pull Container

```bash
# Pull specific version
ngc registry image pull nvidia/tensorflow:22.12-tf2-py3
```

### 3. Development Environment Setup

#### Direct Container Launch

```powershell
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/project -it nvcr.io/nvidia/tensorflow:22.12-tf2-py3
```

#### Using Docker Compose

Create or update docker-compose.yml:

```yaml
version: '3.8'
services:
  lucid-recall:
    image: nvcr.io/nvidia/tensorflow:22.12-tf2-py3
    volumes:
      - .:/workspace/project
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Launch with:

```bash
docker-compose up --build
```

### 4. Environment Verification

#### Basic GPU Detection

```python
import tensorflow as tf

# Check GPU availability
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)

# Verify GPU acceleration
with tf.device('/GPU:0'):
    # Simple matrix multiplication to test GPU
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:", c)
```

#### Performance Test

Save as `test_gpu.py`:

```python
import tensorflow as tf
import time

# Create large matrices
size = 2000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# GPU computation
start_time = time.time()
with tf.device('/GPU:0'):
    c = tf.matmul(a, b)
    _ = c.numpy()  # Force execution
gpu_time = time.time() - start_time
print(f"GPU computation time: {gpu_time:.2f} seconds")
```

### 5. Troubleshooting

#### Common Issues and Solutions

1. NGC Authentication Failures
   - Verify NGC API key: `ngc config verify`
   - Check organization settings: `ngc config get`
   - Ensure proper network connectivity

2. Container Launch Issues
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check Docker GPU support: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`
   - Verify NVIDIA Container Toolkit: `docker info | grep -i runtime`

3. GPU Detection Problems
   - Ensure NVIDIA_VISIBLE_DEVICES is set correctly
   - Check CUDA installation in container: `nvcc --version`
   - Verify TensorFlow can see GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

4. Memory-Related Issues
   - Monitor GPU memory: `nvidia-smi -l 1`
   - Adjust Docker memory limits
   - Check system swap space

### 6. Best Practices

1. Development Workflow
   - Mount code directory as volume for live development
   - Use docker-compose for consistent environment setup
   - Implement proper error handling for GPU operations

2. Resource Management
   - Monitor GPU memory usage
   - Implement proper cleanup of GPU resources
   - Use appropriate batch sizes for your GPU memory

3. Performance Optimization
   - Use mixed precision when possible
   - Implement proper data prefetching
   - Optimize model architecture for GPU execution

### 7. Additional Resources

- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [Docker GPU Guide](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
