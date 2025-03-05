# Setting Up GPU-Accelerated TensorFlow Environment Using NGC and Docker

## Prerequisites

- NVIDIA GPU (Tested with RTX 4090)
- NVIDIA drivers installed
- Docker with NVIDIA Container Toolkit
- NGC CLI installed and configured
- PowerShell or compatible terminal

## Process Documentation

### 1. NGC Setup and Container Acquisition

```bash
# Configure NGC credentials
ngc config set

# List available TensorFlow containers
ngc registry image list nvidia/tensorflow

# Pull specific container version
ngc registry image pull nvidia/tensorflow:22.12-tf2-py3
```

### 2. Docker Container Launch

```powershell
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/project -it nvcr.io/nvidia/tensorflow:22.12-tf2-py3
```

Command Parameters Explained:

- `--gpus all`: Enable GPU access
- `--ipc=host`: Shared memory settings
- `--ulimit memlock=-1`: Remove memory lock limits
- `--ulimit stack=67108864`: Set stack size
- `-v ${PWD}:/workspace/project`: Mount current directory
- `-it`: Interactive terminal mode

### 3. Verification Steps

```python
import tensorflow as tf
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)
```

Would you like me to continue with:

1. Implementation details
2. Testing procedures
3. Common issues and solutions
4. Project structure and organization?
