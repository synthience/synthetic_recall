# GPU Setup Documentation

## Environment Setup Options

### Option 1: Using NGC Container Directly

#### Prerequisites

1. NGC CLI installed and configured

   ```bash
   ngc config set
   ```

1. NVIDIA Container Toolkit installed
1. NVIDIA drivers properly installed on the host system

#### Container Details

- **NGC Container**: nvcr.io/nvidia/tensorflow:22.12-tf2-py3
- **TensorFlow Version**: 2.10.1
- **Verified GPU**: NVIDIA GeForce RTX 4090 (21194 MB memory)

#### Direct Docker Command

```powershell
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/project -it nvcr.io/nvidia/tensorflow:22.12-tf2-py3
```

### Option 2: Using Docker Compose

#### Docker Compose Configuration

The project includes a `docker-compose.yml` that configures:

- GPU access through NVIDIA runtime
- Volume mounting for development
- Resource reservations for GPU capabilities

```yaml
version: '3.8'
services:
  lucid-recall:
    build: .
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

To start using docker-compose:

```bash
docker-compose up --build
```

## Implementation Status

### Verified Components

- [✓] Basic memory management with GPU acceleration
- [✓] Hypersphere normalization
- [✓] Chunking processor
- [✓] Surprise detection
- [✓] Persistence layer
- [✓] TensorFlow GPU verification

### Active Development

- Ephemeral Memory system optimization
- Chunked processing improvements
- GPU performance benchmarking

## Project Structure

Key GPU-accelerated components are located in:

```text
src/
├── managers/
│   └── python/
│       ├── memory/
│       │   ├── ephemeral_memory.py
│       │   ├── chunk_processor.py
│       │   └── surprise_detection.py
│       └── persistence/
```

## Configuration Details

### Memory Settings

- `--ipc=host`: Uses host's IPC namespace for shared memory
- `--ulimit memlock=-1`: Removes memory lock limits
- `--ulimit stack=67108864`: Sets appropriate stack size

### Volume Mounting

- Development files are mounted at `/workspace/project`
- Ensures code changes are immediately reflected

## Verification

To verify GPU support is working:

1. Run the test script:

   ```python
   python test_tf.py
   ```

1. Check GPU detection:

   ```python
   import tensorflow as tf
   print("GPUs Available:", tf.config.list_physical_devices('GPU'))
   ```

## Troubleshooting

1. If GPU is not detected:
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check NVIDIA Container Toolkit: `docker info | grep -i runtime`
   - Ensure NGC container is properly pulled: `docker images`

1. Memory-related issues:
   - Adjust ulimit settings in docker command
   - Monitor GPU memory usage: `nvidia-smi -l 1`

1. Container access issues:
   - Verify user permissions for docker group
   - Check NGC authentication: `ngc config verify`
