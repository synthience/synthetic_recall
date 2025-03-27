# Memory Vector Index with FAISS

## Overview

The `MemoryVectorIndex` class provides an efficient vector similarity search implementation using Facebook AI Similarity Search (FAISS). This implementation supports both CPU and GPU acceleration, with automatic detection and installation of the appropriate FAISS package.

## Features

- Fast vector similarity search using FAISS
- Automatic GPU detection and utilization
- Dynamic FAISS installation if the package is missing
- Persistent storage and loading of indices
- Support for different similarity metrics (L2, Inner Product, Cosine)

## Architecture

The vector index implementation consists of two main components:

1. **Docker Integration**: Pre-installs FAISS during container startup
2. **Dynamic Import**: Auto-installs FAISS if missing during runtime

### FAISS Auto-Installation

The system implements a robust approach to FAISS installation:

```python
# Dynamic FAISS import with auto-installation fallback
try:
    import faiss
except ImportError:
    # Auto-detect GPU and install appropriate FAISS version
    ...
```

This pattern ensures FAISS is available regardless of whether it was pre-installed, making the system more resilient to environment changes.

## GPU Support

The vector index automatically detects GPU availability and uses GPU acceleration when possible:

1. During Docker startup, the system checks for NVIDIA GPUs and installs either `faiss-gpu` or `faiss-cpu`
2. At runtime, if a GPU is detected, the vector index is moved to GPU memory for faster similarity search
3. If the GPU becomes unavailable, the system gracefully falls back to CPU processing

## Usage

### Basic Usage

```python
from synthians_memory_core.vector_index import MemoryVectorIndex

# Create a new index
index = MemoryVectorIndex({
    'embedding_dim': 768,
    'storage_path': '/app/memory/stored/synthians',
    'index_type': 'L2',  # 'L2', 'IP', 'Cosine'
    'use_gpu': True,     # Whether to attempt to use GPU
    'gpu_id': 0          # Which GPU to use
})

# Add vectors to the index
index.add("memory_id_1", embedding_1)  # embedding_1 is a numpy array

# Search for similar vectors
results = index.search(query_embedding, k=10)  # Returns list of (memory_id, score) tuples
```

### Configuration

The `MemoryVectorIndex` accepts the following configuration options:

| Parameter | Description | Default |
|-----------|-------------|--------|
| `embedding_dim` | Dimensionality of the embeddings | 768 |
| `storage_path` | Path to store the index | '/app/memory/stored/synthians' |
| `index_type` | Type of FAISS index to use ('L2', 'IP', 'Cosine') | 'L2' |
| `use_gpu` | Whether to use GPU acceleration if available | True |
| `gpu_id` | Which GPU to use if multiple are available | 0 |

## Implementation Details

### Index Types

- **L2**: Euclidean distance (smaller values = more similar)
- **IP**: Inner Product similarity (larger values = more similar)
- **Cosine**: Cosine similarity with normalized vectors (larger values = more similar)

### Persistence

The vector index is automatically persisted to disk and can be reloaded on restart:

```python
# Save index to disk
index.save()

# Load index from disk
index.load()
```

## Docker Integration

The Docker Compose configuration pre-installs FAISS with GPU support if available:

```yaml
command: >
  /bin/bash -c '
  # Pre-install FAISS before Python importing it
  echo "[+] PRE-INSTALLING FAISS FOR MEMORY VECTOR INDEX" &&
  pip install --upgrade pip setuptools wheel &&
  # Install CPU version first as a fallback
  pip install --no-cache-dir faiss-cpu &&
  # If GPU available, replace with GPU version
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "[+] GPU DETECTED - Installing FAISS-GPU for better performance" &&
    pip uninstall -y faiss-cpu &&
    pip install --no-cache-dir faiss-gpu
  fi &&
  ...
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'faiss'**
   
   This should be handled automatically by the dynamic import system. If it persists:
   - Ensure pip is available in the environment
   - Check if CUDA is properly installed for GPU support
   - Try manually installing: `pip install faiss-cpu` or `pip install faiss-gpu`

2. **GPU Not Detected**

   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Ensure CUDA is properly configured
   - Check if the Docker container has GPU access

3. **Performance Issues**

   - For large indices, adjust memory allocation: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
   - Consider using a different index type for your specific use case
   - For high-dimensional vectors, consider using PCA or product quantization
