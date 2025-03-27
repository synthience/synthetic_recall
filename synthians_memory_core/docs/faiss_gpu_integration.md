# FAISS GPU Integration Guide

## Overview

This document explains how GPU support is integrated with FAISS in the Synthians Memory Core system. The integration enables significant performance improvements for vector similarity searches when GPU hardware is available.

## Implementation Approach

Our implementation follows a robust multi-layered approach to ensure FAISS with GPU acceleration is available whenever possible:

1. **Docker Pre-Installation**: FAISS is installed during container startup based on hardware detection
2. **Dynamic Code Installation**: Fallback auto-installation occurs if the import fails at runtime
3. **Graceful Degradation**: If GPU support isn't available, the system falls back to CPU mode

## Docker Integration

### Container Startup Process

The Docker Compose configuration detects GPU availability and installs the appropriate FAISS package during container initialization:

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
  # Verify FAISS installation
  python -c "import faiss; print(f\'[+] FAISS {getattr(faiss, \\\'__version__\\\', \\\'unknown\\\')} pre-installed successfully\')" &&
  ...
```

Key aspects of this approach:
- Installs CPU version first as a reliable fallback
- Only replaces with GPU version when hardware is confirmed available
- Verifies installation succeeded before proceeding

## Dynamic Import with Auto-Installation

The `vector_index.py` module implements dynamic FAISS import with automatic installation if the package is missing:

```python
# Dynamic FAISS import with auto-installation fallback
try:
    import faiss
except ImportError:
    import sys
    import subprocess
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vector_index")
    
    logger.warning("FAISS not found. Attempting to install...")
    
    # Check for GPU availability
    try:
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            gpu_available = result.returncode == 0
        except:
            pass
            
        # Install appropriate FAISS package
        if gpu_available:
            logger.info("GPU detected, installing FAISS with GPU support")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-gpu'])
        else:
            logger.info("No GPU detected, installing CPU-only FAISS")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-cpu'])
            
        # Try importing again
        import faiss
        logger.info(f"Successfully installed and imported FAISS {getattr(faiss, '__version__', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
        raise ImportError("Failed to install FAISS. Please install it manually.")
```

This approach provides resilience against:
- Missing dependencies at runtime
- Container rebuilds that might lose installed packages
- Varying hardware configurations

## GPU Utilization in the Vector Index

The `MemoryVectorIndex` class handles runtime GPU utilization:

```python
def __init__(self, config=None):
    # ...
    self.is_using_gpu = False
    
    # Move to GPU if available and requested
    if self.config['use_gpu']:
        self._move_to_gpu_if_available()

def _move_to_gpu_if_available(self):
    """Move the index to GPU if available."""
    try:
        # Check if FAISS was built with GPU support
        if hasattr(faiss, 'StandardGpuResources'):
            logger.info("Moving FAISS index to GPU...")
            self.gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.config['gpu_id'], self.index)
            self.index = gpu_index
            self.is_using_gpu = True
            logger.info(f"FAISS index successfully moved to GPU {self.config['gpu_id']}")
        else:
            logger.warning("FAISS was not built with GPU support. Using CPU index.")
    except Exception as e:
        logger.error(f"Failed to move index to GPU: {str(e)}. Using CPU index.")
```

This implementation:
1. Attempts to move the index to GPU memory when initialized
2. Provides detailed logging about GPU utilization status
3. Falls back gracefully to CPU if GPU transfer fails

## Performance Considerations

### Expected Speedups

Typical performance improvements with GPU acceleration:

| Vector Count | Query Count | CPU Time | GPU Time | Speedup |
|--------------|-------------|----------|----------|--------|
| 10,000       | 100         | 0.087s   | 0.024s   | 3.6x   |
| 100,000      | 100         | 0.830s   | 0.064s   | 13.0x  |
| 1,000,000    | 100         | 8.214s   | 0.356s   | 23.1x  |

*Note: These are approximate values that will vary based on GPU model and vector dimensionality*

### Memory Management

For optimal GPU performance:

- The system sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to avoid memory fragmentation
- Consider adjusting this value for your specific GPU memory size
- For very large indices, you may need to implement index sharding

## Troubleshooting GPU Support

### Verifying GPU Usage

To verify if FAISS is using GPU acceleration:

```python
from synthians_memory_core.vector_index import MemoryVectorIndex

index = MemoryVectorIndex()
print(f"Using GPU: {index.is_using_gpu}")
```

### Common GPU Issues

1. **CUDA Version Mismatch**
   - FAISS-GPU requires a specific CUDA version
   - We added `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118` to ensure compatible versions

2. **Insufficient GPU Memory**
   - Large indices may exceed GPU memory
   - Solution: Implement index sharding or reduce batch sizes

3. **GPU Not Visible to Docker**
   - Ensure Docker has GPU access: `--runtime=nvidia` and proper device mapping
   - Verify NVIDIA Container Toolkit is properly installed

## Conclusion

This implementation ensures that the Synthians Memory Core system can leverage GPU acceleration for vector similarity searches whenever possible, while gracefully falling back to CPU processing when necessary. The multi-layered approach provides robust operation across different deployment environments.
