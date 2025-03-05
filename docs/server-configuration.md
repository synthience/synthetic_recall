# Lucid Recall Server Configuration

## Memory System Configuration (tensor_server.py)

### 1. Server Settings
```python
MEMORY_SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'max_connections': 100,
    'connection_timeout': 30,  # seconds
    'ping_interval': None,     # Disabled for testing
    'max_message_size': 1024 * 1024  # 1MB
}
```

### 2. GPU Configuration
```python
GPU_CONFIG = {
    'memory_fraction': 0.8,    # Use 80% of GPU memory
    'cleanup_threshold': 0.9,  # 90% triggers cleanup
    'cache_threshold': 0.95,   # 95% forces cache clear
    'mixed_precision': True,   # Enable automatic mixed precision
    'cudnn_benchmark': True,   # Enable cuDNN auto-tuner
    'tensor_cores': True       # Enable tensor cores
}
```

### 3. Memory Management
```python
EPHEMERAL_MEMORY_CONFIG = {
    'max_size': 1000,          # Maximum number of memories
    'decay_rate': 0.1,         # Memory decay rate
    'retention_threshold': 0.7, # Minimum significance for retention
    'device': 'cuda'           # Use GPU if available
}
```

## HPC Server Configuration (hpc_server.py)

### 1. Server Settings
```python
HPC_SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5004,
    'max_connections': 50,
    'connection_timeout': 30,  # seconds
    'ping_interval': None,     # Disabled for testing
    'max_message_size': 1024 * 1024  # 1MB
}
```

### 2. HPC Processor Configuration
```python
HPC_PROCESSOR_CONFIG = {
    'chunk_size': 512,         # Processing chunk size
    'embedding_dim': 768,      # Embedding dimension
    'batch_size': 32,         # Batch processing size
    'momentum': 0.9,          # Momentum factor
    'diversity_threshold': 0.7,# Pattern diversity threshold
    'cluster_threshold': 0.8,  # Cluster formation threshold
    'max_clusters': 100,      # Maximum number of clusters
    'device': 'cuda'          # Use GPU if available
}
```

### 3. Semantic Processing
```python
SEMANTIC_CONFIG = {
    'hypersphere_norm': 2,    # L2 normalization
    'stability_threshold': 1e-8,# Minimum norm threshold
    'merge_threshold': 0.95,   # Cluster merge similarity
    'shock_threshold': 0.8,    # Surprise threshold
    'momentum_window': 512     # Context window size
}
```

## WebSocket Communication

### 1. Memory WebSocket
```python
MEMORY_WS_CONFIG = {
    'endpoint': 'ws://localhost:5000',
    'protocols': ['memory-protocol'],
    'retry_interval': 5000,   # ms
    'max_retries': 3,
    'heartbeat_interval': None,# Disabled for testing
    'close_timeout': 5000     # ms
}
```

### 2. HPC WebSocket
```python
HPC_WS_CONFIG = {
    'endpoint': 'ws://localhost:5004',
    'protocols': ['hpc-protocol'],
    'retry_interval': 5000,   # ms
    'max_retries': 3,
    'heartbeat_interval': None,# Disabled for testing
    'close_timeout': 5000     # ms
}
```

## Resource Management

### 1. Memory Limits
```python
MEMORY_LIMITS = {
    'max_gpu_memory': 0.8,    # 80% of available GPU memory
    'cleanup_threshold': 0.9,  # 90% triggers cleanup
    'cache_threshold': 0.95,  # 95% forces cache clear
    'min_free_memory': 0.1    # 10% minimum free memory
}
```

### 2. Processing Limits
```python
PROCESSING_LIMITS = {
    'max_batch_size': 32,
    'max_sequence_length': 512,
    'max_clusters': 100,
    'max_context_size': 1000,
    'process_timeout': 30     # seconds
}
```

## Performance Configuration

### 1. GPU Optimization
```python
GPU_OPTIMIZATION = {
    'mixed_precision': True,
    'memory_growth': True,
    'cudnn_benchmark': True,
    'tensor_cores': True,
    'cache_size': '2GB'
}
```

### 2. Processing Optimization
```python
PROCESSING_OPTIMIZATION = {
    'batch_processing': True,
    'hypersphere_projection': True,
    'semantic_clustering': True,
    'momentum_tracking': True,
    'surprise_detection': True
}
```

## Monitoring Configuration

### 1. Logging Settings
```python
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    'file': '/workspace/project/logs/server.log',
    'max_size': '100MB',
    'backup_count': 5,
    'console_output': True
}
```

### 2. Metrics Collection
```python
METRICS_CONFIG = {
    'collection_interval': 1,  # seconds
    'history_window': 3600,   # 1 hour of history
    'detailed_gpu_metrics': True,
    'cluster_metrics': True,
    'memory_metrics': True
}
```

## Error Handling Configuration

### 1. Retry Settings
```python
RETRY_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1000,      # ms
    'exponential_backoff': True,
    'max_delay': 5000,        # ms
    'retry_on_timeout': True
}
```

### 2. Error Reporting
```python
ERROR_CONFIG = {
    'detailed_errors': True,
    'stack_traces': True,
    'error_logging': True,
    'alert_on_critical': True,
    'max_error_log': '50MB'
}
```

## Cluster Management

### 1. Cluster Settings
```python
CLUSTER_CONFIG = {
    'max_clusters': 100,
    'merge_threshold': 0.95,
    'stability_threshold': 0.8,
    'min_cluster_size': 5,
    'cleanup_interval': 3600  # seconds
}
```

### 2. Pattern Detection
```python
PATTERN_CONFIG = {
    'surprise_threshold': 0.8,
    'diversity_factor': 0.7,
    'momentum_weight': 0.7,
    'cluster_weight': 0.3,
    'temporal_decay': 0.1
}