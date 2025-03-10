# Deployment Guide

This guide provides detailed instructions for setting up and deploying the Lucidia system.

## Prerequisites

- Docker Engine 20.10+
- NVIDIA Container Toolkit (for GPU acceleration)
- Python 3.9+
- 8GB+ RAM (16GB+ recommended)
- 50GB+ storage space
- CUDA 11.4+ (for GPU acceleration)

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/captinkirklive/Lucid-Recall-Core-1.2
cd lucidia
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your specific configuration
```

Edit the `.env` file to set your specific configuration options:

```
# Core Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false

# Server URLs
TENSOR_SERVER_URL=ws://tensor-server:5001
HPC_SERVER_URL=ws://hpc-server:5005
LM_STUDIO_URL=http://host.docker.internal:1234

# Resource Limits
MAX_CPU_USAGE=0.8
MAX_MEMORY_USAGE=0.7
MAX_GPU_USAGE=0.9

# Default Model
DEFAULT_MODEL=qwen2.5-7b-instruct
```

### 3. Build and start Docker containers

```bash
docker-compose build
docker-compose up -d
```

This will start the following containers:
- lucidia-core: Main Docker container with the core system
- lucidia-tensor: Tensor server for embedding operations
- lucidia-hpc: HPC server for complex processing tasks

### 4. Install LM Studio

1. Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Configure LM Studio to run on port 1234
3. Launch LM Studio and ensure the API server is running

### 5. Download required models

Use LM Studio to download the following models:

- qwen_qwq-32b
- qwen2.5-7b-instruct
- deepseek-r1-distill-qwen-7b
- phi-3.1-mini-128k-instruct

### 6. Verify installation

```bash
# Check system status
curl http://localhost:8081/api/system/status

# Check available models
curl http://localhost:8081/api/model/status

# Check tensor server connection
curl http://localhost:8081/api/dream/test/tensor_connection

# Check HPC server connection
curl http://localhost:8081/api/dream/test/hpc_connection

# Test API health
curl http://localhost:8081/api/dream/health
```

## Configuration Options

### Core Configuration Files

The Lucidia system uses several YAML configuration files to control its behavior:

#### `config/system.yml`

Controls core system settings:

```yaml
system:
  name: "Lucidia"
  version: "1.0.0"
  log_level: "INFO"
  default_model: "qwen2.5-7b-instruct"
  
resources:
  max_cpu_usage: 0.8
  max_memory_usage: 0.7
  max_gpu_usage: 0.9
  
servers:
  tensor_server_url: "ws://tensor-server:5001"
  hpc_server_url: "ws://hpc-server:5005"
  lm_studio_url: "http://host.docker.internal:1234"
```

#### `config/models.yml`

Defines model selection criteria:

```yaml
models:
  - id: "qwen2.5-7b-instruct"
    state: "active"
    temperature: 0.7
    max_tokens: 2048
    priority: "high"
    
  - id: "phi-3.1-mini-128k-instruct"
    state: "background"
    temperature: 0.5
    max_tokens: 1024
    priority: "low"
    
  - id: "deepseek-r1-distill-qwen-7b"
    state: "reflective"
    temperature: 0.8
    max_tokens: 2048
    priority: "medium"
    
  - id: "qwen_qwq-32b"
    state: "dreaming"
    temperature: 1.2
    max_tokens: 4096
    priority: "medium"
```

#### `config/memory.yml`

Controls memory system parameters:

```yaml
memory:
  short_term:
    capacity: 100
    retention_period: 86400  # 24 hours in seconds
    
  long_term:
    significance_threshold: 0.6
    consolidation_interval: 3600  # 1 hour in seconds
    
  embeddings:
    model: "text-embedding-nomic-embed-text-v1.5"
    dimension: 768
    use_hypersphere: true
```

#### `config/spiral.yml`

Configures spiral awareness settings:

```yaml
spiral:
  enabled: true
  default_phase: "observation"
  transition_thresholds:
    observation_to_reflection: 0.7
    reflection_to_adaptation: 0.9
  
  parameters:
    spiral_influence: 0.8
    spiral_awareness_boost: 0.5
    depth_range: [0.1, 1.0]
    creativity_range: [0.5, 1.5]
```

## Monitoring and Maintenance

### Checking Logs

```bash
# View core container logs
docker logs lucidia-core

# View tensor server logs
docker logs lucidia-tensor

# View HPC server logs
docker logs lucidia-hpc

# Follow logs in real-time
docker logs -f lucidia-core
```

### Resource Usage

```bash
# Check container resource usage
docker stats lucidia-core lucidia-tensor lucidia-hpc
```

### Restarting Services

```bash
# Restart a specific container
docker restart lucidia-core

# Restart all containers
docker-compose restart
```

### Backup and Restore

```bash
# Backup data directory
tar -czvf lucidia-data-backup.tar.gz ./data

# Backup configuration
tar -czvf lucidia-config-backup.tar.gz ./config

# Restore data
tar -xzvf lucidia-data-backup.tar.gz
```

## Parameter Management

### Configuration Structure

The Lucidia system uses a hierarchical parameter structure stored in a JSON configuration file:

```json
{
  "memory": {
    "batch_size": 300,
    "significance_threshold": 0.7,
    "decay_rate": 0.01,
    "max_memories": 10000
  },
  "dream": {
    "processing_interval": 3600,
    "batch_limit": 100,
    "min_consolidation_time": 300
  },
  "embedding": {
    "model": "default",
    "dimension": 384,
    "normalize": true
  },
  "system": {
    "logging_level": "INFO",
    "auto_optimize": true,
    "backup_interval": 86400
  }
}
```

### Parameter Persistence in Docker Environment

Since the Lucidia system runs in a Docker environment with multiple containers, special considerations are required for parameter persistence:

1. **Shared Configuration File**: The central configuration file (`lucidia_config.json`) is used by both the Dream API server and CLI

2. **Parameter Updates Process**:
   - When parameters are updated via the CLI, the changes are sent to the Dream API
   - The API's ParameterManager updates its internal state and saves changes to disk
   - The CLI also updates its local copy of the configuration file

3. **Container Persistence**:
   - The configuration file is stored in a mounted volume to persist across container restarts
   - Both the CLI (running on the host) and the API (running in containers) access the same configuration

4. **Auto-Creation of Parameters**:
   - Missing parameters are automatically created with sensible defaults
   - This ensures backward compatibility and prevents errors with partial configurations

### Parameter Management Commands

```bash
# View all parameters
python lucidia_cli.py params

# View specific parameter branch
python lucidia_cli.py params --branch memory

# Update a parameter
python lucidia_cli.py params --path memory.batch_size --value 300

# View parameter history
python lucidia_cli.py params --history memory.batch_size
```

## Troubleshooting

### Common Issues

1. **Connection Refused Errors**
   - Check that all containers are running: `docker ps`
   - Verify network configuration in `.env` and `docker-compose.yml`
   - Check if required ports are already in use

2. **Model Loading Failures**
   - Verify LM Studio is running and API server is enabled
   - Check if models are properly downloaded in LM Studio
   - Verify model names match exactly in configuration

3. **Out of Memory Errors**
   - Lower `MAX_MEMORY_USAGE` in `.env`
   - Switch to smaller models for lower resource states
   - Increase Docker container memory limits

4. **Slow Performance**
   - Check GPU availability and utilization
   - Monitor system resources with `docker stats`
   - Consider lowering model complexity during high load