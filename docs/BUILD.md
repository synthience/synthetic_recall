# Lucid Recall Build Guide

## Container Build Instructions

### 1. Core Container (nemo_sig_v3)

#### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Git for repository access

#### Build Steps
1. Build core image:
```bash
# From project root
docker build -t lucid-recall-core -f Dockerfile.lucid-recall-core .
```

2. Verify build:
```bash
docker run --rm --gpus all lucid-recall-core nvidia-smi
```

### 2. Port Forward Containers
These use the official alpine/socat image, no build required.

## Deployment Setup

### 1. Environment Configuration
1. Create .env file:
```bash
# Copy example
cp .env.example .env

# Edit with your paths
MODELS_PATH=/path/to/models
```

2. Verify model path:
```bash
# Create if needed
mkdir -p ${MODELS_PATH}
```

### 2. Container Deployment

1. Start all containers:
```bash
docker-compose up -d
```

2. Verify deployment:
```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs -f
```

### 3. Server Initialization

1. Start servers:
```powershell
# Using provided script
./start-lucid-recall.ps1

# Or manually
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python tensor_server.py"
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python hpc_server.py"
```

2. Verify servers:
```bash
# Check processes
docker exec nemo_sig_v3 ps aux | findstr python

# Check ports
docker exec nemo_sig_v3 netstat -tulpn | findstr -E "5000|5004"
```

## Container Details

### 1. lucid-recall-core
- Base: nvcr.io/nvidia/pytorch:23.12-py3
- CUDA support
- Python dependencies
- Project source code
- Entry point script

### 2. port-forward-5000
- Base: alpine/socat
- Port 5000 forwarding
- Links to nemo_sig_v3
- Minimal Alpine Linux

### 3. port-forward-5004
- Base: alpine/socat
- Port 5004 forwarding
- Links to nemo_sig_v3
- Minimal Alpine Linux

## Network Configuration

### Bridge Network
```yaml
networks:
  lucid-net:
    driver: bridge
```

### Port Mappings
- Tensor Server: 5000
- HPC Server: 5004
- Internal communication via Docker network

## Volume Configuration

### Project Mount
```yaml
volumes:
  - ./:/workspace/project
```

### Models Mount
```yaml
volumes:
  - ${MODELS_PATH}:/workspace/models
```

## Maintenance

### 1. Rebuilding Containers
```bash
# Rebuild core
docker-compose build nemo_sig_v3

# Restart all
docker-compose down
docker-compose up -d
```

### 2. Updating Images
```bash
# Update socat images
docker pull alpine/socat

# Restart forwards
docker-compose up -d --force-recreate port-forward-5000 port-forward-5004
```

### 3. Log Management
```bash
# View specific logs
docker-compose logs nemo_sig_v3
docker-compose logs port-forward-5000
docker-compose logs port-forward-5004

# Follow all logs
docker-compose logs -f
```

## Troubleshooting

### 1. GPU Issues
```bash
# Check GPU access
docker exec nemo_sig_v3 nvidia-smi

# Reset GPU if needed
nvidia-smi --gpu-reset
```

### 2. Port Issues
```bash
# Check port availability
netstat -an | findstr "5000 5004"

# Restart port forwards
docker-compose restart port-forward-5000 port-forward-5004
```

### 3. Memory Issues
```bash
# Check container memory
docker stats nemo_sig_v3

# Clear GPU memory
docker exec nemo_sig_v3 python -c "import torch; torch.cuda.empty_cache()"
```

## Development Setup

### 1. Local Development
```bash
# Mount local code
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access container
docker exec -it nemo_sig_v3 bash
```

### 2. Testing
```bash
# Run tests
docker exec nemo_sig_v3 python -m unittest discover -s /workspace/project/server

# Check server status
docker exec nemo_sig_v3 curl -s http://localhost:5000/status
docker exec nemo_sig_v3 curl -s http://localhost:5004/status
```

### 3. Debugging
```bash
# Access logs
docker exec nemo_sig_v3 tail -f /workspace/project/server/*.log

# Check processes
docker exec nemo_sig_v3 ps aux | grep python

# Monitor GPU
docker exec nemo_sig_v3 watch -n1 nvidia-smi