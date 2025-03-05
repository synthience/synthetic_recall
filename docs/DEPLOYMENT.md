# Lucid Recall Deployment Guide

## 1. System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: CUDA-capable with 4GB+ VRAM
- **Storage**: 20GB+ for base system
- **Network**: Gigabit Ethernet recommended

### Software Requirements
- Docker Engine 20.10+
- NVIDIA Container Toolkit
- CUDA 11.x or later
- Python 3.8 or later
- WebSocket-capable browser

## 2. Container Setup

### Primary Container: nemo_sig_v3
This container hosts the core processing servers and memory system.

#### Network Configuration
- Tensor Server: Port 5000
- HPC Server: Port 5004
- LM Studio Connection: Port 1234

#### Resource Allocation
```bash
# Recommended Docker run configuration
docker run \
  --gpus all \
  --name nemo_sig_v3 \
  --network host \
  -v /path/to/project:/workspace/project \
  -e CUDA_VISIBLE_DEVICES=0 \
  nemo_sig_v3:latest
```

## 3. Server Deployment

### Starting the System
1. Start the container:
```bash
docker start nemo_sig_v3
```

2. Launch servers:
```powershell
# Using provided script
./start-lucid-recall.ps1

# Or manually
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python tensor_server.py"
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python hpc_server.py"
```

### Verifying Deployment
1. Check server processes:
```bash
docker exec nemo_sig_v3 ps aux | findstr python
```

2. Verify ports:
```bash
docker exec nemo_sig_v3 netstat -tulpn | findstr -E "5000|5004"
```

3. Check GPU status:
```bash
docker exec nemo_sig_v3 nvidia-smi
```

## 4. Monitoring and Maintenance

### System Monitoring
1. Memory Usage:
```bash
# Container memory
docker stats nemo_sig_v3

# GPU memory
docker exec nemo_sig_v3 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

2. Server Logs:
```bash
# Tensor Server logs
docker exec nemo_sig_v3 tail -f /workspace/project/server/tensor_server.log

# HPC Server logs
docker exec nemo_sig_v3 tail -f /workspace/project/server/hpc_server.log
```

### Maintenance Tasks

#### 1. Server Restart
```powershell
# Stop servers
docker exec nemo_sig_v3 pkill -f "python.*tensor_server.py"
docker exec nemo_sig_v3 pkill -f "python.*hpc_server.py"

# Clear GPU memory
docker exec nemo_sig_v3 nvidia-smi --gpu-reset

# Restart servers
./start-lucid-recall.ps1
```

#### 2. Memory Cleanup
```bash
# Clear memory cache
docker exec nemo_sig_v3 sh -c "rm -rf /workspace/project/memory_store/memories/*"

# Restart servers to clear RAM
./start-lucid-recall.ps1
```

## 5. Troubleshooting

### Common Issues

#### 1. GPU Memory Issues
```bash
# Check GPU memory
docker exec nemo_sig_v3 nvidia-smi

# Reset GPU if needed
docker exec nemo_sig_v3 nvidia-smi --gpu-reset
```

#### 2. Server Connection Issues
```bash
# Check if servers are running
docker exec nemo_sig_v3 ps aux | findstr python

# Check port availability
docker exec nemo_sig_v3 netstat -tulpn | findstr -E "5000|5004"

# Restart servers if needed
./start-lucid-recall.ps1
```

#### 3. Memory System Issues
```bash
# Check memory stats
curl -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Host: localhost:5000" -H "Origin: http://localhost:5000" \
  "ws://localhost:5000" \
  -d '{"type":"stats"}'
```

## 6. Backup and Recovery

### Backup Procedures
1. Container State:
```bash
docker commit nemo_sig_v3 nemo_sig_v3_backup
```

2. Memory Store:
```bash
docker exec nemo_sig_v3 tar -czf /workspace/backup/memory_store.tar.gz /workspace/project/memory_store
```

### Recovery Procedures
1. Container Recovery:
```bash
docker stop nemo_sig_v3
docker rm nemo_sig_v3
docker run --name nemo_sig_v3 nemo_sig_v3_backup
```

2. Memory Store Recovery:
```bash
docker exec nemo_sig_v3 tar -xzf /workspace/backup/memory_store.tar.gz -C /
```

## 7. Performance Optimization

### GPU Optimization
1. Memory Management:
```python
# In tensor_server.py config
torch.cuda.set_per_process_memory_fraction(0.8)
torch.backends.cudnn.benchmark = True
```

2. Batch Processing:
```python
# In hpc_sig_flow_manager.py config
'batch_size': 32,
'chunk_size': 384
```

### Network Optimization
1. WebSocket Configuration:
- Keep-alive interval: 30 seconds
- Message size limit: 10MB
- Connection timeout: 60 seconds

2. Connection Pooling:
- Max connections per server: 1000
- Connection timeout: 5 seconds
- Retry interval: 1 second

## 8. Security Considerations

### Network Security
- Internal network deployment only
- No external port exposure
- Container network isolation
- Regular security updates

### Access Control
- No authentication currently implemented
- Container access restrictions
- File system permissions
- Port access limitations

### Data Protection
- Memory isolation
- No persistent storage
- Container filesystem isolation
- Regular backup procedures