# Lucid Recall System Architecture Documentation

## 1. System Overview

### Container Architecture
The system uses three interconnected containers:

1. **Core Processing Container (nemo_sig_v3)**
   - Based on NVIDIA CUDA-enabled image
   - Hosts main processing servers
   - GPU acceleration support
   - Project and model volume mounts
   - Internal ports 5000, 5004, 6006, 8888

2. **Tensor Server Port Forward (port-forward-5000)**
   - Based on alpine/socat
   - Forwards WebSocket traffic to tensor server
   - Maps host port 5000 to nemo_sig_v3
   - Handles client connections

3. **HPC Server Port Forward (port-forward-5004)**
   - Based on alpine/socat
   - Forwards WebSocket traffic to HPC server
   - Maps host port 5004 to nemo_sig_v3
   - Handles client connections

### Server Components

#### 1. Tensor Server (Port 5000)
- **Purpose**: Handles memory and embedding operations
- **Key Features**:
  - Embedding generation using MiniLM-L6-v2 model
  - Memory storage and retrieval
  - GPU acceleration when available
  - WebSocket interface for real-time communication

#### 2. HPC Server (Port 5004)
- **Purpose**: Hypersphere Processing Chain operations
- **Key Features**:
  - Embedding processing
  - Significance calculation
  - Surprise detection
  - Momentum-based processing

### Memory System

#### Embedding Processing
- Model: all-MiniLM-L6-v2
- Embedding Dimension: 384
- GPU Acceleration: Enabled when available
- Memory Cache: Local storage with significance scoring

#### HPC-SIG Flow Manager
- **Configuration**:
  - Chunk Size: 384 (matches embedding dimension)
  - Batch Size: 32
  - Momentum: 0.9
  - Diversity Threshold: 0.7
  - Surprise Threshold: 0.8

- **Processing Pipeline**:
  1. Embedding Normalization
  2. Surprise Detection
  3. Shock Absorption for High-Surprise Inputs
  4. Significance Calculation
  5. Momentum Buffer Management

#### Significance Calculation
Combines three factors:
- Surprise (40%): Deviation from momentum buffer
- Magnitude (30%): Embedding vector norm
- Diversity (30%): Distinctness from existing memories

## 2. Implementation Status

### Working Components
1. **Container System**
   - nemo_sig_v3 core processing
   - Port forwarding via socat
   - GPU acceleration support
   - Volume mounting for project and models

2. **Memory System**
   - Real-time embedding generation
   - Memory storage and retrieval
   - Significance-based memory ranking
   - GPU-accelerated processing

3. **WebSocket Servers**
   - Tensor Server: Memory operations
   - HPC Server: Processing chain
   - Port forwarding configuration
   - Error handling and recovery

4. **Integration Points**
   - Client-side memory integration
   - LM Studio connectivity
   - GPU acceleration
   - WebSocket message handling

### System Limitations
1. Fixed embedding dimension (384)
2. Memory storage in RAM (no persistence)
3. Single-node deployment
4. GPU dependency for optimal performance
5. Port forwarding overhead

## 3. Technical Details

### Server Paths and Ports
- Tensor Server: ws://localhost:5000 → nemo_sig_v3:5000
- HPC Server: ws://localhost:5004 → nemo_sig_v3:5004
- LM Studio: http://192.168.0.203:1234

### Volume Mounts
```
Project:
Host: G:\Development FAST\Lucid Recall FAST 1.2
Container: /workspace/project

Models:
Host: H:\Models
Container: /workspace/models
```

### Working Commands
```powershell
# Start Servers
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python tensor_server.py"
docker exec -d nemo_sig_v3 sh -c "cd /workspace/project/server && python hpc_server.py"

# Stop Servers
docker exec nemo_sig_v3 pkill -f "python.*tensor_server.py"
docker exec nemo_sig_v3 pkill -f "python.*hpc_server.py"

# Check Status
docker exec nemo_sig_v3 ps aux | findstr python
docker exec nemo_sig_v3 netstat -tulpn | findstr -E "5000|5004"
```

## 4. Future Development

### Planned Features
1. **Container Improvements**
   - Container orchestration
   - Automatic scaling
   - Better resource management
   - Persistent storage volumes

2. **Memory System Enhancements**
   - Persistent storage
   - Distributed memory architecture
   - Advanced significance algorithms

3. **System Expansion**
   - Clustering support
   - Load balancing
   - Failover capabilities

4. **Integration Opportunities**
   - Additional LLM integrations
   - External API connectivity
   - Custom embedding models

### Enhancement Priorities
1. Container orchestration
2. Memory persistence
3. Distributed processing
4. Advanced significance metrics
5. System monitoring
6. Performance optimization

## 5. Deployment Notes

### Requirements
- Docker environment
- CUDA-capable GPU (optional)
- Python 3.8+
- WebSocket support
- LM Studio connection
- Sufficient ports for forwarding

### Performance Considerations
- GPU memory management
- WebSocket connection limits
- Port forwarding overhead
- Memory usage monitoring
- Processing chain optimization

### Security Notes
- Internal network deployment
- No authentication implemented
- Port exposure considerations
- Container isolation
- Volume mount security