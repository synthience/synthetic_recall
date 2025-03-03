# Lucid Recall Technical Specification

## 1. Server Implementations

### Tensor Server
- **Port**: 5000
- **Protocol**: WebSocket
- **Model**: all-MiniLM-L6-v2
- **Embedding Size**: 384
- **GPU Support**: Yes (CUDA)

#### API Endpoints
1. Embedding Generation
```json
Request:
{
    "type": "embed",
    "text": "content to embed"
}

Response:
{
    "type": "embeddings",
    "embeddings": [...],
    "id": "uuid",
    "significance": 0.75,
    "timestamp": 1234567890
}
```

2. Memory Search
```json
Request:
{
    "type": "search",
    "text": "search query",
    "limit": 5
}

Response:
{
    "type": "search_results",
    "results": [
        {
            "id": "memory_id",
            "text": "memory content",
            "similarity": 0.95,
            "significance": 0.8
        }
    ]
}
```

3. System Stats
```json
Request:
{
    "type": "stats"
}

Response:
{
    "type": "stats",
    "gpu_memory": 1.2,
    "gpu_cached": 2.1,
    "device": "cuda",
    "hpc_status": {...},
    "memory_count": 150
}
```

### HPC Server
- **Port**: 5004
- **Protocol**: WebSocket
- **Chunk Size**: 384
- **Batch Size**: 32

#### API Endpoints
1. Process Embedding
```json
Request:
{
    "type": "process",
    "embeddings": [...]
}

Response:
{
    "type": "processed",
    "embeddings": [...],
    "significance": 0.85
}
```

2. System Stats
```json
Request:
{
    "type": "stats"
}

Response:
{
    "type": "stats",
    "has_momentum": true,
    "momentum_size": 384,
    "device": "cuda"
}
```

## 2. Memory System

### Storage Format
```python
Memory = {
    'id': str,          # UUID
    'text': str,        # Original content
    'embedding': tensor, # Processed embedding
    'timestamp': float, # Unix timestamp
    'significance': float # 0.0 to 1.0
}
```

### Significance Calculation
```python
significance = (
    0.4 * surprise +    # Deviation from momentum
    0.3 * magnitude +   # Vector norm
    0.3 * diversity     # Uniqueness
)
```

### Processing Pipeline
1. Input Normalization
2. Surprise Detection
3. Shock Absorption
4. Momentum Update
5. Significance Scoring

## 3. Integration Points

### LM Studio Integration
- **URL**: http://192.168.0.203:1234
- **Model**: qwen2.5-7b-instruct
- **API**: OpenAI-compatible

#### Chat Completion
```json
Request:
POST /v1/chat/completions
{
    "messages": [{
        "role": "user",
        "content": "message"
    }],
    "model": "qwen2.5-7b-instruct",
    "temperature": 0.7,
    "max_tokens": 500
}

Response:
{
    "choices": [{
        "message": {
            "content": "response",
            "role": "assistant"
        }
    }]
}
```

### WebSocket Client Integration
```javascript
// Tensor Server Connection
tensorServer = new WebSocket('ws://192.168.0.203:5000')

// HPC Server Connection
hpcServer = new WebSocket('ws://192.168.0.203:5004')

// Message Handling
tensorServer.onmessage = async (event) => {
    const data = JSON.parse(event.data)
    // Handle different response types
    if (data.type === 'embeddings') {...}
    else if (data.type === 'search_results') {...}
    else if (data.type === 'stats') {...}
}
```

## 4. Container Configuration

### nemo_sig_v3 Container
- Base Image: CUDA-enabled Python
- Working Directory: /workspace/project
- Required Ports: 5000, 5004
- GPU Access: Required
- Memory Limit: None (runtime dependent)

### Environment Requirements
- CUDA 11.x+
- Python 3.8+
- PyTorch with CUDA
- sentence-transformers
- websockets

## 5. Performance Characteristics

### Memory Usage
- Embedding Size: 384 * 4 bytes
- Momentum Buffer: 384 * 4 * chunk_size bytes
- Memory Cache: Variable (RAM-based)

### GPU Requirements
- Minimum VRAM: 2GB
- Recommended VRAM: 4GB+
- Compute Capability: 3.5+

### Processing Times
- Embedding Generation: ~50ms
- Memory Search: ~10ms per 1000 memories
- Significance Calculation: ~5ms
- Total Processing: ~100ms typical

## 6. System Limitations

### Current Constraints
1. Memory Storage
   - RAM-based storage
   - No persistence
   - Single node only

2. Processing
   - Fixed embedding size
   - Single GPU support
   - Synchronous processing

3. Integration
   - Local network only
   - No authentication
   - Limited error recovery

### Known Issues
1. Memory leaks in long-running sessions
2. GPU memory fragmentation
3. WebSocket timeout handling
4. Connection recovery
5. Error propagation

## 7. Monitoring

### Available Metrics
1. Memory Stats
   - Memory count
   - GPU memory usage
   - Cache size
   
2. Processing Stats
   - Embedding times
   - Search latency
   - Significance scores
   
3. System Stats
   - GPU status
   - Connection count
   - Error rates

### Logging
- Location: Container stdout/stderr
- Format: JSON structured logging
- Level: INFO (configurable)
- Components: All servers and managers