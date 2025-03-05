# Titan Memory Server Integration

## Overview

This document details the integration and deployment of the MCP Titan Memory Server, which provides advanced memory management and HPC capabilities for the chat application.

## Architecture

### Components

1. **Titan Memory Server**
   - Memory chunk management
   - Hypersphere token processing
   - Real-time metrics collection
   - HPC flow optimization

2. **Integration Points**
   - MCP Protocol handlers
   - Memory state management
   - Metric collection endpoints
   - WebSocket connections

## Setup and Configuration

### 1. Memory Server Installation

```bash
# Clone Titan Memory repository
git clone [titan-memory-repo]
cd titan-memory

# Install dependencies
npm install

# Build the server
npm run build
```

### 2. Environment Configuration

```env
# Titan Memory Server Settings
TITAN_MEMORY_PORT=3001
TITAN_MEMORY_HOST=localhost
TITAN_MEMORY_WEBSOCKET_PORT=3002

# Memory Management
TITAN_MAX_MEMORY_CHUNKS=1000
TITAN_CHUNK_SIZE_MB=64
TITAN_GC_INTERVAL=300000

# HPC Settings
TITAN_HPC_ENABLED=true
TITAN_GPU_MEMORY_LIMIT_MB=4096
TITAN_TENSOR_POOLING=true
```

## Memory Management

### 1. Chunk Processing

```typescript
// Configure chunk processing
const chunkConfig = {
  maxChunks: process.env.TITAN_MAX_MEMORY_CHUNKS,
  chunkSizeMB: process.env.TITAN_CHUNK_SIZE_MB,
  gcInterval: process.env.TITAN_GC_INTERVAL
};

// Initialize memory manager
const memoryManager = new TitanMemoryManager(chunkConfig);
```

### 2. Memory States

```typescript
interface MemoryState {
  chunks: MemoryChunk[];
  metrics: {
    totalAllocated: number;
    activeChunks: number;
    gcCycles: number;
  };
  status: 'active' | 'gc' | 'error';
}

// Monitor memory state
memoryManager.on('stateChange', (state: MemoryState) => {
  metrics.recordMemoryState(state);
});
```

## MCP Integration

### 1. Server Setup

```typescript
import { MCPServer } from '@modelcontextprotocol/sdk';

const server = new MCPServer({
  name: 'titan-memory',
  version: '1.0.0'
});

// Register memory management tools
server.registerTool('allocateMemory', {
  handler: async (params) => {
    return await memoryManager.allocate(params);
  }
});

server.registerTool('processChunk', {
  handler: async (params) => {
    return await memoryManager.processChunk(params);
  }
});
```

### 2. Resource Exposure

```typescript
// Expose memory metrics as MCP resource
server.exposeResource('memory-metrics', {
  uri: 'titan://memory/metrics',
  handler: () => memoryManager.getMetrics()
});

// Expose chunk status
server.exposeResource('chunk-status', {
  uri: 'titan://memory/chunks',
  handler: () => memoryManager.getChunkStatus()
});
```

## Deployment

### 1. Production Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  titan-memory:
    build: .
    environment:
      - NODE_ENV=production
      - TITAN_MEMORY_PORT=3001
    volumes:
      - memory-data:/var/lib/titan/memory
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 2. Kubernetes Deployment

```yaml
# k8s/titan-memory.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: titan-memory
spec:
  replicas: 1
  selector:
    matchLabels:
      app: titan-memory
  template:
    spec:
      containers:
      - name: titan-memory
        image: titan-memory:latest
        resources:
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: memory-storage
          mountPath: /var/lib/titan/memory
```

## Monitoring

### 1. Memory Metrics

```typescript
// Set up Prometheus metrics
const metrics = {
  memoryUsage: new Gauge({
    name: 'titan_memory_usage_bytes',
    help: 'Current memory usage in bytes'
  }),
  
  chunkCount: new Gauge({
    name: 'titan_memory_chunks_total',
    help: 'Total number of memory chunks'
  }),
  
  gcLatency: new Histogram({
    name: 'titan_memory_gc_duration_seconds',
    help: 'Garbage collection duration'
  })
};
```

### 2. Health Checks

```typescript
// Health check endpoint
app.get('/health', (req, res) => {
  const health = {
    status: memoryManager.getStatus(),
    metrics: memoryManager.getMetrics(),
    lastGC: memoryManager.getLastGCTime()
  };
  
  res.status(200).json(health);
});
```

## Memory Files

The Titan Memory Server stores memory chunks in JSON files with timestamps:

```
test_memory/
├── memory_[timestamp]_[id].json  # Individual memory chunks
└── memory_index.json            # Memory chunk index
```

### Memory File Format

```json
{
  "id": "memory_2025-01-27T00-04-03-288Z_zvfhaj97x",
  "timestamp": "2025-01-27T00:04:03.288Z",
  "size": 1048576,
  "type": "embedding",
  "data": {
    "embeddings": [...],
    "metadata": {
      "model": "gpt-4o-mini",
      "dimensions": 1536
    }
  }
}
```

## Best Practices

1. **Memory Management**
   - Monitor memory usage regularly
   - Configure GC intervals appropriately
   - Set proper memory limits
   - Use chunk pooling when possible

2. **Performance**
   - Enable GPU acceleration when available
   - Use appropriate chunk sizes
   - Implement proper error handling
   - Monitor GC impact

3. **Deployment**
   - Use StatefulSets for persistence
   - Configure resource limits
   - Implement proper backup strategy
   - Monitor system metrics

4. **Integration**
   - Use MCP protocol for communication
   - Implement proper error handling
   - Monitor connection status
   - Handle reconnection gracefully