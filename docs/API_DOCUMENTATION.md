# API Documentation
*Titan Memory Server with GPU Acceleration*

**Version:** 1.0.0  
**Authors:** MEGA, JASON, KEG  
**Last Updated:** January 29, 2025

## Table of Contents
1. [Overview](#overview)
2. [WebSocket API](#websocket-api)
3. [JSON-RPC Methods](#json-rpc-methods)
4. [Data Structures](#data-structures)
5. [Error Handling](#error-handling)
6. [Examples](#examples)

## Overview

The Titan Memory Server exposes its functionality through a WebSocket interface using the JSON-RPC 2.0 protocol. All operations are asynchronous and support both request-response and notification patterns.

### Connection Details
- **URL:** `ws://localhost:3001`
- **Protocol:** `mcp`
- **Format:** JSON-RPC 2.0

## WebSocket API

### Connection Setup

```javascript
const client = new WebSocket('ws://localhost:3001', 'mcp');

client.onopen = () => {
    // Connection established
    console.log('Connected to server');
};

client.onerror = (error) => {
    console.error('WebSocket error:', error);
};

client.onclose = () => {
    console.log('Connection closed');
};
```

### Message Format

All messages must follow the JSON-RPC 2.0 specification:

```typescript
interface JsonRpcRequest {
    jsonrpc: "2.0";
    method: string;
    params?: any;
    id?: number | string;
}

interface JsonRpcResponse {
    jsonrpc: "2.0";
    result?: any;
    error?: {
        code: number;
        message: string;
        data?: any;
    };
    id: number | string | null;
}
```

## JSON-RPC Methods

### 1. Initialize Connection

```typescript
// Request
{
    "jsonrpc": "2.0",
    "method": "initialize",
    "id": 1
}

// Response
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "titan-memory",
            "version": "1.0.0"
        },
        "capabilities": {
            "tools": {
                // Available tools...
            }
        }
    }
}
```

### 2. Process Embeddings

```typescript
// Request
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "processEmbeddings",
        "arguments": {
            "embeddings": Float32Array,
            "chatId": string,
            "options": {
                "dimension_reduction": number,
                "quantization": boolean,
                "quantization_bits": number
            }
        }
    },
    "id": 2
}

// Response
{
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "content": [{
            "type": "text",
            "text": JSON.stringify({
                "processed": Float32Array,
                "metadata": {
                    "processing_time": number,
                    "dimensions": number,
                    "device": string
                }
            })
        }]
    }
}
```

### 3. Allocate Memory

```typescript
// Request
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "allocateMemory",
        "arguments": {
            "size": number,
            "chatId": string
        }
    },
    "id": 3
}

// Response
{
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "content": [{
            "type": "text",
            "text": JSON.stringify({
                "chunkId": string,
                "size": number
            })
        }]
    }
}
```

### 4. Get Metrics

```typescript
// Request
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "getMetrics",
        "arguments": {}
    },
    "id": 4
}

// Response
{
    "jsonrpc": "2.0",
    "id": 4,
    "result": {
        "content": [{
            "type": "text",
            "text": JSON.stringify({
                "memory": {
                    "chunks": number,
                    "usage": number,
                    "available": number
                },
                "hpc": {
                    "gpu_memory": number,
                    "cpu_memory": number,
                    "active_processes": number
                }
            })
        }]
    }
}
```

## Data Structures

### 1. Embedding Format

```typescript
interface Embedding {
    vector: Float32Array;
    metadata?: {
        timestamp: number;
        source: string;
        dimensions: number;
    };
}
```

### 2. Processing Options

```typescript
interface ProcessingOptions {
    dimension_reduction?: number;
    quantization?: boolean;
    quantization_bits?: number;
    batch_size?: number;
    precision?: 'float32' | 'float16';
}
```

### 3. Memory Chunk

```typescript
interface MemoryChunk {
    id: string;
    size: number;
    created: number;
    lastAccessed: number;
    metadata?: Record<string, any>;
}
```

## Error Handling

### Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid Request | Not a valid JSON-RPC request |
| -32601 | Method not found | Method does not exist |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Internal JSON-RPC error |
| -32000 | Resource error | Memory allocation failed |
| -32001 | Processing error | GPU processing failed |
| -32002 | Timeout error | Operation timed out |

### Error Response Example

```typescript
{
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32000,
        "message": "GPU memory allocation failed",
        "data": {
            "requested": "2GB",
            "available": "1GB"
        }
    }
}
```

## Examples

### 1. Basic Processing Flow

```javascript
// Connect to server
const client = new WebSocket('ws://localhost:3001', 'mcp');

// Initialize connection
await send({
    jsonrpc: "2.0",
    method: "initialize",
    id: 1
});

// Process embeddings
const result = await send({
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
        name: "processEmbeddings",
        arguments: {
            embeddings: new Float32Array([...]),
            chatId: "session-123",
            options: {
                dimension_reduction: 256
            }
        }
    },
    id: 2
});
```

### 2. Memory Management

```javascript
// Allocate memory
const chunk = await send({
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
        name: "allocateMemory",
        arguments: {
            size: 1024 * 1024, // 1MB
            chatId: "session-123"
        }
    },
    id: 3
});

// Process chunk
await send({
    jsonrpc: "2.0",
    method: "tools/call",
    params: {
        name: "processChunk",
        arguments: {
            chunkId: chunk.result.content[0].text.chunkId,
            embeddings: new Float32Array([...])
        }
    },
    id: 4
});
```

### 3. Error Handling

```javascript
try {
    const result = await send({
        jsonrpc: "2.0",
        method: "tools/call",
        params: {
            name: "processEmbeddings",
            arguments: {
                embeddings: new Float32Array([...]),
                chatId: "session-123"
            }
        },
        id: 5
    });
} catch (error) {
    if (error.code === -32000) {
        // Handle resource error
        await cleanup();
    }
    throw error;
}
```

## WebSocket Helper Functions

```typescript
function send(message: JsonRpcRequest): Promise<JsonRpcResponse> {
    return new Promise((resolve, reject) => {
        const id = message.id;
        
        const handler = (event) => {
            const response = JSON.parse(event.data);
            if (response.id === id) {
                client.removeEventListener('message', handler);
                if (response.error) {
                    reject(response.error);
                } else {
                    resolve(response);
                }
            }
        };
        
        client.addEventListener('message', handler);
        client.send(JSON.stringify(message));
    });
}
```

## Rate Limiting

- Maximum connections: 1000 per server
- Request timeout: 30 seconds
- Maximum message size: 100MB
- Rate limit: 1000 requests per minute per client

## Security Considerations

1. **Authentication**
   - Use secure WebSocket (wss://)
   - Implement token-based authentication
   - Validate all input data

2. **Data Protection**
   - Encrypt sensitive data
   - Implement proper access controls
   - Regular security audits

## Best Practices

1. **Connection Management**
   - Implement reconnection logic
   - Handle connection timeouts
   - Clean up resources on disconnect

2. **Error Handling**
   - Implement proper error recovery
   - Log all errors
   - Provide meaningful error messages

3. **Performance**
   - Batch operations when possible
   - Monitor memory usage
   - Implement proper cleanup

## Support

For API support and questions:
- Documentation: [docs/](./docs/)
- Issue Tracker: [GitHub Issues](https://github.com/your-org/titan-memory-server/issues)
- Email: support@example.com