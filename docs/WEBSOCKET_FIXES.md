# WebSocket Connection Fixes

## Current Issues

1. Server Configuration:
   - Binding to '0.0.0.0' instead of 'localhost'
   - Missing CORS headers
   - No connection timeouts
   - Incorrect port configuration

2. Client Code:
   - Wrong port numbers
   - Missing error handling
   - No reconnection logic

## Required Changes

### 1. Server Changes

#### Memory Server (tensor_server.py)
```python
# Change port binding
host = 'localhost'  # Instead of '0.0.0.0'
port = 5000        # Fixed port

# Add CORS headers
async def process_request(path, headers):
    if 'origin' in headers:
        return None  # Allow WebSocket upgrade
    return http.HTTPStatus.OK, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': '*'
    }, b''

# Add timeouts
websockets.serve(
    handler,
    host,
    port,
    ping_interval=20,
    ping_timeout=20,
    close_timeout=10,
    process_request=process_request
)
```

#### Inference Server (inference_server.py)
```python
# Use dynamic port finding
port = 5005
while port < 5015:
    try:
        # Same configuration as memory server
        await websockets.serve(...)
        break
    except OSError:
        port += 1
```

### 2. Client Changes

#### memory-interface.js
```javascript
class MemoryInterface {
    constructor() {
        this.config = {
            serverUrl: 'ws://localhost:5000',
            reconnectInterval: 5000,
            maxRetries: 5
        };
        this.connect();
    }

    connect() {
        this.socket = new WebSocket(this.config.serverUrl);
        this.socket.onopen = this.handleOpen.bind(this);
        this.socket.onclose = this.handleClose.bind(this);
        this.socket.onerror = this.handleError.bind(this);
        this.socket.onmessage = this.handleMessage.bind(this);
    }

    handleOpen() {
        console.log('Memory server connected');
        this.sendStats();  // Initial stats request
    }

    handleClose() {
        if (this.shouldReconnect) {
            setTimeout(() => this.connect(), this.config.reconnectInterval);
        }
    }
}
```

#### lora-interface.js
```javascript
class LoRAInterface {
    constructor() {
        this.config = {
            serverUrl: 'ws://localhost:5005',
            reconnectInterval: 5000,
            maxRetries: 5
        };
        this.connect();
    }

    // Similar methods as MemoryInterface
}
```

### 3. Message Format Standardization

```javascript
// Client to Server
{
    type: string,      // Command type
    timestamp: string, // ISO timestamp
    data?: {          // Optional data
        [key: string]: any
    }
}

// Server to Client
{
    status: string,    // 'success' or 'error'
    data?: any,       // Response data
    error?: string    // Error message if status is 'error'
}
```

## Implementation Steps

1. Update Server Code:
   - Apply CORS changes
   - Update port configurations
   - Add timeout settings
   - Implement proper error handling

2. Update Client Code:
   - Fix server URLs
   - Add reconnection logic
   - Implement proper error handling
   - Add message queuing for disconnects

3. Test Connections:
   ```bash
   # Start servers
   python tensor_server.py    # Should show: listening on localhost:5000
   python inference_server.py # Should show: listening on localhost:5005
   ```

4. Verify with test page:
   - Open TEST_WEBSOCKET_CONNECTIONS.md test page
   - Check for successful connections
   - Verify message sending/receiving
   - Test error handling

## Security Considerations

1. Use localhost instead of 0.0.0.0
2. Implement proper CORS
3. Add connection timeouts
4. Validate all messages
5. Handle reconnections gracefully

## Monitoring

1. Check server logs for:
   - Connection attempts
   - Message processing
   - Error occurrences
   - Memory usage

2. Client monitoring:
   - Connection status
   - Message latency
   - Error rates
   - Reconnection attempts

## Next Steps

1. Implement changes in server code
2. Update client interfaces
3. Run connection tests
4. Monitor performance
5. Document any issues