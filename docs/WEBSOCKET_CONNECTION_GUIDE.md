# WebSocket Connection Guide

## Server Configuration Issues

The current WebSocket connection issues are due to:
1. Missing CORS headers in the WebSocket handshake
2. Incorrect port usage in the client code
3. Need for proper error handling in the connection process

## Required Changes

### 1. Client-Side Connection Code

The correct way to connect to the servers:

```javascript
// Memory Server Connection (Port 5000)
const memorySocket = new WebSocket('ws://localhost:5000');
memorySocket.onopen = () => {
    console.log('Memory server connected');
    // Send initial stats request
    memorySocket.send(JSON.stringify({
        type: 'stats',
        timestamp: new Date().toISOString()
    }));
};
memorySocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Memory server:', data);
    if (data.status === 'error') {
        console.error('Memory server error:', data.message);
    }
};
memorySocket.onerror = (error) => {
    console.error('Memory server error:', error);
};
memorySocket.onclose = () => {
    console.log('Memory server disconnected');
};

// Inference Server Connection (Port 5005)
const inferenceSocket = new WebSocket('ws://localhost:5005');
inferenceSocket.onopen = () => {
    console.log('Inference server connected');
    // Send initial stats request
    inferenceSocket.send(JSON.stringify({
        type: 'stats',
        timestamp: new Date().toISOString()
    }));
};
inferenceSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Inference server:', data);
    if (data.status === 'error') {
        console.error('Inference server error:', data.message);
    }
};
inferenceSocket.onerror = (error) => {
    console.error('Inference server error:', error);
};
inferenceSocket.onclose = () => {
    console.log('Inference server disconnected');
};
```

### 2. Required Server Changes

The tensor_server.py needs these modifications:
1. Add CORS headers in the process_request handler
2. Proper handling of connection upgrades
3. Better error handling for connection failures

```python
async def process_request(path: str, headers: Headers) -> Optional[Tuple[http.HTTPStatus, Headers, bytes]]:
    if 'origin' in headers:
        response_headers = Headers()
        response_headers['Access-Control-Allow-Origin'] = headers['origin']
        response_headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response_headers['Access-Control-Allow-Headers'] = 'content-type'
        return None  # Accept the connection
    return None
```

### 3. Message Format

All messages should follow this format:

```javascript
// Client to Server
{
    type: string,      // Command type ('store', 'retrieve', 'stats')
    timestamp: string, // ISO timestamp
    data?: any        // Optional data payload
}

// Server to Client
{
    status: string,   // 'success' or 'error'
    message?: string, // Error message if status is 'error'
    data?: any       // Response data if status is 'success'
}
```

### 4. Connection Testing

To test the connections:

1. Start the servers in order:
   ```bash
   # Start Memory Server (tensor_server.py)
   python tensor_server.py
   
   # Start Inference Server (inference_server.py)
   python inference_server.py
   ```

2. Check server logs for:
   ```
   WebSocket server listening on 0.0.0.0:5000
   WebSocket server listening on 0.0.0.0:5005
   ```

3. Test connections using browser console:
   ```javascript
   // Test memory server
   const testMemory = new WebSocket('ws://localhost:5000');
   testMemory.onopen = () => console.log('Memory connection successful');
   
   // Test inference server
   const testInference = new WebSocket('ws://localhost:5005');
   testInference.onopen = () => console.log('Inference connection successful');
   ```

### 5. Error Handling

Common errors and solutions:
1. "Connection refused" - Server not running or wrong port
2. "Invalid frame header" - Missing CORS headers
3. "Connection closed" - Server terminated connection

### 6. Security Considerations

1. Only accept connections from allowed origins
2. Validate all incoming messages
3. Rate limit connections and messages
4. Handle connection cleanup properly

## Implementation Notes

1. The servers must be running before attempting connections
2. Use proper error handling and reconnection logic
3. Validate message formats before sending
4. Monitor connection status and handle disconnects
5. Clean up connections when done