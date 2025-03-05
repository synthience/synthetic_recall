# Testing WebSocket Connections

## Test Script

Create a file `test_connections.html` with this content:

```html
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Connection Test</title>
    <style>
        .log { font-family: monospace; margin: 5px; }
        .error { color: red; }
        .success { color: green; }
        .info { color: blue; }
    </style>
</head>
<body>
    <h2>WebSocket Connection Test</h2>
    <div id="logs"></div>

    <script>
        function log(message, type = 'info') {
            const logDiv = document.createElement('div');
            logDiv.className = `log ${type}`;
            logDiv.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
            document.getElementById('logs').appendChild(logDiv);
        }

        // Test Memory Server
        function testMemoryServer() {
            log('Testing Memory Server connection...', 'info');
            
            const memorySocket = new WebSocket('ws://localhost:5000');
            
            memorySocket.onopen = () => {
                log('Memory Server connected successfully!', 'success');
                
                // Test stats request
                const statsRequest = {
                    type: 'stats',
                    timestamp: new Date().toISOString()
                };
                memorySocket.send(JSON.stringify(statsRequest));
                log('Sent stats request to Memory Server', 'info');
            };

            memorySocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                log(`Memory Server response: ${JSON.stringify(data, null, 2)}`, 'success');
            };

            memorySocket.onerror = (error) => {
                log(`Memory Server error: ${error.message}`, 'error');
            };

            memorySocket.onclose = () => {
                log('Memory Server connection closed', 'info');
            };
        }

        // Test Inference Server
        function testInferenceServer() {
            log('Testing Inference Server connection...', 'info');
            
            const inferenceSocket = new WebSocket('ws://localhost:5005');
            
            inferenceSocket.onopen = () => {
                log('Inference Server connected successfully!', 'success');
                
                // Test stats request
                const statsRequest = {
                    type: 'stats',
                    timestamp: new Date().toISOString()
                };
                inferenceSocket.send(JSON.stringify(statsRequest));
                log('Sent stats request to Inference Server', 'info');
            };

            inferenceSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                log(`Inference Server response: ${JSON.stringify(data, null, 2)}`, 'success');
            };

            inferenceSocket.onerror = (error) => {
                log(`Inference Server error: ${error.message}`, 'error');
            };

            inferenceSocket.onclose = () => {
                log('Inference Server connection closed', 'info');
            };
        }

        // Start tests
        setTimeout(testMemoryServer, 1000);
        setTimeout(testInferenceServer, 2000);
    </script>
</body>
</html>
```

## Running the Test

1. Save the above HTML file
2. Ensure both servers are running:
   ```bash
   # Terminal 1 - Memory Server
   cd managers/python/nemo_integration
   python tensor_server.py

   # Terminal 2 - Inference Server
   cd managers/python/inference
   python inference_server.py
   ```
3. Open the HTML file in a browser
4. Check the logs for:
   - Connection success messages
   - Server responses
   - Any error messages

## Expected Output

### Successful Test:
```
12:34:56 - Testing Memory Server connection...
12:34:56 - Memory Server connected successfully!
12:34:56 - Sent stats request to Memory Server
12:34:56 - Memory Server response: {
    "status": "success",
    "stats": {
        "memory_count": 0,
        "gpu": {
            "allocated": 0.123,
            "cached": 0.456,
            "max_allocated": 0.789,
            "utilization": 5
        }
    }
}
12:34:57 - Testing Inference Server connection...
12:34:57 - Inference Server connected successfully!
12:34:57 - Sent stats request to Inference Server
12:34:57 - Inference Server response: {
    "status": "success",
    "stats": {
        "model": "all-MiniLM-L6-v2",
        "device": "cuda"
    }
}
```

### Common Error Messages:

1. Connection Refused:
```
12:34:56 - Memory Server error: Connection refused
```
Solution: Ensure the server is running on the correct port

2. Invalid Frame Header:
```
12:34:56 - Memory Server error: Invalid frame header
```
Solution: Check CORS configuration in the server

3. Connection Closed:
```
12:34:56 - Memory Server connection closed
```
Solution: Check server logs for any errors that caused the disconnect

## Troubleshooting

1. If both connections fail:
   - Check if servers are running
   - Verify ports are correct
   - Check for any firewall issues

2. If one connection fails:
   - Check specific server logs
   - Verify the port for that server
   - Check server-specific configuration

3. If connections succeed but messages fail:
   - Check message format
   - Verify JSON structure
   - Check server logs for parsing errors

## Next Steps

After successful connection tests:
1. Integrate the connection code into your application
2. Add proper error handling and reconnection logic
3. Implement message queuing for reliability
4. Add connection status monitoring