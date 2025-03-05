# WebSocket Server Implementations

## tensor_server.py WebSocket Handler

```python
# Inside TensorServer class

async def handle_websocket(self, websocket: 'websockets.WebSocketServerProtocol') -> None:
    client_id = id(websocket)
    logger.info(f"New WebSocket connection established: {client_id} from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command_type = data.get('type')
                
                if command_type == 'store':
                    await self._handle_store(websocket, data)
                elif command_type == 'retrieve':
                    await self._handle_retrieve(websocket, data)
                elif command_type == 'stats':
                    await self._handle_stats(websocket)
                else:
                    await self._send_error(websocket, f'Unknown command type: {command_type}')
            
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await self._send_error(websocket, str(e))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}")
    finally:
        logger.info(f"Connection cleanup: {client_id}")
        self.cleanup_gpu()

async def start(self) -> None:
    """Start server with CORS support."""
    self.running = True

    async def process_request(path: str, headers: Headers) -> Optional[Tuple[http.HTTPStatus, Headers, bytes]]:
        if 'origin' in headers:
            response_headers = Headers([
                ('Access-Control-Allow-Origin', headers['origin']),
                ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                ('Access-Control-Allow-Headers', '*'),
                ('Access-Control-Allow-Credentials', 'true'),
            ])
            if path == '/ws':  # WebSocket endpoint
                return None
            return http.HTTPStatus.OK, response_headers, b''
        return None

    async with websockets.serve(
        self.handle_websocket,
        'localhost',  # Changed from '0.0.0.0' to 'localhost' for security
        self.port,
        process_request=process_request,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10
    ):
        logger.info(f'WebSocket server listening on localhost:{self.port}')
        await asyncio.Future()  # run forever
```

## inference_server.py WebSocket Handler

```python
# Inside InferenceServer class

async def handle_websocket(self, websocket: 'websockets.WebSocketServerProtocol') -> None:
    client_id = id(websocket)
    logger.info(f"New WebSocket connection established: {client_id} from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                command_type = data.get('type')
                
                if command_type == 'embeddings':
                    await self._handle_embeddings(websocket, data)
                elif command_type == 'process':
                    await self._handle_process(websocket, data)
                elif command_type == 'stats':
                    await self._handle_stats(websocket)
                else:
                    await self._send_error(websocket, f'Unknown command type: {command_type}')
            
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await self._send_error(websocket, str(e))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}")
    finally:
        logger.info(f"Connection cleanup: {client_id}")

async def start(self) -> None:
    """Start server with CORS support."""
    self.running = True

    async def process_request(path: str, headers: Headers) -> Optional[Tuple[http.HTTPStatus, Headers, bytes]]:
        if 'origin' in headers:
            response_headers = Headers([
                ('Access-Control-Allow-Origin', headers['origin']),
                ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                ('Access-Control-Allow-Headers', '*'),
                ('Access-Control-Allow-Credentials', 'true'),
            ])
            if path == '/ws':  # WebSocket endpoint
                return None
            return http.HTTPStatus.OK, response_headers, b''
        return None

    # Find available port starting from 5005
    port = 5005
    while port < 5015:  # Try up to port 5014
        try:
            async with websockets.serve(
                self.handle_websocket,
                'localhost',  # Changed from '0.0.0.0' to 'localhost' for security
                port,
                process_request=process_request,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            ):
                logger.info(f'Found available port: {port}')
                logger.info(f'WebSocket server listening on localhost:{port}')
                await asyncio.Future()  # run forever
                break
        except OSError:
            port += 1
    else:
        raise RuntimeError("No available ports found")
```

## Key Changes Made

1. CORS Headers:
   - Added proper CORS headers in process_request
   - Support for WebSocket upgrade requests
   - Allow credentials and required headers

2. Security:
   - Changed binding from '0.0.0.0' to 'localhost'
   - Added connection timeouts
   - Proper error handling and logging

3. Connection Management:
   - Added ping/pong for connection health
   - Proper cleanup on connection close
   - Client ID tracking for debugging

4. Port Configuration:
   - Memory server fixed on port 5000
   - Inference server tries ports 5005-5014
   - Proper port availability checking

5. Message Processing:
   - Structured command handling
   - JSON validation
   - Error response formatting

## Message Format Examples

### Memory Server Messages

```javascript
// Store Memory Request
{
    "type": "store",
    "timestamp": "2025-02-10T02:48:49.000Z",
    "data": {
        "text": "message content",
        "metadata": {
            "type": "user",
            "context": {}
        }
    }
}

// Retrieve Memory Request
{
    "type": "retrieve",
    "timestamp": "2025-02-10T02:48:49.000Z",
    "data": {
        "count": 5,
        "min_significance": 0.7
    }
}
```

### Inference Server Messages

```javascript
// Embeddings Request
{
    "type": "embeddings",
    "timestamp": "2025-02-10T02:48:49.000Z",
    "data": {
        "text": "content to embed",
        "config": {}
    }
}

// Process Request
{
    "type": "process",
    "timestamp": "2025-02-10T02:48:49.000Z",
    "data": {
        "text": "input text",
        "context": []
    }
}
```

## Testing the Changes

1. Start servers with new configuration:
```bash
# Memory Server
python tensor_server.py

# Inference Server
python inference_server.py
```

2. Check server logs for:
- Correct port bindings
- CORS headers in responses
- Connection establishments

3. Use the test page from TEST_WEBSOCKET_CONNECTIONS.md to verify:
- Connection establishment
- Message sending/receiving
- Error handling
- Connection cleanup