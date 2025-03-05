import asyncio
import json
import websockets
from typing import Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    type: str
    data: Dict[str, Any]
    client_id: str

class WebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.handlers: Dict[str, Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]] = {}
        self.server: Optional[websockets.WebSocketServer] = None

    def register_handler(self, message_type: str, handler: Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]):
        """Register a handler for a specific message type"""
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        client_id = str(id(websocket))
        self.clients[client_id] = websocket
        logger.info(f"New client connected. ID: {client_id}")

        try:
            # Send initial connection success message
            await websocket.send(json.dumps({
                "type": "connection_status",
                "status": "connected",
                "client_id": client_id
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type in self.handlers:
                        # Create message object
                        ws_message = WebSocketMessage(
                            type=msg_type,
                            data=data,
                            client_id=client_id
                        )
                        
                        try:
                            # Call appropriate handler
                            response = await self.handlers[msg_type](ws_message)
                            
                            # Ensure response is JSON serializable
                            json.dumps(response)  # Test serialization
                            
                            # Send response back to client
                            await websocket.send(json.dumps(response))
                        except Exception as e:
                            logger.error(f"Handler error: {str(e)}")
                            await websocket.send(json.dumps({
                                "type": "error",
                                "error": f"Handler error: {str(e)}"
                            }))
                    else:
                        logger.warning(f"No handler for message type: {msg_type}")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": f"Unsupported message type: {msg_type}"
                        }))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON format"
                    }))
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Client connection closed: {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    try:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
                    except:
                        pass

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected. ID: {client_id}")
        except Exception as e:
            logger.error(f"Unexpected error in client handler: {str(e)}")
        finally:
            # Clean up client connection
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Cleaned up client: {client_id}")
                
                # Call cleanup on voice handler if available
                try:
                    from voice_core.voice_handler import voice_handler
                    await voice_handler.cleanup_session(client_id)
                except Exception as e:
                    logger.error(f"Error cleaning up voice session: {str(e)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.clients.items():
            try:
                await websocket.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.clients:
                del self.clients[client_id]

    async def start(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await self.server.wait_closed()
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            raise

    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            # Close all client connections
            close_tasks = []
            for websocket in self.clients.values():
                try:
                    close_tasks.append(websocket.close())
                except:
                    pass
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

# Example handlers for voice and memory operations
async def handle_voice_input(message: WebSocketMessage) -> Dict[str, Any]:
    """Handle voice input messages"""
    text = message.data.get('text', '')
    logger.info(f"Received voice input from client {message.client_id}: {text}")
    # Process voice input here
    return {
        "type": "voice_response",
        "text": f"Processed voice input: {text}"
    }

async def handle_memory_operation(message: WebSocketMessage) -> Dict[str, Any]:
    """Handle memory operation messages"""
    operation = message.data.get('operation')
    content = message.data.get('content', '')
    logger.info(f"Received memory operation from client {message.client_id}: {operation}")
    # Process memory operation here
    return {
        "type": "memory_response",
        "operation": operation,
        "status": "success"
    }

# Example usage
if __name__ == "__main__":
    server = WebSocketServer()
    
    # Register handlers
    server.register_handler("voice_input", handle_voice_input)
    server.register_handler("memory_operation", handle_memory_operation)
    
    # Run the server
    asyncio.run(server.start())
