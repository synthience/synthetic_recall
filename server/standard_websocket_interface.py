"""
LUCID RECALL PROJECT
Standardized WebSocket Communication Interface

A unified WebSocket communication interface to standardize
the communication protocol across all memory servers.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Set
import websockets
from dataclasses import dataclass
from enum import Enum, auto

# Configure logging
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Standard message types for WebSocket communication."""
    # Core message types
    CONNECT = auto()
    DISCONNECT = auto()
    ERROR = auto()
    HEARTBEAT = auto()
    
    # Memory operations
    EMBED = auto()
    PROCESS = auto()
    STORE = auto()
    SEARCH = auto()
    UPDATE = auto()
    DELETE = auto()
    
    # Status and control
    STATS = auto()
    CONFIG = auto()
    RESET = auto()
    
    # System messages
    NOTIFICATION = auto()
    LOG = auto()
    WARNING = auto()

@dataclass
class WebSocketMessage:
    """Standard message structure for WebSocket communication."""
    type: Union[MessageType, str]
    data: Dict[str, Any]
    client_id: str
    message_id: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Set defaults for message_id and timestamp if not provided."""
        if not self.message_id:
            self.message_id = f"{int(time.time() * 1000)}-{id(self):x}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        
        # Convert string type to MessageType if needed
        if isinstance(self.type, str):
            try:
                self.type = MessageType[self.type.upper()]
            except KeyError:
                # Keep as string if not a known MessageType
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        type_value = self.type.name if isinstance(self.type, MessageType) else str(self.type)
        return {
            "type": type_value,
            "data": self.data,
            "client_id": self.client_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create message from dictionary."""
        return cls(
            type=data.get("type", "UNKNOWN"),
            data=data.get("data", {}),
            client_id=data.get("client_id", "unknown"),
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", time.time())
        )

class StandardWebSocketInterface:
    """
    Standardized WebSocket server interface.
    
    This class provides a unified interface for WebSocket communication
    across all memory system components, ensuring consistent message
    handling, error recovery, and client management.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, ping_interval: int = 20):
        """
        Initialize the WebSocket interface.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            ping_interval: Interval in seconds for sending ping messages
        """
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_interval * 2
        self.close_timeout = 10
        
        # Client management
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        
        # Message handlers
        self.handlers: Dict[Union[MessageType, str], List[Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]]] = {}
        self.default_handler: Optional[Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]] = None
        
        # Middleware
        self.middleware: List[Callable[[WebSocketMessage], Awaitable[Optional[WebSocketMessage]]]] = []
        
        # Server state
        self.server: Optional[websockets.WebSocketServer] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Connection tracking
        self.connection_count = 0
        self.message_count = 0
        self.error_count = 0
        self.last_error = None
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Performance tracking
        self.start_time = 0
        self.processing_times: List[float] = []
        self.max_tracking_samples = 100
    
    def register_handler(self, 
                         message_type: Union[MessageType, str], 
                         handler: Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function that processes the message and returns a response
        """
        if isinstance(message_type, str):
            try:
                message_type = MessageType[message_type.upper()]
            except KeyError:
                # Keep as string if not a known MessageType
                pass
            
        if message_type not in self.handlers:
            self.handlers[message_type] = []
            
        self.handlers[message_type].append(handler)
        logger.info(f"Registered handler for message type: {message_type}")
    
    def register_default_handler(self, handler: Callable[[WebSocketMessage], Awaitable[Dict[str, Any]]]) -> None:
        """
        Register a default handler for unrecognized message types.
        
        Args:
            handler: Async function that processes the message and returns a response
        """
        self.default_handler = handler
        logger.info("Registered default message handler")
    
    def register_middleware(self, middleware: Callable[[WebSocketMessage], Awaitable[Optional[WebSocketMessage]]]) -> None:
        """
        Register middleware for preprocessing messages.
        
        Middleware can modify messages or prevent processing by returning None.
        
        Args:
            middleware: Async function that processes and potentially modifies the message
        """
        self.middleware.append(middleware)
        logger.info(f"Registered middleware (total: {len(self.middleware)})")
    
    async def _apply_middleware(self, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """
        Apply all middleware to a message.
        
        Args:
            message: Incoming message
            
        Returns:
            Processed message or None if processing should be aborted
        """
        processed_message = message
        
        for mw in self.middleware:
            try:
                processed_message = await mw(processed_message)
                if processed_message is None:
                    # Middleware requested to abort processing
                    return None
            except Exception as e:
                logger.error(f"Error in middleware: {e}")
                # Continue processing with original message
                
        return processed_message
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("Server is already running")
            return
            
        try:
            self.start_time = time.time()
            self._running = True
            self._shutdown_event.clear()
            
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=self.close_timeout
            )
            
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Keep running until stopped
            await self._shutdown_event.wait()
            
        except Exception as e:
            self._running = False
            logger.error(f"Error starting server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            logger.warning("Server is not running")
            return
            
        logger.info("Stopping WebSocket server...")
        self._running = False
        
        # Close all client connections
        close_tasks = []
        for client_id, websocket in list(self.clients.items()):
            try:
                close_tasks.append(self._close_client(client_id, websocket, 1001, "Server shutting down"))
            except Exception as e:
                logger.error(f"Error closing client {client_id}: {e}")
                
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
        # Cancel all active tasks
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            self.active_tasks.clear()
            
        # Close the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            
        # Signal shutdown completion
        self._shutdown_event.set()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """
        Handle a client connection.
        
        Args:
            websocket: WebSocket connection
            path: Request path
        """
        client_id = self._generate_client_id(websocket)
        self.clients[client_id] = websocket
        self.client_info[client_id] = {
            "connect_time": time.time(),
            "path": path,
            "remote": websocket.remote_address,
            "message_count": 0,
            "error_count": 0
        }
        
        self.connection_count += 1
        logger.info(f"Client connected: {client_id} from {websocket.remote_address}")
        
        # Send welcome message
        try:
            welcome_msg = {
                "type": "CONNECT",
                "data": {
                    "client_id": client_id,
                    "server_time": time.time()
                },
                "client_id": client_id,
                "message_id": f"welcome-{client_id}",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(welcome_msg))
        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")
        
        try:
            async for message_text in websocket:
                # Track in active tasks
                task = asyncio.create_task(self._process_message(client_id, websocket, message_text))
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
                
                # Allow other tasks to run
                await asyncio.sleep(0)
                
        except websockets.ConnectionClosed:
            logger.info(f"Client disconnected normally: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            self.error_count += 1
            self.last_error = str(e)
        finally:
            # Clean up client
            await self._close_client(client_id, websocket, 1000, "Connection closed")
    
    async def _process_message(self, 
                              client_id: str, 
                              websocket: websockets.WebSocketServerProtocol, 
                              message_text: str) -> None:
        """
        Process a message from a client.
        
        Args:
            client_id: Client identifier
            websocket: WebSocket connection
            message_text: Raw message text
        """
        start_time = time.time()
        self.message_count += 1
        self.client_info[client_id]["message_count"] += 1
        
        try:
            # Parse message
            try:
                message_data = json.loads(message_text)
                message = WebSocketMessage.from_dict({
                    **message_data,
                    "client_id": client_id  # Ensure client_id is set
                })
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}")
                error_response = {
                    "type": "ERROR",
                    "data": {"error": "Invalid JSON format"},
                    "client_id": client_id,
                    "message_id": f"error-{time.time()}",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(error_response))
                self.error_count += 1
                self.client_info[client_id]["error_count"] += 1
                return
                
            # Apply middleware
            processed_message = await self._apply_middleware(message)
            if processed_message is None:
                # Middleware requested to abort processing
                return
                
            message = processed_message
            
            # Find handler for message type
            message_type = message.type
            handlers = self.handlers.get(message_type, [])
            
            if not handlers and self.default_handler:
                # Use default handler if no specific handlers found
                handlers = [self.default_handler]
                
            if not handlers:
                logger.warning(f"No handler for message type: {message_type}")
                error_response = {
                    "type": "ERROR",
                    "data": {"error": f"Unsupported message type: {message_type}"},
                    "client_id": client_id,
                    "message_id": f"error-{time.time()}",
                    "timestamp": time.time(),
                    "refers_to": message.message_id
                }
                await websocket.send(json.dumps(error_response))
                return
                
            # Call all handlers
            responses = []
            for handler in handlers:
                try:
                    response = await handler(message)
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error in handler for {message_type}: {e}")
                    error_response = {
                        "type": "ERROR",
                        "data": {"error": f"Handler error: {str(e)}"},
                        "client_id": client_id,
                        "message_id": f"error-{time.time()}",
                        "timestamp": time.time(),
                        "refers_to": message.message_id
                    }
                    responses.append(error_response)
                    self.error_count += 1
                    self.client_info[client_id]["error_count"] += 1
            
            # Send responses
            for response in responses:
                if not response:
                    continue
                    
                # Ensure response has all required fields
                if "type" not in response:
                    response["type"] = "RESPONSE"
                if "client_id" not in response:
                    response["client_id"] = client_id
                if "timestamp" not in response:
                    response["timestamp"] = time.time()
                if "message_id" not in response:
                    response["message_id"] = f"resp-{time.time()}"
                if "refers_to" not in response and message.message_id:
                    response["refers_to"] = message.message_id
                    
                try:
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    logger.error(f"Error sending response: {e}")
            
            # Update performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_tracking_samples:
                self.processing_times = self.processing_times[-self.max_tracking_samples:]
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            try:
                error_response = {
                    "type": "ERROR",
                    "data": {"error": f"Server error: {str(e)}"},
                    "client_id": client_id,
                    "message_id": f"error-{time.time()}",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(error_response))
            except:
                pass
            self.error_count += 1
            self.last_error = str(e)
    
    async def _close_client(self, 
                           client_id: str, 
                           websocket: websockets.WebSocketServerProtocol,
                           code: int = 1000,
                           reason: str = "Normal closure") -> None:
        """
        Close a client connection and clean up resources.
        
        Args:
            client_id: Client identifier
            websocket: WebSocket connection
            code: Close code
            reason: Close reason
        """
        try:
            # Remove from active clients
            self.clients.pop(client_id, None)
            
            # Send close message if possible
            if not websocket.closed:
                close_msg = {
                    "type": "DISCONNECT",
                    "data": {"reason": reason},
                    "client_id": client_id,
                    "message_id": f"close-{client_id}",
                    "timestamp": time.time()
                }
                try:
                    await websocket.send(json.dumps(close_msg))
                except:
                    pass
                
                # Close the connection
                await websocket.close(code, reason)
            
            # Log disconnection
            connect_time = self.client_info.get(client_id, {}).get("connect_time", 0)
            connection_duration = time.time() - connect_time if connect_time else 0
            logger.info(f"Client disconnected: {client_id} (duration: {connection_duration:.2f}s)")
            
            # Keep client info for stats
            self.client_info[client_id]["disconnect_time"] = time.time()
            self.client_info[client_id]["connection_duration"] = connection_duration
            
            # Additional cleanup if needed
            # Call any registered disconnect handlers
            disconnect_msg = WebSocketMessage(
                type=MessageType.DISCONNECT,
                data={"client_id": client_id, "reason": reason},
                client_id=client_id
            )
            
            handlers = self.handlers.get(MessageType.DISCONNECT, [])
            for handler in handlers:
                try:
                    await handler(disconnect_msg)
                except Exception as e:
                    logger.error(f"Error in disconnect handler: {e}")
            
        except Exception as e:
            logger.error(f"Error closing client {client_id}: {e}")
    
    async def broadcast(self, message: Dict[str, Any], exclude_clients: Optional[List[str]] = None) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
            exclude_clients: List of client IDs to exclude from broadcast
        """
        if not self.clients:
            return
            
        exclude_clients = exclude_clients or []
        
        # Ensure message has required fields
        if "type" not in message:
            message["type"] = "BROADCAST"
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        if "message_id" not in message:
            message["message_id"] = f"broadcast-{time.time()}"
            
        message_text = json.dumps(message)
        send_tasks = []
        
        for client_id, websocket in list(self.clients.items()):
            if client_id in exclude_clients:
                continue
                
            # Set client_id for each recipient
            client_message = json.loads(message_text)
            client_message["client_id"] = client_id
            
            send_tasks.append(self._safe_send(websocket, json.dumps(client_message)))
            
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
    
    async def _safe_send(self, websocket: websockets.WebSocketServerProtocol, message: str) -> bool:
        """
        Safely send a message to a client.
        
        Args:
            websocket: WebSocket connection
            message: Message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not websocket.closed:
                await websocket.send(message)
                return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
        return False
    
    def _generate_client_id(self, websocket: websockets.WebSocketServerProtocol) -> str:
        """
        Generate a unique client ID.
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Unique client ID
        """
        remote = websocket.remote_address or ('unknown', 0)
        timestamp = int(time.time() * 1000)
        return f"{remote[0]}-{remote[1]}-{timestamp}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = time.time() - self.start_time if self.start_time else 0
        avg_processing_time = sum(self.processing_times) / max(len(self.processing_times), 1)
        
        return {
            "running": self._running,
            "uptime": uptime,
            "connection_count": self.connection_count,
            "active_clients": len(self.clients),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "avg_processing_time": avg_processing_time,
            "handler_count": sum(len(handlers) for handlers in self.handlers.values()),
            "middleware_count": len(self.middleware)
        }