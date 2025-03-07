"""Memory broker for inter-agent communication."""

import asyncio
import inspect
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

# Singleton instance
_broker_instance = None
_broker_lock = asyncio.Lock()

class MemoryBroker:
    """Broker that facilitates communication between memory components and agents."""
    
    def __init__(self, ping_interval: float = 30.0):
        """Initialize the memory broker.
        
        Args:
            ping_interval: Interval in seconds for ping-pong health checks
        """
        self.callbacks = {}
        self.requests = asyncio.Queue()
        self.responses = {}
        self.running = False
        self.clients = set()
        self.tasks = {}
        self.ping_interval = ping_interval
        
    async def start(self):
        """Start the memory broker."""
        if self.running:
            return
            
        logger.info("Starting memory broker")
        self.running = True
        self.tasks["processor"] = asyncio.create_task(self._process_requests())
        self.tasks["ping"] = asyncio.create_task(self._ping_clients())
        logger.info("Memory broker started")
    
    async def stop(self):
        """Stop the memory broker."""
        if not self.running:
            return
            
        logger.info("Stopping memory broker")
        self.running = False
        
        # Cancel all tasks
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear all data
        self.callbacks = {}
        self.responses = {}
        self.clients = set()
        self.tasks = {}
        
        # Drain the queue
        while not self.requests.empty():
            try:
                self.requests.get_nowait()
                self.requests.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info("Memory broker stopped")
    
    async def register_callback(self, operation: str, callback: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        """Register a callback for a specific operation.
        
        Args:
            operation: Name of the operation
            callback: Async function to handle the operation
        """
        if not asyncio.iscoroutinefunction(callback) and not inspect.isawaitable(callback):
            raise ValueError(f"Callback for operation {operation} must be an async function")
            
        self.callbacks[operation] = callback
        logger.info(f"Registered callback for operation: {operation}")
    
    async def unregister_callback(self, operation: str):
        """Unregister a callback for a specific operation.
        
        Args:
            operation: Name of the operation
        """
        if operation in self.callbacks:
            del self.callbacks[operation]
            logger.info(f"Unregistered callback for operation: {operation}")
    
    async def register_client(self, client_id: str = None):
        """Register a client with the broker.
        
        Args:
            client_id: Optional client ID, generated if not provided
            
        Returns:
            str: The client ID
        """
        if client_id is None:
            client_id = str(uuid.uuid4())
            
        self.clients.add(client_id)
        logger.info(f"Registered client: {client_id}")
        return client_id
    
    async def unregister_client(self, client_id: str):
        """Unregister a client from the broker.
        
        Args:
            client_id: Client ID to unregister
        """
        if client_id in self.clients:
            self.clients.remove(client_id)
            logger.info(f"Unregistered client: {client_id}")
    
    async def send_request(self, operation: str, data: Dict[str, Any], client_id: str = None) -> Dict[str, Any]:
        """Send a request to the broker and wait for a response.
        
        Args:
            operation: Name of the operation
            data: Data for the operation
            client_id: Optional client ID
            
        Returns:
            Dict[str, Any]: Response data
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Register the client if not already registered
        if client_id is None:
            client_id = await self.register_client()
        elif client_id not in self.clients:
            await self.register_client(client_id)
        
        # Create a future for the response
        response_future = asyncio.get_event_loop().create_future()
        self.responses[request_id] = response_future
        
        # Prepare the request
        request = {
            "id": request_id,
            "client_id": client_id,
            "operation": operation,
            "data": data
        }
        
        # Queue the request
        await self.requests.put(request)
        
        try:
            # Wait for the response
            return await response_future
        finally:
            # Clean up
            if request_id in self.responses:
                del self.responses[request_id]
    
    async def _process_requests(self):
        """Process requests from the queue."""
        while self.running:
            try:
                # Get the next request
                request = await self.requests.get()
                
                try:
                    # Extract request details
                    request_id = request.get("id")
                    client_id = request.get("client_id")
                    operation = request.get("operation")
                    data = request.get("data", {})
                    
                    # Process the request
                    if operation in self.callbacks:
                        # Call the registered callback
                        callback = self.callbacks[operation]
                        response = await callback(data)
                    else:
                        # No callback registered for this operation
                        response = {
                            "success": False,
                            "error": f"No handler registered for operation: {operation}"
                        }
                    
                    # Set the response future
                    if request_id in self.responses:
                        self.responses[request_id].set_result(response)
                        
                except Exception as e:
                    logger.error(f"Error processing request: {e}", exc_info=True)
                    
                    # Set error response
                    if request_id in self.responses:
                        self.responses[request_id].set_result({
                            "success": False,
                            "error": str(e)
                        })
                
                finally:
                    # Mark the request as done
                    self.requests.task_done()
                    
            except asyncio.CancelledError:
                # Task was cancelled
                break
                
            except Exception as e:
                logger.error(f"Error in request processor: {e}", exc_info=True)
    
    async def _ping_clients(self):
        """Periodically ping clients to check if they're still alive."""
        while self.running:
            try:
                # Sleep for the ping interval
                await asyncio.sleep(self.ping_interval)
                
                # Ping each client (future implementation)
                # This could be used to check if clients are still responsive
                # and clean up resources for disconnected clients
                
            except asyncio.CancelledError:
                # Task was cancelled
                break
                
            except Exception as e:
                logger.error(f"Error in ping task: {e}", exc_info=True)
                # Sleep a bit to avoid tight loop on errors
                await asyncio.sleep(5)

async def get_memory_broker() -> MemoryBroker:
    """Get or create the singleton memory broker instance.
    
    Returns:
        MemoryBroker: The memory broker instance
    """
    global _broker_instance, _broker_lock
    
    async with _broker_lock:
        if _broker_instance is None or not _broker_instance.running:
            _broker_instance = MemoryBroker()
            await _broker_instance.start()
            
    return _broker_instance
