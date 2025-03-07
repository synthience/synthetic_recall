"""
LUCID RECALL PROJECT
Enhanced HPC Client

Client for interacting with the HPC server with robust connectivity
and optimized processing.
"""

import asyncio
import websockets
import json
import logging
import time
import torch
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class EnhancedHPCClient:
    """
    Enhanced client for interacting with the HPC server.
    
    Features:
    - Robust connection handling
    - Request batching
    - Result caching
    - Embedding management
    - Significance calculation
    """
    
    def __init__(self, 
                 server_url: str = "ws://localhost:5005",
                 connection_timeout: float = 10.0,
                 request_timeout: float = 30.0,
                 max_retries: int = 3,
                 ping_interval: float = 15.0,
                 embedding_dim: int = 384):
        """
        Initialize the HPC client.
        
        Args:
            server_url: WebSocket URL for the HPC server
            connection_timeout: Timeout for connection attempts
            request_timeout: Timeout for requests
            max_retries: Maximum number of retries for failed requests
            ping_interval: Interval for ping messages to keep connection alive
            embedding_dim: Dimension of embeddings
        """
        self.server_url = server_url
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.ping_interval = ping_interval
        self.embedding_dim = embedding_dim
        
        # Connection state
        self.connection = None
        self._connecting = False
        self._connection_lock = asyncio.Lock()
        self._last_activity = 0
        self._request_id = 0
        
        # Pending requests
        self._pending_requests = {}
        self._request_timeout_tasks = {}
        
        # Result cache
        self._result_cache = {}
        self._cache_max_size = 1000
        self._cache_ttl = 3600  # 1 hour
        
        # Background tasks
        self._heartbeat_task = None
        
        # Stats
        self.stats = {
            'connect_count': 0,
            'disconnect_count': 0,
            'request_count': 0,
            'error_count': 0,
            'cache_hits': 0,
            'retry_count': 0,
            'timeout_count': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"Initialized EnhancedHPCClient with server_url={server_url}")
    
    async def connect(self) -> bool:
        """
        Connect to the HPC server with robust error handling.
        
        Returns:
            Success status
        """
        # Use lock to prevent multiple concurrent connection attempts
        async with self._connection_lock:
            # Check if already connected
            if self.connection and not self.connection.closed:
                return True
                
            # Check if connection attempt is already in progress
            if self._connecting:
                logger.debug("Connection attempt already in progress, waiting...")
                for _ in range(20):  # Wait up to 2 seconds
                    await asyncio.sleep(0.1)
                    if self.connection and not self.connection.closed:
                        return True
                return False
                
            # Mark as connecting
            self._connecting = True
            
            try:
                self.stats['connect_count'] += 1
                
                # Attempt connection with timeout
                try:
                    self.connection = await asyncio.wait_for(
                        websockets.connect(self.server_url),
                        timeout=self.connection_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Connection to {self.server_url} timed out")
                    self._connecting = False
                    return False
                
                # Update activity timestamp
                self._last_activity = time.time()
                
                # Start heartbeat task if needed
                if not self._heartbeat_task or self._heartbeat_task.done():
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Start message handler
                asyncio.create_task(self._message_handler())
                
                logger.info(f"Connected to HPC server at {self.server_url}")
                self._connecting = False
                return True
                
            except Exception as e:
                logger.error(f"Error connecting to HPC server: {e}")
                self._connecting = False
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from the HPC server."""
        async with self._connection_lock:
            if self.connection and not self.connection.closed:
                try:
                    await self.connection.close()
                    self.stats['disconnect_count'] += 1
                    logger.info("Disconnected from HPC server")
                except Exception as e:
                    logger.error(f"Error disconnecting from HPC server: {e}")
            
            # Cancel heartbeat task
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            
            # Clear connection
            self.connection = None
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                
                # Check if connection is still open
                if not self.connection or self.connection.closed:
                    break
                    
                # Check inactivity
                if time.time() - self._last_activity > self.ping_interval:
                    try:
                        # Send ping to check connection
                        pong_waiter = await self.connection.ping()
                        await asyncio.wait_for(pong_waiter, timeout=5.0)
                        self._last_activity = time.time()
                        
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        logger.warning("Ping failed, reconnecting...")
                        await self.disconnect()
                        await self.connect()
                        break
                    
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
    
    async def _message_handler(self) -> None:
        """Handle incoming messages from the server."""
        if not self.connection:
            return
            
        try:
            async for message in self.connection:
                self._last_activity = time.time()
                
                # Update stats
                self.stats['bytes_received'] += len(message)
                
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Check if this is a response to a pending request
                    request_id = data.get('request_id')
                    if request_id and request_id in self._pending_requests:
                        # Get future for this request
                        future = self._pending_requests.pop(request_id)
                        
                        # Cancel timeout task if exists
                        timeout_task = self._request_timeout_tasks.pop(request_id, None)
                        if timeout_task:
                            timeout_task.cancel()
                            
                        # Set result
                        if not future.done():
                            future.set_result(data)
                            
                    else:
                        logger.warning(f"Received message for unknown request: {request_id}")
                    
                except json.JSONDecodeError:
                    logger.error("Received invalid JSON")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
            await self.disconnect()
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            await self.disconnect()
    
    async def _request_timeout_handler(self, request_id: str) -> None:
        """Handle request timeout."""
        try:
            # Wait for request timeout
            await asyncio.sleep(self.request_timeout)
            
            # Check if request is still pending
            if request_id in self._pending_requests:
                # Get future
                future = self._pending_requests.pop(request_id)
                
                # Set exception if not done
                if not future.done():
                    self.stats['timeout_count'] += 1
                    future.set_exception(asyncio.TimeoutError(f"Request {request_id} timed out"))
                    
                # Remove from pending requests
                self._request_timeout_tasks.pop(request_id, None)
                
        except asyncio.CancelledError:
            # Task was cancelled (this is expected when request completes normally)
            pass
        except Exception as e:
            logger.error(f"Error in timeout handler: {e}")
    
    async def get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor or None on failure
        """
        # Check cache
        cache_key = f"embed:{text}"
        if cache_key in self._result_cache:
            cache_entry = self._result_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                self.stats['cache_hits'] += 1
                return cache_entry['result']
        
        # Send request to get embedding
        response = await self._send_request(
            request_type='embed',
            request_data={'text': text}
        )
        
        if not response:
            return None
            
        # Extract embedding from response
        embedding_data = response.get('data', {}).get('embedding')
        if not embedding_data:
            embedding_data = response.get('embedding')
            
        if not embedding_data:
            logger.error("No embedding in response")
            return None
            
        # Convert to tensor
        try:
            embedding = torch.tensor(embedding_data, dtype=torch.float32)
            
            # Cache result
            self._result_cache[cache_key] = {
                'timestamp': time.time(),
                'result': embedding
            }
            
            # Prune cache if needed
            if len(self._result_cache) > self._cache_max_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._result_cache.keys(), 
                    key=lambda k: self._result_cache[k]['timestamp']
                )
                for key in sorted_keys[:len(sorted_keys) // 10]:  # Remove oldest 10%
                    del self._result_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            return None
    
    async def process_embedding(self, embedding: Union[torch.Tensor, List[float]]) -> Tuple[torch.Tensor, float]:
        """
        Process an embedding through the HPC pipeline.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Tuple of (processed_embedding, significance)
        """
        # Convert to list if tensor
        if isinstance(embedding, torch.Tensor):
            embedding_list = embedding.tolist()
        else:
            embedding_list = embedding
        
        # Send request to process embedding
        response = await self._send_request(
            request_type='process',
            request_data={'embeddings': embedding_list}
        )
        
        if not response:
            # Return original embedding with default significance on failure
            if isinstance(embedding, torch.Tensor):
                return embedding, 0.5
            else:
                return torch.tensor(embedding, dtype=torch.float32), 0.5
                
        # Extract processed embedding and significance
        processed_data = response.get('data', {}).get('embeddings')
        if not processed_data:
            processed_data = response.get('embeddings')
            
        significance = response.get('data', {}).get('significance')
        if significance is None:
            significance = response.get('significance', 0.5)
            
        # Convert to tensor and return
        try:
            processed_embedding = torch.tensor(processed_data, dtype=torch.float32)
            return processed_embedding, float(significance)
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            
            # Return original embedding with default significance on error
            if isinstance(embedding, torch.Tensor):
                return embedding, 0.5
            else:
                return torch.tensor(embedding, dtype=torch.float32), 0.5
    
    async def fetch_relevant_embeddings(self, query_embedding: torch.Tensor, 
                                      limit: int = 10, 
                                      min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Fetch relevant embeddings based on query embedding.
        
        Args:
            query_embedding: Query embedding
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of relevant memories
        """
        # Convert to list if tensor
        if isinstance(query_embedding, torch.Tensor):
            embedding_list = query_embedding.tolist()
        else:
            embedding_list = query_embedding
        
        # Send request to search
        response = await self._send_request(
            request_type='search',
            request_data={
                'embedding': embedding_list,
                'limit': limit,
                'min_significance': min_significance
            }
        )
        
        if not response:
            return []
            
        # Extract results
        results = response.get('data', {}).get('results')
        if not results:
            results = response.get('results', [])
            
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get HPC server stats.
        
        Returns:
            Dict with server stats
        """
        response = await self._send_request(
            request_type='stats',
            request_data={}
        )
        
        if not response:
            return {}
            
        # Extract stats
        server_stats = response.get('data', {})
        if not server_stats:
            server_stats = response
            
        # Combine with client stats
        return {
            'server': server_stats,
            'client': self.stats
        }
    
    async def _send_request(self, request_type: str, request_data: Dict[str, Any],
                          retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Send request to server with retry logic.
        
        Args:
            request_type: Type of request
            request_data: Request data
            retry_count: Current retry count
            
        Returns:
            Response dict or None on failure
        """
        # Check connection
        if not self.connection or self.connection.closed:
            success = await self.connect()
            if not success:
                logger.error("Failed to connect to HPC server")
                return None
        
        # Generate request ID
        self._request_id += 1
        request_id = f"{int(time.time())}:{self._request_id}"
        
        # Create request
        request = {
            'type': request_type,
            'request_id': request_id,
            'timestamp': time.time(),
            **request_data
        }
        
        # Serialize request
        try:
            request_json = json.dumps(request)
        except Exception as e:
            logger.error(f"Error serializing request: {e}")
            return None
        
        # Update stats
        self.stats['request_count'] += 1
        self.stats['bytes_sent'] += len(request_json)
        
        # Create future for response
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future
        
        # Create timeout task
        timeout_task = asyncio.create_task(self._request_timeout_handler(request_id))
        self._request_timeout_tasks[request_id] = timeout_task
        
        # Track start time
        start_time = time.time()
        
        try:
            # Send request
            await self.connection.send(request_json)
            self._last_activity = time.time()
            
            # Wait for response
            response = await response_future
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update average response time
            if self.stats['request_count'] > 1:
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['request_count'] - 1) + response_time) / 
                    self.stats['request_count']
                )
            else:
                self.stats['avg_response_time'] = response_time
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            
            # Retry if not exceeded max retries
            if retry_count < self.max_retries:
                self.stats['retry_count'] += 1
                logger.info(f"Retrying request (attempt {retry_count + 1}/{self.max_retries})")
                
                # Clean up
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
                    
                # Reconnect
                await self.disconnect()
                await self.connect()
                
                # Retry with increased count
                return await self._send_request(request_type, request_data, retry_count + 1)
                
            self.stats['error_count'] += 1
            return None
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed during request")
            
            # Retry if not exceeded max retries
            if retry_count < self.max_retries:
                self.stats['retry_count'] += 1
                logger.info(f"Retrying request (attempt {retry_count + 1}/{self.max_retries})")
                
                # Clean up
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]
                
                # Reconnect
                await self.disconnect()
                await self.connect()
                
                # Retry with increased count
                return await self._send_request(request_type, request_data, retry_count + 1)
                
            self.stats['error_count'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            
            # Clean up
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
                
            if request_id in self._request_timeout_tasks:
                timeout_task = self._request_timeout_tasks.pop(request_id)
                if not timeout_task.done():
                    timeout_task.cancel()
            
            self.stats['error_count'] += 1
            return None