"""
hypersphere_dispatcher.py

This module implements the HypersphereDispatcher class, which serves as the central
coordinator for communication between the Lucidia memory system and external tensor/HPC servers.
It integrates geometry management, confidence handling, memory decay, and batch optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from .manifold_geometry import ManifoldGeometryRegistry
from .confidence_manager import BoundedConfidenceManager
from .memory_decay import StableMemoryDecayManager
from .batch_scheduler import AdaptiveHPCBatchScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSocketConnectionPool:
    """A pool of WebSocket connections for reuse."""
    
    def __init__(self, uri: str, max_connections: int = 5, connection_timeout: float = 10.0):
        """
        Initialize a WebSocket connection pool.
        
        Args:
            uri: WebSocket URI to connect to
            max_connections: Maximum number of connections to maintain
            connection_timeout: Timeout for connection attempts in seconds
        """
        self.uri = uri
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.available_connections = asyncio.Queue()
        self.active_connections = set()
        self.connection_locks = {}  # Locks for each connection
        self._closed = False
        
    async def get_connection(self) -> Tuple[websockets.WebSocketClientProtocol, asyncio.Lock]:
        """Get a connection from the pool or create a new one."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        # Try to get an existing connection
        try:
            while not self.available_connections.empty():
                ws, lock = await self.available_connections.get()
                # Safely check if connection is closed
                try:
                    # Use a safe way to check if connection is closed
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=1.0)
                    return ws, lock
                except Exception:
                    # Connection is closed or ping failed, remove it
                    self.active_connections.discard(ws)
                    if ws in self.connection_locks:
                        del self.connection_locks[ws]
        except Exception as e:
            logger.warning(f"Error retrieving connection from pool: {e}")
        
        # Create new connection if under limit
        if len(self.active_connections) < self.max_connections:
            try:
                ws = await asyncio.wait_for(
                    websockets.connect(self.uri),
                    timeout=self.connection_timeout
                )
                lock = asyncio.Lock()
                self.active_connections.add(ws)
                self.connection_locks[ws] = lock
                return ws, lock
            except Exception as e:
                logger.error(f"Failed to create new WebSocket connection: {e}")
                raise
        
        # Wait for a connection to become available
        return await self.available_connections.get()
    
    async def release_connection(self, ws: websockets.WebSocketClientProtocol):
        """Return a connection to the pool."""
        # Safely check if the connection should be returned to the pool
        try:
            # Check if pool is closed or connection is unusable
            if self._closed or ws not in self.active_connections:
                return
            
            # Test if the connection is still alive
            try:
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=1.0)
                # Connection is good, return it to the pool
                await self.available_connections.put((ws, self.connection_locks.get(ws, asyncio.Lock())))
            except Exception:
                # Connection is not usable, remove it
                self.active_connections.discard(ws)
                if ws in self.connection_locks:
                    del self.connection_locks[ws]
        except Exception as e:
            logger.warning(f"Error releasing connection: {e}")
    
    async def close(self):
        """Close all connections in the pool."""
        self._closed = True
        for ws in self.active_connections:
            try:
                await ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")
        
        self.active_connections.clear()
        self.connection_locks.clear()
        
        # Clear the queue
        while not self.available_connections.empty():
            try:
                await self.available_connections.get_nowait()
            except asyncio.QueueEmpty:
                break

class HypersphereDispatcher:
    """
    Central dispatcher for managing communications with tensor and HPC servers.
    
    This class integrates:
    - ManifoldGeometryRegistry: Ensures geometric consistency across models
    - BoundedConfidenceManager: Manages confidence values with boundaries
    - StableMemoryDecayManager: Handles memory decay consistently
    - AdaptiveHPCBatchScheduler: Optimizes batch sizes for HPC operations
    """
    
    def __init__(
        self,
        tensor_server_uri: str,
        hpc_server_uri: str,
        max_connections: int = 5,
        min_batch_size: int = 4,
        max_batch_size: int = 32,
        target_latency: float = 0.5,
        reconnect_backoff_min: float = 0.1,
        reconnect_backoff_max: float = 30.0,
        reconnect_backoff_factor: float = 2.0,
        health_check_interval: float = 60.0
    ):
        """
        Initialize the HypersphereDispatcher.
        
        Args:
            tensor_server_uri: URI for the tensor server WebSocket
            hpc_server_uri: URI for the HPC server WebSocket
            max_connections: Maximum number of connections per server
            min_batch_size: Minimum batch size for HPC operations
            max_batch_size: Maximum batch size for HPC operations
            target_latency: Target latency for batch processing in seconds
            reconnect_backoff_min: Minimum backoff time for reconnection attempts
            reconnect_backoff_max: Maximum backoff time for reconnection attempts
            reconnect_backoff_factor: Multiplier for exponential backoff
            health_check_interval: Interval for health checks in seconds
        """
        # Initialize connection pools
        self.tensor_pool = WebSocketConnectionPool(tensor_server_uri, max_connections)
        self.hpc_pool = WebSocketConnectionPool(hpc_server_uri, max_connections)
        
        # Initialize component integrations
        self.geometry_registry = ManifoldGeometryRegistry()
        self.confidence_manager = BoundedConfidenceManager()
        self.decay_manager = StableMemoryDecayManager()
        self.batch_scheduler = AdaptiveHPCBatchScheduler(min_batch_size, max_batch_size, target_latency)
        
        # Connection management settings
        self.reconnect_backoff_min = reconnect_backoff_min
        self.reconnect_backoff_max = reconnect_backoff_max
        self.reconnect_backoff_factor = reconnect_backoff_factor
        
        # Health check settings
        self.health_check_interval = health_check_interval
        self.health_check_task = None
        self.is_healthy = {"tensor": False, "hpc": False}
        
        # Processing state
        self.request_queue = asyncio.Queue()
        self.processing_task = None
        self.stopping = False
        
        # Batch management
        self.batch_locks = {}  # Map batch_id -> lock
        
    async def start(self):
        """Start the dispatcher and health check tasks."""
        logger.info("Starting HypersphereDispatcher")
        self.stopping = False
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.processing_task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop the dispatcher and all associated tasks."""
        logger.info("Stopping HypersphereDispatcher")
        self.stopping = True
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        await self.tensor_pool.close()
        await self.hpc_pool.close()
    
    async def get_embedding(self, text: str, model_version: str) -> Dict[str, Any]:
        """
        Get an embedding for the given text using the specified model version.
        
        Args:
            text: Text to embed
            model_version: Model version to use
            
        Returns:
            Dictionary containing the embedding and metadata
        """
        logger.info(f"HypersphereDispatcher: Getting embedding for model {model_version}, text length: {len(text)}")
        try:
            # Ensure we have the geometry for this model
            if not await self.geometry_registry.has_geometry(model_version):
                logger.info(f"HypersphereDispatcher: Fetching geometry for model {model_version}")
                await self._fetch_model_geometry(model_version)
            
            # Create embedding request using the EXACT format expected by the tensor server
            # Refer to the format in updated_hpc_client.py
            request = {
                'type': 'embed',  # This is the correct field name
                'request_id': f"{int(time.time())}:{id(text)}",
                'timestamp': time.time(),
                'text': text,
                'model_version': model_version
            }
            
            # Send request to tensor server
            logger.info(f"HypersphereDispatcher: Sending embedding request to tensor server: {request['type']}")
            try:
                response = await self._send_tensor_request(request)
                logger.info(f"HypersphereDispatcher: Got response from tensor server: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
            except Exception as e:
                logger.error(f"HypersphereDispatcher: Error in _send_tensor_request: {str(e)}")
                return {"status": "error", "message": f"Failed to send tensor request: {str(e)}"}
            
            # Extract embedding from response (handle different response formats)
            embedding = None
            if "data" in response and "embedding" in response["data"]:
                embedding = response["data"]["embedding"]
            elif "embedding" in response:
                embedding = response["embedding"]
            elif "embeddings" in response:
                embedding = response["embeddings"]  # Handle plural 'embeddings' key
            elif "data" in response and "embeddings" in response["data"]:
                embedding = response["data"]["embeddings"]
            elif "result" in response and isinstance(response["result"], dict):
                if "embedding" in response["result"]:
                    embedding = response["result"]["embedding"]
                elif "embeddings" in response["result"]:
                    embedding = response["result"]["embeddings"]
            
            if not embedding:
                logger.error(f"HypersphereDispatcher: No embedding in response: {response}")
                return {"status": "error", "message": "No embedding in response", "response": response}
            
            # Verify and apply geometry constraints
            # Check embedding structure
            if not isinstance(embedding, list) or len(embedding) == 0:
                logger.error(f"HypersphereDispatcher: Invalid embedding format: {type(embedding)}")
                return {"status": "error", "message": "Invalid embedding format received from tensor server"}
            
            logger.info(f"HypersphereDispatcher: Embedding length: {len(embedding)}")
            
            # Verify the embedding is compatible with the model's geometry
            try:
                if not await self.geometry_registry.check_embedding_compatibility(model_version, embedding):
                    logger.warning(f"HypersphereDispatcher: Received incompatible embedding from tensor server for model {model_version}")
                    
                    # Attempt to fix the embedding according to geometry constraints
                    try:
                        geometry = await self.geometry_registry.get_geometry(model_version)
                        
                        # Normalize the embedding if needed (for unit hypersphere)
                        import numpy as np
                        embedding_np = np.array(embedding)
                        norm = np.linalg.norm(embedding_np)
                        
                        if abs(norm - 1.0) > 0.001:  # If not already normalized
                            normalized = embedding_np / norm
                            embedding = normalized.tolist()
                            logger.info(f"HypersphereDispatcher: Normalized embedding to conform to unit hypersphere for model {model_version}")
                    except Exception as e:
                        logger.error(f"HypersphereDispatcher: Failed to normalize embedding: {e}")
                        # Continue with the original embedding
            except Exception as e:
                logger.error(f"HypersphereDispatcher: Error checking embedding compatibility: {str(e)}")
            
            # Return the final result with proper structure
            return {
                "status": "success",
                "embedding": embedding,
                "model_version": model_version,
                "dimensions": len(embedding) if embedding else 0
            }
            
        except Exception as e:
            logger.error(f"HypersphereDispatcher: Unexpected error in get_embedding: {str(e)}")
            import traceback
            logger.error(f"HypersphereDispatcher: Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Error generating embedding: {str(e)}"}
    
    async def batch_similarity_search(
        self, 
        query_embedding: List[float], 
        memory_embeddings: List[List[float]],
        memory_ids: List[str],
        model_version: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform a batch similarity search between a query embedding and memory embeddings.
        
        Args:
            query_embedding: The query embedding vector
            memory_embeddings: List of memory embedding vectors to compare against
            memory_ids: Corresponding memory IDs for the embeddings
            model_version: Model version used for the embeddings
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with match information
        """
        # Check query embedding compatibility with model
        if not await self.geometry_registry.check_embedding_compatibility(model_version, query_embedding):
            raise ValueError(f"Query embedding not compatible with {model_version} geometry")
        
        # Prepare batch information for verification
        batch_id = f"batch_{time.time()}_{id(query_embedding)}"
        model_versions = [model_version] * len(memory_embeddings)
        
        # Verify batch compatibility
        batch_compatible = await self.geometry_registry.verify_batch_compatibility(
            [query_embedding] + memory_embeddings,
            [model_version] + model_versions
        )
        
        if not batch_compatible:
            logger.warning(f"Batch {batch_id} contains incompatible embeddings")
            
            # Try to make the batch compatible by transforming embeddings
            compatible_embeddings = []
            compatible_ids = []
            
            for i, (embedding, memory_id) in enumerate(zip(memory_embeddings, memory_ids)):
                try:
                    # Check if this specific embedding is compatible with the query
                    if await self.geometry_registry.check_embedding_compatibility(model_version, embedding):
                        compatible_embeddings.append(embedding)
                        compatible_ids.append(memory_id)
                    else:
                        # Try to transform the embedding if possible
                        embedding_model = await self.geometry_registry.get_model_for_embedding(memory_id)
                        if embedding_model:
                            transformed = await self.geometry_registry.transform_embedding(
                                embedding, embedding_model, model_version
                            )
                            compatible_embeddings.append(transformed)
                            compatible_ids.append(memory_id)
                            logger.info(f"Transformed embedding for memory {memory_id} from {embedding_model} to {model_version}")
                except Exception as e:
                    logger.warning(f"Failed to transform embedding for memory {memory_id}: {e}")
                    # Skip this embedding
            
            # Update our working set to only compatible embeddings
            memory_embeddings = compatible_embeddings
            memory_ids = compatible_ids
            
            if not memory_embeddings:
                return []  # No compatible embeddings found
        
        # Create HPC request
        request = {
            "type": "similarity_search",
            "batch_id": batch_id,
            "query_embedding": query_embedding,
            "memory_embeddings": memory_embeddings,
            "memory_ids": memory_ids,
            "model_version": model_version,
            "top_k": top_k,
            "timestamp": time.time()
        }
        
        # Create a lock for this batch
        self.batch_locks[batch_id] = asyncio.Lock()
        
        # Queue the request for optimized batch processing
        future = asyncio.Future()
        await self.request_queue.put((request, future))
        
        # Wait for the result
        try:
            result = await future
            return result
        finally:
            # Clean up batch lock
            if batch_id in self.batch_locks:
                del self.batch_locks[batch_id]
    
    async def register_memory(
        self,
        memory_id: str,
        embedding: List[float],
        model_version: str,
        importance: float,
        creation_time: float,
        confidence: float
    ) -> None:
        """
        Register a memory with the dispatcher for decay and confidence management.
        
        Args:
            memory_id: Unique identifier for the memory
            embedding: The memory's embedding vector
            model_version: Model version used for the embedding
            importance: Initial importance score
            creation_time: Creation timestamp
            confidence: Initial confidence value
        """
        # Register with geometry registry
        await self.geometry_registry.register_embedding(memory_id, embedding, model_version)
        
        # Register with decay manager
        await self.decay_manager.register_memory(memory_id, creation_time, importance)
        
        # Register with confidence manager
        adjusted_confidence = await self.confidence_manager.apply_adjustment(confidence)
        # Additional confidence registration logic if needed
        
        logger.info(f"Registered memory {memory_id} with model {model_version}")
    
    async def update_memory_importance(self, memory_id: str, importance_delta: float) -> float:
        """
        Update a memory's importance score.
        
        Args:
            memory_id: Memory identifier
            importance_delta: Change in importance
            
        Returns:
            New importance value
        """
        return await self.decay_manager.update_importance(memory_id, importance_delta)
    
    async def get_decay_weights(self, memory_ids: List[str]) -> Dict[str, float]:
        """
        Get decay weights for a list of memories.
        
        Args:
            memory_ids: List of memory identifiers
            
        Returns:
            Dictionary mapping memory IDs to decay weights
        """
        return await self.decay_manager.get_decay_weights(memory_ids)
    
    async def clean_expired_memories(self, threshold: float = 0.1) -> List[str]:
        """
        Clean up memories with decay weights below the threshold.
        
        Args:
            threshold: Minimum decay weight to retain
            
        Returns:
            List of removed memory IDs
        """
        removed_ids = await self.decay_manager.clean_expired_memories(threshold)
        
        # Also clean up from geometry registry
        for memory_id in removed_ids:
            await self.geometry_registry.remove_embedding(memory_id)
        
        return removed_ids
    
    async def _health_check_loop(self):
        """Periodically check the health of tensor and HPC servers."""
        while not self.stopping:
            try:
                await self._check_tensor_server_health()
                await self._check_hpc_server_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _check_tensor_server_health(self):
        """Check if the tensor server is responsive."""
        try:
            ws, lock = await self.tensor_pool.get_connection()
            try:
                async with lock:
                    health_req = {"type": "health_check", "timestamp": time.time()}
                    await ws.send(json.dumps(health_req))
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    resp_data = json.loads(response)
                    self.is_healthy["tensor"] = resp_data.get("status") == "ok"
            finally:
                await self.tensor_pool.release_connection(ws)
        except Exception as e:
            logger.warning(f"Tensor server health check failed: {e}")
            self.is_healthy["tensor"] = False
    
    async def _check_hpc_server_health(self):
        """Check if the HPC server is responsive."""
        try:
            ws, lock = await self.hpc_pool.get_connection()
            try:
                async with lock:
                    health_req = {"type": "health_check", "timestamp": time.time()}
                    await ws.send(json.dumps(health_req))
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    resp_data = json.loads(response)
                    self.is_healthy["hpc"] = resp_data.get("status") == "ok"
            finally:
                await self.hpc_pool.release_connection(ws)
        except Exception as e:
            logger.warning(f"HPC server health check failed: {e}")
            self.is_healthy["hpc"] = False
    
    async def _fetch_model_geometry(self, model_version: str):
        """Fetch geometry parameters for a model version from the tensor server."""
        request = {
            "type": "get_geometry",
            "model_version": model_version,
            "timestamp": time.time()
        }
        
        try:
            response = await self._send_tensor_request(request)
            if "geometry" in response:
                geo = response["geometry"]
                await self.geometry_registry.register_geometry(
                    model_version,
                    geo.get("dimensions"),
                    geo.get("curvature"),
                    geo.get("parameters", {})
                )
                logger.info(f"Fetched and registered geometry for model {model_version}")
        except Exception as e:
            logger.error(f"Failed to fetch geometry for model {model_version}: {e}")
    
    async def _send_tensor_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the tensor server with retry logic."""
        backoff = self.reconnect_backoff_min
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                ws, lock = await self.tensor_pool.get_connection()
                try:
                    async with lock:
                        await ws.send(json.dumps(request))
                        response = await ws.recv()
                        return json.loads(response)
                finally:
                    await self.tensor_pool.release_connection(ws)
            except (ConnectionClosed, ConnectionClosedError):
                # Connection was closed, try to reconnect
                logger.warning(f"Tensor server connection closed, retrying (attempt {attempt})")
            except Exception as e:
                logger.error(f"Error in tensor server request: {e}")
            
            # Apply exponential backoff
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff = min(backoff * self.reconnect_backoff_factor, self.reconnect_backoff_max)
        
        raise RuntimeError(f"Failed to send tensor request after {max_attempts} attempts")
    
    async def _send_hpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the HPC server with retry logic."""
        backoff = self.reconnect_backoff_min
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                ws, lock = await self.hpc_pool.get_connection()
                try:
                    async with lock:
                        await ws.send(json.dumps(request))
                        response = await ws.recv()
                        return json.loads(response)
                finally:
                    await self.hpc_pool.release_connection(ws)
            except (ConnectionClosed, ConnectionClosedError):
                # Connection was closed, try to reconnect
                logger.warning(f"HPC server connection closed, retrying (attempt {attempt})")
            except Exception as e:
                logger.error(f"Error in HPC server request: {e}")
            
            # Apply exponential backoff
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff = min(backoff * self.reconnect_backoff_factor, self.reconnect_backoff_max)
        
        raise RuntimeError(f"Failed to send HPC request after {max_attempts} attempts")
    
    async def _process_queue(self):
        """Process the request queue, batching requests when possible."""
        while not self.stopping:
            try:
                # Get batch of requests
                batch = await self.batch_scheduler.collect_batch(self.request_queue)
                if not batch:
                    continue
                
                # Process the batch
                start_time = time.time()
                
                # Extract requests and futures
                requests = [req for req, _ in batch]
                futures = [future for _, future in batch]
                
                # Verify geometric compatibility across the batch
                batch_model_versions = []
                batch_embeddings = []
                
                for req in requests:
                    if req["type"] == "similarity_search":
                        batch_model_versions.append(req["model_version"])
                        batch_embeddings.append(req["query_embedding"])
                
                # Only check compatibility if we have more than one request
                if len(batch_model_versions) > 1:
                    batch_compatible = await self.geometry_registry.verify_batch_compatibility(
                        batch_embeddings, batch_model_versions
                    )
                    
                    if not batch_compatible:
                        logger.warning("Batch contains incompatible embeddings, splitting")
                        
                        # Split into compatible sub-batches
                        for i, (req, future) in enumerate(batch):
                            # Put back in queue to be processed in separate batches
                            if i > 0:  # Keep at least the first request in this batch
                                await self.request_queue.put((req, future))
                                futures[i] = None  # Mark as handled
                        
                        # Update batch to only include the first request
                        requests = [requests[0]]
                        futures = [f for f in futures if f is not None]
                
                # Create a batch request
                batch_request = {
                    "type": "batch_processing",
                    "requests": requests,
                    "timestamp": start_time
                }
                
                try:
                    # Send the batch to the HPC server
                    response = await self._send_hpc_request(batch_request)
                    
                    # Set results in futures
                    if "results" in response and len(response["results"]) == len(futures):
                        for i, result in enumerate(response["results"]):
                            if futures[i] and not futures[i].done():
                                futures[i].set_result(result)
                    else:
                        # Handle mismatched response
                        for future in futures:
                            if future and not future.done():
                                future.set_exception(RuntimeError("Invalid batch response"))
                except Exception as e:
                    # Set exception for all futures
                    for future in futures:
                        if future and not future.done():
                            future.set_exception(e)
                
                # Update batch scheduler with processing metrics
                end_time = time.time()
                processing_time = end_time - start_time
                await self.batch_scheduler.record_performance(len(batch), processing_time)
                
            except asyncio.CancelledError:
                # Clean shutdown
                # Set exception for any pending futures
                while not self.request_queue.empty():
                    try:
                        _, future = self.request_queue.get_nowait()
                        if not future.done():
                            future.set_exception(asyncio.CancelledError())
                    except asyncio.QueueEmpty:
                        break
                raise
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def get_model_for_embedding(self, memory_id: str) -> Optional[str]:
        """
        Get the model version used for a memory's embedding.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Model version string or None if not found
        """
        return await self.geometry_registry.get_model_for_embedding(memory_id)
    
    async def transform_embedding_batch(
        self,
        embeddings: List[List[float]],
        source_models: List[str],
        target_model: str
    ) -> List[List[float]]:
        """
        Transform a batch of embeddings to a target model's geometry.
        
        Args:
            embeddings: List of embedding vectors
            source_models: Source model versions for each embedding
            target_model: Target model version
            
        Returns:
            List of transformed embeddings
        """
        transformed = []
        
        for embedding, source_model in zip(embeddings, source_models):
            try:
                # Transform only if source and target models differ
                if source_model != target_model:
                    transformed_embedding = await self.geometry_registry.transform_embedding(
                        embedding, source_model, target_model
                    )
                    transformed.append(transformed_embedding)
                else:
                    transformed.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to transform embedding from {source_model} to {target_model}: {e}")
                # Use original embedding as fallback
                transformed.append(embedding)
        
        return transformed
    
    def register_tensor_client(self, memory_client):
        """
        Register an EnhancedMemoryClient for tensor server communication.
        
        This method allows the HypersphereDispatcher to use an existing
        EnhancedMemoryClient's websocket connections instead of managing its own.
        
        Args:
            memory_client: Instance of EnhancedMemoryClient with tensor server connection
        """
        try:
            # Store the memory client for later use
            self.memory_client = memory_client
            logger.info("Registered tensor client with HypersphereDispatcher")
        except Exception as e:
            logger.error(f"Failed to register tensor client: {e}")
        
    def register_hpc_client(self, memory_client):
        """
        Register an EnhancedMemoryClient for HPC server communication.
        
        This method allows the HypersphereDispatcher to use an existing
        EnhancedMemoryClient's websocket connections instead of managing its own.
        
        Args:
            memory_client: Instance of EnhancedMemoryClient with HPC server connection
        """
        try:
            # If not already registered in tensor_client
            if not hasattr(self, 'memory_client'):
                self.memory_client = memory_client
            logger.info("Registered HPC client with HypersphereDispatcher")
        except Exception as e:
            logger.error(f"Failed to register HPC client: {e}")
    
    async def batch_embed_texts(self, texts: List[str], model_version: str = "latest") -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            model_version: Model version to use for embedding
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            # Use the batch scheduler to determine optimal batch size
            batch_size = self.batch_scheduler.get_optimal_batch_size(len(texts))
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = []
                
                # Process each text in the batch
                for text in batch:
                    embedding_result = await self.get_embedding(text, model_version)
                    if embedding_result and "embedding" in embedding_result:
                        batch_embeddings.append(embedding_result["embedding"])
                    else:
                        # Add a placeholder if embedding failed
                        logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                        batch_embeddings.append([])
                
                embeddings.extend(batch_embeddings)
                
                # Update the batch scheduler with performance metrics
                self.batch_scheduler.update_metrics(len(batch), batch_size)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error in batch_embed_texts: {e}")
            # Return empty embeddings as fallback
            return [[] for _ in texts]

    async def batch_get_embeddings(self, texts: List[str], model_version: str = "latest") -> Dict[str, Any]:
        """
        Process multiple texts into embeddings in a single batch operation.
        
        Args:
            texts: List of texts to embed
            model_version: The model version to use
            
        Returns:
            Dictionary containing all embeddings and metadata
        """
        try:
            embeddings_list = await self.batch_embed_texts(texts, model_version)
            
            # Format the response to match what HypersphereManager expects
            results = []
            for i, embedding in enumerate(embeddings_list):
                if embedding:  # If embedding is not empty
                    results.append({
                        "embedding": embedding,
                        "model_version": model_version,
                        "dimensions": len(embedding),
                        "status": "success"
                    })
                else:
                    results.append({
                        "error": "Failed to generate embedding",
                        "status": "error"
                    })
            
            return {
                "status": "success",
                "embeddings": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in batch_get_embeddings: {e}")
            return {
                "status": "error",
                "message": str(e),
                "embeddings": []
            }

    async def embed_text(self, text: str, model_version: str = "latest") -> Dict[str, Any]:
        """
        Wrapper for get_embedding with simplified return format.
        
        Args:
            text: Text to embed
            model_version: Model version to use
            
        Returns:
            Dictionary with embedding vector and metadata
        """
        return await self.get_embedding(text, model_version)