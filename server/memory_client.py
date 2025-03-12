# server/memory_client.py
import aiohttp
import asyncio
import logging
import json
import time
import websockets
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch

from memory.lucidia_memory_system.core.memory_entry import MemoryEntry

class EnhancedMemoryClient:
    """
    Enhanced client for interacting with the memory system.
    
    This client provides a unified interface for memory operations,
    handling communication with the tensor and HPC servers.
    """
    
    def __init__(self, config: Dict[str, Any] = None, memory_system: Any = None, memory_bridge: Any = None):
        """
        Initialize the enhanced memory client.
        
        Args:
            config: Configuration dictionary
            memory_system: Optional memory system instance
            memory_bridge: Optional memory bridge instance for connecting flat and hierarchical memory
        """
        self.logger = logging.getLogger("EnhancedMemoryClient")
        self.config = config or {}
        
        # Server URLs
        self.tensor_server_url = self.config.get("tensor_server_url", "ws://localhost:5001")
        self.hpc_server_url = self.config.get("hpc_server_url", "ws://localhost:5005")
        
        # WebSocket connection parameters
        self.ping_interval = self.config.get("ping_interval", 30.0)
        self.max_retries = self.config.get("max_retries", 5)
        self.retry_delay = self.config.get("retry_delay", 2.0)
        
        # Connection objects
        self.tensor_connection = None
        self.hpc_connection = None
        
        # Locks for thread safety
        self.tensor_lock = asyncio.Lock()
        self.hpc_lock = asyncio.Lock()
        
        # Memory cache for frequently accessed memories
        self.memory_cache = {}
        self.cache_size = self.config.get("cache_size", 100)
        
        # Operations statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "embedding_requests": 0,
            "retrieval_requests": 0,
            "search_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Store the memory system reference
        self.memory_system = memory_system
        
        # Store the memory bridge reference
        self.memory_bridge = memory_bridge
        
        self.logger.info("Enhanced Memory Client initialized with memory systems: " + 
                        f"Flat: {memory_system is not None}, Bridge: {memory_bridge is not None}")
    
    async def initialize(self) -> None:
        """Initialize the memory client connections."""
        # Establish initial connections (but don't fail if they don't connect immediately)
        try:
            await self.get_tensor_connection()
        except Exception as e:
            self.logger.warning(f"Could not establish initial tensor connection: {e}")
            
        try:
            await self.get_hpc_connection()
        except Exception as e:
            self.logger.warning(f"Could not establish initial HPC connection: {e}")
    
    async def close(self) -> None:
        """Close all connections."""
        try:
            if self.tensor_connection and not self.tensor_connection.closed:
                await self.tensor_connection.close()
                self.tensor_connection = None
                
            if self.hpc_connection and not self.hpc_connection.closed:
                await self.hpc_connection.close()
                self.hpc_connection = None
                
            self.logger.info("All memory client connections closed")
        except Exception as e:
            self.logger.error(f"Error closing memory client connections: {e}")
    
    async def close_connections(self):
        """
        Properly close all WebSocket connections.
        
        This method should be called during system shutdown to ensure clean termination
        of all connections to tensor and HPC servers.
        """
        try:
            self.logger.info("Closing all WebSocket connections...")
            
            # Close tensor connection if it exists
            if self.tensor_connection and not self.tensor_connection.closed:
                try:
                    await self.tensor_connection.close()
                    self.logger.info("Tensor connection closed successfully")
                except Exception as e:
                    self.logger.error(f"Error closing tensor connection: {e}")
            
            # Close HPC connection if it exists
            if self.hpc_connection and not self.hpc_connection.closed:
                try:
                    await self.hpc_connection.close()
                    self.logger.info("HPC connection closed successfully")
                except Exception as e:
                    self.logger.error(f"Error closing HPC connection: {e}")
            
            # Reset connection objects
            self.tensor_connection = None
            self.hpc_connection = None
            
            self.logger.info("All connections closed")
            
        except Exception as e:
            self.logger.error(f"Error during connection shutdown: {e}")
    
    async def get_tensor_connection(self) -> Any:
        """Get or establish connection to tensor server."""
        async with self.tensor_lock:
            # Check if existing connection is still alive
            if self.tensor_connection and not self.tensor_connection.closed:
                try:
                    # Test connection with ping
                    pong_waiter = await self.tensor_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5.0)
                    return self.tensor_connection
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    # Connection is dead, need to reconnect
                    self.logger.info("Tensor connection closed or unresponsive, reconnecting")
                    try:
                        await self.tensor_connection.close()
                    except:
                        pass
                    self.tensor_connection = None
            
            # Establish new connection with retry
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Connecting to tensor server at {self.tensor_server_url} (attempt {attempt+1}/{self.max_retries})")
                    self.tensor_connection = await websockets.connect(
                        self.tensor_server_url,
                        ping_interval=self.ping_interval
                    )
                    self.logger.info("Successfully connected to tensor server")
                    return self.tensor_connection
                except Exception as e:
                    self.logger.warning(f"Failed to connect to tensor server: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to connect to tensor server after {self.max_retries} attempts")
                        raise
    
    async def get_hpc_connection(self) -> Any:
        """Get or establish connection to HPC server."""
        async with self.hpc_lock:
            # Check if existing connection is still alive
            if self.hpc_connection and not self.hpc_connection.closed:
                try:
                    # Test connection with ping
                    pong_waiter = await self.hpc_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5.0)
                    return self.hpc_connection
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    # Connection is dead, need to reconnect
                    self.logger.info("HPC connection closed or unresponsive, reconnecting")
                    try:
                        await self.hpc_connection.close()
                    except:
                        pass
                    self.hpc_connection = None
            
            # Establish new connection with retry
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"Connecting to HPC server at {self.hpc_server_url} (attempt {attempt+1}/{self.max_retries})")
                    self.hpc_connection = await websockets.connect(
                        self.hpc_server_url,
                        ping_interval=self.ping_interval
                    )
                    self.logger.info("Successfully connected to HPC server")
                    return self.hpc_connection
                except Exception as e:
                    self.logger.warning(f"Failed to connect to HPC server: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to connect to HPC server after {self.max_retries} attempts")
                        raise
    
    async def add_memory(self, content: str, memory_type: str = "general", metadata: Optional[Dict[str, Any]] = None) -> Optional[MemoryEntry]:
        """
        Add a new memory to the memory system.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Optional additional metadata
            
        Returns:
            Created memory entry or None if failed
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["embedding_requests"] += 1
            
            # Check if we have the memory bridge for integrated storage
            if self.memory_bridge:
                self.logger.info("Using memory bridge for integrated memory storage")
                
                # Get tensor connection for embedding generation
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request for embedding generation only
                request = {
                    "type": "embed_only",  # Use a different type to avoid storing in the tensor server
                    "text": content,
                    "client_id": "memory_client",
                    "message_id": f"mem_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response with embedding
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error generating embedding: {response_data.get('message', 'Unknown error')}")
                    raise Exception(f"Error generating embedding: {response_data.get('message', 'Unknown error')}")
                
                # Extract embedding
                embedding = response_data.get("embedding")
                if embedding is None:
                    self.logger.error("No embedding received from tensor server")
                    raise Exception("No embedding received from tensor server")
                
                # Get HPC connection for significance calculation
                hpc_conn = await self.get_hpc_connection()
                
                # Prepare request for significance calculation
                request = {
                    "type": "calculate_significance",
                    "content": content,
                    "client_id": "memory_client",
                    "message_id": f"sig_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response with significance
                await hpc_conn.send(json.dumps(request))
                response = await hpc_conn.recv()
                response_data = json.loads(response)
                
                # Extract significance (or use default if not provided)
                significance = response_data.get("significance", 0.5)
                
                # Add categories based on memory type and metadata
                categories = [memory_type]
                if metadata and "categories" in metadata:
                    categories.extend(metadata.get("categories", []))
                
                # Combine metadata
                combined_metadata = metadata or {}
                combined_metadata["memory_type"] = memory_type
                combined_metadata["timestamp"] = time.time()
                
                # Store memory in both systems through the bridge
                memory_id = await self.memory_bridge.store_memory(
                    content=content,
                    significance=significance,
                    metadata=combined_metadata,
                    categories=categories,
                    embedding=embedding
                )
                
                # Create memory entry for return
                memory_entry = MemoryEntry(
                    id=memory_id,
                    content=content,
                    embedding=torch.tensor(embedding) if embedding else None,
                    metadata=combined_metadata,
                    timestamp=time.time(),
                    memory_type=memory_type,
                    significance=significance
                )
                
                # Add to cache
                self._add_to_cache(memory_entry)
                
                self.stats["successful_requests"] += 1
                return memory_entry
            # Check if we have direct access to the memory system
            elif self.memory_system:
                self.logger.info("Using direct memory system access for adding memory")
                
                # Get tensor connection for embedding generation
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request for embedding generation only
                request = {
                    "type": "embed_only",  # Use a different type to avoid storing in the tensor server
                    "text": content,
                    "client_id": "memory_client",
                    "message_id": f"mem_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response with embedding
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error embedding memory: {response_data.get('error')}")
                    self.stats["failed_requests"] += 1
                    return None
                
                # Get embedding from response
                embedding = torch.tensor(response_data.get("embeddings", []))
                
                # Store in unified memory system
                memory_data = await self.memory_system.add_memory(
                    text=content,
                    embedding=embedding
                )
                
                # Extract memory data
                memory_id = memory_data.get("id")
                timestamp = memory_data.get("timestamp", time.time())
                significance = memory_data.get("significance", 0.5)
                
                # Create memory entry
                memory = MemoryEntry(
                    id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    created_at=timestamp,
                    significance=significance,
                    metadata=metadata or {}
                )
                
                # Add to cache
                self._add_to_cache(memory)
                
                self.stats["successful_requests"] += 1
                return memory
            else:
                # Use existing logic with tensor server
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request
                request = {
                    "type": "embed",
                    "text": content,
                    "client_id": "memory_client",
                    "message_id": f"mem_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error embedding memory: {response_data.get('error')}")
                    self.stats["failed_requests"] += 1
                    return None
                
                # Extract memory data
                memory_id = response_data.get("id", f"memory_{int(time.time())}")
                timestamp = response_data.get("timestamp", time.time())
                significance = response_data.get("significance", 0.5)
                
                # Create memory entry
                memory = MemoryEntry(
                    id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    created_at=timestamp,
                    significance=significance,
                    metadata=metadata or {}
                )
                
                # Add to cache
                self._add_to_cache(memory)
                
                self.stats["successful_requests"] += 1
                return memory
                
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            self.stats["failed_requests"] += 1
            return None
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry or None if not found
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["retrieval_requests"] += 1
            
            # Check cache first
            if memory_id in self.memory_cache:
                self.stats["cache_hits"] += 1
                memory = self.memory_cache[memory_id]
                
                # Update access stats and move to front of cache
                memory.record_access()
                self._add_to_cache(memory)  # This will move it to the front
                
                return memory
            
            self.stats["cache_misses"] += 1
            
            # Get tensor connection
            tensor_conn = await self.get_tensor_connection()
            
            # Prepare request
            request = {
                "type": "get_memory",
                "memory_id": memory_id,
                "client_id": "memory_client",
                "message_id": f"get_{int(time.time() * 1000)}",
                "timestamp": time.time()
            }
            
            # Send request and get response
            await tensor_conn.send(json.dumps(request))
            response = await tensor_conn.recv()
            response_data = json.loads(response)
            
            if "type" in response_data and response_data["type"] == "error":
                self.logger.error(f"Error retrieving memory: {response_data.get('error')}")
                self.stats["failed_requests"] += 1
                return None
            
            if "memory" not in response_data:
                self.logger.error("No memory in response")
                self.stats["failed_requests"] += 1
                return None
            
            # Create memory entry from response
            memory_data = response_data["memory"]
            memory = MemoryEntry(
                id=memory_data.get("id", memory_id),
                content=memory_data.get("content", ""),
                memory_type=memory_data.get("type", "general"),
                created_at=memory_data.get("timestamp", time.time()),
                significance=memory_data.get("significance", 0.5),
                metadata=memory_data.get("metadata", {})
            )
            
            # Add to cache
            self._add_to_cache(memory)
            
            self.stats["successful_requests"] += 1
            return memory
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory {memory_id}: {e}")
            self.stats["failed_requests"] += 1
            return None
    
    async def get_memories_by_ids(self, memory_ids: List[str]) -> List[MemoryEntry]:
        """
        Retrieve multiple memories by their IDs.
        
        Args:
            memory_ids: List of memory IDs to retrieve
            
        Returns:
            List of memory entries (any that couldn't be found will be omitted)
        """
        if not memory_ids:
            return []
            
        # Retrieve each memory
        results = await asyncio.gather(
            *[self.get_memory(memory_id) for memory_id in memory_ids],
            return_exceptions=True
        )
        
        # Filter out exceptions and None results
        memories = []
        for result in results:
            if isinstance(result, MemoryEntry):
                memories.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error retrieving memory: {result}")
        
        return memories
    
    async def search_similar(self, query: str, limit: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for memories similar to the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results, each containing memory and similarity score
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["search_requests"] += 1
            
            # Check if we have the memory bridge for integrated storage
            if self.memory_bridge:
                self.logger.info("Using memory bridge for integrated memory search")
                
                # Get tensor connection for embedding generation
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request for embedding generation
                request = {
                    "type": "embed_only",
                    "text": query,
                    "client_id": "memory_client",
                    "message_id": f"search_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response with embeddings
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error generating query embedding: {response_data.get('message', 'Unknown error')}")
                    raise Exception(f"Error generating query embedding: {response_data.get('message', 'Unknown error')}")
                
                # Extract embedding
                query_embedding = response_data.get("embedding")
                if query_embedding is None:
                    self.logger.error("No embedding received from tensor server")
                    raise Exception("No embedding received from tensor server")
                
                # Search in both systems through the bridge
                results = await self.memory_bridge.search_memories(
                    query_embedding=query_embedding,
                    limit=limit,
                    threshold=threshold
                )
                
                # Convert to memory entries
                search_results = []
                for result in results:
                    memory_data = result["memory"]
                    similarity = result["similarity"]
                    
                    if similarity < threshold:
                        continue
                    
                    memory = MemoryEntry(
                        id=memory_data.get("id"),
                        content=memory_data.get("content"),
                        created_at=memory_data.get("timestamp"),
                        significance=memory_data.get("significance", 0.0),
                        memory_type="general"
                    )
                    
                    # Add to cache
                    self._add_to_cache(memory)
                    
                    search_results.append({
                        "memory": memory,
                        "similarity": similarity
                    })
                
                self.stats["successful_requests"] += 1
                return search_results
            # Check if we have direct access to the memory system
            elif self.memory_system:
                self.logger.info("Using direct memory system access for searching memories")
                
                # Get tensor connection for embedding generation only
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request for embedding generation
                request = {
                    "type": "embed_only",
                    "text": query,
                    "client_id": "memory_client",
                    "message_id": f"search_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response with embeddings
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error generating query embedding: {response_data.get('error')}")
                    self.stats["failed_requests"] += 1
                    return []
                
                # Get embeddings from response
                query_embedding = torch.tensor(response_data.get("embeddings", []))
                
                # Search in unified memory system
                results = await self.memory_system.search_memories(
                    query_embedding=query_embedding,
                    limit=limit
                )
                
                # Convert to memory entries
                search_results = []
                for result in results:
                    memory_data = result["memory"]
                    similarity = result["similarity"]
                    
                    if similarity < threshold:
                        continue
                    
                    memory = MemoryEntry(
                        id=memory_data.get("id"),
                        content=memory_data.get("text"),
                        created_at=memory_data.get("timestamp"),
                        significance=memory_data.get("significance", 0.0),
                        memory_type="general"
                    )
                    
                    # Add to cache
                    self._add_to_cache(memory)
                    
                    search_results.append({
                        "memory": memory,
                        "similarity": similarity
                    })
                
                self.stats["successful_requests"] += 1
                return search_results
            else:
                # Use existing logic with tensor server
                # Get tensor connection
                tensor_conn = await self.get_tensor_connection()
                
                # Prepare request
                request = {
                    "type": "search",
                    "text": query,
                    "limit": limit,
                    "client_id": "memory_client",
                    "message_id": f"search_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                # Send request and get response
                await tensor_conn.send(json.dumps(request))
                response = await tensor_conn.recv()
                response_data = json.loads(response)
                
                if "type" in response_data and response_data["type"] == "error":
                    self.logger.error(f"Error searching memories: {response_data.get('error')}")
                    self.stats["failed_requests"] += 1
                    return []
                
                # Process search results
                search_results = []
                results = response_data.get("results", [])
                
                for result in results:
                    memory_id = result.get("id")
                    text = result.get("text")
                    similarity = result.get("similarity", 0.0)
                    significance = result.get("significance", 0.0)
                    
                    if similarity < threshold:
                        continue
                    
                    memory = MemoryEntry(
                        id=memory_id,
                        content=text,
                        created_at=time.time(),  # We don't get timestamp from API
                        significance=significance,
                        memory_type="general"
                    )
                    
                    # Add to cache
                    self._add_to_cache(memory)
                    
                    search_results.append({
                        "memory": memory,
                        "similarity": similarity
                    })
                
                self.stats["successful_requests"] += 1
                return search_results
                
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            self.stats["failed_requests"] += 1
            return []
    
    async def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Get the most recent memories.
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of recent memory entries
        """
        try:
            self.stats["total_requests"] += 1
            
            # Get tensor connection
            tensor_conn = await self.get_tensor_connection()
            
            # Prepare request
            request = {
                "type": "get_recent",
                "limit": limit,
                "client_id": "memory_client",
                "message_id": f"recent_{int(time.time() * 1000)}",
                "timestamp": time.time()
            }
            
            # Send request and get response
            await tensor_conn.send(json.dumps(request))
            response = await tensor_conn.recv()
            response_data = json.loads(response)
            
            if "type" in response_data and response_data["type"] == "error":
                self.logger.error(f"Error retrieving recent memories: {response_data.get('error')}")
                self.stats["failed_requests"] += 1
                return []
            
            # Extract memories
            memories = []
            for memory_data in response_data.get("memories", []):
                memory = MemoryEntry(
                    id=memory_data.get("id", f"unknown_{int(time.time())}"),
                    content=memory_data.get("content", ""),
                    memory_type=memory_data.get("type", "general"),
                    created_at=memory_data.get("timestamp", time.time()),
                    significance=memory_data.get("significance", 0.5),
                    metadata=memory_data.get("metadata", {})
                )
                
                # Add to cache
                self._add_to_cache(memory)
                
                memories.append(memory)
            
            self.stats["successful_requests"] += 1
            return memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent memories: {e}")
            self.stats["failed_requests"] += 1
            return []
    
    async def get_significant_memories(self, limit: int = 10, threshold: float = 0.7) -> List[MemoryEntry]:
        """
        Get the most significant memories.
        
        Args:
            limit: Maximum number of memories to retrieve
            threshold: Minimum significance threshold
            
        Returns:
            List of significant memory entries
        """
        try:
            self.stats["total_requests"] += 1
            
            # Get tensor connection
            tensor_conn = await self.get_tensor_connection()
            
            # Prepare request
            request = {
                "type": "get_significant",
                "limit": limit,
                "threshold": threshold,
                "client_id": "memory_client",
                "message_id": f"significant_{int(time.time() * 1000)}",
                "timestamp": time.time()
            }
            
            # Send request and get response
            await tensor_conn.send(json.dumps(request))
            response = await tensor_conn.recv()
            response_data = json.loads(response)
            
            if "type" in response_data and response_data["type"] == "error":
                self.logger.error(f"Error retrieving significant memories: {response_data.get('error')}")
                self.stats["failed_requests"] += 1
                return []
            
            # Extract memories
            memories = []
            for memory_data in response_data.get("memories", []):
                memory = MemoryEntry(
                    id=memory_data.get("id", f"unknown_{int(time.time())}"),
                    content=memory_data.get("content", ""),
                    memory_type=memory_data.get("type", "general"),
                    created_at=memory_data.get("timestamp", time.time()),
                    significance=memory_data.get("significance", 0.5),
                    metadata=memory_data.get("metadata", {})
                )
                
                # Add to cache
                self._add_to_cache(memory)
                
                memories.append(memory)
            
            self.stats["successful_requests"] += 1
            return memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving significant memories: {e}")
            self.stats["failed_requests"] += 1
            return []
    
    async def get_memories_by_timeframe(
        self, 
        start_time: Union[float, datetime], 
        end_time: Union[float, datetime],
        limit: int = 100
    ) -> List[MemoryEntry]:
        """
        Get memories within a specific timeframe.
        
        Args:
            start_time: Start time (timestamp or datetime)
            end_time: End time (timestamp or datetime)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory entries within the timeframe
        """
        try:
            self.stats["total_requests"] += 1
            
            # Convert datetime to timestamp if needed
            if isinstance(start_time, datetime):
                start_time = start_time.timestamp()
            if isinstance(end_time, datetime):
                end_time = end_time.timestamp()
            
            # Get tensor connection
            tensor_conn = await self.get_tensor_connection()
            
            # Prepare request
            request = {
                "type": "get_by_timeframe",
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
                "client_id": "memory_client",
                "message_id": f"timeframe_{int(time.time() * 1000)}",
                "timestamp": time.time()
            }
            
            # Send request and get response
            await tensor_conn.send(json.dumps(request))
            response = await tensor_conn.recv()
            response_data = json.loads(response)
            
            if "type" in response_data and response_data["type"] == "error":
                self.logger.error(f"Error retrieving memories by timeframe: {response_data.get('error')}")
                self.stats["failed_requests"] += 1
                return []
            
            # Extract memories
            memories = []
            for memory_data in response_data.get("memories", []):
                memory = MemoryEntry(
                    id=memory_data.get("id", f"unknown_{int(time.time())}"),
                    content=memory_data.get("content", ""),
                    memory_type=memory_data.get("type", "general"),
                    created_at=memory_data.get("timestamp", time.time()),
                    significance=memory_data.get("significance", 0.5),
                    metadata=memory_data.get("metadata", {})
                )
                
                # Add to cache
                self._add_to_cache(memory)
                
                memories.append(memory)
            
            self.stats["successful_requests"] += 1
            return memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories by timeframe: {e}")
            self.stats["failed_requests"] += 1
            return []
    
    def _add_to_cache(self, memory: MemoryEntry) -> None:
        """
        Add a memory to the cache, removing oldest entries if needed.
        
        Args:
            memory: Memory entry to add to cache
        """
        # Add or move to front of cache
        self.memory_cache[memory.id] = memory
        
        # Remove oldest entries if cache is too large
        if len(self.memory_cache) > self.cache_size:
            # Find oldest access time
            oldest_id = None
            oldest_time = float('inf')
            
            for mem_id, mem in self.memory_cache.items():
                if mem.last_access < oldest_time:
                    oldest_time = mem.last_access
                    oldest_id = mem_id
            
            # Remove oldest
            if oldest_id:
                del self.memory_cache[oldest_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory client statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
            "embedding_requests": self.stats["embedding_requests"],
            "retrieval_requests": self.stats["retrieval_requests"],
            "search_requests": self.stats["search_requests"],
            "cache_size": len(self.memory_cache),
            "cache_limit": self.cache_size,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
            "tensor_server_url": self.tensor_server_url,
            "hpc_server_url": self.hpc_server_url
        }