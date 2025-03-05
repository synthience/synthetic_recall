# memory_core/base.py

import asyncio
import logging
import time
import uuid
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BaseMemoryClient:
    """
    Base class that sets up fundamental fields and structure. 
    Other mixins will extend this to build the full EnhancedMemoryClient.
    """
    def __init__(
        self,
        session_id: str,
        user_id: str = "default_user",
        tensor_server_url: str = "ws://localhost:5001",
        hpc_server_url: str = "ws://localhost:5005",
        enable_persistence: bool = True,
        persistence_dir: str = "data/memory",
        significance_threshold: float = 0.0
    ):
        # Connection info
        self.tensor_server_url = tensor_server_url
        self.hpc_server_url = hpc_server_url
        
        # Session identifiers
        self.session_id = session_id
        self.user_id = user_id
        
        # Persistence settings
        self.enable_persistence = enable_persistence
        self.persistence_dir = persistence_dir
        self.significance_threshold = significance_threshold
        
        # Connection state
        self._connected = False
        self._tensor_connection = None
        self._hpc_connection = None
        self._closing = False
        
        # Connection settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.connection_timeout = 10.0
        self.ping_interval = 30.0
        
        # Locks for thread safety
        self._tensor_lock = asyncio.Lock()
        self._hpc_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        
        # For background tasks
        self._background_tasks = set()
        self._memory_management_task = None
        
        # Memory state
        self.memories = []
        self.topics_discussed = set()
        
        # Config
        self.max_memory_age = 86400 * 30  # 30 days in seconds
        
    async def initialize(self) -> bool:
        """
        Initialize the memory client and connect to the servers.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Connect to servers
            success = await self.connect()
            if not success:
                logger.error("Failed to connect to memory servers")
                return False
                
            # Create persistence directory if enabled
            if self.enable_persistence and not os.path.exists(self.persistence_dir):
                os.makedirs(self.persistence_dir, exist_ok=True)
                logger.info(f"Created persistence directory: {self.persistence_dir}")
                
            # Start background task for memory management
            self._start_memory_management()
            
            logger.info(f"Memory client initialized (session: {self.session_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing memory client: {e}")
            return False
            
    def _start_memory_management(self):
        """
        Start the background task for memory management.
        This runs asynchronously to handle periodic tasks like:
        - Consolidating memories
        - Pruning old memories
        - Periodic persistence
        """
        if self._memory_management_task is None or self._memory_management_task.done():
            self._memory_management_task = asyncio.create_task(
                self._memory_management_loop()
            )
            self._background_tasks.add(self._memory_management_task)
            self._memory_management_task.add_done_callback(
                self._background_tasks.discard
            )
            
    async def _memory_management_loop(self):
        """
        Background loop for memory management tasks.
        """
        try:
            while not self._closing:
                try:
                    # Persist memories if enabled
                    if self.enable_persistence:
                        await self._persist_memories()
                        
                    # Prune old memories
                    await self._prune_old_memories()
                    
                    # Wait for next cycle (every hour)
                    await asyncio.sleep(3600)
                    
                except asyncio.CancelledError:
                    logger.info("Memory management task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in memory management loop: {e}")
                    await asyncio.sleep(60)  # Wait and retry on error
                    
        except asyncio.CancelledError:
            logger.info("Memory management loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in memory management loop: {e}")
            
    async def close(self):
        """
        Close the memory client and clean up resources.
        """
        logger.info("Closing memory client")
        self._closing = True
        
        # Cancel background tasks
        if self._memory_management_task and not self._memory_management_task.done():
            self._memory_management_task.cancel()
            try:
                await self._memory_management_task
            except asyncio.CancelledError:
                pass
                
        # Persist memories before closing if enabled
        if self.enable_persistence:
            await self._persist_memories()
            
        # Close connections
        await self._close_connections()
        
    async def _close_connections(self):
        """
        Close WebSocket connections to servers.
        """
        try:
            async with self._tensor_lock:
                if self._tensor_connection:
                    await self._tensor_connection.close()
                    self._tensor_connection = None
                    
            async with self._hpc_lock:
                if self._hpc_connection:
                    await self._hpc_connection.close()
                    self._hpc_connection = None
                    
            self._connected = False
            logger.info("Closed all server connections")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
            
    async def _persist_memories(self):
        """
        Persist memories to disk.
        """
        try:
            if not self.enable_persistence or not self.memories:
                return
                
            async with self._memory_lock:
                # Filter memories by significance threshold
                memories_to_save = [
                    mem for mem in self.memories 
                    if mem.get('significance', 0.0) >= self.significance_threshold
                ]
                
                if not memories_to_save:
                    return
                    
                # Save to file
                file_path = os.path.join(
                    self.persistence_dir, 
                    f"{self.user_id}_{self.session_id}_memories.json"
                )
                
                with open(file_path, 'w') as f:
                    json.dump(memories_to_save, f)
                    
                logger.info(f"Persisted {len(memories_to_save)} memories to {file_path}")
                
        except Exception as e:
            logger.error(f"Error persisting memories: {e}")
            
    async def _prune_old_memories(self):
        """
        Remove memories older than max_memory_age.
        """
        try:
            now = time.time()
            
            async with self._memory_lock:
                original_count = len(self.memories)
                
                # Filter out old memories
                self.memories = [
                    mem for mem in self.memories 
                    if now - mem.get('timestamp', now) < self.max_memory_age
                ]
                
                pruned_count = original_count - len(self.memories)
                if pruned_count > 0:
                    logger.info(f"Pruned {pruned_count} old memories")
                    
        except Exception as e:
            logger.error(f"Error pruning old memories: {e}")
            
    def _get_timestamp(self) -> float:
        """
        Get current timestamp in seconds.
        
        Returns:
            float: Current Unix timestamp
        """
        return time.time()
