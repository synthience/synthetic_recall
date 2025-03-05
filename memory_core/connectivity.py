# memory_core/connectivity.py

import json
import asyncio
import websockets
import logging
from typing import Optional, Dict, Any
import time
import traceback

logger = logging.getLogger(__name__)

class ConnectivityMixin:
    """
    Mixin that handles WebSocket connectivity to the tensor and HPC servers.
    Requires self._tensor_lock, self._hpc_lock, etc. from the base class.
    """

    async def connect(self) -> bool:
        """Connect to the tensor and HPC servers."""
        if self._connected:
            return True

        logger.info("Connecting to tensor and HPC servers")
        tensor_connected = await self._connect_to_tensor_server()
        hpc_connected = await self._connect_to_hpc_server()
        
        self._connected = tensor_connected and hpc_connected
        return self._connected
    
    async def _connect_to_tensor_server(self) -> bool:
        """Connect to the tensor server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to tensor server at {self.tensor_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._tensor_lock:
                    self._tensor_connection = connection
                    
                logger.info("Successfully connected to tensor server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to tensor server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for tensor server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to tensor server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _connect_to_hpc_server(self) -> bool:
        """Connect to the HPC server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to HPC server at {self.hpc_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._hpc_lock:
                    self._hpc_connection = connection
                    
                logger.info("Successfully connected to HPC server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to HPC server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for HPC server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to HPC server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _get_tensor_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the tensor server connection, creating a new one if necessary."""
        async with self._tensor_lock:
            # Check if connection exists and is open
            if self._tensor_connection and hasattr(self._tensor_connection, 'open') and self._tensor_connection.open:
                try:
                    # Verify connection is actually responsive with a ping
                    pong_waiter = await self._tensor_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=2.0)
                    return self._tensor_connection
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning("Tensor connection ping failed, will create new connection")
                    # Continue to create new connection
                except Exception as e:
                    logger.warning(f"Tensor connection check failed: {e}")
                    # Continue to create new connection
                    
            # Connection closed or doesn't exist, create new one
            try:
                # Add exponential backoff for reconnection attempts
                retry_count = 0
                max_retries = 3
                base_delay = 0.5
                
                while retry_count < max_retries:
                    try:
                        logger.info("Creating new tensor server connection")
                        connection = await asyncio.wait_for(
                            websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                            timeout=self.connection_timeout
                        )
                        self._tensor_connection = connection
                        return connection
                    except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
                        wait_time = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"Tensor connection attempt {retry_count} failed: {e}. Retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Failed to create tensor connection after retries: {e}")
                return None
    
    async def _get_hpc_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the HPC server connection, creating a new one if necessary."""
        async with self._hpc_lock:
            # Check if connection exists and is open
            if self._hpc_connection and hasattr(self._hpc_connection, 'open') and self._hpc_connection.open:
                try:
                    # Verify connection is actually responsive with a ping
                    pong_waiter = await self._hpc_connection.ping()
                    await asyncio.wait_for(pong_waiter, timeout=2.0)
                    return self._hpc_connection
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.warning("HPC connection ping failed, will create new connection")
                    # Continue to create new connection
                except Exception as e:
                    logger.warning(f"HPC connection check failed: {e}")
                    # Continue to create new connection
                    
            # Connection closed or doesn't exist, create new one
            try:
                # Add exponential backoff for reconnection attempts
                retry_count = 0
                max_retries = 3
                base_delay = 0.5
                
                while retry_count < max_retries:
                    try:
                        logger.info("Creating new HPC server connection")
                        connection = await asyncio.wait_for(
                            websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                            timeout=self.connection_timeout
                        )
                        self._hpc_connection = connection
                        return connection
                    except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise
                        wait_time = base_delay * (2 ** (retry_count - 1))
                        logger.warning(f"HPC connection attempt {retry_count} failed: {e}. Retrying in {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Failed to create HPC connection after retries: {e}")
                return None
