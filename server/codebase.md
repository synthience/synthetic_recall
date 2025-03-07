# __init__.py

```py


```

# chat_processor.py

```py
import torch
import time
import logging
from typing import Dict, Any, List
from memory_index import MemoryIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatProcessor:
    def __init__(self, memory_index: MemoryIndex, config: Dict[str, Any] = None):
        """Initialize chat processor with memory integration."""
        self.config = {
            'max_memories': 5,
            'min_similarity': 0.7,
            'time_decay': 0.01,
            'significance_threshold': 0.5
        }
        if config:
            self.config.update(config)
            
        self.memory_index = memory_index

    async def retrieve_context(self, query_embedding: torch.Tensor) -> List[Dict]:
        """Retrieve relevant memories with significance and time weighting."""
        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
            query_norm = torch.norm(query_embedding, p=2)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
        
        memories = self.memory_index.search(
            query_embedding,
            k=self.config['max_memories']
        )
        
        logger.info(f"Retrieved {len(memories)} memories before filtering")
        for i, m in enumerate(memories):
            logger.info(f"Memory {i}: similarity={m['similarity']:.3f}, content={m['memory'].get('content', 'None')}")
        
        # Filter by similarity threshold
        filtered_memories = [
            m for m in memories 
            if m['similarity'] >= self.config['min_similarity'] 
            and m['memory'].get('content')
        ]
        
        logger.info(f"Filtered to {len(filtered_memories)} memories")
        
        # Sort by significance and similarity
        filtered_memories.sort(
            key=lambda x: (x['memory']['significance'], x['similarity']),
            reverse=True
        )
        
        return filtered_memories

    async def process_chat(self, user_input: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Process chat with memory integration."""
        context = await self.retrieve_context(embedding)
        messages = []
        
        logger.info(f"Processing chat with {len(context)} context memories")
        
        # Add memory context if available
        if context:
            memory_texts = []
            for memory in context:
                if memory['memory'].get('content'):
                    memory_texts.append(
                        f"Previous Memory (Significance: {memory['memory']['significance']:.2f}, "
                        f"Similarity: {memory['similarity']:.2f}):\n"
                        f"{memory['memory']['content']}"
                    )
            
            if memory_texts:
                logger.info(f"Adding {len(memory_texts)} memories to context")
                messages.append({
                    "role": "system",
                    "content": "Relevant context:\n" + "\n\n".join(memory_texts)
                })
            else:
                logger.warning("No memory texts generated despite having context")
        else:
            logger.warning("No context memories retrieved")
        
        # Add user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare response
        response = {
            "messages": messages,
            "model": "qwen2.5-7b-instruct",
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False
        }
        
        return response
```

# hpc_server.py

```py
import asyncio
import websockets
import json
import logging
import torch

# Example HPC manager (you'll see the real code below in hpc_sig_flow_manager.py)
from hpc_sig_flow_manager import HPCSIGFlowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCServer:
    """
    HPCServer listens on ws://0.0.0.0:5005
    Expects JSON messages:
      {
        "type": "process",
        "embeddings": [...]
      }
    or
      {
        "type": "stats"
      }
    """
    def __init__(self, host='0.0.0.0', port=5005):
        self.host = host
        self.port = port
        # HPC manager that does hypothetical processing
        self.hpc_sig_manager = HPCSIGFlowManager({
            'embedding_dim': 384
        })
        logger.info("Initialized HPCServer with HPC-SIG manager")

    def get_stats(self):
        # Return HPC state
        return { 
            'type': 'stats',
            **self.hpc_sig_manager.get_stats()
        }

    async def handle_websocket(self, websocket):
        logger.info(f"New connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'process':
                        # Perform HPC pipeline
                        embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
                        processed_embedding, significance = await self.hpc_sig_manager.process_embedding(embeddings)

                        # Example response
                        response = {
                            'type': 'processed',
                            'embeddings': processed_embedding.tolist(),
                            'significance': significance
                        }
                        logger.info(f"Sending HPC response: {response}")
                        await websocket.send(json.dumps(response))

                    elif data['type'] == 'stats':
                        stats = self.get_stats()
                        await websocket.send(json.dumps(stats))

                    else:
                        # Unknown message type
                        error_msg = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type: {data['type']}")
                        await websocket.send(json.dumps(error_msg))

                except Exception as e:
                    err = {'type': 'error', 'error': str(e)}
                    logger.error(f"Error handling HPC message: {str(e)}")
                    await websocket.send(json.dumps(err))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")

        except Exception as e:
            logger.error(f"Unexpected HPC server error: {str(e)}")

    async def start(self):
        logger.info(f"Starting HPC server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"HPC Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # keep running

if __name__ == '__main__':
    server = HPCServer()
    asyncio.run(server.start())

```

# memory_adapter.py

```py
"""
LUCID RECALL PROJECT
Memory Adapter

This adapter integrates the new unified components with the existing system
without requiring Docker rebuilds or modification of existing code.
"""

import logging
import asyncio
import importlib.util
import os
import sys
import time
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

class MemoryAdapter:
    """
    Adapter for integrating new unified memory components with existing system.
    
    This adapter provides a non-invasive way to use the new unified components
    alongside the existing system without requiring Docker rebuilds or changing
    the original code. It detects what components are available and gracefully
    falls back to existing implementations when needed.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Configuration options including:
                - prefer_unified: Use unified components when available
                - adapter_mode: 'shadow', 'redirect', or 'hybrid'
                - fallback_on_error: Use original components on error
        """
        self.config = {
            'prefer_unified': True,
            'adapter_mode': 'hybrid',  # shadow = run both, redirect = replace, hybrid = gradual transition
            'fallback_on_error': True,
            'log_performance': True,
            'integration_path': os.path.dirname(os.path.abspath(__file__)),
            **(config or {})
        }
        
        # Available components tracking
        self.available_components = {
            'unified_hpc': False,
            'standard_websocket': False,
            'unified_storage': False,
            'unified_significance': False
        }
        
        # Component references
        self.unified_hpc = None
        self.standard_websocket = None
        self.unified_storage = None
        self.unified_significance = None
        
        # Original component references
        self.original_hpc = None
        self.original_websocket = None
        self.original_storage = None
        self.original_significance = None
        
        # Initialize
        self._discover_components()
        
        logger.info(f"Memory adapter initialized in {self.config['adapter_mode']} mode")
        logger.info(f"Available unified components: {[k for k, v in self.available_components.items() if v]}")
    
    def _discover_components(self) -> None:
        """Discover available components without importing directly."""
        integration_path = self.config['integration_path']
        
        # Check for unified HPC flow manager
        component_path = os.path.join(integration_path, 'unified_hpc_flow_manager.py')
        if os.path.exists(component_path):
            try:
                spec = importlib.util.spec_from_file_location("unified_hpc_flow_manager", component_path)
                unified_hpc_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(unified_hpc_module)
                self.unified_hpc = unified_hpc_module.UnifiedHPCFlowManager
                self.available_components['unified_hpc'] = True
                logger.info("Unified HPC flow manager available")
            except Exception as e:
                logger.warning(f"Failed to load unified HPC flow manager: {e}")
        
        # Check for standard websocket interface
        component_path = os.path.join(integration_path, 'standard_websocket_interface.py')
        if os.path.exists(component_path):
            try:
                spec = importlib.util.spec_from_file_location("standard_websocket_interface", component_path)
                standard_websocket_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(standard_websocket_module)
                self.standard_websocket = standard_websocket_module.StandardWebSocketInterface
                self.available_components['standard_websocket'] = True
                logger.info("Standard websocket interface available")
            except Exception as e:
                logger.warning(f"Failed to load standard websocket interface: {e}")
        
        # Check for unified memory storage
        component_path = os.path.join(integration_path, 'unified_memory_storage.py')
        if os.path.exists(component_path):
            try:
                spec = importlib.util.spec_from_file_location("unified_memory_storage", component_path)
                unified_storage_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(unified_storage_module)
                self.unified_storage = unified_storage_module.UnifiedMemoryStorage
                self.available_components['unified_storage'] = True
                logger.info("Unified memory storage available")
            except Exception as e:
                logger.warning(f"Failed to load unified memory storage: {e}")
        
        # Check for unified significance calculator
        component_path = os.path.join(integration_path, 'significance_calculator.py')
        if os.path.exists(component_path):
            try:
                spec = importlib.util.spec_from_file_location("significance_calculator", component_path)
                unified_significance_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(unified_significance_module)
                self.unified_significance = unified_significance_module.UnifiedSignificanceCalculator
                self.available_components['unified_significance'] = True
                logger.info("Unified significance calculator available")
            except Exception as e:
                logger.warning(f"Failed to load unified significance calculator: {e}")
    
    async def get_hpc_manager(self, original_class=None, config: Dict[str, Any] = None) -> Any:
        """
        Get appropriate HPC flow manager based on configuration.
        
        Args:
            original_class: Original class reference if available
            config: Configuration for the component
            
        Returns:
            HPC flow manager instance
        """
        # Store original class reference
        if original_class and not self.original_hpc:
            self.original_hpc = original_class
        
        # Determine which implementation to use
        if self.available_components['unified_hpc'] and self.config['prefer_unified']:
            try:
                # Use unified implementation
                instance = self.unified_hpc(config)
                logger.info("Using unified HPC flow manager")
                return HPCAdapter(instance, original_class(config) if original_class else None, self.config)
            except Exception as e:
                logger.error(f"Error creating unified HPC manager: {e}")
                if self.config['fallback_on_error'] and original_class:
                    logger.info("Falling back to original HPC manager")
                    return original_class(config)
                raise
        elif original_class:
            # Use original implementation
            logger.info("Using original HPC flow manager")
            return original_class(config)
        else:
            raise ValueError("No HPC flow manager implementation available")
    
    async def get_websocket_interface(self, original_class=None, host: str = "0.0.0.0", port: int = 5000) -> Any:
        """
        Get appropriate WebSocket interface based on configuration.
        
        Args:
            original_class: Original class reference if available
            host: Host address to bind to
            port: Port to listen on
            
        Returns:
            WebSocket interface instance
        """
        # Store original class reference
        if original_class and not self.original_websocket:
            self.original_websocket = original_class
        
        # Determine which implementation to use
        if self.available_components['standard_websocket'] and self.config['prefer_unified']:
            try:
                # Use unified implementation
                instance = self.standard_websocket(host=host, port=port)
                logger.info(f"Using standard websocket interface on {host}:{port}")
                
                # In shadow mode, also create original
                if self.config['adapter_mode'] == 'shadow' and original_class:
                    shadow_port = port + 1  # Use next port for shadow
                    shadow_instance = original_class(host=host, port=shadow_port)
                    return WebSocketAdapter(instance, shadow_instance, self.config)
                
                return WebSocketAdapter(instance, None, self.config)
            except Exception as e:
                logger.error(f"Error creating standard websocket interface: {e}")
                if self.config['fallback_on_error'] and original_class:
                    logger.info("Falling back to original websocket implementation")
                    return original_class(host=host, port=port)
                raise
        elif original_class:
            # Use original implementation
            logger.info(f"Using original websocket implementation on {host}:{port}")
            return original_class(host=host, port=port)
        else:
            raise ValueError("No websocket interface implementation available")
    
    async def get_memory_storage(self, original_class=None, config: Dict[str, Any] = None) -> Any:
        """
        Get appropriate memory storage based on configuration.
        
        Args:
            original_class: Original class reference if available
            config: Configuration for the component
            
        Returns:
            Memory storage instance
        """
        # Store original class reference
        if original_class and not self.original_storage:
            self.original_storage = original_class
        
        # Determine which implementation to use
        if self.available_components['unified_storage'] and self.config['prefer_unified']:
            try:
                # Use unified implementation
                instance = self.unified_storage(config)
                logger.info("Using unified memory storage")
                
                # In hybrid mode, also create original but only for reads
                if self.config['adapter_mode'] == 'hybrid' and original_class:
                    original_instance = original_class(config)
                    return StorageAdapter(instance, original_instance, self.config)
                
                return StorageAdapter(instance, None, self.config)
            except Exception as e:
                logger.error(f"Error creating unified memory storage: {e}")
                if self.config['fallback_on_error'] and original_class:
                    logger.info("Falling back to original memory storage")
                    return original_class(config)
                raise
        elif original_class:
            # Use original implementation
            logger.info("Using original memory storage")
            return original_class(config)
        else:
            raise ValueError("No memory storage implementation available")
    
    async def get_significance_calculator(self, original_class=None, config: Dict[str, Any] = None) -> Any:
        """
        Get appropriate significance calculator based on configuration.
        
        Args:
            original_class: Original class reference if available
            config: Configuration for the component
            
        Returns:
            Significance calculator instance
        """
        # Store original class reference
        if original_class and not self.original_significance:
            self.original_significance = original_class
        
        # Determine which implementation to use
        if self.available_components['unified_significance'] and self.config['prefer_unified']:
            try:
                # Use unified implementation
                instance = self.unified_significance(config)
                logger.info("Using unified significance calculator")
                
                # In shadow mode, also create original
                if self.config['adapter_mode'] == 'shadow' and original_class:
                    original_instance = original_class(config)
                    return SignificanceAdapter(instance, original_instance, self.config)
                
                return SignificanceAdapter(instance, None, self.config)
            except Exception as e:
                logger.error(f"Error creating unified significance calculator: {e}")
                if self.config['fallback_on_error'] and original_class:
                    logger.info("Falling back to original significance calculator")
                    return original_class(config)
                raise
        elif original_class:
            # Use original implementation
            logger.info("Using original significance calculator")
            return original_class(config)
        else:
            raise ValueError("No significance calculator implementation available")


# Adapter classes for each component type

class HPCAdapter:
    """Adapter for HPC flow manager."""
    
    def __init__(self, unified_instance, original_instance, config):
        self.unified = unified_instance
        self.original = original_instance
        self.config = config
        self.mode = config['adapter_mode']
        self.logger = logging.getLogger(__name__)
    
    async def process_embedding(self, embedding):
        """Process embedding through appropriate implementation."""
        start_time = time.time()
        
        try:
            if self.mode == 'shadow' and self.original:
                # Run both and compare
                unified_future = asyncio.ensure_future(self.unified.process_embedding(embedding))
                original_future = asyncio.ensure_future(self.original.process_embedding(embedding))
                
                # Wait for both to complete
                await asyncio.gather(unified_future, original_future)
                
                # Get results
                unified_result = unified_future.result()
                original_result = original_future.result()
                
                # Log comparison
                if self.config['log_performance']:
                    self.logger.info(f"HPC comparison - unified: {unified_result[1]:.4f}, original: {original_result[1]:.4f}")
                
                # Return unified result
                return unified_result
            else:
                # Just use unified implementation
                result = await self.unified.process_embedding(embedding)
                
                if self.config['log_performance']:
                    self.logger.debug(f"HPC processing time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in HPC adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original HPC implementation")
                return await self.original.process_embedding(embedding)
            
            raise
    
    def get_stats(self):
        """Get statistics from appropriate implementation."""
        if self.mode == 'shadow' and self.original:
            # Combine stats
            unified_stats = self.unified.get_stats()
            original_stats = self.original.get_stats()
            
            return {
                'unified': unified_stats,
                'original': original_stats,
                'adapter_mode': self.mode
            }
        else:
            return self.unified.get_stats()


class WebSocketAdapter:
    """Adapter for WebSocket interface."""
    
    def __init__(self, unified_instance, original_instance, config):
        self.unified = unified_instance
        self.original = original_instance
        self.config = config
        self.mode = config['adapter_mode']
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the appropriate WebSocket server."""
        try:
            if self.mode == 'shadow' and self.original:
                # Start both servers
                unified_future = asyncio.ensure_future(self.unified.start())
                original_future = asyncio.ensure_future(self.original.start())
                
                # Create task to monitor both
                monitor_task = asyncio.create_task(self._monitor_servers(unified_future, original_future))
                
                # Only await unified server to return control flow
                await self.unified.start()
            else:
                # Just start unified server
                await self.unified.start()
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original WebSocket implementation")
                await self.original.start()
            else:
                raise
    
    async def _monitor_servers(self, unified_future, original_future):
        """Monitor both servers and restart if needed."""
        while True:
            if unified_future.done():
                exception = unified_future.exception()
                if exception:
                    self.logger.error(f"Unified WebSocket server crashed: {exception}")
                    # Restart unified server
                    unified_future = asyncio.ensure_future(self.unified.start())
                
            if original_future.done():
                exception = original_future.exception()
                if exception:
                    self.logger.error(f"Original WebSocket server crashed: {exception}")
                    # Restart original server
                    original_future = asyncio.ensure_future(self.original.start())
            
            # Check every 5 seconds
            await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the appropriate WebSocket server."""
        try:
            if self.mode == 'shadow' and self.original:
                # Stop both servers
                await asyncio.gather(
                    self.unified.stop(),
                    self.original.stop()
                )
            else:
                # Just stop unified server
                await self.unified.stop()
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
            raise
    
    async def broadcast(self, message, exclude_clients=None):
        """Broadcast message to clients."""
        try:
            if self.mode == 'shadow' and self.original:
                # Send to both servers
                await asyncio.gather(
                    self.unified.broadcast(message, exclude_clients),
                    self.original.broadcast(message, exclude_clients)
                )
            else:
                # Just send to unified server
                await self.unified.broadcast(message, exclude_clients)
        except Exception as e:
            self.logger.error(f"Error broadcasting message: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original WebSocket implementation")
                await self.original.broadcast(message, exclude_clients)
            else:
                raise
    
    def get_stats(self):
        """Get statistics from appropriate implementation."""
        if self.mode == 'shadow' and self.original:
            # Combine stats
            unified_stats = self.unified.get_stats()
            original_stats = self.original.get_stats()
            
            return {
                'unified': unified_stats,
                'original': original_stats,
                'adapter_mode': self.mode
            }
        else:
            return self.unified.get_stats()


class StorageAdapter:
    """Adapter for memory storage."""
    
    def __init__(self, unified_instance, original_instance, config):
        self.unified = unified_instance
        self.original = original_instance
        self.config = config
        self.mode = config['adapter_mode']
        self.logger = logging.getLogger(__name__)
    
    async def store(self, memory):
        """Store a memory."""
        start_time = time.time()
        
        try:
            if self.mode == 'hybrid' and self.original:
                # Store in both - start with unified
                unified_result = await self.unified.store(memory)
                
                # Also store in original as backup
                try:
                    await self.original.store(memory)
                except Exception as e:
                    self.logger.warning(f"Failed to store in original storage: {e}")
                
                if self.config['log_performance']:
                    self.logger.debug(f"Storage time: {(time.time() - start_time)*1000:.2f}ms")
                
                return unified_result
            else:
                # Just use unified implementation
                result = await self.unified.store(memory)
                
                if self.config['log_performance']:
                    self.logger.debug(f"Storage time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in storage adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original storage implementation")
                return await self.original.store(memory)
            
            raise
    
    async def retrieve(self, memory_id):
        """Retrieve a memory by ID."""
        start_time = time.time()
        
        try:
            if self.mode == 'hybrid' and self.original:
                # Try unified first
                unified_result = await self.unified.retrieve(memory_id)
                
                # If not found, try original
                if not unified_result:
                    original_result = await self.original.retrieve(memory_id)
                    
                    # If found in original but not unified, copy to unified
                    if original_result:
                        self.logger.info(f"Found memory {memory_id} in original storage, copying to unified")
                        await self.unified.store(original_result)
                        
                    return original_result
                
                if self.config['log_performance']:
                    self.logger.debug(f"Retrieval time: {(time.time() - start_time)*1000:.2f}ms")
                
                return unified_result
            else:
                # Just use unified implementation
                result = await self.unified.retrieve(memory_id)
                
                if self.config['log_performance']:
                    self.logger.debug(f"Retrieval time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in storage adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original storage implementation")
                return await self.original.retrieve(memory_id)
            
            raise
    
    async def search(self, **kwargs):
        """Search for memories."""
        start_time = time.time()
        
        try:
            if self.mode == 'hybrid' and self.original:
                # Try unified first
                unified_results = await self.unified.search(**kwargs)
                
                # Also search original to potentially find memories not yet migrated
                try:
                    original_results = await self.original.search(**kwargs)
                    
                    # Combine results, preferring unified
                    unified_ids = set(result[0].id for result in unified_results)
                    for result in original_results:
                        if result[0].id not in unified_ids:
                            unified_results.append(result)
                            # Copy to unified for future queries
                            await self.unified.store(result[0])
                except Exception as e:
                    self.logger.warning(f"Failed to search original storage: {e}")
                
                if self.config['log_performance']:
                    self.logger.debug(f"Search time: {(time.time() - start_time)*1000:.2f}ms")
                
                return unified_results
            else:
                # Just use unified implementation
                result = await self.unified.search(**kwargs)
                
                if self.config['log_performance']:
                    self.logger.debug(f"Search time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in storage adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original storage implementation")
                return await self.original.search(**kwargs)
            
            raise
    
    def get_stats(self):
        """Get statistics from appropriate implementation."""
        if self.mode == 'hybrid' and self.original:
            # Combine stats
            unified_stats = self.unified.get_stats()
            original_stats = self.original.get_stats()
            
            return {
                'unified': unified_stats,
                'original': original_stats,
                'adapter_mode': self.mode
            }
        else:
            return self.unified.get_stats()


class SignificanceAdapter:
    """Adapter for significance calculator."""
    
    def __init__(self, unified_instance, original_instance, config):
        self.unified = unified_instance
        self.original = original_instance
        self.config = config
        self.mode = config['adapter_mode']
        self.logger = logging.getLogger(__name__)
        
        # Tracking for comparison
        self.comparison_stats = {
            'total_calculations': 0,
            'mean_difference': 0.0,
            'max_difference': 0.0,
            'within_threshold': 0,
            'threshold': 0.1  # Maximum acceptable difference
        }
    
    async def calculate(self, embedding=None, text=None, context=None):
        """Calculate significance score."""
        start_time = time.time()
        
        try:
            if self.mode == 'shadow' and self.original:
                # Run both and compare
                unified_task = asyncio.create_task(self.unified.calculate(embedding, text, context))
                original_task = asyncio.create_task(self.original.calculate(embedding, text, context))
                
                # Wait for both to complete
                unified_result = await unified_task
                original_result = await original_task
                
                # Compare results
                self._compare_results(unified_result, original_result)
                
                if self.config['log_performance']:
                    self.logger.debug(f"Significance calculation time: {(time.time() - start_time)*1000:.2f}ms")
                
                # Return unified result
                return unified_result
            else:
                # Just use unified implementation
                result = await self.unified.calculate(embedding, text, context)
                
                if self.config['log_performance']:
                    self.logger.debug(f"Significance calculation time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in significance adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original significance implementation")
                return await self.original.calculate(embedding, text, context)
            
            raise
    
    def _compare_results(self, unified_result, original_result):
        """Compare results from both implementations."""
        difference = abs(unified_result - original_result)
        
        # Update comparison stats
        self.comparison_stats['total_calculations'] += 1
        self.comparison_stats['max_difference'] = max(self.comparison_stats['max_difference'], difference)
        
        # Update running average
        current_mean = self.comparison_stats['mean_difference']
        n = self.comparison_stats['total_calculations']
        self.comparison_stats['mean_difference'] = current_mean + (difference - current_mean) / n
        
        # Check if within threshold
        if difference <= self.comparison_stats['threshold']:
            self.comparison_stats['within_threshold'] += 1
            
        # Log large differences
        if difference > self.comparison_stats['threshold']:
            self.logger.warning(f"Large significance difference: {difference:.4f} (unified={unified_result:.4f}, original={original_result:.4f})")
    
    def get_stats(self):
        """Get statistics from appropriate implementation."""
        if self.mode == 'shadow' and self.original:
            # Combine stats
            unified_stats = self.unified.get_stats()
            original_stats = self.original.get_stats()
            
            return {
                'unified': unified_stats,
                'original': original_stats,
                'comparison': self.comparison_stats,
                'adapter_mode': self.mode
            }
        else:
            return self.unified.get_stats()
```

# memory_core.py

```py
"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:43 PM EST

MemoryCore: Core memory system with HPC integration
"""

import torch
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from ..server.hpc_flow_manager import HPCFlowManager
from .memory_types import MemoryTypes, MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryCore:
    def __init__(self, config: Dict[str, Any]):
        self.config = {
            'dimension': 768,
            'max_size': 10000,
            'batch_size': 32,
            'cleanup_threshold': 0.7,
            'memory_path': Path('/workspace/memory/stored'),
            **(config or {})
        }
        
        # Initialize memory storage
        self.memories = defaultdict(list)
        self.total_memories = 0
        
        # Initialize HPC Manager
        self.hpc_manager = HPCFlowManager(config)
        
        # Performance tracking
        self.last_cleanup_time = time.time()
        self.stats = {
            'processed': 0,
            'stored': 0,
            'cleaned': 0
        }
        
        logger.info(f"Initialized MemoryCore with config: {self.config}")
        
    async def process_and_store(self, embedding: torch.Tensor, memory_type: MemoryTypes) -> bool:
        """Process embedding through HPC pipeline and store if significant"""
        try:
            # Process through HPC pipeline
            processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
            
            self.stats['processed'] += 1
            
            # Store if significant enough
            if significance > self.config['cleanup_threshold']:
                success = self._store_memory(MemoryEntry(
                    embedding=processed_embedding,
                    memory_type=memory_type,
                    significance=significance,
                    timestamp=time.time()
                ))
                
                if success:
                    self.stats['stored'] += 1
                    
                # Run cleanup if needed
                await self._maybe_cleanup()
                
                return success
                
            return False
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {str(e)}")
            return False
            
    def _store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry in the appropriate type bucket"""
        try:
            # Check if we have room
            if self.total_memories >= self.config['max_size']:
                return False
                
            # Add to appropriate bucket
            self.memories[memory.memory_type].append(memory)
            self.total_memories += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            return False
            
    async def _maybe_cleanup(self):
        """Run cleanup if memory usage is high"""
        current_time = time.time()
        
        # Only clean up periodically
        if (current_time - self.last_cleanup_time < 3600 and  # 1 hour
            self.total_memories < self.config['max_size'] * 0.9):  # 90% full
            return
            
        await self._cleanup()
        
    async def _cleanup(self):
        """Remove least significant memories when storage is full"""
        try:
            logger.info("Starting memory cleanup...")
            
            # Sort all memories by significance
            all_memories = []
            for type_memories in self.memories.values():
                all_memories.extend(type_memories)
                
            all_memories.sort(key=lambda x: x.significance)
            
            # Remove bottom 20%
            num_to_remove = len(all_memories) // 5
            memories_to_keep = all_memories[num_to_remove:]
            
            # Reset storage
            self.memories = defaultdict(list)
            self.total_memories = 0
            
            # Re-add memories to keep
            for memory in memories_to_keep:
                self._store_memory(memory)
                
            self.stats['cleaned'] += num_to_remove
            self.last_cleanup_time = time.time()
            
            logger.info(f"Cleanup complete. Removed {num_to_remove} memories")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def get_recent_memories(self, count: int = 5, memory_type: Optional[MemoryTypes] = None) -> List[MemoryEntry]:
        """Get most recent memories, optionally filtered by type"""
        try:
            if memory_type:
                memories = self.memories[memory_type]
            else:
                memories = []
                for type_memories in self.memories.values():
                    memories.extend(type_memories)
                    
            # Sort by timestamp descending
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return memories[:count]
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory system statistics"""
        return {
            'total_memories': self.total_memories,
            'memory_types': {k: len(v) for k, v in self.memories.items()},
            'processed': self.stats['processed'],
            'stored': self.stats['stored'],
            'cleaned': self.stats['cleaned'],
            'last_cleanup': self.last_cleanup_time,
            'hpc_stats': self.hpc_manager.get_stats()
        }
```

# memory_index.py

```py
import torch
import numpy as np
import time

class MemoryIndex:
    def __init__(self, embedding_dim=384, rebuild_threshold=100, time_decay=0.01, min_similarity=0.7):
        """Initialize memory index with configurable parameters."""
        self.embedding_dim = embedding_dim
        self.rebuild_threshold = rebuild_threshold
        self.time_decay = time_decay
        self.min_similarity = min_similarity
        self.memories = []
        self.index = None

    async def add_memory(self, memory_id, embedding, timestamp, significance=1.0, content=None):
        """Add a memory with an embedding, timestamp, significance score, and content."""
        # Ensure embedding is normalized
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.clone().detach()
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Normalize embedding
        norm = torch.norm(embedding, p=2)
        if norm > 0:
            embedding = embedding / norm

        memory = {
            'id': memory_id,
            'embedding': embedding,
            'timestamp': timestamp,
            'significance': significance,
            'content': content or ""  # Ensure content is never None
        }
        self.memories.append(memory)

        if len(self.memories) % self.rebuild_threshold == 0:
            self.build_index()
        
        return memory

    def build_index(self):
        """Build the search index from stored memories."""
        if not self.memories:
            return
        
        # Stack and normalize embeddings
        embeddings = torch.stack([m['embedding'] for m in self.memories])
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        self.index = embeddings / (norms + 1e-8)  # Add epsilon to prevent division by zero
        print(f"🔹 Built index with {len(self.memories)} memories")

    def search(self, query_embedding, k=5):
        """Search for top-k similar memories with time decay and significance weighting."""
        if self.index is None:
            self.build_index()
            
        if not self.memories:
            return []

        # Normalize query embedding
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.clone().detach()
        else:
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        query_norm = torch.norm(query_embedding, p=2)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            query_normalized = query_embedding

        # Compute cosine similarities
        similarities = torch.matmul(self.index, query_normalized)

        # Apply significance weighting
        significance_scores = torch.tensor([m['significance'] for m in self.memories])
        weighted_similarities = similarities * significance_scores

        # Apply time decay (newer memories get a boost)
        timestamps = torch.tensor([m['timestamp'] for m in self.memories], dtype=torch.float32)
        max_timestamp = torch.max(timestamps)
        time_decay_weights = torch.exp(-self.time_decay * (max_timestamp - timestamps))
        final_scores = weighted_similarities * time_decay_weights

        # Get top k results
        k = min(k, len(self.memories))
        values, indices = torch.topk(final_scores, k)

        results = []
        for val, idx in zip(values, indices):
            results.append({
                'memory': self.memories[idx],
                'similarity': similarities[idx].item()  # Use raw similarity for threshold checks
            })

        return results
```

# memory_system.py

```py
import torch
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'storage_path': Path.cwd() / 'memory/stored',  # Use absolute path
            'embedding_dim': 384,
            'rebuild_threshold': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            **(config or {})
        }
        
        self.memories = []
        self.storage_path = Path(self.config['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Log the actual storage path being used
        logger.info(f"Memory storage path: {self.storage_path.absolute()}")
        
        # Load existing memories
        self._load_memories()
        logger.info(f"Initialized MemorySystem with {len(self.memories)} memories")

    def _load_memories(self):
        """Load all memories from disk."""
        self.memories = []
        try:
            logger.info(f"Loading memories from {self.storage_path}")
            
            # Check if directory exists
            if not self.storage_path.exists():
                logger.warning(f"Memory storage path does not exist: {self.storage_path}")
                self.storage_path.mkdir(parents=True, exist_ok=True)
                return
            
            # Count memory files
            memory_files = list(self.storage_path.glob('*.json'))
            logger.info(f"Found {len(memory_files)} memory files")
            
            # Load each file
            for file_path in memory_files:
                try:
                    with open(file_path, 'r') as f:
                        memory = json.load(f)
                        if isinstance(memory, dict) and 'timestamp' in memory:
                            # Convert embedding from list to tensor if needed
                            if 'embedding' in memory and isinstance(memory['embedding'], list):
                                memory['embedding'] = memory['embedding']  # Keep as list for now
                            
                            self.memories.append(memory)
                            logger.debug(f"Loaded memory {memory.get('id', 'unknown')} from {file_path}")
                        else:
                            logger.warning(f"Invalid memory format in {file_path}")
                except Exception as e:
                    logger.error(f"Error loading memory file {file_path}: {str(e)}")
            
            # Sort by timestamp if memories exist
            if self.memories:
                self.memories.sort(key=lambda x: x.get('timestamp', 0))
                logger.info(f"Successfully loaded {len(self.memories)} memories")
            else:
                logger.info("No valid memories found")
                
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}", exc_info=True)
            self.memories = []

    async def add_memory(self, text: str, embedding: torch.Tensor, 
                        significance: float = None) -> Dict[str, Any]:
        """Add memory with persistence."""
        # Normalize embedding
        embedding = self._normalize_embedding(embedding)
        
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        memory = {
            'id': memory_id,
            'text': text,
            'embedding': embedding.tolist(),
            'timestamp': timestamp,
            'significance': significance
        }
        
        # Add to memory list
        self.memories.append(memory)
        
        # Save to disk
        self._save_memory(memory)
        
        logger.info(f"Stored memory {memory_id} with significance {significance}")
        return memory

    async def search_memories(self, query_embedding: torch.Tensor, 
                            limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        if not self.memories:
            return []
            
        # Normalize query
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            memory_embedding = torch.tensor(memory['embedding'], 
                                         device=self.config['device'])
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                memory_embedding.unsqueeze(0)
            )
            similarities.append({
                'memory': memory,
                'similarity': similarity.item()
            })
        
        # Sort by similarity and significance
        sorted_memories = sorted(
            similarities,
            key=lambda x: (x['similarity'] * 0.7 + 
                          (x['memory']['significance'] or 0) * 0.3),
            reverse=True
        )
        
        return sorted_memories[:limit]

    def _normalize_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Normalize embedding vector."""
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, device=self.config['device'])
        embedding = embedding.to(self.config['device'])
        norm = torch.norm(embedding, p=2)
        return embedding / norm if norm > 0 else embedding

    def _save_memory(self, memory: Dict[str, Any]):
        """Save memory to disk."""
        try:
            memory_id = memory.get('id')
            if not memory_id:
                logger.warning("Cannot save memory without an ID")
                return
                
            file_path = self.storage_path / f"{memory_id}.json"
            logger.debug(f"Saving memory {memory_id} to {file_path}")
            
            # Ensure the directory exists
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Make a copy to avoid modifying the original
            memory_copy = memory.copy()
            
            # Convert any tensor to list for JSON serialization
            if 'embedding' in memory_copy and isinstance(memory_copy['embedding'], torch.Tensor):
                memory_copy['embedding'] = memory_copy['embedding'].tolist()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(memory_copy, f)
                
            logger.info(f"Successfully saved memory {memory_id} to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving memory {memory.get('id', 'unknown')}: {str(e)}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            latest_timestamp = max([m.get('timestamp', 0) for m in self.memories]) if self.memories else 0
        except Exception as e:
            logger.error(f"Error calculating latest timestamp: {str(e)}")
            latest_timestamp = 0

        return {
            'memory_count': len(self.memories),
            'device': self.config['device'],
            'storage_path': str(self.storage_path),
            'latest_timestamp': latest_timestamp
        }
```

# memory_types.py

```py
"""
LUCID RECALL PROJECT
Agent: LucidAurora 1.1
Date: 2/13/25
Time: 4:42 PM EST

Memory Types: Definitions for memory system types and structures
"""

from enum import Enum
from dataclasses import dataclass
import torch
from typing import Optional
import time

class MemoryTypes(Enum):
    """Types of memories that can be stored"""
    EPISODIC = "episodic"      # Event/experience memories
    SEMANTIC = "semantic"       # Factual/conceptual memories
    PROCEDURAL = "procedural"   # Skill/procedure memories
    WORKING = "working"         # Temporary processing memories
    
@dataclass
class MemoryEntry:
    """Container for a single memory entry"""
    embedding: torch.Tensor
    memory_type: MemoryTypes
    significance: float = 0.0
    timestamp: float = time.time()
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate memory entry on creation"""
        if not isinstance(self.embedding, torch.Tensor):
            raise ValueError("Embedding must be a torch.Tensor")
            
        if not isinstance(self.memory_type, MemoryTypes):
            raise ValueError("Invalid memory type")
            
        if self.significance < 0.0 or self.significance > 1.0:
            raise ValueError("Significance must be between 0 and 1")
```

# significance_calculator.py

```py
"""
LUCID RECALL PROJECT
Unified Significance Calculator

Agent: Lucidia 1.1
Date: 05/03/25
Time: 4:43 PM EST
A standardized significance calculator for consistent memory importance
assessment across all memory system components.
"""

import time
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class SignificanceMode(Enum):
    """Operating modes for significance calculation."""
    STANDARD = "standard"       # Balanced approach for general use
    PRECISE = "precise"         # More elaborate calculation for higher quality
    EFFICIENT = "efficient"     # Simplified calculation for speed
    EMOTIONAL = "emotional"     # Prioritizes emotional content
    INFORMATIONAL = "informational"  # Prioritizes information density
    PERSONAL = "personal"       # Prioritizes personal relevance
    CUSTOM = "custom"           # Uses custom weights

class SignificanceComponent(Enum):
    """Components that contribute to significance calculation."""
    SURPRISE = "surprise"             # Novelty of information
    DIVERSITY = "diversity"           # Uniqueness compared to existing memories
    EMOTION = "emotion"               # Emotional intensity
    RECENCY = "recency"               # Temporal relevance
    IMPORTANCE = "importance"         # Explicit importance markers
    PERSONAL = "personal"             # Personal information relevance
    COHERENCE = "coherence"           # Logical consistency
    INFORMATION = "information"       # Information density
    RELEVANCE = "relevance"           # Contextual relevance
    USER_ATTENTION = "user_attention" # User engagement signals

class UnifiedSignificanceCalculator:
    """
    Unified significance calculator for consistent assessment of memory importance.
    
    This class provides a standardized approach to calculating memory significance
    that can be used across all memory system components. It supports multiple
    modes and can be customized with different weights for different factors.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize significance calculator.
        
        Args:
            config: Configuration options including:
                - mode: SignificanceMode for calculation approach
                - component_weights: Custom weights for different components
                - time_decay_rate: Rate at which significance decays over time
                - surprise_threshold: Threshold for considering something surprising
                - min_significance: Minimum significance value
                - max_significance: Maximum significance value
                - adaptive_thresholds: Whether to adjust thresholds based on data
                - history_window: Number of samples to keep for adaptive thresholds
        """
        self.config = {
            'mode': SignificanceMode.STANDARD,
            'component_weights': {},
            'time_decay_rate': 0.1,
            'surprise_threshold': 0.7,
            'min_significance': 0.0,
            'max_significance': 1.0,
            'adaptive_thresholds': True,
            'history_window': 1000,
            'personal_information_keywords': [
                'name', 'address', 'phone', 'email', 'birthday', 'age', 'family',
                'friend', 'password', 'account', 'credit', 'social security',
                'ssn', 'identification', 'id card', 'passport', 'license'
            ],
            'emotional_keywords': [
                'happy', 'sad', 'angry', 'excited', 'love', 'hate', 'scared',
                'anxious', 'proud', 'disappointed', 'hope', 'fear', 'joy',
                'grief', 'frustration', 'satisfaction', 'worry', 'relief'
            ],
            'informational_prefixes': [
                'fact:', 'important:', 'remember:', 'note:', 'key point:',
                'critical:', 'essential:', 'reminder:', 'don\'t forget:'
            ],
            **(config or {})
        }
        
        # Initialize component weights based on mode
        self._init_component_weights()
        
        # Set mode - convert from string if needed
        if isinstance(self.config['mode'], str):
            try:
                self.config['mode'] = SignificanceMode(self.config['mode'].lower())
            except ValueError:
                logger.warning(f"Invalid mode: {self.config['mode']}, using STANDARD")
                self.config['mode'] = SignificanceMode.STANDARD
                
        # History for adaptive thresholds
        self.history = {
            'calculated_significance': [],
            'component_values': {comp: [] for comp in SignificanceComponent},
            'timestamps': []
        }
        
        # Tracking
        self.total_calculations = 0
        self.start_time = time.time()
        self.last_calculation_time = 0
        
        logger.info(f"Initialized UnifiedSignificanceCalculator with mode: {self.config['mode'].value}")
        
    def _init_component_weights(self) -> None:
        """Initialize component weights based on selected mode."""
        # Default weights for STANDARD mode
        standard_weights = {
            SignificanceComponent.SURPRISE: 0.20,
            SignificanceComponent.DIVERSITY: 0.15,
            SignificanceComponent.EMOTION: 0.15,
            SignificanceComponent.RECENCY: 0.10,
            SignificanceComponent.IMPORTANCE: 0.15,
            SignificanceComponent.PERSONAL: 0.15,
            SignificanceComponent.COHERENCE: 0.05,
            SignificanceComponent.INFORMATION: 0.05,
            SignificanceComponent.RELEVANCE: 0.05,
            SignificanceComponent.USER_ATTENTION: 0.00  # Not used by default
        }
        
        # Mode-specific weights
        mode_weights = {
            SignificanceMode.PRECISE: {
                # More thorough analysis with all components
                SignificanceComponent.SURPRISE: 0.15,
                SignificanceComponent.DIVERSITY: 0.15,
                SignificanceComponent.EMOTION: 0.15,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.15,
                SignificanceComponent.PERSONAL: 0.15,
                SignificanceComponent.COHERENCE: 0.10,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.EFFICIENT: {
                # Focus on most important factors for speed
                SignificanceComponent.SURPRISE: 0.25,
                SignificanceComponent.DIVERSITY: 0.20,
                SignificanceComponent.EMOTION: 0.00,
                SignificanceComponent.RECENCY: 0.15,
                SignificanceComponent.IMPORTANCE: 0.20,
                SignificanceComponent.PERSONAL: 0.20,
                SignificanceComponent.COHERENCE: 0.00,
                SignificanceComponent.INFORMATION: 0.00,
                SignificanceComponent.RELEVANCE: 0.00,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.EMOTIONAL: {
                # Prioritize emotional content
                SignificanceComponent.SURPRISE: 0.15,
                SignificanceComponent.DIVERSITY: 0.10,
                SignificanceComponent.EMOTION: 0.35,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.10,
                SignificanceComponent.PERSONAL: 0.15,
                SignificanceComponent.COHERENCE: 0.00,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.INFORMATIONAL: {
                # Prioritize information density
                SignificanceComponent.SURPRISE: 0.20,
                SignificanceComponent.DIVERSITY: 0.15,
                SignificanceComponent.EMOTION: 0.05,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.20,
                SignificanceComponent.PERSONAL: 0.05,
                SignificanceComponent.COHERENCE: 0.10,
                SignificanceComponent.INFORMATION: 0.15,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            },
            SignificanceMode.PERSONAL: {
                # Prioritize personal information
                SignificanceComponent.SURPRISE: 0.10,
                SignificanceComponent.DIVERSITY: 0.10,
                SignificanceComponent.EMOTION: 0.10,
                SignificanceComponent.RECENCY: 0.05,
                SignificanceComponent.IMPORTANCE: 0.10,
                SignificanceComponent.PERSONAL: 0.40,
                SignificanceComponent.COHERENCE: 0.05,
                SignificanceComponent.INFORMATION: 0.05,
                SignificanceComponent.RELEVANCE: 0.05,
                SignificanceComponent.USER_ATTENTION: 0.00
            }
        }
        
        # Set weights based on mode
        mode = self.config['mode']
        if isinstance(mode, str):
            try:
                mode = SignificanceMode(mode.lower())
            except ValueError:
                mode = SignificanceMode.STANDARD
        
        weights = mode_weights.get(mode, standard_weights)
        
        # Override with custom weights if provided
        if mode == SignificanceMode.CUSTOM and self.config['component_weights']:
            weights.update(self.config['component_weights'])
        
        # Store final weights
        self.component_weights = weights
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.component_weights = {comp: weight / total_weight for comp, weight in weights.items()}
    
    async def calculate(self, 
                      embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                      text: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate significance score for a memory.
        
        Args:
            embedding: Vector representation of memory content
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Significance score between 0.0 and 1.0
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Calculate component values
            component_values = {}
            
            # 1. Calculate surprise component if embedding provided
            if embedding is not None:
                component_values[SignificanceComponent.SURPRISE] = await self._calculate_surprise(embedding, context)
                component_values[SignificanceComponent.DIVERSITY] = await self._calculate_diversity(embedding, context)
            else:
                component_values[SignificanceComponent.SURPRISE] = 0.0
                component_values[SignificanceComponent.DIVERSITY] = 0.0
                
            # 2. Calculate text-based components if text provided
            if text:
                component_values[SignificanceComponent.EMOTION] = self._calculate_emotion(text, context)
                component_values[SignificanceComponent.IMPORTANCE] = self._calculate_importance(text, context)
                component_values[SignificanceComponent.PERSONAL] = self._calculate_personal(text, context)
                component_values[SignificanceComponent.INFORMATION] = self._calculate_information(text, context)
                component_values[SignificanceComponent.COHERENCE] = self._calculate_coherence(text, context)
            else:
                component_values[SignificanceComponent.EMOTION] = 0.0
                component_values[SignificanceComponent.IMPORTANCE] = 0.0
                component_values[SignificanceComponent.PERSONAL] = 0.0
                component_values[SignificanceComponent.INFORMATION] = 0.0
                component_values[SignificanceComponent.COHERENCE] = 0.0
                
            # 3. Calculate context-based components
            component_values[SignificanceComponent.RECENCY] = self._calculate_recency(context)
            component_values[SignificanceComponent.RELEVANCE] = self._calculate_relevance(context)
            component_values[SignificanceComponent.USER_ATTENTION] = self._calculate_user_attention(context)
            
            # 4. Apply weights to components
            weighted_sum = 0.0
            for component, value in component_values.items():
                weight = self.component_weights.get(component, 0.0)
                weighted_sum += weight * value
                
            # 5. Apply time decay
            time_decay = self._calculate_time_decay(context)
            significance = weighted_sum * time_decay
            
            # 6. Apply sigmoid function to ensure value is between 0-1
            significance = 1.0 / (1.0 + np.exp(-5.0 * (significance - 0.5)))
            
            # 7. Clamp to configured min/max range
            significance = max(self.config['min_significance'], 
                             min(self.config['max_significance'], significance))
            
            # 8. Update history for adaptive thresholds
            if self.config['adaptive_thresholds']:
                self._update_history(significance, component_values)
            
            # Update tracking
            self.total_calculations += 1
            self.last_calculation_time = time.time()
            
            logger.debug(f"Calculated significance: {significance:.4f} in {(time.time() - start_time)*1000:.2f}ms")
            
            return float(significance)
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            # Return default value on error
            return 0.5
    
    async def _calculate_surprise(self, 
                                embedding: Union[np.ndarray, torch.Tensor, List[float]], 
                                context: Dict[str, Any]) -> float:
        """
        Calculate surprise component.
        
        Args:
            embedding: Vector representation of memory content
            context: Additional contextual information
            
        Returns:
            Surprise score between 0.0 and 1.0
        """
        # If history available in context, use it
        if "embedding_history" in context and context["embedding_history"]:
            history = context["embedding_history"]
            
            # Process embedding
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Ensure embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            # Calculate similarities to history
            similarities = []
            for hist_emb in history:
                if isinstance(hist_emb, torch.Tensor):
                    hist_emb = hist_emb.detach().cpu().numpy()
                elif isinstance(hist_emb, list):
                    hist_emb = np.array(hist_emb, dtype=np.float32)
                    
                # Ensure history embedding is normalized
                hist_norm = np.linalg.norm(hist_emb)
                if hist_norm > 0:
                    hist_emb = hist_emb / hist_norm
                    
                # Calculate cosine similarity
                similarity = np.dot(embedding, hist_emb)
                similarities.append(similarity)
                
            if similarities:
                # Higher surprise = lower similarity
                avg_similarity = np.mean(similarities)
                surprise = 1.0 - avg_similarity
                
                # Adjust based on threshold
                surprise = max(0.0, (surprise - self.config['surprise_threshold'])) / (1.0 - self.config['surprise_threshold'])
                return min(1.0, surprise)
                
        # Default surprise if no history or not enough context
        return 0.5
    
    async def _calculate_diversity(self, 
                                 embedding: Union[np.ndarray, torch.Tensor, List[float]], 
                                 context: Dict[str, Any]) -> float:
        """
        Calculate diversity component.
        
        Args:
            embedding: Vector representation of memory content
            context: Additional contextual information
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        # Similar to surprise but focuses on maximum similarity rather than average
        if "embedding_history" in context and context["embedding_history"]:
            history = context["embedding_history"]
            
            # Process embedding
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Ensure embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            # Calculate similarities to history
            similarities = []
            for hist_emb in history:
                if isinstance(hist_emb, torch.Tensor):
                    hist_emb = hist_emb.detach().cpu().numpy()
                elif isinstance(hist_emb, list):
                    hist_emb = np.array(hist_emb, dtype=np.float32)
                    
                # Ensure history embedding is normalized
                hist_norm = np.linalg.norm(hist_emb)
                if hist_norm > 0:
                    hist_emb = hist_emb / hist_norm
                    
                # Calculate cosine similarity
                similarity = np.dot(embedding, hist_emb)
                similarities.append(similarity)
                
            if similarities:
                # Higher diversity = lower maximum similarity
                max_similarity = max(similarities)
                diversity = 1.0 - max_similarity
                
                return min(1.0, diversity)
                
        # Default diversity if no history or not enough context
        return 0.5
    
    def _calculate_emotion(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate emotion component based on text content.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Emotion score between 0.0 and 1.0
        """
        # Check for emotional keywords
        emotional_keywords = self.config['emotional_keywords']
        
        # Count emotional keywords in text
        text_lower = text.lower()
        emotion_count = sum(1 for keyword in emotional_keywords if keyword in text_lower)
        
        # Normalize count
        normalized_count = min(1.0, emotion_count / 5.0)  # Cap at 5 emotional keywords
        
        # Check for explicit emotion markers in context
        emotion_markers = context.get('emotion_markers', {})
        explicit_emotion = emotion_markers.get('intensity', 0.0)
        
        # Combine text-based and explicit emotion
        emotion_score = max(normalized_count, explicit_emotion)
        
        return emotion_score
    
    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate importance component based on importance markers.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        # Check for informational prefixes like "important:" or "note:"
        text_lower = text.lower()
        prefix_matches = any(text_lower.startswith(prefix) for prefix in self.config['informational_prefixes'])
        
        # Check for explicit importance in context
        explicit_importance = context.get('importance', 0.0)
        
        # Check for imperative verbs that suggest importance
        imperative_markers = ['must', 'should', 'need to', 'have to', 'remember', 'don\'t forget']
        imperative_match = any(marker in text_lower for marker in imperative_markers)
        
        # Calculate importance score
        importance_score = max(
            float(prefix_matches) * 0.8,
            explicit_importance,
            float(imperative_match) * 0.6
        )
        
        return importance_score
    
    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate personal component based on personal information content.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Personal score between 0.0 and 1.0
        """
        # Check for personal information keywords
        personal_keywords = self.config['personal_information_keywords']
        
        # Count personal keywords in text
        text_lower = text.lower()
        personal_count = sum(1 for keyword in personal_keywords if keyword in text_lower)
        
        # Normalize count
        normalized_count = min(1.0, personal_count / 3.0)  # Cap at 3 personal keywords
        
        # Check for first-person pronouns which indicate personal information
        first_person_pronouns = ['i ', 'me ', 'my ', 'mine ', 'we ', 'us ', 'our ', 'ours ']
        pronoun_count = sum(text_lower.count(pronoun) for pronoun in first_person_pronouns)
        pronoun_score = min(1.0, pronoun_count / 10.0)  # Cap at 10 pronouns
        
        # Check for explicit personal markers in context
        explicit_personal = context.get('personal_relevance', 0.0)
        
        # Combine scores
        personal_score = max(normalized_count, pronoun_score * 0.7, explicit_personal)
        
        return personal_score
    
    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate information density component.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Information score between 0.0 and 1.0
        """
        # Basic estimation of information density based on text length
        # Longer text tends to contain more information, but with diminishing returns
        token_count = len(text.split())
        normalized_length = min(1.0, token_count / 100.0)  # Cap at 100 tokens
        
        # Modify based on structural elements like lists or formatting
        list_items = text.count('\n- ')
        has_lists = list_items > 0
        list_bonus = min(0.3, list_items / 10.0)  # Bonus for structured lists, cap at 0.3
        
        # Check for numerical content (dates, quantities, etc.)
        import re
        numerical_content = len(re.findall(r'\d+', text))
        numerical_score = min(0.3, numerical_content / 10.0)  # Cap at 0.3
        
        # Combine scores
        information_score = normalized_length + (list_bonus if has_lists else 0) + numerical_score
        
        return min(1.0, information_score)
    
    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
        """
        Calculate coherence component based on text structure.
        
        Args:
            text: Text content of memory
            context: Additional contextual information
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        # Simple coherence estimation based on sentence structure
        # More sophisticated NLP would be better but beyond scope
        
        # Check sentence length distribution (extreme variation suggests incoherence)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.5  # Default for empty text
            
        # Calculate sentence length stats
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Large variation in sentence length suggests less coherence
        if len(sentence_lengths) > 1:
            variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
            std_dev = variance ** 0.5
            length_coherence = max(0.0, 1.0 - (std_dev / avg_length))
        else:
            length_coherence = 0.5  # Default for single sentence
            
        # Check for connection words that suggest coherent structure
        connection_words = ['therefore', 'thus', 'because', 'since', 'so', 'as a result', 
                          'consequently', 'furthermore', 'moreover', 'in addition']
        text_lower = text.lower()
        connection_count = sum(text_lower.count(word) for word in connection_words)
        connection_score = min(0.5, connection_count / 5.0)  # Cap at 0.5
        
        # Combine scores
        coherence_score = 0.5 * length_coherence + 0.5 * connection_score
        
        return coherence_score
    
    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        """
        Calculate recency component based on time.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Recency score between 0.0 and 1.0
        """
        # Get timestamp from context or use current time
        timestamp = context.get('timestamp', time.time())
        current_time = time.time()
        
        # Calculate time elapsed
        elapsed_seconds = max(0, current_time - timestamp)
        
        # Convert to days
        elapsed_days = elapsed_seconds / (24 * 3600)
        
        # Apply exponential decay
        decay_rate = self.config['time_decay_rate']
        recency = np.exp(-decay_rate * elapsed_days)
        
        return recency
    
    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        """
        Calculate relevance component based on context.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Use explicit relevance if provided
        if 'relevance' in context:
            return min(1.0, max(0.0, context['relevance']))
            
        # Use similarity to query if provided
        if 'query_similarity' in context:
            return min(1.0, max(0.0, context['query_similarity']))
            
        # Default relevance
        return 0.5
    
    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
        """
        Calculate user attention component based on interaction signals.
        
        Args:
            context: Additional contextual information
            
        Returns:
            User attention score between 0.0 and 1.0
        """
        # Check for user interaction markers
        interaction_markers = context.get('user_interaction', {})
        
        # Different types of attention signals
        explicit_focus = interaction_markers.get('explicit_focus', 0.0)
        repeat_count = interaction_markers.get('repeat_count', 0)
        dwell_time = interaction_markers.get('dwell_time', 0.0)
        
        # Normalize signals
        normalized_repeat = min(1.0, repeat_count / 3.0)  # Cap at 3 repeats
        normalized_dwell = min(1.0, dwell_time / 30.0)    # Cap at 30 seconds
        
        # Combine signals
        attention_score = max(explicit_focus, normalized_repeat, normalized_dwell)
        
        return attention_score
    
    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """
        Calculate time decay factor.
        
        Args:
            context: Additional contextual information
            
        Returns:
            Time decay factor between 0.0 and 1.0
        """
        # Get timestamp from context or use current time
        timestamp = context.get('timestamp', time.time())
        current_time = time.time()
        
        # For very recent memories, don't apply decay
        if current_time - timestamp < 60:  # Less than a minute old
            return 1.0
            
        # For older memories, apply configurable decay
        elapsed_days = (current_time - timestamp) / (24 * 3600)
        
        # Different modes have different decay profiles
        if self.config['mode'] == SignificanceMode.PRECISE:
            # Slower decay for precise mode
            decay_rate = self.config['time_decay_rate'] * 0.5
        elif self.config['mode'] == SignificanceMode.EFFICIENT:
            # Faster decay for efficient mode
            decay_rate = self.config['time_decay_rate'] * 1.5
        else:
            # Standard decay rate
            decay_rate = self.config['time_decay_rate']
            
        # Apply exponential decay
        decay_factor = np.exp(-decay_rate * elapsed_days)
        
        # Ensure minimum decay factor based on memory importance
        min_decay = context.get('min_decay_factor', 0.1)
        decay_factor = max(min_decay, decay_factor)
        
        return decay_factor
    
    def _update_history(self, significance: float, component_values: Dict[SignificanceComponent, float]) -> None:
        """
        Update history for adaptive thresholds.
        
        Args:
            significance: Calculated significance score
            component_values: Individual component values
        """
        # Add to history
        self.history['calculated_significance'].append(significance)
        self.history['timestamps'].append(time.time())
        
        for component, value in component_values.items():
            if component in self.history['component_values']:
                self.history['component_values'][component].append(value)
                
        # Trim history if needed
        if len(self.history['calculated_significance']) > self.config['history_window']:
            self.history['calculated_significance'] = self.history['calculated_significance'][-self.config['history_window']:]
            self.history['timestamps'] = self.history['timestamps'][-self.config['history_window']:]
            
            for component in self.history['component_values']:
                if len(self.history['component_values'][component]) > self.config['history_window']:
                    self.history['component_values'][component] = self.history['component_values'][component][-self.config['history_window']:]
    
    def update_adaptive_thresholds(self) -> None:
        """Update adaptive thresholds based on history."""
        if not self.history['calculated_significance']:
            return
            
        # Calculate significance distribution
        significance_values = np.array(self.history['calculated_significance'])
        
        # Adjust surprise threshold based on historical distribution
        if SignificanceComponent.SURPRISE in self.history['component_values'] and len(self.history['component_values'][SignificanceComponent.SURPRISE]) > 10:
            surprise_values = np.array(self.history['component_values'][SignificanceComponent.SURPRISE])
            
            # Set threshold at 70th percentile
            self.config['surprise_threshold'] = np.percentile(surprise_values, 70)
            
        logger.debug(f"Updated adaptive thresholds: surprise_threshold={self.config['surprise_threshold']:.2f}")
    
    def set_mode(self, mode: Union[str, SignificanceMode]) -> None:
        """
        Set the calculation mode.
        
        Args:
            mode: New mode as string or SignificanceMode enum
        """
        if isinstance(mode, str):
            try:
                mode = SignificanceMode(mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode: {mode}, using STANDARD")
                mode = SignificanceMode.STANDARD
        
        self.config['mode'] = mode
        self._init_component_weights()
        logger.info(f"Significance calculator mode set to: {mode.value}")
    
    def set_component_weights(self, weights: Dict[Union[str, SignificanceComponent], float]) -> None:
        """
        Set custom component weights.
        
        Args:
            weights: Dictionary mapping components to weights
        """
        # Convert string keys to SignificanceComponent enum
        component_weights = {}
        for key, value in weights.items():
            if isinstance(key, str):
                try:
                    component = SignificanceComponent[key.upper()]
                    component_weights[component] = value
                except KeyError:
                    logger.warning(f"Unknown component: {key}, ignoring")
            else:
                component_weights[key] = value
        
        # Set custom weights
        self.config['component_weights'] = component_weights
        
        # Set mode to CUSTOM
        self.config['mode'] = SignificanceMode.CUSTOM
        
        # Reinitialize weights
        self._init_component_weights()
        
        logger.info("Custom component weights set")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get significance calculator statistics."""
        # Calculate average significance and component values
        avg_significance = np.mean(self.history['calculated_significance']) if self.history['calculated_significance'] else 0.0
        
        component_stats = {}
        for component in SignificanceComponent:
            values = self.history['component_values'].get(component, [])
            if values:
                component_stats[component.value] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'weight': self.component_weights.get(component, 0.0)
                }
        
        return {
            'mode': self.config['mode'].value,
            'total_calculations': self.total_calculations,
            'avg_significance': avg_significance,
            'surprise_threshold': self.config['surprise_threshold'],
            'time_decay_rate': self.config['time_decay_rate'],
            'adaptive_thresholds': self.config['adaptive_thresholds'],
            'history_size': len(self.history['calculated_significance']),
            'components': component_stats
        }
```

# standard_websocket_interface.py

```py
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
```

# tensor_server.py

```py
"""
LUCID RECALL PROJECT
Agent: Aurora 1.1
Date: 2/13/25
Time: 1:19 AM EST

Tensor Server: Memory & Embedding Operations with Unified Memory System
"""

import asyncio
import websockets
import json
import logging
import torch
import time
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from server.hpc_sig_flow_manager import HPCSIGFlowManager
from server.memory_system import MemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5001):
        self.host = host
        self.port = port
        self.setup_gpu()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if torch.cuda.is_available():
            self.model.to('cuda')
        logger.info(f"Model loaded on {self.model.device}")
        
        # Initialize HPC manager
        self.hpc_manager = HPCSIGFlowManager({
            'embedding_dim': 384,
            'device': self.device
        })
        
        # Initialize unified memory system
        self.memory_system = MemorySystem({
            'device': self.device,
            'embedding_dim': 384
        })
        logger.info("Initialized unified memory system")

    def setup_gpu(self) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            logger.warning("GPU not available, using CPU")

    async def add_memory(self, text: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Add memory with embedding and return metadata."""
        # Process through HPC
        processed_embedding, significance = await self.hpc_manager.process_embedding(embedding)
        
        # Store in unified memory system
        memory = await self.memory_system.add_memory(
            text=text,
            embedding=processed_embedding,
            significance=significance
        )
        
        logger.info(f"Stored memory {memory['id']} with significance {significance}")
        return {
            'id': memory['id'],
            'significance': significance,
            'timestamp': memory['timestamp']
        }

    async def search_memories(self, query_embedding: torch.Tensor, limit: int = 5) -> List[Dict]:
        """Search for similar memories."""
        # Get processed query embedding
        processed_query, _ = await self.hpc_manager.process_embedding(query_embedding)
        
        # Search using unified memory system
        results = await self.memory_system.search_memories(
            query_embedding=processed_query,
            limit=limit
        )
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        memory_stats = self.memory_system.get_stats()
        stats = {
            'type': 'stats',
            'gpu_memory': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'gpu_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            'device': self.device,
            'hpc_status': self.hpc_manager.get_stats(),
            'memory_count': memory_stats['memory_count'],
            'storage_path': memory_stats['storage_path']
        }
        logger.info(f"Stats requested: {stats}")
        return stats

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections and messages."""
        try:
            logger.info(f"New connection from {websocket.remote_address}")
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")

                    if data['type'] == 'embed':
                        # Generate embedding
                        embeddings = self.model.encode(data['text'])
                        
                        # Store memory
                        metadata = await self.add_memory(
                            data['text'], 
                            torch.tensor(embeddings)
                        )
                        
                        response = {
                            'type': 'embeddings',
                            'embeddings': embeddings.tolist(),
                            **metadata
                        }
                        
                    elif data['type'] == 'search':
                        # Generate query embedding
                        query_embedding = self.model.encode(data['text'])
                        
                        # Search memories
                        results = await self.search_memories(
                            torch.tensor(query_embedding),
                            limit=data.get('limit', 5)
                        )
                        
                        response = {
                            'type': 'search_results',
                            'results': [{
                                'id': r['memory']['id'],
                                'text': r['memory']['text'],
                                'similarity': r['similarity'],
                                'significance': r['memory']['significance']
                            } for r in results]
                        }
                        
                    elif data['type'] == 'stats':
                        response = self.get_stats()
                        
                    else:
                        response = {
                            'type': 'error',
                            'error': f"Unknown message type: {data['type']}"
                        }
                        logger.warning(f"Unknown message type received: {data['type']}")

                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent response: {response['type']}")

                except Exception as e:
                    error_msg = {
                        'type': 'error',
                        'error': str(e)
                    }
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")

    async def start(self):
        """Start the WebSocket server."""
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            logger.info(f"Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()

if __name__ == '__main__':
    server = TensorServer()
    asyncio.run(server.start())
```

# unified_memory_storage.py

```py
"""
LUCID RECALL PROJECT
Unified Memory Storage Interface

A standardized interface for memory storage operations across
client and server implementations.
"""

import os
import time
import json
import logging
import numpy as np
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import asyncio
import torch

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories that can be stored."""
    EPISODIC = "episodic"      # Event/experience memories (conversations, interactions)
    SEMANTIC = "semantic"      # Factual/conceptual memories (knowledge, facts)
    PROCEDURAL = "procedural"  # Skill/procedure memories (how to do things)
    WORKING = "working"        # Temporary processing memories (short-term)
    PERSONAL = "personal"      # Personal information about users
    IMPORTANT = "important"    # High-significance memories that should be preserved
    EMOTIONAL = "emotional"    # Memories with emotional context
    SYSTEM = "system"          # System-level memories (configs, settings)

class MemoryEntry:
    """
    Standard memory entry structure for unified access across systems.
    
    This class provides a standard structure for memory entries with
    consistent access patterns and serialization.
    """
    
    def __init__(self, 
                 content: str,
                 embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                 memory_type: Union[MemoryType, str] = MemoryType.EPISODIC,
                 significance: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None,
                 id: Optional[str] = None,
                 timestamp: Optional[float] = None):
        """
        Initialize a memory entry.
        
        Args:
            content: Primary memory content as text
            embedding: Vector representation of content
            memory_type: Type of memory
            significance: Importance score (0.0-1.0)
            metadata: Additional data about this memory
            id: Unique identifier (generated if not provided)
            timestamp: Creation time (current time if not provided)
        """
        self.content = content
        self._embedding = self._process_embedding(embedding)
        
        # Handle memory_type as string or enum
        if isinstance(memory_type, str):
            try:
                self.memory_type = MemoryType[memory_type.upper()]
            except KeyError:
                for mem_type in MemoryType:
                    if mem_type.value == memory_type.lower():
                        self.memory_type = mem_type
                        break
                else:
                    logger.warning(f"Unknown memory type: {memory_type}, defaulting to EPISODIC")
                    self.memory_type = MemoryType.EPISODIC
        else:
            self.memory_type = memory_type
            
        # Ensure significance is in valid range
        self.significance = max(0.0, min(1.0, significance))
        
        self.metadata = metadata or {}
        self.id = id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
        
        # Additional tracking information
        self.access_count = 0
        self.last_access_time = self.timestamp
        self.creation_source = self.metadata.get("source", "unknown")
        
        # Decay parameters
        self.decay_rate = 0.05  # Base decay rate (5% per day)
        self.boost_factor = 0.0  # Significance boost from repeated access
        
    def _process_embedding(self, embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]]) -> Optional[np.ndarray]:
        """Process and normalize embedding input."""
        if embedding is None:
            return None
            
        # Convert to numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Ensure correct dimensionality
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.flatten()
            
        # Normalize if not already normalized
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-5:
            embedding = embedding / norm
            
        return embedding
    
    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Get memory embedding."""
        return self._embedding
    
    @embedding.setter
    def embedding(self, value: Optional[Union[np.ndarray, torch.Tensor, List[float]]]) -> None:
        """Set memory embedding with automatic processing."""
        self._embedding = self._process_embedding(value)
        
    def record_access(self) -> None:
        """Record memory access, updating tracking information."""
        current_time = time.time()
        self.access_count += 1
        self.last_access_time = current_time
        
        # Update boost factor based on access frequency
        time_since_creation = current_time - self.timestamp
        if time_since_creation > 0:
            # More recent accesses provide stronger boost
            recency_factor = min(1.0, 30 * 86400 / max(1, time_since_creation))
            # More accesses provide stronger boost
            access_factor = min(1.0, self.access_count / 10)
            self.boost_factor = 0.3 * recency_factor * access_factor
    
    def get_effective_significance(self) -> float:
        """
        Get effective significance with decay and boost applied.
        
        The significance naturally decays over time but can be boosted
        by frequent access.
        """
        current_time = time.time()
        days_elapsed = (current_time - self.timestamp) / (24 * 3600)
        
        # Calculate decay (lower for important memories)
        importance_factor = 0.5 + 0.5 * self.significance
        effective_decay_rate = self.decay_rate / importance_factor
        decay_factor = np.exp(-effective_decay_rate * days_elapsed)
        
        # Apply decay and boost
        effective_significance = self.significance * decay_factor + self.boost_factor
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, effective_significance))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self._embedding.tolist() if self._embedding is not None else None,
            "memory_type": self.memory_type.value,
            "significance": self.significance,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "effective_significance": self.get_effective_significance()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary."""
        # Convert embedding back to numpy array if present
        embedding_data = data.get("embedding")
        embedding = np.array(embedding_data, dtype=np.float32) if embedding_data else None
        
        return cls(
            content=data.get("content", ""),
            embedding=embedding,
            memory_type=data.get("memory_type", MemoryType.EPISODIC),
            significance=data.get("significance", 0.5),
            metadata=data.get("metadata", {}),
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time())
        )
    
    def __str__(self) -> str:
        """String representation of memory."""
        return f"Memory({self.id[:8]}, type={self.memory_type.value}, sig={self.significance:.2f}): {self.content[:50]}..."

class UnifiedMemoryStorage:
    """
    Unified memory storage interface for consistent operations.
    
    This class provides a standardized interface for memory storage operations
    that can be used consistently across client and server implementations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize memory storage.
        
        Args:
            config: Configuration options
        """
        self.config = {
            'storage_path': os.path.join('data', 'memory'),
            'max_memories': 10000,
            'auto_prune': True,
            'prune_threshold': 0.9,  # Prune when 90% full
            'min_significance_to_store': 0.1,
            'persistence_enabled': True,
            'backup_frequency': 3600,  # Seconds between backups
            'case_sensitive_search': False,
            **(config or {})
        }
        
        # Initialize storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_types: Dict[MemoryType, List[str]] = {mem_type: [] for mem_type in MemoryType}
        
        # Initialize directories
        if self.config['persistence_enabled']:
            os.makedirs(self.config['storage_path'], exist_ok=True)
            
        # Statistics
        self.stats = {
            'memories_stored': 0,
            'memories_retrieved': 0,
            'memories_pruned': 0,
            'memories_purged': 0,
            'memories_updated': 0,
            'last_prune_time': 0,
            'last_backup_time': 0
        }
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._backup_task = None
        
        # Load memories if persistence enabled
        if self.config['persistence_enabled']:
            asyncio.create_task(self._load_memories())
    
    async def store(self, memory: Union[MemoryEntry, Dict[str, Any]]) -> str:
        """
        Store a memory.
        
        Args:
            memory: Memory entry or dictionary
            
        Returns:
            Memory ID
        """
        async with self._lock:
            # Convert dict to MemoryEntry if needed
            if isinstance(memory, dict):
                memory = MemoryEntry.from_dict(memory)
                
            # Skip if below minimum significance
            if memory.significance < self.config['min_significance_to_store']:
                logger.debug(f"Memory below minimum significance threshold ({memory.significance:.2f}), not storing")
                return ""
                
            # Check if we need to prune
            if self.config['auto_prune'] and len(self.memories) >= self.config['max_memories'] * self.config['prune_threshold']:
                await self._prune_memories()
                
            # Check if we're still full after pruning
            if len(self.memories) >= self.config['max_memories']:
                logger.warning(f"Memory storage full ({len(self.memories)}/{self.config['max_memories']}), cannot store new memory")
                return ""
                
            # Store the memory
            self.memories[memory.id] = memory
            
            # Update type index
            self.memory_types[memory.memory_type].append(memory.id)
            
            # Update stats
            self.stats['memories_stored'] += 1
            
            # Persist if enabled
            if self.config['persistence_enabled']:
                await self._persist_memory(memory)
                
            # Start backup task if needed
            if self.config['persistence_enabled'] and (
                self._backup_task is None or 
                self._backup_task.done() or 
                time.time() - self.stats['last_backup_time'] > self.config['backup_frequency']):
                self._backup_task = asyncio.create_task(self._backup_memories())
                
            return memory.id
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory entry or None if not found
        """
        async with self._lock:
            memory = self.memories.get(memory_id)
            
            if memory:
                # Record access
                memory.record_access()
                self.stats['memories_retrieved'] += 1
                
            return memory
    
    async def search(self, 
                   query_embedding: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
                   query_text: Optional[str] = None,
                   memory_type: Optional[Union[MemoryType, str]] = None,
                   min_significance: float = 0.0,
                   max_count: int = 10,
                   time_range: Optional[Tuple[float, float]] = None) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for memories.
        
        Args:
            query_embedding: Vector query for semantic search
            query_text: Text query for keyword search
            memory_type: Optional type filter
            min_significance: Minimum significance threshold
            max_count: Maximum number of results
            time_range: Optional (start_time, end_time) tuple
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        async with self._lock:
            candidates = []
            
            # Convert memory_type string to enum if needed
            if isinstance(memory_type, str):
                try:
                    memory_type = MemoryType[memory_type.upper()]
                except KeyError:
                    for mem_type in MemoryType:
                        if mem_type.value == memory_type.lower():
                            memory_type = mem_type
                            break
            
            # Get candidate memories
            if memory_type:
                # Get only memories of specified type
                candidate_ids = self.memory_types.get(memory_type, [])
                candidates = [self.memories[mid] for mid in candidate_ids if mid in self.memories]
            else:
                # Get all memories
                candidates = list(self.memories.values())
                
            # Apply significance filter
            candidates = [mem for mem in candidates if mem.get_effective_significance() >= min_significance]
                
            # Apply time range filter if provided
            if time_range:
                start_time, end_time = time_range
                candidates = [mem for mem in candidates if start_time <= mem.timestamp <= end_time]
                
            # Process different search types
            if query_embedding is not None:
                # Semantic search using embeddings
                return await self._semantic_search(candidates, query_embedding, max_count)
            elif query_text:
                # Keyword search using text
                return await self._keyword_search(candidates, query_text, max_count)
            else:
                # No query, sort by significance and recency
                return await self._default_search(candidates, max_count)
    
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID
            updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            memory = self.memories.get(memory_id)
            
            if not memory:
                return False
                
            # Apply updates
            if 'content' in updates:
                memory.content = updates['content']
                
            if 'embedding' in updates:
                memory.embedding = updates['embedding']
                
            if 'memory_type' in updates:
                old_type = memory.memory_type
                
                # Update memory type
                if isinstance(updates['memory_type'], str):
                    try:
                        new_type = MemoryType[updates['memory_type'].upper()]
                    except KeyError:
                        for mem_type in MemoryType:
                            if mem_type.value == updates['memory_type'].lower():
                                new_type = mem_type
                                break
                        else:
                            logger.warning(f"Unknown memory type: {updates['memory_type']}, ignoring update")
                            new_type = old_type
                else:
                    new_type = updates['memory_type']
                    
                # Update type index
                if old_type != new_type:
                    if memory_id in self.memory_types[old_type]:
                        self.memory_types[old_type].remove(memory_id)
                    self.memory_types[new_type].append(memory_id)
                    memory.memory_type = new_type
                
            if 'significance' in updates:
                memory.significance = max(0.0, min(1.0, updates['significance']))
                
            if 'metadata' in updates:
                if isinstance(updates['metadata'], dict):
                    memory.metadata.update(updates['metadata'])
                    
            # Update stats
            self.stats['memories_updated'] += 1
            
            # Persist if enabled
            if self.config['persistence_enabled']:
                await self._persist_memory(memory)
                
            return True
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            memory = self.memories.pop(memory_id, None)
            
            if not memory:
                return False
                
            # Update type index
            if memory_id in self.memory_types[memory.memory_type]:
                self.memory_types[memory.memory_type].remove(memory_id)
                
            # Delete from disk if persistence enabled
            if self.config['persistence_enabled']:
                memory_path = os.path.join(self.config['storage_path'], f"{memory_id}.json")
                if os.path.exists(memory_path):
                    try:
                        os.remove(memory_path)
                    except Exception as e:
                        logger.error(f"Error deleting memory file: {e}")
                
            # Update stats
            self.stats['memories_purged'] += 1
            
            return True
    
    async def _semantic_search(self, 
                             candidates: List[MemoryEntry],
                             query_embedding: Union[np.ndarray, torch.Tensor, List[float]],
                             max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            candidates: List of candidate memories
            query_embedding: Query embedding
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        # Convert query embedding to numpy array
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        elif isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
        # Ensure correct dimensionality
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_embedding = query_embedding.flatten()
            
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate similarities and effective significance for each candidate
        results = []
        for memory in candidates:
            # Skip memories without embeddings
            if memory.embedding is None:
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, memory.embedding)
            
            # Combine similarity with significance
            effective_significance = memory.get_effective_significance()
            combined_score = 0.7 * similarity + 0.3 * effective_significance
            
            results.append((memory, combined_score))
            
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def _keyword_search(self, 
                            candidates: List[MemoryEntry],
                            query_text: str,
                            max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Perform keyword search using text.
        
        Args:
            candidates: List of candidate memories
            query_text: Query text
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        # Convert query to lowercase if case-insensitive search
        if not self.config['case_sensitive_search']:
            query_text = query_text.lower()
            
        # Extract keywords from query
        keywords = query_text.split()
        
        # Calculate match scores for each candidate
        results = []
        for memory in candidates:
            content = memory.content
            
            # Convert content to lowercase if case-insensitive search
            if not self.config['case_sensitive_search']:
                content = content.lower()
                
            # Calculate number of matching keywords
            matching_keywords = sum(1 for keyword in keywords if keyword in content)
            match_ratio = matching_keywords / len(keywords) if keywords else 0
            
            # Calculate relevance score
            effective_significance = memory.get_effective_significance()
            combined_score = 0.7 * match_ratio + 0.3 * effective_significance
            
            # Only include if at least one keyword matches
            if match_ratio > 0:
                results.append((memory, combined_score))
                
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def _default_search(self, 
                            candidates: List[MemoryEntry],
                            max_count: int) -> List[Tuple[MemoryEntry, float]]:
        """
        Default search when no query is provided.
        
        Args:
            candidates: List of candidate memories
            max_count: Maximum number of results
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        results = []
        current_time = time.time()
        
        for memory in candidates:
            # Calculate recency (higher is more recent)
            recency = 1.0 / (1.0 + (current_time - memory.timestamp) / (24 * 3600))  # Normalize to days
            
            # Calculate relevance score
            effective_significance = memory.get_effective_significance()
            combined_score = 0.5 * effective_significance + 0.5 * recency
            
            results.append((memory, combined_score))
            
        # Sort by combined score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    async def _prune_memories(self) -> None:
        """Prune least significant memories when storage is full."""
        logger.info("Pruning memories...")
        
        # Get memories sorted by effective significance
        memories_with_significance = [(mid, mem.get_effective_significance()) for mid, mem in self.memories.items()]
        memories_with_significance.sort(key=lambda x: x[1])
        
        # Calculate number to remove (20% of total)
        prune_count = max(1, int(0.2 * len(self.memories)))
        prune_count = min(prune_count, len(memories_with_significance))
        
        # Remove memories
        for i in range(prune_count):
            memory_id, _ = memories_with_significance[i]
            await self.delete(memory_id)
            
        # Update stats
        self.stats['memories_pruned'] += prune_count
        self.stats['last_prune_time'] = time.time()
        
        logger.info(f"Pruned {prune_count} memories")
    
    async def _persist_memory(self, memory: MemoryEntry) -> None:
        """Persist a memory to disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            memory_path = os.path.join(self.config['storage_path'], f"{memory.id}.json")
            
            # Convert to dictionary
            memory_dict = memory.to_dict()
            
            # Write to file
            with open(memory_path, 'w') as f:
                json.dump(memory_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
    
    async def _load_memories(self) -> None:
        """Load memories from disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            # Get all memory files
            memory_files = [f for f in os.listdir(self.config['storage_path']) if f.endswith('.json')]
            
            # Load each memory
            for filename in memory_files:
                try:
                    memory_path = os.path.join(self.config['storage_path'], filename)
                    
                    with open(memory_path, 'r') as f:
                        memory_dict = json.load(f)
                        
                    # Create memory
                    memory = MemoryEntry.from_dict(memory_dict)
                    
                    # Store in memory (bypassing store method to avoid recursion)
                    self.memories[memory.id] = memory
                    
                    # Update type index
                    self.memory_types[memory.memory_type].append(memory.id)
                    
                except Exception as e:
                    logger.error(f"Error loading memory {filename}: {e}")
                    
            logger.info(f"Loaded {len(self.memories)} memories from disk")
            
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
    
    async def _backup_memories(self) -> None:
        """Backup all memories to disk."""
        if not self.config['persistence_enabled']:
            return
            
        try:
            async with self._lock:
                # Create backup directory
                backup_dir = os.path.join(self.config['storage_path'], 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                
                # Create backup file
                backup_time = time.strftime('%Y%m%d-%H%M%S')
                backup_path = os.path.join(backup_dir, f"memory_backup_{backup_time}.json")
                
                # Convert memories to dictionaries
                memories_dict = {mid: mem.to_dict() for mid, mem in self.memories.items()}
                
                # Write to file
                with open(backup_path, 'w') as f:
                    json.dump(memories_dict, f, indent=2)
                    
                # Update stats
                self.stats['last_backup_time'] = time.time()
                
                logger.info(f"Backed up {len(self.memories)} memories to {backup_path}")
                
                # Clean up old backups (keep last 10)
                backup_files = sorted([f for f in os.listdir(backup_dir) if f.startswith('memory_backup_')])
                if len(backup_files) > 10:
                    for old_backup in backup_files[:-10]:
                        try:
                            os.remove(os.path.join(backup_dir, old_backup))
                        except Exception as e:
                            logger.error(f"Error removing old backup {old_backup}: {e}")
                
        except Exception as e:
            logger.error(f"Error backing up memories: {e}")
    
    async def clear(self) -> None:
        """Clear all memories."""
        async with self._lock:
            self.memories.clear()
            self.memory_types = {mem_type: [] for mem_type in MemoryType}
            
            # Update stats
            self.stats['memories_purged'] += len(self.memories)
            
            logger.info("Memory storage cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        type_counts = {mem_type.value: len(ids) for mem_type, ids in self.memory_types.items()}
        
        return {
            'total_memories': len(self.memories),
            'memory_types': type_counts,
            'memories_stored': self.stats['memories_stored'],
            'memories_retrieved': self.stats['memories_retrieved'],
            'memories_pruned': self.stats['memories_pruned'],
            'memories_purged': self.stats['memories_purged'],
            'memories_updated': self.stats['memories_updated'],
            'storage_utilization': len(self.memories) / self.config['max_memories'],
            'persistence_enabled': self.config['persistence_enabled'],
            'last_prune_time': self.stats['last_prune_time'],
            'last_backup_time': self.stats['last_backup_time']
        }

import torch
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import from memory system
from server.memory_system import MemorySystem
from server.memory_index import MemoryIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMemoryStorage:
    """
    Unified memory storage system that combines MemorySystem and MemoryIndex.
    
    This class provides a unified interface for storing and retrieving memories,
    with both persistent storage and fast search capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the unified memory storage system.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - storage_path: Path to store memories
                - embedding_dim: Dimension of embeddings
                - rebuild_threshold: Number of memories before rebuilding index
                - time_decay: Rate at which memory relevance decays over time
                - min_similarity: Minimum similarity threshold for search results
        """
        self.config = config or {}
        
        # Initialize memory system for persistence
        self.memory_system = MemorySystem(config)
        
        # Initialize memory index for fast search
        self.memory_index = MemoryIndex(
            embedding_dim=self.config.get('embedding_dim', 384),
            rebuild_threshold=self.config.get('rebuild_threshold', 100),
            time_decay=self.config.get('time_decay', 0.01),
            min_similarity=self.config.get('min_similarity', 0.7)
        )
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Load memories from system into index
        self._initialize_index()
        
        logger.info(f"Initialized UnifiedMemoryStorage with {len(self.memory_system.memories)} memories")
    
    def _initialize_index(self):
        """
        Initialize the memory index with existing memories from the memory system.
        """
        try:
            for memory in self.memory_system.memories:
                # Skip if missing required fields
                if not all(k in memory for k in ['id', 'embedding', 'timestamp']):
                    continue
                    
                # Add to index
                asyncio.create_task(self.memory_index.add_memory(
                    memory_id=memory['id'],
                    embedding=memory['embedding'],
                    timestamp=memory['timestamp'],
                    significance=memory.get('significance', 0.5),
                    content=memory.get('text', "")
                ))
            
            # Build the index
            self.memory_index.build_index()
            logger.info(f"Initialized memory index with {len(self.memory_system.memories)} memories")
        except Exception as e:
            logger.error(f"Error initializing memory index: {e}")
    
    async def store_memory(self, text: str, embedding: Union[torch.Tensor, List[float]], 
                          significance: float = None) -> Dict[str, Any]:
        """
        Store a memory in both the memory system and index.
        
        Args:
            text: The memory content
            embedding: The embedding vector (tensor or list)
            significance: Optional significance score (0.0-1.0)
            
        Returns:
            The stored memory object
        """
        try:
            # Ensure embedding is a tensor
            if isinstance(embedding, list):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            # Use default significance if not provided
            if significance is None:
                significance = 0.5
            
            async with self._lock:
                # Store in memory system (persistence)
                memory = await self.memory_system.add_memory(
                    text=text,
                    embedding=embedding,
                    significance=significance
                )
                
                # Store in memory index (search)
                await self.memory_index.add_memory(
                    memory_id=memory['id'],
                    embedding=memory['embedding'],
                    timestamp=memory['timestamp'],
                    significance=memory.get('significance', 0.5),
                    content=text
                )
                
                logger.info(f"Stored memory {memory['id']} with significance {significance}")
                return memory
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            # Return minimal memory object on error
            return {
                'id': str(uuid.uuid4()),
                'text': text,
                'timestamp': time.time(),
                'significance': significance or 0.5,
                'error': str(e)
            }
    
    async def search_memories(self, query_embedding: Union[torch.Tensor, List[float]], 
                            limit: int = 5, min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar memories using the memory index.
        
        Args:
            query_embedding: The query embedding vector (tensor or list)
            limit: Maximum number of results to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        try:
            # Ensure embedding is a tensor
            if isinstance(query_embedding, list):
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            
            # Search in memory index
            results = self.memory_index.search(query_embedding, k=limit * 2)
            
            # Filter by significance
            filtered_results = [
                r for r in results 
                if r['memory'].get('significance', 0.0) >= min_significance
            ]
            
            # Format results
            formatted_results = []
            for result in filtered_results[:limit]:
                memory = result['memory']
                formatted_results.append({
                    'id': memory.get('id', ''),
                    'text': memory.get('content', ''),
                    'significance': memory.get('significance', 0.0),
                    'timestamp': memory.get('timestamp', 0),
                    'similarity': result.get('similarity', 0.0)
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory storage system.
        
        Returns:
            Dictionary with memory statistics
        """
        system_stats = self.memory_system.get_stats()
        
        # Add index-specific stats
        stats = {
            **system_stats,
            'index_initialized': self.memory_index.index is not None,
            'index_size': len(self.memory_index.memories),
        }
        
        return stats
```

# websocket_server.py

```py
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

```

