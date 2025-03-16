"""
LUCID RECALL PROJECT
Memory Adapter

This adapter integrates the new unified components with the existing system
without requiring Docker rebuilds or modification of existing code.
Updated to support QuickRecal architecture.
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
            'unified_quickrecal': False  # Updated from 'unified_significance'
        }
        
        # Component references
        self.unified_hpc = None
        self.standard_websocket = None
        self.unified_storage = None
        self.unified_quickrecal = None  # Updated from 'unified_significance'
        
        # Original component references
        self.original_hpc = None
        self.original_websocket = None
        self.original_storage = None
        self.original_quickrecal = None  # Updated from 'original_significance'
        
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
        
        # Check for QuickRecal calculator (formerly significance calculator)
        # Try both file names for backward compatibility
        component_paths = [
            os.path.join(integration_path, 'quickrecal_calculator.py'),  # New name
            os.path.join(integration_path, 'significance_calculator.py')  # Old name for compatibility
        ]
        
        for component_path in component_paths:
            if os.path.exists(component_path):
                try:
                    module_name = os.path.basename(component_path).split('.')[0]
                    spec = importlib.util.spec_from_file_location(module_name, component_path)
                    unified_quickrecal_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(unified_quickrecal_module)
                    
                    # Look for the new class name first, then fall back to old name
                    if hasattr(unified_quickrecal_module, 'UnifiedQuickRecalCalculator'):
                        self.unified_quickrecal = unified_quickrecal_module.UnifiedQuickRecalCalculator
                    elif hasattr(unified_quickrecal_module, 'UnifiedSignificanceCalculator'):
                        # Use old class for compatibility
                        self.unified_quickrecal = unified_quickrecal_module.UnifiedSignificanceCalculator
                        logger.info("Using legacy significance calculator as QuickRecal calculator")
                    
                    if self.unified_quickrecal:
                        self.available_components['unified_quickrecal'] = True
                        logger.info(f"QuickRecal calculator available from {os.path.basename(component_path)}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load QuickRecal calculator from {component_path}: {e}")
    
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
    
    async def get_quickrecal_calculator(self, original_class=None, config: Dict[str, Any] = None) -> Any:
        """
        Get appropriate QuickRecal calculator based on configuration.
        
        Args:
            original_class: Original class reference if available
            config: Configuration for the component
            
        Returns:
            QuickRecal calculator instance
        """
        # Store original class reference
        if original_class and not self.original_quickrecal:
            self.original_quickrecal = original_class
        
        # Determine which implementation to use
        if self.available_components['unified_quickrecal'] and self.config['prefer_unified']:
            try:
                # Use unified implementation
                instance = self.unified_quickrecal(config)
                logger.info("Using unified QuickRecal calculator")
                
                # In shadow mode, also create original
                if self.config['adapter_mode'] == 'shadow' and original_class:
                    original_instance = original_class(config)
                    return QuickRecalAdapter(instance, original_instance, self.config)
                
                return QuickRecalAdapter(instance, None, self.config)
            except Exception as e:
                logger.error(f"Error creating unified QuickRecal calculator: {e}")
                if self.config['fallback_on_error'] and original_class:
                    logger.info("Falling back to original QuickRecal calculator")
                    return original_class(config)
                raise
        elif original_class:
            # Use original implementation
            logger.info("Using original QuickRecal calculator")
            return original_class(config)
        else:
            raise ValueError("No QuickRecal calculator implementation available")
    
    # Legacy method for backward compatibility
    async def get_significance_calculator(self, original_class=None, config: Dict[str, Any] = None) -> Any:
        """
        Legacy method - redirects to get_quickrecal_calculator for backward compatibility.
        """
        logger.warning("get_significance_calculator is deprecated. Use get_quickrecal_calculator instead.")
        return await self.get_quickrecal_calculator(original_class, config)


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
                    # Extract QuickRecal score (second element of tuple)
                    unified_score = unified_result[1]
                    original_score = original_result[1]
                    self.logger.info(f"HPC comparison - unified QuickRecal: {unified_score:.4f}, original: {original_score:.4f}")
                
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
        
        # Convert legacy significance to quickrecal_score if needed
        if isinstance(memory, dict) and 'significance' in memory and 'quickrecal_score' not in memory:
            memory['quickrecal_score'] = memory['significance']
            self.logger.debug("Converted legacy significance to quickrecal_score")
        
        try:
            if self.mode == 'hybrid' and self.original:
                # Store in both - start with unified
                unified_result = await self.unified.store(memory)
                
                # Also store in original as backup
                try:
                    # Make a copy for the original system to avoid format conflicts
                    memory_copy = memory.copy() if isinstance(memory, dict) else memory
                    # For original systems that use significance, ensure it's present
                    if isinstance(memory_copy, dict) and 'quickrecal_score' in memory_copy and 'significance' not in memory_copy:
                        memory_copy['significance'] = memory_copy['quickrecal_score']
                    
                    await self.original.store(memory_copy)
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
                
                # Ensure compatibility with original system
                if isinstance(memory, dict) and 'quickrecal_score' in memory and 'significance' not in memory:
                    memory_copy = memory.copy()
                    memory_copy['significance'] = memory_copy['quickrecal_score']
                    return await self.original.store(memory_copy)
                
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
                        
                        # Convert significance to quickrecal_score if needed
                        if isinstance(original_result, dict) and 'significance' in original_result and 'quickrecal_score' not in original_result:
                            original_result['quickrecal_score'] = original_result['significance']
                            
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
        
        # Convert min_significance to min_quickrecal if present
        if 'min_significance' in kwargs and 'min_quickrecal' not in kwargs:
            kwargs['min_quickrecal'] = kwargs.pop('min_significance')
            self.logger.debug("Converted min_significance to min_quickrecal in search parameters")
        
        try:
            if self.mode == 'hybrid' and self.original:
                # Prepare parameters for original system
                original_kwargs = kwargs.copy()
                if 'min_quickrecal' in original_kwargs:
                    original_kwargs['min_significance'] = original_kwargs.pop('min_quickrecal')
                
                # Try unified first
                unified_results = await self.unified.search(**kwargs)
                
                # Also search original to potentially find memories not yet migrated
                try:
                    original_results = await self.original.search(**original_kwargs)
                    
                    # Combine results, preferring unified
                    unified_ids = set(result[0].id for result in unified_results)
                    for result in original_results:
                        if result[0].id not in unified_ids:
                            # Convert significance to quickrecal_score if needed
                            if hasattr(result[0], 'significance') and not hasattr(result[0], 'quickrecal_score'):
                                result[0].quickrecal_score = result[0].significance
                                
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
                
                # Convert parameters for backward compatibility
                if 'min_quickrecal' in kwargs:
                    kwargs['min_significance'] = kwargs.pop('min_quickrecal')
                
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


class QuickRecalAdapter:
    """Adapter for QuickRecal calculator (formerly Significance calculator)."""
    
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
        """Calculate QuickRecal score."""
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
                    self.logger.debug(f"QuickRecal calculation time: {(time.time() - start_time)*1000:.2f}ms")
                
                # Return unified result
                return unified_result
            else:
                # Just use unified implementation
                result = await self.unified.calculate(embedding, text, context)
                
                if self.config['log_performance']:
                    self.logger.debug(f"QuickRecal calculation time: {(time.time() - start_time)*1000:.2f}ms")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in QuickRecal adapter: {e}")
            
            if self.config['fallback_on_error'] and self.original:
                self.logger.info("Falling back to original QuickRecal implementation")
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
            self.logger.warning(f"Large QuickRecal difference: {difference:.4f} (unified={unified_result:.4f}, original={original_result:.4f})")
    
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

# Legacy alias for backward compatibility
SignificanceAdapter = QuickRecalAdapter