"""
Base Module Framework for Lucidia's Knowledge Graph

This module provides the foundational classes for the modular knowledge graph architecture.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple, Set, Union

class KnowledgeGraphModule(ABC):
    """
    Base class for all knowledge graph modules.
    
    This abstract class defines the standard interface and common functionality
    for all modules in the knowledge graph system, ensuring consistent
    initialization, event handling, and lifecycle management.
    """
    
    def __init__(self, event_bus, module_registry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize module with shared system components.
        
        Args:
            event_bus: Event bus for inter-module communication
            module_registry: Registry for module discovery and access
            config: Optional module-specific configuration
        """
        self.event_bus = event_bus
        self.module_registry = module_registry
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Module state
        self.initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the module and subscribe to events.
        
        This method should be called after module creation to complete
        setup and establish event subscriptions.
        
        Returns:
            Success status
        """
        if self.initialized:
            self.logger.warning("Module already initialized")
            return True
            
        try:
            await self._subscribe_to_events()
            await self._setup_module()
            self.initialized = True
            self.logger.info(f"Module {self.__class__.__name__} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing module: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """
        Shutdown the module gracefully.
        
        This method should be called before system shutdown to allow
        modules to clean up resources.
        
        Returns:
            Success status
        """
        if not self.initialized:
            self.logger.warning("Module not initialized, nothing to shut down")
            return True
            
        try:
            await self._cleanup_resources()
            self.initialized = False
            self.logger.info(f"Module {self.__class__.__name__} shut down")
            return True
        except Exception as e:
            self.logger.error(f"Error shutting down module: {e}")
            return False
    
    @abstractmethod
    async def _subscribe_to_events(self) -> None:
        """
        Subscribe to events on the event bus.
        
        This method should be implemented by each module to establish
        event subscriptions relevant to its functionality.
        """
        pass
    
    @abstractmethod
    async def _setup_module(self) -> None:
        """
        Set up module-specific resources and state.
        
        This method should be implemented by each module to initialize
        any resources or state needed for operation.
        """
        pass
    
    async def _cleanup_resources(self) -> None:
        """
        Clean up module resources before shutdown.
        
        This method can be overridden by modules that need to clean up
        resources before shutdown.
        """
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the module.
        
        Returns:
            Dictionary with module status information
        """
        return {
            "module": self.__class__.__name__,
            "initialized": self.initialized,
            "config_keys": list(self.config.keys())
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)