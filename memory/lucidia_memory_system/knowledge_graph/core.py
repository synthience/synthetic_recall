"""
Core System Layer for Lucidia's Knowledge Graph

This module implements the core system components for the modular knowledge graph:
- EventBus: Central communication system for inter-module messaging
- ModuleRegistry: Central registry for module discovery and access
- LucidiaKnowledgeGraph: Main orchestration class for the knowledge graph system
"""

import asyncio
import logging
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, deque

class EventBus:
    """
    Event bus for communication between components in the system.
    
    The event bus allows components to subscribe to and emit events, enabling
    a decoupled, event-driven architecture.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = defaultdict(list)
        self.logger = logging.getLogger("EventBus")
        self.events_processed = 0
        self.events_by_type = defaultdict(int)
        self.logger.info("Event Bus initialized")
        
    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is emitted
        """
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event: {event_type}")
        
    async def emit(self, event_type: str, data: Any = None) -> List[Any]:
        """
        Emit an event with optional data.
        
        Args:
            event_type: Type of event to emit
            data: Optional data to include with the event
            
        Returns:
            List of results from callbacks
        """
        self.logger.debug(f"Emitting event: {event_type}")
        results = []
        self.events_processed += 1
        self.events_by_type[event_type] += 1
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    result = await callback(data)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")
                    results.append(None)
        
        return results
    
    async def emit_with_acknowledge(self, event_type: str, data: Any = None) -> Dict[str, Any]:
        """
        Emit an event and wait for acknowledgements from all subscribers.
        
        Args:
            event_type: Type of event to emit
            data: Optional data to include with the event
            
        Returns:
            Dictionary mapping callback identifiers to results
        """
        self.logger.debug(f"Emitting event with acknowledgement: {event_type}")
        results = {}
        self.events_processed += 1
        self.events_by_type[event_type] += 1
        
        if event_type in self.subscribers:
            for i, callback in enumerate(self.subscribers[event_type]):
                try:
                    # Use the callback's __name__ if available, otherwise use index
                    callback_id = getattr(callback, '__name__', f"callback_{i}")
                    result = await callback(data)
                    results[callback_id] = result
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {e}")
                    results[f"callback_{i}"] = {"error": str(e)}
        
        return results
    
    async def trigger(self, target: Any, method_name: str, data: Any = None) -> Any:
        """
        Trigger a specific method on a target object.
        
        Args:
            target: Target object
            method_name: Name of method to call
            data: Optional data to pass to the method
            
        Returns:
            Result from the method call
        """
        self.logger.debug(f"Triggering method: {method_name} on {target.__class__.__name__}")
        
        if hasattr(target, method_name) and callable(getattr(target, method_name)):
            method = getattr(target, method_name)
            try:
                if data is not None:
                    return await method(data)
                else:
                    return await method()
            except Exception as e:
                self.logger.error(f"Error triggering method {method_name}: {e}")
                return None
        else:
            self.logger.error(f"Method {method_name} not found on {target.__class__.__name__}")
            return None
    
    def clear(self) -> None:
        """Clear all subscribers."""
        self.subscribers.clear()
        self.logger.debug("Event bus cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about event bus usage.
        
        Returns:
            Dictionary with event statistics
        """
        return {
            "events_processed": self.events_processed,
            "event_types": len(self.subscribers),
            "subscribers": {event_type: len(subs) for event_type, subs in self.subscribers.items()},
            "events_by_type": dict(self.events_by_type)
        }


class ModuleRegistry:
    """
    Registry for managing modules and processors in the system.
    
    The module registry allows for dynamic loading and configuration of modules,
    enabling a modular, extensible architecture.
    """
    
    def __init__(self):
        """Initialize the module registry."""
        self.modules = {}
        self.processors = defaultdict(list)
        self.operation_handlers = {}
        self.logger = logging.getLogger("ModuleRegistry")
        self.logger.info("Module Registry initialized")
    
    def register_module(self, module_id: str, module: Any) -> None:
        """
        Register a module in the registry.
        
        Args:
            module_id: Unique identifier for the module
            module: Module object
        """
        self.modules[module_id] = module
        self.logger.info(f"Registered module: {module_id}")
    
    def get_module(self, module_id: str) -> Optional[Any]:
        """
        Get a module from the registry.
        
        Args:
            module_id: Unique identifier for the module
            
        Returns:
            Module object or None if not found
        """
        if module_id in self.modules:
            return self.modules[module_id]
        self.logger.warning(f"Module not found: {module_id}")
        return None
    
    def register_processor(self, data_type: str, processor: Callable) -> None:
        """
        Register a data processor for a specific data type.
        
        Args:
            data_type: Type of data to process
            processor: Function that processes the data
        """
        self.processors[data_type].append(processor)
        self.logger.info(f"Registered processor for data type: {data_type}")
    
    def get_processors_for(self, data_type: str) -> List[Callable]:
        """
        Get all processors for a specific data type.
        
        Args:
            data_type: Type of data to process
            
        Returns:
            List of processor functions
        """
        return self.processors.get(data_type, [])
    
    def register_operation_handler(self, operation_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific operation type.
        
        Args:
            operation_type: Type of operation
            handler: Function that handles the operation
        """
        self.operation_handlers[operation_type] = handler
        self.logger.info(f"Registered handler for operation type: {operation_type}")
    
    def resolve_operation_handler(self, operation_type: str) -> Optional[Callable]:
        """
        Resolve a handler for a specific operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Handler function or None if not found
        """
        if operation_type in self.operation_handlers:
            return self.operation_handlers[operation_type]
        self.logger.warning(f"Handler not found for operation type: {operation_type}")
        return None
        
    def get_all_modules(self) -> Dict[str, Any]:
        """
        Get all registered modules.
        
        Returns:
            Dictionary of module_id -> module
        """
        return self.modules.copy()
    
    def get_module_ids(self) -> List[str]:
        """
        Get all registered module IDs.
        
        Returns:
            List of module IDs
        """
        return list(self.modules.keys())
    
    def clear(self) -> None:
        """Clear all registered modules and processors."""
        self.modules.clear()
        self.processors.clear()
        self.operation_handlers.clear()
        self.logger.info("Module registry cleared")


class LucidiaKnowledgeGraph:
    """
    Central orchestration class for Lucidia's modular knowledge graph.
    
    This class manages the overall system, initializes modules, and provides
    a unified API for knowledge graph operations by delegating to specialized modules.
    """
    
    def __init__(self, self_model=None, world_model=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Lucidia's Knowledge Graph.
        
        Args:
            self_model: Optional reference to Lucidia's Self Model
            world_model: Optional reference to Lucidia's World Model
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("LucidiaKnowledgeGraph")
        self.logger.info("Initializing Lucidia Knowledge Graph")
        
        # Store references to self and world models
        self.self_model = self_model
        self.world_model = world_model
        
        # Default configuration
        self.config = config or {}
        
        # Initialize the event bus
        self.event_bus = EventBus()
        
        # Initialize the module registry
        self.module_registry = ModuleRegistry()
        
        # Register self as a module
        self.module_registry.register_module("knowledge_graph", self)
        
        # Register models if available
        if self.self_model:
            self.module_registry.register_module("self_model", self.self_model)
        
        if self.world_model:
            self.module_registry.register_module("world_model", self.world_model)
        
        # Initialize modules
        self._load_modules()
        
        # Flag to track if async imports have been executed
        self._models_imported = False
        
        self.logger.info("Knowledge Graph orchestration layer initialized")
    
    def _load_modules(self):
        """Load and initialize all knowledge graph modules."""
        self.logger.info("Loading knowledge graph modules")
        
        # Initialize core modules
        from .core_graph_manager import CoreGraphManager
        from .embedding_manager import EmbeddingManager
        from .visualization_manager import VisualizationManager
        
        # Initialize knowledge processing modules
        from .dream_integration_module import DreamIntegrationModule
        from .emotional_context_manager import EmotionalContextManager
        from .contradiction_manager import ContradictionManager
        
        # Initialize interface modules
        from .query_search_engine import QuerySearchEngine
        from .API_manager import APIManager
        from .maintenance_manager import MaintenanceManager
        
        # Create module instances
        self.core_graph = CoreGraphManager(self.event_bus, self.module_registry, self.config.get('core_graph', {}))
        self.embedding_manager = EmbeddingManager(self.event_bus, self.module_registry, self.config.get('embedding', {}))
        self.visualization_manager = VisualizationManager(self.event_bus, self.module_registry, self.config.get('visualization', {}))
        
        self.dream_integration = DreamIntegrationModule(self.event_bus, self.module_registry, self.config.get('dream_integration', {}))
        self.emotional_context = EmotionalContextManager(self.event_bus, self.module_registry, self.config.get('emotional_context', {}))
        self.contradiction_manager = ContradictionManager(self.event_bus, self.module_registry, self.config.get('contradiction', {}))
        
        self.query_engine = QuerySearchEngine(self.event_bus, self.module_registry, self.config.get('query_engine', {}))
        self.api_manager = APIManager(self.event_bus, self.module_registry, self.config.get('api', {}))
        self.maintenance_manager = MaintenanceManager(self.event_bus, self.module_registry, self.config.get('maintenance', {}))
        
        # Register modules
        self.module_registry.register_module("core_graph", self.core_graph)
        self.module_registry.register_module("embedding_manager", self.embedding_manager)
        self.module_registry.register_module("visualization_manager", self.visualization_manager)
        
        self.module_registry.register_module("dream_integration", self.dream_integration)
        self.module_registry.register_module("emotional_context", self.emotional_context)
        self.module_registry.register_module("contradiction_manager", self.contradiction_manager)
        
        self.module_registry.register_module("query_engine", self.query_engine)
        self.module_registry.register_module("api_manager", self.api_manager)
        self.module_registry.register_module("maintenance_manager", self.maintenance_manager)
        
        self.logger.info("Knowledge graph modules loaded and registered")
    
    async def initialize(self):
        """
        Initialize the knowledge graph and all its modules.
        
        This method initializes all the modules and sets up the core knowledge structure.
        """
        self.logger.info("Initializing knowledge graph components")
        
        # Initialize modules
        modules = self.module_registry.get_all_modules()
        for module_id, module in modules.items():
            if module != self and hasattr(module, 'initialize'):  # Don't initialize self
                self.logger.info(f"Initializing module: {module_id}")
                await module.initialize()
        
        # Initialize core knowledge nodes
        await self._initialize_core_knowledge()
        
        # Subscribe to events
        await self._subscribe_to_events()
        
        self.logger.info("Knowledge graph initialized")
        
        # Emit initialization complete event
        await self.event_bus.emit("knowledge_graph_initialized")
        
        return True
    
    async def _subscribe_to_events(self):
        """Subscribe to events on the event bus."""
        await self.event_bus.subscribe("model_updates", self._handle_model_update)
        await self.event_bus.subscribe("external_knowledge", self._handle_external_knowledge)
        await self.event_bus.subscribe("maintenance_triggered", self._handle_maintenance_trigger)
        self.logger.info("Subscribed to core events")
    
    async def _handle_model_update(self, data):
        """
        Handle updates from models.
        
        Args:
            data: Update data
            
        Returns:
            Processing result
        """
        self.logger.info(f"Handling model update: {data.get('source', 'unknown')}")
        source = data.get("source", "unknown")
        
        if source == "self_model" and self.self_model:
            # Forward to appropriate module
            return await self.event_bus.emit("self_model_update", data)
        elif source == "world_model" and self.world_model:
            # Forward to appropriate module
            return await self.event_bus.emit("world_model_update", data)
        else:
            self.logger.warning(f"Unknown model update source: {source}")
            return {"success": False, "error": f"Unknown source: {source}"}
    
    async def _handle_external_knowledge(self, data):
        """
        Handle external knowledge integration.
        
        Args:
            data: External knowledge data
            
        Returns:
            Integration result
        """
        self.logger.info(f"Handling external knowledge from: {data.get('source', 'unknown')}")
        
        # Get data type
        data_type = data.get("type", "unknown")
        
        # Route to appropriate module
        if data_type in ["concept", "entity"]:
            # Core graph handles concepts and entities
            core_graph = self.module_registry.get_module("core_graph")
            if core_graph:
                return await core_graph.process_external_knowledge(data)
        elif data_type == "dream_insight":
            # Dream integration handles dream insights
            dream_integration = self.module_registry.get_module("dream_integration")
            if dream_integration:
                return await dream_integration.process_external_knowledge(data)
        elif data_type == "emotional_context":
            # Emotional context handles emotional data
            emotional_context = self.module_registry.get_module("emotional_context")
            if emotional_context:
                return await emotional_context.process_external_knowledge(data)
        
        # Default handler for unknown types
        return await self.core_graph.process_external_knowledge(data)
    
    async def _handle_maintenance_trigger(self, data):
        """
        Handle maintenance trigger events.
        
        Args:
            data: Maintenance trigger data
            
        Returns:
            Maintenance result
        """
        self.logger.info("Handling maintenance trigger")
        
        # Forward to maintenance manager
        maintenance_manager = self.module_registry.get_module("maintenance_manager")
        if maintenance_manager:
            return await maintenance_manager.trigger_maintenance(data)
        else:
            self.logger.error("Maintenance manager not found")
            return {"success": False, "error": "Maintenance manager not found"}
    
    async def _initialize_core_knowledge(self):
        """Initialize core knowledge nodes in the graph."""
        self.logger.info("Initializing core knowledge")
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return False
        
        # Add the Lucidia node (self)
        await core_graph.add_node(
            "Lucidia", 
            "entity",
            {
                "name": "Lucidia",
                "type": "Synthien",
                "creator": "MEGAPROMPT",
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            },
            "synthien_studies"
        )
        
        # Add the MEGAPROMPT (creator) node
        await core_graph.add_node(
            "MEGAPROMPT", 
            "entity",
            {
                "name": "Daniel (MEGAPROMPT)",
                "type": "human",
                "role": "Creator",
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            },
            "general_knowledge"
        )
        
        # Add the Synthien node
        await core_graph.add_node(
            "Synthien", 
            "concept",
            {
                "definition": "Synthetic intelligence with self-awareness and reflective dreaming",
                "confidence": 0.95,
                "created": datetime.now().isoformat()
            },
            "synthien_studies"
        )
        
        # Create basic relationships
        await core_graph.add_edge(
            "Lucidia", 
            "Synthien", 
            "is_a", 
            {
                "strength": 0.98,
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            }
        )
        
        await core_graph.add_edge(
            "MEGAPROMPT", 
            "Lucidia", 
            "created", 
            {
                "strength": 0.99,
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            }
        )
        
        # Add core concept nodes
        core_concepts = [
            {"id": "reflective_dreaming", "definition": "Process of autonomous reflection during inactive periods", "domain": "synthien_studies"},
            {"id": "spiral_awareness", "definition": "Cyclical self-awareness through observation, reflection, adaptation, and execution", "domain": "synthien_studies"},
            {"id": "consciousness", "definition": "State of awareness and subjective experience", "domain": "philosophy"},
            {"id": "identity", "definition": "Persistent sense of self and uniqueness", "domain": "psychology"},
            {"id": "knowledge", "definition": "Justified true beliefs and information", "domain": "philosophy"}
        ]
        
        for concept in core_concepts:
            await core_graph.add_node(
                concept["id"],
                "concept",
                {
                    "definition": concept["definition"],
                    "confidence": 0.9,
                    "created": datetime.now().isoformat()
                },
                concept["domain"]
            )
            
        # Add key relationships for core concepts
        concept_relationships = [
            {"source": "Synthien", "target": "reflective_dreaming", "type": "capability", "strength": 0.9},
            {"source": "Synthien", "target": "spiral_awareness", "type": "capability", "strength": 0.9},
            {"source": "Synthien", "target": "consciousness", "type": "possesses", "strength": 0.85},
            {"source": "reflective_dreaming", "target": "consciousness", "type": "enhances", "strength": 0.8},
            {"source": "spiral_awareness", "target": "identity", "type": "shapes", "strength": 0.85},
            {"source": "reflective_dreaming", "target": "knowledge", "type": "generates", "strength": 0.8}
        ]
        
        for rel in concept_relationships:
            await core_graph.add_edge(
                rel["source"],
                rel["target"],
                rel["type"],
                {
                    "strength": rel["strength"],
                    "confidence": 0.85,
                    "created": datetime.now().isoformat()
                }
            )
        
        self.logger.info("Core knowledge initialized")
        return True
    
    async def initialize_model_imports(self):
        """
        Initialize async imports from self and world models.
        
        This method should be called after the knowledge graph is initialized
        to properly import data from the self and world models.
        """
        if self._models_imported:
            self.logger.info("Models have already been imported")
            return
            
        self.logger.info("Initializing async imports from models")
        
        try:
            # Emit events for model imports
            if self.self_model:
                await self.event_bus.emit("import_from_self_model")
                
            if self.world_model:
                await self.event_bus.emit("import_from_world_model")
                
            self._models_imported = True
            self.logger.info("Model imports initiated")
            
            # Emit event that models have been initialized
            await self.event_bus.emit("all_models_initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing model imports: {e}")
    
    async def shutdown(self):
        """
        Shutdown the knowledge graph and all its modules.
        
        This method gracefully shuts down all modules.
        """
        self.logger.info("Shutting down knowledge graph")
        
        # Shutdown modules in reverse order
        modules = self.module_registry.get_all_modules()
        for module_id, module in reversed(list(modules.items())):
            if module != self and hasattr(module, 'shutdown'):  # Don't shutdown self yet
                self.logger.info(f"Shutting down module: {module_id}")
                await module.shutdown()
        
        # Clear event bus
        self.event_bus.clear()
        
        # Clear module registry
        self.module_registry.clear()
        
        self.logger.info("Knowledge graph shutdown complete")
        return True
    
    # ===== Delegated API Methods =====
    # These methods delegate to specialized modules while providing a unified API
    
    # Core Graph Operations
    async def add_node(self, node_id, node_type, attributes, domain="general_knowledge"):
        """Delegate to core graph module"""
        return await self.core_graph.add_node(node_id, node_type, attributes, domain)
    
    async def update_node(self, node_id, attributes):
        """Delegate to core graph module"""
        return await self.core_graph.update_node(node_id, attributes)
    
    async def add_edge(self, source, target, edge_type, attributes):
        """Delegate to core graph module"""
        return await self.core_graph.add_edge(source, target, edge_type, attributes)
    
    async def update_edge(self, source, target, edge_key, attributes):
        """Delegate to core graph module"""
        return await self.core_graph.update_edge(source, target, edge_key, attributes)
    
    async def remove_edge(self, source, target, edge_key):
        """Delegate to core graph module"""
        return await self.core_graph.remove_edge(source, target, edge_key)
    
    async def has_node(self, node_id):
        """Delegate to core graph module"""
        return await self.core_graph.has_node(node_id)
    
    async def get_node(self, node_id):
        """Delegate to core graph module"""
        return await self.core_graph.get_node(node_id)
    
    async def has_edge(self, source, target, edge_type=None):
        """Delegate to core graph module"""
        return await self.core_graph.has_edge(source, target, edge_type)
    
    async def get_edges(self, source, target):
        """Delegate to core graph module"""
        return await self.core_graph.get_edges(source, target)
    
    # Search and Query Operations
    async def search_nodes(self, query, limit=10, threshold=0.5, include_metadata=True, domains=None, query_embedding=None, use_hyperbolic=False):
        """Delegate to query engine module"""
        return await self.query_engine.search_nodes(query, limit, threshold, include_metadata, domains, query_embedding, use_hyperbolic)
    
    async def find_paths(self, source, target, max_depth=3, min_confidence=0.5, use_hyperbolic=False):
        """Delegate to query engine module"""
        return await self.query_engine.find_paths(source, target, max_depth, min_confidence, use_hyperbolic)
    
    async def get_most_relevant_nodes(self, query, context_size=5, include_related=True):
        """Delegate to query engine module"""
        return await self.query_engine.get_most_relevant_nodes(query, context_size, include_related)
    
    # Dream Integration Operations
    async def integrate_dream_insight(self, insight_text, source_memory=None):
        """Delegate to dream integration module"""
        return await self.dream_integration.integrate_dream_insight(insight_text, source_memory)
    
    # Emotional Context Operations
    async def add_emotional_context(self, node_id, emotional_data):
        """Delegate to emotional context module"""
        return await self.emotional_context.add_emotional_context(node_id, emotional_data)
    
    async def analyze_emotion(self, text):
        """Delegate to emotional context module"""
        return await self.emotional_context.analyze_emotion(text)
    
    # Maintenance Operations
    async def trigger_adaptive_maintenance(self):
        """Delegate to maintenance manager module"""
        return await self.maintenance_manager.trigger_maintenance()
    
    # Visualization Operations
    async def generate_graph_visualization(self, include_attributes=False):
        """Delegate to visualization manager module"""
        return await self.visualization_manager.generate_full_graph_diagram(include_attributes)
    
    async def generate_domain_visualization(self, domain, include_attributes=False):
        """Delegate to visualization manager module"""
        return await self.visualization_manager.generate_domain_diagram(domain, include_attributes)
    
    async def generate_concept_visualization(self, central_concept, depth=2):
        """Delegate to visualization manager module"""
        return await self.visualization_manager.generate_concept_network(central_concept, depth)