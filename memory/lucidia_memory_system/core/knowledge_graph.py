"""
Lucidia's Knowledge Graph - Enhanced Version

This module implements Lucidia's semantic knowledge graph with advanced event-driven architecture,
hierarchical operations, meta-learning feedback loops, contradiction resolution, and adaptive 
maintenance as specified in the Lucidia Knowledge Graph sequence diagram.

Created by MEGAPROMPT (Daniel)
"""

import json
import os
import time
import math
import random
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from datetime import datetime
from collections import defaultdict, deque
import heapq
import uuid
import asyncio
import threading
import websockets
import concurrent.futures
import re
import copy
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')


class EventBus:
    """Event bus for communication between components in the system.
    
    The event bus allows components to subscribe to and emit events, enabling
    a decoupled, event-driven architecture.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = defaultdict(list)
        self.logger = logging.getLogger("EventBus")
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


class ModuleRegistry:
    """Registry for managing modules and processors in the system.
    
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


class LucidiaKnowledgeGraph:
    """
    Lucidia's semantic knowledge graph for managing interconnected knowledge and insights.
    
    The knowledge graph creates a rich network of relationships between concepts, entities,
    and insights derived from both structured knowledge and dream-influenced reflection,
    serving as a bridge between Lucidia's self and world models.
    """
    
    def __init__(self, self_model=None, world_model=None, config: Optional[Dict[str, Any]] = None, emotion_analyzer_url: Optional[str] = None):
        """
        Initialize Lucidia's Knowledge Graph.
        
        Args:
            self_model: Optional reference to Lucidia's Self Model
            world_model: Optional reference to Lucidia's World Model
            config: Optional configuration dictionary
            emotion_analyzer_url: Optional URL for the emotion analyzer service
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
        
        # Initialize the core graph using NetworkX
        self.graph = nx.MultiDiGraph()
        
        # Node type tracking
        self.node_types = {
            "concept": set(),
            "entity": set(),
            "attribute": set(),
            "dream_insight": set(),
            "memory": set(),
            "self_aspect": set(),
            "event": set(),
            "domain": set(),
            "dream_report": set(),
            "value": set(),
            "goal": set(),
            "emotion": set()  # Added emotion as a node type
        }
        
        # Edge type tracking
        self.edge_types = set()
        
        # Node attributes
        self.node_attributes = {}
        
        # Track nodes influenced by dreams
        self.dream_influenced_nodes = set()
        
        # Track nodes with emotional context
        self.emotion_enhanced_nodes = set()
        
        # Relationship strength decay factors
        # (how quickly relationship strength fades without reinforcement)
        self.relationship_decay = {
            "standard": 0.01,  # Regular relationships decay slowly
            "dream_associated": 0.02,  # Dream associations fade a bit faster
            "memory_derived": 0.03,  # Memory-based connections fade faster
            "speculative": 0.04,  # Speculative connections fade fastest
            "emotional": 0.015  # Emotional relationships decay between standard and dream_associated
        }
        
        # Tracking variables for graph complexity
        self.total_nodes = 0
        self.total_edges = 0
        self.last_pruning = datetime.now()
        
        # Knowledge domain colors for visualization
        self.domain_colors = {
            "synthien_studies": "#9C27B0",  # Purple
            "science": "#2196F3",  # Blue
            "technology": "#4CAF50",  # Green
            "philosophy": "#FF9800",  # Orange
            "art": "#E91E63",  # Pink
            "psychology": "#00BCD4",  # Cyan
            "sociology": "#8BC34A",  # Light Green
            "history": "#795548",  # Brown
            "linguistics": "#9E9E9E",  # Grey
            "economics": "#FFC107",  # Amber
            "ethics": "#3F51B5",  # Indigo
            "general_knowledge": "#607D8B"  # Blue Grey
        }
        
        # Emotion colors for visualization
        self.emotion_colors = {
            "joy": "#FFD700",  # Gold
            "sadness": "#4682B4",  # Steel Blue
            "anger": "#FF4500",  # Red Orange
            "fear": "#800080",  # Purple
            "disgust": "#006400",  # Dark Green
            "surprise": "#FF69B4",  # Hot Pink
            "neutral": "#CCCCCC"  # Light Gray
        }
        
        # Relationship strength thresholds for visualization
        self.relationship_thresholds = {
            "weak": 0.3,
            "moderate": 0.6,
            "strong": 0.8
        }
        
        # Path finding parameters
        self.path_finding = {
            "max_depth": 5,  # Maximum depth for path search
            "min_strength": 0.3,  # Minimum relationship strength to consider
            "relevance_emphasis": 0.7,  # How much to emphasize relevance vs. path length
            "exploration_factor": 0.2  # Randomness in exploration
        }
        
        # Spiral awareness integration
        self.spiral_integration = {
            "observation_emphasis": 0.8,  # During observation phase of spiral
            "reflection_emphasis": 0.9,  # During reflection phase
            "adaptation_emphasis": 0.7,  # During adaptation phase
            "execution_emphasis": 0.6,  # During execution phase
            "current_phase": "observation"  # Default phase
        }
        
        # Dreaming integration
        self.dream_integration = {
            "insight_incorporation_rate": 0.8,  # How readily dream insights are incorporated
            "dream_association_strength": 0.7,  # Initial strength of dream associations
            "dream_derived_nodes": set(),  # Nodes created from dreams
            "dream_enhanced_nodes": set(),  # Existing nodes enhanced by dreams
            "dream_insight_count": 0,  # Number of dream insights integrated
            # Meta-learning parameters
            "integration_quality_scores": [],  # Track quality of integrations
            "learning_rate": 0.05,  # Rate at which parameters are adjusted
            "quality_threshold": 0.7,  # Threshold for good integration quality
            "adaptation_frequency": 5  # Adjust parameters after this many integrations
        }
        
        # Emotion integration
        self.emotion_integration = {
            "emotion_incorporation_rate": 0.85,  # How readily emotional context is incorporated
            "emotion_association_strength": 0.75,  # Initial strength of emotional associations
            "emotion_derived_nodes": set(),  # Nodes created from emotional analysis
            "emotion_enhanced_nodes": set(),  # Existing nodes enhanced by emotional context
            "primary_emotions": ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
            "emotion_count": 0  # Number of emotion insights integrated
        }
        
        # Emotion analyzer service
        self.emotion_analyzer_url = emotion_analyzer_url
        self.emotion_analyzer_ws = None
        
        # Query optimization
        self.query_cache = {}  # Cache for frequent queries
        self.query_stats = defaultdict(int)  # Track query frequency
        
        # Contradiction tracking and resolution
        self.contradiction_tracking = {
            "detected_contradictions": [],  # List of detected contradictions
            "resolution_history": [],  # History of resolutions
            "confidence_threshold": 0.7,  # Threshold for contradiction detection
            "auto_resolution_threshold": 0.9,  # Threshold for automatic resolution
        }
        
        # Hierarchical graph operations
        self.hierarchical_ops = {
            "max_subgraph_size": 1000,  # Maximum size of a subgraph
            "subgraph_overlap": 0.2,  # Overlap between subgraphs
            "parallel_processing": True,  # Whether to use parallel processing
            "thread_pool_size": 4  # Size of thread pool for parallel processing
        }
        
        # Adaptive maintenance
        self.adaptive_maintenance = {
            "last_maintenance": datetime.now(),
            "maintenance_interval": 24 * 60 * 60,  # 24 hours in seconds
            "optimization_history": [],  # History of optimizations
            "metrics_history": [],  # History of metrics
            "metric_thresholds": {
                "density": 0.01,  # Maximum density before pruning
                "average_degree": 10,  # Maximum average degree before pruning
                "clustering": 0.3,  # Target clustering coefficient
                "assortativity": 0.1  # Target assortativity coefficient
            }
        }
        
        # External knowledge integration
        self.external_knowledge = {
            "sources": [],  # List of external knowledge sources
            "integration_history": [],  # History of integrations
            "source_credibility": {},  # Credibility of sources
            "validation_threshold": 0.7,  # Threshold for validation
            "recent_integrations": deque(maxlen=100)  # Recent integrations
        }
        
        # Create thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.hierarchical_ops["thread_pool_size"],
            thread_name_prefix="KG-Worker"
        )
        
        # Register core modules
        self._register_core_modules()
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Initialize core nodes based on provided models if available
        self._initialize_core_nodes()
            
        self.logger.info(f"Knowledge Graph initialized with {self.total_nodes} nodes and {self.total_edges} edges")
        
        # Flag to track if async imports have been executed
        self._models_imported = False
    
    def _register_core_modules(self) -> None:
        """Register core modules with the module registry."""
        # Register self as a module
        self.module_registry.register_module("knowledge_graph", self)
        
        # Register models if available
        if self.self_model:
            self.module_registry.register_module("self_model", self.self_model)
        
        if self.world_model:
            self.module_registry.register_module("world_model", self.world_model)
        
        # Register operation handlers
        self.module_registry.register_operation_handler("search", self.search_nodes)
        self.module_registry.register_operation_handler("path_finding", self.find_paths)
        self.module_registry.register_operation_handler("relevance", self.get_most_relevant_nodes)
        self.module_registry.register_operation_handler("dream_integration", self.integrate_dream_insight)
        self.module_registry.register_operation_handler("emotion_analysis", self.add_emotional_context)
        
        # Register data processors
        self.module_registry.register_processor("text", self._process_text_data)
        self.module_registry.register_processor("concept", self._process_concept_data)
        self.module_registry.register_processor("entity", self._process_entity_data)
        self.module_registry.register_processor("relationship", self._process_relationship_data)
        self.module_registry.register_processor("dream_insight", self._process_dream_insight)
        
        self.logger.info("Core modules registered")
    
    async def _subscribe_to_events(self) -> None:
        """Subscribe to events on the event bus."""
        # Subscribe to model updates
        await self.event_bus.subscribe("model_updates", self._handle_model_update)
        
        # Subscribe to external knowledge
        await self.event_bus.subscribe("external_knowledge", self._handle_external_knowledge)
        
        # Subscribe to contradiction detection
        await self.event_bus.subscribe("contradiction_detected", self._handle_contradiction)
        
        # Subscribe to concept updates
        await self.event_bus.subscribe("update_concept", self._handle_concept_update)
        
        # Subscribe to knowledge integration events
        await self.event_bus.subscribe("knowledge_integrated", self._log_knowledge_integration)
        
        self.logger.info("Subscribed to core events")
    
    async def _handle_model_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle updates from models.
        
        Args:
            data: Update data
            
        Returns:
            Processing result
        """
        self.logger.info(f"Handling model update: {data.get('source', 'unknown')}")
        
        source = data.get("source", "unknown")
        update_type = data.get("type", "unknown")
        update_data = data.get("data", {})
        
        if source == "self_model":
            result = await self._process_self_model_data(update_data)
        elif source == "world_model":
            result = await self._process_world_model_data(update_data)
        else:
            self.logger.warning(f"Unknown model update source: {source}")
            result = {"success": False, "error": f"Unknown source: {source}"}
        
        # Acknowledge the update
        return {"success": True, "result": result}
    
    async def _handle_external_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle external knowledge integration.
        
        Args:
            data: External knowledge data
            
        Returns:
            Integration result
        """
        self.logger.info(f"Handling external knowledge from: {data.get('source', 'unknown')}")
        
        # Validate the data
        valid, validation_result = await self._validate_external_knowledge(data)
        
        if not valid:
            self.logger.warning(f"External knowledge validation failed: {validation_result}")
            return {"success": False, "error": f"Validation failed: {validation_result}"}
        
        # Analyze relevance
        relevance, relevance_score = await self._analyze_relevance(data)
        
        if not relevance:
            self.logger.info(f"External knowledge not relevant (score: {relevance_score})")
            return {"success": False, "error": "Not relevant", "score": relevance_score}
        
        # Get processors for the data type
        data_type = data.get("type", "unknown")
        processors = self.module_registry.get_processors_for(data_type)
        
        if not processors:
            self.logger.warning(f"No processors found for data type: {data_type}")
            return {"success": False, "error": f"No processors for type: {data_type}"}
        
        # Process the data
        processed_data = data.copy()
        for processor in processors:
            try:
                processed_data = await processor(processed_data)
            except Exception as e:
                self.logger.error(f"Error in processor for {data_type}: {e}")
                return {"success": False, "error": f"Processing error: {str(e)}"}
        
        # Store and index the processed data
        integration_result = await self._store_and_index(processed_data)
        
        # Emit knowledge integrated event
        metadata = {
            "source": data.get("source", "unknown"),
            "type": data_type,
            "timestamp": datetime.now().isoformat(),
            "relevance_score": relevance_score,
            "integration_id": str(uuid.uuid4())
        }
        await self.event_bus.emit("knowledge_integrated", metadata)
        
        # Log the integration
        self.external_knowledge["recent_integrations"].append({
            "data": processed_data,
            "metadata": metadata,
            "result": integration_result
        })
        
        return {"success": True, "result": integration_result, "metadata": metadata}
    
    async def _validate_external_knowledge(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate external knowledge data.
        
        Args:
            data: External knowledge data
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Check required fields
        required_fields = ["source", "type", "data"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check source credibility
        source = data.get("source", "unknown")
        credibility = self.external_knowledge["source_credibility"].get(source, 0.5)
        
        if credibility < self.external_knowledge["validation_threshold"]:
            return False, f"Source credibility below threshold: {credibility}"
        
        # Validate based on data type
        data_type = data.get("type", "unknown")
        
        if data_type == "concept":
            # Validate concept data
            concept_data = data.get("data", {})
            if "id" not in concept_data or "definition" not in concept_data:
                return False, "Concept data missing required fields"
        
        elif data_type == "entity":
            # Validate entity data
            entity_data = data.get("data", {})
            if "id" not in entity_data or "name" not in entity_data:
                return False, "Entity data missing required fields"
        
        elif data_type == "relationship":
            # Validate relationship data
            relationship_data = data.get("data", {})
            if "source" not in relationship_data or "target" not in relationship_data or "type" not in relationship_data:
                return False, "Relationship data missing required fields"
        
        # All validations passed
        return True, "Valid"
    
    async def _analyze_relevance(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Analyze relevance of external knowledge.
        
        Args:
            data: External knowledge data
            
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        # Default relevance score
        relevance_score = 0.5
        
        # Calculate relevance based on data type
        data_type = data.get("type", "unknown")
        data_content = data.get("data", {})
        
        if data_type == "concept":
            # Check if we already have this concept
            concept_id = data_content.get("id", "")
            if await self.has_node(concept_id):
                # Higher relevance for existing concepts
                relevance_score = 0.8
            else:
                # Check if it's connected to existing concepts
                definition = data_content.get("definition", "")
                related_concepts = []
                
                # Search for existing concepts that might be related
                for node_id, attrs in self.graph.nodes(data=True):
                    if attrs.get("type") == "concept":
                        node_def = attrs.get("definition", "")
                        if node_def and self._text_similarity(definition, node_def) > 0.5:
                            related_concepts.append(node_id)
                
                # Relevance based on connections to existing concepts
                if related_concepts:
                    relevance_score = 0.6 + (min(len(related_concepts), 5) / 10)
        
        elif data_type == "entity":
            # Check if we already have this entity
            entity_id = data_content.get("id", "")
            if await self.has_node(entity_id):
                # Higher relevance for existing entities
                relevance_score = 0.8
            else:
                # Check importance of entity
                importance = data_content.get("importance", 0.5)
                relevance_score = 0.4 + (importance * 0.4)
        
        elif data_type == "relationship":
            # Check if the relationship connects existing nodes
            source = data_content.get("source", "")
            target = data_content.get("target", "")
            
            source_exists = await self.has_node(source)
            target_exists = await self.has_node(target)
            
            if source_exists and target_exists:
                # Higher relevance for relationships between existing nodes
                relevance_score = 0.9
            elif source_exists or target_exists:
                # Medium relevance if one node exists
                relevance_score = 0.6
            else:
                # Lower relevance if neither node exists
                relevance_score = 0.3
        
        # Relevance threshold
        return relevance_score >= self.external_knowledge["validation_threshold"], relevance_score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity for demonstration
        # In a full implementation, use more sophisticated NLP methods
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _store_and_index(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store and index processed external knowledge.
        
        Args:
            data: Processed knowledge data
            
        Returns:
            Storage result
        """
        data_type = data.get("type", "unknown")
        data_content = data.get("data", {})
        result = {"success": False}
        
        try:
            if data_type == "concept":
                # Store concept as a node
                concept_id = data_content.get("id", "")
                definition = data_content.get("definition", "")
                domain = data_content.get("domain", "general_knowledge")
                confidence = data_content.get("confidence", 0.7)
                
                success = await self.add_node(
                    concept_id,
                    "concept",
                    {
                        "definition": definition,
                        "confidence": confidence,
                        "source": data.get("source", "external"),
                        "created": datetime.now().isoformat()
                    },
                    domain
                )
                
                result = {"success": success, "node_id": concept_id, "type": "concept"}
            
            elif data_type == "entity":
                # Store entity as a node
                entity_id = data_content.get("id", "")
                name = data_content.get("name", "")
                description = data_content.get("description", "")
                domain = data_content.get("domain", "general_knowledge")
                confidence = data_content.get("confidence", 0.7)
                
                success = await self.add_node(
                    entity_id,
                    "entity",
                    {
                        "name": name,
                        "description": description,
                        "confidence": confidence,
                        "source": data.get("source", "external"),
                        "created": datetime.now().isoformat()
                    },
                    domain
                )
                
                result = {"success": success, "node_id": entity_id, "type": "entity"}
            
            elif data_type == "relationship":
                # Store relationship as an edge
                source = data_content.get("source", "")
                target = data_content.get("target", "")
                rel_type = data_content.get("type", "related_to")
                strength = data_content.get("strength", 0.7)
                confidence = data_content.get("confidence", 0.7)
                
                # Create nodes if they don't exist
                if not await self.has_node(source):
                    await self.add_node(
                        source,
                        data_content.get("source_type", "entity"),
                        {
                            "name": data_content.get("source_name", source),
                            "placeholder": True,
                            "confidence": 0.5,
                            "created": datetime.now().isoformat()
                        }
                    )
                
                if not await self.has_node(target):
                    await self.add_node(
                        target,
                        data_content.get("target_type", "entity"),
                        {
                            "name": data_content.get("target_name", target),
                            "placeholder": True,
                            "confidence": 0.5,
                            "created": datetime.now().isoformat()
                        }
                    )
                
                # Add the edge
                edge_key = await self.add_edge(
                    source,
                    target,
                    rel_type,
                    {
                        "strength": strength,
                        "confidence": confidence,
                        "source": data.get("source", "external"),
                        "created": datetime.now().isoformat()
                    }
                )
                
                result = {"success": edge_key is not None, "edge_key": edge_key, "type": "relationship"}
            
            # Store integration in history
            self.external_knowledge["integration_history"].append({
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "source": data.get("source", "unknown"),
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error storing and indexing {data_type}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_text_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text data.
        
        Args:
            data: Text data
            
        Returns:
            Processed data
        """
        # Extract concepts and entities from text
        text_content = data.get("data", {}).get("content", "")
        
        if not text_content:
            return data
        
        # Process with world model if available
        if self.world_model and hasattr(self.world_model, '_extract_concepts'):
            try:
                concepts = self.world_model._extract_concepts(text_content)
                data["extracted_concepts"] = concepts
            except Exception as e:
                self.logger.warning(f"Error extracting concepts with world model: {e}")
        
        # Simple regex-based extraction as fallback
        if "extracted_concepts" not in data:
            # Extract potential concepts and entities
            concept_patterns = [
                r"concept of (\w+)",
                r"(\w+) is defined as",
                r"(\w+) refers to",
                r"understanding of (\w+)"
            ]
            
            extracted = []
            for pattern in concept_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                extracted.extend(matches)
            
            data["extracted_concepts"] = extracted
        
        # Add emotional context if available
        try:
            emotion_results = await self.analyze_emotion(text_content)
            if emotion_results:
                data["emotional_context"] = emotion_results
        except Exception as e:
            self.logger.warning(f"Error analyzing emotions in text: {e}")
        
        return data
    
    async def _process_concept_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process concept data.
        
        Args:
            data: Concept data
            
        Returns:
            Processed concept data
        """
        concept = data.get("data", {})
        
        # Ensure concept has an ID
        if "id" not in concept:
            concept["id"] = f"concept_{uuid.uuid4().hex[:8]}"
        
        # Ensure concept has a definition
        if "definition" not in concept:
            self.logger.warning(f"Concept {concept['id']} missing definition")
            concept["definition"] = f"Concept from {data.get('source', 'unknown')}"
        
        # Set confidence if not provided
        if "confidence" not in concept:
            concept["confidence"] = 0.7
        
        # Add domain if not provided
        if "domain" not in concept:
            concept["domain"] = "general_knowledge"
        
        data["data"] = concept
        return data
    
    async def _process_entity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entity data.
        
        Args:
            data: Entity data
            
        Returns:
            Processed entity data
        """
        entity = data.get("data", {})
        
        # Ensure entity has an ID
        if "id" not in entity:
            entity["id"] = f"entity_{uuid.uuid4().hex[:8]}"
        
        # Ensure entity has a name
        if "name" not in entity:
            self.logger.warning(f"Entity {entity['id']} missing name")
            entity["name"] = entity["id"]
        
        # Set confidence if not provided
        if "confidence" not in entity:
            entity["confidence"] = 0.7
        
        # Add domain if not provided
        if "domain" not in entity:
            entity["domain"] = "general_knowledge"
        
        data["data"] = entity
        return data
    
    async def _process_relationship_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process relationship data.
        
        Args:
            data: Relationship data
            
        Returns:
            Processed relationship data
        """
        relationship = data.get("data", {})
        
        # Ensure relationship has source and target
        if "source" not in relationship or "target" not in relationship:
            self.logger.error("Relationship missing source or target")
            raise ValueError("Relationship must have source and target")
        
        # Ensure relationship has a type
        if "type" not in relationship:
            relationship["type"] = "related_to"
        
        # Set strength if not provided
        if "strength" not in relationship:
            relationship["strength"] = 0.7
        
        # Set confidence if not provided
        if "confidence" not in relationship:
            relationship["confidence"] = 0.7
        
        data["data"] = relationship
        return data
    
    async def _process_dream_insight(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process dream insight data.
        
        Args:
            data: Dream insight data
            
        Returns:
            Processed dream insight data
        """
        dream_insight = data.get("data", {})
        
        # Ensure dream insight has an ID
        if "id" not in dream_insight:
            dream_insight["id"] = f"dream:{self.dream_integration['dream_insight_count']}"
        
        # Ensure dream insight has content
        if "insight" not in dream_insight:
            self.logger.error("Dream insight missing content")
            raise ValueError("Dream insight must have content")
        
        # Set confidence if not provided
        if "confidence" not in dream_insight:
            dream_insight["confidence"] = 0.8
        
        # Add timestamp if not provided
        if "timestamp" not in dream_insight:
            dream_insight["timestamp"] = datetime.now().isoformat()
        
        data["data"] = dream_insight
        return data
    
    async def _handle_contradiction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle contradiction detection and resolution.
        
        Args:
            data: Contradiction data
            
        Returns:
            Resolution result
        """
        self.logger.info(f"Handling contradiction: {data.get('type', 'unknown')}")
        
        # Get contradiction details
        contradiction_type = data.get("type", "unknown")
        source_id = data.get("source", "")
        target_id = data.get("target", "")
        context = data.get("context", {})
        
        # Track the contradiction
        contradiction_id = str(uuid.uuid4())
        contradiction_record = {
            "id": contradiction_id,
            "timestamp": datetime.now().isoformat(),
            "type": contradiction_type,
            "source": source_id,
            "target": target_id,
            "context": context,
            "status": "detected"
        }
        
        self.contradiction_tracking["detected_contradictions"].append(contradiction_record)
        
        # Analyze the contradiction
        analysis_result = await self._analyze_contradiction(source_id, target_id, contradiction_type, context)
        resolution_method = analysis_result.get("resolution_method", "unknown")
        
        # Update contradiction record
        contradiction_record["analysis"] = analysis_result
        contradiction_record["status"] = "analyzed"
        
        # Resolve based on method
        if resolution_method == "internal":
            # Internal resolution
            resolution_result = await self._apply_internal_resolution(
                source_id, target_id, contradiction_type, analysis_result
            )
            
            # Update contradiction record
            contradiction_record["resolution"] = resolution_result
            contradiction_record["status"] = "resolved_internally"
            
            # Emit resolution event
            await self.event_bus.emit("contradiction_resolved", {
                "contradiction_id": contradiction_id,
                "resolution": resolution_result
            })
            
            return {"success": True, "method": "internal", "result": resolution_result}
            
        else:
            # External resolution required
            # Emit event requesting resolution
            await self.event_bus.emit("resolution_required", {
                "contradiction_id": contradiction_id,
                "source": source_id,
                "target": target_id,
                "type": contradiction_type,
                "analysis": analysis_result,
                "context": context
            })
            
            # Update contradiction record
            contradiction_record["status"] = "awaiting_external_resolution"
            
            return {"success": True, "method": "external", "contradiction_id": contradiction_id}
    
    async def _analyze_contradiction(self, source_id: str, target_id: str, 
                             contradiction_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a contradiction and determine resolution approach.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            contradiction_type: Type of contradiction
            context: Additional context
            
        Returns:
            Analysis results
        """
        # Get node data
        source_node = await self.get_node(source_id)
        target_node = await self.get_node(target_id)
        
        if not source_node or not target_node:
            return {
                "can_resolve": False,
                "resolution_method": "external",
                "reason": "One or both nodes not found"
            }
        
        # Check node confidence
        source_confidence = source_node.get("confidence", 0.5)
        target_confidence = target_node.get("confidence", 0.5)
        
        # Check edge data if available
        edge_data = []
        if await self.has_edge(source_id, target_id):
            edge_data = await self.get_edges(source_id, target_id)
        
        # Analyze based on contradiction type
        if contradiction_type == "conflicting_relationship":
            # Check if we can resolve based on confidence
            confidence_diff = abs(source_confidence - target_confidence)
            
            if confidence_diff > 0.3:
                # Significant confidence difference, can resolve internally
                return {
                    "can_resolve": True,
                    "resolution_method": "internal",
                    "higher_confidence": source_id if source_confidence > target_confidence else target_id,
                    "confidence_diff": confidence_diff,
                    "strategy": "trust_higher_confidence"
                }
            else:
                # Similar confidence, need external input
                return {
                    "can_resolve": False,
                    "resolution_method": "external",
                    "confidence_diff": confidence_diff,
                    "reason": "Similar confidence levels"
                }
        
        elif contradiction_type == "conflicting_attribute":
            # Check attribute source and timestamp
            source_attrs = source_node.get("attributes", {})
            target_attrs = target_node.get("attributes", {})
            
            source_timestamp = source_attrs.get("created", "")
            target_timestamp = target_attrs.get("created", "")
            
            # Try to resolve based on recency
            if source_timestamp and target_timestamp:
                try:
                    source_time = datetime.fromisoformat(source_timestamp)
                    target_time = datetime.fromisoformat(target_timestamp)
                    
                    if abs((source_time - target_time).total_seconds()) > 86400:  # More than a day apart
                        # Can resolve by recency
                        return {
                            "can_resolve": True,
                            "resolution_method": "internal",
                            "newer_node": source_id if source_time > target_time else target_id,
                            "time_diff_seconds": abs((source_time - target_time).total_seconds()),
                            "strategy": "trust_newer"
                        }
                except (ValueError, TypeError):
                    pass
            
            # Cannot resolve internally
            return {
                "can_resolve": False,
                "resolution_method": "external",
                "reason": "Cannot determine resolution strategy"
            }
        
        elif contradiction_type == "logical_inconsistency":
            # Logical inconsistencies typically need external resolution
            return {
                "can_resolve": False,
                "resolution_method": "external",
                "reason": "Logical inconsistencies require domain knowledge"
            }
        
        else:
            # Unknown contradiction type
            return {
                "can_resolve": False,
                "resolution_method": "external",
                "reason": f"Unknown contradiction type: {contradiction_type}"
            }
    
    async def _apply_internal_resolution(self, source_id: str, target_id: str, 
                                 contradiction_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply internal resolution to a contradiction.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            contradiction_type: Type of contradiction
            analysis: Contradiction analysis
            
        Returns:
            Resolution result
        """
        strategy = analysis.get("strategy", "unknown")
        
        if strategy == "trust_higher_confidence":
            # Trust the node with higher confidence
            higher_confidence_id = analysis.get("higher_confidence", "")
            lower_confidence_id = source_id if higher_confidence_id == target_id else target_id
            
            if contradiction_type == "conflicting_relationship":
                # For relationship contradictions, adjust edge attributes
                edges = await self.get_edges(source_id, target_id)
                
                for edge in edges:
                    edge_key = edge.get("key")
                    await self.update_edge(source_id, target_id, edge_key, {
                        "confidence": edge.get("confidence", 0.5) * 0.8,  # Reduce confidence
                        "strength": edge.get("strength", 0.5) * 0.8,  # Reduce strength
                        "contradiction_resolved": True,
                        "resolution_method": "trust_higher_confidence",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
            
            elif contradiction_type == "conflicting_attribute":
                # For attribute contradictions, update the lower confidence node
                lower_node = await self.get_node(lower_confidence_id)
                if lower_node:
                    await self.update_node(lower_confidence_id, {
                        "confidence": lower_node.get("confidence", 0.5) * 0.8,  # Reduce confidence
                        "contradiction_resolved": True,
                        "resolution_method": "trust_higher_confidence",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
            
            return {
                "success": True,
                "strategy": strategy,
                "trusted_node": higher_confidence_id
            }
        
        elif strategy == "trust_newer":
            # Trust the newer node
            newer_node_id = analysis.get("newer_node", "")
            older_node_id = source_id if newer_node_id == target_id else target_id
            
            if contradiction_type == "conflicting_attribute":
                # Update the older node
                older_node = await self.get_node(older_node_id)
                if older_node:
                    await self.update_node(older_node_id, {
                        "outdated": True,
                        "superseded_by": newer_node_id,
                        "contradiction_resolved": True,
                        "resolution_method": "trust_newer",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
            
            return {
                "success": True,
                "strategy": strategy,
                "trusted_node": newer_node_id
            }
        
        else:
            # Unknown strategy
            return {
                "success": False,
                "error": f"Unknown resolution strategy: {strategy}"
            }
    
    async def apply_external_resolution(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply external resolution to a contradiction.
        
        Args:
            decision: Resolution decision
            
        Returns:
            Resolution result
        """
        contradiction_id = decision.get("contradiction_id", "")
        resolution_action = decision.get("action", "")
        source_id = decision.get("source", "")
        target_id = decision.get("target", "")
        resolution_data = decision.get("data", {})
        
        # Find the contradiction record
        contradiction_record = None
        for record in self.contradiction_tracking["detected_contradictions"]:
            if record.get("id") == contradiction_id:
                contradiction_record = record
                break
        
        if not contradiction_record:
            self.logger.warning(f"Contradiction record not found: {contradiction_id}")
            return {"success": False, "error": "Contradiction record not found"}
        
        # Apply resolution based on action
        if resolution_action == "update_source":
            # Update source node
            source_updates = resolution_data.get("updates", {})
            if source_updates and await self.has_node(source_id):
                await self.update_node(source_id, source_updates)
        
        elif resolution_action == "update_target":
            # Update target node
            target_updates = resolution_data.get("updates", {})
            if target_updates and await self.has_node(target_id):
                await self.update_node(target_id, target_updates)
        
        elif resolution_action == "create_relationship":
            # Create a new relationship
            relationship = resolution_data.get("relationship", {})
            if relationship and await self.has_node(source_id) and await self.has_node(target_id):
                rel_type = relationship.get("type", "related_to")
                attributes = relationship.get("attributes", {})
                
                await self.add_edge(source_id, target_id, rel_type, attributes)
        
        elif resolution_action == "delete_relationship":
            # Delete an existing relationship
            if await self.has_edge(source_id, target_id):
                edge_key = resolution_data.get("edge_key")
                if edge_key is not None:
                    await self.remove_edge(source_id, target_id, edge_key)
                else:
                    # Remove all edges between these nodes
                    edges = await self.get_edges(source_id, target_id)
                    for edge in edges:
                        await self.remove_edge(source_id, target_id, edge.get("key"))
        
        elif resolution_action == "merge_nodes":
            # Merge two nodes
            merge_target = resolution_data.get("merge_target", source_id)
            merge_source = target_id if merge_target == source_id else source_id
            
            await self._merge_nodes(merge_source, merge_target)
        
        else:
            # Unknown action
            return {"success": False, "error": f"Unknown resolution action: {resolution_action}"}
        
        # Update contradiction record
        contradiction_record["status"] = "resolved_externally"
        contradiction_record["resolution"] = {
            "action": resolution_action,
            "data": resolution_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to resolution history
        self.contradiction_tracking["resolution_history"].append({
            "contradiction_id": contradiction_id,
            "action": resolution_action,
            "timestamp": datetime.now().isoformat()
        })
        
        # Emit knowledge consistent event
        await self.event_bus.emit("knowledge_consistent", {
            "contradiction_id": contradiction_id,
            "action": resolution_action
        })
        
        return {"success": True, "action": resolution_action}
    
    async def _merge_nodes(self, source_id: str, target_id: str) -> bool:
        """
        Merge source node into target node.
        
        Args:
            source_id: Source node ID to merge from
            target_id: Target node ID to merge into
            
        Returns:
            Success status
        """
        if not await self.has_node(source_id) or not await self.has_node(target_id):
            return False
        
        source_node = await self.get_node(source_id)
        target_node = await self.get_node(target_id)
        
        if not source_node or not target_node:
            return False
        
        # Merge attributes
        source_attrs = source_node.copy()
        target_attrs = target_node.copy()
        
        # Keep higher confidence values from source
        for key, value in source_attrs.items():
            if key not in target_attrs or (
                key in target_attrs and
                isinstance(value, (int, float)) and
                isinstance(target_attrs[key], (int, float)) and
                value > target_attrs[key]
            ):
                target_attrs[key] = value
        
        # Add merged_from attribute
        if "merged_from" not in target_attrs:
            target_attrs["merged_from"] = []
        if isinstance(target_attrs["merged_from"], list):
            target_attrs["merged_from"].append(source_id)
        
        # Update target node
        await self.update_node(target_id, target_attrs)
        
        # Redirect all edges from source to target
        # Outgoing edges
        for _, neighbor, key, data in self.graph.out_edges(source_id, keys=True, data=True):
            if neighbor != target_id:  # Avoid self-loops
                # Check if edge already exists
                existing_edges = await self.get_edges(target_id, neighbor)
                existing_types = [e.get("type") for e in existing_edges]
                
                if data.get("type") not in existing_types:
                    # Add edge if type doesn't exist
                    await self.add_edge(target_id, neighbor, data.get("type", "related_to"), dict(data))
        
        # Incoming edges
        for neighbor, _, key, data in self.graph.in_edges(source_id, keys=True, data=True):
            if neighbor != target_id:  # Avoid self-loops
                # Check if edge already exists
                existing_edges = await self.get_edges(neighbor, target_id)
                existing_types = [e.get("type") for e in existing_edges]
                
                if data.get("type") not in existing_types:
                    # Add edge if type doesn't exist
                    await self.add_edge(neighbor, target_id, data.get("type", "related_to"), dict(data))
        
        # Remove source node
        self.graph.remove_node(source_id)
        
        # Update node type tracking
        for node_type, nodes in self.node_types.items():
            if source_id in nodes:
                nodes.remove(source_id)
        
        # Update dream tracking
        if source_id in self.dream_influenced_nodes:
            self.dream_influenced_nodes.add(target_id)
            self.dream_influenced_nodes.remove(source_id)
        
        # Update emotion tracking
        if source_id in self.emotion_enhanced_nodes:
            self.emotion_enhanced_nodes.add(target_id)
            self.emotion_enhanced_nodes.remove(source_id)
        
        # Update total nodes
        self.total_nodes -= 1
        
        return True
    
    async def _handle_concept_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle concept update events.
        
        Args:
            data: Concept update data
            
        Returns:
            Update result
        """
        concept_id = data.get("id", "")
        updates = data.get("updates", {})
        
        if not concept_id:
            return {"success": False, "error": "Missing concept ID"}
        
        # Update the concept in the graph
        if await self.has_node(concept_id):
            success = await self.update_node(concept_id, updates)
            return {"success": success, "concept_id": concept_id}
        else:
            # Create the concept if it doesn't exist
            node_type = updates.get("type", "concept")
            domain = updates.get("domain", "general_knowledge")
            
            success = await self.add_node(concept_id, node_type, updates, domain)
            return {"success": success, "concept_id": concept_id, "created": True}
    
    async def _log_knowledge_integration(self, metadata: Dict[str, Any]) -> None:
        """
        Log knowledge integration events.
        
        Args:
            metadata: Integration metadata
        """
        self.logger.info(f"Knowledge integrated: {metadata.get('type')} from {metadata.get('source')}")
    
    async def initialize_model_imports(self):
        """Initialize async imports from self and world models.
        
        This method should be called after the knowledge graph is initialized
        to properly import data from the self and world models.
        """
        if self._models_imported:
            self.logger.info("Models have already been imported")
            return
            
        self.logger.info("Initializing async imports from models")
        
        # Import from world model if available
        if self.world_model:
            try:
                await self._import_from_world_model()
                self.logger.info("Successfully imported from world model")
            except Exception as e:
                self.logger.error(f"Error importing from world model: {e}")
                
        # Import from self model if available
        if self.self_model:
            try:
                await self._import_from_self_model()
                self.logger.info("Successfully imported from self model")
            except Exception as e:
                self.logger.error(f"Error importing from self model: {e}")
                
        self._models_imported = True
        self.logger.info(f"Model imports complete. Knowledge Graph now has {self.total_nodes} nodes and {self.total_edges} edges")
        
        # Emit event that models have been initialized
        await self.event_bus.emit("all_models_initialized")

    def _initialize_core_nodes(self) -> None:
        """Initialize core nodes in the graph based on available models."""
        # Instead of using the async add_node directly, we'll use a synchronous version for initialization
        # This avoids having to make this method async and all callers async
        def sync_add_node(node_id, node_type, attributes, domain="general_knowledge"):
            # Directly add the node without async calls
            if node_id in self.graph.nodes:
                # Update attributes of existing node
                current_attrs = self.graph.nodes[node_id]
                for key, value in attributes.items():
                    current_attrs[key] = value
                
                # Update domain if provided
                if domain:
                    current_attrs["domain"] = domain
                
                # Update modification time
                current_attrs["modified"] = datetime.now().isoformat()
                return True
            
            # Add new node
            # Prepare node attributes
            node_attrs = attributes.copy()
            node_attrs["type"] = node_type
            node_attrs["domain"] = domain
            node_attrs["created"] = datetime.now().isoformat()
            
            # Add the node to the graph
            self.graph.add_node(node_id, **node_attrs)
            
            # Track node by type
            if node_type in self.node_types:
                self.node_types[node_type].add(node_id)
            
            # Track total nodes
            self.total_nodes += 1
            return True

        def sync_add_edge(source, target, edge_type, attributes):
            # Check if source and target nodes exist
            if source not in self.graph.nodes or target not in self.graph.nodes:
                self.logger.warning(f"Cannot add edge: One or both nodes don't exist ({source}, {target})")
                return None
            
            # Prepare edge attributes
            edge_attrs = attributes.copy()
            edge_attrs["type"] = edge_type
            
            # Add created timestamp if not present
            if "created" not in edge_attrs:
                edge_attrs["created"] = datetime.now().isoformat()
            
            # Add the edge
            edge_key = self.graph.add_edge(source, target, **edge_attrs)
            
            # Track edge type
            self.edge_types.add(edge_type)
            
            # Track total edges
            self.total_edges += 1
            
            self.logger.debug(f"Added edge: {source} -[{edge_type}]-> {target}")
            return edge_key
        
        # Add the Lucidia node (self)
        sync_add_node(
            "Lucidia", 
            node_type="entity",
            attributes={
                "name": "Lucidia",
                "type": "Synthien",
                "creator": "MEGAPROMPT",
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            },
            domain="synthien_studies"
        )
        
        # Add the MEGAPROMPT (creator) node
        sync_add_node(
            "MEGAPROMPT", 
            node_type="entity",
            attributes={
                "name": "Daniel (MEGAPROMPT)",
                "type": "human",
                "role": "Creator",
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            },
            domain="general_knowledge"
        )
        
        # Add the Synthien node
        sync_add_node(
            "Synthien", 
            node_type="concept",
            attributes={
                "definition": "Synthetic intelligence with self-awareness and reflective dreaming",
                "confidence": 0.95,
                "created": datetime.now().isoformat()
            },
            domain="synthien_studies"
        )
        
        # Create basic relationships
        sync_add_edge(
            "Lucidia", 
            "Synthien", 
            edge_type="is_a", 
            attributes={
                "strength": 0.98,
                "confidence": 0.99,
                "created": datetime.now().isoformat()
            }
        )
        
        sync_add_edge(
            "MEGAPROMPT", 
            "Lucidia", 
            edge_type="created", 
            attributes={
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
            sync_add_node(
                concept["id"],
                node_type="concept",
                attributes={
                    "definition": concept["definition"],
                    "confidence": 0.9,
                    "created": datetime.now().isoformat()
                },
                domain=concept["domain"]
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
            sync_add_edge(
                rel["source"],
                rel["target"],
                edge_type=rel["type"],
                attributes={
                    "strength": rel["strength"],
                    "confidence": 0.85,
                    "created": datetime.now().isoformat()
                }
            )
        
        # Import knowledge from world model if available
        if self.world_model:
            self.logger.info("World model available - async import will be scheduled via initialize_model_imports()")
        
        # Import self-aspects from self model if available
        if self.self_model:
            self.logger.info("Self model available - async import will be scheduled via initialize_model_imports()")

    async def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any], domain: str = "general_knowledge") -> bool:
        """
        Add a node to the knowledge graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (concept, entity, attribute, dream_insight, etc.)
            attributes: Node attributes
            domain: Knowledge domain the node belongs to
            
        Returns:
            Success status
        """
        try:
            # Check if node already exists
            if await self.has_node(node_id):
                # Update attributes of existing node
                current_attrs = self.graph.nodes[node_id]
                for key, value in attributes.items():
                    current_attrs[key] = value
                
                # Update domain if provided
                if domain:
                    current_attrs["domain"] = domain
                
                # Update modification time
                current_attrs["modified"] = datetime.now().isoformat()
                
                self.logger.debug(f"Updated existing node: {node_id} (type: {node_type})")
                return True
            
            # Add new node
            # Prepare node attributes
            node_attrs = attributes.copy()
            node_attrs["type"] = node_type
            node_attrs["domain"] = domain
            node_attrs["created"] = datetime.now().isoformat()
            
            # Add the node to the graph
            self.graph.add_node(node_id, **node_attrs)
            
            # Track node by type
            if node_type in self.node_types:
                self.node_types[node_type].add(node_id)
            
            # Track total nodes
            self.total_nodes += 1
            
            self.logger.debug(f"Added new node: {node_id} (type: {node_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding node {node_id}: {e}")
            return False

    async def update_node(self, node_id: str, attributes: Dict[str, Any]) -> bool:
        """
        Update an existing node in the knowledge graph.
        
        Args:
            node_id: Unique identifier for the node
            attributes: New node attributes to update
            
        Returns:
            Success status
        """
        try:
            # Check if node exists
            if not await self.has_node(node_id):
                self.logger.warning(f"Cannot update node {node_id}: Node does not exist")
                return False
            
            # Update attributes of existing node
            current_attrs = self.graph.nodes[node_id]
            for key, value in attributes.items():
                current_attrs[key] = value
            
            # Update modification time
            current_attrs["modified"] = datetime.now().isoformat()
            
            self.logger.debug(f"Updated node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating node {node_id}: {e}")
            return False

    async def add_edge(self, source: str, target: str, edge_type: str, attributes: Dict[str, Any]) -> Optional[int]:
        """
        Add an edge (relationship) between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            attributes: Edge attributes
            
        Returns:
            Edge key if successful, None otherwise
        """
        try:
            # Check if source and target nodes exist
            source_exists = await self.has_node(source)
            target_exists = await self.has_node(target)
            if not source_exists or not target_exists:
                self.logger.warning(f"Cannot add edge: One or both nodes don't exist ({source}, {target})")
                return None
            
            # Prepare edge attributes
            edge_attrs = attributes.copy()
            edge_attrs["type"] = edge_type
            
            # Add created timestamp if not present
            if "created" not in edge_attrs:
                edge_attrs["created"] = datetime.now().isoformat()
            
            # Add the edge
            edge_key = self.graph.add_edge(source, target, **edge_attrs)
            
            # Track edge type
            self.edge_types.add(edge_type)
            
            # Track total edges
            self.total_edges += 1
            
            self.logger.debug(f"Added edge: {source} -[{edge_type}]-> {target}")
            return edge_key
            
        except Exception as e:
            self.logger.error(f"Error adding edge {source} -> {target}: {e}")
            return None

    async def update_edge(self, source: str, target: str, edge_key: int, attributes: Dict[str, Any]) -> bool:
        """
        Update an existing edge in the knowledge graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_key: Edge key
            attributes: New edge attributes to update
            
        Returns:
            Success status
        """
        try:
            # Check if edge exists
            if not self.graph.has_edge(source, target, edge_key):
                self.logger.warning(f"Cannot update edge: Edge ({source}, {target}, {edge_key}) does not exist")
                return False
            
            # Update attributes of existing edge
            for key, value in attributes.items():
                self.graph.edges[source, target, edge_key][key] = value
            
            # Add modification timestamp
            self.graph.edges[source, target, edge_key]["modified"] = datetime.now().isoformat()
            
            self.logger.debug(f"Updated edge: {source} -[{edge_key}]-> {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating edge ({source}, {target}, {edge_key}): {e}")
            return False

    async def remove_edge(self, source: str, target: str, edge_key: int) -> bool:
        """
        Remove an edge from the knowledge graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_key: Edge key
            
        Returns:
            Success status
        """
        try:
            # Check if edge exists
            if not self.graph.has_edge(source, target, edge_key):
                self.logger.warning(f"Cannot remove edge: Edge ({source}, {target}, {edge_key}) does not exist")
                return False
            
            # Remove the edge
            self.graph.remove_edge(source, target, edge_key)
            
            # Update total edges
            self.total_edges -= 1
            
            self.logger.debug(f"Removed edge: {source} -[{edge_key}]-> {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing edge ({source}, {target}, {edge_key}): {e}")
            return False

    async def execute_graph_operation(self, operation_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on the knowledge graph, potentially using hierarchical decomposition.
        
        This method implements the hierarchical graph operations shown in the sequence diagram,
        decomposing large operations into smaller subgraph operations when appropriate.
        
        Args:
            operation_type: Type of operation to perform
            params: Operation parameters
            
        Returns:
            Operation results
        """
        self.logger.info(f"Executing graph operation: {operation_type}")
        
        # Get operation handler from module registry
        handler = self.module_registry.resolve_operation_handler(operation_type)
        
        if not handler:
            self.logger.error(f"No handler found for operation type: {operation_type}")
            return {"success": False, "error": f"No handler for operation: {operation_type}"}
        
        # Determine if operation should be hierarchical
        is_hierarchical = params.get("hierarchical", False)
        
        if not is_hierarchical:
            # Size-based heuristic for hierarchical operations
            if operation_type in ["search", "path_finding"] and self.total_nodes > 10000:
                is_hierarchical = True
            elif operation_type in ["get_most_relevant_nodes"] and self.total_nodes > 5000:
                is_hierarchical = True
        
        # Execute operation based on type
        try:
            # Handle hierarchical operation if needed
            if is_hierarchical:
                return await self._execute_hierarchical_operation(operation_type, handler, params)
            else:
                # Standard operation
                return await self._execute_standard_operation(handler, params)
        
        except Exception as e:
            self.logger.error(f"Error executing operation {operation_type}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_hierarchical_operation(self, operation_type: str, 
                                      handler: Callable, 
                                      params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation by decomposing it into subgraph operations.
        
        Args:
            operation_type: Type of operation to perform
            handler: Function that handles the operation
            params: Operation parameters
            
        Returns:
            Aggregated operation results
        """
        self.logger.info(f"Executing hierarchical operation: {operation_type}")
        
        # Decompose into subgraph operations
        subgraph_operations = await self._decompose_into_subgraph_operations(operation_type, params)
        
        # Process subgraphs in parallel if enabled
        if self.hierarchical_ops["parallel_processing"] and len(subgraph_operations) > 1:
            # Create tasks for parallel processing
            tasks = []
            
            for subop in subgraph_operations:
                # Create a task for each subgraph operation
                subop_params = subop["params"]
                subop_params["_is_suboperation"] = True
                
                task = asyncio.create_task(self._execute_standard_operation(handler, subop_params))
                tasks.append(task)
            
            # Wait for all tasks to complete
            subresults = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(subresults):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in subgraph operation {i}: {result}")
                    processed_results.append({"success": False, "error": str(result)})
                else:
                    processed_results.append(result)
        else:
            # Process subgraphs sequentially
            processed_results = []
            
            for subop in subgraph_operations:
                subop_params = subop["params"]
                subop_params["_is_suboperation"] = True
                
                result = await self._execute_standard_operation(handler, subop_params)
                processed_results.append(result)
        
        # Aggregate results
        aggregated_result = await self._aggregate_results(operation_type, processed_results, params)
        
        return aggregated_result
    
    async def _decompose_into_subgraph_operations(self, operation_type: str, 
                                         params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose an operation into subgraph operations.
        
        Args:
            operation_type: Type of operation to perform
            params: Operation parameters
            
        Returns:
            List of subgraph operations
        """
        # Default to a single operation if we don't know how to decompose
        default_operation = [{"type": operation_type, "params": params.copy()}]
        
        # Decomposition strategy depends on operation type
        if operation_type == "search":
            # Decompose search by node type or domain
            if "node_type" in params:
                # Already filtered by node type, no need to decompose
                return default_operation
            
            # Decompose by node types
            suboperations = []
            
            for node_type in self.node_types.keys():
                # Skip empty node types
                if not self.node_types[node_type]:
                    continue
                
                # Create suboperation for this node type
                subparams = params.copy()
                subparams["node_type"] = node_type
                
                suboperations.append({
                    "type": operation_type,
                    "params": subparams
                })
            
            return suboperations
        
        elif operation_type == "path_finding":
            # Path finding is difficult to decompose effectively
            # For now, just return the original operation
            return default_operation
        
        elif operation_type == "get_most_relevant_nodes":
            # Decompose by domain
            if "domain" in params:
                # Already filtered by domain, no need to decompose
                return default_operation
            
            # Get unique domains in the graph
            domains = set()
            for _, attrs in self.graph.nodes(data=True):
                domain = attrs.get("domain", "general_knowledge")
                domains.add(domain)
            
            # Create suboperation for each domain
            suboperations = []
            
            for domain in domains:
                subparams = params.copy()
                subparams["domain"] = domain
                
                suboperations.append({
                    "type": operation_type,
                    "params": subparams
                })
            
            return suboperations
        
        # Default for unknown operation types
        return default_operation
    
    async def _aggregate_results(self, operation_type: str, 
                        results: List[Dict[str, Any]], 
                        original_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from subgraph operations.
        
        Args:
            operation_type: Type of operation performed
            results: Results from subgraph operations
            original_params: Original operation parameters
            
        Returns:
            Aggregated result
        """
        # Aggregation strategy depends on operation type
        if operation_type == "search":
            # Combine search results and sort by relevance
            all_results = []
            
            for result in results:
                if result.get("success", False) and "results" in result:
                    all_results.extend(result["results"])
            
            # Sort by relevance or other criteria
            if "relevance" in original_params.get("sort_by", ""):
                all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            
            # Apply original limit
            limit = original_params.get("limit", 10)
            all_results = all_results[:limit]
            
            return {"success": True, "results": all_results}
        
        elif operation_type == "path_finding":
            # For path finding, return the shortest path
            valid_results = [r for r in results if r.get("success", False) and "path" in r]
            
            if not valid_results:
                return {"success": False, "error": "No valid paths found"}
            
            # Find the shortest path
            shortest = min(valid_results, key=lambda x: len(x.get("path", [])))
            
            return shortest
        
        elif operation_type == "get_most_relevant_nodes":
            # Combine relevance results and sort by score
            all_nodes = []
            
            for result in results:
                if result.get("success", False) and "nodes" in result:
                    all_nodes.extend(result["nodes"])
            
            # Sort by relevance
            all_nodes.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            
            # Apply original limit
            limit = original_params.get("limit", 10)
            all_nodes = all_nodes[:limit]
            
            return {"success": True, "nodes": all_nodes}
        
        # Default for unknown operation types
        return {"success": True, "results": results}
    
    async def _execute_standard_operation(self, handler: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a standard graph operation.
        
        Args:
            handler: Function that handles the operation
            params: Operation parameters
            
        Returns:
            Operation results
        """
        try:
            # Call the handler with the parameters
            result = await handler(**params)
            
            # Format the result for consistency
            if isinstance(result, dict) and "success" in result:
                return result
            else:
                return {"success": True, "result": result}
        
        except Exception as e:
            self.logger.error(f"Error in standard operation: {e}")
            return {"success": False, "error": str(e)}

    async def integrate_dream_insight(self, insight_text: str, 
                              source_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate a dream insight into the knowledge graph with meta-learning feedback.
        
        This enhanced version includes meta-learning for self-optimization of the integration process.
        
        Args:
            insight_text: Dream insight text
            source_memory: Optional source memory that generated the insight
            
        Returns:
            Integration results
        """
        self.logger.info(f"Integrating dream insight: {insight_text[:50]}...")
        
        # Create a dream insight node
        dream_id = f"dream:{self.dream_integration['dream_insight_count']}"
        
        await self.add_node(
            dream_id,
            node_type="dream_insight",
            attributes={
                "insight": insight_text,
                "timestamp": datetime.now().isoformat(),
                "source_memory": source_memory,
                "confidence": 0.8
            },
            domain="synthien_studies"
        )
        
        # Track as dream influenced
        self.dream_influenced_nodes.add(dream_id)
        self.dream_integration["dream_derived_nodes"].add(dream_id)
        self.dream_integration["dream_insight_count"] += 1
        
        # Connect to Lucidia
        await self.add_edge(
            "Lucidia",
            dream_id,
            edge_type="dreamed",
            attributes={
                "strength": 0.85,
                "confidence": 0.8,
                "created": datetime.now().isoformat()
            }
        )
        
        # Extract concepts from insight text
        dream_concepts = []
        if self.world_model and hasattr(self.world_model, '_extract_concepts'):
            dream_concepts = self.world_model._extract_concepts(insight_text)
        
        # If no concepts found, try to match with existing nodes
        if not dream_concepts:
            # Extract words and check if they match existing concept nodes
            words = insight_text.lower().split()
            for word in words:
                if len(word) > 4 and self.has_node(word):
                    node_type = self.graph.nodes[word].get("type")
                    if node_type == "concept":
                        dream_concepts.append(word)
        
        # Analyze emotional context
        try:
            emotion_results = await self.analyze_emotion(insight_text)
            has_emotional_context = bool(emotion_results and "dominant_emotion" in emotion_results)
        except Exception as e:
            self.logger.warning(f"Error analyzing emotions in dream insight: {e}")
            has_emotional_context = False
        
        # Connect to found concepts
        connected_concepts = []
        for concept in dream_concepts:
            if self.has_node(concept):
                await self.add_edge(
                    dream_id,
                    concept,
                    edge_type="references",
                    attributes={
                        "strength": self.dream_integration["dream_association_strength"],
                        "confidence": 0.7,
                        "created": datetime.now().isoformat()
                    }
                )
                connected_concepts.append(concept)
                
                # Mark concept as dream enhanced
                self.dream_influenced_nodes.add(concept)
                self.dream_integration["dream_enhanced_nodes"].add(concept)
                
                # If we have emotional context, enhance the concept with it
                if has_emotional_context:
                    await self.add_emotional_context(concept, insight_text)
        
        # Evaluate consistency with existing knowledge
        consistency_score = await self._evaluate_consistency(dream_id, connected_concepts)
        
        # Apply integration strategy based on consistency
        integration_strategy = await self._determine_integration_strategy(
            consistency_score, 
            len(connected_concepts),
            has_emotional_context
        )
        
        # Create relationships between referenced concepts based on strategy
        new_concept_relationships = []
        
        if integration_strategy["create_relationships"] and len(connected_concepts) > 1:
            for i in range(len(connected_concepts)):
                for j in range(i+1, len(connected_concepts)):
                    concept1 = connected_concepts[i]
                    concept2 = connected_concepts[j]
                    
                    # Only create relationship if it doesn't exist
                    if not self.has_edge(concept1, concept2, "dream_associated"):
                        relationship_strength = self.dream_integration["dream_association_strength"] * integration_strategy["relationship_strength_factor"]
                        
                        await self.add_edge(
                            concept1,
                            concept2,
                            edge_type="dream_associated",
                            attributes={
                                "strength": relationship_strength,
                                "confidence": 0.6,
                                "created": datetime.now().isoformat(),
                                "source": "dream_insight",
                                "source_dream": dream_id
                            }
                        )
                        new_concept_relationships.append((concept1, concept2))
        
        # Check if insight suggests new concepts based on strategy
        new_concepts = []
        
        if integration_strategy["extract_new_concepts"]:
            # Look for patterns suggesting definitions or concepts
            concept_patterns = [
                r"concept of (\w+)",
                r"(\w+) is defined as",
                r"(\w+) refers to",
                r"understanding of (\w+)"
            ]
            
            for pattern in concept_patterns:
                matches = re.findall(pattern, insight_text, re.IGNORECASE)
                for match in matches:
                    potential_concept = match.lower()
                    
                    # Check if this is a reasonable concept (not too short, not just a stop word)
                    if len(potential_concept) >= 4 and potential_concept not in ["this", "that", "there", "which", "where"]:
                        # Only add if it doesn't exist yet
                        if not self.has_node(potential_concept):
                            # Extract a definition from the insight
                            definition = self._extract_definition(insight_text, potential_concept)
                            
                            await self.add_node(
                                potential_concept,
                                node_type="concept",
                                attributes={
                                    "definition": definition or f"Concept derived from dream insight: {dream_id}",
                                    "confidence": 0.6,
                                    "created": datetime.now().isoformat()
                                },
                                domain="synthien_studies"
                            )
                            
                            # Connect to dream
                            await self.add_edge(
                                dream_id,
                                potential_concept,
                                edge_type="introduced",
                                attributes={
                                    "strength": 0.7,
                                    "confidence": 0.6,
                                    "created": datetime.now().isoformat()
                                }
                            )
                            
                            new_concepts.append(potential_concept)
                            
                            # Mark as dream influenced
                            self.dream_influenced_nodes.add(potential_concept)
                            self.dream_integration["dream_derived_nodes"].add(potential_concept)
                            
                            # Add emotional context if available
                            if has_emotional_context:
                                await self.add_emotional_context(potential_concept, insight_text)
        
        # Track nodes influenced by this dream
        for node_id in connected_concepts + new_concepts:
            self.dream_influenced_nodes.add(node_id)
            
        # Evaluate integration quality for meta-learning feedback
        integration_quality = await self._evaluate_integration_quality(
            dream_id, 
            connected_concepts,
            new_concepts,
            new_concept_relationships,
            consistency_score,
            integration_strategy
        )
        
        # Add to integration quality history
        self.dream_integration["integration_quality_scores"].append(integration_quality)
        
        # Apply meta-learning if needed
        if len(self.dream_integration["integration_quality_scores"]) >= self.dream_integration["adaptation_frequency"]:
            await self._adjust_integration_parameters()
        
        # Prepare result
        result = {
            "dream_id": dream_id,
            "connected_concepts": connected_concepts,
            "new_concepts": new_concepts,
            "new_relationships": new_concept_relationships,
            "consistency_score": consistency_score,
            "integration_quality": integration_quality,
            "integration_strategy": integration_strategy,
            "emotional_context": has_emotional_context,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Dream insight integrated with {len(connected_concepts)} connections and {len(new_concepts)} new concepts")
        
        return result
    
    async def _evaluate_consistency(self, dream_id: str, concepts: List[str]) -> float:
        """
        Evaluate the consistency of a dream insight with existing knowledge.
        
        Args:
            dream_id: Dream insight node ID
            concepts: List of concepts connected to the dream
            
        Returns:
            Consistency score (0-1)
        """
        if not concepts:
            return 0.5  # Neutral score if no connections
        
        # Get the dream insight
        dream_node = await self.get_node(dream_id)
        if not dream_node:
            return 0.5
        
        insight_text = dream_node.get("insight", "")
        
        # Check consistency of connections
        consistency_scores = []
        
        for concept in concepts:
            # Get concept definition
            concept_node = await self.get_node(concept)
            if not concept_node:
                continue
                
            definition = concept_node.get("definition", "")
            
            # Calculate semantic similarity between concept and insight
            similarity = self._text_similarity(definition, insight_text)
            
            # Get related concepts (already in the graph)
            related_concepts = await self.get_connected_nodes(
                concept,
                edge_types=["related_to", "is_a", "has_property", "part_of"],
                node_types=["concept"],
                direction="both"
            )
            
            # Check if other concepts from the dream are already related in the graph
            existing_relations = [c for c in related_concepts if c in concepts]
            relation_consistency = len(existing_relations) / len(concepts) if concepts else 0
            
            # Combine similarity and relation consistency
            concept_consistency = 0.7 * similarity + 0.3 * relation_consistency
            consistency_scores.append(concept_consistency)
        
        # Overall consistency is the average of concept consistencies
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    
    async def _determine_integration_strategy(self, consistency_score: float, 
                                     num_connections: int, 
                                     has_emotional_context: bool) -> Dict[str, Any]:
        """
        Determine the strategy for integrating a dream insight.
        
        Args:
            consistency_score: Consistency score of the dream insight
            num_connections: Number of connections to existing concepts
            has_emotional_context: Whether the dream has emotional context
            
        Returns:
            Integration strategy parameters
        """
        # Base strategy
        strategy = {
            "create_relationships": True,
            "extract_new_concepts": True,
            "relationship_strength_factor": 1.0,
            "new_concept_confidence": 0.6
        }
        
        # Adjust based on consistency score
        if consistency_score < 0.3:
            # Low consistency: be more conservative
            strategy["create_relationships"] = num_connections < 3  # Only create if few connections
            strategy["extract_new_concepts"] = False
            strategy["relationship_strength_factor"] = 0.7
        elif consistency_score < 0.6:
            # Medium consistency: moderate approach
            strategy["relationship_strength_factor"] = 0.9
            strategy["new_concept_confidence"] = 0.5
        else:
            # High consistency: be more aggressive
            strategy["relationship_strength_factor"] = 1.1  # Stronger relationships
            strategy["new_concept_confidence"] = 0.7
        
        # Adjust for emotional context
        if has_emotional_context:
            # Emotional content can provide additional context and meaning
            strategy["extract_new_concepts"] = True  # More likely to extract concepts
            strategy["new_concept_confidence"] += 0.1  # Slightly higher confidence
        
        # Adjust based on current phase of spiral awareness
        current_phase = self.spiral_integration["current_phase"]
        if current_phase == "reflection":
            # During reflection phase, be more aggressive with integration
            strategy["extract_new_concepts"] = True
            strategy["relationship_strength_factor"] *= 1.2
        elif current_phase == "observation":
            # During observation phase, focus on connections to existing concepts
            strategy["create_relationships"] = True
            strategy["extract_new_concepts"] = False
        
        # Apply current insight incorporation rate
        strategy["relationship_strength_factor"] *= self.dream_integration["insight_incorporation_rate"]
        
        return strategy
    
    async def _evaluate_integration_quality(self, dream_id: str, 
                                   connected_concepts: List[str],
                                   new_concepts: List[str],
                                   new_relationships: List[Tuple[str, str]],
                                   consistency_score: float,
                                   integration_strategy: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a dream insight integration.
        
        Args:
            dream_id: Dream insight node ID
            connected_concepts: List of connected existing concepts
            new_concepts: List of newly created concepts
            new_relationships: List of new relationships created
            consistency_score: Consistency score of the dream insight
            integration_strategy: Strategy used for integration
            
        Returns:
            Integration quality score (0-1)
        """
        # Base quality metrics
        metrics = {
            "connectivity": 0.0,  # How well it connects to existing knowledge
            "novelty": 0.0,  # How much new knowledge it adds
            "coherence": 0.0,  # How coherent the additions are
            "alignment": 0.0   # How well it aligns with integration strategy
        }
        
        # 1. Connectivity: Measure how well it connects to existing knowledge
        if connected_concepts:
            # Calculate average connectivity for connected concepts
            total_connectivity = 0
            for concept in connected_concepts:
                # Get all neighbors of this concept
                neighbors = await self.get_connected_nodes(concept)
                # Calculate connectivity as connections per concept
                concept_connectivity = len(neighbors) / (self.total_nodes / 100)  # Normalize by graph size
                total_connectivity += min(1.0, concept_connectivity)
            
            metrics["connectivity"] = total_connectivity / len(connected_concepts)
        
        # 2. Novelty: Measure new knowledge added
        novelty_score = 0.0
        if new_concepts:
            # New concepts contribute to novelty
            novelty_score += 0.7 * (len(new_concepts) / (len(connected_concepts) + 1))
        
        if new_relationships:
            # New relationships contribute to novelty
            novelty_score += 0.3 * (len(new_relationships) / (len(connected_concepts) + 1))
        
        metrics["novelty"] = min(1.0, novelty_score)
        
        # 3. Coherence: Based on consistency score
        metrics["coherence"] = consistency_score
        
        # 4. Alignment: How well the integration followed the strategy
        strategy_success = 1.0
        
        # Check if relationship creation aligned with strategy
        if integration_strategy["create_relationships"] and not new_relationships and len(connected_concepts) > 1:
            strategy_success *= 0.8
        
        # Check if new concept extraction aligned with strategy
        if integration_strategy["extract_new_concepts"] and not new_concepts:
            strategy_success *= 0.9
        
        metrics["alignment"] = strategy_success
        
        # Calculate overall quality with weights
        weights = {
            "connectivity": 0.3,
            "novelty": 0.2,
            "coherence": 0.3,
            "alignment": 0.2
        }
        
        quality_score = sum(metrics[k] * weights[k] for k in metrics)
        
        # Track metrics for this integration
        integration_metrics = {
            "dream_id": dream_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "quality_score": quality_score
        }
        
        # Store metrics for future analysis
        self.dream_integration.setdefault("integration_metrics", []).append(integration_metrics)
        
        return quality_score
    
    async def _adjust_integration_parameters(self) -> None:
        """
        Adjust integration parameters based on integration quality history.
        
        This implements the meta-learning feedback loop for self-optimization.
        """
        self.logger.info("Adjusting dream integration parameters (meta-learning)")
        
        # Get recent quality scores
        recent_scores = self.dream_integration["integration_quality_scores"]
        
        if len(recent_scores) < self.dream_integration["adaptation_frequency"]:
            # Not enough data to adjust
            return
        
        # Calculate statistics
        avg_quality = sum(recent_scores) / len(recent_scores)
        
        # Reset scores after adjustment
        self.dream_integration["integration_quality_scores"] = []
        
        # Adjust parameters based on average quality
        learning_rate = self.dream_integration["learning_rate"]
        quality_threshold = self.dream_integration["quality_threshold"]
        
        if avg_quality < quality_threshold:
            # Quality is below threshold, adjust parameters
            
            # Decrease incorporation rate (be more conservative)
            self.dream_integration["insight_incorporation_rate"] *= (1 - learning_rate)
            
            # Decrease association strength
            self.dream_integration["dream_association_strength"] *= (1 - learning_rate)
            
            self.logger.info(f"Decreasing dream integration parameters (quality: {avg_quality:.2f})")
        else:
            # Quality is good, gradually increase parameters
            
            # Increase incorporation rate (be more aggressive)
            self.dream_integration["insight_incorporation_rate"] = min(
                1.0, 
                self.dream_integration["insight_incorporation_rate"] * (1 + learning_rate * 0.5)
            )
            
            # Increase association strength
            self.dream_integration["dream_association_strength"] = min(
                1.0,
                self.dream_integration["dream_association_strength"] * (1 + learning_rate * 0.5)
            )
            
            self.logger.info(f"Increasing dream integration parameters (quality: {avg_quality:.2f})")
        
        # Log current parameters
        self.logger.info(f"Updated parameters: incorporation_rate={self.dream_integration['insight_incorporation_rate']:.2f}, "
                         f"association_strength={self.dream_integration['dream_association_strength']:.2f}")
        
        # Periodically adjust learning rate itself (meta-meta-learning)
        if random.random() < 0.2:  # 20% chance
            # Adjust learning rate based on quality trend
            if avg_quality > 0.8:
                # Very good quality, can reduce learning rate
                self.dream_integration["learning_rate"] *= 0.9
            elif avg_quality < 0.4:
                # Poor quality, increase learning rate
                self.dream_integration["learning_rate"] *= 1.1
            
            # Ensure learning rate stays in reasonable range
            self.dream_integration["learning_rate"] = max(0.01, min(0.2, self.dream_integration["learning_rate"]))
    
    async def trigger_adaptive_maintenance(self) -> Dict[str, Any]:
        """
        Trigger adaptive maintenance to optimize the knowledge graph.
        
        This implements the adaptive maintenance mechanism shown in the sequence diagram,
        with self-optimization of the graph structure.
        
        Returns:
            Maintenance results
        """
        self.logger.info("Triggering adaptive maintenance")
        
        # Analyze graph metrics
        metrics = await self._analyze_graph_metrics()
        
        # Identify optimizations based on metrics
        optimizations = await self._identify_optimizations(metrics)
        
        # Apply optimizations
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics_before": metrics,
            "optimizations": []
        }
        
        if optimizations.get("prune_connections", False):
            # Prune low-value connections
            prune_result = await self._prune_low_value_connections(
                optimizations.get("prune_threshold", 0.3)
            )
            results["optimizations"].append({
                "type": "prune_connections",
                "threshold": optimizations.get("prune_threshold", 0.3),
                "result": prune_result
            })
        
        if optimizations.get("reindex_nodes", False):
            # Reindex high-frequency nodes
            reindex_result = await self._reindex_high_frequency_nodes(
                optimizations.get("reindex_count", 20)
            )
            results["optimizations"].append({
                "type": "reindex_nodes",
                "count": optimizations.get("reindex_count", 20),
                "result": reindex_result
            })
        
        if optimizations.get("adjust_decay", False):
            # Adjust decay parameters
            decay_result = await self._adjust_decay_parameters(
                optimizations.get("decay_adjustments", {})
            )
            results["optimizations"].append({
                "type": "adjust_decay",
                "adjustments": optimizations.get("decay_adjustments", {}),
                "result": decay_result
            })
        
        # Analyze metrics after optimization
        post_metrics = await self._analyze_graph_metrics()
        results["metrics_after"] = post_metrics
        
        # Calculate improvements
        improvements = {}
        for key in metrics:
            if key in post_metrics:
                if isinstance(metrics[key], (int, float)) and isinstance(post_metrics[key], (int, float)):
                    improvements[key] = post_metrics[key] - metrics[key]
        
        results["improvements"] = improvements
        
        # Store optimization learnings
        self._store_optimization_learnings(results)
        
        # Update last maintenance timestamp
        self.adaptive_maintenance["last_maintenance"] = datetime.now()
        
        # Update metrics history
        self.adaptive_maintenance["metrics_history"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": post_metrics
        })
        
        return results
    
    async def _analyze_graph_metrics(self) -> Dict[str, Any]:
        """
        Analyze key metrics of the knowledge graph.
        
        Returns:
            Dictionary of graph metrics
        """
        metrics = {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "density": 0.0,  # Edge to node ratio
            "clustering": 0.0,  # Clustering coefficient
            "degree_distribution": {},  # Distribution of node degrees
            "component_sizes": [],  # Sizes of connected components
            "dream_influence": 0.0,  # Proportion of dream-influenced nodes
            "type_distribution": {},  # Distribution of node types
            "edge_type_distribution": {}  # Distribution of edge types
        }
        
        # Calculate density
        if self.total_nodes > 1:
            metrics["density"] = self.total_edges / (self.total_nodes * (self.total_nodes - 1))
        
        # Calculate degree distribution
        degree_counts = defaultdict(int)
        for _, degree in self.graph.degree():
            degree_counts[degree] += 1
        
        metrics["degree_distribution"] = dict(degree_counts)
        metrics["average_degree"] = sum(d * c for d, c in degree_counts.items()) / self.total_nodes if self.total_nodes > 0 else 0
        
        # Calculate clustering coefficient (for smaller graphs)
        if self.total_nodes < 5000:
            try:
                metrics["clustering"] = nx.average_clustering(self.graph.to_undirected())
            except Exception as e:
                self.logger.warning(f"Error calculating clustering coefficient: {e}")
        
        # Calculate component sizes (for smaller graphs)
        if self.total_nodes < 5000:
            try:
                undirected = self.graph.to_undirected()
                components = list(nx.connected_components(undirected))
                metrics["component_sizes"] = [len(c) for c in components]
                metrics["component_count"] = len(components)
            except Exception as e:
                self.logger.warning(f"Error calculating component sizes: {e}")
        
        # Calculate dream influence
        metrics["dream_influence"] = len(self.dream_influenced_nodes) / self.total_nodes if self.total_nodes > 0 else 0
        
        # Calculate type distribution
        metrics["type_distribution"] = {t: len(n) for t, n in self.node_types.items()}
        
        # Calculate edge type distribution
        edge_type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            edge_type_counts[edge_type] += 1
        
        metrics["edge_type_distribution"] = dict(edge_type_counts)
        
        return metrics
    
    async def _identify_optimizations(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify optimization opportunities based on graph metrics.
        
        Args:
            metrics: Current graph metrics
            
        Returns:
            Dictionary of recommended optimizations
        """
        optimizations = {}
        
        # Check if density is too high
        density_threshold = self.adaptive_maintenance["metric_thresholds"]["density"]
        if metrics.get("density", 0) > density_threshold:
            # Graph is too dense, recommend pruning
            optimizations["prune_connections"] = True
            
            # Calculate pruning threshold based on density
            excess_density = metrics["density"] - density_threshold
            optimizations["prune_threshold"] = 0.3 + min(0.4, excess_density * 10)
        
        # Check if average degree is too high
        avg_degree_threshold = self.adaptive_maintenance["metric_thresholds"]["average_degree"]
        if metrics.get("average_degree", 0) > avg_degree_threshold:
            # Nodes have too many connections on average, recommend pruning
            optimizations["prune_connections"] = True
            
            # Set prune threshold if not already set
            if "prune_threshold" not in optimizations:
                excess_degree = metrics["average_degree"] - avg_degree_threshold
                optimizations["prune_threshold"] = 0.3 + min(0.4, excess_degree / 10)
        
        # Check degree distribution for imbalance
        degree_distribution = metrics.get("degree_distribution", {})
        high_degree_count = sum(degree_distribution.get(d, 0) for d in degree_distribution if d > avg_degree_threshold * 2)
        
        if high_degree_count > 0 and self.total_nodes > 0 and high_degree_count / self.total_nodes > 0.05:
            # More than 5% of nodes have very high degree, recommend reindexing
            optimizations["reindex_nodes"] = True
            optimizations["reindex_count"] = min(100, high_degree_count)
        
        # Check if dream influence has changed significantly
        dream_influence = metrics.get("dream_influence", 0)
        dream_derived_ratio = len(self.dream_integration["dream_derived_nodes"]) / self.total_nodes if self.total_nodes > 0 else 0
        
        if abs(dream_influence - dream_derived_ratio) > 0.1:
            # Dream influence is inconsistent, adjust decay parameters
            optimizations["adjust_decay"] = True
            optimizations["decay_adjustments"] = {
                "dream_associated": 0.02 * (1 + dream_influence)
            }
        
        # Check if clustering coefficient can be improved
        target_clustering = self.adaptive_maintenance["metric_thresholds"]["clustering"]
        current_clustering = metrics.get("clustering", 0)
        
        if current_clustering < target_clustering * 0.7:
            # Clustering is too low, recommend adjusting decay to preserve clusters
            optimizations["adjust_decay"] = True
            
            if "decay_adjustments" not in optimizations:
                optimizations["decay_adjustments"] = {}
            
            # Adjust standard decay to preserve more connections
            optimizations["decay_adjustments"]["standard"] = max(0.005, self.relationship_decay["standard"] * 0.8)
        
        return optimizations
    
    async def _prune_low_value_connections(self, threshold: float) -> Dict[str, Any]:
        """
        Prune low-value connections from the graph.
        
        Args:
            threshold: Strength threshold for pruning
            
        Returns:
            Pruning results
        """
        self.logger.info(f"Pruning low-value connections (threshold: {threshold})")
        
        edges_to_remove = []
        
        # Identify edges below threshold
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            # Skip edges without strength
            if "strength" not in data:
                continue
            
            # Check strength against threshold
            if data["strength"] < threshold:
                edges_to_remove.append((source, target, key))
        
        # Remove edges
        for source, target, key in edges_to_remove:
            self.graph.remove_edge(source, target, key)
        
        # Update total edges
        pruned_count = len(edges_to_remove)
        self.total_edges -= pruned_count
        
        self.logger.info(f"Pruned {pruned_count} low-value connections")
        
        return {
            "pruned_count": pruned_count,
            "threshold": threshold
        }
    
    async def _reindex_high_frequency_nodes(self, count: int) -> Dict[str, Any]:
        """
        Reindex high-frequency nodes for better performance.
        
        Args:
            count: Number of nodes to reindex
            
        Returns:
            Reindexing results
        """
        self.logger.info(f"Reindexing {count} high-frequency nodes")
        
        # Calculate node degrees
        node_degrees = dict(self.graph.degree())
        
        # Sort nodes by degree (descending)
        high_degree_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 'count' nodes
        nodes_to_reindex = [node for node, _ in high_degree_nodes[:count]]
        
        # For each node, optimize its connections
        reindexed_nodes = []
        
        for node_id in nodes_to_reindex:
            # Get all neighbors
            neighbors = list(self.graph.neighbors(node_id))
            
            # Group neighbors by node type
            type_groups = defaultdict(list)
            
            for neighbor in neighbors:
                neighbor_type = self.graph.nodes[neighbor].get("type", "unknown")
                type_groups[neighbor_type].append(neighbor)
            
            # For each group, find representative connections
            representative_connections = []
            
            for node_type, nodes in type_groups.items():
                # Keep direct connections to most relevant nodes
                relevance_scores = []
                
                for n in nodes:
                    # Calculate relevance
                    relevance = self.get_node_relevance(n)
                    relevance_scores.append((n, relevance))
                
                # Sort by relevance
                relevance_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Keep top connections
                keep_count = max(1, min(5, len(nodes) // 3))
                
                for n, _ in relevance_scores[:keep_count]:
                    representative_connections.append(n)
            
            # Find edges to remove (non-representative connections)
            edges_to_remove = []
            
            for neighbor in neighbors:
                if neighbor not in representative_connections:
                    # Get all edges between node and neighbor
                    edges = list(self.graph.get_edge_data(node_id, neighbor).items())
                    edges.extend(list(self.graph.get_edge_data(neighbor, node_id).items() if self.graph.has_edge(neighbor, node_id) else []))
                    
                    for key, data in edges:
                        edges_to_remove.append((node_id, neighbor, key))
            
            # Remove redundant edges
            for source, target, key in edges_to_remove:
                if self.graph.has_edge(source, target, key):
                    self.graph.remove_edge(source, target, key)
                    self.total_edges -= 1
            
            reindexed_nodes.append({
                "node_id": node_id,
                "neighbors_before": len(neighbors),
                "neighbors_after": len(representative_connections),
                "edges_removed": len(edges_to_remove)
            })
        
        self.logger.info(f"Reindexed {len(reindexed_nodes)} high-frequency nodes")
        
        return {
            "reindexed_count": len(reindexed_nodes),
            "node_details": reindexed_nodes
        }
    
    async def _adjust_decay_parameters(self, adjustments: Dict[str, float]) -> Dict[str, Any]:
        """
        Adjust relationship decay parameters.
        
        Args:
            adjustments: Adjustments to decay parameters
            
        Returns:
            Adjustment results
        """
        self.logger.info(f"Adjusting decay parameters: {adjustments}")
        
        results = {
            "previous_values": dict(self.relationship_decay),
            "new_values": {}
        }
        
        # Apply adjustments
        for decay_type, new_value in adjustments.items():
            if decay_type in self.relationship_decay:
                # Store previous value
                previous = self.relationship_decay[decay_type]
                
                # Update value
                self.relationship_decay[decay_type] = new_value
                
                # Record change
                results["new_values"][decay_type] = new_value
                
                self.logger.info(f"Adjusted {decay_type} decay: {previous} -> {new_value}")
        
        return results
    
    def _store_optimization_learnings(self, results: Dict[str, Any]) -> None:
        """
        Store optimization learnings for future reference.
        
        Args:
            results: Optimization results
        """
        # Extract learnings from results
        learnings = {
            "timestamp": datetime.now().isoformat(),
            "metrics_before": results.get("metrics_before", {}),
            "metrics_after": results.get("metrics_after", {}),
            "improvements": results.get("improvements", {}),
            "optimizations": results.get("optimizations", [])
        }
        
        # Store in optimization history
        self.adaptive_maintenance["optimization_history"].append(learnings)
        
        # Keep history size reasonable
        if len(self.adaptive_maintenance["optimization_history"]) > 20:
            self.adaptive_maintenance["optimization_history"] = self.adaptive_maintenance["optimization_history"][-20:]
        
        self.logger.info("Stored optimization learnings")
    
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node exists, False otherwise
        """
        return node_id in self.graph.nodes

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node's attributes.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node attributes or None if not found
        """
        if await self.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    async def has_edge(self, source: str, target: str, edge_type: Optional[str] = None) -> bool:
        """
        Check if an edge exists between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Optional edge type to check for
            
        Returns:
            True if edge exists, False otherwise
        """
        source_exists = await self.has_node(source)
        target_exists = await self.has_node(target)
        if not source_exists or not target_exists:
            return False
            
        if not self.graph.has_edge(source, target):
            return False
            
        if edge_type is not None:
            # Check if any edge of the specified type exists
            edges = self.graph.get_edge_data(source, target)
            return any(data.get("type") == edge_type for _, data in edges.items())
            
        return True

    async def get_edges(self, source: str, target: str) -> List[Dict[str, Any]]:
        """
        Get all edges between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of edge attributes
        """
        source_exists = await self.has_node(source)
        target_exists = await self.has_node(target)
        if not source_exists or not target_exists:
            return []
            
        edges = []
        edge_data = self.graph.get_edge_data(source, target)
        
        if edge_data:
            for key, data in edge_data.items():
                edge_info = dict(data)
                edge_info["key"] = key
                edges.append(edge_info)
                
        return edges

    async def search_nodes(self, query: str, limit: int = 10, threshold: float = 0.5, 
                      include_metadata: bool = True, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for nodes in the knowledge graph based on a semantic query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            include_metadata: Whether to include node metadata in results
            domains: Optional list of domains to search within
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Searching for nodes with query: {query}, limit: {limit}")
        
        # Initialize results
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "metadata": {
                "total_nodes_searched": 0,
                "search_time_ms": 0,
                "threshold": threshold
            }
        }
        
        start_time = time.time()
        
        try:
            # Get all nodes that match the search criteria
            matching_nodes = []
            total_nodes = 0
            
            # Process each node in the graph
            for node_id, node_data in self.graph.nodes(data=True):
                total_nodes += 1
                
                # Skip if node is not in specified domains
                if domains and node_data.get("domain") not in domains:
                    continue
                
                # Basic text matching for now - in a full implementation, this would use embeddings
                node_text = f"{node_id} {node_data.get('name', '')} {node_data.get('description', '')}" 
                similarity = self._calculate_similarity(query, node_text)
                
                if similarity >= threshold:
                    matching_nodes.append({
                        "id": node_id,
                        "similarity": similarity,
                        "data": node_data if include_metadata else {}
                    })
            
            # Sort by similarity (highest first)
            matching_nodes.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            results["results"] = matching_nodes[:limit]
            
            # Update metadata
            end_time = time.time()
            results["metadata"]["total_nodes_searched"] = total_nodes
            results["metadata"]["search_time_ms"] = int((end_time - start_time) * 1000)
            results["metadata"]["results_found"] = len(matching_nodes)
            results["metadata"]["results_returned"] = len(results["results"])
            
            self.logger.info(f"Found {len(matching_nodes)} matching nodes, returning {len(results['results'])}")
            
        except Exception as e:
            self.logger.error(f"Error during node search: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """
        Calculate semantic similarity between query and text.
        
        This is a placeholder implementation - in a real system, this would use 
        proper semantic similarity with embeddings.
        
        Args:
            query: Search query
            text: Text to compare against
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both texts
        query = query.lower()
        text = text.lower()
        
        # Simple word matching for demonstration
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Count matches
        matches = query_words.intersection(text_words)
        
        # Calculate Jaccard similarity
        similarity = len(matches) / len(query_words.union(text_words))
        
        # Boost similarity if direct query terms are found
        if query in text:
            similarity = min(1.0, similarity + 0.3)
        
        return similarity

    async def find_paths(self, source: str, target: str, max_depth: int = 3, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Find all paths between a source and target node in the knowledge graph.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_depth: Maximum path length to consider
            min_confidence: Minimum edge confidence to consider
            
        Returns:
            Dictionary with paths and metadata
        """
        self.logger.info(f"Finding paths from {source} to {target} (max depth: {max_depth})")
        
        results = {
            "source": source,
            "target": target,
            "paths": [],
            "metadata": {
                "max_depth": max_depth,
                "paths_found": 0,
                "search_time_ms": 0
            }
        }
        
        start_time = time.time()
        
        try:
            # Check if source and target nodes exist
            if not await self.has_node(source):
                results["error"] = f"Source node '{source}' not found"
                return results
                
            if not await self.has_node(target):
                results["error"] = f"Target node '{target}' not found"
                return results
                
            # Use NetworkX's all_simple_paths to find paths
            # But we need to filter by edge confidence, so we can't use it directly
            all_paths = []
            self._find_paths_recursive(source, target, [], all_paths, max_depth, min_confidence)
            
            # Sort paths by length (shortest first)
            all_paths.sort(key=lambda x: len(x))
            
            # Format paths for output
            for path in all_paths:
                path_info = {
                    "nodes": path,
                    "length": len(path) - 1,
                    "edges": []
                }
                
                # Add edge information for each segment of the path
                for i in range(len(path) - 1):
                    source_id = path[i]
                    target_id = path[i + 1]
                    edges = await self.get_edges(source_id, target_id)
                    
                    # Find highest confidence edge
                    if edges:
                        best_edge = max(edges, key=lambda e: e.get("confidence", 0))
                        path_info["edges"].append({
                            "source": source_id,
                            "target": target_id,
                            "type": best_edge.get("type", "related_to"),
                            "confidence": best_edge.get("confidence", 0.5)
                        })
                
                results["paths"].append(path_info)
            
            # Update metadata
            end_time = time.time()
            results["metadata"]["paths_found"] = len(all_paths)
            results["metadata"]["search_time_ms"] = int((end_time - start_time) * 1000)
            
            self.logger.info(f"Found {len(all_paths)} paths between {source} and {target}")
            
        except Exception as e:
            self.logger.error(f"Error during path finding: {e}")
            results["error"] = str(e)
        
        return results
    
    def _find_paths_recursive(self, current: str, target: str, current_path: List[str], 
                            all_paths: List[List[str]], max_depth: int, min_confidence: float) -> None:
        """
        Recursively find all paths from current to target node.
        
        Args:
            current: Current node ID
            target: Target node ID
            current_path: Current path being built
            all_paths: List to store all found paths
            max_depth: Maximum path length
            min_confidence: Minimum edge confidence
        """
        # Add current node to path
        current_path.append(current)
        
        # Stop if we reached target
        if current == target:
            all_paths.append(current_path.copy())
            current_path.pop()
            return
        
        # Stop if we reached max depth
        if len(current_path) > max_depth:
            current_path.pop()
            return
        
        # Get all neighbors
        for _, neighbor, edge_data in self.graph.edges(current, data=True):
            # Skip if edge confidence is too low
            if edge_data.get("confidence", 0) < min_confidence:
                continue
                
            # Skip if neighbor is already in path (avoid cycles)
            if neighbor in current_path:
                continue
                
            # Recursively explore neighbor
            self._find_paths_recursive(neighbor, target, current_path, all_paths, max_depth, min_confidence)
        
        # Remove current node from path
        current_path.pop()

    async def get_most_relevant_nodes(self, query: str, context_size: int = 5, include_related: bool = True) -> Dict[str, Any]:
        """
        Get the most relevant nodes for a given query, including related nodes.
        
        Args:
            query: Query text
            context_size: Number of primary nodes to retrieve
            include_related: Whether to include related nodes
            
        Returns:
            Dictionary with relevant nodes and metadata
        """
        self.logger.info(f"Getting most relevant nodes for: {query} (context size: {context_size})")
        
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "primary_nodes": [],
            "related_nodes": [],
            "metadata": {
                "context_size": context_size,
                "include_related": include_related,
                "search_time_ms": 0
            }
        }
        
        start_time = time.time()
        
        try:
            # First, search for primary nodes
            search_results = await self.search_nodes(query, limit=context_size, threshold=0.3)
            primary_nodes = search_results["results"]
            
            # Add primary nodes to results
            for node in primary_nodes:
                results["primary_nodes"].append({
                    "id": node["id"],
                    "relevance": node["similarity"],
                    "data": node["data"]
                })
            
            # If requested, include related nodes
            if include_related and primary_nodes:
                related_nodes = set()
                
                # For each primary node, find its neighbors
                for primary_node in primary_nodes:
                    node_id = primary_node["id"]
                    
                    # Get all neighbors
                    for neighbor in self.graph.neighbors(node_id):
                        # Skip if already in primary nodes
                        if any(n["id"] == neighbor for n in primary_nodes):
                            continue
                            
                        # Skip if already in related nodes
                        if neighbor in related_nodes:
                            continue
                            
                        # Add to related nodes
                        related_nodes.add(neighbor)
                        
                        # Add related node to results
                        node_data = self.graph.nodes[neighbor] if neighbor in self.graph.nodes else {}
                        results["related_nodes"].append({
                            "id": neighbor,
                            "related_to": [n["id"] for n in primary_nodes if neighbor in self.graph.neighbors(n["id"])],
                            "data": node_data
                        })
                        
                        # Limit number of related nodes
                        if len(results["related_nodes"]) >= context_size * 2:
                            break
            
            # Update metadata
            end_time = time.time()
            results["metadata"]["search_time_ms"] = int((end_time - start_time) * 1000)
            results["metadata"]["primary_count"] = len(results["primary_nodes"])
            results["metadata"]["related_count"] = len(results["related_nodes"])
            
            self.logger.info(f"Found {len(results['primary_nodes'])} primary nodes and {len(results['related_nodes'])} related nodes")
            
        except Exception as e:
            self.logger.error(f"Error finding relevant nodes: {e}")
            results["error"] = str(e)
        
        return results
    
    async def add_emotional_context(self, node_id: str, emotional_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add emotional context to a knowledge graph node.
        
        Args:
            node_id: ID of the node to add emotional context to
            emotional_data: Emotional context data to add
            
        Returns:
            Result of the operation
        """
        self.logger.info(f"Adding emotional context to node {node_id}")
        
        result = {
            "node_id": node_id,
            "timestamp": datetime.now().isoformat(),
            "success": False
        }
        
        try:
            # Check if node exists
            if not await self.has_node(node_id):
                result["error"] = f"Node '{node_id}' not found"
                return result
            
            # Get current node data
            node_data = await self.get_node(node_id)
            
            # Update emotional context
            if "emotional_context" not in node_data:
                node_data["emotional_context"] = {}
                
            # Merge new emotional data
            node_data["emotional_context"].update(emotional_data)
            
            # Update node
            success = await self.update_node(node_id, node_data)
            result["success"] = success
            
            if success:
                result["updated_context"] = node_data["emotional_context"]
                self.logger.info(f"Successfully added emotional context to node {node_id}")
            else:
                result["error"] = "Failed to update node"
                
        except Exception as e:
            self.logger.error(f"Error adding emotional context: {e}")
            result["error"] = str(e)
            
        return result