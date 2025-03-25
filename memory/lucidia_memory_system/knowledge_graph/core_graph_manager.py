"""
Core Graph Manager Module for Lucidia's Knowledge Graph

This module implements the foundational graph operations for the knowledge graph,
handling node and edge management, type tracking, and basic traversal operations.
"""

import networkx as nx
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, deque

from .base_module import KnowledgeGraphModule

class CoreGraphManager(KnowledgeGraphModule):
    """
    Core Graph Manager responsible for the fundamental graph operations.
    
    This module manages the underlying graph structure, including node and edge
    operations, type tracking, and basic graph traversal functionality.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Core Graph Manager."""
        super().__init__(event_bus, module_registry, config)
        
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
            "emotion": set()
        }
        
        # Edge type tracking
        self.edge_types = set()
        
        # Node attributes
        self.node_attributes = {}
        
        # Tracking variables for graph complexity
        self.total_nodes = 0
        self.total_edges = 0
        
        # Knowledge domain tracking
        self.domains = defaultdict(set)
        
        # Relationship strength decay factors
        self.relationship_decay = {
            "standard": 0.01,  # Regular relationships decay slowly
            "dream_associated": 0.02,  # Dream associations fade a bit faster
            "memory_derived": 0.03,  # Memory-based connections fade faster
            "speculative": 0.04,  # Speculative connections fade fastest
            "emotional": 0.015  # Emotional relationships decay between standard and dream_associated
        }
        
        # Initialize indices to avoid potential race conditions during async initialization
        self.node_by_creation_time = {}
        self.edges_by_type = defaultdict(list)
        
        self.logger.info("Core Graph Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("contradiction_detected", self._handle_contradiction)
        await self.event_bus.subscribe("node_update_requested", self._handle_node_update)
        await self.event_bus.subscribe("edge_update_requested", self._handle_edge_update)
        await self.event_bus.subscribe("graph_consistency_check_requested", self._handle_consistency_check)
        self.logger.info("Subscribed to graph-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # Register operation handlers
        self.module_registry.register_operation_handler("get_connected_nodes", self.get_connected_nodes)
        self.module_registry.register_operation_handler("get_node_types", self.get_node_types)
        self.module_registry.register_operation_handler("get_edge_types", self.get_edge_types)
        
        # Set up graph indices for fast lookups
        # self._setup_graph_indices()
        
        self.logger.info("Core Graph Manager setup complete")
    
    # def _setup_graph_indices(self):
    #     """Set up indices for fast graph lookups."""
    #     # Node type index is already maintained in self.node_types
    #     # Domain index is already maintained in self.domains
        
    #     # Setup other indices as needed
    #     # self.node_by_creation_time = {}  # node_id -> creation_time
    #     # self.edges_by_type = defaultdict(list)  # edge_type -> list of (source, target, key)
        
    #     self.logger.info("Graph indices setup complete")
    
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
                    old_domain = current_attrs.get("domain")
                    if old_domain and old_domain in self.domains and node_id in self.domains[old_domain]:
                        self.domains[old_domain].remove(node_id)
                    
                    current_attrs["domain"] = domain
                    self.domains[domain].add(node_id)
                
                # Update modification time
                current_attrs["modified"] = datetime.now().isoformat()
                
                # Emit node updated event
                await self.event_bus.emit("node_updated", {
                    "node_id": node_id,
                    "node_type": node_type,
                    "domain": domain
                })
                
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
            
            # Track node by domain
            self.domains[domain].add(node_id)
            
            # Track creation time for indexing
            self.node_by_creation_time[node_id] = node_attrs["created"]
            
            # Track total nodes
            self.total_nodes += 1
            
            # Emit node added event
            await self.event_bus.emit("node_added", {
                "node_id": node_id,
                "node_type": node_type,
                "domain": domain
            })
            
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
            
            # Get current node data
            current_attrs = self.graph.nodes[node_id]
            node_type = current_attrs.get("type", "unknown")
            old_domain = current_attrs.get("domain")
            
            # Update attributes
            for key, value in attributes.items():
                current_attrs[key] = value
                
                # Handle special case for domain changes
                if key == "domain" and old_domain and old_domain in self.domains and node_id in self.domains[old_domain]:
                    self.domains[old_domain].remove(node_id)
                    self.domains[value].add(node_id)
            
            # Update modification time
            current_attrs["modified"] = datetime.now().isoformat()
            
            # Emit node updated event
            await self.event_bus.emit("node_updated", {
                "node_id": node_id,
                "node_type": node_type,
                "attributes": attributes
            })
            
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
            
            # Track edge by type
            self.edges_by_type[edge_type].append((source, target, edge_key))
            
            # Track total edges
            self.total_edges += 1
            
            # Emit edge added event
            await self.event_bus.emit("edge_added", {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "edge_key": edge_key
            })
            
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
            
            # Get current edge data
            current_attrs = self.graph.edges[source, target, edge_key]
            old_edge_type = current_attrs.get("type")
            
            # Update attributes
            for key, value in attributes.items():
                current_attrs[key] = value
                
                # Handle special case for edge type changes
                if key == "type" and old_edge_type and old_edge_type != value:
                    # Remove from old type index
                    if old_edge_type in self.edges_by_type:
                        self.edges_by_type[old_edge_type].remove((source, target, edge_key))
                    
                    # Add to new type index
                    self.edges_by_type[value].append((source, target, edge_key))
                    self.edge_types.add(value)
            
            # Add modification timestamp
            current_attrs["modified"] = datetime.now().isoformat()
            
            # Emit edge updated event
            await self.event_bus.emit("edge_updated", {
                "source": source,
                "target": target,
                "edge_key": edge_key,
                "edge_type": current_attrs.get("type")
            })
            
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
            
            # Get edge type before removing
            edge_type = self.graph.edges[source, target, edge_key].get("type")
            
            # Remove the edge
            self.graph.remove_edge(source, target, edge_key)
            
            # Update edge type tracking
            if edge_type in self.edges_by_type:
                if (source, target, edge_key) in self.edges_by_type[edge_type]:
                    self.edges_by_type[edge_type].remove((source, target, edge_key))
            
            # Update total edges
            self.total_edges -= 1
            
            # Emit edge removed event
            await self.event_bus.emit("edge_removed", {
                "source": source,
                "target": target,
                "edge_key": edge_key,
                "edge_type": edge_type
            })
            
            self.logger.debug(f"Removed edge: {source} -[{edge_key}]-> {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing edge ({source}, {target}, {edge_key}): {e}")
            return False

    async def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the knowledge graph.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            Success status
        """
        try:
            # Check if node exists
            if not await self.has_node(node_id):
                self.logger.warning(f"Cannot remove node: Node {node_id} does not exist")
                return False
            
            # Get node data before removal
            node_data = self.graph.nodes[node_id]
            node_type = node_data.get("type")
            domain = node_data.get("domain")
            
            # Remove node from graph
            self.graph.remove_node(node_id)
            
            # Update tracking
            if node_type in self.node_types and node_id in self.node_types[node_type]:
                self.node_types[node_type].remove(node_id)
            
            if domain in self.domains and node_id in self.domains[domain]:
                self.domains[domain].remove(node_id)
            
            if node_id in self.node_by_creation_time:
                del self.node_by_creation_time[node_id]
            
            # Update total nodes
            self.total_nodes -= 1
            
            # Emit node removed event
            await self.event_bus.emit("node_removed", {
                "node_id": node_id,
                "node_type": node_type,
                "domain": domain
            })
            
            self.logger.debug(f"Removed node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing node {node_id}: {e}")
            return False

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

    async def get_connected_nodes(self, node_id: str, edge_types: Optional[List[str]] = None, 
                              node_types: Optional[List[str]] = None, 
                              direction: str = "outgoing") -> List[str]:
        """
        Get nodes connected to a given node.
        
        Args:
            node_id: Node ID to get connections for
            edge_types: Optional list of edge types to filter by
            node_types: Optional list of node types to filter by
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of connected node IDs
        """
        if not await self.has_node(node_id):
            return []
            
        connected_nodes = set()
        
        # Get outgoing connections
        if direction in ["outgoing", "both"]:
            for neighbor in self.graph.successors(node_id):
                # Check edge types if specified
                if edge_types:
                    # Get all edges between nodes
                    edges = self.graph.get_edge_data(node_id, neighbor)
                    # Check if any edge matches the specified types
                    if not any(data.get("type") in edge_types for _, data in edges.items()):
                        continue
                
                # Check node types if specified
                if node_types and self.graph.nodes[neighbor].get("type") not in node_types:
                    continue
                    
                connected_nodes.add(neighbor)
        
        # Get incoming connections
        if direction in ["incoming", "both"]:
            for neighbor in self.graph.predecessors(node_id):
                # Check edge types if specified
                if edge_types:
                    # Get all edges between nodes
                    edges = self.graph.get_edge_data(neighbor, node_id)
                    # Check if any edge matches the specified types
                    if not any(data.get("type") in edge_types for _, data in edges.items()):
                        continue
                
                # Check node types if specified
                if node_types and self.graph.nodes[neighbor].get("type") not in node_types:
                    continue
                    
                connected_nodes.add(neighbor)
        
        return list(connected_nodes)

    async def get_subgraph(self, node_ids: List[str], include_edges: bool = True) -> Dict[str, Any]:
        """
        Get a subgraph containing the specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_edges: Whether to include edges between nodes
            
        Returns:
            Dictionary with subgraph data
        """
        # Filter to only existing nodes
        valid_nodes = [node_id for node_id in node_ids if node_id in self.graph.nodes]
        
        subgraph_data = {
            "nodes": {},
            "edges": []
        }
        
        # Add nodes
        for node_id in valid_nodes:
            subgraph_data["nodes"][node_id] = dict(self.graph.nodes[node_id])
        
        # Add edges if requested
        if include_edges:
            for source in valid_nodes:
                for target in valid_nodes:
                    if self.graph.has_edge(source, target):
                        edge_data = self.graph.get_edge_data(source, target)
                        for key, data in edge_data.items():
                            edge_info = dict(data)
                            edge_info["source"] = source
                            edge_info["target"] = target
                            edge_info["key"] = key
                            subgraph_data["edges"].append(edge_info)
        
        return subgraph_data

    async def get_node_types(self) -> Dict[str, int]:
        """
        Get count of nodes by type.
        
        Returns:
            Dictionary mapping node types to counts
        """
        return {node_type: len(nodes) for node_type, nodes in self.node_types.items()}

    async def get_edge_types(self) -> Dict[str, int]:
        """
        Get count of edges by type.
        
        Returns:
            Dictionary mapping edge types to counts
        """
        return {edge_type: len(edges) for edge_type, edges in self.edges_by_type.items()}

    async def get_nodes_by_type(self, node_type: str) -> List[str]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to get
            
        Returns:
            List of node IDs
        """
        if node_type in self.node_types:
            return list(self.node_types[node_type])
        return []

    async def get_nodes_by_domain(self, domain: str) -> List[str]:
        """
        Get all nodes in a specific domain.
        
        Args:
            domain: Domain to get nodes for
            
        Returns:
            List of node IDs
        """
        if domain in self.domains:
            return list(self.domains[domain])
        return []

    async def get_graph_metrics(self) -> Dict[str, Any]:
        """
        Get basic metrics about the graph.
        
        Returns:
            Dictionary with graph metrics
        """
        metrics = {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "node_types": await self.get_node_types(),
            "edge_types": await self.get_edge_types(),
            "domains": {domain: len(nodes) for domain, nodes in self.domains.items()},
            "density": 0.0
        }
        
        # Calculate density if there are nodes
        if self.total_nodes > 1:
            metrics["density"] = self.total_edges / (self.total_nodes * (self.total_nodes - 1))
        
        return metrics

    async def _handle_contradiction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle contradiction detection events.
        
        Args:
            data: Contradiction data
            
        Returns:
            Processing result
        """
        self.logger.info(f"Handling contradiction event: {data.get('type', 'unknown')}")
        
        # Forward to contradiction manager
        contradiction_manager = self.module_registry.get_module("contradiction_manager")
        if contradiction_manager:
            return await contradiction_manager.handle_contradiction(data)
        
        self.logger.warning("Contradiction manager not found")
        return {"success": False, "error": "Contradiction manager not found"}

    async def _handle_node_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle node update requests.
        
        Args:
            data: Node update data
            
        Returns:
            Update result
        """
        node_id = data.get("node_id")
        attributes = data.get("attributes", {})
        
        if not node_id:
            return {"success": False, "error": "Node ID required"}
        
        result = await self.update_node(node_id, attributes)
        return {"success": result, "node_id": node_id}

    async def _handle_edge_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle edge update requests.
        
        Args:
            data: Edge update data
            
        Returns:
            Update result
        """
        source = data.get("source")
        target = data.get("target")
        edge_key = data.get("edge_key")
        attributes = data.get("attributes", {})
        
        if not all([source, target, edge_key is not None]):
            return {"success": False, "error": "Source, target, and edge_key required"}
        
        result = await self.update_edge(source, target, edge_key, attributes)
        return {"success": result, "source": source, "target": target, "edge_key": edge_key}

    async def _handle_consistency_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle graph consistency check requests.
        
        Args:
            data: Check request data
            
        Returns:
            Check results
        """
        self.logger.info("Performing graph consistency check")
        
        # Verify node type tracking
        node_type_consistent = True
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            if node_type in self.node_types and node_id not in self.node_types[node_type]:
                node_type_consistent = False
                self.logger.warning(f"Node {node_id} missing from node_types[{node_type}]")
                # Repair
                self.node_types[node_type].add(node_id)
        
        # Verify domain tracking
        domain_consistent = True
        for node_id, node_data in self.graph.nodes(data=True):
            domain = node_data.get("domain", "general_knowledge")
            if node_id not in self.domains[domain]:
                domain_consistent = False
                self.logger.warning(f"Node {node_id} missing from domains[{domain}]")
                # Repair
                self.domains[domain].add(node_id)
        
        # Verify edge type tracking
        edge_type_consistent = True
        for source, target, key, data in self.graph.edges(data=True, keys=True):
            edge_type = data.get("type", "unknown")
            if edge_type not in self.edge_types:
                edge_type_consistent = False
                self.logger.warning(f"Edge type {edge_type} missing from edge_types")
                # Repair
                self.edge_types.add(edge_type)
            
            if (source, target, key) not in self.edges_by_type[edge_type]:
                edge_type_consistent = False
                self.logger.warning(f"Edge ({source}, {target}, {key}) missing from edges_by_type[{edge_type}]")
                # Repair
                self.edges_by_type[edge_type].append((source, target, key))
        
        # Verify creation time tracking
        creation_time_consistent = True
        for node_id, node_data in self.graph.nodes(data=True):
            created = node_data.get("created")
            if created and node_id not in self.node_by_creation_time:
                creation_time_consistent = False
                self.logger.warning(f"Node {node_id} missing from node_by_creation_time")
                # Repair
                self.node_by_creation_time[node_id] = created
        
        # Verify counts
        actual_node_count = self.graph.number_of_nodes()
        actual_edge_count = self.graph.number_of_edges()
        
        count_consistent = (actual_node_count == self.total_nodes and 
                           actual_edge_count == self.total_edges)
        
        if not count_consistent:
            self.logger.warning(f"Count inconsistency: tracked {self.total_nodes} nodes (actual {actual_node_count}), " +
                               f"tracked {self.total_edges} edges (actual {actual_edge_count})")
            # Repair
            self.total_nodes = actual_node_count
            self.total_edges = actual_edge_count
        
        results = {
            "success": node_type_consistent and domain_consistent and edge_type_consistent and 
                      creation_time_consistent and count_consistent,
            "node_type_consistent": node_type_consistent,
            "domain_consistent": domain_consistent,
            "edge_type_consistent": edge_type_consistent,
            "creation_time_consistent": creation_time_consistent,
            "count_consistent": count_consistent,
            "actual_node_count": actual_node_count,
            "actual_edge_count": actual_edge_count,
            "tracked_node_count": self.total_nodes,
            "tracked_edge_count": self.total_edges
        }
        
        self.logger.info(f"Graph consistency check completed: {results['success']}")
        return results

    async def process_external_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process external knowledge data.
        
        Args:
            data: External knowledge data
            
        Returns:
            Processing result
        """
        data_type = data.get("type", "unknown")
        
        if data_type == "concept":
            return await self._process_concept_data(data)
        elif data_type == "entity":
            return await self._process_entity_data(data)
        elif data_type == "relationship":
            return await self._process_relationship_data(data)
        else:
            self.logger.warning(f"Unknown data type for processing: {data_type}")
            return {"success": False, "error": f"Unknown data type: {data_type}"}

    async def _process_concept_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process concept data.
        
        Args:
            data: Concept data
            
        Returns:
            Processing result
        """
        try:
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
            
            # Add the concept as a node
            node_id = concept["id"]
            node_type = "concept"
            domain = concept.get("domain", "general_knowledge")
            
            # Create node attributes
            attributes = {
                "definition": concept.get("definition", ""),
                "confidence": concept.get("confidence", 0.7),
                "source": data.get("source", "external"),
            }
            
            # Add any additional attributes
            for key, value in concept.items():
                if key not in ["id", "domain"]:
                    attributes[key] = value
            
            # Add the node
            success = await self.add_node(node_id, node_type, attributes, domain)
            
            return {
                "success": success,
                "node_id": node_id,
                "node_type": node_type,
                "domain": domain
            }
            
        except Exception as e:
            self.logger.error(f"Error processing concept data: {e}")
            return {"success": False, "error": str(e)}

    async def _process_entity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entity data.
        
        Args:
            data: Entity data
            
        Returns:
            Processing result
        """
        try:
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
            
            # Add the entity as a node
            node_id = entity["id"]
            node_type = "entity"
            domain = entity.get("domain", "general_knowledge")
            
            # Create node attributes
            attributes = {
                "name": entity.get("name", ""),
                "description": entity.get("description", ""),
                "confidence": entity.get("confidence", 0.7),
                "source": data.get("source", "external"),
            }
            
            # Add any additional attributes
            for key, value in entity.items():
                if key not in ["id", "domain"]:
                    attributes[key] = value
            
            # Add the node
            success = await self.add_node(node_id, node_type, attributes, domain)
            
            return {
                "success": success,
                "node_id": node_id,
                "node_type": node_type,
                "domain": domain
            }
            
        except Exception as e:
            self.logger.error(f"Error processing entity data: {e}")
            return {"success": False, "error": str(e)}

    async def _process_relationship_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process relationship data.
        
        Args:
            data: Relationship data
            
        Returns:
            Processing result
        """
        try:
            relationship = data.get("data", {})
            
            # Ensure relationship has source and target
            if "source" not in relationship or "target" not in relationship:
                self.logger.error("Relationship missing source or target")
                return {"success": False, "error": "Relationship must have source and target"}
            
            # Ensure relationship has a type
            if "type" not in relationship:
                relationship["type"] = "related_to"
            
            # Set strength if not provided
            if "strength" not in relationship:
                relationship["strength"] = 0.7
            
            # Set confidence if not provided
            if "confidence" not in relationship:
                relationship["confidence"] = 0.7
            
            source = relationship["source"]
            target = relationship["target"]
            rel_type = relationship["type"]
            
            # Create nodes if they don't exist
            if not await self.has_node(source):
                await self.add_node(
                    source,
                    relationship.get("source_type", "entity"),
                    {
                        "name": relationship.get("source_name", source),
                        "placeholder": True,
                        "confidence": 0.5
                    }
                )
            
            if not await self.has_node(target):
                await self.add_node(
                    target,
                    relationship.get("target_type", "entity"),
                    {
                        "name": relationship.get("target_name", target),
                        "placeholder": True,
                        "confidence": 0.5
                    }
                )
            
            # Create edge attributes
            attributes = {
                "strength": relationship.get("strength", 0.7),
                "confidence": relationship.get("confidence", 0.7),
                "source": data.get("source", "external"),
            }
            
            # Add any additional attributes
            for key, value in relationship.items():
                if key not in ["source", "target", "type", "source_type", "target_type", "source_name", "target_name"]:
                    attributes[key] = value
            
            # Add the edge
            edge_key = await self.add_edge(source, target, rel_type, attributes)
            
            return {
                "success": edge_key is not None,
                "source": source,
                "target": target,
                "edge_type": rel_type,
                "edge_key": edge_key
            }
            
        except Exception as e:
            self.logger.error(f"Error processing relationship data: {e}")
            return {"success": False, "error": str(e)}

    async def get_node_attribute(self, node_id: str, attribute_name: str) -> Any:
        """
        Get a specific attribute value for a node.
        
        Args:
            node_id: Unique identifier for the node
            attribute_name: Name of the attribute to retrieve
            
        Returns:
            The attribute value if it exists, None otherwise
        """
        try:
            # Check if node exists
            if not await self.has_node(node_id):
                self.logger.warning(f"Cannot get attribute for node {node_id}: Node does not exist")
                return None
            
            # Get node data
            node_data = self.graph.nodes[node_id]
            
            # Return attribute value or None if it doesn't exist
            return node_data.get(attribute_name)
            
        except Exception as e:
            self.logger.error(f"Error getting attribute {attribute_name} for node {node_id}: {e}")
            return None

    async def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all nodes in the knowledge graph.
        
        Returns:
            Dictionary mapping node IDs to their attribute dictionaries
        """
        try:
            # Return a dictionary with node_id as keys and node attributes as values
            return dict(self.graph.nodes(data=True))
        except Exception as e:
            self.logger.error(f"Error getting all nodes: {e}")
            return {}