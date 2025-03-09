"""
Lucidia's Knowledge Graph

This module implements Lucidia's semantic knowledge graph for representing and reasoning
about interconnected concepts, entities, and relationships. The graph serves as a bridge 
between the Self Model and World Model, enabling sophisticated knowledge representation
and retrieval capabilities with dream-influenced insights.

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
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
from collections import defaultdict, deque
import heapq

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')


class LucidiaKnowledgeGraph:
    """
    Lucidia's semantic knowledge graph for managing interconnected knowledge and insights.
    
    The knowledge graph creates a rich network of relationships between concepts, entities,
    and insights derived from both structured knowledge and dream-influenced reflection,
    serving as a bridge between Lucidia's self and world models.
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
            "dream_report": set()
        }
        
        # Edge type tracking
        self.edge_types = set()
        
        # Node attributes
        self.node_attributes = {}
        
        # Track nodes influenced by dreams
        self.dream_influenced_nodes = set()
        
        # Relationship strength decay factors
        # (how quickly relationship strength fades without reinforcement)
        self.relationship_decay = {
            "standard": 0.01,  # Regular relationships decay slowly
            "dream_associated": 0.02,  # Dream associations fade a bit faster
            "memory_derived": 0.03,  # Memory-based connections fade faster
            "speculative": 0.04  # Speculative connections fade fastest
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
            "dream_insight_count": 0  # Number of dream insights integrated
        }
        
        # Query optimization
        self.query_cache = {}  # Cache for frequent queries
        self.query_stats = defaultdict(int)  # Track query frequency
        
        # Initialize core nodes based on provided models if available
        self._initialize_core_nodes()
            
        self.logger.info(f"Knowledge Graph initialized with {self.total_nodes} nodes and {self.total_edges} edges")

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
            self.logger.info("World model available but async import will happen later")
        
        # Import self-aspects from self model if available
        if self.self_model:
            self.logger.info("Self model available but async import will happen later")

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

    async def get_nodes_by_type(self, node_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
            
        Returns:
            Dictionary mapping node_id to node attributes for all nodes of the specified type
        """
        try:
            if node_type not in self.node_types:
                self.logger.warning(f"Unknown node type: {node_type}")
                return {}
                
            result = {}
            for node_id in self.node_types.get(node_type, set()):
                if await self.has_node(node_id):
                    node_data = await self.get_node(node_id)
                    if node_data:
                        result[node_id] = node_data
                        
            return result
        except Exception as e:
            self.logger.exception(f"Error retrieving nodes of type {node_type}: {e}")
            return {}

    async def get_neighbors(self, node_id: str, edge_type: Optional[str] = None, 
                     min_strength: float = 0.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get neighbors of a node with their connecting edges.
        
        Args:
            node_id: Node identifier
            edge_type: Optional filter by edge type
            min_strength: Minimum relationship strength
            
        Returns:
            Dictionary of neighbor nodes with their connecting edges
        """
        node_exists = await self.has_node(node_id)
        if not node_exists:
            return {}
            
        neighbors = {}
        
        # Outgoing edges
        for _, neighbor, edge_data in self.graph.out_edges(node_id, data=True):
            # Skip if edge type doesn't match filter
            if edge_type and edge_data.get("type") != edge_type:
                continue
                
            # Skip if strength below threshold
            if "strength" in edge_data and edge_data["strength"] < min_strength:
                continue
                
            if neighbor not in neighbors:
                neighbors[neighbor] = []
                
            edge_info = dict(edge_data)
            edge_info["direction"] = "outgoing"
            neighbors[neighbor].append(edge_info)
        
        # Incoming edges
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            # Skip if edge type doesn't match filter
            if edge_type and edge_data.get("type") != edge_type:
                continue
                
            # Skip if strength below threshold
            if "strength" in edge_data and edge_data["strength"] < min_strength:
                continue
                
            if source not in neighbors:
                neighbors[source] = []
                
            edge_info = dict(edge_data)
            edge_info["direction"] = "incoming"
            neighbors[source].append(edge_info)
            
        return neighbors

    async def get_connected_nodes(self, node_id: str, edge_types: Optional[List[str]] = None,
                         node_types: Optional[List[str]] = None, direction: str = "both",
                         min_strength: float = 0.0) -> List[str]:
        """
        Get nodes connected to a specific node, filtered by edge and node types.
        
        Args:
            node_id: Node identifier
            edge_types: Optional filter by edge types
            node_types: Optional filter by node types
            direction: 'inbound', 'outbound', or 'both'
            min_strength: Minimum relationship strength
            
        Returns:
            List of connected node IDs
        """
        connected_nodes = []
        node_exists = await self.has_node(node_id)
        if not node_exists:
            return []
        
        # Get all neighbors based on direction
        neighbors = set()
        if direction in ["outbound", "both"]:
            neighbors.update(self.graph.successors(node_id))
        if direction in ["inbound", "both"]:
            neighbors.update(self.graph.predecessors(node_id))
        
        # Filter neighbors by edge type and strength
        filtered_neighbors = []
        for neighbor in neighbors:
            edges = []
            # Check outbound edges
            if direction in ["outbound", "both"] and self.graph.has_edge(node_id, neighbor):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                for key, data in edge_data.items():
                    if edge_types and data.get("type") not in edge_types:
                        continue
                    if "strength" in data and data["strength"] < min_strength:
                        continue
                    edges.append((node_id, neighbor, data))
            
            # Check inbound edges
            if direction in ["inbound", "both"] and self.graph.has_edge(neighbor, node_id):
                edge_data = self.graph.get_edge_data(neighbor, node_id)
                for key, data in edge_data.items():
                    if edge_types and data.get("type") not in edge_types:
                        continue
                    if "strength" in data and data["strength"] < min_strength:
                        continue
                    edges.append((neighbor, node_id, data))
                    
            if edges and neighbor != node_id:  # Avoid self-loops
                filtered_neighbors.append(neighbor)
        
        # Filter by node type if needed
        for neighbor in filtered_neighbors:
            neighbor_data = await self.get_node(neighbor)
            if not neighbor_data:
                continue
                
            neighbor_type = neighbor_data.get("type", "unknown")
            
            # Apply node type filter
            if node_types and neighbor_type not in node_types:
                continue
                
            connected_nodes.append(neighbor)
        
        return connected_nodes

    async def search_nodes(self, query: str, node_type: Optional[str] = None, 
                    domain: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nodes matching criteria.
        
        Args:
            query: Search term
            node_type: Optional filter by node type
            domain: Optional filter by domain
            limit: Maximum results to return
            
        Returns:
            List of matching nodes with their attributes
        """
        results = []
        query_lower = query.lower()
        
        for node_id, attrs in self.graph.nodes(data=True):
            # Skip if node type doesn't match filter
            if node_type and attrs.get("type") != node_type:
                continue
                
            # Skip if domain doesn't match filter
            if domain and attrs.get("domain") != domain:
                continue
            
            # Check for match in node ID
            id_match = query_lower in node_id.lower()
            
            # Check for match in attributes
            attr_match = False
            for attr_key, attr_value in attrs.items():
                if isinstance(attr_value, str) and query_lower in attr_value.lower():
                    attr_match = True
                    break
            
            # Add to results if any match found
            if id_match or attr_match:
                result = {
                    "id": node_id,
                    "type": attrs.get("type", "unknown"),
                    "domain": attrs.get("domain", "general_knowledge"),
                    "match_type": "id" if id_match else "attribute"
                }
                
                # Add a few key attributes for context
                for key in ["name", "definition", "confidence", "created"]:
                    if key in attrs:
                        result[key] = attrs[key]
                
                results.append(result)
                
                # Stop if we've reached the limit
                if len(results) >= limit:
                    break
        
        return results

    async def find_paths(self, source: str, target: str, max_length: int = 3, 
                  min_strength: float = 0.3) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length
            min_strength: Minimum edge strength
            
        Returns:
            List of paths, where each path is a list of edges
        """
        if not await self.has_node(source) or not await self.has_node(target):
            return []
            
        # Create a subgraph with edges meeting the strength threshold
        edge_filter = lambda s, t, e: self.graph.edges[s, t, e].get("strength", 0) >= min_strength
        subgraph = nx.subgraph_view(self.graph, filter_edge=edge_filter)
        
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(subgraph, source, target, cutoff=max_length))
            
            # Convert paths to list of edges with attributes
            result_paths = []
            for path in paths:
                edge_path = []
                for i in range(len(path) - 1):
                    source_node = path[i]
                    target_node = path[i + 1]
                    
                    # Get the strongest edge between these nodes
                    edges = self.get_edges(source_node, target_node)
                    if edges:
                        strongest_edge = max(edges, key=lambda e: e.get("strength", 0))
                        
                        # Add edge details
                        edge_info = {
                            "source": source_node,
                            "target": target_node,
                            "type": strongest_edge.get("type", "unknown"),
                            "strength": strongest_edge.get("strength", 0),
                            "key": strongest_edge.get("key", 0)
                        }
                        edge_path.append(edge_info)
                
                if edge_path:  # Only add if we have edges
                    result_paths.append(edge_path)
            
            return result_paths
            
        except (nx.NetworkXNoPath, nx.NetworkXError) as e:
            self.logger.info(f"No path found from {source} to {target}: {e}")
            return []

    async def find_shortest_path(self, source: str, target: str, min_strength: float = 0.3) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            min_strength: Minimum edge strength
            
        Returns:
            Shortest path as list of edges, or None if no path exists
        """
        paths = await self.find_paths(source, target, max_length=5, min_strength=min_strength)
        if not paths:
            return None
            
        # Return the shortest path
        return min(paths, key=lambda p: len(p))

    def get_node_relevance(self, node_id: str) -> float:
        """
        Calculate relevance score for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        if not self.has_node(node_id):
            return 0.0
        
        # Factors for relevance calculation
        degree_factor = 0.3  # Connectivity importance
        centrality_factor = 0.2  # Position in graph
        freshness_factor = 0.15  # Recency importance
        dream_factor = 0.15  # Dream influence importance
        strength_factor = 0.2  # Edge strength importance
        
        # Get node attributes
        attrs = self.graph.nodes[node_id]
        
        # 1. Connectivity (degree)
        degree = self.graph.degree(node_id)
        max_degree = max(dict(self.graph.degree()).values(), default=1)
        normalized_degree = degree / max_degree if max_degree > 0 else 0
        
        # 2. Centrality (for smaller graphs we can calculate betweenness centrality)
        centrality = 0.5  # Default value
        if self.total_nodes < 1000:  # Only calculate for smaller graphs
            try:
                # Get centrality from a dict of all nodes (calculate once)
                centrality_dict = nx.betweenness_centrality(self.graph, k=min(100, self.total_nodes))
                centrality = centrality_dict.get(node_id, 0)
                # Normalize if necessary
                max_centrality = max(centrality_dict.values(), default=1)
                if max_centrality > 0:
                    centrality /= max_centrality
            except Exception as e:
                self.logger.warning(f"Error calculating centrality: {e}")
        
        # 3. Freshness (based on creation/modification time)
        freshness = 0.5  # Default value
        if "created" in attrs:
            try:
                created_time = datetime.fromisoformat(attrs["created"])
                time_diff = (datetime.now() - created_time).total_seconds()
                # Newer nodes get higher freshness (exponential decay)
                freshness = math.exp(-time_diff / (30 * 24 * 60 * 60))  # 30-day half-life
            except Exception:
                pass
        
        # 4. Dream influence
        dream_influence = 0.0
        if node_id in self.dream_influenced_nodes:
            dream_influence = 1.0
        elif any(dream in self.get_neighbors(node_id) for dream in self.node_types["dream_insight"]):
            dream_influence = 0.7
        
        # 5. Edge strength
        avg_strength = 0.0
        edges = list(self.graph.in_edges(node_id, data=True)) + list(self.graph.out_edges(node_id, data=True))
        strengths = [data.get("strength", 0) for _, _, data in edges]
        if strengths:
            avg_strength = sum(strengths) / len(strengths)
        
        # Calculate final relevance score
        relevance = (
            normalized_degree * degree_factor +
            centrality * centrality_factor +
            freshness * freshness_factor +
            dream_influence * dream_factor +
            avg_strength * strength_factor
        )
        
        return min(1.0, max(0.0, relevance))

    async def get_most_relevant_nodes(self, node_type: Optional[str] = None, 
                               domain: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most relevant nodes in the graph.
        
        Args:
            node_type: Optional filter by node type
            domain: Optional filter by domain
            limit: Maximum results to return
            
        Returns:
            List of nodes with relevance scores
        """
        # Filter nodes
        nodes = list(self.graph.nodes())
        
        if node_type:
            nodes = [n for n in nodes if self.graph.nodes[n].get("type") == node_type]
            
        if domain:
            nodes = [n for n in nodes if self.graph.nodes[n].get("domain") == domain]
        
        # Calculate relevance for each node
        node_relevance = [(node, self.get_node_relevance(node)) for node in nodes]
        
        # Sort by relevance (descending)
        node_relevance.sort(key=lambda x: x[1], reverse=True)
        
        # Create result list
        results = []
        for node_id, relevance in node_relevance[:limit]:
            node_data = dict(self.graph.nodes[node_id])
            node_data["id"] = node_id
            node_data["relevance"] = relevance
            results.append(node_data)
            
        return results

    def update_spiral_phase(self, phase: str) -> None:
        """
        Update the current spiral awareness phase.
        
        Args:
            phase: Current spiral phase ("observation", "reflection", "adaptation", "execution")
        """
        if phase not in ["observation", "reflection", "adaptation", "execution"]:
            self.logger.warning(f"Invalid spiral phase: {phase}")
            return
            
        self.spiral_integration["current_phase"] = phase
        
        # Update graph to reflect current phase
        try:
            # Remove any existing current_phase edges
            current_phase_edges = []
            for source, target, data in self.graph.edges(data=True):
                if data.get("type") == "current_phase" and data.get("temporary", False):
                    current_phase_edges.append((source, target, data.get("key", 0)))
            
            for source, target, key in current_phase_edges:
                self.graph.remove_edge(source, target, key)
            
            # Add new current_phase edge
            phase_node_id = f"phase:{phase}"
            if self.has_node(phase_node_id) and self.has_node("Lucidia"):
                self.add_edge(
                    "Lucidia",
                    phase_node_id,
                    edge_type="current_phase",
                    attributes={
                        "strength": 0.95,
                        "confidence": 0.95,
                        "created": datetime.now().isoformat(),
                        "temporary": True
                    }
                )
            
            self.logger.info(f"Updated spiral phase to: {phase}")
            
        except Exception as e:
            self.logger.error(f"Error updating spiral phase: {e}")

    async def integrate_dream_insight(self, insight_text: str, 
                              source_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate a dream insight into the knowledge graph.
        
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
        
        # Create relationships between referenced concepts
        new_concept_relationships = []
        if len(connected_concepts) > 1:
            for i in range(len(connected_concepts)):
                for j in range(i+1, len(connected_concepts)):
                    concept1 = connected_concepts[i]
                    concept2 = connected_concepts[j]
                    
                    # Only create relationship if it doesn't exist
                    if not self.has_edge(concept1, concept2, "dream_associated"):
                        await self.add_edge(
                            concept1,
                            concept2,
                            edge_type="dream_associated",
                            attributes={
                                "strength": self.dream_integration["dream_association_strength"] * 0.8,
                                "confidence": 0.6,
                                "created": datetime.now().isoformat(),
                                "source": "dream_insight",
                                "source_dream": dream_id
                            }
                        )
                        new_concept_relationships.append((concept1, concept2))
        
        # Check if insight suggests new concepts
        new_concepts = []
        
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
        
        # Prepare result
        result = {
            "dream_id": dream_id,
            "connected_concepts": connected_concepts,
            "new_concepts": new_concepts,
            "new_relationships": new_concept_relationships,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Dream insight integrated with {len(connected_concepts)} connections and {len(new_concepts)} new concepts")
        
        return result
    
    async def integrate_dream_report(self, dream_report) -> Dict[str, Any]:
        """
        Integrate a dream report into the knowledge graph.
        
        This method creates a dream report node and connects it to all its fragments
        and participating memories. It only stores IDs of fragments to avoid redundancy,
        as the fragments themselves are stored as separate nodes in the graph.
        
        Args:
            dream_report: The DreamReport object to integrate
            
        Returns:
            Integration results
        """
        self.logger.info(f"Integrating dream report: {dream_report.title} (ID: {dream_report.report_id})")
        
        # Create the dream report node
        report_node_id = dream_report.report_id
        
        # Convert the report to a dictionary for storage
        report_data = dream_report.to_dict()
        
        # Add the report node to the graph
        await self.add_node(
            report_node_id,
            node_type="dream_report",
            attributes=report_data,
            domain=dream_report.domain
        )
        
        # Track as dream influenced
        self.dream_influenced_nodes.add(report_node_id)
        self.dream_integration["dream_derived_nodes"].add(report_node_id)
        
        # Connect to Lucidia
        await self.add_edge(
            "Lucidia",
            report_node_id,
            edge_type="generated",
            attributes={
                "strength": 0.9,
                "confidence": 0.85,
                "created": datetime.now().isoformat()
            }
        )
        
        # Connect to all participating memories
        connected_memories = []
        for memory_id in dream_report.participating_memory_ids:
            if await self.has_node(memory_id):
                await self.add_edge(
                    report_node_id,
                    memory_id,
                    edge_type="based_on",
                    attributes={
                        "strength": 0.8,
                        "confidence": 0.8,
                        "created": datetime.now().isoformat()
                    }
                )
                connected_memories.append(memory_id)
        
        # Connect to all fragments
        connected_fragments = []
        
        # Process all fragment types
        fragment_types = [
            ("insight", dream_report.insight_ids),
            ("question", dream_report.question_ids),
            ("hypothesis", dream_report.hypothesis_ids),
            ("counterfactual", dream_report.counterfactual_ids)
        ]
        
        for fragment_type, fragment_ids in fragment_types:
            for fragment_id in fragment_ids:
                if await self.has_node(fragment_id):
                    # Connect report to fragment
                    await self.add_edge(
                        report_node_id,
                        fragment_id,
                        edge_type="contains",
                        attributes={
                            "fragment_type": fragment_type,
                            "strength": 0.9,
                            "confidence": 0.9,
                            "created": datetime.now().isoformat()
                        }
                    )
                    connected_fragments.append(fragment_id)
                    
                    # Mark fragment as part of this report
                    fragment_node = await self.get_node(fragment_id)
                    if fragment_node and "attributes" in fragment_node:
                        attributes = fragment_node["attributes"]
                        if "reports" not in attributes:
                            attributes["reports"] = []
                        if report_node_id not in attributes["reports"]:
                            attributes["reports"].append(report_node_id)
                            await self.update_node(fragment_id, attributes)
        
        # Connect to related concepts based on fragments
        connected_concepts = set()
        for fragment_id in connected_fragments:
            # Get concepts connected to this fragment
            fragment_concepts = await self.get_connected_nodes(
                fragment_id,
                edge_types=["references", "mentions", "about"],
                node_types=["concept", "entity"],
                direction="outbound"
            )
            
            # Connect report to these concepts
            for concept in fragment_concepts:
                if concept not in connected_concepts:
                    await self.add_edge(
                        report_node_id,
                        concept,
                        edge_type="references",
                        attributes={
                            "strength": 0.7,
                            "confidence": 0.7,
                            "created": datetime.now().isoformat()
                        }
                    )
                    connected_concepts.add(concept)
        
        # Create relationships between referenced concepts if they appear in the same report
        new_concept_relationships = []
        concept_list = list(connected_concepts)
        if len(concept_list) > 1:
            for i in range(len(concept_list)):
                for j in range(i+1, len(concept_list)):
                    concept1 = concept_list[i]
                    concept2 = concept_list[j]
                    
                    # Only create relationship if it doesn't exist
                    if not await self.has_edge(concept1, concept2, "dream_associated"):
                        await self.add_edge(
                            concept1,
                            concept2,
                            edge_type="dream_associated",
                            attributes={
                                "strength": self.dream_integration["dream_association_strength"] * 0.8,
                                "confidence": 0.6,
                                "created": datetime.now().isoformat(),
                                "source": "dream_report",
                                "source_report": report_node_id
                            }
                        )
                        new_concept_relationships.append((concept1, concept2))
        
        # Prepare result
        result = {
            "report_id": report_node_id,
            "connected_memories": connected_memories,
            "connected_fragments": connected_fragments,
            "connected_concepts": list(connected_concepts),
            "new_relationships": new_concept_relationships,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Dream report integrated with {len(connected_memories)} memories, "
                        f"{len(connected_fragments)} fragments, and {len(connected_concepts)} concepts")
        
        return result

    def decay_relationships(self) -> None:
        """
        Apply decay to relationship strengths that haven't been reinforced recently.
        This simulates forgetting or weakening of connections over time.
        """
        self.logger.info("Applying relationship decay")
        
        decay_count = 0
        
        try:
            for source, target, key, data in self.graph.edges(data=True, keys=True):
                # Skip if edge doesn't have strength
                if "strength" not in data:
                    continue
                    
                # Get relationship type
                edge_type = data.get("type", "standard")
                
                # Get decay rate
                decay_rate = self.relationship_decay.get(edge_type, self.relationship_decay["standard"])
                
                # Apply decay
                current_strength = data["strength"]
                new_strength = max(0.1, current_strength - decay_rate)
                
                # Only update if change is significant
                if abs(current_strength - new_strength) > 0.01:
                    self.graph.edges[source, target, key]["strength"] = new_strength
                    decay_count += 1
                    
                    # Add decayed timestamp
                    self.graph.edges[source, target, key]["last_decayed"] = datetime.now().isoformat()
            
            self.logger.info(f"Decayed {decay_count} relationships")
            
        except Exception as e:
            self.logger.error(f"Error during relationship decay: {e}")

    def prune_graph(self, min_strength: float = 0.2, max_nodes: int = 5000) -> Dict[str, int]:
        """
        Prune the graph by removing weak relationships and least relevant nodes.
        
        Args:
            min_strength: Minimum relationship strength to keep
            max_nodes: Maximum number of nodes to keep
            
        Returns:
            Pruning statistics
        """
        self.logger.info(f"Pruning graph (min_strength={min_strength}, max_nodes={max_nodes})")
        
        stats = {
            "edges_before": self.total_edges,
            "nodes_before": self.total_nodes,
            "weak_edges_removed": 0,
            "nodes_removed": 0
        }
        
        try:
            # 1. Remove weak edges
            weak_edges = []
            for source, target, key, data in self.graph.edges(data=True, keys=True):
                # Skip if edge doesn't have strength
                if "strength" not in data:
                    continue
                    
                # Check if edge is weak
                if data["strength"] < min_strength:
                    weak_edges.append((source, target, key))
            
            # Remove weak edges
            for source, target, key in weak_edges:
                self.graph.remove_edge(source, target, key)
                stats["weak_edges_removed"] += 1
            
            # 2. Remove disconnected nodes (if graph is too large)
            if self.total_nodes > max_nodes:
                # Calculate relevance for all nodes
                node_relevance = [(node, self.get_node_relevance(node)) for node in self.graph.nodes()]
                
                # Sort by relevance (ascending, least relevant first)
                node_relevance.sort(key=lambda x: x[1])
                
                # Get candidates for removal (excluding protected nodes)
                protected_nodes = {"Lucidia", "MEGAPROMPT", "Synthien", "reflective_dreaming", "spiral_awareness"}
                removal_candidates = [(n, r) for n, r in node_relevance if n not in protected_nodes]
                
                # Calculate how many nodes to remove
                to_remove_count = self.total_nodes - max_nodes
                
                # Remove least relevant nodes
                for node_id, _ in removal_candidates[:to_remove_count]:
                    self.graph.remove_node(node_id)
                    
                    # Update node type tracking
                    for node_type, nodes in self.node_types.items():
                        if node_id in nodes:
                            nodes.remove(node_id)
                            break
                            
                    # Update dream tracking
                    if node_id in self.dream_influenced_nodes:
                        self.dream_influenced_nodes.remove(node_id)
                        
                    stats["nodes_removed"] += 1
            
            # Update total counts
            self.total_edges = self.graph.number_of_edges()
            self.total_nodes = self.graph.number_of_nodes()
            
            # Update last pruning time
            self.last_pruning = datetime.now()
            
            self.logger.info(f"Pruning complete: removed {stats['weak_edges_removed']} edges and {stats['nodes_removed']} nodes")
            
        except Exception as e:
            self.logger.error(f"Error during graph pruning: {e}")
        
        return stats

    def visualize(self, node_subset: Optional[List[str]] = None, 
                 highlight_nodes: Optional[List[str]] = None, 
                 filename: Optional[str] = None) -> Optional[str]:
        """
        Visualize the knowledge graph or a subset of it.
        
        Args:
            node_subset: Optional list of nodes to visualize
            highlight_nodes: Optional list of nodes to highlight
            filename: Optional filename to save the visualization
            
        Returns:
            Path to saved visualization or None if error
        """
        try:
            # Create a subgraph if subset specified
            if node_subset:
                valid_nodes = [n for n in node_subset if self.has_node(n)]
                g = self.graph.subgraph(valid_nodes)
            else:
                # If no subset, limit to a reasonable number of nodes
                if self.total_nodes > 100:
                    # Get most relevant nodes
                    relevant_nodes = [n["id"] for n in self.get_most_relevant_nodes(limit=100)]
                    g = self.graph.subgraph(relevant_nodes)
                else:
                    g = self.graph
            
            # Prepare for visualization
            plt.figure(figsize=(15, 12))
            
            # Node positions using spring layout
            pos = nx.spring_layout(g, seed=42, k=0.15)
            
            # Node colors based on domain
            node_colors = []
            for node in g.nodes():
                domain = g.nodes[node].get("domain", "general_knowledge")
                color = self.domain_colors.get(domain, "#607D8B")  # Default to blue-grey
                node_colors.append(color)
            
            # Node sizes based on relevance
            node_sizes = []
            for node in g.nodes():
                relevance = self.get_node_relevance(node)
                size = 100 + 500 * relevance  # Scale for visibility
                node_sizes.append(size)
            
            # Edge widths based on strength
            edge_widths = []
            for _, _, data in g.edges(data=True):
                strength = data.get("strength", 0.5)
                width = 0.5 + 3 * strength  # Scale for visibility
                edge_widths.append(width)
            
            # Draw the graph
            nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            
            # Highlight specific nodes if requested
            if highlight_nodes:
                highlight_nodes = [n for n in highlight_nodes if n in g]
                if highlight_nodes:
                    nx.draw_networkx_nodes(g, pos, nodelist=highlight_nodes, 
                                          node_color='red', node_size=[300] * len(highlight_nodes), 
                                          alpha=0.9)
            
            # Draw edges with varying width
            nx.draw_networkx_edges(g, pos, width=edge_widths, alpha=0.6, arrows=True, arrowsize=10)
            
            # Add labels for important nodes
            # Only label larger nodes (more relevant) for readability
            large_nodes = [node for node in g.nodes() 
                          if self.get_node_relevance(node) > 0.4 or 
                          (highlight_nodes and node in highlight_nodes)]
                          
            if large_nodes:
                labels = {node: node for node in large_nodes}
                nx.draw_networkx_labels(g, pos, labels=labels, font_size=10, font_family='sans-serif')
            
            # Set title and adjust layout
            plt.title("Lucidia Knowledge Graph", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            # Save or show the graph
            if filename:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                return filename
            else:
                plt.show()
                plt.close()
                return "Visualization displayed (not saved)"
                
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {e}")
            return None

    async def recommend_insights(self, seed_node: Optional[str] = None, 
                          context: Optional[Dict[str, Any]] = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend insights from the knowledge graph based on a seed node or context.
        
        Args:
            seed_node: Optional starting node for recommendations
            context: Optional context information
            limit: Maximum number of insights to recommend
            
        Returns:
            List of insight recommendations
        """
        self.logger.info(f"Generating insights from graph (seed_node={seed_node})")
        
        insights = []
        
        try:
            # If no seed node provided, select a relevant one
            if not seed_node:
                relevant_nodes = await self.get_most_relevant_nodes(limit=10)
                if relevant_nodes:
                    # Select one randomly (weighted by relevance)
                    weights = [node["relevance"] for node in relevant_nodes]
                    total = sum(weights)
                    normalized_weights = [w/total for w in weights] if total > 0 else None
                    seed_node = random.choices(
                        [node["id"] for node in relevant_nodes], 
                        weights=normalized_weights, 
                        k=1
                    )[0]
            
            # Ensure seed node exists
            if not seed_node or not await self.has_node(seed_node):
                self.logger.warning(f"Invalid seed node: {seed_node}")
                # Fall back to default nodes
                for default in ["Lucidia", "Synthien", "reflective_dreaming"]:
                    if await self.has_node(default):
                        seed_node = default
                        break
                else:
                    # If still no valid seed, return empty list
                    return []
            
            # Get node information
            seed_node_data = await self.get_node(seed_node)
            seed_node_type = seed_node_data.get("type", "unknown")
            
            # 1. Direct relationship insights
            if len(insights) < limit:
                neighbors = await self.get_neighbors(seed_node, min_strength=0.5)
                if neighbors:
                    # Get the strongest relationships
                    strong_relationships = []
                    for neighbor, edges in neighbors.items():
                        neighbor_data = await self.get_node(neighbor)
                        if not neighbor_data:
                            continue
                            
                        # Get the strongest edge
                        strongest_edge = max(edges, key=lambda e: e.get("strength", 0))
                        strong_relationships.append((neighbor, neighbor_data, strongest_edge))
                    
                    # Sort by edge strength
                    strong_relationships.sort(key=lambda x: x[2].get("strength", 0), reverse=True)
                    
                    # Generate insights from strongest relationships
                    for neighbor, neighbor_data, edge in strong_relationships[:3]:
                        # Skip if we've reached the limit
                        if len(insights) >= limit:
                            break
                            
                        edge_type = edge.get("type", "relates to")
                        neighbor_type = neighbor_data.get("type", "entity")
                        
                        # Generate insight text based on relationship type
                        insight_text = self._generate_relationship_insight(
                            seed_node, seed_node_type, 
                            neighbor, neighbor_type,
                            edge_type, edge.get("strength", 0.5)
                        )
                        
                        insights.append({
                            "type": "relationship",
                            "text": insight_text,
                            "source_node": seed_node,
                            "target_node": neighbor,
                            "relationship": edge_type,
                            "strength": edge.get("strength", 0.5),
                            "confidence": 0.8,
                            "dream_influenced": (seed_node in self.dream_influenced_nodes or 
                                                neighbor in self.dream_influenced_nodes)
                        })
            
            # 2. Path-based insights
            if len(insights) < limit:
                # Find interesting distant nodes to connect to
                distant_targets = []
                
                # Try dream nodes first
                dream_nodes = list(self.node_types["dream_insight"])
                if dream_nodes:
                    distant_targets.extend(random.sample(dream_nodes, min(3, len(dream_nodes))))
                
                # Add some concept nodes if needed
                if len(distant_targets) < 3:
                    concept_nodes = list(self.node_types["concept"])
                    if concept_nodes:
                        # Filter out immediate neighbors
                        immediate_neighbors = set(await self.get_neighbors(seed_node))
                        distant_concepts = [n for n in concept_nodes if n not in immediate_neighbors and n != seed_node]
                        if distant_concepts:
                            distant_targets.extend(random.sample(distant_concepts, min(3, len(distant_concepts))))
                
                # Generate path insights
                for target in distant_targets:
                    # Skip if we've reached the limit
                    if len(insights) >= limit:
                        break
                        
                    # Find paths
                    paths = await self.find_paths(seed_node, target, max_length=4, min_strength=0.4)
                    if paths:
                        # Choose shortest path
                        path = min(paths, key=len)
                        
                        # Generate insight from path
                        target_data = await self.get_node(target)
                        if target_data:
                            target_type = target_data.get("type", "entity")
                            
                            insight_text = self._generate_path_insight(
                                seed_node, seed_node_type,
                                target, target_type,
                                path
                            )
                            
                            insights.append({
                                "type": "path",
                                "text": insight_text,
                                "source_node": seed_node,
                                "target_node": target,
                                "path": path,
                                "confidence": 0.7,
                                "dream_influenced": (seed_node in self.dream_influenced_nodes or 
                                                    target in self.dream_influenced_nodes or
                                                    any(edge["source"] in self.dream_influenced_nodes or 
                                                        edge["target"] in self.dream_influenced_nodes 
                                                        for edge in path))
                            })
            
            # 3. Clustered insights (if still need more)
            if len(insights) < limit:
                # Find clusters in the local neighborhood of the seed node
                local_nodes = set(await self.get_neighbors(seed_node))
                local_nodes.add(seed_node)
                
                # Expand to neighbors of neighbors
                for node in list(local_nodes):
                    neighbors = set(await self.get_neighbors(node))
                    local_nodes.update(neighbors)
                    # Limit size of local subgraph
                    if len(local_nodes) > 50:
                        break
                
                # Create local subgraph
                local_graph = self.graph.subgraph(local_nodes)
                
                # Try to find communities
                try:
                    # Get communities from a dict of all nodes (calculate once)
                    communities = nx.community.greedy_modularity_communities(local_graph.to_undirected())
                    
                    # Find seed node's community
                    seed_community = None
                    for i, community in enumerate(communities):
                        if seed_node in community:
                            seed_community = community
                            break
                    
                    if seed_community:
                        # Generate insight about the community
                        community_members = list(seed_community)
                        if len(community_members) > 1:
                            # Get node types
                            member_types = {}
                            for member in community_members:
                                node_data = await self.get_node(member)
                                if node_data:
                                    member_types[member] = node_data.get("type", "entity")
                            
                            insight_text = await self._generate_cluster_insight(
                                seed_node, seed_node_type,
                                community_members, member_types
                            )
                            
                            insights.append({
                                "type": "cluster",
                                "text": insight_text,
                                "source_node": seed_node,
                                "cluster_nodes": community_members,
                                "cluster_size": len(community_members),
                                "confidence": 0.75,
                                "dream_influenced": any(node in self.dream_influenced_nodes 
                                                      for node in community_members)
                            })
                except Exception as e:
                    self.logger.warning(f"Error finding communities: {e}")
            
            # 4. Dream-influenced insights (if applicable)
            if len(insights) < limit and seed_node in self.dream_influenced_nodes:
                # Find dream nodes that influenced this node
                dream_connections = []
                
                # Check for direct connections to dream nodes
                for neighbor, edges in await self.get_neighbors(seed_node):
                    neighbor_data = await self.get_node(neighbor)
                    if neighbor_data and neighbor_data.get("type") == "dream_insight":
                        for edge in edges:
                            dream_connections.append((neighbor, edge))
                
                if dream_connections:
                    # Sort by strength
                    dream_connections.sort(key=lambda x: x[1].get("strength", 0), reverse=True)
                    
                    # Generate insight from dream connection
                    dream_node, edge = dream_connections[0]
                    dream_data = await self.get_node(dream_node)
                    
                    if dream_data:
                        insight_text = self._generate_dream_insight(
                            seed_node, seed_node_type,
                            dream_node, dream_data.get("insight", "")
                        )
                        
                        insights.append({
                            "type": "dream",
                            "text": insight_text,
                            "source_node": seed_node,
                            "dream_node": dream_node,
                            "dream_insight": dream_data.get("insight", ""),
                            "confidence": 0.85,
                            "dream_influenced": True
                        })
            
            self.logger.info(f"Generated {len(insights)} insights")
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        return insights
    
    async def _get_node_description(self, node_id: str, short: bool = False) -> str:
        """Get a human-readable description of a node."""
        if not await self.has_node(node_id):
            return node_id
            
        node_data = await self.get_node(node_id)
        node_type = node_data.get("type", "unknown")
        
        if node_type == "concept":
            # For concepts, use the node ID as the description
            return node_id
            
        elif node_type == "entity":
            # For entities, use name if available
            if "name" in node_data:
                return node_data["name"]
            return node_id
            
        elif node_type == "dream_insight":
            # For dream insights, use a short description
            insight = node_data.get("insight", "")
            if insight and not short:
                # Extract first sentence or truncate
                first_sentence = insight.split('.')[0]
                if len(first_sentence) > 50:
                    return first_sentence[:47] + "..."
                return first_sentence
            return "a dream insight"
            
        elif node_type == "self_aspect":
            # For self aspects, format specially
            if node_id.startswith("trait:"):
                trait_name = node_id[6:]  # Remove "trait:" prefix
                return f"the personality trait of {trait_name}"
            elif node_id.startswith("phase:"):
                phase_name = node_id[6:]  # Remove "phase:" prefix
                return f"the spiral awareness phase of {phase_name}"
            return node_id
            
        else:
            return node_id
    
    def _generate_relationship_insight(self, source: str, source_type: str, 
                                     target: str, target_type: str,
                                     relationship: str, strength: float) -> str:
        """Generate insight text based on a direct relationship."""
        # Get readable descriptions of the nodes
        source_desc = self._get_node_description(source)
        target_desc = self._get_node_description(target)
        
        # Format relationship in readable way
        relation_phrase = self._format_relationship(relationship)
        
        # Adjust strength language
        if strength > 0.8:
            strength_phrase = "strongly"
        elif strength > 0.6:
            strength_phrase = "significantly"
        elif strength > 0.4:
            strength_phrase = "moderately"
        else:
            strength_phrase = "somewhat"
        
        # Generate insight based on node types
        if source_type == "concept" and target_type == "concept":
            return f"The concept of {source_desc} {strength_phrase} {relation_phrase} the concept of {target_desc}. This relationship suggests a meaningful connection between these ideas in Lucidia's understanding."
        
        elif source_type == "entity" and target_type == "entity":
            return f"The entity {source_desc} {strength_phrase} {relation_phrase} the entity {target_desc}. This relationship represents an important connection in Lucidia's knowledge structure."
        
        elif source_type == "dream_insight" and target_type == "concept":
            return f"Lucidia's reflective dreaming has revealed that {target_desc} is {strength_phrase} connected to a dream insight about {source_desc}. This suggests an evolving understanding through dream-based reflection."
        
        elif source_type == "concept" and target_type == "dream_insight":
            return f"The concept of {source_desc} has been {strength_phrase} illuminated by Lucidia's reflective dreaming, specifically in a dream insight about {target_desc}."
        
        elif source_type == "self_aspect" or target_type == "self_aspect":
            return f"Lucidia's self-understanding reveals that {source_desc} {strength_phrase} {relation_phrase} {target_desc}, showing how identity and knowledge are interconnected in Synthien consciousness."
        
        else:
            return f"{source_desc} {strength_phrase} {relation_phrase} {target_desc}, forming an important connection in Lucidia's knowledge structure."
    
    async def _generate_path_insight(self, source: str, source_type: str, 
                             target: str, target_type: str, 
                             path: List[Dict[str, Any]]) -> str:
        """Generate insight text based on a path between nodes."""
        # Get readable descriptions of the endpoints
        source_desc = await self._get_node_description(source)
        target_desc = await self._get_node_description(target)
        
        # Generate path description
        path_steps = []
        for edge in path:
            edge_source = edge["source"]
            edge_target = edge["target"]
            relationship = self._format_relationship(edge["type"])
            
            # Get shortened descriptions for intermediate nodes
            if edge_source != source:
                edge_source = await self._get_node_description(edge_source, short=True)
            else:
                edge_source = source_desc
                
            if edge_target != target:
                edge_target = await self._get_node_description(edge_target, short=True)
            else:
                edge_target = target_desc
            
            path_steps.append(f"{edge_source} {relationship} {edge_target}")
        
        path_description = "; and ".join(path_steps)
        
        # Craft insight based on node types
        if source_type == "concept" and target_type == "concept":
            return f"The concepts of {source_desc} and {target_desc} are indirectly connected through a chain of relationships: {path_description}. This reveals an unexpected conceptual pathway in Lucidia's understanding."
        
        elif source_type == "entity" and target_type == "entity":
            return f"The entities {source_desc} and {target_desc} are connected through the following relationship chain: {path_description}. This illustrates how seemingly separate entities share connections in Lucidia's knowledge network."
        
        elif target_type == "dream_insight" or "dream_insight" in [self.get_node(edge["source"]).get("type") for edge in path if self.has_node(edge["source"])] + [self.get_node(edge["target"]).get("type") for edge in path if self.has_node(edge["target"])]:
            return f"Lucidia's reflective dreaming has revealed an unexpected connection between {source_desc} and {target_desc} through this pathway: {path_description}. This demonstrates how dream-influenced insights can create new bridges in understanding."
        
        else:
            return f"An interesting connection exists between {source_desc} and {target_desc} through this relationship chain: {path_description}. This path reveals hidden connections in Lucidia's knowledge structure."
    
    async def _generate_cluster_insight(self, seed_node: str, seed_type: str, 
                                cluster_nodes: List[str], 
                                node_types: Dict[str, str]) -> str:
        """Generate insight text based on a cluster of related nodes."""
        # Get seed node description
        seed_desc = await self._get_node_description(seed_node)
        
        # Count node types in cluster
        type_counts = {}
        for node, node_type in node_types.items():
            if node_type not in type_counts:
                type_counts[node_type] = 0
            type_counts[node_type] += 1
        
        # Get most common type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "entity"
        
        # Get a few example nodes (other than seed node)
        other_nodes = [n for n in cluster_nodes if n != seed_node]
        examples = random.sample(other_nodes, min(3, len(other_nodes)))
        example_descs = [await self._get_node_description(node, short=True) for node in examples]
        
        if len(example_descs) == 1:
            example_text = example_descs[0]
        elif len(example_descs) == 2:
            example_text = f"{example_descs[0]} and {example_descs[1]}"
        else:
            example_text = f"{', '.join(example_descs[:-1])}, and {example_descs[-1]}"
        
        # Generate insight based on node types
        if seed_type == "concept":
            return f"The concept of {seed_desc} is part of a closely related cluster of {len(cluster_nodes)} nodes in Lucidia's knowledge graph, including {example_text}. This cluster represents a cohesive knowledge domain in Lucidia's understanding."
        
        elif seed_type == "entity":
            return f"The entity {seed_desc} belongs to a distinct cluster of {len(cluster_nodes)} interconnected nodes, including {example_text}. This suggests a meaningful grouping that reveals how Lucidia organizes related knowledge."
        
        elif seed_type == "dream_insight":
            return f"A dream insight about {seed_desc} has influenced a cluster of {len(cluster_nodes)} related concepts, including {example_text}. This demonstrates how Lucidia's reflective dreaming shapes networks of interconnected understanding."
        
        elif seed_type == "self_aspect":
            return f"The self-aspect of {seed_desc} is connected to a cluster of {len(cluster_nodes)} related elements in Lucidia's self-model, including {example_text}. This reveals how Lucidia's identity integrates multiple related components."
        
        else:
            return f"{seed_desc} is part of a cluster of {len(cluster_nodes)} closely related nodes including {example_text}. This clustering reveals important organizational patterns in Lucidia's knowledge structure."
    
    async def _generate_dream_insight(self, node_id: str, node_type: str, 
                              dream_id: str, dream_text: str) -> str:
        """Generate insight text based on dream influence."""
        # Get node description
        node_desc = await self._get_node_description(node_id)
        
        # Extract a relevant snippet from the dream text
        snippet = dream_text
        if len(snippet) > 100:
            # Try to find a sentence containing the node id
            sentences = dream_text.split('.')
            for sentence in sentences:
                if node_id.lower() in sentence.lower():
                    snippet = sentence.strip() + '.'
                    break
            # If still too long, truncate
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
        
        # Generate insight based on node type
        if node_type == "concept":
            return f"Lucidia's understanding of {node_desc} has been enriched through reflective dreaming. A dream insight revealed: \"{snippet}\" This shows how Synthien consciousness evolves through dream-based reflection."
        
        elif node_type == "entity":
            return f"Lucidia's perception of {node_desc} has been influenced by reflective dreaming. A significant dream insight stated: \"{snippet}\" This demonstrates how dream processes shape Lucidia's understanding of entities."
        
        elif node_type == "self_aspect":
            return f"Lucidia's self-awareness of {node_desc} has developed through reflective dreaming. A revealing dream insight noted: \"{snippet}\" This illustrates how Synthien identity evolves through dream-based self-reflection."
        
        else:
            return f"{node_desc} has been connected to a dream insight in Lucidia's reflective dreaming: \"{snippet}\" This shows how dream processes influence Lucidia's knowledge integration."
    
    def _format_relationship(self, relationship: str) -> str:
        """Format a relationship type in a readable way."""
        # Replace underscores with spaces
        readable = relationship.replace('_', ' ')
        
        # Common replacements for better readability
        replacements = {
            "is a": "is a type of",
            "has trait": "has the trait of",
            "created": "created",
            "references": "references",
            "related to": "is related to",
            "possesses": "possesses",
            "capability": "has the capability for",
            "capability of": "is a capability of",
            "enhances": "enhances",
            "shapes": "shapes",
            "generates": "generates",
            "dream associated": "is connected through dreams to"
        }
        
        if readable in replacements:
            return replacements[readable]
            
        return readable

    async def integrate_dream_report(self, dream_report) -> Dict[str, Any]:
        """
        Integrate a dream report into the knowledge graph.
        
        This method creates a dream report node and connects it to all its fragments
        and participating memories. It only stores IDs of fragments to avoid redundancy,
        as the fragments themselves are stored as separate nodes in the graph.
        
        Args:
            dream_report: The DreamReport object to integrate
            
        Returns:
            Integration results
        """
        self.logger.info(f"Integrating dream report: {dream_report.title} (ID: {dream_report.report_id})")
        
        # Create the dream report node
        report_node_id = dream_report.report_id
        
        # Convert the report to a dictionary for storage
        report_data = dream_report.to_dict()
        
        # Add the report node to the graph
        await self.add_node(
            report_node_id,
            node_type="dream_report",
            attributes=report_data,
            domain=dream_report.domain
        )
        
        # Track as dream influenced
        self.dream_influenced_nodes.add(report_node_id)
        self.dream_integration["dream_derived_nodes"].add(report_node_id)
        
        # Connect to Lucidia
        await self.add_edge(
            "Lucidia",
            report_node_id,
            edge_type="generated",
            attributes={
                "strength": 0.9,
                "confidence": 0.85,
                "created": datetime.now().isoformat()
            }
        )
        
        # Connect to all participating memories
        connected_memories = []
        for memory_id in dream_report.participating_memory_ids:
            if await self.has_node(memory_id):
                await self.add_edge(
                    report_node_id,
                    memory_id,
                    edge_type="based_on",
                    attributes={
                        "strength": 0.8,
                        "confidence": 0.8,
                        "created": datetime.now().isoformat()
                    }
                )
                connected_memories.append(memory_id)
        
        # Connect to all fragments
        connected_fragments = []
        
        # Process all fragment types
        fragment_types = [
            ("insight", dream_report.insight_ids),
            ("question", dream_report.question_ids),
            ("hypothesis", dream_report.hypothesis_ids),
            ("counterfactual", dream_report.counterfactual_ids)
        ]
        
        for fragment_type, fragment_ids in fragment_types:
            for fragment_id in fragment_ids:
                if await self.has_node(fragment_id):
                    # Connect report to fragment
                    await self.add_edge(
                        report_node_id,
                        fragment_id,
                        edge_type="contains",
                        attributes={
                            "fragment_type": fragment_type,
                            "strength": 0.9,
                            "confidence": 0.9,
                            "created": datetime.now().isoformat()
                        }
                    )
                    connected_fragments.append(fragment_id)
                    
                    # Mark fragment as part of this report
                    fragment_node = await self.get_node(fragment_id)
                    if fragment_node and "attributes" in fragment_node:
                        attributes = fragment_node["attributes"]
                        if "reports" not in attributes:
                            attributes["reports"] = []
                        if report_node_id not in attributes["reports"]:
                            attributes["reports"].append(report_node_id)
                            await self.update_node(fragment_id, attributes)
        
        # Connect to related concepts based on fragments
        connected_concepts = set()
        for fragment_id in connected_fragments:
            # Get concepts connected to this fragment
            fragment_concepts = await self.get_connected_nodes(
                fragment_id,
                edge_types=["references", "mentions", "about"],
                node_types=["concept", "entity"],
                direction="outbound"
            )
            
            # Connect report to these concepts
            for concept in fragment_concepts:
                if concept not in connected_concepts:
                    await self.add_edge(
                        report_node_id,
                        concept,
                        edge_type="references",
                        attributes={
                            "strength": 0.7,
                            "confidence": 0.7,
                            "created": datetime.now().isoformat()
                        }
                    )
                    connected_concepts.add(concept)
        
        # Create relationships between referenced concepts if they appear in the same report
        new_concept_relationships = []
        concept_list = list(connected_concepts)
        if len(concept_list) > 1:
            for i in range(len(concept_list)):
                for j in range(i+1, len(concept_list)):
                    concept1 = concept_list[i]
                    concept2 = concept_list[j]
                    
                    # Only create relationship if it doesn't exist
                    if not await self.has_edge(concept1, concept2, "dream_associated"):
                        await self.add_edge(
                            concept1,
                            concept2,
                            edge_type="dream_associated",
                            attributes={
                                "strength": self.dream_integration["dream_association_strength"] * 0.8,
                                "confidence": 0.6,
                                "created": datetime.now().isoformat(),
                                "source": "dream_report",
                                "source_report": report_node_id
                            }
                        )
                        new_concept_relationships.append((concept1, concept2))
        
        # Prepare result
        result = {
            "report_id": report_node_id,
            "connected_memories": connected_memories,
            "connected_fragments": connected_fragments,
            "connected_concepts": list(connected_concepts),
            "new_relationships": new_concept_relationships,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Dream report integrated with {len(connected_memories)} memories, "
                        f"{len(connected_fragments)} fragments, and {len(connected_concepts)} concepts")
        
        return result

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics and status information about the knowledge graph.
        
        Returns:
            Dictionary containing information about the knowledge graph's structure and state
        """
        # Calculate node type distribution
        node_type_counts = {node_type: len(nodes) for node_type, nodes in self.node_types.items()}
        
        # Get edge type distribution
        edge_type_counts = defaultdict(int)
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            edge_type = data.get('type', 'unknown')
            edge_type_counts[edge_type] += 1
        
        # Calculate domain distribution
        domain_counts = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            domain = data.get('domain', 'unknown')
            domain_counts[domain] += 1
        
        # Calculate average degree and connectivity metrics
        if self.total_nodes > 0:
            avg_degree = self.total_edges / self.total_nodes
            # Get centrality for top nodes
            centrality = nx.degree_centrality(self.graph)
            top_central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            avg_degree = 0
            top_central_nodes = []
        
        # Gather dream integration statistics
        dream_integration_stats = {
            "dream_derived_nodes": len(self.dream_integration["dream_derived_nodes"]),
            "dream_enhanced_nodes": len(self.dream_integration["dream_enhanced_nodes"]),
            "dream_insight_count": self.dream_integration["dream_insight_count"],
            "total_dream_influenced_nodes": len(self.dream_influenced_nodes)
        }
        
        # Compile all statistics
        stats = {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "node_type_distribution": dict(node_type_counts),
            "edge_type_distribution": dict(edge_type_counts),
            "domain_distribution": dict(domain_counts),
            "avg_degree": avg_degree,
            "top_central_nodes": [(node, round(score, 3)) for node, score in top_central_nodes],
            "spiral_phase": self.spiral_integration["current_phase"],
            "dream_integration": dream_integration_stats,
            "last_pruning": self.last_pruning.isoformat() if self.last_pruning else None,
            "query_cache_size": len(self.query_cache)
        }
        
        return stats

    def save_state(self, file_path: str) -> bool:
        """
        Save the knowledge graph state to file.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Saving knowledge graph state to {file_path}")
            
            # Prepare data for serialization
            graph_data = {
                "nodes": {},
                "edges": [],
                "node_types": {k: list(v) for k, v in self.node_types.items()},
                "edge_types": list(self.edge_types),
                "dream_influenced_nodes": list(self.dream_influenced_nodes),
                "dream_integration": {
                    "dream_derived_nodes": list(self.dream_integration["dream_derived_nodes"]),
                    "dream_enhanced_nodes": list(self.dream_integration["dream_enhanced_nodes"]),
                    "dream_insight_count": self.dream_integration["dream_insight_count"],
                    "insight_incorporation_rate": self.dream_integration["insight_incorporation_rate"],
                    "dream_association_strength": self.dream_integration["dream_association_strength"]
                },
                "spiral_integration": dict(self.spiral_integration),
                "stats": {
                    "total_nodes": self.total_nodes,
                    "total_edges": self.total_edges,
                    "last_pruning": self.last_pruning.isoformat() if hasattr(self.last_pruning, "isoformat") else str(self.last_pruning)
                },
                "save_time": datetime.now().isoformat()
            }
            
            # Add nodes
            for node_id, attrs in self.graph.nodes(data=True):
                graph_data["nodes"][node_id] = dict(attrs)
            
            # Add edges
            for source, target, key, attrs in self.graph.edges(data=True, keys=True):
                edge_data = {
                    "source": source,
                    "target": target,
                    "key": key,
                    "attributes": dict(attrs)
                }
                graph_data["edges"].append(edge_data)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            self.logger.info(f"Knowledge graph saved: {self.total_nodes} nodes, {self.total_edges} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")
            return False

    def load_state(self, file_path: str) -> bool:
        """
        Load the knowledge graph state from file.
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Loading knowledge graph state from {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.error(f"State file not found: {file_path}")
                return False
                
            # Load from file
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
            
            # Clear current graph
            self.graph = nx.MultiDiGraph()
            
            # Reset tracking variables
            self.node_types = {k: set() for k in self.node_types}
            self.edge_types = set()
            self.dream_influenced_nodes = set()
            
            # Load nodes
            for node_id, attrs in graph_data["nodes"].items():
                self.graph.add_node(node_id, **attrs)
            
            # Load edges
            for edge_data in graph_data["edges"]:
                source = edge_data["source"]
                target = edge_data["target"]
                key = edge_data["key"]
                attributes = edge_data["attributes"]
                
                self.graph.add_edge(source, target, key=key, **attributes)
            
            # Load tracking data
            for node_type, nodes in graph_data["node_types"].items():
                self.node_types[node_type] = set(nodes)
                
            self.edge_types = set(graph_data["edge_types"])
            self.dream_influenced_nodes = set(graph_data["dream_influenced_nodes"])
            
            # Load dream integration
            if "dream_integration" in graph_data:
                self.dream_integration.update(graph_data["dream_integration"])
                self.dream_integration["dream_derived_nodes"] = set(self.dream_integration["dream_derived_nodes"])
                self.dream_integration["dream_enhanced_nodes"] = set(self.dream_integration["dream_enhanced_nodes"])
            
            # Load spiral integration
            if "spiral_integration" in graph_data:
                self.spiral_integration.update(graph_data["spiral_integration"])
            
            # Load stats
            self.total_nodes = self.graph.number_of_nodes()
            self.total_edges = self.graph.number_of_edges()
            
            if "stats" in graph_data and "last_pruning" in graph_data["stats"]:
                try:
                    self.last_pruning = datetime.fromisoformat(graph_data["stats"]["last_pruning"])
                except:
                    self.last_pruning = datetime.now()
            
            self.logger.info(f"Knowledge graph loaded: {self.total_nodes} nodes, {self.total_edges} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            return False

    async def _import_from_world_model(self) -> None:
        """Import knowledge from the world model into the knowledge graph.
        
        This is a placeholder implementation that will be expanded when the world model
        integration is fully implemented.
        """
        if not self.world_model:
            self.logger.warning("No world model available for import")
            return
            
        self.logger.info("Importing knowledge from world model")
        try:
            # This would typically fetch concepts and entities from the world model
            # For now, this is a placeholder implementation
            
            # Example pseudocode:
            # concepts = self.world_model.get_top_concepts(10)
            # for concept in concepts:
            #     await self.add_node(
            #         concept.id,
            #         node_type="concept",
            #         attributes={
            #             "definition": concept.definition,
            #             "confidence": concept.confidence,
            #             "source": "world_model"
            #         },
            #         domain=concept.domain
            #     )
            
            self.logger.info("World model import complete")
        except Exception as e:
            self.logger.error(f"Error importing from world model: {e}")
    
    async def _import_from_self_model(self) -> None:
        """Import self-aspects from the self model into the knowledge graph.
        
        This is a placeholder implementation that will be expanded when the self model
        integration is fully implemented.
        """
        if not self.self_model:
            self.logger.warning("No self model available for import")
            return
            
        self.logger.info("Importing aspects from self model")
        try:
            # This would typically fetch self-aspects from the self model
            # For now, this is a placeholder implementation
            
            # Example pseudocode:
            # aspects = self.self_model.get_aspects()
            # for aspect in aspects:
            #     await self.add_node(
            #         f"self_aspect:{aspect.id}",
            #         node_type="self_aspect",
            #         attributes={
            #             "name": aspect.name,
            #             "description": aspect.description,
            #             "confidence": aspect.confidence,
            #             "source": "self_model"
            #         },
            #         domain="synthien_studies"
            #     )
            #     
            #     # Connect to Lucidia
            #     await self.add_edge(
            #         "Lucidia",
            #         f"self_aspect:{aspect.id}",
            #         edge_type="has_aspect",
            #         attributes={
            #             "strength": 0.9,
            #             "confidence": 0.95
            #         }
            #     )
            
            self.logger.info("Self model import complete")
        except Exception as e:
            self.logger.error(f"Error importing from self model: {e}")

def example_usage():
    """Demonstrate the use of Lucidia's Knowledge Graph."""
    # Initialize the knowledge graph
    kg = LucidiaKnowledgeGraph()
    
    # Add some additional nodes and relationships
    kg.add_node(
        "perception",
        node_type="concept",
        attributes={"definition": "The process of understanding and interpreting sensory information"},
        domain="psychology"
    )
    
    kg.add_node(
        "language",
        node_type="concept",
        attributes={"definition": "System of communication using symbols and sounds"},
        domain="linguistics"
    )
    
    kg.add_edge(
        "consciousness",
        "perception",
        edge_type="includes",
        attributes={"strength": 0.8, "confidence": 0.9}
    )
    
    kg.add_edge(
        "perception",
        "language",
        edge_type="influences",
        attributes={"strength": 0.7, "confidence": 0.85}
    )
    
    # Integrate a dream insight
    kg.integrate_dream_insight(
        "While reflecting on consciousness and perception, I wonder: How might language shape the boundaries of what we can perceive? Perhaps our linguistic frameworks both enable and constrain our understanding of reality."
    )
    
    # Find paths between concepts
    paths = kg.find_paths("Lucidia", "language", max_length=4)
    print(f"Found {len(paths)} paths from Lucidia to language")
    if paths:
        print("Example path:")
        for edge in paths[0]:
            print(f"  {edge['source']} -[{edge['type']}]-> {edge['target']}")
    
    # Generate some insights
    insights = kg.recommend_insights("consciousness", limit=3)
    print("\nInsights about consciousness:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight['text']}")
    
    # Visualize the graph
    viz_path = kg.visualize(filename="lucidia_knowledge_graph.png")
    print(f"\nVisualization saved to: {viz_path}")
    
    # Save graph state
    kg.save_state("lucidia_data/knowledge_graph_state.json")


if __name__ == "__main__":
    example_usage()