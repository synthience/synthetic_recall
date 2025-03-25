"""
Visualization Manager Module for Lucidia's Knowledge Graph

This module implements visualization capabilities for the knowledge graph,
focusing on Mermaid diagram generation for different views of the graph.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict

from .base_module import KnowledgeGraphModule

class VisualizationManager(KnowledgeGraphModule):
    """
    Visualization Manager responsible for knowledge graph visualization.
    
    This module generates different visualizations of the knowledge graph,
    with a focus on Mermaid diagram generation.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Visualization Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # Styling configuration
        self.domain_colors = self.get_config('domain_colors', {
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
        })
        
        self.emotion_colors = self.get_config('emotion_colors', {
            "joy": "#FFD700",  # Gold
            "sadness": "#4682B4",  # Steel Blue
            "anger": "#FF4500",  # Red Orange
            "fear": "#800080",  # Purple
            "disgust": "#006400",  # Dark Green
            "surprise": "#FF69B4",  # Hot Pink
            "neutral": "#CCCCCC"  # Light Gray
        })
        
        self.node_styles = self.get_config('node_styles', {
            "concept": "fill:#f9f,stroke:#333,stroke-width:2px",
            "entity": "fill:#bbf,stroke:#333,stroke-width:2px",
            "dream_insight": "fill:#bfb,stroke:#333,stroke-width:2px",
            "attribute": "fill:#fbb,stroke:#333,stroke-width:1px",
            "memory": "fill:#bff,stroke:#333,stroke-width:2px",
            "self_aspect": "fill:#fbf,stroke:#333,stroke-width:2px",
            "event": "fill:#ffb,stroke:#333,stroke-width:2px",
            "domain": "fill:#ccc,stroke:#333,stroke-width:2px",
            "dream_report": "fill:#beb,stroke:#333,stroke-width:2px",
            "value": "fill:#caf,stroke:#333,stroke-width:2px",
            "goal": "fill:#fca,stroke:#333,stroke-width:2px",
            "emotion": "fill:#fcf,stroke:#333,stroke-width:2px"
        })
        
        # Maximum nodes to include in visualizations
        self.max_nodes = self.get_config('max_nodes', 200)
        
        # Default depth for subgraph visualization
        self.default_depth = self.get_config('default_depth', 2)
        
        self.logger.info("Visualization Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("visualization_requested", self._handle_visualization_request)
        await self.event_bus.subscribe("node_added", self._handle_node_change)
        await self.event_bus.subscribe("node_updated", self._handle_node_change)
        await self.event_bus.subscribe("node_removed", self._handle_node_change)
        await self.event_bus.subscribe("edge_added", self._handle_edge_change)
        await self.event_bus.subscribe("edge_removed", self._handle_edge_change)
        self.logger.info("Subscribed to visualization-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # No special setup needed
        pass
    
    async def _handle_visualization_request(self, data):
        """
        Handle visualization request events.
        
        Args:
            data: Visualization request data
            
        Returns:
            Visualization result
        """
        vis_type = data.get("type", "full")
        
        if vis_type == "full":
            result = await self.generate_full_graph_diagram(
                include_attributes=data.get("include_attributes", False)
            )
        elif vis_type == "domain":
            result = await self.generate_domain_diagram(
                data.get("domain", "general_knowledge"),
                include_attributes=data.get("include_attributes", False)
            )
        elif vis_type == "concept":
            result = await self.generate_concept_network(
                data.get("central_concept"),
                depth=data.get("depth", self.default_depth)
            )
        elif vis_type == "subgraph":
            result = await self.generate_subgraph_diagram(
                data.get("central_node"),
                node_types=data.get("node_types"),
                depth=data.get("depth", self.default_depth)
            )
        else:
            result = f"graph TD\n    error[\"Unknown visualization type: {vis_type}\"]"
        
        return {
            "visualization_type": vis_type,
            "mermaid_code": result
        }
    
    async def _handle_node_change(self, data):
        """
        Handle node change events (add, update, remove).
        
        Args:
            data: Node change event data
        """
        # Cache invalidation could be implemented here if needed
        pass
    
    async def _handle_edge_change(self, data):
        """
        Handle edge change events (add, remove).
        
        Args:
            data: Edge change event data
        """
        # Cache invalidation could be implemented here if needed
        pass
    
    async def generate_full_graph_diagram(self, include_attributes=False):
        """
        Generate a Mermaid diagram for the full knowledge graph.
        
        Args:
            include_attributes: Whether to include node attributes in the diagram
            
        Returns:
            Mermaid diagram code
        """
        self.logger.info("Generating full graph diagram")
        
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return "graph TD\n    error[\"Core graph module not found\"]"
        
        # Check if graph is too large
        if core_graph.total_nodes > self.max_nodes:
            self.logger.warning(f"Graph too large for full visualization ({core_graph.total_nodes} nodes)")
            return self._generate_too_large_message(core_graph.total_nodes)
        
        # Generate for all nodes and edges
        return await self.generate_mermaid_diagram(include_attributes=include_attributes)
    
    async def generate_domain_diagram(self, domain, include_attributes=False):
        """
        Generate a Mermaid diagram for nodes in a specific domain.
        
        Args:
            domain: Domain to visualize
            include_attributes: Whether to include node attributes
            
        Returns:
            Mermaid diagram code
        """
        self.logger.info(f"Generating diagram for domain: {domain}")
        
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return "graph TD\n    error[\"Core graph module not found\"]"
        
        # Get nodes in the domain
        domain_nodes = await core_graph.get_nodes_by_domain(domain)
        
        # Check if result set is too large
        if len(domain_nodes) > self.max_nodes:
            self.logger.warning(f"Domain too large for visualization ({len(domain_nodes)} nodes)")
            return self._generate_too_large_message(len(domain_nodes), f"domain '{domain}'")
        
        # Get edges between domain nodes
        domain_edges = []
        for source_id in domain_nodes:
            for target_id in domain_nodes:
                if await core_graph.has_edge(source_id, target_id):
                    edges = await core_graph.get_edges(source_id, target_id)
                    for edge in edges:
                        domain_edges.append((source_id, target_id, edge["key"]))
        
        return await self.generate_mermaid_diagram(
            nodes=domain_nodes,
            edges=domain_edges,
            include_attributes=include_attributes
        )
    
    async def generate_concept_network(self, central_concept, depth=2):
        """
        Generate a Mermaid diagram centered on a specific concept.
        
        Args:
            central_concept: Central concept to visualize
            depth: Depth of connections to include
            
        Returns:
            Mermaid diagram code
        """
        self.logger.info(f"Generating concept network for: {central_concept}")
        
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return "graph TD\n    error[\"Core graph module not found\"]"
        
        # Check if central concept exists
        if not await core_graph.has_node(central_concept):
            self.logger.error(f"Central concept '{central_concept}' not found")
            return f"graph TD\n    error[\"Concept '{central_concept}' not found\"]"
        
        # Generate subgraph focusing on concept nodes
        return await self.generate_subgraph_diagram(
            central_concept,
            node_types=["concept"],
            depth=depth
        )
    
    async def generate_subgraph_diagram(self, central_node, node_types=None, depth=2):
        """
        Generate a Mermaid diagram for a subgraph around a central node.
        
        Args:
            central_node: Central node to visualize
            node_types: Optional list of node types to include
            depth: Depth of connections to include
            
        Returns:
            Mermaid diagram code
        """
        self.logger.info(f"Generating subgraph diagram for: {central_node}")
        
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return "graph TD\n    error[\"Core graph module not found\"]"
        
        # Check if central node exists
        if not await core_graph.has_node(central_node):
            self.logger.error(f"Central node '{central_node}' not found")
            return f"graph TD\n    error[\"Node '{central_node}' not found\"]"
        
        # Build subgraph through breadth-first traversal
        subgraph_nodes = {central_node}
        current_frontier = {central_node}
        
        for _ in range(depth):
            next_frontier = set()
            for node in current_frontier:
                # Get connected nodes
                connected = await core_graph.get_connected_nodes(
                    node, node_types=node_types, direction="both"
                )
                next_frontier.update(connected)
            
            # Update subgraph nodes
            subgraph_nodes.update(next_frontier)
            
            # Check if subgraph is getting too large
            if len(subgraph_nodes) > self.max_nodes:
                self.logger.warning(f"Subgraph too large for visualization ({len(subgraph_nodes)} nodes)")
                return self._generate_too_large_message(len(subgraph_nodes), f"subgraph around '{central_node}'")
            
            # Update frontier for next iteration
            current_frontier = next_frontier - subgraph_nodes
        
        # Get edges between subgraph nodes
        subgraph_edges = []
        for source_id in subgraph_nodes:
            for target_id in subgraph_nodes:
                if await core_graph.has_edge(source_id, target_id):
                    edges = await core_graph.get_edges(source_id, target_id)
                    for edge in edges:
                        subgraph_edges.append((source_id, target_id, edge["key"]))
        
        return await self.generate_mermaid_diagram(
            nodes=subgraph_nodes,
            edges=subgraph_edges,
            include_attributes=True,
            highlight_node=central_node
        )
    
    async def generate_mermaid_diagram(self, nodes=None, edges=None, include_attributes=False, highlight_node=None):
        """
        Generate a Mermaid diagram for the specified nodes and edges.
        
        Args:
            nodes: List of node IDs to include (None for all)
            edges: List of (source, target, key) tuples to include (None for all)
            include_attributes: Whether to include node attributes
            highlight_node: Optional node to highlight
            
        Returns:
            Mermaid diagram code
        """
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return "graph TD\n    error[\"Core graph module not found\"]"
        
        # Start building Mermaid diagram
        mermaid_code = "graph TD\n"
        
        # Process nodes
        processed_nodes = set()
        
        # Group nodes by domain for subgraph organization
        domain_nodes = defaultdict(list)
        
        # Get node list
        node_list = list(nodes) if nodes is not None else list(core_graph.graph.nodes())
        
        # Prepare node data
        for node_id in node_list:
            if node_id not in core_graph.graph.nodes:
                continue
                
            node_data = core_graph.graph.nodes[node_id]
            domain = node_data.get("domain", "general_knowledge")
            
            # Add to domain grouping
            domain_nodes[domain].append(node_id)
            processed_nodes.add(node_id)
        
        # Generate domain subgraphs
        for domain, domain_node_list in domain_nodes.items():
            if len(domain_node_list) > 0:
                # Sanitize domain name for Mermaid
                safe_domain = domain.replace(" ", "_")
                mermaid_code += f"    subgraph {safe_domain}\n"
                
                # Generate nodes in this domain
                for node_id in domain_node_list:
                    node_data = core_graph.graph.nodes[node_id]
                    node_type = node_data.get("type", "unknown")
                    
                    # Determine display label
                    if node_type == "concept" and "definition" in node_data:
                        display_name = f"{node_id}: {node_data['definition'][:30]}..."
                    elif "name" in node_data:
                        display_name = f"{node_id}: {node_data['name']}"
                    else:
                        display_name = node_id
                    
                    # Truncate if too long
                    if len(display_name) > 50:
                        display_name = display_name[:47] + "..."
                    
                    # Add attributes if requested
                    if include_attributes:
                        # Add confidence if available
                        if "confidence" in node_data:
                            display_name += f"<br/>(conf: {node_data['confidence']:.2f})"
                        
                        # Add emotional context if available
                        if "emotional_context" in node_data and "dominant_emotion" in node_data["emotional_context"]:
                            emotion = node_data["emotional_context"]["dominant_emotion"]
                            display_name += f"<br/>({emotion})"
                    
                    # Special highlighting for central node
                    if highlight_node and node_id == highlight_node:
                        mermaid_code += f"        {node_id}[\"<b>{display_name}</b>\"]::highlighted\n"
                    else:
                        mermaid_code += f"        {node_id}[\"{display_name}\"]::type{node_type}\n"
                
                mermaid_code += "    end\n"
        
        # Process edges
        processed_edges = 0
        
        # Get edge list
        if edges is not None:
            edge_list = edges
        else:
            edge_list = []
            for u, v, k in core_graph.graph.edges(keys=True):
                if u in processed_nodes and v in processed_nodes:
                    edge_list.append((u, v, k))
        
        # Generate edges
        for source, target, key in edge_list:
            if source not in processed_nodes or target not in processed_nodes:
                continue
                
            if not core_graph.graph.has_edge(source, target, key):
                continue
                
            edge_data = core_graph.graph.edges[source, target, key]
            edge_type = edge_data.get("type", "related_to")
            
            # Add edge
            mermaid_code += f"    {source} -->|{edge_type}| {target}\n"
            processed_edges += 1
            
            # Add edge styling based on strength if available
            if "strength" in edge_data:
                strength = edge_data["strength"]
                if strength > 0.8:
                    thickness = 3
                elif strength > 0.5:
                    thickness = 2
                else:
                    thickness = 1
                
                mermaid_code += f"    linkStyle {processed_edges - 1} stroke-width:{thickness}px\n"
        
        # Add style definitions
        for node_type, style in self.node_styles.items():
            mermaid_code += f"    classDef type{node_type} {style}\n"
        
        # Add highlighted style
        mermaid_code += "    classDef highlighted fill:#ff5733,stroke:#333,stroke-width:3px\n"
        
        return mermaid_code
    
    def _generate_too_large_message(self, node_count, context="graph"):
        """
        Generate a message for when a graph is too large to visualize.
        
        Args:
            node_count: Number of nodes in the graph
            context: Context description for the message
            
        Returns:
            Mermaid diagram code with warning message
        """
        return f"""graph TD
    warning["The {context} is too large to visualize<br/>{node_count} nodes (max {self.max_nodes})"]
    recommendations["Try one of these approaches:"]
    recommendation1["1. Visualize a specific domain"]
    recommendation2["2. Generate a subgraph around a specific node"]
    recommendation3["3. Increase max_nodes in configuration"]
    
    warning --> recommendations
    recommendations --> recommendation1
    recommendations --> recommendation2
    recommendations --> recommendation3
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef warning fill:#ff9800,stroke:#333,stroke-width:2px
    
    class warning warning
"""
    
    async def get_visualization_stats(self):
        """
        Get statistics about visualizations.
        
        Returns:
            Dictionary with visualization statistics
        """
        # Get the core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"error": "Core graph module not found"}
        
        # Get graph metrics
        metrics = await core_graph.get_graph_metrics()
        
        # Get domain statistics
        domain_stats = {}
        for domain, nodes in core_graph.domains.items():
            domain_stats[domain] = len(nodes)
        
        # Get node type statistics
        node_type_stats = {}
        for node_type, nodes in core_graph.node_types.items():
            node_type_stats[node_type] = len(nodes)
        
        # Get edge type statistics
        edge_type_stats = {}
        for edge_type, edges in core_graph.edges_by_type.items():
            edge_type_stats[edge_type] = len(edges)
        
        return {
            "total_nodes": metrics["total_nodes"],
            "total_edges": metrics["total_edges"],
            "domains": domain_stats,
            "node_types": node_type_stats,
            "edge_types": edge_type_stats,
            "visualization_limit": self.max_nodes,
            "can_visualize_full_graph": metrics["total_nodes"] <= self.max_nodes
        }