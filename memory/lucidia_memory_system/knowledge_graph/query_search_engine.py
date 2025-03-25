"""
Query Search Engine Module for Lucidia's Knowledge Graph

This module implements semantic search, path finding, relevance ranking,
and context generation capabilities.
"""

import logging
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, deque
import heapq
import numpy as np

from .base_module import KnowledgeGraphModule

class QuerySearchEngine(KnowledgeGraphModule):
    """
    Query Search Engine responsible for search and query operations.
    
    This module provides semantic search functionality, path finding between
    knowledge nodes, relevance ranking algorithms, and context generation for queries.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Query Search Engine."""
        super().__init__(event_bus, module_registry, config)
        
        # Query configuration
        self.search_config = {
            "default_limit": self.get_config("default_limit", 10),
            "default_threshold": self.get_config("default_threshold", 0.5),
            "use_embeddings": self.get_config("use_embeddings", True),
            "use_hyperbolic": self.get_config("use_hyperbolic", False),
            "search_boost": {
                "concept": self.get_config("concept_boost", 1.2),
                "entity": self.get_config("entity_boost", 1.1),
                "dream_insight": self.get_config("dream_boost", 1.3)
            }
        }
        
        # Path finding configuration
        self.path_finding = {
            "max_depth": self.get_config("max_depth", 5),
            "min_confidence": self.get_config("min_confidence", 0.3),
            "relevance_emphasis": self.get_config("relevance_emphasis", 0.7),
            "use_hyperbolic": self.get_config("path_use_hyperbolic", False),
            "exploration_factor": self.get_config("exploration_factor", 0.2)
        }
        
        # Cache for frequent queries
        self.query_cache = {}
        self.query_stats = defaultdict(int)
        self.max_cache_size = self.get_config("max_cache_size", 1000)
        
        # Statistics
        self.search_stats = {
            "total_searches": 0,
            "total_path_finding": 0,
            "cache_hits": 0,
            "embedding_searches": 0,
            "text_searches": 0,
            "avg_search_time_ms": 0
        }
        
        self.logger.info("Query Search Engine initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("search_requested", self._handle_search_request)
        await self.event_bus.subscribe("path_finding_requested", self._handle_path_request)
        await self.event_bus.subscribe("relevance_requested", self._handle_relevance_request)
        await self.event_bus.subscribe("context_generation_requested", self._handle_context_request)
        self.logger.info("Subscribed to query-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # No special setup needed
        pass
    
    async def _handle_search_request(self, data):
        """
        Handle search request events.
        
        Args:
            data: Search request data
            
        Returns:
            Search results
        """
        query = data.get("query")
        limit = data.get("limit", self.search_config["default_limit"])
        threshold = data.get("threshold", self.search_config["default_threshold"])
        include_metadata = data.get("include_metadata", True)
        domains = data.get("domains")
        query_embedding = data.get("query_embedding")
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        results = await self.search_nodes(
            query,
            limit,
            threshold,
            include_metadata,
            domains,
            query_embedding
        )
        
        return results
    
    async def _handle_path_request(self, data):
        """
        Handle path finding request events.
        
        Args:
            data: Path request data
            
        Returns:
            Path results
        """
        source = data.get("source")
        target = data.get("target")
        max_depth = data.get("max_depth", self.path_finding["max_depth"])
        min_confidence = data.get("min_confidence", self.path_finding["min_confidence"])
        use_hyperbolic = data.get("use_hyperbolic", self.path_finding["use_hyperbolic"])
        
        if not source or not target:
            return {"success": False, "error": "Source and target required"}
        
        results = await self.find_paths(
            source,
            target,
            max_depth,
            min_confidence,
            use_hyperbolic
        )
        
        return results
    
    async def _handle_relevance_request(self, data):
        """
        Handle relevance request events.
        
        Args:
            data: Relevance request data
            
        Returns:
            Relevance results
        """
        query = data.get("query")
        context_size = data.get("context_size", 5)
        include_related = data.get("include_related", True)
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        results = await self.get_most_relevant_nodes(
            query,
            context_size,
            include_related
        )
        
        return results
    
    async def _handle_context_request(self, data):
        """
        Handle context generation request events.
        
        Args:
            data: Context request data
            
        Returns:
            Generated context
        """
        query = data.get("query")
        max_tokens = data.get("max_tokens", 512)
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        context = await self.generate_context(query, max_tokens)
        
        return {"success": True, "context": context}
    
    async def search_nodes(self, query: str, limit: int = 10, threshold: float = 0.5, 
                      include_metadata: bool = True, domains: Optional[List[str]] = None,
                      query_embedding: Optional[List[float]] = None, use_hyperbolic: bool = False) -> Dict[str, Any]:
        """
        Search for nodes in the knowledge graph based on a semantic query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            include_metadata: Whether to include node metadata in results
            domains: Optional list of domains to search within
            query_embedding: Optional pre-computed embedding for the query
            use_hyperbolic: Whether to use hyperbolic embeddings for similarity calculations
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(f"Searching for nodes with query: {query}, limit: {limit}")
        
        # Update statistics
        self.search_stats["total_searches"] += 1
        
        # Check cache
        cache_key = f"{query}:{limit}:{threshold}:{str(domains)}"
        if cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            self.query_stats[query] += 1
            return self.query_cache[cache_key]
        
        # Initialize results
        results = {
            "success": True,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": [],
            "metadata": {
                "total_nodes_searched": 0,
                "search_time_ms": 0,
                "threshold": threshold,
                "using_embeddings": False,
                "using_hyperbolic": False
            }
        }
        start_time = time.time()
        
        try:
            # Get core graph for node access
            core_graph = self.module_registry.get_module("core_graph")
            if not core_graph:
                self.logger.error("Core graph module not found")
                return {"success": False, "error": "Core graph module not found"}
            
            # Get embedding manager for similarity calculations
            embedding_manager = self.module_registry.get_module("embedding_manager")
            use_embeddings = self.search_config["use_embeddings"] and embedding_manager is not None
            use_hyperbolic = use_hyperbolic and use_embeddings
            
            # Process query embedding if provided or generate one
            query_emb = None
            if use_embeddings:
                if query_embedding is not None:
                    # Use provided embedding
                    query_emb = query_embedding
                    self.search_stats["embedding_searches"] += 1
                elif embedding_manager:
                    # Generate embedding for query
                    try:
                        # Generate text embedding
                        query_emb = embedding_manager._generate_embedding_from_text(query).tolist()
                        self.search_stats["embedding_searches"] += 1
                    except Exception as e:
                        self.logger.error(f"Error generating query embedding: {e}")
                        use_embeddings = False
            
            # Get all nodes that match the search criteria
            matching_nodes = []
            total_nodes = 0
            
            # Get node list
            if domains:
                # Get nodes from specified domains
                node_ids = set()
                for domain in domains:
                    domain_nodes = await core_graph.get_nodes_by_domain(domain)
                    node_ids.update(domain_nodes)
            else:
                # Get all nodes
                node_ids = core_graph.graph.nodes()
            
            # Process each node
            for node_id in node_ids:
                total_nodes += 1
                
                # Get node data
                node_data = await core_graph.get_node(node_id)
                if not node_data:
                    continue
                
                # Calculate similarity
                similarity = 0.0
                
                if use_embeddings and query_emb is not None:
                    # Use embedding-based similarity if available
                    if embedding_manager:
                        if node_id in node_data:
                            # It's a node ID, calculate node similarity
                            similarity = await embedding_manager.calculate_node_similarity(
                                node_id, query, use_hyperbolic
                            )
                        else:
                            # Extract text for similarity calculation
                            node_text = self._extract_node_text(node_data)
                            
                            # Calculate similarity between embeddings
                            if node_data.get("embedding") is not None:
                                node_emb = node_data["embedding"]
                                similarity = embedding_manager.calculate_similarity(
                                    query_emb, node_emb, use_hyperbolic
                                )
                            else:
                                # Fallback to text similarity
                                similarity = self._calculate_text_similarity(query, node_text)
                else:
                    # Use text-based similarity
                    self.search_stats["text_searches"] += 1
                    node_text = self._extract_node_text(node_data)
                    similarity = self._calculate_text_similarity(query, node_text)
                
                # Apply type-based boosting
                node_type = node_data.get("type", "unknown")
                if node_type in self.search_config["search_boost"]:
                    similarity *= self.search_config["search_boost"][node_type]
                
                # Add to results if above threshold
                if similarity >= threshold:
                    matching_nodes.append({
                        "id": node_id,
                        "similarity": similarity,
                        "data": node_data if include_metadata else {"type": node_type}
                    })
            
            # Sort by similarity (highest first)
            matching_nodes.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            results["results"] = matching_nodes[:limit]
            
            # Update metadata
            end_time = time.time()
            search_time_ms = int((end_time - start_time) * 1000)
            
            results["metadata"]["total_nodes_searched"] = total_nodes
            results["metadata"]["search_time_ms"] = search_time_ms
            results["metadata"]["using_embeddings"] = use_embeddings
            results["metadata"]["using_hyperbolic"] = use_hyperbolic
            results["metadata"]["results_found"] = len(matching_nodes)
            results["metadata"]["results_returned"] = len(results["results"])
            
            # Update statistics
            self.search_stats["avg_search_time_ms"] = (
                (self.search_stats["avg_search_time_ms"] * 
                 (self.search_stats["total_searches"] - 1) + 
                 search_time_ms) / 
                self.search_stats["total_searches"]
            )
            
            # Update query stats
            self.query_stats[query] += 1
            
            # Cache results if not too large
            if len(self.query_cache) >= self.max_cache_size:
                # Remove least used query if cache is full
                least_used = min(self.query_stats.items(), key=lambda x: x[1])[0]
                if least_used in self.query_cache:
                    del self.query_cache[least_used]
                    del self.query_stats[least_used]
            
            self.query_cache[cache_key] = results
            self.query_stats[query] = self.query_stats.get(query, 0) + 1
            
            # Convert any NumPy arrays to Python lists for JSON serialization
            results = self._convert_numpy_to_lists(results)
            
            self.logger.info(f"Found {len(matching_nodes)} matching nodes in {search_time_ms}ms")
            
        except Exception as e:
            self.logger.error(f"Error during node search: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def find_paths(self, source: str, target: str, max_depth: int = 3, 
                   min_confidence: float = 0.5, use_hyperbolic: bool = False) -> Dict[str, Any]:
        """
        Find all paths between a source and target node in the knowledge graph.

        Args:
            source: Source node ID
            target: Target node ID
            max_depth: Maximum path length to consider
            min_confidence: Minimum edge confidence to consider
            use_hyperbolic: Whether to use hyperbolic embeddings for path finding

        Returns:
            Dictionary with paths and metadata
        """
        self.logger.info(f"Finding paths from {source} to {target} (max depth: {max_depth})")
        
        # Update statistics
        self.search_stats["total_path_finding"] += 1
        
        # Check cache
        cache_key = f"path:{source}:{target}:{max_depth}:{min_confidence}:{use_hyperbolic}"
        if cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            return self.query_cache[cache_key]
        
        results = {
            "success": True,
            "source": source,
            "target": target,
            "paths": [],
            "metadata": {
                "max_depth": max_depth,
                "min_confidence": min_confidence,
                "use_hyperbolic": use_hyperbolic,
                "search_time_ms": 0
            }
        }
        start_time = time.time()
        
        try:
            # Get core graph for node access
            core_graph = self.module_registry.get_module("core_graph")
            if not core_graph:
                self.logger.error("Core graph module not found")
                return {"success": False, "error": "Core graph module not found"}
                
            # Check if source and target nodes exist
            if not await core_graph.has_node(source):
                self.logger.error(f"Source node {source} does not exist")
                return {"success": False, "error": f"Source node {source} does not exist"}
            if not await core_graph.has_node(target):
                self.logger.error(f"Target node {target} does not exist")
                return {"success": False, "error": f"Target node {target} does not exist"}
                
            # Find paths using modified BFS
            all_paths = []
            visited = {source: True}
            queue = deque([(source, [source], 0)])  # (node, path_so_far, depth)
            
            while queue:
                current, path, depth = queue.popleft()
                
                # If we reached the target, add the path
                if current == target:
                    all_paths.append(self._format_path(path, core_graph))
                    continue
                
                # If we reached max depth, skip further exploration
                if depth >= max_depth:
                    continue
                
                # Get neighbors
                neighbors = await core_graph.get_connected_nodes(current, direction="outgoing")
                
                for neighbor in neighbors:
                    # Skip if already in path
                    if neighbor in path:
                        continue
                    
                    # Get edge data
                    edges = await core_graph.get_edges(current, neighbor)
                    
                    # Check if any edge meets confidence threshold
                    valid_edge = False
                    for edge in edges:
                        if edge.get("confidence", 0) >= min_confidence:
                            valid_edge = True
                            break
                    
                    if not valid_edge:
                        continue
                    
                    # Add to queue
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, depth + 1))
                    visited[neighbor] = True
            
            # Update results
            results["paths"] = all_paths
            
            # Update metadata
            end_time = time.time()
            search_time_ms = int((end_time - start_time) * 1000)
            
            results["metadata"]["search_time_ms"] = search_time_ms
            results["metadata"]["paths_found"] = len(all_paths)
            
            # Cache results if not too large
            if len(self.query_cache) < self.max_cache_size:
                self.query_cache[cache_key] = results
            
            self.logger.info(f"Found {len(all_paths)} paths from {source} to {target}")
            
        except Exception as e:
            self.logger.error(f"Error during path finding: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _format_path(self, path: List[str], core_graph) -> Dict[str, Any]:
        """
        Format a path for output.
        
        Args:
            path: List of node IDs in the path
            core_graph: Core graph module for node access
            
        Returns:
            Formatted path with node information
        """
        formatted_path = {
            "length": len(path) - 1,  # Number of edges
            "nodes": [],
            "edges": []
        }
        
        # Add node information
        for node_id in path:
            node_data = core_graph.graph.nodes.get(node_id, {})
            formatted_path["nodes"].append({
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "name": node_data.get("name", node_id)
            })
        
        # Add edge information
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Get edge data
            edge_data = core_graph.graph.get_edge_data(source, target)
            
            if edge_data:
                # Get first edge (multi-graph can have multiple edges)
                first_edge_key = list(edge_data.keys())[0]
                edge = edge_data[first_edge_key]
                
                formatted_path["edges"].append({
                    "source": source,
                    "target": target,
                    "type": edge.get("type", "related_to"),
                    "confidence": edge.get("confidence", 0.5)
                })
            else:
                # Default edge if data not available
                formatted_path["edges"].append({
                    "source": source,
                    "target": target,
                    "type": "related_to",
                    "confidence": 0.5
                })
        
        return formatted_path
    
    async def get_most_relevant_nodes(self, query: str, context_size: int = 5, 
                               include_related: bool = True) -> Dict[str, Any]:
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
        
        # Check cache
        cache_key = f"relevance:{query}:{context_size}:{include_related}"
        if cache_key in self.query_cache:
            self.search_stats["cache_hits"] += 1
            return self.query_cache[cache_key]
        
        results = {
            "success": True,
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
            # Get core graph for node access
            core_graph = self.module_registry.get_module("core_graph")
            if not core_graph:
                self.logger.error("Core graph module not found")
                return {"success": False, "error": "Core graph module not found"}
            
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
                related_nodes = {}
                
                # For each primary node, find its neighbors
                for primary_node in primary_nodes:
                    node_id = primary_node["id"]
                    
                    # Get all neighbors
                    neighbors = await core_graph.get_connected_nodes(node_id)
                    
                    for neighbor in neighbors:
                        # Skip if already in primary nodes
                        if any(n["id"] == neighbor for n in primary_nodes):
                            continue
                            
                        # Skip if already in related nodes
                        if neighbor in related_nodes:
                            # Update related_to list
                            related_nodes[neighbor]["related_to"].append(node_id)
                            continue
                            
                        # Get node data
                        neighbor_data = await core_graph.get_node(neighbor)
                        
                        # Add to related nodes
                        related_nodes[neighbor] = {
                            "id": neighbor,
                            "related_to": [node_id],
                            "data": neighbor_data
                        }
                        
                        # Limit number of related nodes
                        if len(related_nodes) >= context_size * 2:
                            break
                
                # Add related nodes to results
                results["related_nodes"] = list(related_nodes.values())
            
            # Update metadata
            end_time = time.time()
            search_time_ms = int((end_time - start_time) * 1000)
            
            results["metadata"]["search_time_ms"] = search_time_ms
            results["metadata"]["primary_count"] = len(results["primary_nodes"])
            results["metadata"]["related_count"] = len(results["related_nodes"])
            
            # Cache results if not too large
            if len(self.query_cache) < self.max_cache_size:
                self.query_cache[cache_key] = results
            
            self.logger.info(f"Found {len(results['primary_nodes'])} primary nodes and {len(results['related_nodes'])} related nodes")
            
        except Exception as e:
            self.logger.error(f"Error finding relevant nodes: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def generate_context(self, query: str, max_tokens: int = 512) -> str:
        """
        Generate context from the knowledge graph based on the query.
        
        Args:
            query: The query to generate context for
            max_tokens: Maximum context tokens to generate
            
        Returns:
            Formatted context string for LLM consumption
        """
        self.logger.info(f"Generating knowledge graph context for query: {query}")
        
        # Get relevant nodes
        relevant_result = await self.get_most_relevant_nodes(query, context_size=3, include_related=True)
        
        if not relevant_result["success"]:
            return "Error generating context: " + relevant_result.get("error", "Unknown error")
        
        # Create a list to hold context parts
        context_parts = []
        
        # Extract entities and concepts from the query for organization
        query_elements = self._extract_query_elements(query)
        
        # Process primary nodes
        primary_nodes = relevant_result["primary_nodes"]
        if primary_nodes:
            context_parts.append("# Relevant Knowledge")
            
            for node in primary_nodes:
                node_id = node["id"]
                node_data = node["data"]
                node_type = node_data.get("type", "unknown")
                
                # Format based on node type
                if node_type == "concept":
                    context_parts.append(f"## Concept: {node_id}")
                    context_parts.append(f"Definition: {node_data.get('definition', 'No definition')}")
                    
                    # Add confidence if available
                    if "confidence" in node_data:
                        context_parts.append(f"Confidence: {node_data.get('confidence', 0.0):.2f}")
                
                elif node_type == "entity":
                    context_parts.append(f"## Entity: {node_data.get('name', node_id)}")
                    context_parts.append(f"Description: {node_data.get('description', 'No description')}")
                    
                    # Add confidence if available
                    if "confidence" in node_data:
                        context_parts.append(f"Confidence: {node_data.get('confidence', 0.0):.2f}")
                
                elif node_type == "dream_insight":
                    context_parts.append(f"## Dream Insight")
                    context_parts.append(f"Insight: {node_data.get('insight', 'No insight')}")
                    
                    # Add timestamp if available
                    if "timestamp" in node_data:
                        context_parts.append(f"Timestamp: {node_data.get('timestamp')}")
                
                else:
                    # Generic node format
                    context_parts.append(f"## {node_type.capitalize()}: {node_id}")
                    
                    # Add key attributes
                    for key, value in node_data.items():
                        if key not in ["id", "type", "created", "modified"] and isinstance(value, (str, int, float, bool)):
                            context_parts.append(f"{key}: {value}")
        
        # Process related nodes - more condensed format
        related_nodes = relevant_result["related_nodes"]
        if related_nodes:
            context_parts.append("\n# Related Information")
            
            # Group by node type for better organization
            nodes_by_type = defaultdict(list)
            for node in related_nodes:
                node_type = node["data"].get("type", "unknown")
                nodes_by_type[node_type].append(node)
            
            # Process each type group
            for node_type, nodes in nodes_by_type.items():
                context_parts.append(f"## Related {node_type.capitalize()}s")
                
                for node in nodes[:5]:  # Limit to 5 per type
                    node_id = node["id"]
                    node_data = node["data"]
                    
                    # Brief summary format based on type
                    if node_type == "concept":
                        definition = node_data.get("definition", "")
                        # Truncate long definitions
                        if len(definition) > 100:
                            definition = definition[:97] + "..."
                        context_parts.append(f"- {node_id}: {definition}")
                    
                    elif node_type == "entity":
                        name = node_data.get("name", node_id)
                        context_parts.append(f"- {name}")
                    
                    else:
                        # Generic format
                        context_parts.append(f"- {node_id}")
        
        # Combine all parts with newlines
        context = "\n".join(context_parts)
        
        # Approximate token count and truncate if needed
        # Rough approximation: 1 token ≈ 4 characters for English text
        estimated_tokens = len(context) // 4
        if estimated_tokens > max_tokens:
            # Truncate at paragraph boundaries if possible
            paragraphs = context.split("\n\n")
            truncated_context = []
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = len(para) // 4
                if current_tokens + para_tokens <= max_tokens:
                    truncated_context.append(para)
                    current_tokens += para_tokens
                else:
                    break
            
            context = "\n\n".join(truncated_context)
            if not truncated_context:
                # If we couldn't fit even one paragraph, truncate by characters
                context = context[:max_tokens * 4] + "..."
        
        return context
    
    def _extract_node_text(self, node_data: Dict[str, Any]) -> str:
        """
        Extract searchable text from node data.
        
        Args:
            node_data: Node data
            
        Returns:
            Extracted text
        """
        node_type = node_data.get("type", "unknown")
        
        if node_type == "concept":
            # For concepts, use id and definition
            return f"{node_data.get('id', '')} {node_data.get('definition', '')}"
        elif node_type == "entity":
            # For entities, use id, name, and description
            return f"{node_data.get('id', '')} {node_data.get('name', '')} {node_data.get('description', '')}"
        elif node_type == "dream_insight":
            # For dream insights, use id and insight
            return f"{node_data.get('id', '')} {node_data.get('insight', '')}"
        else:
            # For other types, combine all string values
            text_parts = [node_data.get('id', '')]
            
            for key, value in node_data.items():
                if isinstance(value, str) and key not in ["id", "type", "created", "modified"]:
                    text_parts.append(value)
            
            return " ".join(text_parts)
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """
        Calculate similarity between query and text using simple techniques.
        
        Args:
            query: Search query
            text: Text to compare
            
        Returns:
            Similarity score (0-1)
        """
        if not query or not text:
            return 0.0
        
        # Normalize and split into words
        query_words = set(query.lower().strip().split())
        text_words = set(text.lower().strip().split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for exact phrase matches
        phrase_bonus = 0.0
        
        # Check for exact phrases
        query_lower = query.lower()
        text_lower = text.lower()
        
        if query_lower in text_lower:
            # Exact query match
            phrase_bonus = 0.3
        else:
            # Check for query word sequences
            query_seq = query_lower.split()
            if len(query_seq) > 1:
                for i in range(len(query_seq) - 1):
                    bigram = f"{query_seq[i]} {query_seq[i+1]}"
                    if bigram in text_lower:
                        phrase_bonus = 0.15
                        break
        
        # Combine scores
        similarity = jaccard + phrase_bonus
        
        # Ensure score is in [0, 1] range
        return min(1.0, similarity)
    
    def _extract_query_elements(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities and concepts from a query string.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with extracted entities and concepts
        """
        # This is a simplified version - in a real implementation
        # this would use more sophisticated NLP techniques
        
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Extract potential concepts (lowercase words excluding stopwords)
        stopwords = {"a", "an", "the", "in", "on", "at", "with", "by", "for", "from", "of", "to"}
        concepts = [word.lower() for word in re.findall(r'\b[a-z]{4,}\b', query.lower())
                    if word.lower() not in stopwords]
        
        return {
            'entities': entities,
            'concepts': concepts
        }
    
    def _convert_numpy_to_lists(self, obj):
        """Convert NumPy arrays to Python lists recursively in dictionaries and lists.
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object with all NumPy arrays replaced by Python lists
        """
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_lists(item) for item in obj)
        else:
            return obj
    
    async def analyze_query_patterns(self) -> Dict[str, Any]:
        """
        Analyze query patterns from statistics.
        
        Returns:
            Dictionary with query pattern analysis
        """
        # Get most frequent queries
        sorted_queries = sorted(self.query_stats.items(), key=lambda x: x[1], reverse=True)
        top_queries = sorted_queries[:10]
        
        # Analyze word frequency in queries
        word_counts = defaultdict(int)
        
        for query, count in self.query_stats.items():
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += count
        
        # Get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:20]
        
        return {
            "total_unique_queries": len(self.query_stats),
            "total_query_count": sum(self.query_stats.values()),
            "top_queries": top_queries,
            "top_words": top_words,
            "cache_size": len(self.query_cache),
            "search_stats": {
                "total_searches": self.search_stats["total_searches"],
                "total_path_finding": self.search_stats["total_path_finding"],
                "cache_hits": self.search_stats["cache_hits"],
                "embedding_searches": self.search_stats["embedding_searches"],
                "text_searches": self.search_stats["text_searches"],
                "avg_search_time_ms": self.search_stats["avg_search_time_ms"]
            }
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the query cache.
        
        Returns:
            Clear result
        """
        cache_size = len(self.query_cache)
        self.query_cache.clear()
        
        return {
            "success": True,
            "cleared_entries": cache_size
        }