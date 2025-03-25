"""
API Manager Module for Lucidia's Knowledge Graph

This module handles external system integration, service endpoints,
authentication, and cross-system communication.
"""

import logging
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict
import asyncio
import websockets

from .base_module import KnowledgeGraphModule

class APIManager(KnowledgeGraphModule):
    """
    API Manager responsible for external system integration and communication.
    
    This module provides APIs for external systems to interact with the knowledge graph,
    handles authentication, request routing, and cross-system communication.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the API Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # API configuration
        self.api_config = {
            "host": self.get_config("host", "localhost"),
            "port": self.get_config("port", 8765),
            "enabled": self.get_config("enabled", True),
            "auth_required": self.get_config("auth_required", True),
            "auth_tokens": self.get_config("auth_tokens", ["lucidia_test_token"]),
            "max_request_size": self.get_config("max_request_size", 1024 * 1024),  # 1MB
            "rate_limit": self.get_config("rate_limit", 100),  # Requests per minute
            "cors_origins": self.get_config("cors_origins", ["*"])
        }
        
        # Server instance
        self.server = None
        self.running = False
        
        # Request handlers
        self.request_handlers = {}
        
        # Request tracking
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "auth_failures": 0,
            "by_endpoint": defaultdict(int),
            "response_times": []
        }
        
        # Client sessions
        self.active_connections = set()
        self.client_sessions = {}
        
        self.logger.info("API Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("api_call", self._handle_api_call)
        await self.event_bus.subscribe("external_knowledge_submitted", self._handle_external_knowledge)
        await self.event_bus.subscribe("knowledge_update", self._broadcast_knowledge_update)
        self.logger.info("Subscribed to API-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # Register request handlers
        self._register_handlers()
        
        # Start API server if enabled
        if self.api_config["enabled"]:
            await self._start_server()
    
    async def _cleanup_resources(self):
        """Clean up resources before shutdown."""
        # Shutdown server if running
        if self.running:
            await self._stop_server()
    
    def _register_handlers(self):
        """Register request handlers for various endpoints."""
        # Core graph operations
        self.request_handlers["node/add"] = self._handle_add_node
        self.request_handlers["node/update"] = self._handle_update_node
        self.request_handlers["node/get"] = self._handle_get_node
        self.request_handlers["edge/add"] = self._handle_add_edge
        self.request_handlers["edge/update"] = self._handle_update_edge
        
        # Search and query operations
        self.request_handlers["search"] = self._handle_search
        self.request_handlers["path/find"] = self._handle_find_path
        self.request_handlers["relevance"] = self._handle_relevance
        
        # Knowledge integration
        self.request_handlers["dream/integrate"] = self._handle_dream_integrate
        self.request_handlers["emotion/analyze"] = self._handle_emotion_analyze
        self.request_handlers["external/knowledge"] = self._handle_external_knowledge_api
        
        # Maintenance operations
        self.request_handlers["maintenance/trigger"] = self._handle_maintenance_trigger
        self.request_handlers["health/check"] = self._handle_health_check
        
        # Visualization operations
        self.request_handlers["visualize/graph"] = self._handle_visualize_graph
        self.request_handlers["visualize/domain"] = self._handle_visualize_domain
        self.request_handlers["visualize/concept"] = self._handle_visualize_concept
        
        # Meta operations
        self.request_handlers["meta/stats"] = self._handle_get_stats
        self.request_handlers["meta/status"] = self._handle_get_status
    
    async def _start_server(self):
        """Start the WebSocket API server."""
        if self.running:
            self.logger.warning("Server already running")
            return
        
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.api_config["host"],
                self.api_config["port"]
            )
            self.running = True
            self.logger.info(f"API server started on {self.api_config['host']}:{self.api_config['port']}")
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
    
    async def _stop_server(self):
        """Stop the WebSocket API server."""
        if not self.running:
            return
        
        # Close all active connections
        for connection in self.active_connections:
            await connection.close()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.running = False
        self.logger.info("API server stopped")
    
    async def _handle_connection(self, websocket, path):
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Add to active connections
        self.active_connections.add(websocket)
        
        # Initialize session
        self.client_sessions[session_id] = {
            "id": session_id,
            "authenticated": False,
            "connection_time": datetime.now(),
            "ip": websocket.remote_address[0] if websocket.remote_address else "unknown",
            "last_request_time": None,
            "request_count": 0
        }
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "system",
                "message": "Welcome to Lucidia Knowledge Graph API",
                "session_id": session_id,
                "requires_auth": self.api_config["auth_required"]
            }))
            
            # Handle messages
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Update session
                    self.client_sessions[session_id]["last_request_time"] = datetime.now()
                    self.client_sessions[session_id]["request_count"] += 1
                    
                    # Handle authentication
                    if not self.client_sessions[session_id]["authenticated"] and self.api_config["auth_required"]:
                        if data.get("type") == "auth" and "token" in data:
                            # Authenticate
                            auth_result = self._authenticate(data["token"])
                            self.client_sessions[session_id]["authenticated"] = auth_result
                            
                            # Send auth result
                            await websocket.send(json.dumps({
                                "type": "auth_result",
                                "success": auth_result,
                                "message": "Authentication successful" if auth_result else "Authentication failed"
                            }))
                            
                            if not auth_result:
                                self.request_stats["auth_failures"] += 1
                            
                            continue
                        else:
                            # Require authentication
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": "Authentication required"
                            }))
                            
                            continue
                    
                    # Process request
                    request_id = data.get("request_id", str(uuid.uuid4()))
                    endpoint = data.get("endpoint")
                    request_data = data.get("data", {})
                    
                    # Update request stats
                    self.request_stats["total_requests"] += 1
                    self.request_stats["by_endpoint"][endpoint] += 1
                    
                    # Handle request
                    if endpoint in self.request_handlers:
                        # Track response time
                        start_time = time.time()
                        
                        # Call handler
                        response = await self.request_handlers[endpoint](request_data)
                        
                        # Calculate response time
                        response_time = time.time() - start_time
                        self.request_stats["response_times"].append(response_time)
                        
                        # Keep only last 100 response times
                        if len(self.request_stats["response_times"]) > 100:
                            self.request_stats["response_times"] = self.request_stats["response_times"][-100:]
                        
                        # Send response
                        await websocket.send(json.dumps({
                            "type": "response",
                            "request_id": request_id,
                            "success": True,
                            "data": response
                        }))
                        
                        self.request_stats["successful_requests"] += 1
                    else:
                        # Unknown endpoint
                        await websocket.send(json.dumps({
                            "type": "response",
                            "request_id": request_id,
                            "success": False,
                            "error": f"Unknown endpoint: {endpoint}"
                        }))
                        
                        self.request_stats["failed_requests"] += 1
                    
                except json.JSONDecodeError:
                    # Invalid JSON
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                    
                    self.request_stats["failed_requests"] += 1
                    
                except Exception as e:
                    # Other errors
                    self.logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error: {str(e)}"
                    }))
                    
                    self.request_stats["failed_requests"] += 1
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed for session {session_id}")
        finally:
            # Clean up
            self.active_connections.remove(websocket)
            if session_id in self.client_sessions:
                del self.client_sessions[session_id]
    
    def _authenticate(self, token: str) -> bool:
        """
        Authenticate a client token.
        
        Args:
            token: Authentication token
            
        Returns:
            Authentication success
        """
        return token in self.api_config["auth_tokens"]
    
    async def _handle_api_call(self, data):
        """
        Handle API call events.
        
        Args:
            data: API call data
            
        Returns:
            Call result
        """
        endpoint = data.get("endpoint")
        request_data = data.get("data", {})
        
        if endpoint in self.request_handlers:
            return await self.request_handlers[endpoint](request_data)
        else:
            return {"success": False, "error": f"Unknown endpoint: {endpoint}"}
    
    async def _handle_external_knowledge(self, data):
        """
        Handle external knowledge events.
        
        Args:
            data: External knowledge data
            
        Returns:
            Processing result
        """
        self.logger.info(f"Handling external knowledge from API: {data.get('source', 'unknown')}")
        
        # Forward to event bus for processing
        result = await self.event_bus.emit("external_knowledge", data)
        
        if result:
            return result[0]  # Return first result (should be from the correct handler)
        else:
            return {"success": False, "error": "No handler for external knowledge"}
    
    async def _broadcast_knowledge_update(self, data):
        """
        Broadcast knowledge update to connected clients.
        
        Args:
            data: Knowledge update data
        """
        # Only broadcast to authenticated clients
        authenticated_connections = [
            conn for conn in self.active_connections 
            if self.client_sessions.get(conn.id, {}).get("authenticated", False)
        ]
        
        # Prepare broadcast message
        message = {
            "type": "knowledge_update",
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Broadcast to all authenticated clients
        for connection in authenticated_connections:
            try:
                await connection.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
    
    # === Request Handlers ===
    
    async def _handle_add_node(self, data):
        """
        Handle add node request.
        
        Args:
            data: Request data
            
        Returns:
            Operation result
        """
        node_id = data.get("node_id")
        node_type = data.get("node_type")
        attributes = data.get("attributes", {})
        domain = data.get("domain", "general_knowledge")
        
        if not all([node_id, node_type]):
            return {"success": False, "error": "Missing required fields"}
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Add node
        result = await core_graph.add_node(node_id, node_type, attributes, domain)
        
        return {"success": result, "node_id": node_id}
    
    async def _handle_update_node(self, data):
        """
        Handle update node request.
        
        Args:
            data: Request data
            
        Returns:
            Operation result
        """
        node_id = data.get("node_id")
        attributes = data.get("attributes", {})
        
        if not node_id:
            return {"success": False, "error": "Node ID required"}
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Update node
        result = await core_graph.update_node(node_id, attributes)
        
        return {"success": result, "node_id": node_id}
    
    async def _handle_get_node(self, data):
        """
        Handle get node request.
        
        Args:
            data: Request data
            
        Returns:
            Node data
        """
        node_id = data.get("node_id")
        
        if not node_id:
            return {"success": False, "error": "Node ID required"}
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Get node
        node_data = await core_graph.get_node(node_id)
        
        if node_data:
            return {"success": True, "node_id": node_id, "data": node_data}
        else:
            return {"success": False, "error": f"Node {node_id} not found"}
    
    async def _handle_add_edge(self, data):
        """
        Handle add edge request.
        
        Args:
            data: Request data
            
        Returns:
            Operation result
        """
        source = data.get("source")
        target = data.get("target")
        edge_type = data.get("edge_type")
        attributes = data.get("attributes", {})
        
        if not all([source, target, edge_type]):
            return {"success": False, "error": "Missing required fields"}
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Add edge
        edge_key = await core_graph.add_edge(source, target, edge_type, attributes)
        
        if edge_key is not None:
            return {"success": True, "source": source, "target": target, "edge_key": edge_key}
        else:
            return {"success": False, "error": "Failed to add edge"}
    
    async def _handle_update_edge(self, data):
        """
        Handle update edge request.
        
        Args:
            data: Request data
            
        Returns:
            Operation result
        """
        source = data.get("source")
        target = data.get("target")
        edge_key = data.get("edge_key")
        attributes = data.get("attributes", {})
        
        if not all([source, target, edge_key is not None]):
            return {"success": False, "error": "Missing required fields"}
        
        # Get core graph module
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Update edge
        result = await core_graph.update_edge(source, target, edge_key, attributes)
        
        return {"success": result, "source": source, "target": target, "edge_key": edge_key}
    
    async def _handle_search(self, data):
        """
        Handle search request.
        
        Args:
            data: Request data
            
        Returns:
            Search results
        """
        query = data.get("query")
        limit = data.get("limit", 10)
        threshold = data.get("threshold", 0.5)
        include_metadata = data.get("include_metadata", True)
        domains = data.get("domains")
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        # Get query engine module
        query_engine = self.module_registry.get_module("query_engine")
        if not query_engine:
            return {"success": False, "error": "Query engine module not found"}
        
        # Search nodes
        results = await query_engine.search_nodes(
            query, limit, threshold, include_metadata, domains
        )
        
        return results
    
    async def _handle_find_path(self, data):
        """
        Handle find path request.
        
        Args:
            data: Request data
            
        Returns:
            Path results
        """
        source = data.get("source")
        target = data.get("target")
        max_depth = data.get("max_depth", 3)
        min_confidence = data.get("min_confidence", 0.5)
        
        if not source or not target:
            return {"success": False, "error": "Source and target required"}
        
        # Get query engine module
        query_engine = self.module_registry.get_module("query_engine")
        if not query_engine:
            return {"success": False, "error": "Query engine module not found"}
        
        # Find paths
        results = await query_engine.find_paths(
            source, target, max_depth, min_confidence
        )
        
        return results
    
    async def _handle_relevance(self, data):
        """
        Handle relevance request.
        
        Args:
            data: Request data
            
        Returns:
            Relevance results
        """
        query = data.get("query")
        context_size = data.get("context_size", 5)
        include_related = data.get("include_related", True)
        
        if not query:
            return {"success": False, "error": "Query required"}
        
        # Get query engine module
        query_engine = self.module_registry.get_module("query_engine")
        if not query_engine:
            return {"success": False, "error": "Query engine module not found"}
        
        # Get most relevant nodes
        results = await query_engine.get_most_relevant_nodes(
            query, context_size, include_related
        )
        
        return results
    
    async def _handle_dream_integrate(self, data):
        """
        Handle dream integration request.
        
        Args:
            data: Request data
            
        Returns:
            Integration results
        """
        insight_text = data.get("insight")
        source_memory = data.get("source_memory")
        
        if not insight_text:
            return {"success": False, "error": "Dream insight text required"}
        
        # Get dream integration module
        dream_integration = self.module_registry.get_module("dream_integration")
        if not dream_integration:
            return {"success": False, "error": "Dream integration module not found"}
        
        # Integrate dream insight
        results = await dream_integration.integrate_dream_insight(
            insight_text, source_memory
        )
        
        return results
    
    async def _handle_emotion_analyze(self, data):
        """
        Handle emotion analysis request.
        
        Args:
            data: Request data
            
        Returns:
            Analysis results
        """
        text = data.get("text")
        
        if not text:
            return {"success": False, "error": "Text required"}
        
        # Get emotional context module
        emotional_context = self.module_registry.get_module("emotional_context")
        if not emotional_context:
            return {"success": False, "error": "Emotional context module not found"}
        
        # Analyze emotion
        results = await emotional_context.analyze_emotion(text)
        
        return {"success": True, "analysis": results}
    
    async def _handle_external_knowledge_api(self, data):
        """
        Handle external knowledge API request.
        
        Args:
            data: Request data
            
        Returns:
            Processing results
        """
        source = data.get("source")
        data_type = data.get("type")
        knowledge_data = data.get("data")
        
        if not all([source, data_type, knowledge_data]):
            return {"success": False, "error": "Missing required fields"}
        
        # Prepare external knowledge event
        external_knowledge = {
            "source": source,
            "type": data_type,
            "data": knowledge_data
        }
        
        # Emit event for processing
        await self.event_bus.emit("external_knowledge_submitted", external_knowledge)
        
        # Forward to event bus for processing
        result = await self.event_bus.emit("external_knowledge", external_knowledge)
        
        if result:
            return result[0]  # Return first result (should be from the correct handler)
        else:
            return {"success": False, "error": "No handler for external knowledge"}
    
    async def _handle_maintenance_trigger(self, data):
        """
        Handle maintenance trigger request.
        
        Args:
            data: Request data
            
        Returns:
            Maintenance results
        """
        force = data.get("force", False)
        operation = data.get("operation")
        
        # Get maintenance manager module
        maintenance_manager = self.module_registry.get_module("maintenance_manager")
        if not maintenance_manager:
            return {"success": False, "error": "Maintenance manager module not found"}
        
        # Trigger maintenance
        results = await maintenance_manager.trigger_maintenance(operation)
        
        return results
    
    async def _handle_health_check(self, data):
        """
        Handle health check request.
        
        Args:
            data: Request data
            
        Returns:
            Health check results
        """
        force = data.get("force", False)
        
        # Get maintenance manager module
        maintenance_manager = self.module_registry.get_module("maintenance_manager")
        if not maintenance_manager:
            return {"success": False, "error": "Maintenance manager module not found"}
        
        # Run health check
        results = await maintenance_manager.run_health_check()
        
        return results
    
    async def _handle_visualize_graph(self, data):
        """
        Handle graph visualization request.
        
        Args:
            data: Request data
            
        Returns:
            Visualization results
        """
        include_attributes = data.get("include_attributes", False)
        
        # Get visualization manager module
        visualization_manager = self.module_registry.get_module("visualization_manager")
        if not visualization_manager:
            return {"success": False, "error": "Visualization manager module not found"}
        
        # Generate visualization
        mermaid_code = await visualization_manager.generate_full_graph_diagram(include_attributes)
        
        return {"success": True, "mermaid_code": mermaid_code}
    
    async def _handle_visualize_domain(self, data):
        """
        Handle domain visualization request.
        
        Args:
            data: Request data
            
        Returns:
            Visualization results
        """
        domain = data.get("domain")
        include_attributes = data.get("include_attributes", False)
        
        if not domain:
            return {"success": False, "error": "Domain required"}
        
        # Get visualization manager module
        visualization_manager = self.module_registry.get_module("visualization_manager")
        if not visualization_manager:
            return {"success": False, "error": "Visualization manager module not found"}
        
        # Generate visualization
        mermaid_code = await visualization_manager.generate_domain_diagram(domain, include_attributes)
        
        return {"success": True, "domain": domain, "mermaid_code": mermaid_code}
    
    async def _handle_visualize_concept(self, data):
        """
        Handle concept visualization request.
        
        Args:
            data: Request data
            
        Returns:
            Visualization results
        """
        concept = data.get("concept")
        depth = data.get("depth", 2)
        
        if not concept:
            return {"success": False, "error": "Concept required"}
        
        # Get visualization manager module
        visualization_manager = self.module_registry.get_module("visualization_manager")
        if not visualization_manager:
            return {"success": False, "error": "Visualization manager module not found"}
        
        # Generate visualization
        mermaid_code = await visualization_manager.generate_concept_network(concept, depth)
        
        return {"success": True, "concept": concept, "mermaid_code": mermaid_code}
    
    async def _handle_get_stats(self, data):
        """
        Handle stats request.
        
        Args:
            data: Request data
            
        Returns:
            System statistics
        """
        # Get stats from various modules
        stats = {
            "api": {
                "total_requests": self.request_stats["total_requests"],
                "successful_requests": self.request_stats["successful_requests"],
                "failed_requests": self.request_stats["failed_requests"],
                "auth_failures": self.request_stats["auth_failures"],
                "avg_response_time": (
                    sum(self.request_stats["response_times"]) / len(self.request_stats["response_times"])
                    if self.request_stats["response_times"] else 0
                ),
                "active_connections": len(self.active_connections)
            }
        }
        
        # Get core graph stats
        core_graph = self.module_registry.get_module("core_graph")
        if core_graph:
            core_stats = await core_graph.get_graph_metrics()
            stats["graph"] = core_stats
        
        # Get query engine stats
        query_engine = self.module_registry.get_module("query_engine")
        if query_engine and hasattr(query_engine, "analyze_query_patterns"):
            query_stats = await query_engine.analyze_query_patterns()
            stats["query"] = query_stats
        
        # Get maintenance stats
        maintenance_manager = self.module_registry.get_module("maintenance_manager")
        if maintenance_manager and hasattr(maintenance_manager, "get_maintenance_stats"):
            maintenance_stats = await maintenance_manager.get_maintenance_stats()
            stats["maintenance"] = maintenance_stats
        
        # Get dream integration stats
        dream_integration = self.module_registry.get_module("dream_integration")
        if dream_integration and hasattr(dream_integration, "get_integration_stats"):
            dream_stats = await dream_integration.get_integration_stats()
            stats["dream_integration"] = dream_stats
        
        # Get emotional context stats
        emotional_context = self.module_registry.get_module("emotional_context")
        if emotional_context and hasattr(emotional_context, "get_emotion_stats"):
            emotion_stats = await emotional_context.get_emotion_stats()
            stats["emotional_context"] = emotion_stats
        
        # Get contradiction stats
        contradiction_manager = self.module_registry.get_module("contradiction_manager")
        if contradiction_manager and hasattr(contradiction_manager, "get_contradiction_stats"):
            contradiction_stats = await contradiction_manager.get_contradiction_stats()
            stats["contradictions"] = contradiction_stats
        
        return {"success": True, "stats": stats}
    
    async def _handle_get_status(self, data):
        """
        Handle status request.
        
        Args:
            data: Request data
            
        Returns:
            System status
        """
        # Get all modules
        modules = self.module_registry.get_all_modules()
        
        # Check status of each module
        module_status = {}
        for module_id, module in modules.items():
            if hasattr(module, "get_status"):
                module_status[module_id] = await module.get_status()
            else:
                module_status[module_id] = {"initialized": True}  # Assume initialized if no status method
        
        return {
            "success": True,
            "status": {
                "api_server": {
                    "running": self.running,
                    "host": self.api_config["host"],
                    "port": self.api_config["port"],
                    "active_connections": len(self.active_connections)
                },
                "modules": module_status
            }
        }
    
    async def get_api_stats(self):
        """
        Get API statistics.
        
        Returns:
            Dictionary with API statistics
        """
        return {
            "total_requests": self.request_stats["total_requests"],
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "auth_failures": self.request_stats["auth_failures"],
            "by_endpoint": dict(self.request_stats["by_endpoint"]),
            "avg_response_time": (
                sum(self.request_stats["response_times"]) / len(self.request_stats["response_times"])
                if self.request_stats["response_times"] else 0
            ),
            "active_connections": len(self.active_connections),
            "active_sessions": len(self.client_sessions),
            "server_running": self.running
        }