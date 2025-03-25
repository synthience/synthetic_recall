"""
Maintenance Manager Module for Lucidia's Knowledge Graph

This module implements graph health monitoring, performance optimization,
low-value connection pruning, and adaptive maintenance scheduling.
"""

import logging
import uuid
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict

from .base_module import KnowledgeGraphModule

class MaintenanceManager(KnowledgeGraphModule):
    """
    Maintenance Manager responsible for knowledge graph maintenance and optimization.
    
    This module handles health monitoring, performance optimization, connection pruning,
    metric tracking, and adaptive maintenance scheduling.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Maintenance Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # Adaptive maintenance configuration
        self.adaptive_maintenance = {
            "last_maintenance": datetime.now() - timedelta(days=1),  # Start one day ago to enable initial maintenance
            "maintenance_interval": self.get_config("maintenance_interval", 24 * 60 * 60),  # Default: 24 hours in seconds
            "optimization_history": [],  # History of optimizations
            "metrics_history": [],  # History of metrics
            "maintenance_enabled": self.get_config("maintenance_enabled", True),
            "auto_maintenance": self.get_config("auto_maintenance", False),
            "metric_thresholds": {
                "density": self.get_config("density_threshold", 0.01),  # Maximum density before pruning
                "average_degree": self.get_config("average_degree_threshold", 10),  # Maximum average degree before pruning
                "clustering": self.get_config("clustering_threshold", 0.3),  # Target clustering coefficient
                "assortativity": self.get_config("assortativity_threshold", 0.1)  # Target assortativity coefficient
            }
        }
        
        # Health checks configuration
        self.health_checks = {
            "last_check": datetime.now() - timedelta(hours=1),  # Start one hour ago to enable initial check
            "check_interval": self.get_config("health_check_interval", 60 * 60),  # Default: 1 hour in seconds
            "check_enabled": self.get_config("health_check_enabled", True),
            "auto_repair": self.get_config("auto_repair", True),
            "health_history": []  # History of health checks
        }
        
        # Statistics tracking
        self.maintenance_stats = {
            "total_maintenance_runs": 0,
            "total_connections_pruned": 0,
            "total_nodes_reindexed": 0,
            "total_optimizations": 0,
            "health_check_runs": 0,
            "health_issues_detected": 0,
            "health_issues_repaired": 0
        }
        
        self.logger.info("Maintenance Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("maintenance_triggered", self._handle_maintenance_trigger)
        await self.event_bus.subscribe("health_check_triggered", self._handle_health_check_trigger)
        await self.event_bus.subscribe("automatic_maintenance_check", self._handle_auto_check)
        self.logger.info("Subscribed to maintenance-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # No special setup needed
        pass
    
    async def _handle_maintenance_trigger(self, data):
        """
        Handle maintenance trigger events.
        
        Args:
            data: Trigger data
            
        Returns:
            Maintenance result
        """
        force = data.get("force", False)
        specific_operation = data.get("operation")
        
        if not self.adaptive_maintenance["maintenance_enabled"] and not force:
            return {"success": False, "error": "Maintenance is disabled"}
        
        # Check if it's time for maintenance
        current_time = datetime.now()
        last_maintenance = self.adaptive_maintenance["last_maintenance"]
        interval = self.adaptive_maintenance["maintenance_interval"]
        
        if not force and (current_time - last_maintenance).total_seconds() < interval:
            return {
                "success": False, 
                "error": "Maintenance interval not reached",
                "next_maintenance": last_maintenance + timedelta(seconds=interval)
            }
        
        # Trigger maintenance
        result = await self.trigger_maintenance(specific_operation)
        
        return result
    
    async def _handle_health_check_trigger(self, data):
        """
        Handle health check trigger events.
        
        Args:
            data: Trigger data
            
        Returns:
            Health check result
        """
        force = data.get("force", False)
        
        if not self.health_checks["check_enabled"] and not force:
            return {"success": False, "error": "Health checks are disabled"}
        
        # Check if it's time for a health check
        current_time = datetime.now()
        last_check = self.health_checks["last_check"]
        interval = self.health_checks["check_interval"]
        
        if not force and (current_time - last_check).total_seconds() < interval:
            return {
                "success": False, 
                "error": "Health check interval not reached",
                "next_check": last_check + timedelta(seconds=interval)
            }
        
        # Trigger health check
        result = await self.run_health_check()
        
        return result
    
    async def _handle_auto_check(self, data):
        """
        Handle automatic maintenance check events.
        
        Args:
            data: Check data
        """
        # Check if auto maintenance is enabled
        if not self.adaptive_maintenance["auto_maintenance"]:
            return
        
        # Check if it's time for maintenance
        current_time = datetime.now()
        last_maintenance = self.adaptive_maintenance["last_maintenance"]
        interval = self.adaptive_maintenance["maintenance_interval"]
        
        if (current_time - last_maintenance).total_seconds() >= interval:
            # Trigger maintenance
            await self.trigger_maintenance()
        
        # Check if it's time for a health check
        last_check = self.health_checks["last_check"]
        check_interval = self.health_checks["check_interval"]
        
        if (current_time - last_check).total_seconds() >= check_interval:
            # Trigger health check
            await self.run_health_check()
    
    async def trigger_maintenance(self, specific_operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Trigger adaptive maintenance to optimize the knowledge graph.
        
        Args:
            specific_operation: Optional specific operation to perform
            
        Returns:
            Maintenance results
        """
        self.logger.info(f"Triggering adaptive maintenance{f' (operation: {specific_operation})' if specific_operation else ''}")
        
        # Update statistics
        self.maintenance_stats["total_maintenance_runs"] += 1
        
        # Analyze graph metrics
        metrics = await self._analyze_graph_metrics()
        
        # Identify optimizations based on metrics (unless specific operation is requested)
        optimizations = {}
        if specific_operation:
            # Only perform the requested operation
            if specific_operation == "prune_connections":
                optimizations["prune_connections"] = True
                optimizations["prune_threshold"] = 0.3
            elif specific_operation == "reindex_nodes":
                optimizations["reindex_nodes"] = True
                optimizations["reindex_count"] = 20
            elif specific_operation == "adjust_decay":
                optimizations["adjust_decay"] = True
                optimizations["decay_adjustments"] = {
                    "standard": 0.01,
                    "dream_associated": 0.02
                }
            elif specific_operation == "compress_data":
                optimizations["compress_data"] = True
        else:
            # Identify all needed optimizations
            optimizations = await self._identify_optimizations(metrics)
        
        # Apply optimizations
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics_before": metrics,
            "optimizations": [],
            "total_improvements": 0
        }
        
        # Track start time for performance measurement
        start_time = time.time()
        
        # Perform optimizations
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
            
            # Update statistics
            self.maintenance_stats["total_connections_pruned"] += prune_result.get("pruned_count", 0)
        
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
            
            # Update statistics
            self.maintenance_stats["total_nodes_reindexed"] += reindex_result.get("reindexed_count", 0)
        
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
        
        if optimizations.get("compress_data", False):
            # Compress graph data
            compress_result = await self._compress_graph_data()
            results["optimizations"].append({
                "type": "compress_data",
                "result": compress_result
            })
        
        # Analyze metrics after optimization
        post_metrics = await self._analyze_graph_metrics()
        results["metrics_after"] = post_metrics
        
        # Calculate improvements
        improvements = {}
        for key in metrics:
            if key in post_metrics and isinstance(metrics[key], (int, float)) and isinstance(post_metrics[key], (int, float)):
                # For density and average_degree, lower is better
                if key in ["density", "average_degree"]:
                    improvements[key] = metrics[key] - post_metrics[key]
                else:
                    improvements[key] = post_metrics[key] - metrics[key]
        
        results["improvements"] = improvements
        results["total_improvements"] = sum(improvements.values())
        
        # Store optimization learnings
        await self._store_optimization_learnings(results)
        
        # Update last maintenance timestamp
        self.adaptive_maintenance["last_maintenance"] = datetime.now()
        
        # Update metrics history
        self.adaptive_maintenance["metrics_history"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": post_metrics
        })
        
        # Limit metrics history size
        if len(self.adaptive_maintenance["metrics_history"]) > 30:
            self.adaptive_maintenance["metrics_history"] = self.adaptive_maintenance["metrics_history"][-30:]
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results["execution_time_seconds"] = execution_time
        
        # Update total optimizations count
        self.maintenance_stats["total_optimizations"] += len(results["optimizations"])
        
        self.logger.info(f"Completed maintenance in {execution_time:.2f} seconds with {len(results['optimizations'])} optimizations")
        
        # Emit maintenance completed event
        await self.event_bus.emit("maintenance_completed", {
            "timestamp": datetime.now().isoformat(),
            "optimizations": len(results["optimizations"]),
            "execution_time": execution_time
        })
        
        return results
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Run a health check on the knowledge graph.
        
        Returns:
            Health check results
        """
        self.logger.info("Running knowledge graph health check")
        
        # Update statistics
        self.maintenance_stats["health_check_runs"] += 1
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "issues_detected": 0,
            "issues_repaired": 0,
            "overall_health": "good"
        }
        
        # Track start time for performance measurement
        start_time = time.time()
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Run various health checks
        
        # 1. Node type consistency
        node_type_check = await self._check_node_type_consistency(core_graph)
        results["checks"].append({
            "name": "node_type_consistency",
            "result": node_type_check
        })
        
        if not node_type_check["consistent"]:
            results["issues_detected"] += 1
            if node_type_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # 2. Domain consistency
        domain_check = await self._check_domain_consistency(core_graph)
        results["checks"].append({
            "name": "domain_consistency",
            "result": domain_check
        })
        
        if not domain_check["consistent"]:
            results["issues_detected"] += 1
            if domain_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # 3. Edge type consistency
        edge_type_check = await self._check_edge_type_consistency(core_graph)
        results["checks"].append({
            "name": "edge_type_consistency",
            "result": edge_type_check
        })
        
        if not edge_type_check["consistent"]:
            results["issues_detected"] += 1
            if edge_type_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # 4. Count consistency
        count_check = await self._check_count_consistency(core_graph)
        results["checks"].append({
            "name": "count_consistency",
            "result": count_check
        })
        
        if not count_check["consistent"]:
            results["issues_detected"] += 1
            if count_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # 5. Check for orphaned nodes (nodes with no edges)
        orphan_check = await self._check_orphaned_nodes(core_graph)
        results["checks"].append({
            "name": "orphaned_nodes",
            "result": orphan_check
        })
        
        if orphan_check["orphaned_count"] > 0:
            results["issues_detected"] += 1
            if orphan_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # 6. Check for dangling edges (edges to non-existent nodes)
        dangling_check = await self._check_dangling_edges(core_graph)
        results["checks"].append({
            "name": "dangling_edges",
            "result": dangling_check
        })
        
        if dangling_check["dangling_count"] > 0:
            results["issues_detected"] += 1
            if dangling_check.get("repaired", False):
                results["issues_repaired"] += 1
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results["execution_time_seconds"] = execution_time
        
        # Update statistics
        self.maintenance_stats["health_issues_detected"] += results["issues_detected"]
        self.maintenance_stats["health_issues_repaired"] += results["issues_repaired"]
        
        # Determine overall health
        if results["issues_detected"] == 0:
            results["overall_health"] = "excellent"
        elif results["issues_detected"] <= 2:
            results["overall_health"] = "good"
        elif results["issues_detected"] <= 5:
            results["overall_health"] = "fair"
        else:
            results["overall_health"] = "poor"
        
        # Store health check history
        self.health_checks["health_history"].append({
            "timestamp": datetime.now().isoformat(),
            "issues_detected": results["issues_detected"],
            "issues_repaired": results["issues_repaired"],
            "overall_health": results["overall_health"]
        })
        
        # Limit health history size
        if len(self.health_checks["health_history"]) > 30:
            self.health_checks["health_history"] = self.health_checks["health_history"][-30:]
        
        # Update last check timestamp
        self.health_checks["last_check"] = datetime.now()
        
        self.logger.info(f"Completed health check in {execution_time:.2f} seconds: {results['issues_detected']} issues detected, {results['issues_repaired']} repaired")
        
        # Emit health check completed event
        await self.event_bus.emit("health_check_completed", {
            "timestamp": datetime.now().isoformat(),
            "issues_detected": results["issues_detected"],
            "issues_repaired": results["issues_repaired"],
            "overall_health": results["overall_health"]
        })
        
        return results
    
    async def _check_node_type_consistency(self, core_graph) -> Dict[str, Any]:
        """
        Check node type tracking consistency.
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        inconsistencies = []
        
        # Check if each node is properly tracked in its type
        for node_id, node_data in core_graph.graph.nodes(data=True):
            node_type = node_data.get("type", "unknown")
            
            if node_type in core_graph.node_types and node_id not in core_graph.node_types[node_type]:
                inconsistencies.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "issue": "node not in type tracking"
                })
        
        # Check if tracked nodes exist
        for node_type, nodes in core_graph.node_types.items():
            for node_id in nodes:
                if node_id not in core_graph.graph.nodes:
                    inconsistencies.append({
                        "node_id": node_id,
                        "node_type": node_type,
                        "issue": "tracked node doesn't exist"
                    })
        
        # Repair if needed and allowed
        repaired = False
        if inconsistencies and self.health_checks["auto_repair"]:
            await self._repair_node_type_consistency(core_graph, inconsistencies)
            repaired = True
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies[:100],  # Limit to 100 for report
            "total_inconsistencies": len(inconsistencies),
            "repaired": repaired
        }
    
    async def _repair_node_type_consistency(self, core_graph, inconsistencies: List[Dict[str, Any]]) -> int:
        """
        Repair node type tracking inconsistencies.
        
        Args:
            core_graph: Core graph module
            inconsistencies: List of inconsistencies to repair
            
        Returns:
            Number of repaired inconsistencies
        """
        repaired = 0
        
        for issue in inconsistencies:
            node_id = issue["node_id"]
            node_type = issue["node_type"]
            issue_type = issue["issue"]
            
            if issue_type == "node not in type tracking":
                # Add node to type tracking
                if node_id in core_graph.graph.nodes:
                    core_graph.node_types[node_type].add(node_id)
                    repaired += 1
            
            elif issue_type == "tracked node doesn't exist":
                # Remove node from type tracking
                core_graph.node_types[node_type].remove(node_id)
                repaired += 1
        
        return repaired
    
    async def _check_domain_consistency(self, core_graph) -> Dict[str, Any]:
        """
        Check domain tracking consistency.
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        inconsistencies = []
        
        # Check if each node is properly tracked in its domain
        for node_id, node_data in core_graph.graph.nodes(data=True):
            domain = node_data.get("domain", "general_knowledge")
            
            if node_id not in core_graph.domains[domain]:
                inconsistencies.append({
                    "node_id": node_id,
                    "domain": domain,
                    "issue": "node not in domain tracking"
                })
        
        # Check if tracked nodes exist
        for domain, nodes in core_graph.domains.items():
            for node_id in nodes:
                if node_id not in core_graph.graph.nodes:
                    inconsistencies.append({
                        "node_id": node_id,
                        "domain": domain,
                        "issue": "tracked node doesn't exist"
                    })
        
        # Repair if needed and allowed
        repaired = False
        if inconsistencies and self.health_checks["auto_repair"]:
            await self._repair_domain_consistency(core_graph, inconsistencies)
            repaired = True
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies[:100],  # Limit to 100 for report
            "total_inconsistencies": len(inconsistencies),
            "repaired": repaired
        }
    
    async def _repair_domain_consistency(self, core_graph, inconsistencies: List[Dict[str, Any]]) -> int:
        """
        Repair domain tracking inconsistencies.
        
        Args:
            core_graph: Core graph module
            inconsistencies: List of inconsistencies to repair
            
        Returns:
            Number of repaired inconsistencies
        """
        repaired = 0
        
        for issue in inconsistencies:
            node_id = issue["node_id"]
            domain = issue["domain"]
            issue_type = issue["issue"]
            
            if issue_type == "node not in domain tracking":
                # Add node to domain tracking
                if node_id in core_graph.graph.nodes:
                    core_graph.domains[domain].add(node_id)
                    repaired += 1
            
            elif issue_type == "tracked node doesn't exist":
                # Remove node from domain tracking
                core_graph.domains[domain].remove(node_id)
                repaired += 1
        
        return repaired
    
    async def _check_edge_type_consistency(self, core_graph) -> Dict[str, Any]:
        """
        Check edge type tracking consistency.
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        inconsistencies = []
        
        # Check if each edge type is properly tracked
        for source, target, key, data in core_graph.graph.edges(data=True, keys=True):
            edge_type = data.get("type", "unknown")
            
            if edge_type not in core_graph.edge_types:
                inconsistencies.append({
                    "source": source,
                    "target": target,
                    "edge_key": key,
                    "edge_type": edge_type,
                    "issue": "edge type not tracked"
                })
            
            # Check if edge is in type-specific tracking
            if (source, target, key) not in core_graph.edges_by_type[edge_type]:
                inconsistencies.append({
                    "source": source,
                    "target": target,
                    "edge_key": key,
                    "edge_type": edge_type,
                    "issue": "edge not in type tracking"
                })
        
        # Check if tracked edges exist
        for edge_type, edges in core_graph.edges_by_type.items():
            for source, target, key in edges:
                if not core_graph.graph.has_edge(source, target, key):
                    inconsistencies.append({
                        "source": source,
                        "target": target,
                        "edge_key": key,
                        "edge_type": edge_type,
                        "issue": "tracked edge doesn't exist"
                    })
        
        # Repair if needed and allowed
        repaired = False
        if inconsistencies and self.health_checks["auto_repair"]:
            await self._repair_edge_type_consistency(core_graph, inconsistencies)
            repaired = True
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies[:100],  # Limit to 100 for report
            "total_inconsistencies": len(inconsistencies),
            "repaired": repaired
        }
    
    async def _repair_edge_type_consistency(self, core_graph, inconsistencies: List[Dict[str, Any]]) -> int:
        """
        Repair edge type tracking inconsistencies.
        
        Args:
            core_graph: Core graph module
            inconsistencies: List of inconsistencies to repair
            
        Returns:
            Number of repaired inconsistencies
        """
        repaired = 0
        
        for issue in inconsistencies:
            source = issue["source"]
            target = issue["target"]
            key = issue["edge_key"]
            edge_type = issue["edge_type"]
            issue_type = issue["issue"]
            
            if issue_type == "edge type not tracked":
                # Add edge type to tracking
                core_graph.edge_types.add(edge_type)
                repaired += 1
            
            elif issue_type == "edge not in type tracking":
                # Add edge to type tracking
                if core_graph.graph.has_edge(source, target, key):
                    core_graph.edges_by_type[edge_type].append((source, target, key))
                    repaired += 1
            
            elif issue_type == "tracked edge doesn't exist":
                # Remove edge from type tracking
                if (source, target, key) in core_graph.edges_by_type[edge_type]:
                    core_graph.edges_by_type[edge_type].remove((source, target, key))
                    repaired += 1
        
        return repaired
    
    async def _check_count_consistency(self, core_graph) -> Dict[str, Any]:
        """
        Check node and edge count consistency.
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        # Get actual counts
        actual_node_count = core_graph.graph.number_of_nodes()
        actual_edge_count = core_graph.graph.number_of_edges()
        
        # Check tracking counts
        node_count_consistent = (actual_node_count == core_graph.total_nodes)
        edge_count_consistent = (actual_edge_count == core_graph.total_edges)
        
        # Repair if needed and allowed
        repaired = False
        if (not node_count_consistent or not edge_count_consistent) and self.health_checks["auto_repair"]:
            # Update tracking counts
            core_graph.total_nodes = actual_node_count
            core_graph.total_edges = actual_edge_count
            repaired = True
        
        return {
            "consistent": node_count_consistent and edge_count_consistent,
            "actual_node_count": actual_node_count,
            "tracked_node_count": core_graph.total_nodes,
            "node_count_consistent": node_count_consistent,
            "actual_edge_count": actual_edge_count,
            "tracked_edge_count": core_graph.total_edges,
            "edge_count_consistent": edge_count_consistent,
            "repaired": repaired
        }
    
    async def _check_orphaned_nodes(self, core_graph) -> Dict[str, Any]:
        """
        Check for orphaned nodes (nodes with no connections).
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        orphaned_nodes = []
        
        # Find nodes with no edges
        for node_id in core_graph.graph.nodes():
            if core_graph.graph.degree(node_id) == 0:
                # Get node data
                node_data = core_graph.graph.nodes[node_id]
                
                orphaned_nodes.append({
                    "node_id": node_id,
                    "node_type": node_data.get("type", "unknown"),
                    "domain": node_data.get("domain", "general_knowledge")
                })
        
        # Repair if needed and allowed
        repaired = False
        if orphaned_nodes and self.health_checks["auto_repair"]:
            # In this case, repair means connecting orphaned nodes to Lucidia
            # or removing them if they're less important
            await self._repair_orphaned_nodes(core_graph, orphaned_nodes)
            repaired = True
        
        return {
            "orphaned_count": len(orphaned_nodes),
            "orphaned_nodes": orphaned_nodes[:100],  # Limit to 100 for report
            "repaired": repaired
        }
    
    async def _repair_orphaned_nodes(self, core_graph, orphaned_nodes: List[Dict[str, Any]]) -> int:
        """
        Repair orphaned nodes.
        
        Args:
            core_graph: Core graph module
            orphaned_nodes: List of orphaned nodes to repair
            
        Returns:
            Number of repaired nodes
        """
        repaired = 0
        
        for node_info in orphaned_nodes:
            node_id = node_info["node_id"]
            node_type = node_info["node_type"]
            
            # Get node data
            node_data = await core_graph.get_node(node_id)
            if not node_data:
                continue
            
            # Check confidence - if low, remove node
            confidence = node_data.get("confidence", 0.5)
            
            if confidence < 0.4:
                # Low confidence, remove node
                await core_graph.remove_node(node_id)
            else:
                # Higher confidence, connect to Lucidia
                if await core_graph.has_node("Lucidia"):
                    relation_type = "references" if node_type == "concept" else "knows_about"
                    
                    await core_graph.add_edge(
                        "Lucidia",
                        node_id,
                        edge_type=relation_type,
                        attributes={
                            "strength": 0.5,
                            "confidence": 0.6,
                            "created": datetime.now().isoformat(),
                            "source": "maintenance_repair"
                        }
                    )
            
            repaired += 1
        
        return repaired
    
    async def _check_dangling_edges(self, core_graph) -> Dict[str, Any]:
        """
        Check for dangling edges (edges to non-existent nodes).
        
        Note: This should not happen with NetworkX, but we check just in case.
        
        Args:
            core_graph: Core graph module
            
        Returns:
            Check results
        """
        dangling_edges = []
        
        # This shouldn't happen with NetworkX, but check edge references anyway
        for edge_type, edges in core_graph.edges_by_type.items():
            for source, target, key in edges:
                if not core_graph.graph.has_node(source) or not core_graph.graph.has_node(target):
                    dangling_edges.append({
                        "source": source,
                        "target": target,
                        "edge_key": key,
                        "edge_type": edge_type
                    })
        
        # Repair if needed and allowed
        repaired = False
        if dangling_edges and self.health_checks["auto_repair"]:
            await self._repair_dangling_edges(core_graph, dangling_edges)
            repaired = True
        
        return {
            "dangling_count": len(dangling_edges),
            "dangling_edges": dangling_edges[:100],  # Limit to 100 for report
            "repaired": repaired
        }
    
    async def _repair_dangling_edges(self, core_graph, dangling_edges: List[Dict[str, Any]]) -> int:
        """
        Repair dangling edges.
        
        Args:
            core_graph: Core graph module
            dangling_edges: List of dangling edges to repair
            
        Returns:
            Number of repaired edges
        """
        repaired = 0
        
        for edge_info in dangling_edges:
            source = edge_info["source"]
            target = edge_info["target"]
            key = edge_info["edge_key"]
            edge_type = edge_info["edge_type"]
            
            # Remove from tracking
            if (source, target, key) in core_graph.edges_by_type[edge_type]:
                core_graph.edges_by_type[edge_type].remove((source, target, key))
                repaired += 1
        
        return repaired
    
    async def _analyze_graph_metrics(self) -> Dict[str, Any]:
        """
        Analyze key metrics of the knowledge graph.
        
        Returns:
            Dictionary of graph metrics
        """
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {}
        
        # Get basic metrics from core graph
        metrics = await core_graph.get_graph_metrics()
        
        # Calculate additional metrics
        metrics["node_type_distribution"] = {}
        for node_type, nodes in core_graph.node_types.items():
            metrics["node_type_distribution"][node_type] = len(nodes)
        
        metrics["edge_type_distribution"] = {}
        for edge_type, edges in core_graph.edges_by_type.items():
            metrics["edge_type_distribution"][edge_type] = len(edges)
        
        metrics["domain_distribution"] = {}
        for domain, nodes in core_graph.domains.items():
            metrics["domain_distribution"][domain] = len(nodes)
        
        # Calculate degree statistics
        if core_graph.total_nodes > 0:
            degrees = [degree for _, degree in core_graph.graph.degree()]
            metrics["max_degree"] = max(degrees) if degrees else 0
            metrics["min_degree"] = min(degrees) if degrees else 0
            metrics["median_degree"] = sorted(degrees)[len(degrees) // 2] if degrees else 0
        
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
        
        # Get metric thresholds
        density_threshold = self.adaptive_maintenance["metric_thresholds"]["density"]
        avg_degree_threshold = self.adaptive_maintenance["metric_thresholds"]["average_degree"]
        
        # Check if density is too high
        if metrics.get("density", 0) > density_threshold:
            # Graph is too dense, recommend pruning
            optimizations["prune_connections"] = True
            
            # Calculate pruning threshold based on density
            excess_density = metrics["density"] - density_threshold
            optimizations["prune_threshold"] = 0.3 + min(0.4, excess_density * 10)
        
        # Check if average degree is too high
        if metrics.get("average_degree", 0) > avg_degree_threshold:
            # Nodes have too many connections on average, recommend pruning
            optimizations["prune_connections"] = True
            
            # Set prune threshold if not already set
            if "prune_threshold" not in optimizations:
                excess_degree = metrics["average_degree"] - avg_degree_threshold
                optimizations["prune_threshold"] = 0.3 + min(0.4, excess_degree / 10)
        
        # Check for high degree nodes
        if metrics.get("max_degree", 0) > avg_degree_threshold * 3:
            # Some nodes have very high degree, recommend reindexing
            optimizations["reindex_nodes"] = True
            
            # Calculate number of nodes to reindex
            # Estimate that roughly 5% of nodes are high-degree
            reindex_count = max(10, int(metrics.get("total_nodes", 100) * 0.05))
            optimizations["reindex_count"] = min(100, reindex_count)  # Cap at 100
        
        # Check if dream influence has changed significantly
        # This would require data from dream_integration module
        dream_integration = self.module_registry.get_module("dream_integration")
        if dream_integration:
            # Check if dream-related decay needs adjustment
            if hasattr(dream_integration, "dream_integration") and "dream_insight_count" in dream_integration.dream_integration:
                # If we have many dream insights, adjust decay
                if dream_integration.dream_integration["dream_insight_count"] > 10:
                    optimizations["adjust_decay"] = True
                    optimizations["decay_adjustments"] = {
                        "dream_associated": 0.02
                    }
        
        # Check if graph size warrants data compression
        if metrics.get("total_nodes", 0) > 10000 or metrics.get("total_edges", 0) > 50000:
            optimizations["compress_data"] = True
        
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
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Track edges to remove by type
        edges_to_remove = []
        pruned_by_type = defaultdict(int)
        
        # Identify edges below threshold
        for source, target, key, data in core_graph.graph.edges(data=True, keys=True):
            # Skip edges without strength
            if "strength" not in data:
                continue
            
            # Check strength against threshold
            if data["strength"] < threshold:
                edges_to_remove.append((source, target, key))
                
                # Track by type
                edge_type = data.get("type", "unknown")
                pruned_by_type[edge_type] += 1
        
        # Remove edges
        for source, target, key in edges_to_remove:
            await core_graph.remove_edge(source, target, key)
        
        # Track pruning count
        pruned_count = len(edges_to_remove)
        
        self.logger.info(f"Pruned {pruned_count} low-value connections")
        
        return {
            "success": True,
            "pruned_count": pruned_count,
            "threshold": threshold,
            "pruned_by_type": dict(pruned_by_type)
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
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Calculate node degrees
        node_degrees = dict(core_graph.graph.degree())
        
        # Sort nodes by degree (descending)
        high_degree_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 'count' nodes
        nodes_to_reindex = [node for node, _ in high_degree_nodes[:count]]
        
        # For each node, optimize its connections
        reindexed_nodes = []
        
        for node_id in nodes_to_reindex:
            # Get all neighbors
            neighbors = list(core_graph.graph.neighbors(node_id))
            
            # Group neighbors by node type
            type_groups = defaultdict(list)
            
            for neighbor in neighbors:
                # Skip if already in primary nodes
                if neighbor in nodes_to_reindex:
                    continue
                
                # Get neighbor type
                neighbor_data = await core_graph.get_node(neighbor)
                if not neighbor_data:
                    continue
                
                neighbor_type = neighbor_data.get("type", "unknown")
                type_groups[neighbor_type].append(neighbor)
            
            # For each group, find representative connections
            representative_connections = []
            
            for node_type, nodes in type_groups.items():
                # Get existing connections for calculation
                node_connections = {}
                for n in nodes:
                    edges = await core_graph.get_edges(node_id, n)
                    node_connections[n] = edges
                
                # Keep direct connections to most relevant nodes
                # Here we use the count of existing connections as a relevance proxy
                relevance_scores = []
                
                for n, edges in node_connections.items():
                    # Calculate relevance based on edge strength
                    avg_strength = sum(edge.get("strength", 0.5) for edge in edges) / max(1, len(edges))
                    relevance_scores.append((n, avg_strength))
                
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
                    edges = await core_graph.get_edges(node_id, neighbor)
                    
                    for edge in edges:
                        edges_to_remove.append((node_id, neighbor, edge["key"]))
            
            # Remove redundant edges
            for source, target, key in edges_to_remove:
                await core_graph.remove_edge(source, target, key)
            
            reindexed_nodes.append({
                "node_id": node_id,
                "neighbors_before": len(neighbors),
                "neighbors_after": len(representative_connections),
                "edges_removed": len(edges_to_remove)
            })
        
        self.logger.info(f"Reindexed {len(reindexed_nodes)} high-frequency nodes")
        
        return {
            "success": True,
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
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        results = {
            "success": True,
            "previous_values": dict(core_graph.relationship_decay),
            "new_values": {}
        }
        
        # Apply adjustments
        for decay_type, new_value in adjustments.items():
            if decay_type in core_graph.relationship_decay:
                # Store previous value
                previous = core_graph.relationship_decay[decay_type]
                
                # Update value
                core_graph.relationship_decay[decay_type] = new_value
                
                # Record change
                results["new_values"][decay_type] = new_value
                
                self.logger.info(f"Adjusted {decay_type} decay: {previous} -> {new_value}")
        
        return results
    
    async def _compress_graph_data(self) -> Dict[str, Any]:
        """
        Compress graph data to reduce memory usage.
        
        Returns:
            Compression results
        """
        self.logger.info("Compressing graph data")
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        compression_stats = {
            "nodes_processed": 0,
            "edges_processed": 0,
            "storage_info_before": {"node_attrs": 0, "edge_attrs": 0},
            "storage_info_after": {"node_attrs": 0, "edge_attrs": 0}
        }
        
        # Measure storage before
        node_attr_size = 0
        for _, data in core_graph.graph.nodes(data=True):
            node_attr_size += len(str(data))
        
        edge_attr_size = 0
        for _, _, data in core_graph.graph.edges(data=True):
            edge_attr_size += len(str(data))
        
        compression_stats["storage_info_before"]["node_attrs"] = node_attr_size
        compression_stats["storage_info_before"]["edge_attrs"] = edge_attr_size
        
        # Process nodes
        for node_id, data in core_graph.graph.nodes(data=True):
            compression_stats["nodes_processed"] += 1
            
            # Compress string attributes
            for key, value in list(data.items()):
                if isinstance(value, str) and len(value) > 100:
                    # Truncate long text attributes that aren't essential
                    if key not in ["id", "type", "name", "domain"]:
                        data[key] = value[:100] + "..."
        
        # Process edges
        for source, target, key, data in core_graph.graph.edges(data=True, keys=True):
            compression_stats["edges_processed"] += 1
            
            # Remove redundant attributes
            for key in list(data.keys()):
                if key in ["modified"] and "created" in data:
                    # Keep created but remove modified
                    del data[key]
        
        # Measure storage after
        node_attr_size = 0
        for _, data in core_graph.graph.nodes(data=True):
            node_attr_size += len(str(data))
        
        edge_attr_size = 0
        for _, _, data in core_graph.graph.edges(data=True):
            edge_attr_size += len(str(data))
        
        compression_stats["storage_info_after"]["node_attrs"] = node_attr_size
        compression_stats["storage_info_after"]["edge_attrs"] = edge_attr_size
        
        # Calculate compression ratio
        before_size = (compression_stats["storage_info_before"]["node_attrs"] + 
                      compression_stats["storage_info_before"]["edge_attrs"])
        after_size = (compression_stats["storage_info_after"]["node_attrs"] + 
                     compression_stats["storage_info_after"]["edge_attrs"])
        
        compression_ratio = before_size / after_size if after_size > 0 else 1
        compression_stats["compression_ratio"] = compression_ratio
        
        self.logger.info(f"Compressed graph data with ratio: {compression_ratio:.2f}")
        
        return {
            "success": True,
            "stats": compression_stats
        }
    
    async def _store_optimization_learnings(self, results: Dict[str, Any]) -> None:
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
        if len(self.adaptive_maintenance["optimization_history"]) > 30:
            self.adaptive_maintenance["optimization_history"] = self.adaptive_maintenance["optimization_history"][-30:]
        
        self.logger.info("Stored optimization learnings")
    
    async def set_maintenance_interval(self, interval_hours: int) -> Dict[str, Any]:
        """
        Set the maintenance interval.
        
        Args:
            interval_hours: Maintenance interval in hours
            
        Returns:
            Update result
        """
        if interval_hours < 1:
            return {"success": False, "error": "Interval must be at least 1 hour"}
        
        previous = self.adaptive_maintenance["maintenance_interval"] / 3600
        self.adaptive_maintenance["maintenance_interval"] = interval_hours * 3600
        
        return {
            "success": True,
            "previous_interval_hours": previous,
            "new_interval_hours": interval_hours
        }
    
    async def toggle_maintenance(self, enabled: bool) -> Dict[str, Any]:
        """
        Toggle maintenance on or off.
        
        Args:
            enabled: Whether maintenance is enabled
            
        Returns:
            Toggle result
        """
        previous = self.adaptive_maintenance["maintenance_enabled"]
        self.adaptive_maintenance["maintenance_enabled"] = enabled
        
        return {
            "success": True,
            "previous": previous,
            "current": enabled
        }
    
    async def toggle_auto_maintenance(self, enabled: bool) -> Dict[str, Any]:
        """
        Toggle automatic maintenance on or off.
        
        Args:
            enabled: Whether automatic maintenance is enabled
            
        Returns:
            Toggle result
        """
        previous = self.adaptive_maintenance["auto_maintenance"]
        self.adaptive_maintenance["auto_maintenance"] = enabled
        
        return {
            "success": True,
            "previous": previous,
            "current": enabled
        }
    
    async def get_maintenance_stats(self) -> Dict[str, Any]:
        """
        Get maintenance statistics.
        
        Returns:
            Dictionary with maintenance statistics
        """
        return {
            "total_maintenance_runs": self.maintenance_stats["total_maintenance_runs"],
            "total_connections_pruned": self.maintenance_stats["total_connections_pruned"],
            "total_nodes_reindexed": self.maintenance_stats["total_nodes_reindexed"],
            "total_optimizations": self.maintenance_stats["total_optimizations"],
            "maintenance_enabled": self.adaptive_maintenance["maintenance_enabled"],
            "auto_maintenance": self.adaptive_maintenance["auto_maintenance"],
            "maintenance_interval_hours": self.adaptive_maintenance["maintenance_interval"] / 3600,
            "last_maintenance": self.adaptive_maintenance["last_maintenance"].isoformat(),
            "next_scheduled_maintenance": (
                self.adaptive_maintenance["last_maintenance"] + 
                timedelta(seconds=self.adaptive_maintenance["maintenance_interval"])
            ).isoformat(),
            "health_check_runs": self.maintenance_stats["health_check_runs"],
            "health_issues_detected": self.maintenance_stats["health_issues_detected"],
            "health_issues_repaired": self.maintenance_stats["health_issues_repaired"],
            "health_check_enabled": self.health_checks["check_enabled"],
            "auto_repair": self.health_checks["auto_repair"],
            "health_check_interval_hours": self.health_checks["check_interval"] / 3600,
            "last_health_check": self.health_checks["last_check"].isoformat(),
            "next_scheduled_check": (
                self.health_checks["last_check"] + 
                timedelta(seconds=self.health_checks["check_interval"])
            ).isoformat()
        }