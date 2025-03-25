"""
Contradiction Manager Module for Lucidia's Knowledge Graph

This module implements contradiction detection algorithms, consistency evaluation,
resolution strategy determination, and confidence-based reconciliation.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict

from .base_module import KnowledgeGraphModule

class ContradictionManager(KnowledgeGraphModule):
    """
    Contradiction Manager responsible for handling contradictions in the knowledge graph.
    
    This module detects and resolves contradictions between nodes and relationships,
    using confidence-based resolution strategies and maintaining a resolution history.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Contradiction Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # Contradiction tracking configuration
        self.contradiction_tracking = {
            "detected_contradictions": [],  # List of detected contradictions
            "resolution_history": [],  # History of resolutions
            "confidence_threshold": self.get_config("confidence_threshold", 0.7),  # Threshold for contradiction detection
            "auto_resolution_threshold": self.get_config("auto_resolution_threshold", 0.9),  # Threshold for automatic resolution
            "contradiction_types": {
                "conflicting_relationship": {
                    "description": "Relationships between the same nodes that contradict each other",
                    "resolution_priority": 1
                },
                "conflicting_attribute": {
                    "description": "Node attributes that contradict each other",
                    "resolution_priority": 2
                },
                "logical_inconsistency": {
                    "description": "Logical contradictions in the knowledge structure",
                    "resolution_priority": 3
                }
            }
        }
        
        # Statistics tracking
        self.contradiction_stats = {
            "detected_count": 0,
            "resolved_count": 0,
            "auto_resolved_count": 0,
            "manual_resolved_count": 0,
            "by_type": defaultdict(int)
        }
        
        self.logger.info("Contradiction Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("contradiction_detected", self.handle_contradiction)
        await self.event_bus.subscribe("external_resolution_applied", self._handle_external_resolution)
        await self.event_bus.subscribe("edge_added", self._check_edge_contradictions)
        await self.event_bus.subscribe("node_updated", self._check_node_contradictions)
        self.logger.info("Subscribed to contradiction-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # No special setup needed
        pass
    
    async def handle_contradiction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a detected contradiction.
        
        Args:
            data: Contradiction data
            
        Returns:
            Handling result
        """
        self.logger.info(f"Handling contradiction: {data.get('type', 'unknown')}")
        
        # Get contradiction details
        contradiction_type = data.get("type", "unknown")
        source_id = data.get("source", "")
        target_id = data.get("target", "")
        context = data.get("context", {})
        
        # Update statistics
        self.contradiction_stats["detected_count"] += 1
        self.contradiction_stats["by_type"][contradiction_type] += 1
        
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
        
        # Update contradiction record
        contradiction_record["analysis"] = analysis_result
        contradiction_record["status"] = "analyzed"
        
        # Determine if we can auto-resolve
        can_resolve = analysis_result.get("can_resolve", False)
        resolution_method = analysis_result.get("resolution_method", "external")
        
        if can_resolve and resolution_method == "internal":
            # Apply internal resolution
            resolution_result = await self._apply_internal_resolution(
                source_id, target_id, contradiction_type, analysis_result
            )
            
            # Update contradiction record
            contradiction_record["resolution"] = resolution_result
            contradiction_record["status"] = "resolved_internally"
            
            # Update statistics
            self.contradiction_stats["resolved_count"] += 1
            self.contradiction_stats["auto_resolved_count"] += 1
            
            # Emit resolution event
            await self.event_bus.emit("contradiction_resolved", {
                "contradiction_id": contradiction_id,
                "resolution": resolution_result,
                "method": "internal"
            })
            
            return {
                "success": True,
                "contradiction_id": contradiction_id,
                "method": "internal",
                "resolution": resolution_result
            }
        else:
            # Request external resolution
            contradiction_record["status"] = "awaiting_resolution"
            
            # Emit event requesting resolution
            await self.event_bus.emit("resolution_required", {
                "contradiction_id": contradiction_id,
                "source": source_id,
                "target": target_id,
                "type": contradiction_type,
                "analysis": analysis_result,
                "context": context
            })
            
            return {
                "success": True,
                "contradiction_id": contradiction_id,
                "method": "external",
                "requires_resolution": True
            }
    
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
        # Get core graph for node/edge operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {
                "can_resolve": False,
                "resolution_method": "external",
                "reason": "Core graph module not found"
            }
        
        # Get node data
        source_node = await core_graph.get_node(source_id)
        target_node = await core_graph.get_node(target_id)
        
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
        if await core_graph.has_edge(source_id, target_id):
            edge_data = await core_graph.get_edges(source_id, target_id)
        
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
            source_created = source_node.get("created", "")
            target_created = target_node.get("created", "")
            
            # Try to resolve based on recency
            if source_created and target_created:
                try:
                    source_time = datetime.fromisoformat(source_created)
                    target_time = datetime.fromisoformat(target_created)
                    
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
            
            # If recency check fails, try confidence check
            if confidence_diff > 0.3:
                # Significant confidence difference, can resolve internally
                return {
                    "can_resolve": True,
                    "resolution_method": "internal",
                    "higher_confidence": source_id if source_confidence > target_confidence else target_id,
                    "confidence_diff": confidence_diff,
                    "strategy": "trust_higher_confidence"
                }
            
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
        # Get core graph for node/edge operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        strategy = analysis.get("strategy", "unknown")
        
        if strategy == "trust_higher_confidence":
            # Trust the node with higher confidence
            higher_confidence_id = analysis.get("higher_confidence", "")
            lower_confidence_id = source_id if higher_confidence_id == target_id else target_id
            
            if contradiction_type == "conflicting_relationship":
                # For relationship contradictions, adjust edge attributes
                edges = await core_graph.get_edges(source_id, target_id)
                
                for edge in edges:
                    edge_key = edge.get("key")
                    await core_graph.update_edge(source_id, target_id, edge_key, {
                        "confidence": edge.get("confidence", 0.5) * 0.8,  # Reduce confidence
                        "strength": edge.get("strength", 0.5) * 0.8,  # Reduce strength
                        "contradiction_resolved": True,
                        "resolution_method": "trust_higher_confidence",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
            
            elif contradiction_type == "conflicting_attribute":
                # For attribute contradictions, update the lower confidence node
                lower_node = await core_graph.get_node(lower_confidence_id)
                if lower_node:
                    await core_graph.update_node(lower_confidence_id, {
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
                older_node = await core_graph.get_node(older_node_id)
                if older_node:
                    await core_graph.update_node(older_node_id, {
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
    
    async def _handle_external_resolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle external resolution decision.
        
        Args:
            data: External resolution data
            
        Returns:
            Resolution result
        """
        contradiction_id = data.get("contradiction_id", "")
        resolution_action = data.get("action", "")
        resolution_data = data.get("data", {})
        
        if not contradiction_id:
            return {"success": False, "error": "Contradiction ID required"}
        
        # Find the contradiction record
        contradiction_record = None
        for record in self.contradiction_tracking["detected_contradictions"]:
            if record.get("id") == contradiction_id:
                contradiction_record = record
                break
        
        if not contradiction_record:
            self.logger.warning(f"Contradiction record not found: {contradiction_id}")
            return {"success": False, "error": "Contradiction record not found"}
        
        # Get node IDs
        source_id = contradiction_record.get("source", "")
        target_id = contradiction_record.get("target", "")
        
        # Apply resolution
        result = await self.apply_external_resolution(
            contradiction_id, 
            resolution_action, 
            source_id, 
            target_id, 
            resolution_data
        )
        
        # Update statistics
        if result.get("success", False):
            self.contradiction_stats["resolved_count"] += 1
            self.contradiction_stats["manual_resolved_count"] += 1
        
        return result
    
    async def apply_external_resolution(self, contradiction_id: str, action: str, 
                                source_id: str, target_id: str, 
                                resolution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply external resolution to a contradiction.
        
        Args:
            contradiction_id: Contradiction ID
            action: Resolution action
            source_id: Source node ID
            target_id: Target node ID
            resolution_data: Resolution data
            
        Returns:
            Resolution result
        """
        # Get core graph for node/edge operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
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
        if action == "update_source":
            # Update source node
            source_updates = resolution_data.get("updates", {})
            if source_updates and await core_graph.has_node(source_id):
                await core_graph.update_node(source_id, source_updates)
                
                # Add resolution metadata
                await core_graph.update_node(source_id, {
                    "contradiction_resolved": True,
                    "resolution_method": "external_update",
                    "resolution_timestamp": datetime.now().isoformat()
                })
        
        elif action == "update_target":
            # Update target node
            target_updates = resolution_data.get("updates", {})
            if target_updates and await core_graph.has_node(target_id):
                await core_graph.update_node(target_id, target_updates)
                
                # Add resolution metadata
                await core_graph.update_node(target_id, {
                    "contradiction_resolved": True,
                    "resolution_method": "external_update",
                    "resolution_timestamp": datetime.now().isoformat()
                })
        
        elif action == "create_relationship":
            # Create a new relationship
            relationship = resolution_data.get("relationship", {})
            if relationship and await core_graph.has_node(source_id) and await core_graph.has_node(target_id):
                rel_type = relationship.get("type", "related_to")
                attributes = relationship.get("attributes", {})
                
                # Add resolution metadata
                attributes.update({
                    "contradiction_resolved": True,
                    "resolution_method": "external_create",
                    "resolution_timestamp": datetime.now().isoformat()
                })
                
                await core_graph.add_edge(source_id, target_id, rel_type, attributes)
        
        elif action == "delete_relationship":
            # Delete an existing relationship
            if await core_graph.has_edge(source_id, target_id):
                edge_key = resolution_data.get("edge_key")
                if edge_key is not None:
                    await core_graph.remove_edge(source_id, target_id, edge_key)
                else:
                    # Remove all edges between these nodes
                    edges = await core_graph.get_edges(source_id, target_id)
                    for edge in edges:
                        await core_graph.remove_edge(source_id, target_id, edge.get("key"))
        
        elif action == "merge_nodes":
            # Merge two nodes
            merge_target = resolution_data.get("merge_target", source_id)
            merge_source = target_id if merge_target == source_id else source_id
            
            await self._merge_nodes(merge_source, merge_target, core_graph)
        
        else:
            # Unknown action
            return {"success": False, "error": f"Unknown resolution action: {action}"}
        
        # Update contradiction record
        contradiction_record["status"] = "resolved_externally"
        contradiction_record["resolution"] = {
            "action": action,
            "data": resolution_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to resolution history
        self.contradiction_tracking["resolution_history"].append({
            "contradiction_id": contradiction_id,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
        
        # Emit knowledge consistent event
        await self.event_bus.emit("knowledge_consistent", {
            "contradiction_id": contradiction_id,
            "action": action
        })
        
        return {"success": True, "action": action}
    
    async def _merge_nodes(self, source_id: str, target_id: str, core_graph) -> bool:
        """
        Merge source node into target node.
        
        Args:
            source_id: Source node ID to merge from
            target_id: Target node ID to merge into
            core_graph: Core graph module for operations
            
        Returns:
            Success status
        """
        if not await core_graph.has_node(source_id) or not await core_graph.has_node(target_id):
            return False
        
        source_node = await core_graph.get_node(source_id)
        target_node = await core_graph.get_node(target_id)
        
        if not source_node or not target_node:
            return False
        
        # Merge attributes
        merged_attributes = target_node.copy()
        
        # Keep higher confidence values from source
        for key, value in source_node.items():
            if key not in merged_attributes or (
                key in merged_attributes and
                isinstance(value, (int, float)) and
                isinstance(merged_attributes[key], (int, float)) and
                value > merged_attributes[key]
            ):
                merged_attributes[key] = value
        
        # Add merged_from attribute
        if "merged_from" not in merged_attributes:
            merged_attributes["merged_from"] = []
        if isinstance(merged_attributes["merged_from"], list):
            merged_attributes["merged_from"].append(source_id)
        
        # Add resolution metadata
        merged_attributes.update({
            "contradiction_resolved": True,
            "resolution_method": "node_merge",
            "resolution_timestamp": datetime.now().isoformat()
        })
        
        # Update target node
        await core_graph.update_node(target_id, merged_attributes)
        
        # Redirect all edges from source to target
        # Get outgoing edges from source
        source_neighbors = await core_graph.get_connected_nodes(source_id, direction="outgoing")
        for neighbor in source_neighbors:
            if neighbor != target_id:  # Avoid self-loops
                edges = await core_graph.get_edges(source_id, neighbor)
                for edge in edges:
                    # Create edge from target to neighbor
                    edge_attrs = edge.copy()
                    if "key" in edge_attrs:
                        del edge_attrs["key"]
                    
                    # Add merge metadata
                    edge_attrs.update({
                        "merged_from": source_id,
                        "resolution_method": "edge_merge",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
                    
                    await core_graph.add_edge(target_id, neighbor, edge.get("type", "related_to"), edge_attrs)
        
        # Get incoming edges to source
        source_incoming = await core_graph.get_connected_nodes(source_id, direction="incoming")
        for neighbor in source_incoming:
            if neighbor != target_id:  # Avoid self-loops
                edges = await core_graph.get_edges(neighbor, source_id)
                for edge in edges:
                    # Create edge from neighbor to target
                    edge_attrs = edge.copy()
                    if "key" in edge_attrs:
                        del edge_attrs["key"]
                    
                    # Add merge metadata
                    edge_attrs.update({
                        "merged_from": source_id,
                        "resolution_method": "edge_merge",
                        "resolution_timestamp": datetime.now().isoformat()
                    })
                    
                    await core_graph.add_edge(neighbor, target_id, edge.get("type", "related_to"), edge_attrs)
        
        # Remove source node
        await core_graph.remove_node(source_id)
        
        return True
    
    async def _check_edge_contradictions(self, data: Dict[str, Any]) -> None:
        """
        Check for contradictions when an edge is added.
        
        Args:
            data: Edge added event data
        """
        source = data.get("source")
        target = data.get("target")
        edge_type = data.get("edge_type")
        
        if not all([source, target, edge_type]):
            return
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return
        
        # Check for contradictory edges
        edges = await core_graph.get_edges(source, target)
        
        # If there's only one edge, no contradiction
        if len(edges) <= 1:
            return
        
        # Check for conflicting relationship types
        edge_types = set(edge.get("type") for edge in edges)
        contradictory_types = self._get_contradictory_types(edge_types)
        
        if contradictory_types:
            # We found a contradiction
            self.logger.info(f"Detected contradictory edge types: {contradictory_types}")
            
            # Emit contradiction detected event
            await self.event_bus.emit("contradiction_detected", {
                "type": "conflicting_relationship",
                "source": source,
                "target": target,
                "context": {
                    "edge_types": list(edge_types),
                    "contradictory_types": list(contradictory_types),
                    "total_edges": len(edges)
                }
            })
    
    async def _check_node_contradictions(self, data: Dict[str, Any]) -> None:
        """
        Check for contradictions when a node is updated.
        
        Args:
            data: Node updated event data
        """
        node_id = data.get("node_id")
        attributes = data.get("attributes", {})
        
        if not node_id or not attributes:
            return
        
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return
        
        # Get node neighbors
        neighbors = await core_graph.get_connected_nodes(node_id)
        
        for neighbor in neighbors:
            # Check for attribute conflicts
            await self._check_attribute_conflicts(node_id, neighbor)
    
    async def _check_attribute_conflicts(self, node1_id: str, node2_id: str) -> None:
        """
        Check for conflicting attributes between nodes.
        
        Args:
            node1_id: First node ID
            node2_id: Second node ID
        """
        # Get core graph for operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return
        
        # Get node data
        node1_data = await core_graph.get_node(node1_id)
        node2_data = await core_graph.get_node(node2_id)
        
        if not node1_data or not node2_data:
            return
        
        # Check for conflicting attributes
        conflicts = []
        
        # Check defining attributes
        if node1_data.get("type") == node2_data.get("type"):
            if node1_data.get("type") == "concept":
                # For concepts, check definitions
                def1 = node1_data.get("definition", "")
                def2 = node2_data.get("definition", "")
                
                if def1 and def2 and self._are_contradictory_definitions(def1, def2):
                    conflicts.append({
                        "attribute": "definition",
                        "values": [def1, def2]
                    })
            
            elif node1_data.get("type") == "entity":
                # For entities, check descriptions
                desc1 = node1_data.get("description", "")
                desc2 = node2_data.get("description", "")
                
                if desc1 and desc2 and self._are_contradictory_descriptions(desc1, desc2):
                    conflicts.append({
                        "attribute": "description",
                        "values": [desc1, desc2]
                    })
        
        # If conflicts found, emit contradiction event
        if conflicts:
            self.logger.info(f"Detected attribute conflicts between {node1_id} and {node2_id}")
            
            # Emit contradiction detected event
            await self.event_bus.emit("contradiction_detected", {
                "type": "conflicting_attribute",
                "source": node1_id,
                "target": node2_id,
                "context": {
                    "conflicts": conflicts,
                    "node1_type": node1_data.get("type"),
                    "node2_type": node2_data.get("type")
                }
            })
    
    def _get_contradictory_types(self, edge_types: Set[str]) -> Set[str]:
        """
        Identify contradictory edge types.
        
        Args:
            edge_types: Set of edge types
            
        Returns:
            Set of contradictory types
        """
        # Define contradictory type pairs
        contradictory_pairs = [
            {"is_a", "is_not_a"},
            {"part_of", "separate_from"},
            {"greater_than", "less_than"},
            {"causes", "prevents"},
            {"supports", "opposes"},
            {"before", "after"},
            {"requires", "excludes"},
            {"parent_of", "child_of"}
        ]
        
        # Check if any contradictory pairs exist in edge types
        contradictory = set()
        for pair in contradictory_pairs:
            if len(pair.intersection(edge_types)) > 1:
                contradictory.update(pair.intersection(edge_types))
        
        return contradictory
    
    def _are_contradictory_definitions(self, def1: str, def2: str) -> bool:
        """
        Check if two concept definitions are contradictory.
        
        Args:
            def1: First definition
            def2: Second definition
            
        Returns:
            True if contradictory, False otherwise
        """
        # Simple check for contradictory keywords
        contradictory_pairs = [
            ("always", "never"),
            ("must", "cannot"),
            ("is", "is not"),
            ("all", "none"),
            ("required", "optional"),
            ("positive", "negative"),
            ("true", "false")
        ]
        
        for term1, term2 in contradictory_pairs:
            if term1 in def1.lower() and term2 in def2.lower():
                return True
            if term2 in def1.lower() and term1 in def2.lower():
                return True
        
        return False
    
    def _are_contradictory_descriptions(self, desc1: str, desc2: str) -> bool:
        """
        Check if two entity descriptions are contradictory.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            True if contradictory, False otherwise
        """
        # Similar to contradictory definitions
        return self._are_contradictory_definitions(desc1, desc2)
    
    async def get_contradictions(self, status: Optional[str] = None, 
                          contradiction_type: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get contradictions from the tracking system.
        
        Args:
            status: Optional filter by status
            contradiction_type: Optional filter by type
            limit: Maximum number of results
            
        Returns:
            List of contradictions
        """
        results = []
        
        for record in self.contradiction_tracking["detected_contradictions"]:
            # Apply filters
            if status and record.get("status") != status:
                continue
                
            if contradiction_type and record.get("type") != contradiction_type:
                continue
                
            results.append(record)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def get_contradiction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about contradictions.
        
        Returns:
            Dictionary with contradiction statistics
        """
        # Count by status
        status_counts = defaultdict(int)
        for record in self.contradiction_tracking["detected_contradictions"]:
            status = record.get("status", "unknown")
            status_counts[status] += 1
        
        return {
            "detected_count": self.contradiction_stats["detected_count"],
            "resolved_count": self.contradiction_stats["resolved_count"],
            "auto_resolved_count": self.contradiction_stats["auto_resolved_count"],
            "manual_resolved_count": self.contradiction_stats["manual_resolved_count"],
            "unresolved_count": self.contradiction_stats["detected_count"] - self.contradiction_stats["resolved_count"],
            "by_type": dict(self.contradiction_stats["by_type"]),
            "by_status": dict(status_counts),
            "confidence_threshold": self.contradiction_tracking["confidence_threshold"],
            "auto_resolution_threshold": self.contradiction_tracking["auto_resolution_threshold"]
        }