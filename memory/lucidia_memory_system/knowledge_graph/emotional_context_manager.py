"""
Emotional Context Manager for Lucidia's Knowledge Graph

This module handles emotional context integration, analysis, and tracking,
providing affective dimensions to the knowledge graph.
"""

import logging
import uuid
import re
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict

from .base_module import KnowledgeGraphModule

class EmotionalContextManager(KnowledgeGraphModule):
    """
    Emotional Context Manager for integrating emotional dimensions into the knowledge graph.
    
    This module handles emotion analysis and extraction, affective context integration,
    emotional relationship tracking, and multi-dimensional sentiment mapping.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Emotional Context Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # Emotion integration configuration
        self.emotion_integration = {
            "emotion_incorporation_rate": self.get_config("emotion_incorporation_rate", 0.85),
            "emotion_association_strength": self.get_config("emotion_association_strength", 0.75),
            "emotion_derived_nodes": set(),
            "emotion_enhanced_nodes": set(),
            "primary_emotions": self.get_config("primary_emotions", 
                                               ["joy", "sadness", "anger", "fear", 
                                                "disgust", "surprise", "neutral"]),
            "emotion_count": 0,
            "emotion_thresholds": self.get_config("emotion_thresholds", 
                                                 {"dominant": 0.5, "secondary": 0.25})
        }
        
        # Track nodes with emotional context
        self.emotion_enhanced_nodes = set()
        
        # Emotion color mapping for visualization
        self.emotion_colors = self.get_config("emotion_colors", {
            "joy": "#FFD700",  # Gold
            "sadness": "#4682B4",  # Steel Blue
            "anger": "#FF4500",  # Red Orange
            "fear": "#800080",  # Purple
            "disgust": "#006400",  # Dark Green
            "surprise": "#FF69B4",  # Hot Pink
            "neutral": "#CCCCCC"  # Light Gray
        })
        
        # Emotion analyzer service configuration
        self.emotion_analyzer_url = self.get_config("emotion_analyzer_url")
        self.emotion_analyzer_enabled = self.emotion_analyzer_url is not None
        self.emotion_analyzer_ws = None
        
        # Emotion pattern for simulated analysis
        self.emotion_patterns = {
            "joy": [r"happiness", r"joy", r"delight", r"pleasure", r"content", r"satisfied", r"happy", r"excited", r"love"],
            "sadness": [r"sad", r"grief", r"sorrow", r"depression", r"despair", r"melancholy", r"misery", r"unhappy", r"regret"],
            "anger": [r"anger", r"rage", r"fury", r"hostile", r"irritate", r"annoyed", r"hate", r"resent", r"wrath"],
            "fear": [r"fear", r"anxiety", r"worry", r"terror", r"dread", r"panic", r"scared", r"frightened", r"phobia"],
            "disgust": [r"disgust", r"revulsion", r"aversion", r"distaste", r"repulsed", r"loathing", r"repel", r"gross"],
            "surprise": [r"surprise", r"astonishment", r"amazement", r"shock", r"wonder", r"stun", r"unexpected", r"startled"]
        }
        
        # Cache for frequent emotion analyses
        self.emotion_cache = {}
        self.cache_size_limit = self.get_config("cache_size_limit", 1000)
        
        self.logger.info("Emotional Context Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("emotional_context_requested", self._handle_emotional_context_request)
        await self.event_bus.subscribe("analyze_emotion_requested", self._handle_analyze_emotion_request)
        await self.event_bus.subscribe("node_added", self._handle_node_added)
        self.logger.info("Subscribed to emotion-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # Register operation handlers
        self.module_registry.register_operation_handler("analyze_emotion", self.analyze_emotion)
        self.module_registry.register_operation_handler("add_emotional_context", self.add_emotional_context)
        self.module_registry.register_operation_handler("get_emotional_nodes", self.get_emotional_nodes)
        
        # Try to connect to emotion analyzer service if configured
        if self.emotion_analyzer_enabled:
            try:
                await self._connect_to_emotion_analyzer()
            except Exception as e:
                self.logger.warning(f"Could not connect to emotion analyzer service: {e}")
                self.emotion_analyzer_enabled = False
        
        self.logger.info("Emotional Context Manager setup complete")
    
    async def _connect_to_emotion_analyzer(self):
        """
        Connect to the emotion analyzer service.
        
        This would establish a connection to an external emotion analysis service.
        For the simplified implementation, this is a placeholder.
        """
        # In a real implementation, this would connect to a websocket or REST API
        self.logger.info(f"Would connect to emotion analyzer at {self.emotion_analyzer_url}")
        # For now, just mark as connected
        self.emotion_analyzer_enabled = True
    
    async def _handle_emotional_context_request(self, data):
        """
        Handle requests to add emotional context to nodes.
        
        Args:
            data: Request data
            
        Returns:
            Processing result
        """
        node_id = data.get("node_id")
        emotional_data = data.get("emotional_data")
        
        if not node_id or not emotional_data:
            return {"success": False, "error": "Node ID and emotional data required"}
        
        result = await self.add_emotional_context(node_id, emotional_data)
        return result
    
    async def _handle_analyze_emotion_request(self, data):
        """
        Handle requests to analyze emotion in text.
        
        Args:
            data: Request data
            
        Returns:
            Analysis result
        """
        text = data.get("text")
        
        if not text:
            return {"success": False, "error": "Text required for emotion analysis"}
        
        result = await self.analyze_emotion(text)
        return {"success": True, "analysis": result}
    
    async def _handle_node_added(self, data):
        """
        Handle node added events for automatic emotional analysis.
        
        Args:
            data: Node added event data
        """
        node_id = data.get("node_id")
        node_type = data.get("node_type")
        
        if not node_id:
            return
        
        # Check if we should automatically analyze node content
        auto_analyze = self.get_config("auto_analyze_nodes", False)
        if not auto_analyze:
            return
        
        # Get the node data
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return
        
        node_data = await core_graph.get_node(node_id)
        if not node_data:
            return
        
        # Extract text to analyze
        text_to_analyze = self._extract_node_text(node_data)
        
        # Don't process if no substantial text
        if not text_to_analyze or len(text_to_analyze) < 10:
            return
        
        # Analyze emotion
        emotion_results = await self.analyze_emotion(text_to_analyze)
        
        # Add emotional context if we got results
        if emotion_results:
            await self.add_emotional_context(node_id, emotion_results)
    
    def _extract_node_text(self, node_data):
        """
        Extract text from node data for emotion analysis.
        
        Args:
            node_data: Node data
            
        Returns:
            Text for analysis
        """
        node_type = node_data.get("type", "unknown")
        
        if node_type == "concept":
            # For concepts, use definition
            return node_data.get("definition", "")
        elif node_type == "entity":
            # For entities, use name and description
            return f"{node_data.get('name', '')} {node_data.get('description', '')}"
        elif node_type == "dream_insight":
            # For dream insights, use insight text
            return node_data.get("insight", "")
        elif node_type == "memory":
            # For memories, use content
            return node_data.get("content", "")
        else:
            # For other types, combine available text fields
            text = ""
            for key, value in node_data.items():
                if isinstance(value, str) and key not in ["id", "type", "domain", "created", "modified"]:
                    text += f" {value}"
            return text.strip()
    
    async def analyze_emotion(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze emotion in text.
        
        This method would typically call an external emotion analysis service.
        For the simplified implementation, it uses pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis or None if analysis fails
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.emotion_cache:
            return self.emotion_cache[cache_key]
        
        # If external analyzer is available, use it
        if self.emotion_analyzer_enabled and self.emotion_analyzer_url:
            try:
                # In a real implementation, this would call the external service
                # For now, use the simulated analysis
                analysis = self._simulate_emotion_analysis(text)
            except Exception as e:
                self.logger.error(f"Error calling emotion analyzer: {e}")
                analysis = None
        else:
            # Use simplified pattern-based analysis
            analysis = self._simulate_emotion_analysis(text)
        
        # Update cache if we got results
        if analysis and len(self.emotion_cache) < self.cache_size_limit:
            self.emotion_cache[cache_key] = analysis
        
        return analysis
    
    def _simulate_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """
        Simulate emotion analysis using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with simulated emotion analysis
        """
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_patterns}
        emotion_scores["neutral"] = 0.5  # Default neutral score
        
        # Count emotion pattern matches
        total_matches = 0
        emotion_matches = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(r'\b' + pattern + r'\b', text_lower))
            
            emotion_matches[emotion] = matches
            total_matches += matches
        
        # Calculate emotion scores
        if total_matches > 0:
            for emotion in self.emotion_patterns:
                # Base score from pattern matches
                base_score = emotion_matches[emotion] / max(1, total_matches)
                
                # Add some randomness to make it more realistic
                variance = random.uniform(-0.1, 0.1)
                
                # Ensure score is in [0, 1] range
                score = max(0.0, min(1.0, base_score + variance))
                
                emotion_scores[emotion] = score
            
            # Reduce neutral score if we found emotions
            total_emotion_score = sum(emotion_scores[e] for e in self.emotion_patterns)
            emotion_scores["neutral"] = max(0.1, 1.0 - min(0.9, total_emotion_score))
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Only consider as dominant if above threshold
        if dominant_emotion[1] < self.emotion_integration["emotion_thresholds"]["dominant"]:
            dominant_emotion = ("neutral", emotion_scores["neutral"])
        
        # Find secondary emotions
        secondary_emotions = []
        for emotion, score in emotion_scores.items():
            if (emotion != dominant_emotion[0] and 
                score >= self.emotion_integration["emotion_thresholds"]["secondary"]):
                secondary_emotions.append((emotion, score))
        
        # Sort secondary emotions by score
        secondary_emotions.sort(key=lambda x: x[1], reverse=True)
        
        # Construct analysis result
        analysis = {
            "dominant_emotion": dominant_emotion[0],
            "dominant_score": dominant_emotion[1],
            "emotion_scores": emotion_scores,
            "secondary_emotions": [(e, s) for e, s in secondary_emotions[:2]],  # Top 2 secondary emotions
            "analysis_type": "simulated"
        }
        
        return analysis
    
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
        
        # Get core graph for node operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Check if node exists
        if not await core_graph.has_node(node_id):
            return {"success": False, "error": f"Node '{node_id}' not found"}
        
        # Get current node data
        node_data = await core_graph.get_node(node_id)
        
        # Update emotional context
        if "emotional_context" not in node_data:
            node_data["emotional_context"] = {}
        
        # Merge new emotional data with existing
        updated_context = node_data["emotional_context"].copy()
        updated_context.update(emotional_data)
        
        # Update node
        update_result = await core_graph.update_node(node_id, {"emotional_context": updated_context})
        
        if update_result:
            # Track as emotion enhanced
            self.emotion_enhanced_nodes.add(node_id)
            self.emotion_integration["emotion_enhanced_nodes"].add(node_id)
            self.emotion_integration["emotion_count"] += 1
            
            # Emit event for emotional context added
            await self.event_bus.emit("emotional_context_added", {
                "node_id": node_id,
                "emotion_data": emotional_data
            })
            
            return {
                "success": True,
                "node_id": node_id,
                "emotional_context": updated_context
            }
        else:
            return {
                "success": False,
                "error": "Failed to update node with emotional context"
            }
    
    async def add_emotional_relationship(self, source_id: str, target_id: str, 
                                  emotion: str, strength: float = 0.7) -> Dict[str, Any]:
        """
        Add an emotional relationship between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            emotion: Emotion characterizing the relationship
            strength: Strength of the relationship
            
        Returns:
            Result of the operation
        """
        self.logger.info(f"Adding emotional relationship from {source_id} to {target_id}: {emotion}")
        
        # Get core graph for node and edge operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Check if nodes exist
        if not await core_graph.has_node(source_id):
            return {"success": False, "error": f"Source node '{source_id}' not found"}
        
        if not await core_graph.has_node(target_id):
            return {"success": False, "error": f"Target node '{target_id}' not found"}
        
        # Validate emotion
        if emotion not in self.emotion_integration["primary_emotions"]:
            return {"success": False, "error": f"Unknown emotion: {emotion}"}
        
        # Add edge with emotional context
        edge_key = await core_graph.add_edge(
            source_id,
            target_id,
            edge_type="emotional",
            attributes={
                "emotion": emotion,
                "strength": strength,
                "confidence": 0.7
            }
        )
        
        if edge_key is not None:
            # Track affected nodes
            self.emotion_enhanced_nodes.add(source_id)
            self.emotion_enhanced_nodes.add(target_id)
            
            return {
                "success": True,
                "source": source_id,
                "target": target_id,
                "emotion": emotion,
                "edge_key": edge_key
            }
        else:
            return {
                "success": False,
                "error": "Failed to add emotional relationship"
            }
    
    async def get_emotional_nodes(self, emotion: Optional[str] = None, 
                           threshold: float = 0.5,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get nodes with specific emotional context.
        
        Args:
            emotion: Optional specific emotion to filter by
            threshold: Minimum emotion score threshold
            limit: Maximum number of nodes to return
            
        Returns:
            List of nodes with emotional context
        """
        self.logger.info(f"Getting emotional nodes for emotion: {emotion}")
        
        # Get core graph for node operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return []
        
        # Check all emotion enhanced nodes
        matching_nodes = []
        
        for node_id in self.emotion_enhanced_nodes:
            # Check if node still exists
            if not await core_graph.has_node(node_id):
                continue
            
            # Get node data
            node_data = await core_graph.get_node(node_id)
            
            # Check if node has emotional context
            if "emotional_context" not in node_data:
                continue
            
            emotional_context = node_data["emotional_context"]
            
            # Check for specific emotion if requested
            if emotion:
                # Check dominant emotion
                if emotional_context.get("dominant_emotion") != emotion:
                    continue
                
                # Check score
                score = emotional_context.get("dominant_score", 0)
                if score < threshold:
                    continue
            
            # Node passes filters, add to results
            matching_nodes.append({
                "node_id": node_id,
                "node_type": node_data.get("type", "unknown"),
                "emotional_context": emotional_context
            })
            
            # Check limit
            if len(matching_nodes) >= limit:
                break
        
        return matching_nodes
    
    async def analyze_emotional_relationships(self, node_id: str) -> Dict[str, Any]:
        """
        Analyze emotional relationships for a node.
        
        Args:
            node_id: Node ID to analyze
            
        Returns:
            Analysis of emotional relationships
        """
        self.logger.info(f"Analyzing emotional relationships for node: {node_id}")
        
        # Get core graph for node operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Check if node exists
        if not await core_graph.has_node(node_id):
            return {"success": False, "error": f"Node '{node_id}' not found"}
        
        # Get emotional edges
        emotional_edges = []
        
        # Outgoing edges
        outgoing_neighbors = await core_graph.get_connected_nodes(
            node_id, edge_types=["emotional"], direction="outgoing"
        )
        
        for neighbor in outgoing_neighbors:
            edges = await core_graph.get_edges(node_id, neighbor)
            for edge in edges:
                if edge.get("type") == "emotional":
                    emotional_edges.append({
                        "direction": "outgoing",
                        "target": neighbor,
                        "emotion": edge.get("emotion", "unknown"),
                        "strength": edge.get("strength", 0)
                    })
        
        # Incoming edges
        incoming_neighbors = await core_graph.get_connected_nodes(
            node_id, edge_types=["emotional"], direction="incoming"
        )
        
        for neighbor in incoming_neighbors:
            edges = await core_graph.get_edges(neighbor, node_id)
            for edge in edges:
                if edge.get("type") == "emotional":
                    emotional_edges.append({
                        "direction": "incoming",
                        "source": neighbor,
                        "emotion": edge.get("emotion", "unknown"),
                        "strength": edge.get("strength", 0)
                    })
        
        # Group by emotion
        emotions_by_type = defaultdict(list)
        for edge in emotional_edges:
            emotions_by_type[edge["emotion"]].append(edge)
        
        # Calculate summary
        emotion_summary = {}
        for emotion, edges in emotions_by_type.items():
            emotion_summary[emotion] = {
                "count": len(edges),
                "average_strength": sum(e["strength"] for e in edges) / len(edges) if edges else 0,
                "edges": edges
            }
        
        return {
            "success": True,
            "node_id": node_id,
            "total_emotional_relationships": len(emotional_edges),
            "emotion_summary": emotion_summary
        }
    
    async def process_external_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process external knowledge data.
        
        Args:
            data: External knowledge data
            
        Returns:
            Processing result
        """
        data_type = data.get("type", "unknown")
        
        if data_type == "emotional_context":
            node_id = data.get("data", {}).get("node_id")
            emotional_data = data.get("data", {}).get("emotional_data")
            
            if not node_id or not emotional_data:
                return {"success": False, "error": "Node ID and emotional data required"}
            
            result = await self.add_emotional_context(node_id, emotional_data)
            return result
        else:
            self.logger.warning(f"Unknown data type for processing: {data_type}")
            return {"success": False, "error": f"Unknown data type: {data_type}"}
    
    async def get_emotion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about emotional context in the knowledge graph.
        
        Returns:
            Dictionary with emotion statistics
        """
        # Count nodes with emotional context
        emotion_enhanced_count = len(self.emotion_enhanced_nodes)
        
        # Count by emotion
        emotion_counts = defaultdict(int)
        nodes_by_emotion = defaultdict(list)
        
        # Get core graph for node operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"error": "Core graph module not found"}
        
        # Count edges with emotional context
        emotional_edge_count = 0
        emotion_edge_counts = defaultdict(int)
        
        # Count node emotions
        for node_id in self.emotion_enhanced_nodes:
            if await core_graph.has_node(node_id):
                node_data = await core_graph.get_node(node_id)
                if "emotional_context" in node_data and "dominant_emotion" in node_data["emotional_context"]:
                    emotion = node_data["emotional_context"]["dominant_emotion"]
                    emotion_counts[emotion] += 1
                    nodes_by_emotion[emotion].append(node_id)
        
        # Count edge emotions
        for source, target, key, data in core_graph.graph.edges(data=True, keys=True):
            if data.get("type") == "emotional" and "emotion" in data:
                emotional_edge_count += 1
                emotion = data["emotion"]
                emotion_edge_counts[emotion] += 1
        
        return {
            "total_nodes_with_emotion": emotion_enhanced_count,
            "nodes_by_emotion": {k: len(v) for k, v in nodes_by_emotion.items()},
            "total_emotional_edges": emotional_edge_count,
            "edges_by_emotion": dict(emotion_edge_counts),
            "cache_size": len(self.emotion_cache),
            "emotion_analyzer_enabled": self.emotion_analyzer_enabled
        }