"""
Emotion Graph Enhancer - Integrates emotion analysis with knowledge graph
Adds emotional context to nodes and relationships in the knowledge graph
"""

import logging
import asyncio
import json
import websockets
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class EmotionGraphEnhancer:
    """Enhances Knowledge Graph with emotional context from the emotion analyzer.
    
    Connects to the emotion analyzer service via WebSocket to analyze text content
    and adds emotional metadata to nodes and relationships in the knowledge graph.
    """
    
    def __init__(self, url: str = "ws://localhost:5007", 
                 knowledge_graph = None,
                 emotion_threshold: float = 0.3):
        """Initialize the EmotionGraphEnhancer.
        
        Args:
            url: WebSocket URL for the emotion analyzer service
            knowledge_graph: LucidiaKnowledgeGraph instance
            emotion_threshold: Confidence threshold for emotions
        """
        self.url = url
        self.knowledge_graph = knowledge_graph
        self.emotion_threshold = emotion_threshold
        self.websocket = None
        self.connected = False
        logger.info(f"Initialized EmotionGraphEnhancer, will connect to {url}")
        
    async def connect(self):
        """Connect to the emotion analyzer service."""
        try:
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to emotion analyzer service at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to emotion analyzer: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the emotion analyzer service."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from emotion analyzer service")
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for emotional content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not self.connected:
            success = await self.connect()
            if not success:
                logger.error("Cannot analyze text: not connected to emotion analyzer")
                return {}
                
        try:
            # Send analysis request
            await self.websocket.send(json.dumps({
                "type": "analyze",
                "text": text,
                "threshold": self.emotion_threshold
            }))
            
            # Get response
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "error":
                logger.error(f"Error analyzing text: {result.get('message')}")
                return {}
                
            return result
        except Exception as e:
            logger.error(f"Error in analyze_text: {e}")
            # Try to reconnect
            self.connected = False
            return {}
    
    async def enhance_node_with_emotions(self, node_id: str, text: Optional[str] = None) -> bool:
        """Enhance a knowledge graph node with emotional metadata.
        
        Args:
            node_id: ID of the node to enhance
            text: Text to analyze. If None, uses node name and description
            
        Returns:
            True if successful, False otherwise
        """
        if not self.knowledge_graph:
            logger.error("No knowledge graph provided")
            return False
            
        # Get node information
        node = self.knowledge_graph.get_node_by_id(node_id)
        if not node:
            logger.error(f"Node {node_id} not found")
            return False
            
        # If no text provided, use node name and description
        if not text:
            node_name = node.get("name", "")
            node_description = node.get("description", "")
            text = f"{node_name}. {node_description}".strip()
            
        if not text:
            logger.warning(f"No text available for node {node_id}")
            return False
            
        # Analyze text
        emotion_data = await self.analyze_text(text)
        if not emotion_data:
            return False
            
        # Extract emotion data
        primary_emotions = emotion_data.get("primary_emotions", {})
        dominant_emotion = emotion_data.get("dominant_primary", {}).get("emotion")
        dominant_confidence = emotion_data.get("dominant_primary", {}).get("confidence")
        detailed_emotions = emotion_data.get("detailed_emotions", {})
        
        # Prepare emotion metadata
        emotion_metadata = {
            "primary_emotions": primary_emotions,
            "dominant_emotion": dominant_emotion,
            "confidence": dominant_confidence,
            "detailed_emotions": detailed_emotions
        }
        
        # Update node metadata
        metadata = node.get("metadata", {})
        metadata["emotions"] = emotion_metadata
        
        # Update node
        return self.knowledge_graph.update_node(node_id, {"metadata": metadata})
    
    async def enhance_relationship_with_emotions(self, source_id: str, target_id: str, 
                                             relation_type: str, text: Optional[str] = None) -> bool:
        """Enhance a knowledge graph relationship with emotional metadata.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            text: Text to analyze. If None, uses relation_type
            
        Returns:
            True if successful, False otherwise
        """
        if not self.knowledge_graph:
            logger.error("No knowledge graph provided")
            return False
            
        # Get relationship
        edges = self.knowledge_graph.get_relationships(source_id=source_id, target_id=target_id, relation_type=relation_type)
        if not edges:
            logger.error(f"Relationship not found: {source_id} -> {relation_type} -> {target_id}")
            return False
            
        edge = edges[0]  # Take the first matching edge
        edge_id = edge.get("id")
        
        # If no text provided, use relation_type and any description
        if not text:
            # Get source and target node names to provide more context
            source_node = self.knowledge_graph.get_node_by_id(source_id)
            target_node = self.knowledge_graph.get_node_by_id(target_id)
            source_name = source_node.get("name", "") if source_node else ""
            target_name = target_node.get("name", "") if target_node else ""
            
            description = edge.get("description", "")
            text = f"{source_name} {relation_type} {target_name}. {description}".strip()
            
        if not text:
            text = relation_type  # Fallback to just the relation type
            
        # Analyze text
        emotion_data = await self.analyze_text(text)
        if not emotion_data:
            return False
            
        # Extract emotion data
        primary_emotions = emotion_data.get("primary_emotions", {})
        dominant_emotion = emotion_data.get("dominant_primary", {}).get("emotion")
        dominant_confidence = emotion_data.get("dominant_primary", {}).get("confidence")
        detailed_emotions = emotion_data.get("detailed_emotions", {})
        
        # Prepare emotion metadata
        emotion_metadata = {
            "primary_emotions": primary_emotions,
            "dominant_emotion": dominant_emotion,
            "confidence": dominant_confidence,
            "detailed_emotions": detailed_emotions
        }
        
        # Update edge metadata
        metadata = edge.get("metadata", {})
        metadata["emotions"] = emotion_metadata
        
        # Update edge
        return self.knowledge_graph.update_relationship(edge_id, {"metadata": metadata})
    
    async def enhance_subgraph(self, central_node_id: str, depth: int = 1) -> Dict[str, Any]:
        """Enhance a subgraph with emotional metadata.
        
        Args:
            central_node_id: ID of the central node
            depth: Depth of traversal (1 = immediate neighbors)
            
        Returns:
            Dictionary with enhancement statistics
        """
        if not self.knowledge_graph:
            logger.error("No knowledge graph provided")
            return {"success": False, "error": "No knowledge graph provided"}
            
        stats = {
            "nodes_enhanced": 0,
            "relationships_enhanced": 0,
            "failed_nodes": 0,
            "failed_relationships": 0
        }
        
        # Get subgraph
        subgraph = self.knowledge_graph.get_subgraph(central_node_id, depth)
        if not subgraph:
            return {"success": False, "error": f"Failed to get subgraph for node {central_node_id}"}
            
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        
        # Enhance nodes
        for node in nodes:
            node_id = node.get("id")
            success = await self.enhance_node_with_emotions(node_id)
            if success:
                stats["nodes_enhanced"] += 1
            else:
                stats["failed_nodes"] += 1
                
        # Enhance relationships
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            relation_type = edge.get("type")
            
            success = await self.enhance_relationship_with_emotions(
                source_id, target_id, relation_type)
                
            if success:
                stats["relationships_enhanced"] += 1
            else:
                stats["failed_relationships"] += 1
                
        stats["success"] = True
        return stats
    
    async def batch_enhance_nodes(self, node_ids: List[str]) -> Dict[str, Any]:
        """Enhance multiple nodes with emotional metadata in batch.
        
        Args:
            node_ids: List of node IDs to enhance
            
        Returns:
            Dictionary with enhancement statistics
        """
        stats = {
            "nodes_enhanced": 0,
            "failed_nodes": 0,
            "node_details": {}
        }
        
        for node_id in node_ids:
            success = await self.enhance_node_with_emotions(node_id)
            
            if success:
                stats["nodes_enhanced"] += 1
                stats["node_details"][node_id] = "success"
            else:
                stats["failed_nodes"] += 1
                stats["node_details"][node_id] = "failed"
                
        stats["success"] = stats["nodes_enhanced"] > 0
        return stats
    
    async def enhance_new_content(self, text: str, extract_concepts: bool = True) -> Dict[str, Any]:
        """Analyze new content, extract concepts, and enhance the knowledge graph.
        
        Args:
            text: Text content to analyze
            extract_concepts: Whether to extract concepts from text
            
        Returns:
            Dictionary with enhancement statistics and emotion analysis
        """
        if not self.knowledge_graph:
            logger.error("No knowledge graph provided")
            return {"success": False, "error": "No knowledge graph provided"}
            
        # Analyze text for emotions
        emotion_data = await self.analyze_text(text)
        if not emotion_data:
            return {"success": False, "error": "Failed to analyze emotions"}
            
        result = {
            "emotion_analysis": emotion_data,
            "nodes_enhanced": 0,
            "relationships_enhanced": 0
        }
        
        # If extract_concepts is True, use knowledge graph to extract concepts
        if extract_concepts and hasattr(self.knowledge_graph, "extract_concepts_and_relationships"):
            extraction_result = self.knowledge_graph.extract_concepts_and_relationships(text)
            
            if extraction_result and "concepts" in extraction_result:
                concepts = extraction_result.get("concepts", [])
                relationships = extraction_result.get("relationships", [])
                
                # Enhance extracted concepts
                for concept in concepts:
                    # If concept is new, add it to the graph
                    concept_id = concept.get("id")
                    if not self.knowledge_graph.get_node_by_id(concept_id):
                        # Add emotion metadata
                        metadata = concept.get("metadata", {})
                        metadata["emotions"] = emotion_data
                        concept["metadata"] = metadata
                        
                        # Add to graph
                        self.knowledge_graph.add_node(concept)
                    else:
                        # Enhance existing concept
                        await self.enhance_node_with_emotions(concept_id)
                        
                    result["nodes_enhanced"] += 1
                    
                # Enhance extracted relationships
                for relationship in relationships:
                    source_id = relationship.get("source")
                    target_id = relationship.get("target")
                    relation_type = relationship.get("type")
                    
                    # Add emotion metadata
                    metadata = relationship.get("metadata", {})
                    metadata["emotions"] = emotion_data
                    relationship["metadata"] = metadata
                    
                    # Add to graph
                    self.knowledge_graph.add_relationship(relationship)
                    result["relationships_enhanced"] += 1
                    
                result["extraction_result"] = extraction_result
                
        result["success"] = True
        return result
