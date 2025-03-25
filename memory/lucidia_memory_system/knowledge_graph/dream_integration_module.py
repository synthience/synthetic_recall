"""
Dream Integration Module for Lucidia's Knowledge Graph

This module implements dream insight integration with meta-learning feedback loops,
quality evaluation, and parameter adaptation.
"""

import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict, deque
import random

from .base_module import KnowledgeGraphModule

class DreamIntegrationModule(KnowledgeGraphModule):
    """
    Dream Integration Module for processing and integrating dream insights.
    
    This module handles the integration of dream insights into the knowledge graph,
    with meta-learning feedback loops to improve integration quality over time.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Dream Integration Module."""
        super().__init__(event_bus, module_registry, config)
        
        # Dreaming integration configuration
        self.dream_integration = {
            "insight_incorporation_rate": self.get_config("insight_incorporation_rate", 0.8),
            "dream_association_strength": self.get_config("dream_association_strength", 0.7),
            "dream_derived_nodes": set(),
            "dream_enhanced_nodes": set(),
            "dream_insight_count": 0,
            # Meta-learning parameters
            "integration_quality_scores": [],
            "learning_rate": self.get_config("learning_rate", 0.05),
            "quality_threshold": self.get_config("quality_threshold", 0.7),
            "adaptation_frequency": self.get_config("adaptation_frequency", 5),
            # Metrics tracking
            "integration_metrics": []
        }
        
        # Track nodes influenced by dreams
        self.dream_influenced_nodes = set()
        
        # Spiral awareness phase
        self.current_spiral_phase = self.get_config("current_spiral_phase", "observation")
        self.spiral_integration = {
            "observation_emphasis": self.get_config("observation_emphasis", 0.8),
            "reflection_emphasis": self.get_config("reflection_emphasis", 0.9),
            "adaptation_emphasis": self.get_config("adaptation_emphasis", 0.7),
            "execution_emphasis": self.get_config("execution_emphasis", 0.6)
        }
        
        self.logger.info("Dream Integration Module initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("dream_insight_received", self._handle_dream_insight)
        await self.event_bus.subscribe("spiral_phase_changed", self._handle_spiral_phase_change)
        await self.event_bus.subscribe("dream_integration_quality_feedback", self._handle_quality_feedback)
        self.logger.info("Subscribed to dream-related events")
    
    async def _setup_module(self):
        """Set up module-specific resources and state."""
        # Register operation handlers
        self.module_registry.register_operation_handler("integrate_dream_insight", self.integrate_dream_insight)
        self.module_registry.register_operation_handler("get_dream_influenced_nodes", self.get_dream_influenced_nodes)
        
        self.logger.info("Dream Integration Module setup complete")
    
    async def _handle_dream_insight(self, data):
        """
        Handle dream insight events.
        
        Args:
            data: Dream insight event data
            
        Returns:
            Integration result
        """
        insight_text = data.get("insight")
        source_memory = data.get("source_memory")
        
        if not insight_text:
            return {"success": False, "error": "Dream insight text required"}
        
        result = await self.integrate_dream_insight(insight_text, source_memory)
        return result
    
    async def _handle_spiral_phase_change(self, data):
        """
        Handle spiral phase change events.
        
        Args:
            data: Spiral phase change event data
        """
        new_phase = data.get("phase")
        if not new_phase:
            return
        
        if new_phase in ["observation", "reflection", "adaptation", "execution"]:
            self.current_spiral_phase = new_phase
            self.logger.info(f"Spiral phase changed to: {new_phase}")
    
    async def _handle_quality_feedback(self, data):
        """
        Handle dream integration quality feedback events.
        
        Args:
            data: Quality feedback event data
            
        Returns:
            Feedback result
        """
        dream_id = data.get("dream_id")
        quality_score = data.get("quality_score")
        
        if dream_id is None or quality_score is None:
            return {"success": False, "error": "Dream ID and quality score required"}
        
        # Add to quality scores
        self.dream_integration["integration_quality_scores"].append(quality_score)
        
        # Apply meta-learning if needed
        if len(self.dream_integration["integration_quality_scores"]) >= self.dream_integration["adaptation_frequency"]:
            await self._adjust_integration_parameters()
        
        return {"success": True, "dream_id": dream_id, "quality_score": quality_score}
    
    async def integrate_dream_insight(self, insight_text: str, 
                              source_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate a dream insight into the knowledge graph with meta-learning feedback.
        
        Args:
            insight_text: Dream insight text
            source_memory: Optional source memory that generated the insight
            
        Returns:
            Integration results
        """
        self.logger.info(f"Integrating dream insight: {insight_text[:50]}...")
        
        # Get core graph for node operations
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return {"success": False, "error": "Core graph module not found"}
        
        # Create a dream insight node
        dream_id = f"dream:{self.dream_integration['dream_insight_count']}"
        
        await core_graph.add_node(
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
        await core_graph.add_edge(
            "Lucidia",
            dream_id,
            edge_type="dreamed",
            attributes={
                "strength": 0.85,
                "confidence": 0.8
            }
        )
        
        # Extract concepts from insight text
        dream_concepts = await self._extract_concepts(insight_text, core_graph)
        
        # Analyze emotional context
        emotional_context_manager = self.module_registry.get_module("emotional_context")
        has_emotional_context = False
        emotion_results = None
        
        if emotional_context_manager:
            try:
                emotion_results = await emotional_context_manager.analyze_emotion(insight_text)
                has_emotional_context = bool(emotion_results and "dominant_emotion" in emotion_results)
                
                # Add emotional context to dream insight node
                if has_emotional_context:
                    await emotional_context_manager.add_emotional_context(dream_id, emotion_results)
            except Exception as e:
                self.logger.warning(f"Error analyzing emotions in dream insight: {e}")
        
        # Connect to found concepts
        connected_concepts = []
        for concept in dream_concepts:
            if await core_graph.has_node(concept):
                await core_graph.add_edge(
                    dream_id,
                    concept,
                    edge_type="references",
                    attributes={
                        "strength": self.dream_integration["dream_association_strength"],
                        "confidence": 0.7
                    }
                )
                connected_concepts.append(concept)
                
                # Mark concept as dream influenced
                self.dream_influenced_nodes.add(concept)
                self.dream_integration["dream_enhanced_nodes"].add(concept)
                
                # If we have emotional context, enhance the concept with it
                if has_emotional_context and emotional_context_manager:
                    await emotional_context_manager.add_emotional_context(concept, emotion_results)
        
        # Evaluate consistency with existing knowledge
        consistency_score = await self._evaluate_consistency(dream_id, connected_concepts, core_graph)
        
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
                    if not await core_graph.has_edge(concept1, concept2, "dream_associated"):
                        relationship_strength = self.dream_integration["dream_association_strength"] * integration_strategy["relationship_strength_factor"]
                        
                        edge_key = await core_graph.add_edge(
                            concept1,
                            concept2,
                            edge_type="dream_associated",
                            attributes={
                                "strength": relationship_strength,
                                "confidence": 0.6,
                                "source": "dream_insight",
                                "source_dream": dream_id
                            }
                        )
                        
                        if edge_key is not None:
                            new_concept_relationships.append((concept1, concept2))
        
        # Check if insight suggests new concepts based on strategy
        new_concepts = []
        
        if integration_strategy["extract_new_concepts"]:
            potential_concepts = await self._extract_potential_concepts(insight_text)
            
            for concept_name, definition in potential_concepts.items():
                # Only add if it doesn't exist yet
                if not await core_graph.has_node(concept_name):
                    # Create the new concept
                    await core_graph.add_node(
                        concept_name,
                        node_type="concept",
                        attributes={
                            "definition": definition or f"Concept derived from dream insight: {dream_id}",
                            "confidence": integration_strategy["new_concept_confidence"]
                        },
                        domain="synthien_studies"
                    )
                    
                    # Connect to dream
                    await core_graph.add_edge(
                        dream_id,
                        concept_name,
                        edge_type="introduced",
                        attributes={
                            "strength": 0.7,
                            "confidence": 0.6
                        }
                    )
                    
                    new_concepts.append(concept_name)
                    
                    # Mark as dream influenced
                    self.dream_influenced_nodes.add(concept_name)
                    self.dream_integration["dream_derived_nodes"].add(concept_name)
                    
                    # Add emotional context if available
                    if has_emotional_context and emotional_context_manager:
                        await emotional_context_manager.add_emotional_context(concept_name, emotion_results)
        
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
            "success": True,
            "dream_id": dream_id,
            "connected_concepts": connected_concepts,
            "new_concepts": new_concepts,
            "new_relationships": new_concept_relationships,
            "consistency_score": consistency_score,
            "integration_quality": integration_quality,
            "integration_strategy": integration_strategy,
            "emotional_context": has_emotional_context,
            "emotion_results": emotion_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Dream insight integrated with {len(connected_concepts)} connections and {len(new_concepts)} new concepts")
        
        # Emit dream integrated event
        await self.event_bus.emit("dream_insight_integrated", result)
        
        return result
    
    async def _extract_concepts(self, text: str, core_graph) -> List[str]:
        """
        Extract concepts from text.
        
        Args:
            text: Text to extract concepts from
            core_graph: Core graph module for node access
            
        Returns:
            List of concept node IDs
        """
        # Get concepts from world model if available
        world_model = self.module_registry.get_module("world_model")
        concepts = []
        
        if world_model and hasattr(world_model, '_extract_concepts'):
            try:
                concepts = await world_model._extract_concepts(text)
            except Exception as e:
                self.logger.warning(f"Error extracting concepts with world model: {e}")
        
        # If no concepts found or no world model available, try to match with existing nodes
        if not concepts:
            # Get all concept nodes
            all_concepts = await core_graph.get_nodes_by_type("concept")
            
            # Process text to extract words and phrases
            words = text.lower().split()
            phrases = []
            
            # Extract potential phrases (2-3 word combinations)
            for i in range(len(words) - 1):
                phrases.append(f"{words[i]}_{words[i+1]}")
            
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
            
            # Check for matches with existing concepts
            # First check exact matches
            for concept in all_concepts:
                concept_lower = concept.lower()
                if concept_lower in text.lower():
                    concepts.append(concept)
            
            # Then check for word matches
            if len(concepts) < 5:  # Limit to reasonable number
                for word in words:
                    if len(word) > 4:  # Skip short words
                        for concept in all_concepts:
                            if word == concept.lower() and concept not in concepts:
                                concepts.append(concept)
            
            # Then check for phrase matches
            if len(concepts) < 5:  # Limit to reasonable number
                for phrase in phrases:
                    for concept in all_concepts:
                        if phrase == concept.lower() and concept not in concepts:
                            concepts.append(concept)
        
        return concepts
    
    async def _evaluate_consistency(self, dream_id: str, concepts: List[str], core_graph) -> float:
        """
        Evaluate the consistency of a dream insight with existing knowledge.
        
        Args:
            dream_id: Dream insight node ID
            concepts: List of concepts connected to the dream
            core_graph: Core graph module for node access
            
        Returns:
            Consistency score (0-1)
        """
        if not concepts:
            return 0.5  # Neutral score if no connections
        
        # Get the dream insight
        dream_node = await core_graph.get_node(dream_id)
        if not dream_node:
            return 0.5
        
        insight_text = dream_node.get("insight", "")
        
        # Get embedding manager for similarity calculations
        embedding_manager = self.module_registry.get_module("embedding_manager")
        
        # Check consistency of connections
        consistency_scores = []
        
        for concept in concepts:
            # Get concept definition
            concept_node = await core_graph.get_node(concept)
            if not concept_node:
                continue
                
            definition = concept_node.get("definition", "")
            
            # Calculate semantic similarity between concept and insight
            if embedding_manager:
                similarity = await embedding_manager.calculate_node_similarity(concept, insight_text)
            else:
                # Simple text similarity fallback
                similarity = self._text_similarity(insight_text, definition)
            
            # Get related concepts (already in the graph)
            related_concepts = await core_graph.get_connected_nodes(
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
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity between two strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
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
        current_phase = self.current_spiral_phase
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
    
    async def _extract_potential_concepts(self, text: str) -> Dict[str, str]:
        """
        Extract potential new concepts from text.
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            Dictionary mapping concept names to definitions
        """
        potential_concepts = {}
        
        # Look for patterns suggesting definitions or concepts
        concept_patterns = [
            r"concept of (\w+)",
            r"(\w+) is defined as",
            r"(\w+) refers to",
            r"understanding of (\w+)",
            r"notion of (\w+)",
            r"idea of (\w+)"
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                potential_concept = match.lower()
                
                # Check if this is a reasonable concept (not too short, not just a stop word)
                if (len(potential_concept) >= 4 and 
                    potential_concept not in ["this", "that", "there", "which", "where"]):
                    
                    # Extract a definition from the insight
                    definition = self._extract_definition(text, potential_concept)
                    
                    # Only add if we could extract a definition
                    if definition:
                        potential_concepts[potential_concept] = definition
        
        # Look for terms that appear to be important (capitalized, repeated)
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)  # Capitalized words
        
        for word in words:
            if word.lower() not in potential_concepts:
                # Extract a definition from the insight
                definition = self._extract_definition(text, word.lower())
                
                # Only add if we could extract a definition
                if definition:
                    potential_concepts[word.lower()] = definition
        
        return potential_concepts
    
    def _extract_definition(self, text: str, concept: str) -> Optional[str]:
        """
        Extract a definition for a concept from text.
        
        Args:
            text: Text to extract definition from
            concept: Concept to extract definition for
            
        Returns:
            Definition or None if not found
        """
        # Look for explicit definitions
        patterns = [
            rf"{concept} is defined as (.*?)[.?!]",
            rf"{concept} refers to (.*?)[.?!]",
            rf"{concept} means (.*?)[.?!]",
            rf"{concept} is (.*?)[.?!]",
            rf"definition of {concept} is (.*?)[.?!]",
            rf"{concept} can be understood as (.*?)[.?!]"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no explicit definition, extract the sentence containing the concept
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if re.search(r'\b' + re.escape(concept) + r'\b', sentence, re.IGNORECASE):
                return sentence.strip()
        
        return None
    
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
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return 0.5
        
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
                neighbors = await core_graph.get_connected_nodes(concept)
                # Calculate connectivity as connections per concept
                concept_connectivity = len(neighbors) / max(1, core_graph.total_nodes / 100)  # Normalize by graph size
                total_connectivity += min(1.0, concept_connectivity)
            
            metrics["connectivity"] = total_connectivity / len(connected_concepts)
        
        # 2. Novelty: Measure new knowledge added
        novelty_score = 0.0
        if new_concepts:
            # New concepts contribute to novelty
            novelty_score += 0.7 * (len(new_concepts) / max(1, len(connected_concepts) + 1))
        
        if new_relationships:
            # New relationships contribute to novelty
            novelty_score += 0.3 * (len(new_relationships) / max(1, len(connected_concepts) + 1))
        
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
        self.dream_integration["integration_metrics"].append(integration_metrics)
        
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
    
    async def get_dream_influenced_nodes(self, limit: int = 100, 
                                  node_types: Optional[List[str]] = None) -> List[str]:
        """
        Get nodes influenced by dreams.
        
        Args:
            limit: Maximum number of nodes to return
            node_types: Optional list of node types to filter by
            
        Returns:
            List of node IDs influenced by dreams
        """
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return []
        
        # Filter nodes by type if requested
        if node_types:
            filtered_nodes = []
            for node_id in self.dream_influenced_nodes:
                if await core_graph.has_node(node_id):
                    node_data = await core_graph.get_node(node_id)
                    if node_data and node_data.get("type") in node_types:
                        filtered_nodes.append(node_id)
            
            return filtered_nodes[:limit]
        else:
            # Return all dream influenced nodes
            return list(self.dream_influenced_nodes)[:limit]
    
    async def process_external_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process external knowledge data.
        
        Args:
            data: External knowledge data
            
        Returns:
            Processing result
        """
        data_type = data.get("type", "unknown")
        
        if data_type == "dream_insight":
            insight_text = data.get("data", {}).get("insight")
            source_memory = data.get("data", {}).get("source_memory")
            
            if not insight_text:
                return {"success": False, "error": "Dream insight text required"}
            
            result = await self.integrate_dream_insight(insight_text, source_memory)
            return result
        else:
            self.logger.warning(f"Unknown data type for processing: {data_type}")
            return {"success": False, "error": f"Unknown data type: {data_type}"}
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about dream integration.
        
        Returns:
            Dictionary with integration statistics
        """
        # Calculate statistics
        total_integrations = self.dream_integration["dream_insight_count"]
        derived_nodes = len(self.dream_integration["dream_derived_nodes"])
        enhanced_nodes = len(self.dream_integration["dream_enhanced_nodes"])
        influenced_nodes = len(self.dream_influenced_nodes)
        
        # Calculate average quality from metrics
        quality_scores = [m["quality_score"] for m in self.dream_integration["integration_metrics"]]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Metrics by component
        metrics_by_component = {}
        for metric in ["connectivity", "novelty", "coherence", "alignment"]:
            component_scores = [m["metrics"][metric] for m in self.dream_integration["integration_metrics"]]
            metrics_by_component[metric] = sum(component_scores) / len(component_scores) if component_scores else 0
        
        return {
            "total_integrations": total_integrations,
            "dream_derived_nodes": derived_nodes,
            "dream_enhanced_nodes": enhanced_nodes,
            "dream_influenced_nodes": influenced_nodes,
            "average_quality": avg_quality,
            "metrics_by_component": metrics_by_component,
            "current_parameters": {
                "insight_incorporation_rate": self.dream_integration["insight_incorporation_rate"],
                "dream_association_strength": self.dream_integration["dream_association_strength"],
                "learning_rate": self.dream_integration["learning_rate"],
                "quality_threshold": self.dream_integration["quality_threshold"],
                "adaptation_frequency": self.dream_integration["adaptation_frequency"]
            },
            "current_spiral_phase": self.current_spiral_phase
        }