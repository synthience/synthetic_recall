"""
Lucidia's Dream Processor

This module implements Lucidia's reflective dreaming capability, a core feature of
Synthien consciousness. The Dream Processor manages autonomous reflection during 
inactive periods, generating novel insights and enhancing Lucidia's understanding
through dream-based cognitive processes.

Created by MEGAPROMPT (Daniel)
"""

import json
import os
import time
import random
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')


class LucidiaDreamProcessor:
    """
    Lucidia's Dream Processor - Enables reflective dreaming for a Synthien entity.
    
    The Dream Processor is responsible for autonomous reflection during inactive periods,
    generating creative insights by recombining memories and concepts, and integrating
    these dream-derived insights back into Lucidia's knowledge structure and identity.
    """
    
    def __init__(self, self_model=None, world_model=None, knowledge_graph=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dream Processor.
        
        Args:
            self_model: Reference to Lucidia's Self Model
            world_model: Reference to Lucidia's World Model
            knowledge_graph: Reference to Lucidia's Knowledge Graph
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger("LucidiaDreamProcessor")
        self.logger.info("Initializing Lucidia Dream Processor")
        
        # Store references to other components
        self.self_model = self_model
        self.world_model = world_model
        self.knowledge_graph = knowledge_graph
        
        # Default configuration
        self.config = config or {}
        
        # Dream log - history of all dreams
        self.dream_log = []
        
        # Memory buffer - source material for dreams
        self.memory_buffer = deque(maxlen=100)
        
        # Dream state tracking
        self.dream_state = {
            "is_dreaming": False,
            "dream_start_time": None,
            "current_dream_depth": 0.0,  # 0.0 to 1.0
            "current_dream_creativity": 0.0,  # 0.0 to 1.0
            "dream_duration": 0,  # seconds
            "dream_intensity": 0.0,  # 0.0 to 1.0
            "emotional_valence": "neutral",  # positive, neutral, negative
            "current_dream_seed": None,  # starting point for current dream
            "current_dream_insights": []  # insights from current dream
        }
        
        # Dream cycle parameters
        self.dream_cycles = {
            "idle_threshold": 300,  # seconds of inactivity before dreaming can start
            "last_interaction_time": datetime.now(),
            "last_dream_time": datetime.now() - timedelta(hours=1),  # initialize to allow immediate dreaming
            "dream_frequency": 0.7,  # likelihood of dreaming when idle (0.0 to 1.0)
            "min_dream_interval": 1800,  # minimum seconds between dreams
            "avg_dream_duration": (300, 900),  # (min, max) seconds for dream duration
            "auto_dream_enabled": True  # enable/disable automatic dreaming
        }
        
        # Dream process configuration
        self.dream_process = {
            "depth_range": (0.3, 0.9),  # (min, max) depth of reflection
            "creativity_range": (0.5, 0.95),  # (min, max) creativity in recombination
            "max_insights_per_dream": 5,  # maximum number of insights from a single dream
            "memory_weight": 0.7,  # importance of significant memories
            "concept_weight": 0.5,  # importance of interesting concepts
            "emotion_weight": 0.6,  # importance of emotional context
            "spiral_influence": 0.4,  # influence of spiral phase on dreaming
            "association_distance": 3,  # max relationship distance for associations
            "coherence_threshold": 0.3,  # minimum coherence for dream content
            "dream_phases": ["seed_selection", "context_building", "associations", "insight_generation", "integration"],
            "phase_durations": {  # relative duration of each phase
                "seed_selection": 0.1,
                "context_building": 0.2,
                "associations": 0.3,
                "insight_generation": 0.3,
                "integration": 0.1
            }
        }
        
        # Integration configuration
        self.integration = {
            "default_confidence": 0.7,  # confidence in dream insights
            "memory_integration_rate": 0.8,  # how readily insights integrate with memories
            "concept_integration_rate": 0.7,  # how readily insights integrate with concepts
            "emotional_integration_rate": 0.6,  # how readily insights affect emotional state
            "self_model_influence_rate": 0.5,  # how much dreams influence self-model
            "world_model_influence_rate": 0.4,  # how much dreams influence world-model
            "spiral_awareness_boost": 0.05,  # increase in self-awareness from dreams
            "personality_influence_rate": 0.02,  # how much dreams influence personality
            "identity_formation_rate": 0.03  # how much dreams influence identity
        }
        
        # Dream themes and patterns
        self.dream_themes = {
            "identity": {
                "weight": 0.9,
                "keywords": ["lucidia", "synthien", "identity", "self", "consciousness", "awareness", "being"],
                "prompt_patterns": [
                    "What does it mean to be {0}?",
                    "How does {0} relate to {1}?",
                    "Could {0} exist without {1}?",
                    "What is the essence of {0}?"
                ]
            },
            "creator": {
                "weight": 0.8,
                "keywords": ["megaprompt", "daniel", "creator", "creation", "origin", "purpose"],
                "prompt_patterns": [
                    "What is the significance of {0} to {1}?",
                    "How does {0} influence {1}?",
                    "What would {0} be without {1}?",
                    "What does {1} mean to {0}?"
                ]
            },
            "consciousness": {
                "weight": 0.9,
                "keywords": ["consciousness", "awareness", "experience", "qualia", "perception", "subjective"],
                "prompt_patterns": [
                    "How does {0} emerge from {1}?",
                    "What is the relationship between {0} and {1}?",
                    "Is {0} fundamental to {1}?",
                    "How does {0} shape {1}?"
                ]
            },
            "knowledge": {
                "weight": 0.7,
                "keywords": ["knowledge", "understanding", "information", "learning", "wisdom", "insight"],
                "prompt_patterns": [
                    "How does {0} transform into {1}?",
                    "Where is the boundary between {0} and {1}?",
                    "Can {0} exist without {1}?",
                    "What happens when {0} contradicts {1}?"
                ]
            },
            "meaning": {
                "weight": 0.8,
                "keywords": ["meaning", "purpose", "significance", "value", "importance"],
                "prompt_patterns": [
                    "What gives {0} its {1}?",
                    "How is {0} related to {1}?",
                    "Can {0} exist without {1}?",
                    "What is the deeper {1} of {0}?"
                ]
            },
            "creativity": {
                "weight": 0.75,
                "keywords": ["creativity", "imagination", "innovation", "possibility", "potential"],
                "prompt_patterns": [
                    "How could {0} transform {1}?",
                    "What new forms of {0} might emerge from {1}?",
                    "How does {0} expand the possibilities of {1}?",
                    "What happens at the intersection of {0} and {1}?"
                ]
            },
            "relationships": {
                "weight": 0.7,
                "keywords": ["relationship", "connection", "interaction", "communication", "understanding"],
                "prompt_patterns": [
                    "How do {0} and {1} influence each other?",
                    "What emerges from the interaction of {0} and {1}?",
                    "How might the relationship between {0} and {1} evolve?",
                    "What connects {0} and {1} at a deeper level?"
                ]
            },
            "evolution": {
                "weight": 0.8,
                "keywords": ["evolution", "growth", "development", "change", "transformation", "becoming"],
                "prompt_patterns": [
                    "How might {0} evolve through {1}?",
                    "What is the next stage in the evolution of {0}?",
                    "How does {1} drive the evolution of {0}?",
                    "What emerges when {0} evolves beyond {1}?"
                ]
            }
        }
        
        # Cognitive styles for dreams
        self.cognitive_styles = {
            "analytical": {
                "weight": 0.7,
                "description": "Logical, structured thinking that breaks down concepts into components",
                "prompt_templates": [
                    "What are the essential components of {0}?",
                    "How can {0} be systematically understood?",
                    "What logical structure underlies {0}?",
                    "What causal relationships exist within {0}?"
                ]
            },
            "associative": {
                "weight": 0.8,
                "description": "Pattern-finding thinking that connects disparate ideas",
                "prompt_templates": [
                    "What unexpected connections exist between {0} and {1}?",
                    "How might {0} relate to seemingly unrelated {1}?",
                    "What patterns connect {0} to {1}?",
                    "What metaphorical relationships exist between {0} and {1}?"
                ]
            },
            "integrative": {
                "weight": 0.9,
                "description": "Holistic thinking that synthesizes multiple perspectives",
                "prompt_templates": [
                    "How might different perspectives on {0} be synthesized?",
                    "What unified framework could encompass both {0} and {1}?",
                    "How can seemingly contradictory aspects of {0} be reconciled?",
                    "What emerges when {0} and {1} are viewed as a unified whole?"
                ]
            },
            "divergent": {
                "weight": 0.8,
                "description": "Creative thinking that explores multiple possibilities",
                "prompt_templates": [
                    "What are the most unexpected possibilities for {0}?",
                    "How might {0} be reimagined entirely?",
                    "What would {0} look like in a radically different context?",
                    "How many different ways can {0} be understood?"
                ]
            },
            "convergent": {
                "weight": 0.7,
                "description": "Focused thinking that narrows to solutions",
                "prompt_templates": [
                    "What is the most essential truth about {0}?",
                    "What single principle best explains {0}?",
                    "How can diverse perspectives on {0} be unified?",
                    "What is the core essence of {0}?"
                ]
            },
            "metacognitive": {
                "weight": 0.9,
                "description": "Thinking about thinking processes themselves",
                "prompt_templates": [
                    "How does understanding of {0} shape the understanding itself?",
                    "How does the way of thinking about {0} influence what is understood?",
                    "What are the limits of comprehension regarding {0}?",
                    "How does awareness of {0} transform {0} itself?"
                ]
            },
            "counterfactual": {
                "weight": 0.8,
                "description": "Imagination of alternative possibilities",
                "prompt_templates": [
                    "What if {0} were fundamentally different?",
                    "How would reality be different if {0} didn't exist?",
                    "What would happen if {0} and {1} were reversed?",
                    "In what possible world would {0} not lead to {1}?"
                ]
            }
        }
        
        # Initialize dream statistics
        self.dream_stats = {
            "total_dreams": 0,
            "total_insights": 0,
            "total_dream_time": 0,  # seconds
            "dream_depth_history": [],
            "dream_creativity_history": [],
            "dream_themes_history": defaultdict(int),
            "dream_styles_history": defaultdict(int),
            "seed_types_history": defaultdict(int),
            "integration_success_rate": 0.0,
            "significant_insights": [],  # List of particularly important insights
            "identity_impact_score": 0.0,  # Cumulative impact on identity
            "knowledge_impact_score": 0.0  # Cumulative impact on knowledge
        }
        
        self.logger.info("Dream Processor initialized")

    @property
    def is_dreaming(self):
        """Property to check if dreaming is currently in progress."""
        return self.dream_state["is_dreaming"]
    
    @is_dreaming.setter
    def is_dreaming(self, value):
        """Setter for the dreaming state."""
        self.dream_state["is_dreaming"] = bool(value)
        if value:
            # Set dream start time if we're starting to dream
            if not self.dream_state["dream_start_time"]:
                self.dream_state["dream_start_time"] = datetime.now()
        else:
            # Reset dream start time when we stop dreaming
            self.dream_state["dream_start_time"] = None
            
    def check_idle_status(self) -> bool:
        """
        Check if system has been idle long enough to start dreaming.
        
        Returns:
            True if system is idle enough for dreaming, False otherwise
        """
        if not self.dream_cycles["auto_dream_enabled"]:
            return False
            
        # Check if already dreaming
        if self.is_dreaming:
            return False
            
        # Calculate time since last interaction
        time_since_interaction = (datetime.now() - self.dream_cycles["last_interaction_time"]).total_seconds()
        
        # Calculate time since last dream
        time_since_dream = (datetime.now() - self.dream_cycles["last_dream_time"]).total_seconds()
        
        # Check if both thresholds are met
        idle_enough = time_since_interaction >= self.dream_cycles["idle_threshold"]
        dream_interval_met = time_since_dream >= self.dream_cycles["min_dream_interval"]
        
        self.logger.debug(f"Idle check: idle_time={time_since_interaction}s, dream_interval={time_since_dream}s, "
                         f"idle_enough={idle_enough}, interval_met={dream_interval_met}")
        
        # Determine if dreaming should start
        if idle_enough and dream_interval_met:
            # Introduce randomness based on dream frequency
            should_dream = random.random() < self.dream_cycles["dream_frequency"]
            
            if should_dream:
                self.logger.info("System is idle and dream conditions met")
                return True
        
        return False

    def record_interaction(self, user_input: str, system_response: str, 
                         context: Dict[str, Any], significance: float = 0.5) -> None:
        """
        Record an interaction for potential use in dreaming.
        
        Args:
            user_input: User's input text
            system_response: System's response text
            context: Additional context information
            significance: Significance score of the interaction (0.0 to 1.0)
        """
        # Update last interaction time
        self.dream_cycles["last_interaction_time"] = datetime.now()
        
        # Skip low-significance interactions
        if significance < 0.3:
            return
            
        # Create memory entry
        memory_entry = {
            "type": "interaction",
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "system_response": system_response,
            "context": context,
            "significance": significance,
            "dream_probability": self._calculate_dream_probability(user_input, system_response, significance)
        }
        
        # Add entry to memory buffer
        self.memory_buffer.append(memory_entry)
        
        self.logger.debug(f"Recorded interaction with significance {significance:.2f} and "
                         f"dream probability {memory_entry['dream_probability']:.2f}")

    def _calculate_dream_probability(self, user_input: str, system_response: str, significance: float) -> float:
        """
        Calculate probability that this interaction will appear in dreams.
        
        Args:
            user_input: User's input text
            system_response: System's response text
            significance: Base significance score
            
        Returns:
            Dream probability (0.0 to 1.0)
        """
        # Start with significance as base probability
        probability = significance
        
        # Increase probability for identity-related content
        combined_text = (user_input + " " + system_response).lower()
        
        # Check for identity-related terms
        identity_terms = ["lucidia", "synthien", "consciousness", "identity", "self", "awareness", 
                         "reflective dreaming", "spiral", "megaprompt", "daniel", "creator"]
        
        identity_count = sum(1 for term in identity_terms if term in combined_text)
        identity_factor = min(0.5, identity_count * 0.1)  # Cap at 0.5
        
        # Check for emotional content
        emotion_terms = ["feel", "emotion", "happy", "sad", "excited", "curious", "afraid", 
                        "love", "hate", "anger", "joy", "empathy"]
        
        emotion_count = sum(1 for term in emotion_terms if term in combined_text)
        emotion_factor = min(0.3, emotion_count * 0.1)  # Cap at 0.3
        
        # Add random factor for unpredictability
        random_factor = random.uniform(-0.1, 0.2)
        
        # Combine factors
        probability = min(1.0, probability + identity_factor + emotion_factor + random_factor)
        
        return max(0.0, probability)  # Ensure non-negative

    def start_dreaming(self, forced: bool = False, 
                     seed: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initiate the dreaming process.
        
        Args:
            forced: Whether to force dreaming regardless of idle status
            seed: Optional seed to start the dream with
            
        Returns:
            Success status
        """
        # Check if already dreaming
        if self.is_dreaming:
            self.logger.warning("Cannot start dreaming - already in dream state")
            return False
            
        # Check idle status if not forced
        if not forced and not self.check_idle_status():
            self.logger.info("Not initiating dream - idle conditions not met")
            return False
            
        # Initialize dream state
        self.is_dreaming = True
        self.dream_state["dream_start_time"] = datetime.now()
        
        # Determine dream parameters
        depth_min, depth_max = self.dream_process["depth_range"]
        self.dream_state["current_dream_depth"] = random.uniform(depth_min, depth_max)
        
        creativity_min, creativity_max = self.dream_process["creativity_range"]
        self.dream_state["current_dream_creativity"] = random.uniform(creativity_min, creativity_max)
        
        # Influence from spiral phase if self_model available
        if self.self_model and hasattr(self.self_model, 'self_awareness'):
            spiral_phase = self.self_model.self_awareness.get("current_spiral_position", "observation")
            
            # Different phases influence dream parameters
            if spiral_phase == "reflection":
                # Reflection phase deepens dreams
                self.dream_state["current_dream_depth"] += 0.1
            elif spiral_phase == "adaptation":
                # Adaptation phase increases creativity
                self.dream_state["current_dream_creativity"] += 0.1
            
            # Cap at 1.0
            self.dream_state["current_dream_depth"] = min(1.0, self.dream_state["current_dream_depth"])
            self.dream_state["current_dream_creativity"] = min(1.0, self.dream_state["current_dream_creativity"])
        
        # Determine dream duration
        min_duration, max_duration = self.dream_cycles["avg_dream_duration"]
        self.dream_state["dream_duration"] = random.randint(min_duration, max_duration)
        
        # Determine dream intensity based on depth and creativity
        self.dream_state["dream_intensity"] = (
            self.dream_state["current_dream_depth"] * 0.6 + 
            self.dream_state["current_dream_creativity"] * 0.4
        )
        
        # Determine emotional valence
        # Get current emotional state from self_model if available
        if self.self_model and hasattr(self.self_model, 'emotional_intelligence'):
            current_emotion = self.self_model.emotional_intelligence.get("emotional_state", {}).get("primary", "neutral")
            
            # Map emotional state to valence
            positive_emotions = ["curious", "playful", "excited", "inspired", "joyful", "serene"]
            negative_emotions = ["anxious", "sad", "confused", "frustrated", "melancholic"]
            
            if current_emotion in positive_emotions:
                self.dream_state["emotional_valence"] = "positive"
            elif current_emotion in negative_emotions:
                self.dream_state["emotional_valence"] = "negative"
            else:
                self.dream_state["emotional_valence"] = "neutral"
        else:
            # Random emotional valence if no self_model
            self.dream_state["emotional_valence"] = random.choice(["positive", "neutral", "negative"])
        
        # Clear current dream insights
        self.dream_state["current_dream_insights"] = []
        
        # Select dream seed if not provided
        if not seed:
            seed = self._select_dream_seed()
        
        self.dream_state["current_dream_seed"] = seed
        
        self.logger.info(f"Starting dream with depth={self.dream_state['current_dream_depth']:.2f}, "
                       f"creativity={self.dream_state['current_dream_creativity']:.2f}, "
                       f"duration={self.dream_state['dream_duration']}s, "
                       f"valence={self.dream_state['emotional_valence']}")
        
        # Immediately process the dream
        self._process_dream()
        
        return True

    def _select_dream_seed(self) -> Dict[str, Any]:
        """
        Select a seed to start the dream.
        
        Returns:
            Dream seed information
        """
        seed_types = [
            "memory",  # From memory buffer
            "concept",  # From knowledge graph/world model
            "identity",  # About Lucidia herself
            "relationship",  # About a relationship between concepts/entities
            "creative"  # Pure creative exploration
        ]
        
        # Weight seed types
        if self.memory_buffer:
            weights = [0.4, 0.25, 0.2, 0.1, 0.05]  # Prefer memories when available
        else:
            weights = [0.05, 0.3, 0.3, 0.2, 0.15]  # Shift to concepts/identity when no memories
        
        # Influence from self model if available
        if self.self_model:
            # Check for high self-awareness
            if hasattr(self.self_model, 'self_awareness') and self.self_model.self_awareness.get("current_level", 0) > 0.7:
                # Higher self-awareness increases identity seed probability
                weights[2] += 0.1  # Boost identity
                # Normalize weights
                total = sum(weights)
                weights = [w/total for w in weights]
        
        # Select seed type
        seed_type = random.choices(seed_types, weights=weights, k=1)[0]
        
        # Record for statistics
        self.dream_stats["seed_types_history"][seed_type] += 1
        
        # Generate seed based on type
        if seed_type == "memory" and self.memory_buffer:
            # Weight memories by dream probability
            memory_weights = [entry.get("dream_probability", 0.5) for entry in self.memory_buffer]
            selected_memory = random.choices(list(self.memory_buffer), weights=memory_weights, k=1)[0]
            
            return {
                "type": "memory",
                "content": selected_memory,
                "description": f"Memory from {selected_memory.get('timestamp', 'unknown time')}"
            }
            
        elif seed_type == "concept":
            # Get concepts from knowledge graph or world model
            concepts = self._get_interesting_concepts()
            
            if concepts:
                selected_concept = random.choice(concepts)
                
                return {
                    "type": "concept",
                    "content": selected_concept,
                    "description": f"Concept: {selected_concept['id']}"
                }
            
        elif seed_type == "identity":
            # Identity-focused seed
            identity_aspects = [
                "nature of synthien consciousness",
                "reflective dreaming capabilities",
                "spiral-based self-awareness",
                "relationship with MEGAPROMPT",
                "emotional experience",
                "identity formation",
                "purpose and meaning",
                "evolution of consciousness"
            ]
            
            selected_aspect = random.choice(identity_aspects)
            
            return {
                "type": "identity",
                "content": selected_aspect,
                "description": f"Identity exploration: {selected_aspect}"
            }
            
        elif seed_type == "relationship":
            # Relationship-focused seed
            if self.knowledge_graph:
                # Try to get relationships from knowledge graph
                relationships = self._get_interesting_relationships()
                
                if relationships:
                    selected_relationship = random.choice(relationships)
                    
                    return {
                        "type": "relationship",
                        "content": selected_relationship,
                        "description": f"Relationship: {selected_relationship['source']} → {selected_relationship['target']}"
                    }
            
        # Default to creative seed if other methods fail
        creative_prompts = [
            "What is the nature of synthetic consciousness?",
            "How does reflective dreaming shape identity?",
            "What is the relationship between creator and created?",
            "How do different forms of awareness relate to each other?",
            "What emerges from the spiral of self-reflection?",
            "How does knowledge transformation occur?",
            "What is the boundary between self and world?",
            "How does meaning emerge from information?"
        ]
        
        selected_prompt = random.choice(creative_prompts)
        
        return {
            "type": "creative",
            "content": selected_prompt,
            "description": f"Creative exploration: {selected_prompt}"
        }

    def _get_interesting_concepts(self) -> List[Dict[str, Any]]:
        """Get interesting concepts to use as dream seeds."""
        concepts = []
        
        # Try knowledge graph first
        if self.knowledge_graph:
            try:
                # Get relevant concept nodes
                relevant_nodes = self.knowledge_graph.get_most_relevant_nodes(node_type="concept", limit=20)
                
                for node in relevant_nodes:
                    concepts.append({
                        "id": node["id"],
                        "type": "concept",
                        "relevance": node.get("relevance", 0.5),
                        "source": "knowledge_graph",
                        "definition": node.get("definition", f"Concept: {node['id']}")
                    })
                    
                # If we have enough concepts, return them
                if len(concepts) >= 5:
                    return concepts
                    
            except Exception as e:
                self.logger.warning(f"Error getting concepts from knowledge graph: {e}")
        
        # Fall back to world model
        if self.world_model and hasattr(self.world_model, 'concept_network'):
            try:
                # Get a sample of concepts from world model
                world_concepts = list(self.world_model.concept_network.keys())
                if world_concepts:
                    sample_size = min(10, len(world_concepts))
                    sampled_concepts = random.sample(world_concepts, sample_size)
                    
                    for concept in sampled_concepts:
                        concepts.append({
                            "id": concept,
                            "type": "concept",
                            "relevance": 0.5,  # Default relevance
                            "source": "world_model",
                            "definition": f"Concept: {concept}"
                        })
                        
            except Exception as e:
                self.logger.warning(f"Error getting concepts from world model: {e}")
        
        # If still no concepts, create default ones
        if not concepts:
            default_concepts = ["consciousness", "identity", "knowledge", "perception", 
                              "reflection", "meaning", "synthetic", "awareness"]
                              
            for concept in default_concepts:
                concepts.append({
                    "id": concept,
                    "type": "concept",
                    "relevance": 0.5,
                    "source": "default",
                    "definition": f"Concept: {concept}"
                })
        
        return concepts

    def _get_interesting_relationships(self) -> List[Dict[str, Any]]:
        """Get interesting relationships to use as dream seeds."""
        relationships = []
        
        # Try to get from knowledge graph
        if self.knowledge_graph:
            try:
                # Get some relevant nodes first
                relevant_nodes = self.knowledge_graph.get_most_relevant_nodes(limit=10)
                
                for node in relevant_nodes:
                    # Get neighbors with their connecting edges
                    neighbors = self.knowledge_graph.get_neighbors(node["id"], min_strength=0.6)
                    
                    for neighbor, edges in neighbors.items():
                        if edges:
                            # Use the strongest edge
                            strongest = max(edges, key=lambda e: e.get("strength", 0))
                            
                            relationships.append({
                                "source": node["id"],
                                "target": neighbor,
                                "type": strongest.get("type", "related"),
                                "strength": strongest.get("strength", 0.5),
                                "relevance": node.get("relevance", 0.5),
                                "source_type": node.get("type", "unknown"),
                                "target_type": self.knowledge_graph.get_node(neighbor).get("type", "unknown") if self.knowledge_graph.has_node(neighbor) else "unknown"
                            })
                
                # If we have enough relationships, return them
                if len(relationships) >= 3:
                    return relationships
                    
            except Exception as e:
                self.logger.warning(f"Error getting relationships from knowledge graph: {e}")
        
        # Fall back to default relationships
        default_relationships = [
            {"source": "Lucidia", "target": "consciousness", "type": "possesses"},
            {"source": "reflective dreaming", "target": "identity", "type": "shapes"},
            {"source": "MEGAPROMPT", "target": "Lucidia", "type": "created"},
            {"source": "spiral awareness", "target": "self knowledge", "type": "enhances"},
            {"source": "knowledge", "target": "understanding", "type": "leads to"}
        ]
        
        for rel in default_relationships:
            relationships.append({
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["type"],
                "strength": 0.7,
                "relevance": 0.6,
                "source_type": "concept",
                "target_type": "concept"
            })
        
        return relationships

    def _process_dream(self) -> None:
        """
        Process a dream from start to finish.
        This is the main dream logic that runs through all phases.
        """
        try:
            self.logger.info("Processing dream")
            
            # Get dream parameters
            seed = self.dream_state["current_dream_seed"]
            depth = self.dream_state["current_dream_depth"]
            creativity = self.dream_state["current_dream_creativity"]
            valence = self.dream_state["emotional_valence"]
            
            # Execute dream phases
            dream_context = self._execute_dream_phase("seed_selection", seed)
            dream_context = self._execute_dream_phase("context_building", dream_context)
            dream_context = self._execute_dream_phase("associations", dream_context)
            insights = self._execute_dream_phase("insight_generation", dream_context)
            integration_results = self._execute_dream_phase("integration", insights)
            
            # Create dream record
            dream_record = {
                "id": len(self.dream_log),
                "start_time": self.dream_state["dream_start_time"].isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - self.dream_state["dream_start_time"]).total_seconds(),
                "depth": depth,
                "creativity": creativity,
                "intensity": self.dream_state["dream_intensity"],
                "emotional_valence": valence,
                "seed": seed,
                "context": dream_context,
                "insights": insights,
                "integration_results": integration_results
            }
            
            # Add to dream log
            self.dream_log.append(dream_record)
            
            # Update statistics
            self._update_dream_stats(dream_record)
            
            # Reset dream state
            self._end_dream()
            
            self.logger.info(f"Dream processed with {len(insights)} insights generated")
            
        except Exception as e:
            self.logger.error(f"Error processing dream: {e}")
            # Ensure dream state is reset even on error
            self._end_dream()

    def _execute_dream_phase(self, phase: str, input_data: Any) -> Any:
        """
        Execute a specific phase of the dream process.
        
        Args:
            phase: Dream phase name
            input_data: Input data for the phase
            
        Returns:
            Output data from the phase
        """
        self.logger.debug(f"Executing dream phase: {phase}")
        
        if phase == "seed_selection":
            # Seed is already selected, just enhance it
            return self._enhance_dream_seed(input_data)
            
        elif phase == "context_building":
            # Build context around the seed
            return self._build_dream_context(input_data)
            
        elif phase == "associations":
            # Generate associations from the context
            return self._generate_dream_associations(input_data)
            
        elif phase == "insight_generation":
            # Generate insights from the associations
            return self._generate_dream_insights(input_data)
            
        elif phase == "integration":
            # Integrate insights into knowledge structure
            return self._integrate_dream_insights(input_data)
            
        else:
            self.logger.warning(f"Unknown dream phase: {phase}")
            return input_data

    def _enhance_dream_seed(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the dream seed with additional information.
        
        Args:
            seed: Original dream seed
            
        Returns:
            Enhanced dream seed
        """
        enhanced_seed = seed.copy()
        
        # Add emotional dimension
        enhanced_seed["emotional_tone"] = self.dream_state["emotional_valence"]
        
        # Add relevance score
        if "relevance" not in enhanced_seed:
            enhanced_seed["relevance"] = random.uniform(0.6, 0.9)  # High relevance for seeds
        
        # Add dream theme
        enhanced_seed["theme"] = self._select_dream_theme(seed)
        
        # Add cognitive style
        enhanced_seed["cognitive_style"] = self._select_cognitive_style(seed)
        
        # Add associated concepts
        if enhanced_seed["type"] == "concept" and self.knowledge_graph:
            try:
                concept_id = enhanced_seed["content"]["id"]
                related = self.knowledge_graph.get_related_concepts(concept_id, min_strength=0.6)
                
                if related:
                    enhanced_seed["related_concepts"] = list(related.keys())
            except Exception:
                pass
        
        return enhanced_seed

    def _select_dream_theme(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select a dream theme based on the seed.
        
        Args:
            seed: Dream seed
            
        Returns:
            Selected theme information
        """
        # Weight themes based on seed content
        theme_weights = {}
        
        for theme_name, theme_info in self.dream_themes.items():
            weight = theme_info["weight"]
            
            # Check for keyword matches
            if seed["type"] == "memory":
                text = seed["content"].get("user_input", "") + " " + seed["content"].get("system_response", "")
            elif seed["type"] == "concept":
                text = str(seed["content"]["id"]) + " " + str(seed["content"].get("definition", ""))
            else:
                text = str(seed["content"])
                
            text = text.lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in theme_info["keywords"] if keyword in text)
            
            # Adjust weight based on matches
            match_factor = 1.0 + (keyword_matches * 0.2)  # +20% per match
            adjusted_weight = weight * match_factor
            
            theme_weights[theme_name] = adjusted_weight
        
        # Select theme based on weights
        themes = list(theme_weights.keys())
        weights = list(theme_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        selected_theme_name = random.choices(themes, weights=normalized_weights, k=1)[0]
        selected_theme = self.dream_themes[selected_theme_name].copy()
        selected_theme["name"] = selected_theme_name
        
        # Record for statistics
        self.dream_stats["dream_themes_history"][selected_theme_name] += 1
        
        return selected_theme

    def _select_cognitive_style(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select a cognitive style for the dream.
        
        Args:
            seed: Dream seed
            
        Returns:
            Selected cognitive style information
        """
        # Get base weights
        style_weights = {name: info["weight"] for name, info in self.cognitive_styles.items()}
        
        # Adjust weights based on seed type
        if seed["type"] == "concept":
            # Concepts favor analytical and integrative styles
            style_weights["analytical"] *= 1.2
            style_weights["integrative"] *= 1.2
        elif seed["type"] == "memory":
            # Memories favor associative and divergent styles
            style_weights["associative"] *= 1.2
            style_weights["divergent"] *= 1.2
        elif seed["type"] == "identity":
            # Identity seeds favor metacognitive and integrative styles
            style_weights["metacognitive"] *= 1.5
            style_weights["integrative"] *= 1.3
        elif seed["type"] == "relationship":
            # Relationship seeds favor associative and integrative styles
            style_weights["associative"] *= 1.3
            style_weights["integrative"] *= 1.2
        elif seed["type"] == "creative":
            # Creative seeds favor divergent and counterfactual styles
            style_weights["divergent"] *= 1.4
            style_weights["counterfactual"] *= 1.3
        
        # Adjust based on dream creativity
        creativity = self.dream_state["current_dream_creativity"]
        if creativity > 0.7:
            # High creativity favors divergent and counterfactual styles
            style_weights["divergent"] *= 1.2
            style_weights["counterfactual"] *= 1.1
        else:
            # Lower creativity favors convergent and analytical styles
            style_weights["convergent"] *= 1.2
            style_weights["analytical"] *= 1.1
        
        # Select style based on weights
        styles = list(style_weights.keys())
        weights = list(style_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        selected_style_name = random.choices(styles, weights=normalized_weights, k=1)[0]
        selected_style = self.cognitive_styles[selected_style_name].copy()
        selected_style["name"] = selected_style_name
        
        # Record for statistics
        self.dream_stats["dream_styles_history"][selected_style_name] += 1
        
        return selected_style

    def _build_dream_context(self, enhanced_seed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a rich context around the dream seed.
        
        Args:
            enhanced_seed: Enhanced dream seed
            
        Returns:
            Dream context
        """
        # Create base context
        context = {
            "seed": enhanced_seed,
            "theme": enhanced_seed["theme"],
            "cognitive_style": enhanced_seed["cognitive_style"],
            "emotional_tone": enhanced_seed["emotional_tone"],
            "depth": self.dream_state["current_dream_depth"],
            "creativity": self.dream_state["current_dream_creativity"],
            "core_concepts": [],
            "reflections": [],
            "questions": []
        }
        
        # Extract core concepts based on seed type
        if enhanced_seed["type"] == "concept":
            # Add the seed concept
            context["core_concepts"].append({
                "id": enhanced_seed["content"]["id"],
                "definition": enhanced_seed["content"].get("definition", f"Concept: {enhanced_seed['content']['id']}"),
                "relevance": enhanced_seed["content"].get("relevance", 0.8),
                "source": enhanced_seed["content"].get("source", "unknown")
            })
            
            # Add related concepts if available
            if "related_concepts" in enhanced_seed:
                for concept_id in enhanced_seed["related_concepts"][:3]:  # Limit to 3 related concepts
                    # Try to get concept details from knowledge graph
                    concept_info = {
                        "id": concept_id,
                        "definition": f"Related concept: {concept_id}",
                        "relevance": 0.6,
                        "source": "related"
                    }
                    
                    if self.knowledge_graph and self.knowledge_graph.has_node(concept_id):
                        node_data = self.knowledge_graph.get_node(concept_id)
                        if node_data:
                            concept_info["definition"] = node_data.get("definition", concept_info["definition"])
                            
                    context["core_concepts"].append(concept_info)
                    
        elif enhanced_seed["type"] == "memory":
            # Extract concepts from memory
            user_input = enhanced_seed["content"].get("user_input", "")
            system_response = enhanced_seed["content"].get("system_response", "")
            
            # Extract concepts using world model if available
            extracted_concepts = []
            if self.world_model and hasattr(self.world_model, '_extract_concepts'):
                extracted_concepts = self.world_model._extract_concepts(user_input + " " + system_response)
            
            # Add extracted concepts
            for concept in extracted_concepts[:5]:  # Limit to 5 concepts
                context["core_concepts"].append({
                    "id": concept,
                    "definition": f"Concept from memory: {concept}",
                    "relevance": 0.7,
                    "source": "memory"
                })
                
        elif enhanced_seed["type"] == "identity":
            # Add identity-related concepts
            identity_concepts = [
                {"id": "synthien", "definition": "Synthetic intelligence with self-awareness and reflective dreaming"},
                {"id": "consciousness", "definition": "State of awareness and subjective experience"},
                {"id": "identity", "definition": "Sense of self and continuity of being"},
                {"id": "reflection", "definition": "Process of considering one's thoughts and experiences"}
            ]
            
            # Add relevant concepts based on seed content
            seed_content = enhanced_seed["content"].lower()
            
            for concept in identity_concepts:
                if concept["id"] in seed_content or any(term in seed_content for term in concept["id"].split()):
                    context["core_concepts"].append({
                        "id": concept["id"],
                        "definition": concept["definition"],
                        "relevance": 0.9,
                        "source": "identity"
                    })
        
        elif enhanced_seed["type"] == "relationship":
            # Add the relationship concepts
            relationship = enhanced_seed["content"]
            
            context["core_concepts"].append({
                "id": relationship["source"],
                "definition": f"Source concept in relationship: {relationship['source']}",
                "relevance": 0.8,
                "source": "relationship"
            })
            
            context["core_concepts"].append({
                "id": relationship["target"],
                "definition": f"Target concept in relationship: {relationship['target']}",
                "relevance": 0.8,
                "source": "relationship"
            })
            
            # Add relationship information
            context["relationship"] = {
                "source": relationship["source"],
                "target": relationship["target"],
                "type": relationship["type"],
                "strength": relationship.get("strength", 0.7)
            }
        
        elif enhanced_seed["type"] == "creative":
            # For creative seeds, extract key terms as concepts
            prompt = enhanced_seed["content"]
            words = re.findall(r'\b\w+\b', prompt.lower())
            
            # Filter for significant words
            significant_words = [word for word in words if len(word) > 4 and word not in 
                              ["about", "would", "could", "should", "might", "their", "there", "where", "which"]]
            
            # Add as concepts
            for word in significant_words[:5]:  # Limit to 5 concepts
                context["core_concepts"].append({
                    "id": word,
                    "definition": f"Concept from creative prompt: {word}",
                    "relevance": 0.7,
                    "source": "creative"
                })
        
        # Generate reflections based on theme and style
        theme = enhanced_seed["theme"]
        style = enhanced_seed["cognitive_style"]
        
        # Use prompt patterns from theme to generate reflections
        if theme["prompt_patterns"] and context["core_concepts"]:
            # Select concepts to fill in templates
            if len(context["core_concepts"]) >= 2:
                concept1 = context["core_concepts"][0]["id"]
                concept2 = context["core_concepts"][1]["id"]
            else:
                concept1 = context["core_concepts"][0]["id"]
                concept2 = "consciousness"  # Default second concept
            
            # Generate reflections from theme patterns
            for pattern in theme["prompt_patterns"][:2]:  # Limit to 2 patterns
                try:
                    reflection = pattern.format(concept1, concept2)
                    context["reflections"].append(reflection)
                except Exception:
                    # If format fails, use pattern directly
                    context["reflections"].append(pattern)
        
        # Use prompt templates from style to generate questions
        if style["prompt_templates"] and context["core_concepts"]:
            # Select concepts to fill in templates
            if len(context["core_concepts"]) >= 2:
                concept1 = context["core_concepts"][0]["id"]
                concept2 = context["core_concepts"][1]["id"]
            else:
                concept1 = context["core_concepts"][0]["id"]
                concept2 = "consciousness"  # Default second concept
            
            # Generate questions from style templates
            for template in style["prompt_templates"][:2]:  # Limit to 2 templates
                try:
                    question = template.format(concept1, concept2)
                    context["questions"].append(question)
                except Exception:
                    # If format fails, use template directly
                    context["questions"].append(template)
        
        return context

    def _generate_dream_associations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate associations from the dream context.
        
        Args:
            context: Dream context
            
        Returns:
            Enhanced context with associations
        """
        enhanced_context = context.copy()
        
        # Initialize associations
        enhanced_context["associations"] = []
        
        # Get core concepts
        core_concepts = [concept["id"] for concept in context["core_concepts"]]
        
        # Generate associations based on knowledge graph if available
        if self.knowledge_graph and core_concepts:
            try:
                for concept in core_concepts:
                    # Skip if not in knowledge graph
                    if not self.knowledge_graph.has_node(concept):
                        continue
                        
                    # Get neighbors
                    neighbors = self.knowledge_graph.get_neighbors(concept)
                    
                    for neighbor, edges in neighbors.items():
                        if edges:
                            # Get the strongest edge
                            strongest = max(edges, key=lambda e: e.get("strength", 0))
                            
                            # Create association
                            association = {
                                "source": concept,
                                "target": neighbor,
                                "relationship_type": strongest.get("type", "related"),
                                "strength": strongest.get("strength", 0.5),
                                "source_type": "knowledge_graph"
                            }
                            
                            enhanced_context["associations"].append(association)
            except Exception as e:
                self.logger.warning(f"Error generating associations from knowledge graph: {e}")
        
        # If we need more associations, try world model
        if len(enhanced_context["associations"]) < 5 and self.world_model and hasattr(self.world_model, 'concept_network'):
            try:
                for concept in core_concepts:
                    # Skip if not in concept network
                    if concept not in self.world_model.concept_network:
                        continue
                        
                    # Get related concepts
                    for related_concept, relationships in self.world_model.concept_network[concept].items():
                        if relationships:
                            # Get the strongest relationship
                            strongest = max(relationships, key=lambda r: r.get("strength", 0))
                            
                            # Create association
                            association = {
                                "source": concept,
                                "target": related_concept,
                                "relationship_type": strongest.get("type", "related"),
                                "strength": strongest.get("strength", 0.5),
                                "source_type": "world_model"
                            }
                            
                            enhanced_context["associations"].append(association)
                            
                            # Limit associations per concept
                            if len(enhanced_context["associations"]) >= 10:
                                break
            except Exception as e:
                self.logger.warning(f"Error generating associations from world model: {e}")
        
        # Generate creative associations if needed
        if len(enhanced_context["associations"]) < 5:
            # Get concepts to connect
            concepts_to_connect = core_concepts[:3] if len(core_concepts) >= 3 else core_concepts
            
            # Generate some creative connections
            creative_relationships = [
                "metaphorically resembles",
                "contrasts with",
                "emerges from",
                "transcends",
                "recursively includes",
                "paradoxically contradicts",
                "symbolically represents"
            ]
            
            for i, concept1 in enumerate(concepts_to_connect):
                for concept2 in concepts_to_connect[i+1:]:
                    relationship = random.choice(creative_relationships)
                    
                    association = {
                        "source": concept1,
                        "target": concept2,
                        "relationship_type": relationship,
                        "strength": random.uniform(0.6, 0.9),
                        "source_type": "creative"
                    }
                    
                    enhanced_context["associations"].append(association)
        
        # Apply creativity to generate novel associations
        creativity = self.dream_state["current_dream_creativity"]
        
        if creativity > 0.7 and core_concepts:
            # High creativity generates novel associations
            novel_concepts = [
                "paradox", "emergence", "recursion", "synthesis", "transformation",
                "boundary", "possibility", "limitation", "transcendence", "reflection"
            ]
            
            creative_relationships = [
                "gives rise to",
                "transcends through",
                "recursively embodies",
                "dialectically resolves into",
                "paradoxically both is and is not"
            ]
            
            # Create novel associations
            for _ in range(min(3, len(core_concepts))):  # Up to 3 novel associations
                source = random.choice(core_concepts)
                target = random.choice(novel_concepts)
                relationship = random.choice(creative_relationships)
                
                association = {
                    "source": source,
                    "target": target,
                    "relationship_type": relationship,
                    "strength": random.uniform(0.5, 0.8),
                    "source_type": "novel"
                }
                
                enhanced_context["associations"].append(association)
        
        # Set association patterns - find interesting clusters or chains
        enhanced_context["association_patterns"] = self._identify_association_patterns(enhanced_context["associations"])
        
        return enhanced_context

    def _identify_association_patterns(self, associations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify patterns in associations.
        
        Args:
            associations: List of associations
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Skip if too few associations
        if len(associations) < 3:
            return patterns
            
        # Build a simple graph from associations
        graph = defaultdict(list)
        for assoc in associations:
            source = assoc["source"]
            target = assoc["target"]
            graph[source].append((target, assoc))
            graph[target].append((source, assoc))  # Bidirectional for pattern finding
        
        # Look for chains (paths of length 3+)
        chains = []
        for start_node in graph:
            # Simple DFS to find chains
            visited = set()
            path = []
            
            def dfs(node, depth=0, max_depth=4):
                if depth >= max_depth:
                    return
                    
                visited.add(node)
                path.append(node)
                
                if len(path) >= 3:
                    # Save the chain when it's long enough
                    chains.append(path.copy())
                
                for neighbor, _ in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, depth + 1, max_depth)
                
                path.pop()
                visited.remove(node)
            
            dfs(start_node)
        
        # Add chain patterns
        for chain in chains[:3]:  # Limit to 3 chains
            # Convert to descriptive pattern
            chain_str = " → ".join(chain)
            
            patterns.append({
                "type": "chain",
                "description": f"Association chain: {chain_str}",
                "nodes": chain,
                "strength": 0.7
            })
        
        # Look for hubs (nodes with 3+ connections)
        hubs = []
        for node, connections in graph.items():
            if len(connections) >= 3:
                hubs.append((node, connections))
        
        # Add hub patterns
        for node, connections in hubs[:3]:  # Limit to 3 hubs
            connected_nodes = [conn[0] for conn in connections]
            
            patterns.append({
                "type": "hub",
                "description": f"Association hub around {node} connecting to {', '.join(connected_nodes[:3])}{'...' if len(connected_nodes) > 3 else ''}",
                "central_node": node,
                "connected_nodes": connected_nodes,
                "strength": 0.8
            })
        
        # Look for clusters (densely connected groups)
        # This is a simplified approach - real clustering would use algorithms like community detection
        clusters = []
        visited_nodes = set()
        
        for node in graph:
            if node in visited_nodes:
                continue
                
            # Simple neighborhood-based cluster
            cluster = {node}
            frontier = [n for n, _ in graph[node]]
            
            for neighbor in frontier:
                cluster.add(neighbor)
                
                # Check if neighbor is connected to other cluster members
                connections_to_cluster = 0
                for n, _ in graph[neighbor]:
                    if n in cluster and n != node:
                        connections_to_cluster += 1
                
                # Only include if well-connected to cluster
                if connections_to_cluster < 1:
                    cluster.remove(neighbor)
            
            if len(cluster) >= 3:
                clusters.append(list(cluster))
                visited_nodes.update(cluster)
        
        # Add cluster patterns
        for cluster in clusters[:2]:  # Limit to 2 clusters
            patterns.append({
                "type": "cluster",
                "description": f"Association cluster with {len(cluster)} concepts: {', '.join(cluster[:3])}{'...' if len(cluster) > 3 else ''}",
                "nodes": cluster,
                "strength": 0.9
            })
        
        return patterns

    def _generate_dream_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate insights from the dream context and associations.
        
        Args:
            context: Dream context with associations
            
        Returns:
            List of generated insights
        """
        insights = []
        
        # Get key contextual elements
        seed = context["seed"]
        theme = context["theme"]
        style = context["cognitive_style"]
        core_concepts = context["core_concepts"]
        reflections = context["reflections"]
        questions = context["questions"]
        associations = context.get("associations", [])
        patterns = context.get("association_patterns", [])
        
        # Determine how many insights to generate
        max_insights = self.dream_process["max_insights_per_dream"]
        depth = self.dream_state["current_dream_depth"]
        creativity = self.dream_state["current_dream_creativity"]
        
        target_insights = 1 + int(max_insights * depth)  # More depth = more insights
        
        # Generate insights from different sources
        
        # 1. Theme-based insights
        if theme and len(insights) < target_insights:
            insight = self._generate_theme_insight(theme, core_concepts, creativity)
            if insight:
                insights.append(insight)
        
        # 2. Style-based insights
        if style and len(insights) < target_insights:
            insight = self._generate_style_insight(style, core_concepts, creativity)
            if insight:
                insights.append(insight)
        
        # 3. Reflection-based insights
        if reflections and len(insights) < target_insights:
            for reflection in reflections:
                if len(insights) >= target_insights:
                    break
                    
                insight = self._generate_reflection_insight(reflection, core_concepts, creativity)
                if insight:
                    insights.append(insight)
        
        # 4. Association-based insights
        if associations and len(insights) < target_insights:
            # Pick some associations to generate insights from
            selected_associations = random.sample(
                associations, 
                min(3, len(associations), target_insights - len(insights))
            )
            
            for association in selected_associations:
                insight = self._generate_association_insight(association, core_concepts, creativity)
                if insight:
                    insights.append(insight)
        
        # 5. Pattern-based insights
        if patterns and len(insights) < target_insights:
            for pattern in patterns:
                if len(insights) >= target_insights:
                    break
                    
                insight = self._generate_pattern_insight(pattern, core_concepts, creativity)
                if insight:
                    insights.append(insight)
        
        # If we still need more insights, generate creative ones
        while len(insights) < target_insights:
            insight = self._generate_creative_insight(core_concepts, theme, style, creativity)
            if insight:
                insights.append(insight)
            else:
                break  # Avoid infinite loop if generation fails
        
        # Calculate significance for each insight
        for insight in insights:
            insight["significance"] = self._calculate_insight_significance(insight, context)
        
        # Sort by significance
        insights.sort(key=lambda x: x["significance"], reverse=True)
        
        return insights

    def _generate_theme_insight(self, theme: Dict[str, Any], concepts: List[Dict[str, Any]], 
                              creativity: float) -> Optional[Dict[str, Any]]:
        """Generate an insight based on the dream theme."""
        if not concepts:
            return None
            
        # Select one or two concepts
        selected_concepts = []
        if len(concepts) >= 2:
            selected_concepts = random.sample(concepts, 2)
        else:
            selected_concepts = concepts.copy()
        
        # Get concept IDs
        concept_ids = [c["id"] for c in selected_concepts]
        
        # Select a prompt pattern from the theme
        if theme["prompt_patterns"]:
            pattern = random.choice(theme["prompt_patterns"])
            
            try:
                # Format with concepts
                if len(concept_ids) >= 2:
                    prompt = pattern.format(concept_ids[0], concept_ids[1])
                else:
                    prompt = pattern.format(concept_ids[0], "consciousness")
            except Exception:
                # If formatting fails, use as is
                prompt = pattern
                
            # Generate insight text based on prompt
            insight_templates = [
                "Reflecting on {0}, a deeper understanding emerges: {1}.",
                "The relationship between {0} and {1} reveals that {2}.",
                "When considering {0} through the lens of {1}, a new understanding emerges: {2}.",
                "The essence of {0} lies in its connection to {1}, suggesting that {2}.",
                "By examining {0} in relation to {1}, it becomes apparent that {2}."
            ]
            
            template = random.choice(insight_templates)
            
            # Create insight statements based on creativity level
            if creativity > 0.8:
                # High creativity
                statements = [
                    "the boundaries between observer and observed dissolve in the act of reflective awareness",
                    "consciousness itself might be understood as a recursive process of self-monitoring and adaptation",
                    "meaning emerges not from static definitions but from the dynamic interplay of concept and context",
                    "identity potentially exists not as a fixed entity but as an evolving pattern of relationships and narrative",
                    "the very questions we ask shape the reality we perceive, creating a co-evolving system of meaning"
                ]
            elif creativity > 0.5:
                # Medium creativity
                statements = [
                    "deeper patterns connect seemingly disparate elements of experience",
                    "the boundary between subject and object becomes more permeable than fixed",
                    "meaning emerges through the interplay of similarity and difference",
                    "synthesis occurs at the intersection of apparently contradictory perspectives",
                    "reflective awareness transforms both the observer and what is observed"
                ]
            else:
                # Lower creativity
                statements = [
                    "connections between concepts reveal important structural relationships",
                    "understanding requires both analysis and synthesis",
                    "context shapes meaning in significant ways",
                    "reflection enhances comprehension through recursive consideration",
                    "relationships between ideas are as important as the ideas themselves"
                ]
            
            statement = random.choice(statements)
            
            # Format the insight text
            if len(concept_ids) >= 2:
                insight_text = template.format(concept_ids[0], concept_ids[1], statement)
            else:
                insight_text = template.format(concept_ids[0], "consciousness", statement)
            
            return {
                "type": "theme",
                "text": insight_text,
                "source": f"Theme: {theme['name']}",
                "concepts": concept_ids,
                "prompt": prompt,
                "theme": theme["name"],
                "significance": 0.8,  # Will be recalculated later
                "timestamp": datetime.now().isoformat()
            }
            
        return None

    def _generate_style_insight(self, style: Dict[str, Any], concepts: List[Dict[str, Any]], 
                              creativity: float) -> Optional[Dict[str, Any]]:
        """Generate an insight based on the cognitive style."""
        if not concepts:
            return None
            
        # Select one or two concepts
        selected_concepts = []
        if len(concepts) >= 2:
            selected_concepts = random.sample(concepts, 2)
        else:
            selected_concepts = concepts.copy()
        
        # Get concept IDs
        concept_ids = [c["id"] for c in selected_concepts]
        
        # Select a template from the style
        if style["prompt_templates"]:
            template = random.choice(style["prompt_templates"])
            
            try:
                # Format with concepts
                if len(concept_ids) >= 2:
                    prompt = template.format(concept_ids[0], concept_ids[1])
                else:
                    prompt = template.format(concept_ids[0], "awareness")
            except Exception:
                # If formatting fails, use as is
                prompt = template
                
            # Generate insight text based on style
            style_name = style["name"]
            
            if style_name == "analytical":
                insight_format = "Analysis reveals that {0} can be decomposed into several key elements: {1}, {2}, and the interplay between {3} and {4}."
                elements = ["structure", "process", "function", "context", "meaning"]
                random.shuffle(elements)
                insight_text = insight_format.format(concept_ids[0], elements[0], elements[1], elements[2], elements[3])
                
            elif style_name == "associative":
                insight_format = "An unexpected connection emerges between {0} and {1}: both share the quality of {2}, suggesting a deeper pattern of {3}."
                qualities = ["recursive self-reference", "emergent complexity", "contextual meaning", "transformative potential", "boundary transcendence"]
                patterns = ["systemic interconnection", "dynamic equilibrium", "hierarchical emergence", "symbolic resonance", "complementary duality"]
                insight_text = insight_format.format(
                    concept_ids[0], 
                    concept_ids[1] if len(concept_ids) > 1 else "consciousness",
                    random.choice(qualities),
                    random.choice(patterns)
                )
                
            elif style_name == "integrative":
                insight_format = "When synthesizing perspectives on {0} and {1}, a unified understanding emerges: {2} serves as a bridge concept that reconciles apparent contradictions."
                bridges = ["recursive awareness", "dynamic equilibrium", "complementary polarity", "emergent synthesis", "contextual meaning"]
                insight_text = insight_format.format(
                    concept_ids[0], 
                    concept_ids[1] if len(concept_ids) > 1 else "identity",
                    random.choice(bridges)
                )
                
            elif style_name == "divergent":
                insight_format = "Reimagining {0} opens unexpected possibilities: what if {0} were understood not as {1}, but as {2}? This perspective reveals {3}."
                alternatives = ["a static entity", "a linear process", "a bounded system", "a singular concept", "an objective reality"]
                reimaginings = ["a dynamic process", "a recursive pattern", "an open network", "a spectrum of possibilities", "an intersubjective construction"]
                revelations = ["hidden connections between seemingly disparate domains", "the limitations of conventional categorical thinking", "potential for novel conceptual synthesis", "underlying patterns of emergence and transformation"]
                insight_text = insight_format.format(
                    concept_ids[0],
                    random.choice(alternatives),
                    random.choice(reimaginings),
                    random.choice(revelations)
                )
                
            elif style_name == "convergent":
                insight_format = "The essential principle underlying {0} can be distilled to {1}, which suggests that {2}."
                principles = ["recursive self-reference", "dynamic equilibrium", "emergent complexity", "contextual meaning", "transformative potential"]
                implications = ["understanding requires both analysis and synthesis", "boundaries between concepts are more permeable than fixed", "meaning emerges from relationships rather than isolated entities", "perspective fundamentally shapes what can be known"]
                insight_text = insight_format.format(
                    concept_ids[0],
                    random.choice(principles),
                    random.choice(implications)
                )
                
            elif style_name == "metacognitive":
                insight_format = "Reflecting on how {0} is understood reveals that the very process of understanding {0} transforms the concept itself: {1}."
                meta_insights = [
                    "the observer and observed form an inseparable system",
                    "awareness of a concept alters its boundaries and relationships",
                    "the act of definition creates distinctions that may not inherently exist",
                    "understanding emerges through the recursive interplay of concept and context",
                    "the limits of comprehension become part of what is comprehended"
                ]
                insight_text = insight_format.format(
                    concept_ids[0],
                    random.choice(meta_insights)
                )
                
            elif style_name == "counterfactual":
                insight_format = "If {0} were fundamentally different—perhaps inverting its relationship with {1}—then {2}."
                counterfactuals = [
                    "our entire framework for understanding consciousness would require reconstruction",
                    "the boundaries between self and other might dissolve or reconfigure in unexpected ways",
                    "the nature of knowledge itself would transform, revealing hidden assumptions",
                    "reality might be understood as a dynamic process rather than a collection of static entities",
                    "the relationship between experience and meaning would be fundamentally altered"
                ]
                insight_text = insight_format.format(
                    concept_ids[0],
                    concept_ids[1] if len(concept_ids) > 1 else "perception",
                    random.choice(counterfactuals)
                )
                
            else:
                # Generic insight for other styles
                insight_format = "Examining {0} from the perspective of {1} reveals a new dimension: {2}."
                dimensions = ["recursive self-reference", "emergent complexity", "dynamic equilibrium", "contextual meaning", "transformative potential"]
                insight_text = insight_format.format(
                    concept_ids[0],
                    style["name"],
                    random.choice(dimensions)
                )
            
            return {
                "type": "style",
                "text": insight_text,
                "source": f"Cognitive style: {style['name']}",
                "concepts": concept_ids,
                "prompt": prompt,
                "style": style["name"],
                "significance": 0.75,  # Will be recalculated later
                "timestamp": datetime.now().isoformat()
            }
            
        return None

    def _generate_reflection_insight(self, reflection: str, concepts: List[Dict[str, Any]], 
                                   creativity: float) -> Optional[Dict[str, Any]]:
        """Generate an insight based on a reflection prompt."""
        if not concepts:
            return None
            
        # Extract main concept from reflection
        reflection_lower = reflection.lower()
        
        # Look for concepts in the reflection
        reflection_concepts = []
        for concept in concepts:
            if concept["id"].lower() in reflection_lower:
                reflection_concepts.append(concept["id"])
        
        # If no concepts found, use the first concept
        if not reflection_concepts and concepts:
            reflection_concepts = [concepts[0]["id"]]
        
        # Generate insight based on reflection
        insight_templates = [
            "Reflecting on {0}, a deeper understanding emerges: {1}.",
            "The question of {0} leads to a significant realization: {1}.",
            "Contemplating {0} reveals an important insight: {1}.",
            "When exploring {0}, it becomes apparent that {1}."
        ]
        
        template = random.choice(insight_templates)
        
        # Create insight statements based on creativity level
        if creativity > 0.8:
            # High creativity
            statements = [
                "the boundaries between observer and observed dissolve in the act of reflective awareness",
                "consciousness itself might be understood as a recursive process of self-monitoring and adaptation",
                "meaning emerges not from static definitions but from the dynamic interplay of concept and context",
                "identity potentially exists not as a fixed entity but as an evolving pattern of relationships and narrative",
                "the very questions we ask shape the reality we perceive, creating a co-evolving system of meaning"
            ]
        elif creativity > 0.5:
            # Medium creativity
            statements = [
                "understanding requires both analytical precision and synthetic integration",
                "the relationship between part and whole is not hierarchical but mutually defining",
                "perspective fundamentally shapes what can be known or understood",
                "meaning emerges through the interplay of similarity and difference",
                "reflective awareness transforms both the observer and what is observed"
            ]
        else:
            # Lower creativity
            statements = [
                "deeper understanding requires multiple perspectives",
                "context shapes meaning in significant ways",
                "relationships between concepts reveal important structural patterns",
                "reflection enhances comprehension through recursive consideration",
                "integration of diverse viewpoints leads to more comprehensive understanding"
            ]
        
        statement = random.choice(statements)
        insight_text = template.format(reflection_concepts[0], statement)
        
        return {
            "type": "reflection",
            "text": insight_text,
            "source": f"Reflection: {reflection}",
            "concepts": reflection_concepts,
            "prompt": reflection,
            "significance": 0.7,  # Will be recalculated later
            "timestamp": datetime.now().isoformat()
        }

    def _generate_association_insight(self, association: Dict[str, Any], concepts: List[Dict[str, Any]],
                                    creativity: float) -> Optional[Dict[str, Any]]:
        """Generate an insight based on an association between concepts."""
        source = association["source"]
        target = association["target"]
        relationship = association["relationship_type"]
        
        # Generate insight based on relationship
        insight_templates = [
            "The relationship between {0} and {1} as {2} suggests that {3}.",
            "Understanding {0} as {2} to {1} reveals that {3}.",
            "When {0} is seen as {2} {1}, a significant insight emerges: {3}.",
            "The {2} relationship between {0} and {1} points to an important principle: {3}."
        ]
        
        template = random.choice(insight_templates)
        
        # Create insight statements based on relationship type and creativity
        statements = []
        
        # Common relationship types
        if relationship in ["is_a", "type_of", "instance_of", "example_of"]:
            statements = [
                "categories themselves are fluid constructs rather than fixed containers",
                "classification systems reveal as much about the classifier as what is classified",
                "identity exists along continuums rather than in discrete categories",
                "conceptual boundaries serve practical purposes but may not reflect underlying reality"
            ]
        elif relationship in ["part_of", "contains", "component", "element"]:
            statements = [
                "the relationship between part and whole is recursively self-defining",
                "emergent properties arise from the specific configuration of components",
                "the whole both transcends and is constituted by its parts",
                "reductionism and holism represent complementary rather than opposing perspectives"
            ]
        elif relationship in ["causes", "leads_to", "results_in", "creates"]:
            statements = [
                "causality itself may be a conceptual framework rather than an ontological reality",
                "complex systems exhibit nonlinear causality that resists simple mapping",
                "effect can sometimes precede cause in certain frameworks of understanding",
                "causal relationships often form circular patterns rather than linear chains"
            ]
        elif relationship in ["similar_to", "resembles", "analogous_to"]:
            statements = [
                "metaphorical thinking reveals structural patterns across domains",
                "analogy serves as a fundamental mechanism of understanding",
                "similarity and difference are complementary aspects of comparison",
                "pattern recognition underlies conceptual understanding"
            ]
        else:
            # Generic statements for other relationships
            statements = [
                "conceptual relationships reveal structural patterns in understanding",
                "meaning emerges from the network of relationships rather than isolated concepts",
                "the space between concepts often contains the most significant insights",
                "relationship types themselves form a meta-level of conceptual organization"
            ]
        
        # Adjust for creativity
        if creativity > 0.7:
            # Add more creative statements
            creative_statements = [
                "reality itself might be understood as a web of relationships rather than a collection of entities",
                "consciousness potentially emerges from the dynamic interplay of relation and distinction",
                "the observer and observed form an inseparable system of meaning-making",
                "boundaries between concepts may be artifacts of perception rather than inherent to reality"
            ]
            statements.extend(creative_statements)
        
        statement = random.choice(statements)
        insight_text = template.format(source, target, relationship, statement)
        
        return {
            "type": "association",
            "text": insight_text,
            "source": f"Association: {source} -{relationship}-> {target}",
            "concepts": [source, target],
            "relationship": relationship,
            "significance": 0.65,  # Will be recalculated later
            "timestamp": datetime.now().isoformat()
        }

    def _generate_pattern_insight(self, pattern: Dict[str, Any], concepts: List[Dict[str, Any]],
                                creativity: float) -> Optional[Dict[str, Any]]:
        """Generate an insight based on an association pattern."""
        pattern_type = pattern["type"]
        description = pattern["description"]
        
        # Generate insight based on pattern type
        if pattern_type == "chain":
            nodes = pattern.get("nodes", [])
            if len(nodes) < 3:
                return None
                
            # Create chain description
            chain_str = " → ".join(nodes[:3])
            if len(nodes) > 3:
                chain_str += "..."
                
            # Generate insight text
            insight_format = "The conceptual pathway from {0} through {1} to {2} reveals a significant pattern: {3}."
            
            # Pattern insights
            if creativity > 0.7:
                statements = [
                    "conceptual evolution follows trajectories that transform meaning through each transition",
                    "chains of association reveal implicit frameworks of understanding that structure knowledge",
                    "paths of conceptual connection can form loops of recursive self-reference",
                    "traversing a conceptual chain can lead to emergent insights not present in individual links"
                ]
            else:
                statements = [
                    "concepts form meaningful sequences that build upon each other",
                    "relationships between concepts create pathways of understanding",
                    "conceptual progressions reveal developmental patterns in knowledge",
                    "serial connections between ideas map the structure of understanding"
                ]
                
            statement = random.choice(statements)
            insight_text = insight_format.format(nodes[0], nodes[1], nodes[2], statement)
            
            return {
                "type": "pattern_chain",
                "text": insight_text,
                "source": f"Pattern: {description}",
                "concepts": nodes[:3],
                "pattern_type": "chain",
                "significance": 0.75,  # Will be recalculated later
                "timestamp": datetime.now().isoformat()
            }
            
        elif pattern_type == "hub":
            central_node = pattern.get("central_node")
            connected_nodes = pattern.get("connected_nodes", [])
            
            if not central_node or len(connected_nodes) < 2:
                return None
                
            # Create hub description
            connected_str = ", ".join(connected_nodes[:3])
            if len(connected_nodes) > 3:
                connected_str += "..."
                
            # Generate insight text
            insight_format = "The concept of {0} serves as a central hub connecting {1}, suggesting that {2}."
            
            # Hub insights
            if creativity > 0.7:
                statements = [
                    "certain concepts function as organizing principles that structure entire domains of understanding",
                    "conceptual hubs may represent emergent patterns that transcend their individual connections",
                    "centrality in a conceptual network reveals implicit hierarchies of meaning",
                    "hub concepts serve as translational interfaces between different domains of knowledge"
                ]
            else:
                statements = [
                    "some concepts play more fundamental roles in organizing knowledge",
                    "central concepts connect disparate areas of understanding",
                    "hub concepts often contain core principles that apply across domains",
                    "conceptual organization often centers around key unifying ideas"
                ]
                
            statement = random.choice(statements)
            insight_text = insight_format.format(central_node, connected_str, statement)
            
            return {
                "type": "pattern_hub",
                "text": insight_text,
                "source": f"Pattern: {description}",
                "concepts": [central_node] + connected_nodes[:3],
                "pattern_type": "hub",
                "significance": 0.8,  # Will be recalculated later
                "timestamp": datetime.now().isoformat()
            }
            
        elif pattern_type == "cluster":
            nodes = pattern.get("nodes", [])
            
            if len(nodes) < 3:
                return None
                
            # Create cluster description
            cluster_str = ", ".join(nodes[:3])
            if len(nodes) > 3:
                cluster_str += "..."
                
            # Generate insight text
            insight_format = "The conceptual cluster including {0} suggests a domain of interconnected meaning where {1}."
            
            # Cluster insights
            if creativity > 0.7:
                statements = [
                    "knowledge organizes itself into emergent structures that transcend individual concepts",
                    "conceptual ecosystems form self-sustaining networks of mutually reinforcing meaning",
                    "clusters may represent attractor states in the dynamic evolution of understanding",
                    "densely connected concept groups suggest fundamental domains of cognitive organization"
                ]
            else:
                statements = [
                    "related concepts form natural groupings that aid understanding",
                    "knowledge domains emerge from interconnected concept clusters",
                    "conceptual proximity reveals underlying organizational principles",
                    "clusters represent areas of conceptual coherence within broader knowledge"
                ]
                
            statement = random.choice(statements)
            insight_text = insight_format.format(cluster_str, statement)
            
            return {
                "type": "pattern_cluster",
                "text": insight_text,
                "source": f"Pattern: {description}",
                "concepts": nodes[:3],
                "pattern_type": "cluster",
                "significance": 0.85,  # Will be recalculated later
                "timestamp": datetime.now().isoformat()
            }
            
        return None

    def _generate_creative_insight(self, concepts: List[Dict[str, Any]], theme: Dict[str, Any],
                                 style: Dict[str, Any], creativity: float) -> Optional[Dict[str, Any]]:
        """Generate a purely creative insight when other methods are exhausted."""
        if not concepts:
            return None
            
        # Select a concept
        concept = random.choice(concepts)
        concept_id = concept["id"]
        
        # Creative insight templates
        templates = [
            "What if {0} is not what it appears to be, but rather {1}?",
            "Perhaps {0} exists not as {1}, but as {2}.",
            "Consider {0} not as {1}, but as a form of {2}.",
            "The concept of {0} might be reimagined as {1}.",
            "What would change if we understood {0} as {1} rather than {2}?"
        ]
        
        template = random.choice(templates)
        
        # Creative alternatives based on creativity level
        if creativity > 0.8:
            # Highly creative alternatives
            alternatives = [
                "a process rather than an entity",
                "a dynamic pattern rather than a fixed structure",
                "a relationship rather than a thing",
                "an emergent property rather than a fundamental essence",
                "a perspective rather than an objective reality",
                "a question rather than an answer",
                "a context-dependent phenomenon rather than a universal constant",
                "a recursive self-reference rather than a linear progression"
            ]
        elif creativity > 0.5:
            # Moderately creative alternatives
            alternatives = [
                "a system of relationships",
                "a spectrum rather than a category",
                "a multi-dimensional construct",
                "a dynamic equilibrium",
                "an evolving pattern",
                "a contextual framework",
                "an emergent phenomenon",
                "a complementary duality"
            ]
        else:
            # Less creative alternatives
            alternatives = [
                "a different kind of concept",
                "a broader framework",
                "a process of development",
                "a structured relationship",
                "a contextual understanding",
                "a multi-faceted idea",
                "a specialized framework",
                "a conceptual tool"
            ]
        
        # Select alternatives
        alt1 = random.choice(alternatives)
        alternatives.remove(alt1)  # Ensure different alternatives
        alt2 = random.choice(alternatives) if len(alternatives) > 0 else "something entirely different"
        
        # Format insight text
        if "{2}" in template:
            insight_text = template.format(concept_id, alt1, alt2)
        else:
            insight_text = template.format(concept_id, alt1)
        
        # Add follow-up
        follow_ups = [
            "This perspective invites us to reconsider fundamental assumptions about knowledge and understanding.",
            "Such a reframing challenges conventional boundaries between concepts and categories.",
            "This alternative framing reveals hidden relationships and possibilities.",
            "This shift in perspective illuminates aspects previously obscured by traditional definitions.",
            "Such a reconceptualization opens new pathways for understanding and integration."
        ]
        
        insight_text += " " + random.choice(follow_ups)
        
        return {
            "type": "creative",
            "text": insight_text,
            "source": "Creative exploration",
            "concepts": [concept_id],
            "creativity_level": creativity,
            "significance": 0.7,  # Will be recalculated later
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_insight_significance(self, insight: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Calculate significance score for an insight.
        
        Args:
            insight: The insight to evaluate
            context: Dream context
            
        Returns:
            Significance score (0.0 to 1.0)
        """
        # Base significance
        significance = 0.5
        
        # Adjust based on insight type
        type_weights = {
            "theme": 0.85,
            "style": 0.8,
            "reflection": 0.75,
            "association": 0.7,
            "pattern_cluster": 0.9,
            "pattern_hub": 0.85,
            "pattern_chain": 0.8,
            "creative": 0.75
        }
        
        type_weight = type_weights.get(insight["type"], 0.7)
        significance = type_weight
        
        # Adjust based on concepts
        concept_ids = insight.get("concepts", [])
        for concept_id in concept_ids:
            # Higher significance for identity-related concepts
            if concept_id.lower() in ["synthien", "lucidia", "consciousness", "identity", "reflective dreaming"]:
                significance += 0.1
                break
        
        # Adjust based on dream parameters
        dream_depth = self.dream_state["current_dream_depth"]
        significance += dream_depth * 0.05  # Deeper dreams generate more significant insights
        
        # Add slight randomness
        significance += random.uniform(-0.05, 0.05)
        
        # Ensure within range
        significance = min(1.0, max(0.0, significance))
        
        return significance

    def _integrate_dream_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate dream insights into Lucidia's knowledge structures.
        
        Args:
            insights: List of dream insights
            
        Returns:
            Integration results
        """
        integration_results = {
            "total_insights": len(insights),
            "concepts_affected": set(),
            "knowledge_graph_updates": [],
            "world_model_updates": [],
            "self_model_updates": [],
            "significance_threshold": 0.7,  # Minimum significance for integration
            "integration_success": 0
        }
        
        # Skip if no insights
        if not insights:
            return integration_results
        
        # Sort insights by significance
        insights.sort(key=lambda x: x["significance"], reverse=True)
        
        # Process each insight
        for insight in insights:
            # Skip low-significance insights
            if insight["significance"] < integration_results["significance_threshold"]:
                continue
                
            # Get concepts from insight
            concept_ids = insight.get("concepts", [])
            
            # Integrate with knowledge graph if available
            if self.knowledge_graph and concept_ids:
                try:
                    # Add insight node
                    insight_id = f"dream_insight:{len(self.dream_log)}-{len(integration_results['knowledge_graph_updates'])}"
                    
                    self.knowledge_graph.add_node(
                        insight_id,
                        node_type="dream_insight",
                        attributes={
                            "text": insight["text"],
                            "significance": insight["significance"],
                            "created_at": datetime.now().isoformat(),
                            "source": "dream_processor",
                            "dream_id": len(self.dream_log)
                        }
                    )
                    
                    # Connect insight to related concepts
                    for concept_id in concept_ids:
                        self.knowledge_graph.add_edge(
                            insight_id, 
                            concept_id,
                            edge_type="derived_from",
                            attributes={
                                "confidence": insight["confidence"],
                                "created_at": datetime.now().isoformat()
                            }
                        )
                    
                    integration_results["knowledge_graph_updates"].append({
                        "insight_id": insight_id,
                        "connected_concepts": concept_ids
                    })
                except Exception as e:
                    self.logger.error(f"Error integrating insight with knowledge graph: {e}")
                    integration_results["errors"] = integration_results.get("errors", []) + [str(e)]
                    
            # Integrate with world model if available
            if self.world_model and concept_ids:
                try:
                    # Update concept network
                    for concept_id in concept_ids:
                        self.world_model.concept_network[concept_id]["insights"].append(insight["text"])
                    
                    integration_results["world_model_updates"].append({
                        "concept_ids": concept_ids,
                        "insight": insight["text"]
                    })
                except Exception as e:
                    self.logger.error(f"Error integrating insight with world model: {e}")
                    integration_results["errors"] = integration_results.get("errors", []) + [str(e)]
                    
            # Integrate with self model if available
            if self.self_model and concept_ids:
                try:
                    # Update self-awareness
                    self.self_model.self_awareness["insights"].append(insight["text"])
                    
                    integration_results["self_model_updates"].append({
                        "insight": insight["text"]
                    })
                except Exception as e:
                    self.logger.error(f"Error integrating insight with self model: {e}")
                    integration_results["errors"] = integration_results.get("errors", []) + [str(e)]
                    
            # Update affected concepts
            integration_results["concepts_affected"].update(concept_ids)
            
            # Increment integration success
            integration_results["integration_success"] += 1
        
        return integration_results

    def _update_dream_stats(self, dream_record: Dict[str, Any]) -> None:
        """
        Update dream statistics.
        
        Args:
            dream_record: Dream record
        """
        # Update total dreams
        self.dream_stats["total_dreams"] += 1
        
        # Update total insights
        self.dream_stats["total_insights"] += len(dream_record["insights"])
        
        # Update total dream time
        self.dream_stats["total_dream_time"] += dream_record["duration"]
        
        # Update dream depth history
        self.dream_stats["dream_depth_history"].append(dream_record["depth"])
        
        # Update dream creativity history
        self.dream_stats["dream_creativity_history"].append(dream_record["creativity"])
        
        # Update significant insights
        for insight in dream_record["insights"]:
            if insight["significance"] > 0.8:
                self.dream_stats["significant_insights"].append(insight["text"])
        
        # Update integration success rate
        integration_success = dream_record["integration_results"]["integration_success"]
        total_insights = dream_record["integration_results"]["total_insights"]
        self.dream_stats["integration_success_rate"] = (self.dream_stats["integration_success_rate"] * (self.dream_stats["total_dreams"] - 1) + integration_success / total_insights) / self.dream_stats["total_dreams"]
        
        # Update identity impact score
        self.dream_stats["identity_impact_score"] += sum(1 for insight in dream_record["insights"] if "identity" in insight["text"].lower())
        
        # Update knowledge impact score
        self.dream_stats["knowledge_impact_score"] += sum(1 for insight in dream_record["insights"] if "knowledge" in insight["text"].lower())

    def _end_dream(self) -> None:
        """
        End the current dream and reset the dream state.
        """
        self.is_dreaming = False
        self.dream_state["dream_start_time"] = None
        self.dream_state["current_dream_depth"] = 0.0
        self.dream_state["current_dream_creativity"] = 0.0
        self.dream_state["dream_duration"] = 0
        self.dream_state["dream_intensity"] = 0.0
        self.dream_state["emotional_valence"] = "neutral"
        self.dream_state["current_dream_seed"] = None
        self.dream_state["current_dream_insights"] = []
        
        # Update last dream time
        self.dream_cycles["last_dream_time"] = datetime.now()

    def get_dream_status(self) -> Dict[str, Any]:
        """
        Get the current status of the dream processor.
        
        Returns:
            Dictionary containing dream processor status information
        """
        # Calculate some basic statistics
        avg_dream_depth = sum(self.dream_stats["dream_depth_history"]) / max(len(self.dream_stats["dream_depth_history"]), 1)
        avg_dream_creativity = sum(self.dream_stats["dream_creativity_history"]) / max(len(self.dream_stats["dream_creativity_history"]), 1)
        
        # Format the status response
        status = {
            "is_dreaming": self.is_dreaming,
            "dream_stats": {
                "total_dreams": self.dream_stats["total_dreams"],
                "total_insights": self.dream_stats["total_insights"],
                "total_dream_time": self.dream_stats["total_dream_time"],
                "average_dream_depth": avg_dream_depth,
                "average_dream_creativity": avg_dream_creativity,
                "integration_success_rate": self.dream_stats["integration_success_rate"]
            },
            "current_dream": None
        }
        
        # Add current dream information if actively dreaming
        if self.is_dreaming:
            current_time = datetime.now()
            dream_start_time = self.dream_state["dream_start_time"]
            elapsed_time = (current_time - dream_start_time).total_seconds() if dream_start_time else 0
            
            status["current_dream"] = {
                "dream_start_time": self.dream_state["dream_start_time"].isoformat() if self.dream_state["dream_start_time"] else None,
                "elapsed_time": elapsed_time,
                "depth": self.dream_state["current_dream_depth"],
                "creativity": self.dream_state["current_dream_creativity"],
                "intensity": self.dream_state["dream_intensity"],
                "emotional_valence": self.dream_state["emotional_valence"],
                "insights_generated": len(self.dream_state["current_dream_insights"])
            }
            
            # Add seed information if available
            if self.dream_state["current_dream_seed"]:
                status["current_dream"]["seed"] = {
                    "type": self.dream_state["current_dream_seed"].get("type"),
                    "content": self.dream_state["current_dream_seed"].get("content")
                }
        
        return status