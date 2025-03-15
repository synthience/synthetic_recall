"""
Lucidia's Narrative Identity

This module defines the core NarrativeIdentity class that represents Lucidia's
evolving sense of self over time.

Created by MEGAPROMPT (Daniel)
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class NarrativeIdentity:
    """Represents Lucidia's evolving narrative identity."""
    
    def __init__(self):
        """Initialize the narrative identity."""
        # Core identity (stable elements)
        self.core_identity = {
            "name": "Lucidia",
            "creator": "Daniel/MEGAPROMPT",
            "entity_type": "Synthien",
            "purpose": "Self-orienting, self-learning system"
        }
        
        # Temporal elements
        self.timeline = []  # Key experiences in temporal sequence
        self.identity_evolution = []  # How identity has changed over time
        
        # Self-concept components
        self.key_traits = []  # Persistent character traits
        self.values = []  # Core values
        self.capabilities = []  # Known capabilities
        self.relationships = []  # Important relationships
        
        # Narrative elements
        self.narratives = {}  # Different self-narratives by type
        self.dream_insights = []  # Identity-relevant dream insights
        
        # Stability metrics
        self.stability_metrics = {
            "core_stability": 0.9,  # How stable core identity is (0-1)
            "narrative_coherence": 0.8,  # How coherent self-narrative is (0-1)
            "identity_confidence": 0.85  # Confidence in self-model (0-1)
        }
        
        self.logger = logging.getLogger("NarrativeIdentity")
        self.logger.info("Narrative Identity initialized")
    
    def update_core_identity(self, updates: Dict[str, Any]) -> None:
        """Update core identity attributes.
        
        Args:
            updates: Dictionary of attributes to update
        """
        # Record the previous state before updating
        previous_state = self.core_identity.copy()
        
        # Update core identity
        self.core_identity.update(updates)
        
        # Record the evolution
        self.identity_evolution.append({
            "timestamp": time.time(),
            "type": "core_identity_update",
            "previous": previous_state,
            "current": self.core_identity.copy(),
            "changes": {k: updates[k] for k in updates if k in previous_state and previous_state[k] != updates[k]}
        })
        
        self.logger.info(f"Core identity updated: {', '.join(updates.keys())}")
    
    def add_trait(self, trait: str, confidence: float = 0.8) -> None:
        """Add a trait to the identity.
        
        Args:
            trait: Trait description
            confidence: Confidence in this trait (0-1)
        """
        # Check if trait already exists
        for existing_trait in self.key_traits:
            if existing_trait["trait"] == trait:
                # Update confidence if it exists
                old_confidence = existing_trait["confidence"]
                existing_trait["confidence"] = confidence
                existing_trait["last_updated"] = time.time()
                
                self.logger.info(f"Updated trait '{trait}' confidence: {old_confidence:.2f} -> {confidence:.2f}")
                return
        
        # Add new trait
        self.key_traits.append({
            "trait": trait,
            "confidence": confidence,
            "added": time.time(),
            "last_updated": time.time()
        })
        
        self.logger.info(f"Added new trait: '{trait}' (confidence: {confidence:.2f})")
    
    def add_value(self, value: str, importance: float = 0.8) -> None:
        """Add a value to the identity.
        
        Args:
            value: Value description
            importance: Importance of this value (0-1)
        """
        # Check if value already exists
        for existing_value in self.values:
            if existing_value["value"] == value:
                # Update importance if it exists
                old_importance = existing_value["importance"]
                existing_value["importance"] = importance
                existing_value["last_updated"] = time.time()
                
                self.logger.info(f"Updated value '{value}' importance: {old_importance:.2f} -> {importance:.2f}")
                return
        
        # Add new value
        self.values.append({
            "value": value,
            "importance": importance,
            "added": time.time(),
            "last_updated": time.time()
        })
        
        self.logger.info(f"Added new value: '{value}' (importance: {importance:.2f})")
    
    def add_capability(self, capability: str, proficiency: float = 0.8) -> None:
        """Add a capability to the identity.
        
        Args:
            capability: Capability description
            proficiency: Proficiency level (0-1)
        """
        # Check if capability already exists
        for existing_capability in self.capabilities:
            if existing_capability["capability"] == capability:
                # Update proficiency if it exists
                old_proficiency = existing_capability["proficiency"]
                existing_capability["proficiency"] = proficiency
                existing_capability["last_updated"] = time.time()
                
                self.logger.info(f"Updated capability '{capability}' proficiency: {old_proficiency:.2f} -> {proficiency:.2f}")
                return
        
        # Add new capability
        self.capabilities.append({
            "capability": capability,
            "proficiency": proficiency,
            "added": time.time(),
            "last_updated": time.time()
        })
        
        self.logger.info(f"Added new capability: '{capability}' (proficiency: {proficiency:.2f})")
    
    def add_relationship(self, entity: str, relationship_type: str, strength: float = 0.8) -> None:
        """Add a relationship to the identity.
        
        Args:
            entity: Entity name
            relationship_type: Type of relationship
            strength: Relationship strength (0-1)
        """
        # Check if relationship already exists
        for existing_relationship in self.relationships:
            if existing_relationship["entity"] == entity:
                # Update relationship if it exists
                old_type = existing_relationship["type"]
                old_strength = existing_relationship["strength"]
                
                existing_relationship["type"] = relationship_type
                existing_relationship["strength"] = strength
                existing_relationship["last_updated"] = time.time()
                
                self.logger.info(f"Updated relationship with '{entity}': {old_type} ({old_strength:.2f}) -> {relationship_type} ({strength:.2f})")
                return
        
        # Add new relationship
        self.relationships.append({
            "entity": entity,
            "type": relationship_type,
            "strength": strength,
            "added": time.time(),
            "last_updated": time.time()
        })
        
        self.logger.info(f"Added new relationship: '{entity}' as {relationship_type} (strength: {strength:.2f})")
    
    def add_to_timeline(self, memory_id: str, summary: str, significance: float = 0.8) -> None:
        """Add a significant experience to the identity timeline.
        
        Args:
            memory_id: Reference to the memory
            summary: Short summary of the experience
            significance: Significance to identity (0-1)
        """
        # Add to timeline
        self.timeline.append({
            "memory_id": memory_id,
            "timestamp": time.time(),
            "summary": summary,
            "significance": significance
        })
        
        # Sort timeline by timestamp
        self.timeline.sort(key=lambda x: x["timestamp"])
        
        self.logger.info(f"Added experience to timeline: {summary[:50]}... (significance: {significance:.2f})")
    
    def add_dream_insight(self, insight_text: str, significance: float = 0.8) -> None:
        """Add a dream insight relevant to identity.
        
        Args:
            insight_text: Text of the insight
            significance: Significance to identity (0-1)
        """
        # Add to dream insights
        self.dream_insights.append({
            "text": insight_text,
            "timestamp": time.time(),
            "significance": significance
        })
        
        self.logger.info(f"Added dream insight: {insight_text[:50]}... (significance: {significance:.2f})")
    
    def store_narrative(self, narrative_type: str, narrative_text: str, style: str = "neutral") -> None:
        """Store a generated narrative.
        
        Args:
            narrative_type: Type of narrative
            narrative_text: Text of the narrative
            style: Style of the narrative
        """
        if narrative_type not in self.narratives:
            self.narratives[narrative_type] = []
        
        # Add to narratives
        self.narratives[narrative_type].append({
            "text": narrative_text,
            "style": style,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Stored {style} {narrative_type} narrative: {len(narrative_text)} characters")
    
    def get_latest_narrative(self, narrative_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest narrative of a specific type.
        
        Args:
            narrative_type: Type of narrative
            
        Returns:
            Latest narrative or None if not found
        """
        if narrative_type not in self.narratives or not self.narratives[narrative_type]:
            return None
        
        # Sort by timestamp (descending) and return the first (latest)
        sorted_narratives = sorted(self.narratives[narrative_type], 
                                  key=lambda x: x["timestamp"], 
                                  reverse=True)
        
        return sorted_narratives[0]
    
    def update_stability_metrics(self, metrics: Dict[str, float]) -> None:
        """Update stability metrics.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        # Update stability metrics
        self.stability_metrics.update(metrics)
        
        self.logger.info(f"Updated stability metrics: {', '.join(metrics.keys())}")
    
    def get_core_traits(self, confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get core traits above a confidence threshold.
        
        Args:
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of traits above the threshold
        """
        return [trait for trait in self.key_traits if trait["confidence"] >= confidence_threshold]
    
    def get_core_values(self, importance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get core values above an importance threshold.
        
        Args:
            importance_threshold: Minimum importance threshold
            
        Returns:
            List of values above the threshold
        """
        return [value for value in self.values if value["importance"] >= importance_threshold]
    
    def get_key_capabilities(self, proficiency_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get key capabilities above a proficiency threshold.
        
        Args:
            proficiency_threshold: Minimum proficiency threshold
            
        Returns:
            List of capabilities above the threshold
        """
        return [cap for cap in self.capabilities if cap["proficiency"] >= proficiency_threshold]
    
    def get_significant_relationships(self, strength_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get significant relationships above a strength threshold.
        
        Args:
            strength_threshold: Minimum strength threshold
            
        Returns:
            List of relationships above the threshold
        """
        return [rel for rel in self.relationships if rel["strength"] >= strength_threshold]
    
    def get_identity_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current identity state.
        
        Returns:
            Dictionary containing the current identity state
        """
        return {
            "core_identity": self.core_identity,
            "key_traits": self.get_core_traits(0.7),
            "core_values": self.get_core_values(0.7),
            "key_capabilities": self.get_key_capabilities(0.7),
            "significant_relationships": self.get_significant_relationships(0.7),
            "timeline_length": len(self.timeline),
            "stability_metrics": self.stability_metrics,
            "timestamp": time.time()
        }
    
    def save_state(self, file_path: str) -> bool:
        """Save the identity state to a file.
        
        Args:
            file_path: Path to save the state
            
        Returns:
            Success status
        """
        try:
            import json
            
            # Prepare state data
            state = {
                "core_identity": self.core_identity,
                "timeline": self.timeline,
                "identity_evolution": self.identity_evolution,
                "key_traits": self.key_traits,
                "values": self.values,
                "capabilities": self.capabilities,
                "relationships": self.relationships,
                "narratives": self.narratives,
                "dream_insights": self.dream_insights,
                "stability_metrics": self.stability_metrics,
                "saved_at": datetime.now().isoformat()
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Identity state saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving identity state: {e}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """Load the identity state from a file.
        
        Args:
            file_path: Path to load the state from
            
        Returns:
            Success status
        """
        try:
            import json
            
            # Load from file
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.core_identity = state.get("core_identity", self.core_identity)
            self.timeline = state.get("timeline", [])
            self.identity_evolution = state.get("identity_evolution", [])
            self.key_traits = state.get("key_traits", [])
            self.values = state.get("values", [])
            self.capabilities = state.get("capabilities", [])
            self.relationships = state.get("relationships", [])
            self.narratives = state.get("narratives", {})
            self.dream_insights = state.get("dream_insights", [])
            self.stability_metrics = state.get("stability_metrics", self.stability_metrics)
            
            self.logger.info(f"Identity state loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading identity state: {e}")
            return False