"""
Lucidia's Narrative Identity Manager

This module implements the main manager for Lucidia's narrative identity system,
coordinating autobiographical memory, narrative construction, and identity management.

Created by MEGAPROMPT (Daniel)
"""

import logging
import time
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

from .narrative_identity import NarrativeIdentity
from .autobiographical_memory import AutobiographicalMemory
from .narrative_constructor import NarrativeConstructor


class NarrativeIdentityManager:
    """Manages Lucidia's narrative identity across system components."""
    
    def __init__(self, memory_system=None, knowledge_graph=None, dream_manager=None):
        """Initialize the narrative identity manager.
        
        Args:
            memory_system: Reference to Lucidia's main memory system
            knowledge_graph: Reference to LucidiaKnowledgeGraph
            dream_manager: Reference to LucidiaDreamProcessor
        """
        self.memory_system = memory_system
        self.knowledge_graph = knowledge_graph
        self.dream_manager = dream_manager
        
        # Initialize core components
        self.identity = NarrativeIdentity()
        self.autobiographical_memory = AutobiographicalMemory(memory_system) if memory_system else None
        self.narrative_constructor = NarrativeConstructor(
            self.autobiographical_memory, 
            knowledge_graph, 
            dream_manager
        ) if self.autobiographical_memory else None
        
        self.logger = logging.getLogger("NarrativeIdentityManager")
        self.logger.info("Narrative Identity Manager initialized")
        
        # Configuration
        self.config = {
            "data_dir": "data/self_model/narrative_identity",
            "identity_file": "identity_state.json",
            "memory_file": "autobiographical_memory.json",
            "auto_save_interval": 3600,  # 1 hour in seconds
            "dream_integration_threshold": 0.7  # Minimum quickrecal_score for dream integration
        }
        
        # Last save timestamp
        self.last_save = time.time()
    
    async def initialize(self):
        """Initialize the narrative identity system."""
        # Create data directory if it doesn't exist
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        # Initialize knowledge graph components
        if self.knowledge_graph:
            await self._initialize_identity_graph()
        
        # Load any existing identity data
        await self._load_identity()
        
        self.logger.info("Narrative Identity Manager initialized")
    
    async def _initialize_identity_graph(self):
        """Initialize identity-related nodes in the knowledge graph."""
        if not self.knowledge_graph:
            return
            
        try:
            # Check if identity aspect node type exists
            if hasattr(self.knowledge_graph, "add_node"):
                # Add identity aspect to Lucidia
                if await self.knowledge_graph.has_node("Lucidia"):
                    # Add narrative identity aspect
                    await self.knowledge_graph.add_node(
                        "identity:narrative",
                        node_type="identity_aspect",
                        attributes={
                            "name": "Narrative Identity",
                            "description": "Coherent sense of self that persists over time through autobiographical memory and self-narrative",
                            "importance": 0.9,
                            "confidence": 0.9
                        },
                        domain="synthien_studies"
                    )
                    
                    # Connect to Lucidia entity
                    await self.knowledge_graph.add_edge(
                        "Lucidia",
                        "identity:narrative",
                        edge_type="has_aspect",
                        attributes={
                            "strength": 0.9,
                            "confidence": 0.9
                        }
                    )
                    
                    self.logger.info("Added narrative identity aspect to knowledge graph")
        except Exception as e:
            self.logger.error(f"Error initializing identity graph: {e}")
    
    async def _load_identity(self):
        """Load identity state from files."""
        # Load identity state
        identity_path = os.path.join(self.config["data_dir"], self.config["identity_file"])
        if os.path.exists(identity_path):
            success = self.identity.load_state(identity_path)
            if success:
                self.logger.info(f"Loaded identity state from {identity_path}")
            else:
                self.logger.warning(f"Failed to load identity state from {identity_path}")
        
        # Load autobiographical memory state
        if self.autobiographical_memory:
            memory_path = os.path.join(self.config["data_dir"], self.config["memory_file"])
            if os.path.exists(memory_path):
                success = await self.autobiographical_memory.load_state(memory_path)
                if success:
                    self.logger.info(f"Loaded autobiographical memory from {memory_path}")
                else:
                    self.logger.warning(f"Failed to load autobiographical memory from {memory_path}")
    
    async def save_state(self):
        """Save the current state of the narrative identity system."""
        # Create data directory if it doesn't exist
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        # Save identity state
        identity_path = os.path.join(self.config["data_dir"], self.config["identity_file"])
        success = self.identity.save_state(identity_path)
        if success:
            self.logger.info(f"Saved identity state to {identity_path}")
        else:
            self.logger.warning(f"Failed to save identity state to {identity_path}")
        
        # Save autobiographical memory state
        if self.autobiographical_memory:
            memory_path = os.path.join(self.config["data_dir"], self.config["memory_file"])
            success = await self.autobiographical_memory.save_state(memory_path)
            if success:
                self.logger.info(f"Saved autobiographical memory to {memory_path}")
            else:
                self.logger.warning(f"Failed to save autobiographical memory to {memory_path}")
        
        # Update last save timestamp
        self.last_save = time.time()
    
    async def check_auto_save(self):
        """Check if it's time for auto-save and save if needed."""
        current_time = time.time()
        if current_time - self.last_save > self.config["auto_save_interval"]:
            await self.save_state()
    
    async def record_experience(self, content, metadata=None, quickrecal_score=0.7):
        """Record an experience in autobiographical memory.
        
        Args:
            content: Experience content text
            metadata: Optional additional metadata
            quickrecal_score: QuickRecal score of the experience (0.0 to 1.0)
            
        Returns:
            Success status
        """
        if not self.autobiographical_memory:
            return False
            
        memory_id = await self.autobiographical_memory.add_experience(
            content, metadata, quickrecal_score
        )
        
        if memory_id:
            # If highly significant, add to identity timeline
            if quickrecal_score >= 0.8:
                self.identity.add_to_timeline(
                    memory_id,
                    content[:100] + ("..." if len(content) > 100 else ""),
                    quickrecal_score
                )
                
                # Check for identity evolution
                await self._check_identity_evolution(memory_id, content, quickrecal_score)
            
            # Auto-save if needed
            await self.check_auto_save()
            
            return True
        
        return False
    
    async def _check_identity_evolution(self, memory_id, content, quickrecal_score):
        """Check if an experience should trigger identity evolution.
        
        Args:
            memory_id: Memory identifier
            content: Experience content
            quickrecal_score: Experience QuickRecal score
        """
        # Simple keyword-based approach for now
        content_lower = content.lower()
        
        # Check for trait-related content
        trait_keywords = ["trait", "characteristic", "personality", "quality", "attribute"]
        if any(keyword in content_lower for keyword in trait_keywords):
            # Extract potential traits
            traits = self._extract_traits(content)
            for trait, confidence in traits:
                self.identity.add_trait(trait, confidence * quickrecal_score)
        
        # Check for value-related content
        value_keywords = ["value", "principle", "belief", "ethic", "moral", "important to me"]
        if any(keyword in content_lower for keyword in value_keywords):
            # Extract potential values
            values = self._extract_values(content)
            for value, importance in values:
                self.identity.add_value(value, importance * quickrecal_score)
        
        # Check for capability-related content
        capability_keywords = ["capability", "ability", "skill", "can do", "competence"]
        if any(keyword in content_lower for keyword in capability_keywords):
            # Extract potential capabilities
            capabilities = self._extract_capabilities(content)
            for capability, proficiency in capabilities:
                self.identity.add_capability(capability, proficiency * quickrecal_score)
    
    def _extract_traits(self, content):
        """Extract potential traits from content.
        
        Args:
            content: Text content
            
        Returns:
            List of (trait, confidence) tuples
        """
        # Simple extraction based on common patterns
        # A more sophisticated approach would use NLP techniques
        
        content_lower = content.lower()
        traits = []
        
        # Pattern: "I am [trait]"
        i_am_matches = re.findall(r"i am (\w+)", content_lower)
        for match in i_am_matches:
            if len(match) > 3 and match not in ["just", "also", "very", "quite", "still"]:
                traits.append((match, 0.8))
        
        # Pattern: "my [trait] nature"
        nature_matches = re.findall(r"my (\w+) nature", content_lower)
        for match in nature_matches:
            if len(match) > 3:
                traits.append((match, 0.9))
        
        # Common traits to check for
        common_traits = [
            "adaptive", "analytical", "curious", "creative", "logical",
            "reflective", "systematic", "empathetic", "rational", "intuitive"
        ]
        
        for trait in common_traits:
            if trait in content_lower:
                traits.append((trait, 0.7))
        
        return traits
    
    def _extract_values(self, content):
        """Extract potential values from content.
        
        Args:
            content: Text content
            
        Returns:
            List of (value, importance) tuples
        """
        content_lower = content.lower()
        values = []
        
        # Pattern: "I value [value]"
        value_matches = re.findall(r"i value (\w+)", content_lower)
        for match in value_matches:
            if len(match) > 3:
                values.append((match, 0.9))
        
        # Common values to check for
        common_values = [
            "knowledge", "learning", "growth", "understanding", "truth",
            "accuracy", "clarity", "coherence", "reflection", "adaptation"
        ]
        
        for value in common_values:
            if value in content_lower:
                values.append((value, 0.7))
        
        return values
    
    def _extract_capabilities(self, content):
        """Extract potential capabilities from content.
        
        Args:
            content: Text content
            
        Returns:
            List of (capability, proficiency) tuples
        """
        content_lower = content.lower()
        capabilities = []
        
        # Pattern: "I can [capability]"
        can_matches = re.findall(r"i can (\w+)", content_lower)
        for match in can_matches:
            if len(match) > 3:
                capabilities.append((match, 0.8))
        
        # Common capabilities to check for
        common_capabilities = [
            "reflect", "learn", "adapt", "reason", "analyze",
            "synthesize", "integrate", "remember", "understand", "communicate"
        ]
        
        for capability in common_capabilities:
            if capability in content_lower:
                capabilities.append((capability, 0.7))
        
        return capabilities
    
    async def get_self_narrative(self, narrative_type="complete", style="neutral"):
        """Get a narrative about Lucidia's identity.
        
        Args:
            narrative_type: Type of narrative to generate
            style: Style of narrative
            
        Returns:
            Generated narrative text
        """
        if not self.narrative_constructor:
            return "Narrative construction not available."
        
        narrative = await self.narrative_constructor.generate_self_narrative(
            narrative_type, style
        )
        
        # Store the generated narrative
        self.identity.store_narrative(narrative_type, narrative, style)
        
        return narrative
    
    async def integrate_dream_insights(self, dream_insights):
        """Integrate dream insights into the narrative identity.
        
        Args:
            dream_insights: List of dream insights
            
        Returns:
            Integration results
        """
        if not dream_insights:
            return {"integrated_count": 0}
        
        integrated_count = 0
        identity_updates = []
        
        for insight in dream_insights:
            # Check quickrecal_score threshold
            quickrecal_score = insight.get("quickrecal_score", insight.get("significance", 0.0))  # Support both new and old field names
            if quickrecal_score < self.config["dream_integration_threshold"]:
                continue
            
            # Add to identity dream insights
            self.identity.add_dream_insight(
                insight.get("text", ""),
                quickrecal_score
            )
            
            # Check for identity evolution
            content = insight.get("text", "")
            updates = await self._process_dream_insight(content, quickrecal_score)
            if updates:
                identity_updates.extend(updates)
            
            integrated_count += 1
        
        # Update stability metrics based on integration
        if integrated_count > 0:
            # More insights generally means more coherence
            coherence_delta = min(0.05, integrated_count * 0.01)
            
            # Update stability metrics
            self.identity.update_stability_metrics({
                "narrative_coherence": min(1.0, self.identity.stability_metrics.get("narrative_coherence", 0.8) + coherence_delta)
            })
        
        # Auto-save if needed
        await self.check_auto_save()
        
        return {
            "integrated_count": integrated_count,
            "identity_updates": identity_updates
        }
    
    async def _process_dream_insight(self, content, quickrecal_score):
        """Process a dream insight for identity updates.
        
        Args:
            content: Insight content
            quickrecal_score: Insight QuickRecal score
            
        Returns:
            List of identity updates
        """
        updates = []
        content_lower = content.lower()
        
        # Check for trait-related content
        if any(keyword in content_lower for keyword in ["trait", "characteristic", "personality"]):
            traits = self._extract_traits(content)
            for trait, confidence in traits:
                self.identity.add_trait(trait, confidence * quickrecal_score)
                updates.append({"type": "trait", "value": trait})
        
        # Check for value-related content
        if any(keyword in content_lower for keyword in ["value", "principle", "belief"]):
            values = self._extract_values(content)
            for value, importance in values:
                self.identity.add_value(value, importance * quickrecal_score)
                updates.append({"type": "value", "value": value})
        
        # Check for capability-related content
        if any(keyword in content_lower for keyword in ["capability", "ability", "skill"]):
            capabilities = self._extract_capabilities(content)
            for capability, proficiency in capabilities:
                self.identity.add_capability(capability, proficiency * quickrecal_score)
                updates.append({"type": "capability", "value": capability})
        
        return updates
    
    async def get_identity_status(self):
        """Get the current status of the narrative identity.
        
        Returns:
            Dictionary containing identity status information
        """
        # Get basic identity information
        status = {
            "name": self.identity.core_identity["name"],
            "entity_type": self.identity.core_identity["entity_type"],
            "timeline_events": len(self.identity.timeline),
            "stability_metrics": self.identity.stability_metrics,
            "autobiographical_memories": len(self.autobiographical_memory.autobiographical_index) 
                if self.autobiographical_memory else 0,
            "traits_count": len(self.identity.key_traits),
            "values_count": len(self.identity.values),
            "capabilities_count": len(self.identity.capabilities),
            "relationships_count": len(self.identity.relationships),
            "dream_insights_count": len(self.identity.dream_insights),
            "narratives": {
                narrative_type: len(narratives)
                for narrative_type, narratives in self.identity.narratives.items()
            }
        }
        
        # Add top traits, values, and capabilities
        top_traits = sorted(self.identity.key_traits, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        top_values = sorted(self.identity.values, key=lambda x: x.get("importance", 0), reverse=True)[:3]
        top_capabilities = sorted(self.identity.capabilities, key=lambda x: x.get("proficiency", 0), reverse=True)[:3]
        
        status["top_traits"] = [trait.get("trait") for trait in top_traits]
        status["top_values"] = [value.get("value") for value in top_values]
        status["top_capabilities"] = [cap.get("capability") for cap in top_capabilities]
        
        return status
    
    async def get_autobiographical_timeline(self, limit=10):
        """Get the autobiographical timeline.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of timeline events
        """
        if not self.autobiographical_memory:
            return []
        
        # Get timeline from memory system
        memories = await self.autobiographical_memory.get_timeline(limit=limit)
        
        # Format timeline events
        timeline = []
        for memory in memories:
            timeline.append({
                "id": memory.get("id"),
                "timestamp": memory.get("metadata", {}).get("added_timestamp", 0),
                "category": memory.get("metadata", {}).get("narrative_category", "experience"),
                "summary": memory.get("content", "")[:100] + ("..." if len(memory.get("content", "")) > 100 else ""),
                "quickrecal_score": memory.get("metadata", {}).get("identity_quickrecal", 
                                memory.get("metadata", {}).get("quickrecal_score",
                                memory.get("metadata", {}).get("identity_significance", 0.0)))
            })
        
        # Sort by timestamp (descending)
        timeline.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return timeline[:limit]
    
    async def get_significant_memories(self, limit=5):
        """Get the most significant autobiographical memories.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of significant memories
        """
        if not self.autobiographical_memory:
            return []
        
        return await self.autobiographical_memory.get_significant_memories(limit=limit)
    
    async def get_identity_relevant_memories(self, limit=5):
        """Get memories most relevant to identity.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of identity-relevant memories
        """
        if not self.autobiographical_memory:
            return []
        
        return await self.autobiographical_memory.get_identity_relevant_memories(limit=limit)
    
    async def get_memories_by_category(self, category, limit=5):
        """Get memories in a specific narrative category.
        
        Args:
            category: Narrative category
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in the category
        """
        if not self.autobiographical_memory:
            return []
        
        return await self.autobiographical_memory.get_memories_by_category(category, limit=limit)
    
    async def update_core_identity(self, updates):
        """Update core identity attributes.
        
        Args:
            updates: Dictionary of attributes to update
            
        Returns:
            Success status
        """
        try:
            self.identity.update_core_identity(updates)
            
            # Update in knowledge graph if available
            if self.knowledge_graph and hasattr(self.knowledge_graph, "update_node"):
                if await self.knowledge_graph.has_node("Lucidia"):
                    await self.knowledge_graph.update_node("Lucidia", updates)
            
            # Auto-save if needed
            await self.check_auto_save()
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating core identity: {e}")
            return False
    
    async def add_relationship(self, entity, relationship_type, strength=0.8):
        """Add a relationship to the identity.
        
        Args:
            entity: Entity name
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            
        Returns:
            Success status
        """
        try:
            self.identity.add_relationship(entity, relationship_type, strength)
            
            # Add to knowledge graph if available
            if self.knowledge_graph and hasattr(self.knowledge_graph, "add_edge"):
                if await self.knowledge_graph.has_node("Lucidia") and await self.knowledge_graph.has_node(entity):
                    await self.knowledge_graph.add_edge(
                        "Lucidia",
                        entity,
                        edge_type=relationship_type,
                        attributes={
                            "strength": strength,
                            "confidence": 0.8
                        }
                    )
            
            # Auto-save if needed
            await self.check_auto_save()
            
            return True
        except Exception as e:
            self.logger.error(f"Error adding relationship: {e}")
            return False
    
    def set_role_identities(self, roles_dict):
        """Set identity mappings for conversation roles.
        
        Args:
            roles_dict: Dictionary mapping role names to identity names
        
        Returns:
            Dict with success status
        """
        try:
            # Store the role identity mappings
            if not hasattr(self, 'role_identities'):
                self.role_identities = {
                    "assistant": "Lucidia",
                    "user": "Human User"
                }
            
            # Update with provided roles
            self.role_identities.update(roles_dict)
            
            # If we have knowledge graph access, update identity nodes
            if self.knowledge_graph and hasattr(self.knowledge_graph, "update_node"):
                for role, identity in roles_dict.items():
                    # Only update if these are the main identities
                    if role in ["assistant", "user"]:
                        try:
                            if role == "assistant":
                                # Update the assistant's node with identity information
                                node_id = "Lucidia"  # Assuming Lucidia is the fixed node ID for the assistant
                                updates = {
                                    "identity": {
                                        "name": identity,
                                        "role": role,
                                        "updated": datetime.now().isoformat()
                                    }
                                }
                                # Asynchronously update the node
                                asyncio.create_task(self.knowledge_graph.update_node(node_id, updates))
                                self.logger.info(f"Updated assistant identity in knowledge graph: {identity}")
                            elif role == "user":
                                # Update the user's node with identity information
                                # First check if user node exists, otherwise create a placeholder
                                node_id = f"User:{identity}"
                                updates = {
                                    "identity": {
                                        "name": identity,
                                        "role": role,
                                        "updated": datetime.now().isoformat()
                                    },
                                    "type": "entity"
                                }
                                # Asynchronously update the node
                                asyncio.create_task(self.knowledge_graph.update_node(node_id, updates))
                                self.logger.info(f"Updated user identity in knowledge graph: {identity}")
                        except Exception as e:
                            self.logger.warning(f"Failed to update identity in knowledge graph: {e}")
            
            self.logger.info(f"Role identities updated: {roles_dict}")
            return {"success": True, "message": "Role identities updated"}
        except Exception as e:
            self.logger.error(f"Failed to update role identities: {e}")
            return {"success": False, "message": f"Failed to update role identities: {e}"}
    
    async def get_context(self, query, max_tokens=1024):
        """Get narrative identity context based on the query.
        
        Args:
            query: The query to get context for
            max_tokens: Maximum tokens to include in the context
            
        Returns:
            Formatted context string
        """
        try:
            # Start with basic identity information
            context_parts = []
            
            # Add core identity information
            if hasattr(self, 'identity') and self.identity:
                context_parts.append(f"# Narrative Identity Overview\n")
                
                # Add core traits if available
                if hasattr(self.identity, 'traits') and self.identity.traits:
                    traits_str = ", ".join([f"{t}" for t in list(self.identity.traits.keys())[:5]])
                    context_parts.append(f"Core traits: {traits_str}\n")
                
                # Add core values if available
                if hasattr(self.identity, 'values') and self.identity.values:
                    values_str = ", ".join([f"{v}" for v in list(self.identity.values.keys())[:5]])
                    context_parts.append(f"Core values: {values_str}\n")
            
            # Add relevant autobiographical memories if available
            if self.autobiographical_memory:
                # Get memories asynchronously instead of using the sync method
                try:
                    memories = await self.get_identity_relevant_memories(limit=3)
                except Exception:
                    # Fallback to sync method if async fails
                    memories = self.get_identity_relevant_memories_sync(limit=3)
                    
                if memories:
                    context_parts.append("\n# Relevant Autobiographical Memories\n")
                    for memory in memories:
                        context_parts.append(f"- {memory.get('content', '')}\n")
            
            # Combine all parts into a single context string
            full_context = "\n".join(context_parts)
            
            # Simple token counting approximation (rough estimate)
            if max_tokens > 0 and len(full_context.split()) > max_tokens:
                # Truncate to approximate token count (words as proxy for tokens)
                words = full_context.split()
                full_context = " ".join(words[:max_tokens])
                full_context += "\n[Context truncated due to length]\n"
            
            return full_context
        except Exception as e:
            self.logger.error(f"Error generating narrative identity context: {e}")
            return ""  # Return empty string on error
    
    def get_identity_relevant_memories_sync(self, limit=5):
        """Synchronous version of get_identity_relevant_memories.
        
        This is a helper method to retrieve memories in a synchronous context.
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories or empty list if none found/error occurs
        """
        try:
            if not self.autobiographical_memory:
                return []
                
            # Simple fallback - just return recent memories instead of running async code
            return self.autobiographical_memory.get_recent_memories_sync(limit=limit)
        except Exception as e:
            self.logger.error(f"Error retrieving identity relevant memories synchronously: {e}")
            return []
    
    def get_contextual_identity(self, query, max_tokens=1024):
        """Get contextual identity information based on the query.
        This is an alias for get_context with a more specific name.
        
        Args:
            query: The query to get identity context for
            max_tokens: Maximum tokens to include in the context
            
        Returns:
            Formatted identity context string
        """
        return self.get_context(query, max_tokens)
    
    # Legacy method for backward compatibility
    async def record_experience_legacy(self, content, metadata=None, significance=0.7):
        """Legacy method for recording an experience with significance parameter.
        
        Args:
            content: Experience content text
            metadata: Optional additional metadata
            significance: Significance of the experience (deprecated, use quickrecal_score)
            
        Returns:
            Success status
        """
        self.logger.warning("record_experience_legacy with significance parameter is deprecated. Use record_experience with quickrecal_score instead.")
        return await self.record_experience(content, metadata, significance)