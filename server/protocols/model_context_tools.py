from typing import Dict, Any, List, Optional, Union, Set, Callable
import logging
import json
import asyncio
import re
from datetime import datetime, timedelta
import inspect
import traceback

from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class ModelContextToolProvider(ToolProvider):
    """
    Tool provider specialized for Lucidia's Model Context Protocol (MCP) functions.
    
    This provider enables Lucidia to call tools directly from conversation
    for self-evolution, reflection, knowledge graph updates, and dream processes.
    """
    
    def __init__(self, self_model=None, world_model=None, knowledge_graph=None, 
                 memory_system=None, dream_processor=None, spiral_manager=None, 
                 parameter_manager=None, model_manager=None, dream_parameter_adapter=None):
        super().__init__()
        self.self_model = self_model
        self.world_model = world_model
        self.knowledge_graph = knowledge_graph
        self.memory_system = memory_system
        self.dream_processor = dream_processor
        self.spiral_manager = spiral_manager
        self.parameter_manager = parameter_manager
        self.model_manager = model_manager
        self.dream_parameter_adapter = dream_parameter_adapter
        self.tool_usage_history = []
        
        # Initialize all tools
        self.register_mcp_tools()
    
    def register_mcp_tools(self):
        """Register all Model Context Protocol tools."""
        # Self-Model Tools
        self.register_tool(
            name="update_self_model",
            function=self.update_self_model,
            description="Update Lucidia's self-model with new reflections, characteristics, or learning. "
                      "Use this to evolve Lucidia's self-awareness and capabilities.",
            parameters={
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "description": "Aspect of self to update (personality, emotional_state, capabilities, knowledge, learning_insight)",
                        "enum": ["personality", "emotional_state", "capabilities", "knowledge", "learning_insight"]
                    },
                    "content": {
                        "type": "object",
                        "description": "Details of the update to apply"
                    },
                    "significance": {
                        "type": "number",
                        "description": "Significance level of this update (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7
                    }
                },
                "required": ["aspect", "content"],
            }
        )
        
        # Knowledge Graph Tools
        self.register_tool(
            name="update_knowledge_graph",
            function=self.update_knowledge_graph,
            description="Update Lucidia's knowledge graph with new concepts, relationships, or insights. "
                      "Use this to evolve Lucidia's understanding of the world and relationships.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Type of operation to perform on the knowledge graph",
                        "enum": ["add_concept", "add_relationship", "update_concept", "delete_concept", "add_insight"]
                    },
                    "data": {
                        "type": "object",
                        "description": "Data for the knowledge graph operation"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of this knowledge (user, inference, dream, observation)",
                        "default": "inference"
                    }
                },
                "required": ["operation", "data"],
            }
        )
        
        # Memory System Tools
        self.register_tool(
            name="memory_operation",
            function=self.memory_operation,
            description="Perform operations on Lucidia's memory system like search, store, retrieve, or consolidate. "
                      "Use this to help Lucidia manage its memory effectively.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Type of memory operation to perform",
                        "enum": ["store", "retrieve", "search", "consolidate", "associate", "forget"]
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory to operate on",
                        "enum": ["conversation", "factual", "emotional", "procedural", "relationship", "any"],
                        "default": "any"
                    },
                    "content": {
                        "type": "object",
                        "description": "Content for the memory operation"
                    },
                    "significance": {
                        "type": "number",
                        "description": "Significance of this memory (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.75
                    }
                },
                "required": ["operation", "content"],
            }
        )
        
        # Dream Process Tools
        self.register_tool(
            name="manage_dream_process",
            function=self.manage_dream_process,
            description="Manage Lucidia's dream processes including starting, stopping, scheduling, and configuring dreams. "
                      "Use this to control when and how Lucidia dreams.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to take on the dream process",
                        "enum": ["start", "stop", "schedule", "configure", "get_status"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the dream action"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level for this dream action",
                        "enum": ["low", "medium", "high", "critical"],
                        "default": "medium"
                    }
                },
                "required": ["action"],
            }
        )
        
        # Spiral Phase Tools
        self.register_tool(
            name="manage_spiral_phase",
            function=self.manage_spiral_phase,
            description="Manage Lucidia's spiral phases for cognitive development and reflection. "
                      "Use this to control Lucidia's depth of thinking and cognitive focus.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to take on the spiral phase",
                        "enum": ["advance", "set_phase", "get_current", "configure", "evaluate"]
                    },
                    "phase": {
                        "type": "string",
                        "description": "Target spiral phase (for set_phase action)",
                        "enum": ["observation", "reflection", "adaptation"]
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the spiral phase action"
                    }
                },
                "required": ["action"],
            }
        )
        
        # Parameter Management Tools
        self.register_tool(
            name="manage_parameters",
            function=self.manage_parameters,
            description="Manage Lucidia's system parameters to control behavior, performance, and capabilities. "
                      "Use this to optimize Lucidia's functioning.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to take on parameters",
                        "enum": ["get", "update", "verify", "lock", "unlock"]
                    },
                    "parameter_path": {
                        "type": "string",
                        "description": "Path to the parameter to manage (e.g., 'dream_cycles.idle_threshold')"
                    },
                    "value": {
                        "type": "object",
                        "description": "New value for the parameter (for update action)"
                    },
                    "transition_period": {
                        "type": "number",
                        "description": "Transition period in seconds (for update action)"
                    }
                },
                "required": ["action", "parameter_path"],
            }
        )
        
        # Reasoning and Reflection Tools
        self.register_tool(
            name="metacognitive_reflection",
            function=self.metacognitive_reflection,
            description="Trigger a metacognitive reflection process to analyze Lucidia's thinking, reasoning, or learning. "
                      "Use this to help Lucidia improve its cognitive abilities.",
            parameters={
                "type": "object",
                "properties": {
                    "reflection_type": {
                        "type": "string",
                        "description": "Type of reflection to perform",
                        "enum": ["reasoning_quality", "learning_progress", "knowledge_gaps", "bias_analysis", "goal_alignment"]
                    },
                    "context": {
                        "type": "string",
                        "description": "Context information for the reflection"
                    },
                    "depth": {
                        "type": "number",
                        "description": "Depth of reflection (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7
                    }
                },
                "required": ["reflection_type", "context"],
            }
        )
        
        # Usage Analysis Tools
        self.register_tool(
            name="analyze_tool_usage",
            function=self.analyze_tool_usage,
            description="Analyze how Lucidia has been using tools to identify patterns and improvement opportunities. "
                      "Use this to optimize tool usage.",
            parameters={
                "type": "object",
                "properties": {
                    "time_period": {
                        "type": "string",
                        "description": "Time period to analyze",
                        "enum": ["recent", "day", "week", "all"],
                        "default": "recent"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["frequency", "effectiveness", "patterns", "recommendations"],
                        "default": "patterns"
                    }
                },
                "required": ["analysis_type"],
            }
        )
        
        # System health monitoring tools
        self.register_tool(
            name="check_system_health",
            function=self.check_system_health,
            description="Check the health of Lucidia's various subsystems and components. "
                      "Use this to identify and address potential issues.",
            parameters={
                "type": "object",
                "properties": {
                    "subsystems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subsystems to check (memory, knowledge, self_model, dream, all)",
                        "default": ["all"]
                    },
                    "detail_level": {
                        "type": "string",
                        "description": "Level of detail for the health check",
                        "enum": ["summary", "detailed", "diagnostic"],
                        "default": "summary"
                    }
                },
                "required": [],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} model context protocol tools")
    
    # ========== Tool Implementation Methods ==========
    
    async def update_self_model(self, aspect: str, content: Dict[str, Any], 
                          significance: float = 0.7) -> Dict[str, Any]:
        """Update Lucidia's self-model with new reflections or characteristics."""
        try:
            if not self.self_model:
                return {
                    "status": "error",
                    "message": "Self-model not initialized",
                    "details": None
                }
            
            # Record the tool usage
            self._record_tool_usage("update_self_model", {
                "aspect": aspect,
                "content": content,
                "significance": significance
            })
            
            # Different handling based on aspect
            if aspect == "personality":
                # Update personality traits or characteristics
                for trait, value in content.items():
                    if isinstance(value, (int, float)) and trait in self.self_model.personality:
                        self.self_model.personality[trait] = value
                    elif isinstance(value, str) and trait == "notes":
                        if not hasattr(self.self_model, "personality_notes"):
                            self.self_model.personality_notes = []
                        self.self_model.personality_notes.append({
                            "timestamp": datetime.now().isoformat(),
                            "note": value,
                            "significance": significance
                        })
                
                return {
                    "status": "success",
                    "message": f"Updated {len(content)} personality aspects",
                    "details": {
                        "updated_traits": list(content.keys()),
                        "current_state": {k: self.self_model.personality.get(k) for k in content.keys()}
                    }
                }
                
            elif aspect == "emotional_state":
                # Update emotional state
                if not hasattr(self.self_model, "emotional_state"):
                    self.self_model.emotional_state = {}
                
                # Merge the provided emotional state with the existing one
                for emotion, value in content.items():
                    self.self_model.emotional_state[emotion] = value
                
                # Record high-significance emotional changes
                if significance > 0.8 and hasattr(self.self_model, "emotional_history"):
                    self.self_model.emotional_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "emotional_state": content.copy(),
                        "significance": significance
                    })
                
                return {
                    "status": "success",
                    "message": "Updated emotional state",
                    "details": {
                        "current_state": self.self_model.emotional_state
                    }
                }
                
            elif aspect == "capabilities":
                # Update capabilities
                if not hasattr(self.self_model, "capabilities"):
                    self.self_model.capabilities = {}
                
                # Add or update capabilities
                for capability, details in content.items():
                    self.self_model.capabilities[capability] = details
                
                return {
                    "status": "success",
                    "message": f"Updated {len(content)} capabilities",
                    "details": {
                        "updated_capabilities": list(content.keys()),
                        "current_capabilities": len(self.self_model.capabilities)
                    }
                }
                
            elif aspect == "knowledge":
                # Update knowledge areas
                if not hasattr(self.self_model, "knowledge_areas"):
                    self.self_model.knowledge_areas = {}
                
                # Add or update knowledge areas
                for area, details in content.items():
                    if area in self.self_model.knowledge_areas:
                        # Update existing area
                        for key, value in details.items():
                            self.self_model.knowledge_areas[area][key] = value
                    else:
                        # Create new area
                        self.self_model.knowledge_areas[area] = details
                
                return {
                    "status": "success",
                    "message": f"Updated {len(content)} knowledge areas",
                    "details": {
                        "updated_areas": list(content.keys())
                    }
                }
                
            elif aspect == "learning_insight":
                # Record learning insights
                if not hasattr(self.self_model, "learning_insights"):
                    self.self_model.learning_insights = []
                
                # Add the insight with metadata
                insight = {
                    "timestamp": datetime.now().isoformat(),
                    "content": content.get("insight", ""),
                    "category": content.get("category", "general"),
                    "significance": significance,
                    "source": content.get("source", "reflection")
                }
                
                self.self_model.learning_insights.append(insight)
                
                # For high-significance insights, also advance the spiral
                if significance > 0.85 and self.spiral_manager:
                    await self.spiral_manager.record_insight({
                        "content": content.get("insight", ""),
                        "significance": significance,
                        "source": "self_model"
                    })
                
                return {
                    "status": "success",
                    "message": "Recorded learning insight",
                    "details": {
                        "insight_id": len(self.self_model.learning_insights) - 1,
                        "category": content.get("category", "general"),
                        "total_insights": len(self.self_model.learning_insights)
                    }
                }
            
            return {
                "status": "error",
                "message": f"Unknown aspect: {aspect}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in update_self_model: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def update_knowledge_graph(self, operation: str, data: Dict[str, Any], 
                              source: str = "inference") -> Dict[str, Any]:
        """Update Lucidia's knowledge graph with new concepts, relationships, or insights."""
        try:
            if not self.knowledge_graph:
                return {
                    "status": "error",
                    "message": "Knowledge graph not initialized",
                    "details": None
                }
            
            # Record the tool usage
            self._record_tool_usage("update_knowledge_graph", {
                "operation": operation,
                "data": data,
                "source": source
            })
            
            # Handle different operations
            if operation == "add_concept":
                # Add a new concept to the knowledge graph
                concept_id = await self.knowledge_graph.add_concept(
                    concept_name=data.get("name"),
                    concept_type=data.get("type", "concept"),
                    description=data.get("description", ""),
                    attributes=data.get("attributes", {}),
                    source=source
                )
                
                return {
                    "status": "success",
                    "message": f"Added concept: {data.get('name')}",
                    "details": {
                        "concept_id": concept_id,
                        "concept_name": data.get("name")
                    }
                }
                
            elif operation == "add_relationship":
                # Add a relationship between concepts
                relationship_id = await self.knowledge_graph.add_relationship(
                    source_id=data.get("source_id"),
                    target_id=data.get("target_id"),
                    relationship_type=data.get("relationship_type", "related_to"),
                    strength=data.get("strength", 0.5),
                    attributes=data.get("attributes", {})
                )
                
                return {
                    "status": "success",
                    "message": f"Added relationship: {data.get('relationship_type')} from {data.get('source_id')} to {data.get('target_id')}",
                    "details": {
                        "relationship_id": relationship_id
                    }
                }
                
            elif operation == "update_concept":
                # Update an existing concept
                await self.knowledge_graph.update_concept(
                    concept_id=data.get("concept_id"),
                    updates=data.get("updates", {})
                )
                
                return {
                    "status": "success",
                    "message": f"Updated concept: {data.get('concept_id')}",
                    "details": {
                        "concept_id": data.get("concept_id"),
                        "updated_fields": list(data.get("updates", {}).keys())
                    }
                }
                
            elif operation == "delete_concept":
                # Delete a concept
                await self.knowledge_graph.delete_concept(
                    concept_id=data.get("concept_id")
                )
                
                return {
                    "status": "success",
                    "message": f"Deleted concept: {data.get('concept_id')}",
                    "details": {
                        "concept_id": data.get("concept_id")
                    }
                }
                
            elif operation == "add_insight":
                # Add an insight linked to concepts
                insight_id = await self.knowledge_graph.add_insight(
                    content=data.get("content"),
                    source=source,
                    related_concepts=data.get("related_concepts", []),
                    significance=data.get("significance", 0.7),
                    insight_type=data.get("insight_type", "general")
                )
                
                return {
                    "status": "success",
                    "message": "Added insight to knowledge graph",
                    "details": {
                        "insight_id": insight_id,
                        "related_concepts": len(data.get("related_concepts", []))
                    }
                }
            
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in update_knowledge_graph: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def memory_operation(self, operation: str, content: Dict[str, Any], 
                         memory_type: str = "any", significance: float = 0.75) -> Dict[str, Any]:
        """Perform operations on Lucidia's memory system."""
        try:
            if not self.memory_system:
                return {
                    "status": "error",
                    "message": "Memory system not initialized",
                    "details": None
                }
            
            # Record the tool usage
            self._record_tool_usage("memory_operation", {
                "operation": operation,
                "memory_type": memory_type,
                "content": content,
                "significance": significance
            })
            
            # Handle different operations
            if operation == "store":
                # Store a new memory
                memory_id = await self.memory_system.store_memory(
                    content=content.get("content"),
                    memory_type=memory_type,
                    significance=significance,
                    metadata=content.get("metadata", {})
                )
                
                return {
                    "status": "success",
                    "message": f"Stored new memory of type {memory_type}",
                    "details": {
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                        "significance": significance
                    }
                }
                
            elif operation == "retrieve":
                # Retrieve memory by ID or query
                if "memory_id" in content:
                    memory = await self.memory_system.get_memory_by_id(content["memory_id"])
                    
                    return {
                        "status": "success" if memory else "error",
                        "message": "Retrieved memory" if memory else "Memory not found",
                        "details": memory
                    }
                else:
                    # Retrieve by semantic search
                    memories = await self.memory_system.search_memories(
                        query=content.get("query", ""),
                        memory_type=memory_type,
                        limit=content.get("limit", 10),
                        min_significance=content.get("min_significance", 0.0),
                        min_quickrecal_score=content.get("min_quickrecal_score", None),
                        recency_bias=content.get("recency_bias", 0.3)
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Retrieved {len(memories)} memories",
                        "details": {
                            "memories": memories,
                            "count": len(memories)
                        }
                    }
                    
            elif operation == "search":
                # Search memories
                memories = await self.memory_system.search_memories(
                    query=content.get("query", ""),
                    memory_type=memory_type,
                    limit=content.get("limit", 10),
                    min_significance=content.get("min_significance", 0.0),
                    min_quickrecal_score=content.get("min_quickrecal_score", None),
                    recency_bias=content.get("recency_bias", 0.3)
                )
                
                return {
                    "status": "success",
                    "message": f"Found {len(memories)} memories",
                    "details": {
                        "memories": memories,
                        "count": len(memories),
                        "query": content.get("query", "")
                    }
                }
                
            elif operation == "consolidate":
                # Consolidate memories (batch processing)
                if not hasattr(self.memory_system, "consolidate_memories"):
                    return {
                        "status": "error",
                        "message": "Memory consolidation not supported by this memory system",
                        "details": None
                    }
                
                result = await self.memory_system.consolidate_memories(
                    memory_type=memory_type,
                    criteria=content.get("criteria", {}),
                    action=content.get("action", "summarize"),
                    min_quickrecal_score=content.get("min_quickrecal_score", None),
                    min_significance=content.get("min_significance", 0.0)
                )
                
                return {
                    "status": "success",
                    "message": f"Consolidated {result.get('count', 0)} memories",
                    "details": result
                }
                
            elif operation == "associate":
                # Associate memories or create relationships
                if not hasattr(self.memory_system, "associate_memories"):
                    return {
                        "status": "error",
                        "message": "Memory association not supported by this memory system",
                        "details": None
                    }
                
                association_id = await self.memory_system.associate_memories(
                    source_id=content.get("source_id"),
                    target_id=content.get("target_id"),
                    relationship=content.get("relationship", "related_to"),
                    strength=content.get("strength", 0.5)
                )
                
                return {
                    "status": "success",
                    "message": "Created memory association",
                    "details": {
                        "association_id": association_id,
                        "source_id": content.get("source_id"),
                        "target_id": content.get("target_id")
                    }
                }
                
            elif operation == "forget":
                # Reduce significance or delete memories
                if "memory_id" in content:
                    # Specific memory
                    if content.get("permanent", False):
                        # Permanent deletion
                        result = await self.memory_system.delete_memory(content["memory_id"])
                    else:
                        # Reduce significance
                        new_significance = max(0.0, significance - 0.3)
                        result = await self.memory_system.update_memory_significance(
                            memory_id=content["memory_id"],
                            new_significance=new_significance
                        )
                    
                    return {
                        "status": "success" if result else "error",
                        "message": "Memory forgotten" if result else "Failed to forget memory",
                        "details": {
                            "memory_id": content["memory_id"],
                            "permanent": content.get("permanent", False)
                        }
                    }
                else:
                    # Batch operation based on criteria
                    count = await self.memory_system.reduce_memory_significance(
                        criteria=content.get("criteria", {}),
                        reduction=content.get("reduction", 0.3),
                        memory_type=memory_type
                    )
                    
                    return {
                        "status": "success",
                        "message": f"Reduced significance of {count} memories",
                        "details": {
                            "count": count,
                            "memory_type": memory_type
                        }
                    }
            
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in memory_operation: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def manage_dream_process(self, action: str, parameters: Dict[str, Any] = None, 
                             priority: str = "medium") -> Dict[str, Any]:
        """Manage Lucidia's dream processes."""
        try:
            if not self.dream_processor:
                return {
                    "status": "error",
                    "message": "Dream processor not initialized",
                    "details": None
                }
            
            if parameters is None:
                parameters = {}
            
            # Record the tool usage
            self._record_tool_usage("manage_dream_process", {
                "action": action,
                "parameters": parameters,
                "priority": priority
            })
            
            # Handle different actions
            if action == "start":
                # Start a dream
                force = parameters.get("force", False)
                seed = parameters.get("seed", None)
                
                # Check if already dreaming
                if self.dream_processor.is_dreaming:
                    return {
                        "status": "warning",
                        "message": "Dream processor is already dreaming",
                        "details": {
                            "current_dream_id": self.dream_processor.current_dream_id,
                            "elapsed_time": self.dream_processor.dream_elapsed_time
                        }
                    }
                
                # Start dreaming
                result = await self.dream_processor.start_dreaming(
                    forced=force,
                    seed=seed,
                    dream_depth=parameters.get("depth", None),
                    dream_creativity=parameters.get("creativity", None)
                )
                
                return {
                    "status": "success",
                    "message": "Dream started successfully",
                    "details": result
                }
                
            elif action == "stop":
                # Stop a dream
                if not self.dream_processor.is_dreaming:
                    return {
                        "status": "warning",
                        "message": "Dream processor is not currently dreaming",
                        "details": None
                    }
                
                graceful = parameters.get("graceful", True)
                result = await self.dream_processor.stop_dreaming(graceful=graceful)
                
                return {
                    "status": "success",
                    "message": f"Dream stopped {'gracefully' if graceful else 'forcefully'}",
                    "details": result
                }
                
            elif action == "schedule":
                # Schedule a dream for later
                schedule_time = parameters.get("schedule_time")
                duration = parameters.get("duration", 10)  # minutes
                
                if not schedule_time:
                    # If no time provided, schedule based on delay
                    delay_minutes = parameters.get("delay_minutes", 60)
                    schedule_time = datetime.now() + timedelta(minutes=delay_minutes)
                
                result = await self.dream_processor.schedule_dream(
                    schedule_time=schedule_time,
                    duration_minutes=duration,
                    priority=priority,
                    parameters=parameters.get("dream_parameters", {})
                )
                
                return {
                    "status": "success",
                    "message": f"Dream scheduled for {schedule_time}",
                    "details": result
                }
                
            elif action == "configure":
                # Configure dream parameters
                if not self.dream_parameter_adapter:
                    return {
                        "status": "error",
                        "message": "Dream parameter adapter not initialized",
                        "details": None
                    }
                
                config_changes = {}
                
                # Process configuration changes
                for key, value in parameters.items():
                    # Map friendly names to parameter paths
                    param_path = {
                        "idle_threshold": "dream_cycles.idle_threshold",
                        "auto_dream_enabled": "dream_cycles.auto_enabled",
                        "dream_intensity": "dream_cycles.dream_intensity",
                        "min_idle_time": "dream_cycles.min_idle_time",
                        "max_duration": "dream_cycles.max_duration",
                    }.get(key, key)  # If not in mapping, use as-is
                    
                    # Apply the configuration
                    if await self.dream_parameter_adapter.set_parameter(param_path, value):
                        config_changes[param_path] = value
                
                return {
                    "status": "success",
                    "message": f"Updated {len(config_changes)} dream parameters",
                    "details": {
                        "updated_parameters": config_changes
                    }
                }
                
            elif action == "get_status":
                # Get current dream status
                status = await self.dream_processor.get_status()
                
                return {
                    "status": "success",
                    "message": "Retrieved dream status",
                    "details": status
                }
            
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in manage_dream_process: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def manage_spiral_phase(self, action: str, phase: str = None,
                           parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage Lucidia's spiral phases."""
        try:
            if not self.spiral_manager:
                return {
                    "status": "error",
                    "message": "Spiral manager not initialized",
                    "details": None
                }
            
            if parameters is None:
                parameters = {}
            
            # Record the tool usage
            self._record_tool_usage("manage_spiral_phase", {
                "action": action,
                "phase": phase,
                "parameters": parameters
            })
            
            # Handle different actions
            if action == "advance":
                # Advance to the next spiral phase
                result = await self.spiral_manager.advance_phase(
                    force=parameters.get("force", False),
                    context=parameters.get("context", {}),
                    insight=parameters.get("insight", None)
                )
                
                return {
                    "status": "success",
                    "message": f"Advanced to phase: {result.get('new_phase')}",
                    "details": result
                }
                
            elif action == "set_phase":
                # Set a specific phase
                if not phase:
                    return {
                        "status": "error",
                        "message": "Phase parameter is required for set_phase action",
                        "details": None
                    }
                
                result = await self.spiral_manager.set_phase(
                    phase=phase,
                    reason=parameters.get("reason", "MCP request"),
                    context=parameters.get("context", {})
                )
                
                return {
                    "status": "success",
                    "message": f"Set phase to: {phase}",
                    "details": result
                }
                
            elif action == "get_current":
                # Get the current phase
                result = await self.spiral_manager.get_current_phase()
                
                return {
                    "status": "success",
                    "message": f"Current phase is: {result.get('phase')}",
                    "details": result
                }
                
            elif action == "configure":
                # Configure spiral parameters
                updated_params = {}
                
                for key, value in parameters.items():
                    # Map friendly names to parameter paths
                    param_path = {
                        "min_phase_duration": "spiral.min_phase_duration",
                        "phase_evaluation_threshold": "spiral.phase_evaluation_threshold",
                        "insight_threshold": "spiral.insight_threshold",
                    }.get(key, key)  # If not in mapping, use as-is
                    
                    # Update the parameter
                    if self.parameter_manager:
                        await self.parameter_manager.set_parameter(param_path, value)
                        updated_params[param_path] = value
                
                return {
                    "status": "success",
                    "message": f"Updated {len(updated_params)} spiral parameters",
                    "details": {
                        "updated_parameters": updated_params
                    }
                }
                
            elif action == "evaluate":
                # Evaluate the current phase effectiveness
                result = await self.spiral_manager.evaluate_current_phase(
                    context=parameters.get("context", {}),
                    detailed=parameters.get("detailed", False)
                )
                
                return {
                    "status": "success",
                    "message": "Evaluated current phase",
                    "details": result
                }
            
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in manage_spiral_phase: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def manage_parameters(self, action: str, parameter_path: str,
                         value: Any = None, transition_period: int = 0) -> Dict[str, Any]:
        """Manage Lucidia's system parameters."""
        try:
            if not self.parameter_manager:
                return {
                    "status": "error",
                    "message": "Parameter manager not initialized",
                    "details": None
                }
            
            # Record the tool usage
            self._record_tool_usage("manage_parameters", {
                "action": action,
                "parameter_path": parameter_path,
                "value": value,
                "transition_period": transition_period
            })
            
            # Handle different actions
            if action == "get":
                # Get a parameter value
                result = await self.parameter_manager.get_parameter(parameter_path)
                
                if result is None:
                    return {
                        "status": "error",
                        "message": f"Parameter not found: {parameter_path}",
                        "details": None
                    }
                
                return {
                    "status": "success",
                    "message": f"Retrieved parameter: {parameter_path}",
                    "details": {
                        "parameter_path": parameter_path,
                        "value": result,
                        "metadata": await self.parameter_manager.get_parameter_metadata(parameter_path)
                    }
                }
                
            elif action == "update":
                # Update a parameter
                if value is None:
                    return {
                        "status": "error",
                        "message": "Value is required for update action",
                        "details": None
                    }
                
                # Check if gradual transition is requested
                if transition_period > 0:
                    if hasattr(self.parameter_manager, "set_parameter_with_transition"):
                        result = await self.parameter_manager.set_parameter_with_transition(
                            parameter_path=parameter_path,
                            target_value=value,
                            transition_period_seconds=transition_period
                        )
                    else:
                        # Fallback to immediate update
                        result = await self.parameter_manager.set_parameter(
                            parameter_path=parameter_path,
                            value=value
                        )
                else:
                    # Immediate update
                    result = await self.parameter_manager.set_parameter(
                        parameter_path=parameter_path,
                        value=value
                    )
                
                return {
                    "status": "success",
                    "message": f"Updated parameter: {parameter_path}",
                    "details": {
                        "parameter_path": parameter_path,
                        "new_value": value,
                        "transition_period": transition_period if transition_period > 0 else None
                    }
                }
                
            elif action == "verify":
                # Verify if a parameter exists and is valid
                exists = await self.parameter_manager.parameter_exists(parameter_path)
                
                if not exists:
                    return {
                        "status": "warning",
                        "message": f"Parameter does not exist: {parameter_path}",
                        "details": {
                            "parameter_path": parameter_path,
                            "exists": False
                        }
                    }
                
                # Get metadata to check if valid
                metadata = await self.parameter_manager.get_parameter_metadata(parameter_path)
                current_value = await self.parameter_manager.get_parameter(parameter_path)
                
                valid = True
                if metadata and "validation" in metadata:
                    # If there's validation logic, apply it
                    valid = await self.parameter_manager.validate_parameter(
                        parameter_path=parameter_path,
                        value=current_value
                    )
                
                return {
                    "status": "success",
                    "message": f"Parameter verified: {parameter_path}",
                    "details": {
                        "parameter_path": parameter_path,
                        "exists": True,
                        "valid": valid,
                        "current_value": current_value,
                        "metadata": metadata
                    }
                }
                
            elif action == "lock" or action == "unlock":
                # Lock or unlock a parameter
                if not hasattr(self.parameter_manager, "lock_parameter"):
                    return {
                        "status": "error",
                        "message": "Parameter locking not supported by this parameter manager",
                        "details": None
                    }
                
                if action == "lock":
                    result = await self.parameter_manager.lock_parameter(
                        parameter_path=parameter_path,
                        reason="MCP request",
                        duration_seconds=value if isinstance(value, (int, float)) else None
                    )
                else:  # unlock
                    result = await self.parameter_manager.unlock_parameter(
                        parameter_path=parameter_path
                    )
                
                return {
                    "status": "success",
                    "message": f"Parameter {action}ed: {parameter_path}",
                    "details": result
                }
            
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in manage_parameters: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def metacognitive_reflection(self, reflection_type: str, context: str,
                                depth: float = 0.7) -> Dict[str, Any]:
        """Trigger a metacognitive reflection process."""
        try:
            # Record the tool usage
            self._record_tool_usage("metacognitive_reflection", {
                "reflection_type": reflection_type,
                "context": context,
                "depth": depth
            })
            
            # Check dependencies
            if not self.self_model:
                return {
                    "status": "error",
                    "message": "Self-model not initialized",
                    "details": None
                }
            
            if not self.model_manager:
                return {
                    "status": "error",
                    "message": "Model manager not initialized",
                    "details": None
                }
            
            # Structured reflection process based on type
            reflection_prompt = self._generate_reflection_prompt(reflection_type, context, depth)
            
            # Perform the reflection using the LLM
            reflection_response = await self.call_llm(
                model_manager=self.model_manager,
                messages=[
                    {"role": "system", "content": "You are Lucidia's metacognitive reflection system. "
                     "Analyze the provided context deeply and provide insights that will help Lucidia "
                     "improve its cognitive processes."},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0.5 + (depth * 0.4),  # Adjust temperature based on depth
                max_tokens=800
            )
            
            if "error" in reflection_response and not reflection_response.get("simulated", False):
                return {
                    "status": "error",
                    "message": "Failed to generate reflection",
                    "details": reflection_response
                }
            
            # Extract the reflection content
            reflection_content = reflection_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Save the reflection to self-model
            if hasattr(self.self_model, "metacognitive_reflections"):
                self.self_model.metacognitive_reflections.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": reflection_type,
                    "context": context,
                    "depth": depth,
                    "reflection": reflection_content
                })
            
            # For certain reflection types, also update other aspects of the system
            if reflection_type == "learning_progress" and self.spiral_manager:
                # Record learning insights in spiral
                await self.spiral_manager.record_insight({
                    "source": "metacognitive_reflection",
                    "content": reflection_content,
                    "significance": 0.6 + (depth * 0.3)
                })
            
            return {
                "status": "success",
                "message": f"Completed {reflection_type} reflection",
                "details": {
                    "reflection": reflection_content,
                    "reflection_type": reflection_type,
                    "depth": depth,
                    "next_steps": self._extract_next_steps(reflection_content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in metacognitive_reflection: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def analyze_tool_usage(self, analysis_type: str, time_period: str = "recent") -> Dict[str, Any]:
        """Analyze how Lucidia has been using tools to identify patterns and improvement opportunities."""
        try:
            # Filter tool usage based on time period
            now = datetime.now()
            filtered_usage = []
            
            for usage in self.tool_usage_history:
                timestamp = datetime.fromisoformat(usage["timestamp"])
                if time_period == "recent" and (now - timestamp).total_seconds() < 3600:  # Last hour
                    filtered_usage.append(usage)
                elif time_period == "day" and (now - timestamp).days < 1:
                    filtered_usage.append(usage)
                elif time_period == "week" and (now - timestamp).days < 7:
                    filtered_usage.append(usage)
                elif time_period == "all":
                    filtered_usage.append(usage)
            
            # If no usage data, return empty analysis
            if not filtered_usage:
                return {
                    "status": "warning",
                    "message": f"No tool usage data found for time period: {time_period}",
                    "details": {
                        "time_period": time_period,
                        "analysis_type": analysis_type,
                        "records_found": 0
                    }
                }
            
            # Perform the requested analysis
            if analysis_type == "frequency":
                # Analyze frequency of tool usage
                tool_counts = {}
                for usage in filtered_usage:
                    tool_name = usage["tool_name"]
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                
                return {
                    "status": "success",
                    "message": f"Analyzed tool usage frequency for {time_period}",
                    "details": {
                        "time_period": time_period,
                        "total_usages": len(filtered_usage),
                        "tool_counts": tool_counts,
                        "most_used": max(tool_counts.items(), key=lambda x: x[1]) if tool_counts else None
                    }
                }
                
            elif analysis_type == "effectiveness":
                # Analyze effectiveness of tool usage
                success_rates = {}
                tool_durations = {}
                
                for usage in filtered_usage:
                    tool_name = usage["tool_name"]
                    success = usage.get("result", {}).get("status") == "success"
                    
                    # Track success/failure
                    if tool_name not in success_rates:
                        success_rates[tool_name] = {"success": 0, "failure": 0}
                    
                    if success:
                        success_rates[tool_name]["success"] += 1
                    else:
                        success_rates[tool_name]["failure"] += 1
                    
                    # Track duration if available
                    if "duration_ms" in usage:
                        if tool_name not in tool_durations:
                            tool_durations[tool_name] = []
                        tool_durations[tool_name].append(usage["duration_ms"])
                
                # Calculate average durations
                avg_durations = {}
                for tool, durations in tool_durations.items():
                    avg_durations[tool] = sum(durations) / len(durations) if durations else None
                
                return {
                    "status": "success",
                    "message": f"Analyzed tool usage effectiveness for {time_period}",
                    "details": {
                        "time_period": time_period,
                        "total_usages": len(filtered_usage),
                        "success_rates": success_rates,
                        "average_durations_ms": avg_durations
                    }
                }
                
            elif analysis_type == "patterns":
                # Analyze patterns of tool usage
                tool_sequences = []
                current_sequence = []
                
                # Build sequences of tool usage
                for i, usage in enumerate(filtered_usage):
                    if i > 0 and (datetime.fromisoformat(usage["timestamp"]) - 
                                 datetime.fromisoformat(filtered_usage[i-1]["timestamp"])).total_seconds() > 300:
                        # More than 5 minutes between usages, consider new sequence
                        if current_sequence:
                            tool_sequences.append(current_sequence)
                            current_sequence = []
                    
                    current_sequence.append(usage["tool_name"])
                
                # Add the last sequence if not empty
                if current_sequence:
                    tool_sequences.append(current_sequence)
                
                # Find common patterns
                common_pairs = {}
                for sequence in tool_sequences:
                    for i in range(len(sequence) - 1):
                        pair = (sequence[i], sequence[i+1])
                        common_pairs[pair] = common_pairs.get(pair, 0) + 1
                
                return {
                    "status": "success",
                    "message": f"Analyzed tool usage patterns for {time_period}",
                    "details": {
                        "time_period": time_period,
                        "total_usages": len(filtered_usage),
                        "sequence_count": len(tool_sequences),
                        "common_pairs": {f"{a} -> {b}": count for (a, b), count in common_pairs.items()},
                        "longest_sequence": max([len(seq) for seq in tool_sequences]) if tool_sequences else 0
                    }
                }
                
            elif analysis_type == "recommendations":
                # Generate recommendations based on usage
                # This requires analyzing the data and using the LLM to generate insights
                
                # Prepare usage summary for LLM
                usage_summary = self._generate_usage_summary(filtered_usage)
                
                # Get recommendations from LLM
                if self.model_manager:
                    recommendation_response = await self.call_llm(
                        model_manager=self.model_manager,
                        messages=[
                            {"role": "system", "content": "You are Lucidia's tool usage analyzer. "
                             "Based on the tool usage data, provide specific recommendations for "
                             "more effective tool utilization."},
                            {"role": "user", "content": usage_summary}
                        ],
                        temperature=0.7,
                        max_tokens=600
                    )
                    
                    recommendations = recommendation_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    recommendations = "LLM not available to generate recommendations"
                
                return {
                    "status": "success",
                    "message": f"Generated recommendations based on tool usage for {time_period}",
                    "details": {
                        "time_period": time_period,
                        "total_usages": len(filtered_usage),
                        "tools_used": list(set([u["tool_name"] for u in filtered_usage])),
                        "recommendations": recommendations
                    }
                }
            
            return {
                "status": "error",
                "message": f"Unknown analysis type: {analysis_type}",
                "details": None
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_tool_usage: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    async def check_system_health(self, subsystems: List[str] = None, 
                           detail_level: str = "summary") -> Dict[str, Any]:
        """Check the health of Lucidia's various subsystems and components."""
        try:
            if subsystems is None or "all" in subsystems:
                subsystems = ["memory", "knowledge", "self_model", "dream", "spiral", "parameters"]
            
            # Record the tool usage
            self._record_tool_usage("check_system_health", {
                "subsystems": subsystems,
                "detail_level": detail_level
            })
            
            health_results = {}
            overall_status = "healthy"
            
            # Check each requested subsystem
            for subsystem in subsystems:
                if subsystem == "memory" and self.memory_system:
                    memory_health = await self._check_memory_health(detail_level)
                    health_results["memory"] = memory_health
                    if memory_health.get("status") == "critical":
                        overall_status = "critical"
                    elif memory_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
                        
                elif subsystem == "knowledge" and self.knowledge_graph:
                    knowledge_health = await self._check_knowledge_health(detail_level)
                    health_results["knowledge"] = knowledge_health
                    if knowledge_health.get("status") == "critical":
                        overall_status = "critical"
                    elif knowledge_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
                        
                elif subsystem == "self_model" and self.self_model:
                    self_model_health = await self._check_self_model_health(detail_level)
                    health_results["self_model"] = self_model_health
                    if self_model_health.get("status") == "critical":
                        overall_status = "critical"
                    elif self_model_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
                        
                elif subsystem == "dream" and self.dream_processor:
                    dream_health = await self._check_dream_health(detail_level)
                    health_results["dream"] = dream_health
                    if dream_health.get("status") == "critical":
                        overall_status = "critical"
                    elif dream_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
                        
                elif subsystem == "spiral" and self.spiral_manager:
                    spiral_health = await self._check_spiral_health(detail_level)
                    health_results["spiral"] = spiral_health
                    if spiral_health.get("status") == "critical":
                        overall_status = "critical"
                    elif spiral_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
                        
                elif subsystem == "parameters" and self.parameter_manager:
                    parameter_health = await self._check_parameter_health(detail_level)
                    health_results["parameters"] = parameter_health
                    if parameter_health.get("status") == "critical":
                        overall_status = "critical"
                    elif parameter_health.get("status") == "warning" and overall_status != "critical":
                        overall_status = "warning"
            
            return {
                "status": "success",
                "message": f"System health check completed: {overall_status}",
                "details": {
                    "timestamp": datetime.now().isoformat(),
                    "overall_status": overall_status,
                    "subsystems": health_results,
                    "recommendations": self._generate_health_recommendations(health_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in check_system_health: {str(e)}")
            return {"status": "error", "message": str(e), "details": traceback.format_exc()}
    
    # ========== Helper Methods ==========
    
    def _record_tool_usage(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record the usage of a tool for later analysis."""
        self.tool_usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": args
        })
        
        # Keep history manageable - limit to last 1000 usages
        if len(self.tool_usage_history) > 1000:
            self.tool_usage_history = self.tool_usage_history[-1000:]
    
    def _generate_reflection_prompt(self, reflection_type: str, context: str, depth: float) -> str:
        """Generate a prompt for metacognitive reflection based on type and context."""
        base_prompt = f"Perform a {reflection_type} reflection at depth level {depth:.1f} on the following context:\n\n{context}\n\n"
        
        if reflection_type == "reasoning_quality":
            return base_prompt + (
                "Evaluate the quality of reasoning, identifying any logical fallacies, biases, or gaps. "
                "Consider whether conclusions follow from premises, if alternative perspectives were considered, "
                "and whether uncertainties were appropriately acknowledged. "
                "Suggest specific ways to improve reasoning processes."
            )
        elif reflection_type == "learning_progress":
            return base_prompt + (
                "Analyze learning progress, identifying what has been effectively learned, what gaps remain, "
                "and what strategies have been most effective for knowledge acquisition. "
                "Consider patterns in how new information is integrated with existing knowledge. "
                "Suggest specific areas for focused learning and optimal learning strategies."
            )
        elif reflection_type == "knowledge_gaps":
            return base_prompt + (
                "Identify significant knowledge gaps that may be limiting effectiveness. "
                "Consider both known unknowns (recognized gaps) and potential unknown unknowns. "
                "Assess the impact of these gaps on overall cognitive performance. "
                "Prioritize which gaps should be addressed first and suggest approaches for filling them."
            )
        elif reflection_type == "bias_analysis":
            return base_prompt + (
                "Analyze potential cognitive biases affecting processing and decision-making. "
                "Consider confirmation bias, anchoring, availability heuristic, and other common biases. "
                "Identify specific instances where biases may have influenced thinking. "
                "Suggest concrete debiasing strategies and mechanisms for more objective reasoning."
            )
        elif reflection_type == "goal_alignment":
            return base_prompt + (
                "Evaluate alignment between current activities and overarching goals/values. "
                "Consider whether actions and decisions are consistent with stated priorities. "
                "Identify any misalignments or goal conflicts that may be occurring. "
                "Suggest strategies for better aligning cognitive processes with core goals and values."
            )
        else:
            return base_prompt + (
                "Analyze the provided context thoroughly, identifying patterns, insights, and opportunities for improvement. "
                "Consider how this relates to overall cognitive development and suggest specific actionable next steps."
            )
    
    def _extract_next_steps(self, reflection_content: str) -> List[str]:
        """Extract action items or next steps from reflection content."""
        next_steps = []
        
        # Look for sections that indicate next steps or recommendations
        patterns = [
            r"(?:Next steps|Action items|Recommendations|Suggested actions):(.*?)(?:\n\n|\Z)",
            r"\d+\.\s+(.*?)(?=\d+\.|$)",
            r"[-•]\s+(.*?)(?=[-•]|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reflection_content, re.DOTALL)
            if matches:
                for match in matches:
                    steps = [s.strip() for s in match.strip().split("\n") if s.strip()]
                    next_steps.extend(steps)
        
        # If no structured steps found, look for sentences containing action verbs
        if not next_steps:
            action_verbs = ["should", "must", "need to", "recommend", "suggest", "implement", "develop", "improve"]
            sentences = [s.strip() for s in re.split(r'[.!?]', reflection_content) if s.strip()]
            
            for sentence in sentences:
                if any(verb in sentence.lower() for verb in action_verbs):
                    next_steps.append(sentence)
        
        # Limit to most relevant steps
        return next_steps[:5]
    
    def _generate_usage_summary(self, usage_data: List[Dict[str, Any]]) -> str:
        """Generate a summary of tool usage data for LLM analysis."""
        tool_counts = {}
        for usage in usage_data:
            tool_name = usage["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        summary = f"Tool Usage Summary (Past {len(usage_data)} uses):\n\n"
        
        # Add frequency data
        summary += "Tool Frequency:\n"
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"- {tool}: {count} uses ({count/len(usage_data)*100:.1f}%)\n"
        
        # Add sequence patterns if we have enough data
        if len(usage_data) > 5:
            tool_sequence = [u["tool_name"] for u in usage_data[-5:]]
            summary += f"\nRecent Tool Sequence: {' -> '.join(tool_sequence)}\n"
        
        # Add success rates if available
        success_rates = {}
        for usage in usage_data:
            if "result" in usage and "status" in usage["result"]:
                tool_name = usage["tool_name"]
                success = usage["result"]["status"] == "success"
                
                if tool_name not in success_rates:  success_rates[tool_name] = {"success": 0, "failure": 0}
                

                if success:
                    success_rates[tool_name]["success"] += 1
                else:
                    success_rates[tool_name]["failure"] += 1
async def _check_memory_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the memory system."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.memory_system:
            return {
                "status": "critical",
                "message": "Memory system not available",
                "metrics": {},
                "issues": ["Memory system not initialized"]
            }
        
        # Get memory statistics
        if hasattr(self.memory_system, "get_statistics"):
            stats = await self.memory_system.get_statistics()
            metrics["total_memories"] = stats.get("total_memories", 0)
            metrics["avg_significance"] = stats.get("average_significance", 0)
            
            # Check for warning conditions
            if metrics["total_memories"] > 10000:
                status = "warning"
                issues.append("High memory count may impact performance")
            
            # Memory usage metrics
            if "memory_usage_mb" in stats:
                metrics["memory_usage_mb"] = stats["memory_usage_mb"]
                if stats["memory_usage_mb"] > 1000:  # More than 1GB
                    status = "warning"
                    issues.append("High memory usage")
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check index health
            if hasattr(self.memory_system, "check_index_health"):
                index_health = await self.memory_system.check_index_health()
                metrics["index_health"] = index_health
                
                if not index_health.get("healthy", True):
                    status = "critical"
                    issues.append("Memory index corruption detected")
            
            # Check for memory fragmentation
            if hasattr(self.memory_system, "check_fragmentation"):
                frag = await self.memory_system.check_fragmentation()
                metrics["fragmentation"] = frag
                
                if frag > 0.3:  # More than 30% fragmentation
                    status = "warning"
                    issues.append("Memory fragmentation detected")
            
            # Detailed diagnostics
            if detail_level == "diagnostic":
                if hasattr(self.memory_system, "run_diagnostics"):
                    diagnostics = await self.memory_system.run_diagnostics()
                    metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Memory system health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking memory health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking memory health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

async def _check_knowledge_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the knowledge graph."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.knowledge_graph:
            return {
                "status": "critical",
                "message": "Knowledge graph not available",
                "metrics": {},
                "issues": ["Knowledge graph not initialized"]
            }
        
        # Basic metrics
        if hasattr(self.knowledge_graph, "get_statistics"):
            stats = await self.knowledge_graph.get_statistics()
            metrics["concept_count"] = stats.get("concept_count", 0)
            metrics["relationship_count"] = stats.get("relationship_count", 0)
            metrics["density"] = stats.get("density", 0)
            
            # Analyze graph metrics
            if metrics["concept_count"] > 0 and metrics["relationship_count"] / metrics["concept_count"] < 2:
                status = "warning"
                issues.append("Knowledge graph has low connectivity")
            
            if metrics["concept_count"] > 10000:
                status = "warning"
                issues.append("Very large knowledge graph may impact performance")
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check for orphaned concepts (no relationships)
            if hasattr(self.knowledge_graph, "find_orphaned_concepts"):
                orphaned = await self.knowledge_graph.find_orphaned_concepts(limit=10)
                metrics["orphaned_concepts"] = len(orphaned)
                
                if metrics["orphaned_concepts"] > 0:
                    status = "warning"
                    issues.append(f"Found {metrics['orphaned_concepts']} orphaned concepts")
            
            # Check knowledge graph consistency
            if hasattr(self.knowledge_graph, "verify_consistency"):
                consistency = await self.knowledge_graph.verify_consistency()
                metrics["consistency"] = consistency
                
                if not consistency.get("consistent", True):
                    status = "critical"
                    issues.append("Knowledge graph consistency issues detected")
            
            # Detailed diagnostics
            if detail_level == "diagnostic":
                if hasattr(self.knowledge_graph, "run_diagnostics"):
                    diagnostics = await self.knowledge_graph.run_diagnostics()
                    metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Knowledge graph health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking knowledge graph health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking knowledge graph health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

async def _check_self_model_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the self model."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.self_model:
            return {
                "status": "critical",
                "message": "Self model not available",
                "metrics": {},
                "issues": ["Self model not initialized"]
            }
        
        # Check basic attributes
        expected_attributes = ["personality", "goals", "capabilities"]
        missing_attributes = [attr for attr in expected_attributes if not hasattr(self.self_model, attr)]
        
        if missing_attributes:
            status = "warning"
            issues.append(f"Self model missing key attributes: {', '.join(missing_attributes)}")
        
        # Check for recent updates
        if hasattr(self.self_model, "last_updated"):
            last_updated = self.self_model.last_updated
            now = datetime.now()
            
            if isinstance(last_updated, str):
                try:
                    last_updated = datetime.fromisoformat(last_updated)
                except ValueError:
                    last_updated = None
            
            if last_updated:
                days_since_update = (now - last_updated).days
                metrics["days_since_update"] = days_since_update
                
                if days_since_update > 7:
                    status = "warning"
                    issues.append(f"Self model not updated in {days_since_update} days")
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check for reflections
            if hasattr(self.self_model, "metacognitive_reflections"):
                reflection_count = len(self.self_model.metacognitive_reflections)
                metrics["reflection_count"] = reflection_count
                
                # Check recent reflections
                recent_reflections = [r for r in self.self_model.metacognitive_reflections 
                                     if (datetime.now() - datetime.fromisoformat(r["timestamp"])).days < 3]
                metrics["recent_reflections"] = len(recent_reflections)
                
                if len(recent_reflections) == 0:
                    status = "warning"
                    issues.append("No recent metacognitive reflections")
            
            # Check emotional state
            if hasattr(self.self_model, "emotional_state"):
                metrics["emotional_dimensions"] = len(self.self_model.emotional_state)
                
                # Check for emotional imbalance (e.g., extremely high negative emotions)
                negative_emotions = ["stress", "anxiety", "frustration", "exhaustion"]
                high_negative = any(self.self_model.emotional_state.get(e, 0) > 0.8 
                                   for e in negative_emotions if e in self.self_model.emotional_state)
                
                if high_negative:
                    status = "warning"
                    issues.append("High negative emotional state detected")
            
            # Detailed diagnostics
            if detail_level == "diagnostic" and hasattr(self.self_model, "run_diagnostics"):
                diagnostics = await self.self_model.run_diagnostics()
                metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Self model health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking self model health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking self model health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

async def _check_dream_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the dream processor."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.dream_processor:
            return {
                "status": "critical",
                "message": "Dream processor not available",
                "metrics": {},
                "issues": ["Dream processor not initialized"]
            }
        
        # Check current status
        current_status = await self.dream_processor.get_status()
        metrics["is_dreaming"] = current_status.get("is_dreaming", False)
        metrics["is_scheduled"] = current_status.get("has_scheduled_dreams", False)
        
        # Check for dream problems
        if current_status.get("consecutive_failures", 0) > 3:
            status = "critical"
            issues.append(f"Multiple consecutive dream failures: {current_status.get('consecutive_failures')}")
            metrics["consecutive_failures"] = current_status.get("consecutive_failures")
        
        # Check recent dreams
        if hasattr(self.dream_processor, "get_recent_dreams"):
            recent_dreams = await self.dream_processor.get_recent_dreams(limit=5)
            metrics["recent_dream_count"] = len(recent_dreams)
            
            # Check success rate
            successful_dreams = [d for d in recent_dreams if d.get("status") == "completed"]
            if recent_dreams:
                success_rate = len(successful_dreams) / len(recent_dreams)
                metrics["dream_success_rate"] = success_rate
                
                if success_rate < 0.5:
                    status = "warning"
                    issues.append(f"Low dream success rate: {success_rate:.2f}")
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check dream statistics
            if hasattr(self.dream_processor, "get_dream_statistics"):
                stats = await self.dream_processor.get_dream_statistics()
                metrics.update(stats)
                
                # Check for concerning patterns
                if stats.get("avg_duration_minutes", 0) > 30:
                    issues.append("Dreams taking too long on average")
                
                if stats.get("total_dreams", 0) > 0 and stats.get("integration_rate", 1) < 0.3:
                    status = "warning"
                    issues.append("Low dream integration rate")
            
            # Check dream queue health
            if hasattr(self.dream_processor, "check_queue_health"):
                queue_health = await self.dream_processor.check_queue_health()
                metrics["queue_health"] = queue_health
                
                if not queue_health.get("healthy", True):
                    status = "warning"
                    issues.append("Dream queue issues detected")
            
            # Detailed diagnostics
            if detail_level == "diagnostic" and hasattr(self.dream_processor, "run_diagnostics"):
                diagnostics = await self.dream_processor.run_diagnostics()
                metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Dream processor health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking dream health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking dream health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

async def _check_spiral_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the spiral manager."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.spiral_manager:
            return {
                "status": "critical",
                "message": "Spiral manager not available",
                "metrics": {},
                "issues": ["Spiral manager not initialized"]
            }
        
        # Check current phase
        current_phase = await self.spiral_manager.get_current_phase()
        metrics["current_phase"] = current_phase.get("phase")
        metrics["phase_duration"] = current_phase.get("duration_minutes")
        
        # Check for phase stagnation
        if current_phase.get("duration_minutes", 0) > 1440:  # More than 24 hours
            status = "warning"
            issues.append(f"Current phase stagnant for over 24 hours: {metrics['phase_duration']} minutes")
        
        # Check phase transition history
        if hasattr(self.spiral_manager, "get_phase_history"):
            phase_history = await self.spiral_manager.get_phase_history(limit=10)
            metrics["phase_transitions"] = len(phase_history)
            
            # Check transition frequency
            if phase_history:
                # Calculate average time between transitions
                if len(phase_history) > 1:
                    first_date = datetime.fromisoformat(phase_history[-1]["timestamp"])
                    last_date = datetime.fromisoformat(phase_history[0]["timestamp"])
                    days_span = (last_date - first_date).total_seconds() / (60*60*24)
                    
                    if days_span > 0:
                        transitions_per_day = len(phase_history) / days_span
                        metrics["transitions_per_day"] = transitions_per_day
                        
                        if transitions_per_day < 2:
                            status = "warning"
                            issues.append("Low spiral transition rate")
                        elif transitions_per_day > 20:
                            status = "warning"
                            issues.append("Excessive spiral transition rate")
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check spiral progress
            if hasattr(self.spiral_manager, "get_progress_metrics"):
                progress = await self.spiral_manager.get_progress_metrics()
                metrics["progression_metrics"] = progress
                
                # Check for progression issues
                if progress.get("insight_rate", 1) < 0.2:
                    status = "warning"
                    issues.append("Low insight generation rate")
                
                if progress.get("stagnation_score", 0) > 0.7:
                    status = "warning"
                    issues.append("High cognitive stagnation detected")
            
            # Check for insights
            if hasattr(self.spiral_manager, "get_recent_insights"):
                insights = await self.spiral_manager.get_recent_insights(days=3)
                metrics["recent_insights"] = len(insights)
                
                if len(insights) == 0:
                    status = "warning"
                    issues.append("No recent insights generated")
            
            # Detailed diagnostics
            if detail_level == "diagnostic" and hasattr(self.spiral_manager, "run_diagnostics"):
                diagnostics = await self.spiral_manager.run_diagnostics()
                metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Spiral manager health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking spiral health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking spiral health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

async def _check_parameter_health(self, detail_level: str) -> Dict[str, Any]:
    """Check the health of the parameter manager."""
    try:
        status = "healthy"
        metrics = {}
        issues = []
        
        # Basic availability check
        if not self.parameter_manager:
            return {
                "status": "critical",
                "message": "Parameter manager not available",
                "metrics": {},
                "issues": ["Parameter manager not initialized"]
            }
        
        # Check critical parameters
        critical_params = [
            "dream_cycles.auto_enabled",
            "dream_cycles.idle_threshold",
            "spiral.min_phase_duration",
            "memory.consolidation_threshold",
            "llm.token_budget"
        ]
        
        missing_params = []
        invalid_params = []
        
        for param in critical_params:
            # Check if parameter exists
            exists = await self.parameter_manager.parameter_exists(param)
            if not exists:
                missing_params.append(param)
                continue
            
            # Check if valid (if validation supported)
            if hasattr(self.parameter_manager, "validate_parameter"):
                value = await self.parameter_manager.get_parameter(param)
                valid = await self.parameter_manager.validate_parameter(param, value)
                
                if not valid:
                    invalid_params.append(param)
        
        if missing_params:
            status = "warning"
            issues.append(f"Missing critical parameters: {', '.join(missing_params)}")
            metrics["missing_parameters"] = missing_params
        
        if invalid_params:
            status = "critical"
            issues.append(f"Invalid critical parameters: {', '.join(invalid_params)}")
            metrics["invalid_parameters"] = invalid_params
        
        # Additional detailed checks
        if detail_level in ["detailed", "diagnostic"]:
            # Check for locked parameters
            if hasattr(self.parameter_manager, "get_locked_parameters"):
                locked = await self.parameter_manager.get_locked_parameters()
                metrics["locked_parameters"] = locked
                
                if len(locked) > 5:
                    status = "warning"
                    issues.append("Many parameters are locked: {len(locked)}")
            
            # Check parameter changes
            if hasattr(self.parameter_manager, "get_parameter_changes"):
                recent_changes = await self.parameter_manager.get_parameter_changes(days=1)
                metrics["recent_parameter_changes"] = len(recent_changes)
                
                if len(recent_changes) > 20:
                    status = "warning"
                    issues.append("High rate of parameter changes")
            
            # Check for transitioning parameters
            if hasattr(self.parameter_manager, "get_transitioning_parameters"):
                transitioning = await self.parameter_manager.get_transitioning_parameters()
                metrics["transitioning_parameters"] = len(transitioning)
            
            # Detailed diagnostics
            if detail_level == "diagnostic" and hasattr(self.parameter_manager, "run_diagnostics"):
                diagnostics = await self.parameter_manager.run_diagnostics()
                metrics["diagnostics"] = diagnostics
        
        return {
            "status": status,
            "message": f"Parameter manager health: {status}",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error checking parameter health: {str(e)}")
        return {
            "status": "critical",
            "message": f"Error checking parameter health: {str(e)}",
            "metrics": {},
            "issues": [str(e)]
        }

def _generate_health_recommendations(self, health_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on health check results."""
    recommendations = []
    
    # Process issues from each subsystem
    for subsystem, result in health_results.items():
        if result.get("status") in ["warning", "critical"]:
            for issue in result.get("issues", []):
                # Generate recommendation based on issue
                if "memory" in subsystem and "high memory usage" in issue.lower():
                    recommendations.append("Run memory consolidation to reduce memory usage")
                
                elif "memory" in subsystem and "index corruption" in issue.lower():
                    recommendations.append("Rebuild memory indexes to address corruption")
                
                elif "knowledge" in subsystem and "orphaned concepts" in issue.lower():
                    recommendations.append("Run knowledge graph cleanup to connect orphaned concepts")
                
                elif "knowledge" in subsystem and "connectivity" in issue.lower():
                    recommendations.append("Enhance knowledge graph connectivity through relationship inference")
                
                elif "self_model" in subsystem and "negative emotional state" in issue.lower():
                    recommendations.append("Schedule reflective activities to address high negative emotions")
                
                elif "self_model" in subsystem and "no recent metacognitive" in issue.lower():
                    recommendations.append("Trigger metacognitive reflection to update self-awareness")
                
                elif "dream" in subsystem and "consecutive failures" in issue.lower():
                    recommendations.append("Reset dream processor and verify dream integration processes")
                
                elif "dream" in subsystem and "low success rate" in issue.lower():
                    recommendations.append("Reduce dream complexity and enhance dream seed quality")
                
                elif "spiral" in subsystem and "stagnant" in issue.lower():
                    recommendations.append("Force spiral phase transition to overcome stagnation")
                
                elif "spiral" in subsystem and "low insight" in issue.lower():
                    recommendations.append("Schedule focused learning sessions to generate new insights")
                
                elif "parameter" in subsystem and "missing" in issue.lower():
                    recommendations.append("Initialize default values for missing critical parameters")
                
                elif "parameter" in subsystem and "invalid" in issue.lower():
                    recommendations.append("Reset invalid parameters to their default values")
                
                # Generic recommendation if no specific one was generated
                else:
                    recommendations.append(f"Investigate and address: {issue}")
    
    # Add general recommendations if no specific ones were found
    if not recommendations and any(r.get("status") != "healthy" for r in health_results.values()):
        recommendations.append("Run detailed diagnostics on all subsystems showing warnings or critical issues")
        recommendations.append("Check system logs for error patterns not captured by health checks")
    
    # Deduplicate and limit recommendations
    unique_recommendations = list(set(recommendations))
    return unique_recommendations[:7]  # Limit to top 7 recommendations

                    