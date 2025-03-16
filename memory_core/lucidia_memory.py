# memory_core/lucidia_memory.py

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio

# Import Lucidia memory system components
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel
from memory.lucidia_memory_system.narrative_identity.identity_manager import NarrativeIdentityManager

# Configure logger
logger = logging.getLogger(__name__)

class LucidiaMemorySystemMixin:
    """
    Mixin for integrating Lucidia's advanced memory system components:
    - Knowledge Graph: Represents interconnected concepts and relationships
    - World Model: Models external reality, knowledge structures, and conceptual relationships
    - Self Model: Implements core identity, self-reflection, and adaptive behavior
    - Narrative Identity Manager: Manages the narrative identity of the system
    
    These components work together to provide a sophisticated memory and knowledge
    representation system for enhanced contextual understanding and reasoning.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Lucidia memory system components.
        
        Args:
            **kwargs: Additional configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Lucidia Memory System components")
        
        # Get configuration options
        lucidia_config = kwargs.get('lucidia_config', {})
        
        # Initialize the Self Model first (as it doesn't depend on the others)
        self.self_model = LucidiaSelfModel(config=lucidia_config.get('self_model', {}))
        
        # Initialize the World Model (can reference the Self Model)
        self.world_model = LucidiaWorldModel(
            self_model=self.self_model,
            config=lucidia_config.get('world_model', {})
        )
        
        # Initialize the Knowledge Graph (references both Self and World models)
        self.knowledge_graph = LucidiaKnowledgeGraph(
            self_model=self.self_model,
            world_model=self.world_model,
            config=lucidia_config.get('knowledge_graph', {})
        )
        
        # Initialize the Narrative Identity Manager (references all other components)
        self.narrative_identity = NarrativeIdentityManager(
            memory_system=kwargs.get('memory_system', None),
            knowledge_graph=self.knowledge_graph,
            dream_manager=None  # For now, we don't have a dream manager
        )
        
        # State tracking
        self._lucidia_initialized = True
        self.logger.info("Lucidia Memory System components initialized successfully")
    
    async def process_for_lucidia_memory(self, text: str, role: str = "user") -> None:
        """
        Process incoming messages through Lucidia's memory system components.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
        """
        try:
            # Process through self model for identity and self-awareness components
            if hasattr(self.self_model, 'process_message'):
                await asyncio.to_thread(self.self_model.process_message, text, role)
            
            # Process through world model for external knowledge and concepts
            if hasattr(self.world_model, 'process_message'):
                await asyncio.to_thread(self.world_model.process_message, text, role)
                
            # Process through knowledge graph to update relationships and insights
            if hasattr(self.knowledge_graph, 'process_message'):
                await asyncio.to_thread(self.knowledge_graph.process_message, text, role)
                
            # Process through narrative identity manager
            if hasattr(self.narrative_identity, 'process_message'):
                await asyncio.to_thread(self.narrative_identity.process_message, text, role)
                
            self.logger.debug(f"Processed message through Lucidia memory system: {text[:50]}...")
        except Exception as e:
            self.logger.error(f"Error processing message through Lucidia memory system: {e}")
    
    async def get_lucidia_insights(self, query: str) -> Dict[str, Any]:
        """
        Get insights from Lucidia's memory system components.
        
        Args:
            query: The query to get insights for
            
        Returns:
            Dictionary containing insights from all components
        """
        insights = {
            "self_model": {},
            "world_model": {},
            "knowledge_graph": {},
            "narrative_identity": {}
        }
        
        try:
            # Get insights from self model
            if hasattr(self.self_model, 'get_insights'):
                self_insights = await asyncio.to_thread(self.self_model.get_insights, query)
                insights["self_model"] = self_insights
            
            # Get insights from world model
            if hasattr(self.world_model, 'get_insights'):
                world_insights = await asyncio.to_thread(self.world_model.get_insights, query)
                insights["world_model"] = world_insights
                
            # Get insights from knowledge graph
            if hasattr(self.knowledge_graph, 'get_insights'):
                kg_insights = await asyncio.to_thread(self.knowledge_graph.get_insights, query)
                insights["knowledge_graph"] = kg_insights
                
            # Get insights from narrative identity manager
            if hasattr(self.narrative_identity, 'get_insights'):
                narrative_insights = await asyncio.to_thread(self.narrative_identity.get_insights, query)
                insights["narrative_identity"] = narrative_insights
                
            self.logger.debug(f"Retrieved Lucidia insights for query: {query[:50]}...")
        except Exception as e:
            self.logger.error(f"Error retrieving Lucidia insights: {e}")
            
        return insights
    
    async def get_lucidia_rag_context(self, query: str, max_tokens: int = 1024) -> str:
        """
        Get enhanced RAG context using Lucidia's memory system components.
        
        Args:
            query: The query to get context for
            max_tokens: Maximum tokens for the context
            
        Returns:
            String containing formatted context from Lucidia components
        """
        context_parts = []
        
        try:
            # Get Self Model context
            if hasattr(self.self_model, 'generate_context'):
                self_context = await asyncio.to_thread(self.self_model.generate_context, query)
                if self_context:
                    context_parts.append("### Lucidia Self Insights")
                    context_parts.append(self_context)
            
            # Get World Model context
            if hasattr(self.world_model, 'generate_context'):
                world_context = await asyncio.to_thread(self.world_model.generate_context, query)
                if world_context:
                    context_parts.append("### Lucidia World Knowledge")
                    context_parts.append(world_context)
            
            # Get Knowledge Graph context
            if hasattr(self.knowledge_graph, 'generate_context'):
                kg_context = await asyncio.to_thread(self.knowledge_graph.generate_context, query)
                if kg_context:
                    context_parts.append("### Lucidia Knowledge Insights")
                    context_parts.append(kg_context)
                    
            # Get Narrative Identity Manager context
            if hasattr(self.narrative_identity, 'generate_context'):
                narrative_context = await asyncio.to_thread(self.narrative_identity.generate_context, query)
                if narrative_context:
                    context_parts.append("### Lucidia Narrative Identity")
                    context_parts.append(narrative_context)
                    
            self.logger.debug(f"Generated Lucidia RAG context for query: {query[:50]}...")
        except Exception as e:
            self.logger.error(f"Error generating Lucidia RAG context: {e}")
        
        return "\n\n".join(context_parts) if context_parts else ""

    async def get_lucidia_memory_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools for accessing Lucidia memory system components.
        
        Returns:
            List of tool definitions formatted for LLM access
        """
        tools = []
        
        # Knowledge graph query tool
        tools.append({
            "type": "function",
            "function": {
                "name": "query_knowledge_graph",
                "description": "Query Lucidia's knowledge graph for concepts, relationships and insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for in the knowledge graph"
                        },
                        "concept_type": {
                            "type": "string",
                            "description": "Optional type of concept to filter by (e.g., 'person', 'place', 'event', 'idea')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        
        # Self model reflection tool
        tools.append({
            "type": "function",
            "function": {
                "name": "self_model_reflection",
                "description": "Access Lucidia's self-awareness and introspection capabilities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection_type": {
                            "type": "string",
                            "description": "Type of reflection to perform (identity, values, capabilities, limitations, etc.)",
                            "enum": ["identity", "values", "capabilities", "limitations", "goals", "general"]
                        }
                    },
                    "required": ["reflection_type"]
                }
            }
        })
        
        # World model insight tool
        tools.append({
            "type": "function",
            "function": {
                "name": "world_model_insight",
                "description": "Retrieve Lucidia's understanding of concepts, domains, and external reality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "The concept or domain to retrieve information about"
                        },
                        "perspective": {
                            "type": "string",
                            "description": "Optional perspective to view the concept from (objective, subjective, analytical, etc.)",
                            "enum": ["objective", "subjective", "analytical", "emotional", "ethical", "general"]
                        }
                    },
                    "required": ["concept"]
                }
            }
        })
        
        # Narrative identity manager tool
        tools.append({
            "type": "function",
            "function": {
                "name": "narrative_identity_insight",
                "description": "Retrieve Lucidia's narrative identity insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to retrieve narrative identity insights for"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        
        return tools
        
    async def handle_lucidia_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool call to one of the Lucidia memory system components.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Parameters for the tool call
            
        Returns:
            Result of the tool call
        """
        result = {"result": None, "error": None}
        
        try:
            if tool_name == "query_knowledge_graph":
                if hasattr(self.knowledge_graph, 'query'):
                    query = parameters.get("query", "")
                    concept_type = parameters.get("concept_type", None)
                    max_results = parameters.get("max_results", 5)
                    
                    kg_result = await asyncio.to_thread(
                        self.knowledge_graph.query,
                        query=query,
                        concept_type=concept_type,
                        max_results=max_results
                    )
                    result["result"] = kg_result
                    
            elif tool_name == "self_model_reflection":
                if hasattr(self.self_model, 'reflect'):
                    reflection_type = parameters.get("reflection_type", "general")
                    
                    reflection = await asyncio.to_thread(
                        self.self_model.reflect,
                        reflection_type=reflection_type
                    )
                    result["result"] = reflection
                    
            elif tool_name == "world_model_insight":
                if hasattr(self.world_model, 'get_concept_insight'):
                    concept = parameters.get("concept", "")
                    perspective = parameters.get("perspective", "general")
                    
                    insight = await asyncio.to_thread(
                        self.world_model.get_concept_insight,
                        concept=concept,
                        perspective=perspective
                    )
                    result["result"] = insight
                    
            elif tool_name == "narrative_identity_insight":
                if hasattr(self.narrative_identity, 'get_insight'):
                    query = parameters.get("query", "")
                    
                    insight = await asyncio.to_thread(
                        self.narrative_identity.get_insight,
                        query=query
                    )
                    result["result"] = insight
            
            else:
                result["error"] = f"Unknown Lucidia tool: {tool_name}"
                
        except Exception as e:
            self.logger.error(f"Error handling Lucidia tool call {tool_name}: {e}")
            result["error"] = str(e)
            
        return result
