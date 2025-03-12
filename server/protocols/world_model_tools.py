from typing import Dict, Any, List, Optional, Union
import logging
import json
import asyncio
import re
from datetime import datetime

from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class WorldModelToolProvider(ToolProvider):
    """Tool provider specialized for world model functions in Lucidia's architecture."""
    
    def __init__(self, world_model=None, self_model=None, knowledge_graph=None, 
                 memory_system=None, parameter_manager=None, model_manager=None):
        super().__init__()
        self.world_model = world_model
        self.self_model = self_model
        self.knowledge_graph = knowledge_graph
        self.memory_system = memory_system
        self.parameter_manager = parameter_manager
        self.model_manager = model_manager
        self.register_world_model_tools()
    
    def register_world_model_tools(self):
        """Register world model-specific tools."""
        # Entity extraction tool
        self.register_tool(
            name="extract_entities",
            function=self.extract_entities,
            description="Extract entities (people, places, concepts, objects) from provided text. "
                      "This tool analyzes text to identify important entities and their attributes.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract entities from"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of entities to extract (person, location, organization, concept, object, event)",
                        "default": ["person", "location", "organization", "concept", "object", "event"]
                    },
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Whether to extract attributes for each entity",
                        "default": True
                    }
                },
                "required": ["text"],
            }
        )
        
        # Relationship inference tool
        self.register_tool(
            name="infer_relationships",
            function=self.infer_relationships,
            description="Infer relationships between entities based on context. "
                      "Identifies connections, relation types, and confidence levels.",
            parameters={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entities to analyze for relationships"
                    },
                    "context": {
                        "type": "string",
                        "description": "Context information to help infer relationships"
                    },
                    "max_relationships": {
                        "type": "integer",
                        "description": "Maximum number of relationships to infer",
                        "default": 10
                    }
                },
                "required": ["entities", "context"],
            }
        )
        
        # Entity update tool
        self.register_tool(
            name="update_entity_model",
            function=self.update_entity_model,
            description="Update or create an entity in the world model with new information. "
                      "Use this to evolve Lucidia's understanding of entities over time.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the entity to update"
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Type of entity (person, location, organization, concept, object, event)",
                        "enum": ["person", "location", "organization", "concept", "object", "event"]
                    },
                    "attributes": {
                        "type": "object",
                        "description": "Key-value pairs of entity attributes to update"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level in this entity information (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8
                    }
                },
                "required": ["entity_name", "entity_type", "attributes"],
            }
        )
        
        # Causal reasoning tool
        self.register_tool(
            name="perform_causal_reasoning",
            function=self.perform_causal_reasoning,
            description="Perform causal reasoning to determine relationships between events or actions. "
                      "Identifies cause-effect relationships and their strength.",
            parameters={
                "type": "object",
                "properties": {
                    "event_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sequence of events to analyze for causal relationships"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context to inform causal reasoning"
                    },
                    "reasoning_depth": {
                        "type": "integer",
                        "description": "Depth of causal chain to analyze (1-5)",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 2
                    }
                },
                "required": ["event_sequence"],
            }
        )
        
        # Knowledge graph query tool
        self.register_tool(
            name="query_knowledge_graph",
            function=self.query_knowledge_graph,
            description="Query the knowledge graph for information about entities, relationships, or concepts. "
                      "Returns structured information from Lucidia's knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search the knowledge graph"
                    },
                    "entity_focus": {
                        "type": "string",
                        "description": "Optional entity to focus the query on"
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional specific relationship types to query for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} world model tools")
    
    # ========== Tool Implementation Methods ==========
    
    async def extract_entities(self, text: str, entity_types: List[str] = None, 
                         include_attributes: bool = True) -> Dict[str, Any]:
        """Extract entities from provided text."""
        try:
            if not text:
                return {
                    "status": "error",
                    "message": "No text provided",
                    "entities": []
                }
            
            if entity_types is None:
                entity_types = ["person", "location", "organization", "concept", "object", "event"]
                
            # First try to use world model's entity extractor if available
            if self.world_model and hasattr(self.world_model, "extract_entities"):
                try:
                    entities = await self.world_model.extract_entities(
                        text=text, 
                        entity_types=entity_types,
                        include_attributes=include_attributes
                    )
                    return {
                        "status": "success",
                        "entities": entities,
                        "count": len(entities)
                    }
                except Exception as e:
                    logger.warning(f"World model entity extraction failed: {str(e)}. Falling back to LLM.")
            
            # Fall back to LLM-based entity extraction
            if self.model_manager:
                entity_types_str = ", ".join(entity_types)
                attributes_instruction = "For each entity, also identify key attributes and properties." if include_attributes else ""
                
                messages = [
                    {"role": "system", "content": "You are an entity extraction system for Lucidia's world model. "
                                            "Your task is to identify and classify entities mentioned in text, "
                                            "extracting relevant information about each entity."},
                    {"role": "user", "content": f"Text: {text}\n\n"
                                            f"Entity types to extract: {entity_types_str}\n\n"
                                            f"{attributes_instruction}\n\n"
                                            f"Please extract entities from the text and return them in a structured format. "
                                            f"For each entity, provide its name, type, and any relevant attributes or "
                                            f"properties mentioned in the text. Format your response as a list of entities."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.3,    # Lower temperature for more reliable extraction
                    max_tokens=1000,    # Allow for thorough entity extraction
                    timeout=45          # Give enough time for complex text
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    extraction_text = response["choices"][0]["message"].get("content", "")
                    
                    # Process extraction text to structured format
                    entities = []
                    
                    # Extract entities from the response using regex patterns
                    entity_blocks = re.split(r'\n\s*\d+\.\s*|\n\s*-|\n\s*\*', extraction_text)
                    for block in entity_blocks:
                        if not block.strip():
                            continue
                            
                        entity = {}
                        
                        # Look for entity name and type
                        name_match = re.search(r'(?:Entity|Name)[:\s]*([^\n]+)', block, re.IGNORECASE)
                        if name_match:
                            entity["name"] = name_match.group(1).strip()
                        else:
                            # Try to get the first line as the name
                            first_line = block.strip().split('\n')[0].strip()
                            if first_line and not first_line.lower().startswith(('entity', 'name', 'type')):
                                entity["name"] = first_line
                        
                        # Look for entity type
                        type_match = re.search(r'[Tt]ype[:\s]*([^\n,]+)', block)
                        if type_match:
                            entity["type"] = type_match.group(1).strip().lower()
                        
                        # Default type if not found
                        if "name" in entity and "type" not in entity:
                            # Try to infer type from context
                            for t in entity_types:
                                if t.lower() in block.lower():
                                    entity["type"] = t.lower()
                                    break
                            if "type" not in entity:
                                entity["type"] = "unknown"
                        
                        # Extract attributes if requested
                        if include_attributes and "name" in entity:
                            attributes = {}
                            # Look for attribute sections
                            attr_section = re.search(r'[Aa]ttributes[:\s]*(.+?)(?:\n\n|$)', block, re.DOTALL)
                            if attr_section:
                                attr_text = attr_section.group(1)
                                # Parse key-value pairs
                                attr_pairs = re.finditer(r'([\w\s]+)[:\s]*([^,\n]+)(?:,|\n|$)', attr_text)
                                for pair in attr_pairs:
                                    key = pair.group(1).strip().lower()
                                    value = pair.group(2).strip()
                                    if key and value and key not in ('name', 'type'):
                                        attributes[key] = value
                            
                            # If no structured attributes found, extract them from whole block
                            if not attributes:
                                attr_pairs = re.finditer(r'([\w\s]+)[:\s]*([^,\n]+)(?:,|\n|$)', block)
                                for pair in attr_pairs:
                                    key = pair.group(1).strip().lower()
                                    value = pair.group(2).strip()
                                    if key and value and key not in ('name', 'type', 'entity', 'attributes'):
                                        attributes[key] = value
                            
                            if attributes:
                                entity["attributes"] = attributes
                        
                        # Add to entities if it has at least a name
                        if "name" in entity and entity["name"]:
                            entities.append(entity)
                    
                    return {
                        "status": "success",
                        "entities": entities,
                        "count": len(entities)
                    }
            
            # Fallback if extraction failed
            return {
                "status": "error",
                "message": "Entity extraction failed",
                "entities": []
            }
        except Exception as e:
            logger.error(f"Error in extract_entities: {str(e)}")
            return {"status": "error", "message": str(e), "entities": []}
    
    async def infer_relationships(self, entities: List[str], context: str, 
                            max_relationships: int = 10) -> Dict[str, Any]:
        """Infer relationships between entities based on context."""
        try:
            if not entities or len(entities) < 2:
                return {
                    "status": "error",
                    "message": "At least two entities are required to infer relationships",
                    "relationships": []
                }
            
            # Try to use world model's relationship inference if available
            if self.world_model and hasattr(self.world_model, "infer_relationships"):
                try:
                    relationships = await self.world_model.infer_relationships(
                        entities=entities,
                        context=context,
                        max_relationships=max_relationships
                    )
                    return {
                        "status": "success",
                        "relationships": relationships,
                        "count": len(relationships)
                    }
                except Exception as e:
                    logger.warning(f"World model relationship inference failed: {str(e)}. Falling back to LLM.")
            
            # Fall back to LLM-based relationship inference
            if self.model_manager:
                entities_text = ", ".join(entities)
                
                messages = [
                    {"role": "system", "content": "You are a relationship inference system for Lucidia's world model. "
                                            "Your task is to identify potential relationships between entities "
                                            "based on the provided context, indicating the type of relationship "
                                            "and confidence level for each."},
                    {"role": "user", "content": f"Entities: {entities_text}\n\n"
                                            f"Context: {context}\n\n"
                                            f"Please infer up to {max_relationships} relationships between these entities "
                                            f"based on the provided context. For each relationship, specify the source entity, "
                                            f"target entity, relationship type, and a confidence level (0-1). "}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.4,    # Lower temperature for more reliable inference
                    max_tokens=1000,    # Allow for thorough relationship analysis
                    timeout=45          # Give enough time for complex reasoning
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    inference_text = response["choices"][0]["message"].get("content", "")
                    
                    # Process inference text to structured format
                    relationships = []
                    
                    # Extract relationships from the response using regex patterns
                    relationship_blocks = re.split(r'\n\s*\d+\.\s*|\n\s*-|\n\s*\*', inference_text)
                    for block in relationship_blocks:
                        if not block.strip():
                            continue
                            
                        relationship = {}
                        
                        # Look for source, target, type, and confidence
                        source_match = re.search(r'[Ss]ource[:\s]*([^\n]+)', block)
                        target_match = re.search(r'[Tt]arget[:\s]*([^\n]+)', block)
                        type_match = re.search(r'(?:[Rr]elationship)?\s*[Tt]ype[:\s]*([^\n,]+)', block)
                        confidence_match = re.search(r'[Cc]onfidence[:\s]*([0-9]\.[0-9]+)', block)
                        
                        # Process direct relationship expression (Entity1 -> Relationship -> Entity2)
                        direct_match = re.search(r'([^\n:]+)\s*(?:->|→|–)\s*([^\n:]+)\s*(?:->|→|–)\s*([^\n:]+)', block)
                        
                        if source_match and target_match:
                            relationship["source"] = source_match.group(1).strip()
                            relationship["target"] = target_match.group(1).strip()
                        elif direct_match:
                            relationship["source"] = direct_match.group(1).strip()
                            relationship["type"] = direct_match.group(2).strip()
                            relationship["target"] = direct_match.group(3).strip()
                        
                        # Get type if not already found
                        if "type" not in relationship and type_match:
                            relationship["type"] = type_match.group(1).strip()
                        
                        # Get confidence
                        if confidence_match:
                            try:
                                confidence = float(confidence_match.group(1))
                                relationship["confidence"] = min(max(confidence, 0.0), 1.0)  # Clamp to 0-1
                            except ValueError:
                                relationship["confidence"] = 0.5  # Default confidence
                        else:
                            relationship["confidence"] = 0.5  # Default confidence
                        
                        # Add to relationships if it has at least source and target
                        if "source" in relationship and "target" in relationship:
                            if "type" not in relationship:
                                relationship["type"] = "related"  # Default type
                            relationships.append(relationship)
                            
                            # Stop if we've reached max_relationships
                            if len(relationships) >= max_relationships:
                                break
                    
                    return {
                        "status": "success",
                        "relationships": relationships,
                        "count": len(relationships)
                    }
            
            # Fallback if inference failed
            return {
                "status": "error",
                "message": "Relationship inference failed",
                "relationships": []
            }
        except Exception as e:
            logger.error(f"Error in infer_relationships: {str(e)}")
            return {"status": "error", "message": str(e), "relationships": []}
    
    async def update_entity_model(self, entity_name: str, entity_type: str, 
                           attributes: Dict[str, Any], confidence: float = 0.8) -> Dict[str, Any]:
        """Update or create an entity in the world model with new information."""
        try:
            if not entity_name or not entity_type:
                return {
                    "status": "error",
                    "message": "Entity name and type are required",
                    "updated": False
                }
            
            # Try to use world model's entity update if available
            if self.world_model and hasattr(self.world_model, "update_entity"):
                try:
                    result = await self.world_model.update_entity(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        attributes=attributes,
                        confidence=confidence
                    )
                    return {
                        "status": "success",
                        "updated": True,
                        "entity": {
                            "name": entity_name,
                            "type": entity_type,
                            "attributes": attributes
                        },
                        **result
                    }
                except Exception as e:
                    logger.warning(f"World model entity update failed: {str(e)}. Falling back to memory system.")
            
            # Fall back to memory system if available
            if self.memory_system and hasattr(self.memory_system, "store_entity"):
                try:
                    # Store entity in memory system
                    entity_data = {
                        "name": entity_name,
                        "type": entity_type,
                        "attributes": attributes,
                        "confidence": confidence,
                        "updated": datetime.now().isoformat()
                    }
                    
                    memory_id = await self.memory_system.store_entity(
                        entity_data=entity_data,
                        significance=confidence
                    )
                    
                    return {
                        "status": "success",
                        "updated": True,
                        "memory_id": memory_id,
                        "entity": entity_data
                    }
                except Exception as e:
                    logger.warning(f"Memory system entity storage failed: {str(e)}.")
            
            # If both approaches failed, return minimal success
            # This ensures the called code can continue, but logs the issue
            logger.warning(f"No persistent storage available for entity update: {entity_name}")
            return {
                "status": "partial_success",
                "updated": False,
                "message": "Entity processed but not persistently stored",
                "entity": {
                    "name": entity_name,
                    "type": entity_type,
                    "attributes": attributes
                }
            }
        except Exception as e:
            logger.error(f"Error in update_entity_model: {str(e)}")
            return {"status": "error", "message": str(e), "updated": False}
    
    async def perform_causal_reasoning(self, event_sequence: List[str], context: str = "", 
                               reasoning_depth: int = 2) -> Dict[str, Any]:
        """Perform causal reasoning to determine relationships between events or actions."""
        try:
            if not event_sequence or len(event_sequence) < 2:
                return {
                    "status": "error",
                    "message": "At least two events are required for causal reasoning",
                    "causal_chains": []
                }
            
            # Ensure reasonable reasoning depth
            reasoning_depth = max(1, min(reasoning_depth, 5))  # Clamp to 1-5
            
            # Try to use world model's causal reasoning if available
            if self.world_model and hasattr(self.world_model, "perform_causal_reasoning"):
                try:
                    causal_chains = await self.world_model.perform_causal_reasoning(
                        event_sequence=event_sequence,
                        context=context,
                        reasoning_depth=reasoning_depth
                    )
                    return {
                        "status": "success",
                        "causal_chains": causal_chains
                    }
                except Exception as e:
                    logger.warning(f"World model causal reasoning failed: {str(e)}. Falling back to LLM.")
            
            # Fall back to LLM-based causal reasoning
            if self.model_manager:
                events_text = "\n".join([f"{i+1}. {event}" for i, event in enumerate(event_sequence)])
                
                messages = [
                    {"role": "system", "content": "You are a causal reasoning system for Lucidia's world model. "
                                            "Your task is to analyze a sequence of events and identify causal "
                                            "relationships between them, determining what caused what and with what probability."},
                    {"role": "user", "content": f"Event sequence:\n{events_text}\n\n"
                                            f"Context: {context}\n\n"
                                            f"Reasoning depth: {reasoning_depth}\n\n"
                                            f"Please perform causal reasoning on this event sequence, identifying cause-effect "
                                            f"relationships between events. For each causal link, specify the cause event, "
                                            f"effect event, strength of causation (0-1), and any intermediary mechanisms. "
                                            f"Analyze up to {reasoning_depth} levels of causation depth."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.5,    # Balanced temperature for reasoning
                    max_tokens=1200,    # Allow for thorough causal analysis
                    timeout=50          # Give enough time for complex reasoning
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    reasoning_text = response["choices"][0]["message"].get("content", "")
                    
                    # Process reasoning text to structured format
                    causal_chains = []
                    
                    # Extract causal relationships from the response
                    causal_blocks = re.split(r'\n\s*\d+\.\s*|\n\s*-|\n\s*\*', reasoning_text)
                    for block in causal_blocks:
                        if not block.strip():
                            continue
                            
                        causal_link = {}
                        
                        # Look for cause, effect, strength
                        cause_match = re.search(r'[Cc]ause[:\s]*([^\n]+)', block)
                        effect_match = re.search(r'[Ee]ffect[:\s]*([^\n]+)', block)
                        strength_match = re.search(r'[Ss]trength[:\s]*([0-9]\.[0-9]+)', block)
                        mechanism_match = re.search(r'[Mm]echanism[:\s]*([^\n]+)', block)
                        
                        # Process direct causal expression (Cause -> Effect)
                        direct_match = re.search(r'([^\n:]+)\s*(?:->|→|–|causes|led to)\s*([^\n:]+)', block)
                        
                        if cause_match and effect_match:
                            causal_link["cause"] = cause_match.group(1).strip()
                            causal_link["effect"] = effect_match.group(1).strip()
                        elif direct_match:
                            causal_link["cause"] = direct_match.group(1).strip()
                            causal_link["effect"] = direct_match.group(2).strip()
                        
                        # Get strength
                        if strength_match:
                            try:
                                strength = float(strength_match.group(1))
                                causal_link["strength"] = min(max(strength, 0.0), 1.0)  # Clamp to 0-1
                            except ValueError:
                                causal_link["strength"] = 0.7  # Default strength
                        else:
                            causal_link["strength"] = 0.7  # Default strength
                        
                        # Get mechanism if available
                        if mechanism_match:
                            causal_link["mechanism"] = mechanism_match.group(1).strip()
                        
                        # Add to causal chains if it has at least cause and effect
                        if "cause" in causal_link and "effect" in causal_link:
                            causal_chains.append(causal_link)
                    
                    return {
                        "status": "success",
                        "causal_chains": causal_chains,
                        "count": len(causal_chains)
                    }
            
            # Fallback if reasoning failed
            return {
                "status": "error",
                "message": "Causal reasoning failed",
                "causal_chains": []
            }
        except Exception as e:
            logger.error(f"Error in perform_causal_reasoning: {str(e)}")
            return {"status": "error", "message": str(e), "causal_chains": []}
    
    async def query_knowledge_graph(self, query: str, entity_focus: str = None,
                             relationship_types: List[str] = None, max_results: int = 10) -> Dict[str, Any]:
        """Query the knowledge graph for information about entities, relationships, or concepts."""
        try:
            if not query:
                return {
                    "status": "error",
                    "message": "Query is required",
                    "results": []
                }
            
            # Try to use knowledge graph's query function if available
            if self.knowledge_graph and hasattr(self.knowledge_graph, "query"):
                try:
                    results = await self.knowledge_graph.query(
                        query=query,
                        entity_focus=entity_focus,
                        relationship_types=relationship_types,
                        max_results=max_results
                    )
                    return {
                        "status": "success",
                        "results": results,
                        "count": len(results)
                    }
                except Exception as e:
                    logger.warning(f"Knowledge graph query failed: {str(e)}. Falling back to LLM.")
            
            # Fall back to memory system if available
            if self.memory_system and hasattr(self.memory_system, "semantic_search"):
                try:
                    # Search memory for relevant information
                    memories = await self.memory_system.semantic_search(
                        query=query,
                        limit=max_results
                    )
                    
                    return {
                        "status": "success",
                        "results": memories,
                        "count": len(memories),
                        "source": "memory_system"
                    }
                except Exception as e:
                    logger.warning(f"Memory system search failed: {str(e)}. Falling back to LLM.")
            
            # Fall back to LLM for simulated knowledge graph response
            if self.model_manager:
                relationship_filter = ", ".join(relationship_types) if relationship_types else "any"
                entity_context = f" about {entity_focus}" if entity_focus else ""
                
                messages = [
                    {"role": "system", "content": "You are a knowledge graph query system for Lucidia's world model. "
                                            "Your task is to respond to queries about entities, relationships, and concepts "
                                            "as if you were querying a knowledge graph. Return structured information."},
                    {"role": "user", "content": f"Query: {query}{entity_context}\n\n"
                                            f"Relationship types: {relationship_filter}\n"
                                            f"Max results: {max_results}\n\n"
                                            f"Please provide relevant information from the knowledge graph in response to this query. "
                                            f"Format your response as a structured list of results, with each result including "
                                            f"relevant entities, relationships, and attributes. If appropriate, include confidence "
                                            f"levels and source information."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.4,    # Lower temperature for more factual responses
                    max_tokens=1000,    # Enough for thorough results
                    timeout=45          # Reasonable timeout
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    query_text = response["choices"][0]["message"].get("content", "")
                    
                    # Return the LLM response with a flag indicating it's a simulated response
                    return {
                        "status": "success",
                        "results": query_text,
                        "simulated": True,
                        "message": "Response generated using LLM as knowledge graph not available."
                    }
            
            # If all approaches failed
            return {
                "status": "error",
                "message": "Knowledge graph query failed - no suitable query mechanism available",
                "results": []
            }
        except Exception as e:
            logger.error(f"Error in query_knowledge_graph: {str(e)}")
            return {"status": "error", "message": str(e), "results": []}