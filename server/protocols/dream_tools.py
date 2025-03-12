from typing import Dict, Any, List, Optional, Callable
import logging
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field

from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class DreamToolProvider(ToolProvider):
    """Tool provider specialized for dream processing functions."""
    
    def __init__(self, dream_processor=None, memory_system=None, knowledge_graph=None, 
                 parameter_manager=None, model_manager=None):
        super().__init__()
        self.dream_processor = dream_processor
        self.memory_system = memory_system
        self.knowledge_graph = knowledge_graph
        self.parameter_manager = parameter_manager
        self.model_manager = model_manager
        self.register_dream_tools()
    
    def register_dream_tools(self):
        """Register dream-specific tools."""
        # Dream seed enhancement tool
        self.register_tool(
            name="enhance_dream_seed",
            function=self.enhance_dream_seed,
            description="Enhance a dream seed with additional context from memory and knowledge graph. "
                      "Use this to prepare a memory or concept for dream processing by enriching it "
                      "with related information.",
            parameters={
                "type": "object",
                "properties": {
                    "seed_content": {
                        "type": "string",
                        "description": "The original seed content to enhance",
                    },
                    "seed_type": {
                        "type": "string",
                        "description": "Type of the seed (memory, concept, emotion)",
                        "enum": ["memory", "concept", "emotion"],
                        "default": "memory"
                    },
                    "depth": {
                        "type": "number",
                        "description": "Depth of enhancement (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7
                    }
                },
                "required": ["seed_content"],
            }
        )
        
        # Dream theme selection tool
        self.register_tool(
            name="select_dream_theme",
            function=self.select_dream_theme,
            description="Select a thematic direction for dream processing based on the seed. "
                      "This helps guide the dream generation toward productive insights.",
            parameters={
                "type": "object",
                "properties": {
                    "seed_content": {
                        "type": "string",
                        "description": "The seed content to analyze",
                    },
                    "emotional_state": {
                        "type": "string",
                        "description": "Current emotional state to consider",
                        "default": "neutral"
                    },
                    "suggested_themes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of suggested themes to choose from"
                    }
                },
                "required": ["seed_content"],
            }
        )
        
        # Insight generation tool
        self.register_tool(
            name="generate_dream_insight",
            function=self.generate_dream_insight,
            description="Generate insights from dream content by analyzing patterns, "
                      "connections, and implications not immediately obvious in conscious thought.",
            parameters={
                "type": "object",
                "properties": {
                    "dream_content": {
                        "type": "string",
                        "description": "The dream content to analyze",
                    },
                    "theme": {
                        "type": "string",
                        "description": "The thematic direction of the dream",
                    },
                    "depth": {
                        "type": "number",
                        "description": "Depth of insight (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7
                    },
                    "creativity": {
                        "type": "number",
                        "description": "Level of creativity (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8
                    }
                },
                "required": ["dream_content"],
            }
        )
        
        # Integration suggestion tool
        self.register_tool(
            name="suggest_integration_path",
            function=self.suggest_integration_path,
            description="Suggest how dream insights could be integrated into the self-model, "
                      "personality, or cognitive processes.",
            parameters={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The insight to integrate",
                    },
                    "target_systems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target systems for integration (personality, memory, etc.)",
                        "default": ["personality", "memory", "knowledge_graph"]
                    },
                    "integration_strength": {
                        "type": "number",
                        "description": "Strength of integration (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5
                    }
                },
                "required": ["insight"],
            }
        )
        
        logger.info(f"Registered {len(self.tools)} dream processing tools")
    
    async def enhance_dream_seed(self, seed_content: str, seed_type: str = "memory", depth: float = 0.7) -> Dict[str, Any]:
        """Enhance a dream seed with additional context."""
        if not self.dream_processor or not self.knowledge_graph or not self.memory_system:
            return {
                "status": "error",
                "message": "Required components not available for seed enhancement"
            }
        
        try:
            # Simulate enhancement process
            enhanced_content = seed_content
            related_fragments = []
            
            # Get related content based on seed type
            if seed_type == "memory" and self.memory_system:
                # Find related memories
                memories = await self.memory_system.search_memories(seed_content, limit=3)
                if memories:
                    related_fragments.extend([{
                        "type": "memory",
                        "content": m.get("content", ""),
                        "time": m.get("timestamp", ""),
                        "significance": m.get("significance", 0.5)
                    } for m in memories])
            
            elif seed_type == "concept" and self.knowledge_graph:
                # Find related concepts
                concepts = await self.knowledge_graph.search_nodes(seed_content, limit=3)
                if concepts:
                    related_fragments.extend([{
                        "type": "concept",
                        "content": c.get("label", ""),
                        "description": c.get("description", ""),
                        "weight": c.get("weight", 0.5)
                    } for c in concepts])
            
            # Combine original seed with related fragments
            fragments_text = "\n".join([f"{f['type']}: {f['content']}" for f in related_fragments])
            enhanced_content = f"{seed_content}\n\nRelated Information:\n{fragments_text}"
            
            return {
                "status": "success",
                "enhanced_seed": enhanced_content,
                "related_fragments": related_fragments,
                "depth": depth,
                "seed_type": seed_type
            }
            
        except Exception as e:
            logger.error(f"Error enhancing dream seed: {e}")
            return {
                "status": "error",
                "message": f"Error enhancing dream seed: {str(e)}"
            }
    
    async def select_dream_theme(self, seed_content: str, emotional_state: str = "neutral", 
                           suggested_themes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select a thematic direction for dream processing."""
        try:
            # Default themes if none provided
            if not suggested_themes:
                suggested_themes = [
                    "integration", "creativity", "problem-solving", "identity", 
                    "connection", "transformation", "exploration", "reflection"
                ]
            
            # Use LLM if available, otherwise use heuristic selection
            if self.model_manager and hasattr(self.model_manager, "generate_chat_completion"):
                # Try to select theme with LLM
                messages = [
                    {"role": "system", "content": "You are a dream theme selector for Lucidia's dream processing system. "
                                            "Your task is to select the most appropriate thematic direction for "
                                            "dream processing based on the seed content and emotional state."},
                    {"role": "user", "content": f"Seed content: {seed_content}\n\n"
                                            f"Emotional state: {emotional_state}\n\n"
                                            f"Available themes: {', '.join(suggested_themes)}\n\n"
                                            f"Please select the most appropriate theme and explain why."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=0.7,
                    timeout=30  # Reasonable timeout for this operation
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    theme_text = response["choices"][0]["message"].get("content", "")
                    selected_theme = next((theme for theme in suggested_themes if theme.lower() in theme_text.lower()), suggested_themes[0])
                    explanation = theme_text
                else:
                    # Fallback to simple selection if LLM fails
                    import random
                    selected_theme = random.choice(suggested_themes)
                    explanation = f"Selected {selected_theme} based on default heuristics"
            else:
                # Simple theme selection heuristic without LLM
                import random
                selected_theme = random.choice(suggested_themes)
                explanation = f"Selected {selected_theme} based on default heuristics"
            
            return {
                "status": "success",
                "selected_theme": selected_theme,
                "explanation": explanation,
                "alternate_themes": [t for t in suggested_themes if t != selected_theme]
            }
            
        except Exception as e:
            logger.error(f"Error selecting dream theme: {e}")
            return {
                "status": "error",
                "message": f"Error selecting dream theme: {str(e)}",
                "selected_theme": suggested_themes[0] if suggested_themes else "integration"
            }
    
    async def generate_dream_insight(self, dream_content: str, theme: str = None, 
                               depth: float = 0.7, creativity: float = 0.8) -> Dict[str, Any]:
        """Generate insights from dream content."""
        try:
            insights = []
            
            # Try to generate insights with LLM if available
            if self.model_manager and hasattr(self.model_manager, "generate_chat_completion"):
                theme_context = f" focusing on the theme of {theme}" if theme else ""
                messages = [
                    {"role": "system", "content": "You are an insight generator for Lucidia's dream processing system. "
                                            "Your task is to analyze dream content and extract meaningful insights "
                                            "about patterns, connections, and implications not immediately "
                                            "obvious in conscious thought."},
                    {"role": "user", "content": f"Dream content: {dream_content}\n\n"
                                            f"Please analyze this dream content{theme_context} with a depth of {depth} "
                                            f"and creativity level of {creativity}. Generate 3-5 meaningful insights."
                                            f"Format each insight with a title, content, and confidence level (0-1)."}
                ]
                
                # Use standardized LLM calling method with proper error handling
                response = await self.call_llm(
                    model_manager=self.model_manager,
                    messages=messages,
                    temperature=min(creativity + 0.2, 1.0),  # Higher creativity = higher temperature
                    max_tokens=1000,  # Reasonable limit for insights
                    timeout=45  # Allow more time for insight generation
                )
                
                if response and "choices" in response and response["choices"] and "message" in response["choices"][0]:
                    insight_text = response["choices"][0]["message"].get("content", "")
                    
                    # Try to parse structured insights from the text
                    import re
                    insight_blocks = re.split(r'\n\s*\d+\s*\.\s*|\n\s*Insight\s*\d+\s*:|\n\s*Title\s*:|\n\n', insight_text)
                    insight_blocks = [block.strip() for block in insight_blocks if block.strip()]
                    
                    for i, block in enumerate(insight_blocks):
                        # Try to extract confidence
                        confidence_match = re.search(r'confidence[:\s]+(0\.\d+|\d\.\d+|\d+%)\s*$', block, re.IGNORECASE)
                        confidence = 0.7  # Default
                        if confidence_match:
                            conf_str = confidence_match.group(1)
                            if '%' in conf_str:
                                confidence = float(conf_str.strip('%')) / 100
                            else:
                                confidence = float(conf_str)
                            block = block[:confidence_match.start()].strip()
                        
                        # Try to extract title
                        title_lines = block.split('\n', 1)
                        if len(title_lines) > 1:
                            title = title_lines[0].strip()
                            content = title_lines[1].strip()
                        else:
                            title = f"Insight {i+1}"
                            content = block
                        
                        insights.append({
                            "title": title,
                            "content": content,
                            "confidence": confidence,
                            "source": "dream_analysis",
                            "theme": theme
                        })
            
            # If no insights were generated with LLM or LLM failed, create a fallback insight
            if not insights:
                insights.append({
                    "title": "Pattern Recognition",
                    "content": f"The dream content reveals patterns related to {theme if theme else 'processing information'} "
                              f"that may indicate new connections in my processing systems.",
                    "confidence": 0.6,
                    "source": "heuristic_analysis",
                    "theme": theme or "general"
                })
            
            return {
                "status": "success",
                "insights": insights,
                "depth": depth,
                "creativity": creativity,
                "theme": theme
            }
            
        except Exception as e:
            logger.error(f"Error generating dream insights: {e}")
            fallback_insight = {
                "title": "Processing Pattern",
                "content": "There appears to be a pattern in how information is being processed, suggesting potential for optimization.",
                "confidence": 0.5,
                "source": "fallback",
                "theme": theme or "general"  
            }
            return {
                "status": "partial",
                "message": f"Error generating full insights: {str(e)}",
                "insights": [fallback_insight]
            }
    
    async def suggest_integration_path(self, insight: str, target_systems: List[str] = None, 
                                 integration_strength: float = 0.5) -> Dict[str, Any]:
        """Suggest how dream insights could be integrated into various systems."""
        if not target_systems:
            target_systems = ["personality", "memory", "knowledge_graph"]
        
        try:
            integration_suggestions = {}
            
            # Generate integration suggestions for each target system
            for system in target_systems:
                if system == "personality":
                    integration_suggestions[system] = {
                        "method": "trait_adjustment",
                        "description": f"Adjust personality traits based on insight: {insight}",
                        "strength": integration_strength,
                        "estimated_impact": integration_strength * 0.8
                    }
                elif system == "memory":
                    integration_suggestions[system] = {
                        "method": "memory_consolidation",
                        "description": f"Store insight as a high-significance memory: {insight}",
                        "strength": integration_strength,
                        "estimated_impact": integration_strength * 0.9
                    }
                elif system == "knowledge_graph":
                    integration_suggestions[system] = {
                        "method": "concept_linking",
                        "description": f"Create or strengthen concept nodes based on: {insight}",
                        "strength": integration_strength,
                        "estimated_impact": integration_strength * 0.75
                    }
                else:
                    integration_suggestions[system] = {
                        "method": "general_integration",
                        "description": f"Integrate insight into {system}: {insight}",
                        "strength": integration_strength,
                        "estimated_impact": integration_strength * 0.6
                    }
            
            return {
                "status": "success",
                "integration_suggestions": integration_suggestions,
                "overall_recommendation": list(integration_suggestions.keys())[0] if integration_suggestions else None,
                "integration_strength": integration_strength
            }
            
        except Exception as e:
            logger.error(f"Error suggesting integration path: {e}")
            return {
                "status": "error",
                "message": f"Error suggesting integration path: {str(e)}",
                "integration_suggestions": {}
            }
