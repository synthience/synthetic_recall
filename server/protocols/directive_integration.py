"""
Directive Integration for Lucidia Memory System

This module ties together the ContextualToolDetector with the DreamToolProvider
and existing chat processing pipeline, enabling automatic directive detection
and tool execution during conversations.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from server.protocols.contextual_tool_detector import ContextualToolDetector
from server.protocols.dream_tools import DreamToolProvider
from server.protocols.tool_protocol import ToolProvider

logger = logging.getLogger(__name__)

class DirectiveIntegration:
    """
    Integration layer connecting directive detection with tool execution.
    
    This class wraps a tool provider and processes user messages to detect
    and execute directives based on natural language inputs.
    """
    
    def __init__(self, 
                 tool_provider: Optional[ToolProvider] = None,
                 dream_processor = None,
                 memory_system = None,
                 knowledge_graph = None,
                 parameter_manager = None,
                 model_manager = None):
        """
        Initialize the directive integration.
        
        Args:
            tool_provider: Existing tool provider to wrap or None to create a new one
            dream_processor: Dream processor component
            memory_system: Memory system component
            knowledge_graph: Knowledge graph component
            parameter_manager: Parameter manager component
            model_manager: Model manager component
        """
        # Use existing tool provider or create a new DreamToolProvider
        if tool_provider is None and dream_processor is not None:
            self.tool_provider = DreamToolProvider(
                dream_processor=dream_processor,
                memory_system=memory_system,
                knowledge_graph=knowledge_graph,
                parameter_manager=parameter_manager,
                model_manager=model_manager
            )
            logger.info("Created new DreamToolProvider for directive integration")
        else:
            self.tool_provider = tool_provider
            logger.info("Using existing tool provider for directive integration")
            
        # Create and configure the directive detector
        self.detector = ContextualToolDetector(tool_provider=self.tool_provider)
        
        # Register additional custom directives beyond the defaults
        self._register_custom_directives()
        
    def _register_custom_directives(self):
        """Register additional custom directives specific to this implementation."""
        # This can be extended with more domain-specific directives
        pass
        
    async def process_message_for_directives(self, 
                                          message: str, 
                                          conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user message to detect and execute any directives.
        
        Args:
            message: The user message text
            conversation_context: Additional context from the conversation
            
        Returns:
            Dictionary with results from any executed tools, or empty dict if none
        """
        results = await self.detector.process_message(message)
        
        if results and results.get("detected_directives", 0) > 0:
            logger.info(f"Executed {results['detected_directives']} directive(s) based on user message")
            
            # Create a summary for the conversation
            results["response_prefix"] = self._create_response_prefix(results) 
            
        return results or {}
    
    def _create_response_prefix(self, results: Dict[str, Any]) -> str:
        """
        Create a natural language prefix to add to the response based on executed tools.
        
        Args:
            results: Results from directive processing
            
        Returns:
            A string to prefix to the response
        """
        tool_results = results.get("results", [])
        if not tool_results:
            return ""
            
        # Get the first successful tool result
        successful_tools = [r for r in tool_results if "error" not in r]
        if not successful_tools:
            return ""
            
        tool = successful_tools[0]["tool"]
        
        # Create appropriate prefix based on the tool type
        if tool == "generate_dream_cycle":
            return "I've initiated a dream cycle as requested. "
        elif tool == "search_memories":
            return "I've searched my memories as requested. "
        elif tool == "generate_self_reflection":
            return "I've reflected on my recent interactions. "
        elif tool == "explore_knowledge_graph":
            return "I've explored my knowledge graph for those concepts. "
        elif tool == "generate_insight":
            return "I've generated some insights based on your request. "
        else:
            return f"I've processed your request using {tool}. "
            
    def inject_directive_results(self, 
                                response: str, 
                                directive_results: Dict[str, Any]) -> str:
        """
        Inject directive execution results into the response.
        
        Args:
            response: Original response text
            directive_results: Results from directive processing
            
        Returns:
            Modified response with tool results injected
        """
        if not directive_results or not directive_results.get("results"):
            return response
            
        # Add the prefix to the response
        prefix = directive_results.get("response_prefix", "")
        if prefix:
            response = prefix + response
            
        return response
