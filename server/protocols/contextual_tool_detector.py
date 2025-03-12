"""
Contextual Tool Detector for Lucidia Memory System

This module provides a mechanism to detect when tool calls should be invoked based on
natural language directives in conversation, allowing Lucidia to proactively respond
to specific intents without requiring explicit function calls.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import json

logger = logging.getLogger(__name__)

class DirectivePattern:
    """A pattern for matching user directives to specific tool invocations."""
    
    def __init__(
        self, 
        tool_name: str,
        patterns: List[str],
        description: str,
        parameter_extractors: Dict[str, Callable[[str], Any]] = None,
        default_params: Dict[str, Any] = None,
        priority: int = 0
    ):
        """
        Initialize a directive pattern.
        
        Args:
            tool_name: Name of the tool to invoke
            patterns: List of regex patterns to match against user input
            description: Description of what this directive does
            parameter_extractors: Functions to extract parameters from the matched text
            default_params: Default parameters to use if not extracted
            priority: Priority of this directive (higher numbers take precedence)
        """
        self.tool_name = tool_name
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.description = description
        self.parameter_extractors = parameter_extractors or {}
        self.default_params = default_params or {}
        self.priority = priority
        
    def match(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the directive matches the given text.
        
        Args:
            text: User input text to check
            
        Returns:
            Tuple of (match_found, extracted_parameters)
        """
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                # Extract parameters from the match
                params = self.default_params.copy()
                
                # Use parameter extractors if available
                for param_name, extractor in self.parameter_extractors.items():
                    try:
                        params[param_name] = extractor(text)
                    except Exception as e:
                        logger.warning(f"Error extracting parameter {param_name}: {e}")
                
                return True, params
        
        return False, {}


class ContextualToolDetector:
    """
    Detects intent in user messages and maps them to tool calls.
    
    This class enables Lucidia to recognize when the user's input indicates
    a specific tool should be called, without requiring explicit function call syntax.
    """
    
    def __init__(self, tool_provider=None):
        """
        Initialize the contextual tool detector.
        
        Args:
            tool_provider: The tool provider instance to use for executing tools
        """
        self.tool_provider = tool_provider
        self.directives: List[DirectivePattern] = []
        self.register_default_directives()
        
    def register_directive(self, directive: DirectivePattern) -> None:
        """
        Register a new directive pattern.
        
        Args:
            directive: The directive pattern to register
        """
        self.directives.append(directive)
        # Sort by priority (higher first)
        self.directives.sort(key=lambda d: d.priority, reverse=True)
        logger.info(f"Registered directive for tool {directive.tool_name} with priority {directive.priority}")
        
    def register_default_directives(self) -> None:
        """Register the default set of directives for common tools."""
        
        # Dream cycle directive
        self.register_directive(DirectivePattern(
            tool_name="generate_dream_cycle",
            patterns=[
                r"(?:initiate|start|begin|run|execute)\s+(?:a\s+)?dream(?:ing)?\s+cycle",
                r"(?:make|let)\s+(?:lucidia\s+)?dream\s+(?:about|on)"
            ],
            description="Initiates a dreaming cycle to process memories and generate insights",
            default_params={"time_budget": 180},
            priority=10
        ))
        
        # Memory search directive
        self.register_directive(DirectivePattern(
            tool_name="search_memories",
            patterns=[
                r"(?:find|search|look\s+for|retrieve)\s+(?:memories|memory)\s+(?:about|related\s+to|containing)\s+(.+)",
                r"(?:what\s+do\s+you\s+remember\s+about)\s+(.+)"
            ],
            description="Searches for memories related to a specific topic",
            parameter_extractors={
                "query": lambda text: re.search(r"(?:about|related\s+to|containing|remember\s+about)\s+(.+)(?:\?|\.|$)", text).group(1)
            },
            default_params={"limit": 5},
            priority=8
        ))
        
        # Self-reflection directive
        self.register_directive(DirectivePattern(
            tool_name="generate_self_reflection",
            patterns=[
                r"(?:reflect|think)\s+(?:about|on)\s+(?:yourself|your\s+(?:thoughts|behavior|actions|responses))",
                r"(?:perform|do)\s+(?:a\s+)?self[-\s]reflection"
            ],
            description="Generates a self-reflection on Lucidia's recent interactions and behavior",
            default_params={"depth": 0.7},
            priority=7
        ))
        
        # Knowledge graph exploration directive
        self.register_directive(DirectivePattern(
            tool_name="explore_knowledge_graph",
            patterns=[
                r"(?:explore|examine|investigate|analyze)\s+(?:the\s+)?(?:knowledge\s+graph|concept\s+network)\s+(?:for|about|related\s+to)\s+(.+)",
                r"(?:how\s+are\s+concepts\s+related\s+to)\s+(.+)"
            ],
            description="Explores connections in the knowledge graph related to a concept",
            parameter_extractors={
                "concept": lambda text: re.search(r"(?:for|about|related\s+to|related\s+to)\s+(.+)(?:\?|\.|$)", text).group(1)
            },
            default_params={"depth": 2},
            priority=6
        ))
        
        # Generate insight directive
        self.register_directive(DirectivePattern(
            tool_name="generate_insight",
            patterns=[
                r"(?:generate|create|provide|give\s+me)\s+(?:an\s+)?insight(?:s)?\s+(?:about|on|for|related\s+to)\s+(.+)",
                r"(?:what\s+insights\s+do\s+you\s+have\s+(?:about|on|for))\s+(.+)"
            ],
            description="Generates insights about a specific topic using memory and knowledge graph",
            parameter_extractors={
                "topic": lambda text: re.search(r"(?:about|on|for|related\s+to)\s+(.+)(?:\?|\.|$)", text).group(1)
            },
            default_params={},
            priority=5
        ))
    
    def detect_directives(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Detect directives in the given text.
        
        Args:
            text: User input text to analyze
            
        Returns:
            List of (tool_name, parameters) tuples for matched directives
        """
        matched_directives = []
        
        for directive in self.directives:
            match_found, params = directive.match(text)
            if match_found:
                matched_directives.append((directive.tool_name, params))
                logger.info(f"Detected directive for tool {directive.tool_name}")
                
                # Optionally stop after first match - uncomment if you want this behavior
                # return matched_directives
        
        return matched_directives
    
    async def process_message(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process a message and execute any detected directives.
        
        Args:
            text: User input text to process
            
        Returns:
            Result of the tool execution or None if no directives matched
        """
        if not self.tool_provider:
            logger.warning("No tool provider available for executing detected directives")
            return None
            
        directives = self.detect_directives(text)
        if not directives:
            return None
            
        results = []
        for tool_name, params in directives:
            try:
                logger.info(f"Executing tool {tool_name} with params {params}")
                result = await self.tool_provider.execute_tool(tool_name, params)
                results.append({
                    "tool": tool_name,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
                
        return {
            "detected_directives": len(directives),
            "results": results
        }
    
    def set_tool_provider(self, tool_provider) -> None:
        """
        Set the tool provider to use for executing detected directives.
        
        Args:
            tool_provider: The tool provider instance
        """
        self.tool_provider = tool_provider
