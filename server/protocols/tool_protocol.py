from typing import Dict, Any, Callable, List, Optional, Protocol, runtime_checkable
import logging
import json
import aiohttp
import asyncio
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for any component that supports tool registration and execution."""
    tools: Dict[str, Any]
    http_session: Optional[aiohttp.ClientSession]
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """Register a new tool."""
        ...
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool with the given arguments."""
        ...

@dataclass
class ToolProvider:
    """Mixin class for providing tool functionality to any component."""
    tools: Dict[str, Any] = field(default_factory=dict)
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """Register a new tool with the provider."""
        self.tools[name] = {
            "function": function,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        }
        logger.info(f"Registered tool: {name}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool with the given arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
            result = await self.tools[tool_name]["function"](**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get a list of all available tools in OpenAI schema format."""
        return [tool["schema"] for tool in self.tools.values()]

    def register_standard_tools(self) -> None:
        """Placeholder for registering standard tools common to all components."""
        pass
        
    async def call_llm(self, model_manager, messages, model=None, temperature=0.7, max_tokens=None,
                       top_p=None, frequency_penalty=None, presence_penalty=None, timeout=60):
        """Standardized method for calling an LLM with proper error handling.
        
        Args:
            model_manager: The LLM manager instance to use for the call
            messages: List of message dictionaries in the format expected by the API
            model: Optional model override
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            timeout: Timeout in seconds for the API call
            
        Returns:
            Dictionary with the LLM response or error information
        """
        if not model_manager or not hasattr(model_manager, "generate_chat_completion"):
            logger.warning("LLM Manager not available for tool")
            return {
                "error": "LLM Manager not available",
                "simulated": True,
                "choices": [{
                    "message": {
                        "content": "This is a simulated response because the LLM is not available.",
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }]
            }
        
        # Prepare parameters for the API call
        params = {
            "messages": messages,
            "temperature": temperature
        }
        
        # Add optional parameters if provided
        if model is not None:
            params["model"] = model
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
            
        # Call the LLM with timeout
        try:
            # Use asyncio.wait_for to apply timeout
            result = await asyncio.wait_for(
                model_manager.generate_chat_completion(params),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM API call timed out after {timeout} seconds")
            return {
                "error": f"Request timed out after {timeout} seconds",
                "simulated": True,
                "choices": [{
                    "message": {
                        "content": "Response generation timed out. Please try again with a simpler request.",
                        "role": "assistant"
                    },
                    "finish_reason": "timeout"
                }]
            }
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return {
                "error": str(e),
                "simulated": True,
                "choices": [{
                    "message": {
                        "content": "An error occurred while generating a response.",
                        "role": "assistant"
                    },
                    "finish_reason": "error"
                }]
            }
