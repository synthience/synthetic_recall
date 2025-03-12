"""
RAG Integration for Lucidia Self-Model

This module implements integration between Lucidia's self-model and 
Retrieval-Augmented Generation (RAG) capabilities using the LM Studio
tool schema.

Created by CASCADE
"""

import json
import logging
import aiohttp
import os
from typing import Dict, List, Any, Optional, Union, Callable
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGToolIntegration:
    """
    Integrates Retrieval-Augmented Generation capabilities with Lucidia's self-model
    using the LM Studio tool schema.
    """
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:1234/v1", 
                 api_key: str = "lm-studio",
                 model: str = "qwen2.5-7b-instruct"):
        """
        Initialize the RAG Tool Integration.
        
        Args:
            base_url: Base URL for the LM Studio API
            api_key: API key for authentication
            model: Model to use for completions
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.session = None
        self.tools = {}
        logger.info(f"Initialized RAG Tool Integration with API base: {self.base_url}, model: {self.model}")
    
    async def initialize(self):
        """Initialize the aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            })
            logger.info("Initialized aiohttp session for RAG tool integration")
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Closed aiohttp session for RAG tool integration")
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]):
        """
        Register a new tool for the RAG integration.
        
        Args:
            name: Name of the tool
            function: The function to execute when the tool is called
            description: Description of the tool
            parameters: Parameters schema for the tool in JSON Schema format
        """
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
    
    async def process_rag_query(self, query: str, messages: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a RAG query using the LM Studio API.
        
        Args:
            query: The query to process
            messages: Optional list of previous messages for context
            
        Returns:
            Dictionary containing the response and any tool calls
        """
        await self.initialize()
        
        if messages is None:
            messages = []
        
        # Add the user query to messages
        messages.append({"role": "user", "content": query})
        
        # Prepare tools for the API request
        tools_list = [tool["schema"] for tool in self.tools.values()]
        
        try:
            async with self.session.post(
                urljoin(self.base_url, "chat/completions"),
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools_list
                }
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    logger.error(f"Error from LM Studio API: {result}")
                    return {"status": "error", "message": str(result)}
                
                # Process tool calls if present
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    message = result["choices"][0]["message"]
                    
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_results = await self._process_tool_calls(message["tool_calls"], messages)
                        
                        # Add tool results to messages
                        messages.extend(tool_results["tool_messages"])
                        
                        # Get final response after tool execution
                        final_response = await self._get_final_response(messages)
                        return {
                            "status": "success",
                            "tool_results": tool_results["results"],
                            "response": final_response,
                            "messages": messages
                        }
                    else:
                        # No tool calls, just return the response
                        return {
                            "status": "success",
                            "response": message.get("content", ""),
                            "messages": messages
                        }
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return {"status": "error", "message": "Unexpected response format"}
                    
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _process_tool_calls(self, tool_calls, messages):
        """
        Process tool calls from the LM Studio API response.
        
        Args:
            tool_calls: List of tool calls from the API response
            messages: Current message history
            
        Returns:
            Dictionary containing results and messages
        """
        results = []
        tool_messages = []
        
        # Add the assistant message with tool calls
        tool_messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call["id"],
                "type": tool_call["type"],
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                }
            } for tool_call in tool_calls]
        })
        
        # Process each tool call
        for tool_call in tool_calls:
            if tool_call["type"] == "function":
                function_name = tool_call["function"]["name"]
                
                if function_name in self.tools:
                    # Parse arguments
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    # Execute the function
                    function = self.tools[function_name]["function"]
                    result = await function(**arguments) if callable(function) else {"error": "Function not callable"}
                    
                    # Add result to the list
                    results.append(result)
                    
                    # Add tool response message
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result)
                    })
                else:
                    logger.warning(f"Tool not found: {function_name}")
                    results.append({"error": f"Tool not found: {function_name}"})
        
        return {"results": results, "tool_messages": tool_messages}
    
    async def _get_final_response(self, messages):
        """
        Get the final response after tool execution.
        
        Args:
            messages: Current message history
            
        Returns:
            Final response text
        """
        try:
            async with self.session.post(
                urljoin(self.base_url, "chat/completions"),
                json={
                    "model": self.model,
                    "messages": messages
                }
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    logger.error(f"Error getting final response: {result}")
                    return "Error getting final response after tool execution"
                
                if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                    return result["choices"][0]["message"].get("content", "")
                else:
                    return "Unexpected response format for final response"
                    
        except Exception as e:
            logger.error(f"Error getting final response: {e}")
            return f"Error: {str(e)}"


# Example knowledge retrieval function
async def fetch_knowledge(query: str, sources: List[str] = None, max_results: int = 5):
    """
    Fetch knowledge from various sources based on the query.
    
    Args:
        query: The knowledge query
        sources: List of sources to query (wikipedia, knowledge_base, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing the retrieved knowledge
    """
    # This would be replaced with actual implementation
    # connecting to vector databases, knowledge bases, etc.
    return {
        "status": "success",
        "sources": sources or ["internal_knowledge_base"],
        "results": [
            {
                "content": f"Sample knowledge result for query: {query}",
                "source": "Sample Source",
                "confidence": 0.95
            }
        ]
    }


# Example Wikipedia integration similar to the example in the prompt
async def fetch_wikipedia_content(search_query: str):
    """
    Fetches wikipedia content for a given search query.
    
    Args:
        search_query: The search term for Wikipedia
        
    Returns:
        Dictionary containing the retrieved Wikipedia content
    """
    import urllib.parse
    import urllib.request
    
    try:
        # Search for most relevant article
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_query,
            "srlimit": 1,
        }

        url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        with urllib.request.urlopen(url) as response:
            search_data = json.loads(response.read().decode())

        if not search_data["query"]["search"]:
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'"
            }

        # Get the normalized title from search results
        normalized_title = search_data["query"]["search"][0]["title"]

        # Now fetch the actual content with the normalized title
        content_params = {
            "action": "query",
            "format": "json",
            "titles": normalized_title,
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "redirects": 1,
        }

        url = f"{search_url}?{urllib.parse.urlencode(content_params)}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        pages = data["query"]["pages"]
        page_id = list(pages.keys())[0]

        if page_id == "-1":
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'"
            }

        content = pages[page_id]["extract"].strip()
        return {
            "status": "success",
            "content": content,
            "title": pages[page_id]["title"]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
