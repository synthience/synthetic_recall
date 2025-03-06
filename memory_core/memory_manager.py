# memory_core/memory_manager.py

import logging
import asyncio
from typing import Dict, Any, Optional, List

from memory_core.enhanced_memory_client import EnhancedMemoryClient

# Add this method to the MemoryManager class in memory_manager.py

async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
    """
    Detect emotional context in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dict with emotional context information
    """
    try:
        # Just pass through to the memory client
        return await self.memory_client.detect_emotional_context(text)
    except Exception as e:
        logger.error(f"Error detecting emotional context: {e}")
        # Return default neutral context on error
        return {
            "timestamp": time.time(),
            "text": text,
            "emotions": {"neutral": 1.0},
            "sentiment": 0.0,
            "emotional_state": "neutral",
            "error": str(e)
        }

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    High-level memory system manager that provides a simplified interface 
    for interacting with the memory system.
    
    This class serves as the main entry point for applications to interact
    with the memory system, hiding the complexity of the underlying
    implementation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.tensor_server_url = self.config.get("tensor_server_url", "ws://localhost:5001")
        self.hpc_server_url = self.config.get("hpc_server_url", "ws://localhost:5005")
        self.session_id = self.config.get("session_id")
        self.user_id = self.config.get("user_id")
        
        # Create memory client
        self.memory_client = EnhancedMemoryClient(
            tensor_server_url=self.tensor_server_url,
            hpc_server_url=self.hpc_server_url,
            session_id=self.session_id,
            user_id=self.user_id,
            **self.config
        )
        
        logger.info(f"Initialized MemoryManager with session_id={self.session_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the memory system.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize the memory client
            await self.memory_client.initialize()
            return True
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            return False
    
    async def process_message(self, text: str, role: str = "user") -> None:
        """
        Process a message and extract relevant information.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
        """
        await self.memory_client.process_message(text, role)
    
    async def search_memory(self, query: str, limit: int = 5, min_significance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for memories based on semantic similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching memories
        """
        return await self.memory_client.search_memory(query, limit, min_significance)
    
    async def store_memory(self, content: str, significance: float = None) -> bool:
        """
        Store a new memory.
        
        Args:
            content: The memory content
            significance: Optional significance override
            
        Returns:
            bool: Success status
        """
        return await self.memory_client.store_memory(content, significance=significance)
    
    async def get_memory_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for LLM integration.
        
        Returns:
            List of tool definitions
        """
        return await self.memory_client.get_memory_tools_for_llm()
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool call from the LLM.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments for the tool
            
        Returns:
            The result of the tool call
        """
        return await self.memory_client.handle_tool_call(tool_name, args)
    
    async def cleanup(self) -> None:
        """
        Clean up resources and persist memories.
        """
        await self.memory_client.cleanup()
        logger.info("Memory manager cleanup complete")
