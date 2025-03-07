"""Memory Client Proxy for interfacing with the Memory Agent."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from memory_core.memory_broker import get_memory_broker

logger = logging.getLogger(__name__)

class MemoryClientProxy:
    """Client proxy for interfacing with the Memory Agent via the broker."""
    
    def __init__(self):
        """Initialize the memory client proxy."""
        self.broker = None
    
    async def connect(self):
        """Connect to the memory broker."""
        if self.broker is None:
            self.broker = await get_memory_broker()
            logger.info("Memory client proxy connected to broker")
    
    async def close(self):
        """Close the connection to the memory broker."""
        # No need to explicitly close the broker connection
        # It's a shared resource managed globally
        self.broker = None
        logger.info("Memory client proxy disconnected from broker")
    
    async def classify_query(self, query: str) -> str:
        """Classify a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "classify_query", 
            {
                "query": query
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("query_type", "unknown")
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_memories", 
            {
                "query": query,
                "limit": limit,
                "min_significance": min_significance
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("memories", [])
    
    async def retrieve_information(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """Retrieve specific information via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "retrieve_information", 
            {
                "query": query,
                "context_type": context_type
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("information", {})

    async def store_and_retrieve(self, content: str, query: str = "", memory_type: str = "conversation") -> Dict[str, Any]:
        """Store content and retrieve related memories via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "store_and_retrieve", 
            {
                "content": content,
                "query": query,
                "memory_type": memory_type
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", {})

    async def store_emotional_context(self, user_input: str, emotions: Dict[str, float], timestamp: str = None) -> bool:
        """Store emotional context via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        response = await self.broker.send_request(
            "store_emotional_context", 
            {
                "user_input": user_input,
                "emotions": emotions,
                "timestamp": timestamp
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def get_emotional_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve emotional history via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_emotional_history", 
            {
                "limit": limit
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("history", [])

    async def detect_and_store_personal_details(self, text: str) -> Dict[str, Any]:
        """Detect and store personal details via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "detect_and_store_personal_details", 
            {
                "text": text
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("details", {})

    async def store_transcript(self, transcript: str, metadata: Dict[str, Any] = None) -> bool:
        """Store a conversation transcript via the memory agent."""
        if not self.broker:
            await self.connect()
        
        if metadata is None:
            metadata = {}
            
        response = await self.broker.send_request(
            "store_transcript", 
            {
                "transcript": transcript,
                "metadata": metadata
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("result", False)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for text via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "generate_embedding", 
            {
                "text": text
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("embedding", [])

    async def get_rag_context(self, query: str, max_tokens: int = 1000, min_significance: float = 0.3) -> str:
        """Get RAG context for a query via the memory agent."""
        if not self.broker:
            await self.connect()
            
        response = await self.broker.send_request(
            "get_rag_context", 
            {
                "query": query,
                "max_tokens": max_tokens,
                "min_significance": min_significance
            }
        )
        
        if not response.get("success", False):
            raise Exception(response.get("error", "Unknown error"))
            
        return response.get("context", "")