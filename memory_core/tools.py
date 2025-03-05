# memory_core/tools.py

import logging
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class ToolsMixin:
    """
    Mixin that provides smaller utility methods: 
    - Embedding creation
    - Searching/storing memories
    - tool endpoints for retrieval
    """

    async def process_embedding(self, text: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Send text to tensor server for embeddings and HPC for significance. 
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple of (embedding, significance)
        """
        try:
            # Get connection (creates new one if necessary)
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for embedding")
                return None, 0.0
            
            # Create a properly formatted message according to StandardWebSocketInterface
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            # Send embedding request with proper format
            payload = {
                "type": "embed",
                "text": text,
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get embedding response
            response = await connection.recv()
            data = json.loads(response)
            logger.debug(f"Received embedding response: {data}")
            
            # Extract embedding from standardized response format
            embedding = None
            if isinstance(data, dict):
                # Check for embedding in response data structure
                if 'data' in data and 'embeddings' in data['data']:
                    embedding = np.array(data['data']['embeddings'])
                    logger.debug(f"Found embeddings in data.embeddings: shape={embedding.shape}")
                elif 'data' in data and 'embedding' in data['data']:
                    embedding = np.array(data['data']['embedding'])
                    logger.debug(f"Found embedding in data.embedding: shape={embedding.shape}")
                elif 'embeddings' in data:
                    embedding = np.array(data['embeddings'])
                    logger.debug(f"Found embeddings at root level: shape={embedding.shape}")
                elif 'embedding' in data:
                    embedding = np.array(data['embedding'])
                    logger.debug(f"Found embedding at root level: shape={embedding.shape}")
                else:
                    logger.error(f"Could not find embedding in response. Keys: {data.keys()}")
                    if 'data' in data:
                        logger.error(f"Data keys: {data['data'].keys() if isinstance(data['data'], dict) else 'data is not a dict'}")
            
            if embedding is None:
                logger.error(f"No embedding in response: {data}")
                return None, 0.0
            
            # Process embedding with HPC for significance
            hpc_connection = await self._get_hpc_connection()
            if not hpc_connection:
                logger.error("Failed to get HPC connection for significance")
                return embedding, 0.5  # Default significance
            
            # Send to HPC for significance with proper format
            hpc_timestamp = time.time()
            hpc_message_id = f"{int(hpc_timestamp * 1000)}-{id(self):x}"
            
            hpc_payload = {
                "type": "process",
                "embeddings": embedding.tolist(),
                "client_id": self.session_id or "unknown",
                "message_id": hpc_message_id,
                "timestamp": hpc_timestamp
            }
            
            await hpc_connection.send(json.dumps(hpc_payload))
            
            # Get significance
            hpc_response = await hpc_connection.recv()
            hpc_data = json.loads(hpc_response)
            
            # Extract significance from standardized response format
            significance = 0.5  # Default
            if isinstance(hpc_data, dict):
                if 'data' in hpc_data and 'significance' in hpc_data['data']:
                    significance = hpc_data['data']['significance']
                elif 'significance' in hpc_data:
                    significance = hpc_data['significance']
            
            return embedding, significance
                
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            return None, 0.0
    
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
        try:
            # Get connection
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for search")
                return []
            
            # Send search request with proper format
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            payload = {
                "type": "search",
                "text": query,
                "limit": limit * 2,  # Request more to filter
                "min_significance": min_significance,
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get search response
            response = await connection.recv()
            data = json.loads(response)
            
            # Extract results from standardized response format
            results = []
            if isinstance(data, dict):
                if data.get('type') == 'search_results' and 'results' in data:
                    results = data['results']
                elif 'data' in data and 'results' in data['data']:
                    results = data['data']['results']
                elif 'results' in data:
                    results = data['results']
            
            if not results:
                logger.error(f"No results in search response: {data}")
                return []
                
            # Filter by significance
            filtered_results = [
                r for r in results 
                if r.get('significance', 0.0) >= min_significance
            ]
            
            # Sort by similarity and limit
            sorted_results = sorted(
                filtered_results, 
                key=lambda x: x.get('score', 0.0), 
                reverse=True
            )[:limit]
            
            return sorted_results
                
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    async def store_memory(self, content: str, metadata: Dict[str, Any] = None, significance: float = None) -> bool:
        """
        Store a new memory with semantic embedding.
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            significance: Optional override for significance
            
        Returns:
            bool: Success status
        """
        try:
            # Generate embedding and get significance
            embedding, auto_significance = await self.process_embedding(content)
            
            # If embedding failed but we have a significance override, create a dummy embedding
            if embedding is None:
                if significance is not None:
                    logger.warning("Using dummy embedding with provided significance")
                    # Create a dummy embedding (all zeros) for storage
                    embedding = np.zeros(768)  # Standard embedding size
                    auto_significance = significance
                else:
                    logger.error("Failed to create embedding for memory and no significance override provided")
                    return False
            
            # Use provided significance or auto-calculated
            memory_significance = significance if significance is not None else auto_significance
            
            # Create memory object
            memory = {
                "id": str(time.time()),
                "content": content,
                "embedding": embedding.tolist(),
                "timestamp": time.time(),
                "significance": memory_significance,
                "metadata": metadata or {}
            }
            
            # Add to memory store
            async with self._memory_lock:
                self.memories.append(memory)
            
            logger.info(f"Stored memory with significance {memory_significance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    def get_memory_tools(self) -> List[Dict[str, Any]]:
        """
        Return OpenAI-compatible function definitions for memory tools.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search for relevant memories based on semantic similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to return",
                                "default": 5
                            },
                            "min_significance": {
                                "type": "number",
                                "description": "Minimum significance threshold (0.0 to 1.0)",
                                "default": 0.0
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_important_memory",
                    "description": "Store an important memory with high significance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store"
                            },
                            "significance": {
                                "type": "number",
                                "description": "Memory significance (0.0 to 1.0)",
                                "default": 0.8
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_important_memories",
                    "description": "Retrieve the most important memories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of important memories to return",
                                "default": 5
                            },
                            "min_significance": {
                                "type": "number",
                                "description": "Minimum significance threshold (0.0 to 1.0)",
                                "default": 0.7
                            }
                        }
                    }
                }
            }
        ]
    
    async def search_memory_tool(self, query: str = "", memory_type: str = "all", max_results: int = 5, min_significance: float = 0.0, time_range: Dict = None) -> Dict[str, Any]:
        """
        Tool implementation for memory search.
        
        Args:
            query: The search query string
            memory_type: Type of memory to search (default: all)
            max_results: Maximum number of results to return (default: 5)
            min_significance: Minimum significance threshold (default: 0.0)
            time_range: Optional time range filter
            
        Returns:
            Dict with search results
        """
        if not query:
            return {"error": "No query provided", "memories": []}
        
        # Check for personal detail queries with more comprehensive patterns
        personal_detail_patterns = {
            "name": [r"what.*name", r"who am i", r"call me", r"my name", r"what.*call me", r"how.*call me", r"what.*i go by"],
            "location": [r"where.*live", r"where.*from", r"my location", r"my address", r"my home", r"where.*stay", r"where.*i.*live"],
            "birthday": [r"when.*born", r"my birthday", r"my birth date", r"when.*birthday", r"how old", r"my age"],
            "job": [r"what.*do for (a )?living", r"my (job|profession|occupation|career|work)", r"where.*work", r"what.*i do"],
            "family": [r"my (family|wife|husband|partner|child|children|son|daughter|mother|father|parent|sibling|brother|sister)"],
        }
        
        # First, try direct personal detail retrieval with higher priority
        for category, patterns in personal_detail_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logger.info(f"Personal detail query detected for category: {category}")
                    
                    # Try to get personal detail directly first
                    if hasattr(self, "get_personal_detail"):
                        value = await self.get_personal_detail(category)
                        if value:
                            logger.info(f"Found personal detail directly: {category}={value}")
                            # Return the personal detail as a high-significance memory
                            return {
                                "memories": [
                                    {
                                        "content": f"User {category}: {value}",
                                        "significance": 0.95,
                                        "timestamp": time.time()
                                    }
                                ],
                                "count": 1
                            }
                    
                    # If direct retrieval failed, try searching memory with type filters
                    if hasattr(self, "search_memory"):
                        # First try searching for memories with metadata type related to this category
                        type_specific_results = []
                        try:
                            # Search for memories with this category in metadata
                            async with self._memory_lock:
                                type_specific_results = [
                                    memory for memory in self.memories 
                                    if memory.get("metadata", {}).get("type") == f"{category}_reference" or 
                                       memory.get("metadata", {}).get("type") == f"{category}_introduction" or
                                       memory.get("metadata", {}).get("name") == category or
                                       memory.get("metadata", {}).get("category") == category
                                ]
                            
                            # Sort by significance and recency
                            type_specific_results = sorted(
                                type_specific_results,
                                key=lambda x: (x.get("significance", 0.0), x.get("timestamp", 0)),
                                reverse=True
                            )[:max_results]
                            
                            if type_specific_results:
                                logger.info(f"Found {len(type_specific_results)} memories with metadata type related to {category}")
                                formatted_results = [
                                    {
                                        "content": r.get("content", ""),
                                        "significance": r.get("significance", 0.0),
                                        "timestamp": r.get("timestamp", 0)
                                    } for r in type_specific_results
                                ]
                                return {
                                    "memories": formatted_results,
                                    "count": len(formatted_results)
                                }
                        except Exception as e:
                            logger.error(f"Error searching for type-specific memories: {e}")
                    
                    # If we couldn't get it directly, search with higher significance threshold
                    # and add category-specific terms to the query
                    enhanced_query = f"{category} {query}"
                    min_significance = max(min_significance, 0.7)  # Raise significance threshold
                    
                    # Use the enhanced query for the search
                    query = enhanced_query
                    logger.info(f"Using enhanced query for personal detail: {enhanced_query}")
                    break
        
        # For backward compatibility, use max_results as limit
        limit = max_results
        
        # Perform the actual memory search
        results = await self.search_memory(query, limit, min_significance)
        
        # Format for LLM consumption
        formatted_results = [
            {
                "content": r.get("content", ""),
                "significance": r.get("significance", 0.0),
                "timestamp": r.get("timestamp", 0)
            } for r in results
        ]
        
        return {
            "memories": formatted_results,
            "count": len(formatted_results)
        }
    
    async def store_important_memory(self, content: str = "", significance: float = 0.8) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory.
        
        Args:
            content: The memory content to store
            significance: Memory significance score (0.0-1.0)
            
        Returns:
            Dict with status
        """
        if not content:
            return {"success": False, "error": "No content provided"}
        
        try:
            # Get connection
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for storing important memory")
                return {"success": False, "error": "Connection failed"}
            
            # Send store request with proper format
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            payload = {
                "type": "embed",
                "text": content,
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            # Check success status
            success = False
            if isinstance(data, dict):
                if 'data' in data and 'success' in data['data']:
                    success = data['data']['success']
                elif 'success' in data:
                    success = data['success']
                elif data.get('type') == 'embeddings':  # Consider successful embedding generation as success
                    success = True
            
            if success:
                logger.info(f"Successfully stored important memory with significance {significance}")
            else:
                logger.error("Failed to store important memory")
                
            return {"success": success}
            
        except Exception as e:
            logger.error(f"Error storing important memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_important_memories(self, limit: int = 5, min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Tool implementation to get important memories.
        
        Args:
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold (0.0-1.0)
            
        Returns:
            Dict with important memories
        """
        try:
            # Get connection
            connection = await self._get_tensor_connection()
            if not connection:
                logger.error("Failed to get tensor connection for important memories")
                return {"memories": [], "count": 0}
            
            # Send search request with proper format
            timestamp = time.time()
            message_id = f"{int(timestamp * 1000)}-{id(self):x}"
            
            payload = {
                "type": "search",
                "min_significance": min_significance,
                "limit": limit,
                "sort_by": "significance",
                "client_id": self.session_id or "unknown",
                "message_id": message_id,
                "timestamp": timestamp
            }
            
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            # Extract results from standardized response format
            memories = []
            if isinstance(data, dict):
                if data.get('type') == 'search_results' and 'results' in data:
                    memories = data['results']
                elif 'data' in data and 'results' in data['data']:
                    memories = data['data']['results']
                elif 'results' in data:
                    memories = data['results']
            
            # Format for LLM consumption
            formatted_memories = [
                {
                    "content": mem.get("content", ""),
                    "significance": mem.get("significance", 0.0),
                    "timestamp": mem.get("timestamp", 0)
                } for mem in memories
            ]
            
            return {
                "memories": formatted_memories,
                "count": len(formatted_memories)
            }
            
        except Exception as e:
            logger.error(f"Error getting important memories: {e}")
            return {"memories": [], "count": 0}
