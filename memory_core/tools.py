# memory_core/tools.py

import logging
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import re
import uuid
import torch
import asyncio
import os
import shutil
import random
import copy

logger = logging.getLogger(__name__)

class ToolsMixin:
    """
    Mixin that provides smaller utility methods: 
    - Embedding creation
    - Searching/storing memories
    - tool endpoints for retrieval
    - emotional context detection
    - personal details management
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
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Generate embedding and significance
            try:
                embedding, memory_significance = await self.process_embedding(content)
                
                # Use provided significance if available
                if significance is not None:
                    try:
                        memory_significance = float(significance)
                        # Clamp to valid range
                        memory_significance = max(0.0, min(1.0, memory_significance))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid significance value: {significance}, using calculated value: {memory_significance}")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback: Use zero embedding and default significance
                if hasattr(self, 'embedding_dim'):
                    embedding = torch.zeros(self.embedding_dim)
                else:
                    embedding = torch.zeros(384)  # Default embedding dimension
                memory_significance = 0.5 if significance is None else significance
            
            # Create memory object
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "timestamp": time.time(),
                "significance": memory_significance,
                "metadata": metadata or {}
            }
            
            # Add to memory list
            async with self._memory_lock:
                self.memories.append(memory)
                
            logger.info(f"Stored new memory with ID {memory_id} and significance {memory_significance:.2f}")
            
            # For high-significance memories, force immediate persistence
            if memory_significance >= 0.7 and hasattr(self, 'persistence_enabled') and self.persistence_enabled:
                logger.info(f"Forcing immediate persistence for high-significance memory: {memory_id}")
                
                # Set up retry parameters
                max_retries = 3
                retry_delay = 0.5  # seconds
                success = False
                
                for retry in range(max_retries):
                    try:
                        # Save memory immediately instead of waiting for background task
                        memory_copy = copy.deepcopy(memory)
                        
                        # Convert complex types for JSON serialization
                        memory_copy = self._convert_numpy_to_python(memory_copy)
                        
                        # Ensure storage directory exists
                        os.makedirs(self.storage_path, exist_ok=True)
                        
                        # Use atomic write pattern with temporary file
                        file_path = self.storage_path / f"{memory_id}.json"
                        temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
                        backup_file_path = self.storage_path / f"{memory_id}.json.bak"
                        
                        try:
                            # Write to temporary file first (atomic write operation)
                            with open(temp_file_path, 'w', encoding='utf-8') as f:
                                json.dump(memory_copy, f, ensure_ascii=False, indent=2)
                                
                            # If the file exists, create a backup before overwriting
                            if file_path.exists():
                                try:
                                    shutil.copy2(file_path, backup_file_path)
                                except Exception as e:
                                    logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
                            
                            # Rename temporary file to actual file (atomic operation)
                            os.replace(temp_file_path, file_path)
                            
                            # Verify file integrity
                            with open(file_path, 'r', encoding='utf-8') as f:
                                _ = json.load(f)  # Just load to verify it's valid JSON
                            
                            # If we get here, the file is valid JSON
                            logger.info(f"Successfully persisted high-significance memory immediately: {memory_id}")
                            
                            # Remove backup if everything succeeded
                            if backup_file_path.exists():
                                os.remove(backup_file_path)
                                
                            success = True
                            break  # Exit retry loop on success
                        except json.JSONDecodeError:
                            logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                            # Restore from backup if verification failed
                            if backup_file_path.exists():
                                try:
                                    os.replace(backup_file_path, file_path)
                                    logger.info(f"Restored memory {memory_id} from backup after verification failure")
                                except Exception as e:
                                    logger.error(f"Failed to restore backup for memory {memory_id}: {e}")
                            # Continue with retry
                        except Exception as e:
                            # Clean up temp file if it exists
                            if temp_file_path.exists():
                                try:
                                    os.remove(temp_file_path)
                                except Exception:
                                    pass
                            raise e  # Re-raise for retry handling
                    
                    except Exception as e:
                        error_msg = f"Error persisting high-significance memory {memory_id} (attempt {retry+1}/{max_retries}): {e}"
                        if retry < max_retries - 1:
                            logger.warning(error_msg + ", retrying...")
                            # Exponential backoff with jitter
                            backoff_time = retry_delay * (2 ** retry) * (0.5 + 0.5 * random.random())
                            await asyncio.sleep(backoff_time)
                        else:
                            logger.error(error_msg + ", giving up")
                
                if not success:
                    logger.error(f"Failed to persist high-significance memory after {max_retries} attempts")
                    # Note: We still return True because the memory was added to the in-memory list
                    # It will be persisted later by the background task
            
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
            
    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: The object to convert
            
        Returns:
            The converted object with numpy types replaced by Python native types
        """
        # Handle None
        if obj is None:
            return None
            
        # Handle NumPy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)
            
        # Handle PyTorch tensors
        if hasattr(obj, '__module__') and 'torch' in obj.__module__:
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if hasattr(obj, 'detach') and hasattr(obj.detach(), 'numpy') and hasattr(obj.detach().numpy(), 'tolist'):
                return obj.detach().numpy().tolist()
            return str(obj)
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
            
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle sets
        if isinstance(obj, set):
            return [self._convert_numpy_to_python(item) for item in obj]
            
        # Handle other non-serializable types
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
    
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

    async def get_emotional_context(self, args: Dict = None, limit: int = 5) -> Dict[str, Any]:
        """
        Tool implementation to get emotional context information.
        
        Args:
            args: Optional arguments (unused)
            limit: Maximum number of emotions to include
            
        Returns:
            Dict with emotional context information
        """
        try:
            if not hasattr(self, "emotions") or not self.emotions:
                return {
                    "success": True,
                    "summary": "No emotional context information available yet.",
                    "emotions": {}
                }
            
            # Get recent emotions (limited by the limit parameter)
            recent_emotions = list(self.emotions.values())[-limit:] if self.emotions else []
            
            # Calculate average sentiment
            avg_sentiment = sum(e.get("sentiment", 0) for e in recent_emotions) / max(len(recent_emotions), 1)
            
            # Get dominant emotions
            all_detected = {}
            for emotion_data in recent_emotions:
                for emotion, score in emotion_data.get("emotions", {}).items():
                    if emotion not in all_detected:
                        all_detected[emotion] = []
                    all_detected[emotion].append(score)
            
            # Average the scores
            dominant_emotions = {}
            for emotion, scores in all_detected.items():
                dominant_emotions[emotion] = sum(scores) / len(scores)
            
            # Sort dominant emotions by score
            sorted_emotions = sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)
            top_emotions = dict(sorted_emotions[:3])  # Top 3 emotions
            
            # Create emotional context summary
            if avg_sentiment > 0.6:
                sentiment_desc = "very positive"
            elif avg_sentiment > 0.2:
                sentiment_desc = "positive"
            elif avg_sentiment > -0.2:
                sentiment_desc = "neutral"
            elif avg_sentiment > -0.6:
                sentiment_desc = "negative"
            else:
                sentiment_desc = "very negative"
                
            emotion_list = ", ".join([f"{emotion}" for emotion, _ in sorted_emotions[:3]])
            summary = f"User's recent emotional state appears {sentiment_desc} with prevalent emotions of {emotion_list}."
            
            return {
                "success": True,
                "summary": summary,
                "sentiment": avg_sentiment,
                "emotions": top_emotions,
                "recent_emotions": recent_emotions
            }
            
        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "Failed to retrieve emotional context."
            }
    
    async def get_personal_details_tool(self, args: Dict = None, category: str = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details about the user.
        
        Args:
            args: Optional arguments (unused)
            category: Optional category of personal details to retrieve
            
        Returns:
            Dict with personal details
        """
        try:
            details = {}
            
            # Try to get from personal details cache first
            if hasattr(self, "personal_details") and self.personal_details:
                # If category is specified, only return that category
                if category and category in self.personal_details:
                    details = {category: self.personal_details[category]}
                    logger.info(f"Retrieved personal detail for category: {category}")
                else:
                    details = self.personal_details.copy()
                    logger.info(f"Retrieved {len(details)} personal details from cache")
            
            # If no details in cache, try to extract from memories
            if not details and hasattr(self, "memories") and self.memories:
                # Look for memories with personal detail metadata
                async with self._memory_lock:
                    personal_memories = [m for m in self.memories 
                                        if m.get("metadata", {}).get("type") in 
                                        ["personal_detail", "name_reference", "location_reference", 
                                         "birthday_reference", "job_reference", "family_reference"]]
                
                # Extract details from memory content
                if personal_memories:
                    logger.info(f"Found {len(personal_memories)} personal detail memories")
                    
                    # Look for name references
                    name_memories = [m for m in personal_memories 
                                   if m.get("metadata", {}).get("type") == "name_reference" or 
                                      "name" in m.get("content", "").lower()]
                    if name_memories:
                        # Sort by significance and recency
                        name_memories = sorted(name_memories, 
                                              key=lambda x: (x.get("significance", 0), x.get("timestamp", 0)), 
                                              reverse=True)
                        details["name"] = name_memories[0].get("content")
                    
                    # Look for location references
                    location_memories = [m for m in personal_memories 
                                      if m.get("metadata", {}).get("type") == "location_reference" or 
                                         "location" in m.get("content", "").lower() or 
                                         "live" in m.get("content", "").lower()]
                    if location_memories:
                        location_memories = sorted(location_memories, 
                                                 key=lambda x: (x.get("significance", 0), x.get("timestamp", 0)), 
                                                 reverse=True)
                        details["location"] = location_memories[0].get("content")
                    
                    # Add other detail types as needed
            
            # If we found details, cache them for future use
            if details and hasattr(self, "personal_details"):
                self.personal_details.update(details)
            
            return {
                "success": len(details) > 0,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error getting personal details: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": {}
            }

    async def get_memory_tools(self) -> List[Dict[str, Any]]:
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
            },
            {
                "type": "function",
                "function": {
                    "name": "get_emotional_context",
                    "description": "Get the current emotional context and patterns from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_personal_details",
                    "description": "Get personal details about the user from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
