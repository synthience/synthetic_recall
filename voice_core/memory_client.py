"""
Memory system client for connecting to the tensor and HPC servers.
Handles embedding generation and memory operations.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
import websockets

logger = logging.getLogger(__name__)

class MemoryClient:
    """Client for interacting with the memory system servers."""
    
    def __init__(self, 
                 tensor_url: str = "ws://localhost:5001",
                 hpc_url: str = "ws://localhost:5005",
                 session_id: str = None):
        """
        Initialize memory client.
        
        Args:
            tensor_url: WebSocket URL for tensor server
            hpc_url: WebSocket URL for HPC server
            session_id: Unique session identifier
        """
        self.tensor_url = tensor_url
        self.hpc_url = hpc_url
        self.session_id = session_id or str(time.time())
        
        # Connection state
        self._tensor_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._hpc_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Synchronization locks
        self._tensor_lock = asyncio.Lock()
        self._hpc_lock = asyncio.Lock()
        
        # Local cache
        self._conversation_history: List[Dict[str, Any]] = []
        self._embeddings_cache: Dict[str, List[float]] = {}
        
    async def initialize(self) -> bool:
        """Initialize connections to memory servers."""
        try:
            # Start connection handler
            self._reconnect_task = asyncio.create_task(self._maintain_connections())
            
            # Wait for tensor server connection first
            for _ in range(3):  # Try for 3 seconds
                if self._tensor_ws:
                    break
                await asyncio.sleep(1)
                
            if not self._tensor_ws:
                logger.error("Failed to connect to tensor server")
                return False
                
            # Consider initialization successful if tensor server is connected
            # HPC connection will be maintained in background
            logger.info("Memory client initialized with tensor server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory client: {e}")
            return False
            
    async def _maintain_connections(self) -> None:
        """Maintain WebSocket connections with reconnection."""
        while True:
            try:
                # Connect to tensor server if needed
                if not self._tensor_ws:
                    try:
                        async with websockets.connect(self.tensor_url) as ws:
                            self._tensor_ws = ws
                            logger.info("Connected to tensor server")
                            
                            # Handle tensor server messages
                            async for message in ws:
                                try:
                                    data = json.loads(message)
                                    await self._handle_tensor_message(data)
                                except json.JSONDecodeError:
                                    logger.error("Invalid JSON from tensor server")
                                except Exception as e:
                                    logger.error(f"Error handling tensor message: {e}")
                    except Exception as e:
                        logger.error(f"Error connecting to tensor server: {e}")
                        self._tensor_ws = None
                        await asyncio.sleep(1)  # Delay before retry
                
                # Connect to HPC server if needed
                if not self._hpc_ws:
                    try:
                        async with websockets.connect(self.hpc_url) as ws:
                            self._hpc_ws = ws
                            logger.info("Connected to HPC server")
                            self._connected = True
                            
                            # Handle HPC server messages
                            async for message in ws:
                                try:
                                    data = json.loads(message)
                                    await self._handle_hpc_message(data)
                                except json.JSONDecodeError:
                                    logger.error("Invalid JSON from HPC server")
                                except Exception as e:
                                    logger.error(f"Error handling HPC message: {e}")
                    except Exception as e:
                        logger.error(f"Error connecting to HPC server: {e}")
                        self._hpc_ws = None
                        await asyncio.sleep(1)  # Delay before retry
                        
            except Exception as e:
                logger.error(f"Error in connection maintenance: {e}")
                
            await asyncio.sleep(1)  # Main loop delay
            
    async def _handle_tensor_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming tensor server message."""
        msg_type = data.get("type")
        
        if msg_type == "embeddings":
            # Cache embeddings
            timestamp = data.get("timestamp")
            embeddings = data.get("embeddings")
            if timestamp and embeddings:
                self._embeddings_cache[timestamp] = embeddings
                
                # Forward to HPC server
                if self._hpc_ws:
                    async with self._hpc_lock:
                        try:
                            await self._hpc_ws.send(json.dumps({
                                "type": "process_embeddings",
                                "session_id": self.session_id,
                                "timestamp": timestamp,
                                "embeddings": embeddings
                            }))
                        except Exception as e:
                            logger.error(f"Error forwarding embeddings to HPC: {e}")
                    
    async def _handle_hpc_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming HPC server message."""
        msg_type = data.get("type")
        
        if msg_type == "memory_processed":
            # Update local cache with processed memory
            memory_id = data.get("memory_id")
            if memory_id:
                logger.info(f"Memory processed: {memory_id}")
                
    async def store_transcript(self, text: str, sender: str) -> bool:
        """
        Store a conversation transcript.
        
        Args:
            text: The transcript text
            sender: Who sent the message ("user" or "assistant")
            
        Returns:
            True if stored successfully
        """
        try:
            # Add to local cache
            entry = {
                "text": text,
                "sender": sender,
                "timestamp": time.time()
            }
            self._conversation_history.append(entry)
            
            # Request embeddings from tensor server
            if self._tensor_ws:
                async with self._tensor_lock:
                    try:
                        await self._tensor_ws.send(json.dumps({
                            "type": "embed",
                            "session_id": self.session_id,
                            "text": text,
                            "timestamp": entry["timestamp"]
                        }))
                        return True
                    except Exception as e:
                        logger.error(f"Error sending transcript to tensor server: {e}")
                        return False
            return False
            
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return False
            
    async def store_conversation(self, text: str, role: str = "assistant") -> bool:
        """Store conversation entry with embeddings."""
        try:
            # Add to conversation history
            entry = {
                "text": text,
                "role": role,
                "timestamp": time.time()
            }
            self._conversation_history.append(entry)
            
            # Request embeddings from tensor server
            if self._tensor_ws:
                async with self._tensor_lock:
                    try:
                        await self._tensor_ws.send(json.dumps({
                            "type": "embed",
                            "session_id": self.session_id,
                            "text": text,
                            "timestamp": entry["timestamp"]
                        }))
                        return True
                    except Exception as e:
                        logger.error(f"Error sending conversation to tensor server: {e}")
                        return False
            return False
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return False

    async def retrieve_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from memory for a given query.
        
        Args:
            query: The query text to search for relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories with similarity scores
        """
        try:
            if self._tensor_ws:
                async with self._tensor_lock:
                    await self._tensor_ws.send(json.dumps({
                        "type": "search",
                        "session_id": self.session_id,
                        "text": query,
                        "limit": limit
                    }))
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(self._tensor_ws.recv(), timeout=5.0)
                        data = json.loads(response)
                        if data["type"] == "search_results":
                            return data["results"]
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for search results")
                    except Exception as e:
                        logger.error(f"Error receiving search results: {e}")
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []

    def format_context(self, memories: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        """
        Format retrieved memories into a context string for the LLM.
        
        Args:
            memories: List of memory objects with text and metadata
            max_tokens: Maximum approximate token length for context
            
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
            
        context_parts = []
        total_length = 0  # Rough token estimation
        
        # Sort by similarity * significance
        sorted_memories = sorted(
            memories,
            key=lambda x: (x.get("similarity", 0) * 0.7 + x.get("significance", 0) * 0.3),
            reverse=True
        )
        
        for memory in sorted_memories:
            text = memory.get("text", "").strip()
            if not text:
                continue
                
            # Rough token estimation (4 chars ≈ 1 token)
            est_tokens = len(text) // 4
            if total_length + est_tokens > max_tokens:
                break
                
            context_parts.append(text)
            total_length += est_tokens
            
        if context_parts:
            return "Previous relevant context:\n" + "\n---\n".join(context_parts) + "\n\nCurrent conversation:"
        return ""

    async def get_rag_context(self, query: str) -> str:
        """
        Get formatted RAG context for a query.
        
        Args:
            query: The query to find relevant context for
            
        Returns:
            Formatted context string for the LLM
        """
        memories = await self.retrieve_context(query)
        return self.format_context(memories)

    async def cleanup(self) -> None:
        """Clean up WebSocket connections."""
        try:
            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                
            if self._tensor_ws:
                await self._tensor_ws.close()
                self._tensor_ws = None
                
            if self._hpc_ws:
                await self._hpc_ws.close()
                self._hpc_ws = None
                
            self._connected = False
            logger.info("Memory client cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get cached conversation history."""
        return self._conversation_history.copy()
