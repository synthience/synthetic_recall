# __init__.py

```py
# memory_core/__init__.py

"""Modular memory system for Lucid Recall"""

__version__ = "0.1.0"

from memory_core.enhanced_memory_client import EnhancedMemoryClient
from memory_core.memory_manager import MemoryManager

__all__ = ["EnhancedMemoryClient", "MemoryManager"]

```

# base.py

```py
# memory_core/base.py

import asyncio
import logging
import time
import uuid
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BaseMemoryClient:
    """
    Base class that sets up fundamental fields and structure. 
    Other mixins will extend this to build the full EnhancedMemoryClient.
    """
    def __init__(
        self,
        session_id: str,
        user_id: str = "default_user",
        tensor_server_url: str = "ws://localhost:5001",
        hpc_server_url: str = "ws://localhost:5005",
        enable_persistence: bool = True,
        persistence_dir: str = "data/memory",
        significance_threshold: float = 0.0
    ):
        # Connection info
        self.tensor_server_url = tensor_server_url
        self.hpc_server_url = hpc_server_url
        
        # Session identifiers
        self.session_id = session_id
        self.user_id = user_id
        
        # Persistence settings
        self.enable_persistence = enable_persistence
        self.persistence_dir = persistence_dir
        self.significance_threshold = significance_threshold
        
        # Connection state
        self._connected = False
        self._tensor_connection = None
        self._hpc_connection = None
        self._closing = False
        
        # Memory state
        self.memories = []
        self.topics_discussed = set()
        
        # Config
        self.max_memory_age = 86400 * 30  # 30 days in seconds
        self.max_retries = 5
        self.retry_delay = 1.0
        self.ping_interval = 30
        self.connection_timeout = 10
        
        # Threading locks
        self._tensor_lock = asyncio.Lock()
        self._hpc_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        
        # For background tasks
        self._background_tasks = set()
        self._memory_management_task = None
        
    async def initialize(self) -> bool:
        """
        Initialize the memory client and connect to the servers.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Connect to servers
            success = await self.connect()
            if not success:
                logger.error("Failed to connect to memory servers")
                return False
                
            # Create persistence directory if enabled
            if self.enable_persistence and not os.path.exists(self.persistence_dir):
                os.makedirs(self.persistence_dir, exist_ok=True)
                logger.info(f"Created persistence directory: {self.persistence_dir}")
                
            # Start background task for memory management
            self._start_memory_management()
            
            logger.info(f"Memory client initialized (session: {self.session_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing memory client: {e}")
            return False
            
    def _start_memory_management(self):
        """
        Start the background task for memory management.
        This runs asynchronously to handle periodic tasks like:
        - Consolidating memories
        - Pruning old memories
        - Periodic persistence
        """
        if self._memory_management_task is None or self._memory_management_task.done():
            self._memory_management_task = asyncio.create_task(
                self._memory_management_loop()
            )
            self._background_tasks.add(self._memory_management_task)
            self._memory_management_task.add_done_callback(
                self._background_tasks.discard
            )
            
    async def _memory_management_loop(self):
        """
        Background loop for memory management tasks.
        """
        try:
            while not self._closing:
                try:
                    # Persist memories if enabled
                    if self.enable_persistence:
                        await self._persist_memories()
                        
                    # Prune old memories
                    await self._prune_old_memories()
                    
                    # Wait for next cycle (every hour)
                    await asyncio.sleep(3600)
                    
                except asyncio.CancelledError:
                    logger.info("Memory management task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in memory management loop: {e}")
                    await asyncio.sleep(60)  # Wait and retry on error
                    
        except asyncio.CancelledError:
            logger.info("Memory management loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in memory management loop: {e}")
            
    async def close(self):
        """
        Close the memory client and clean up resources.
        """
        logger.info("Closing memory client")
        self._closing = True
        
        # Cancel background tasks
        if self._memory_management_task and not self._memory_management_task.done():
            self._memory_management_task.cancel()
            try:
                await self._memory_management_task
            except asyncio.CancelledError:
                pass
                
        # Persist memories before closing if enabled
        if self.enable_persistence:
            await self._persist_memories()
            
        # Close connections
        await self._close_connections()
        
    async def _close_connections(self):
        """
        Close WebSocket connections to servers.
        """
        try:
            async with self._tensor_lock:
                if self._tensor_connection:
                    await self._tensor_connection.close()
                    self._tensor_connection = None
                    
            async with self._hpc_lock:
                if self._hpc_connection:
                    await self._hpc_connection.close()
                    self._hpc_connection = None
                    
            self._connected = False
            logger.info("Closed all server connections")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
            
    async def _persist_memories(self):
        """
        Persist memories to disk.
        """
        try:
            if not self.enable_persistence or not self.memories:
                return
                
            async with self._memory_lock:
                # Filter memories by significance threshold
                memories_to_save = [
                    mem for mem in self.memories 
                    if mem.get('significance', 0.0) >= self.significance_threshold
                ]
                
                if not memories_to_save:
                    return
                    
                # Save to file
                file_path = os.path.join(
                    self.persistence_dir, 
                    f"{self.user_id}_{self.session_id}_memories.json"
                )
                
                with open(file_path, 'w') as f:
                    json.dump(memories_to_save, f)
                    
                logger.info(f"Persisted {len(memories_to_save)} memories to {file_path}")
                
        except Exception as e:
            logger.error(f"Error persisting memories: {e}")
            
    async def _prune_old_memories(self):
        """
        Remove memories older than max_memory_age.
        """
        try:
            now = time.time()
            
            async with self._memory_lock:
                original_count = len(self.memories)
                
                # Filter out old memories
                self.memories = [
                    mem for mem in self.memories 
                    if now - mem.get('timestamp', now) < self.max_memory_age
                ]
                
                pruned_count = original_count - len(self.memories)
                if pruned_count > 0:
                    logger.info(f"Pruned {pruned_count} old memories")
                    
        except Exception as e:
            logger.error(f"Error pruning old memories: {e}")
            
    def _get_timestamp(self) -> float:
        """
        Get current timestamp in seconds.
        
        Returns:
            float: Current Unix timestamp
        """
        return time.time()

```

# cognitive.py

```py
# memory_client/cognitive.py

import logging
import math
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class CognitiveMemoryMixin:
    """
    Mixin that applies a cognitive-inspired memory approach: 
    e.g., forgetting curve, spaced repetition, etc.
    """

    def __init__(self):
        self._memory_access_counts = defaultdict(int)
        self._memory_last_access = {}
        self._memory_decay_rate = 0.05  # 5% decay per day

    async def _apply_memory_decay(self):
        """Periodically apply memory decay."""
        pass

    async def record_memory_access(self, memory_id: str):
        """Record that a memory was accessed (reinforcement)."""
        pass

    async def associate_memories(self, memory_id1: str, memory_id2: str, strength: float = 0.5) -> bool:
        """Create an association between two memories."""
        return False

    async def get_associated_memories(self, memory_id: str, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        return []

```

# connectivity.py

```py
# memory_core/connectivity.py

import json
import asyncio
import websockets
import logging
from typing import Optional, Dict, Any
import time
import traceback

logger = logging.getLogger(__name__)

class ConnectivityMixin:
    """
    Mixin that handles WebSocket connectivity to the tensor and HPC servers.
    Requires self._tensor_lock, self._hpc_lock, etc. from the base class.
    """

    async def connect(self) -> bool:
        """Connect to the tensor and HPC servers."""
        if self._connected:
            return True

        logger.info("Connecting to tensor and HPC servers")
        tensor_connected = await self._connect_to_tensor_server()
        hpc_connected = await self._connect_to_hpc_server()
        
        self._connected = tensor_connected and hpc_connected
        return self._connected
    
    async def _connect_to_tensor_server(self) -> bool:
        """Connect to the tensor server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to tensor server at {self.tensor_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._tensor_lock:
                    self._tensor_connection = connection
                    
                logger.info("Successfully connected to tensor server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to tensor server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for tensor server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to tensor server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _connect_to_hpc_server(self) -> bool:
        """Connect to the HPC server with retry logic."""
        retry_count = 0
        max_retries = self.max_retries
        delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to HPC server at {self.hpc_server_url} (attempt {retry_count + 1}/{max_retries})")
                
                # Use a timeout for the connection attempt
                connection = await asyncio.wait_for(
                    websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                
                async with self._hpc_lock:
                    self._hpc_connection = connection
                    
                logger.info("Successfully connected to HPC server")
                return True
                
            except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to connect to HPC server: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached for HPC server connection")
                    return False
                    
                # Exponential backoff
                wait_time = delay * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Unexpected error connecting to HPC server: {e}")
                logger.error(traceback.format_exc())
                return False
    
    async def _get_tensor_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the tensor server connection, creating a new one if necessary."""
        async with self._tensor_lock:
            if self._tensor_connection and self._tensor_connection.open:
                return self._tensor_connection
                
            # Connection closed or doesn't exist, create new one
            try:
                logger.debug("Creating new tensor server connection")
                connection = await asyncio.wait_for(
                    websockets.connect(self.tensor_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                self._tensor_connection = connection
                return connection
            except Exception as e:
                logger.error(f"Failed to create tensor connection: {e}")
                return None
    
    async def _get_hpc_connection(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Get the HPC server connection, creating a new one if necessary."""
        async with self._hpc_lock:
            if self._hpc_connection and self._hpc_connection.open:
                return self._hpc_connection
                
            # Connection closed or doesn't exist, create new one
            try:
                logger.debug("Creating new HPC server connection")
                connection = await asyncio.wait_for(
                    websockets.connect(self.hpc_server_url, ping_interval=self.ping_interval),
                    timeout=self.connection_timeout
                )
                self._hpc_connection = connection
                return connection
            except Exception as e:
                logger.error(f"Failed to create HPC connection: {e}")
                return None

```

# consolidation.py

```py
# memory_client/consolidation.py

import time
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

class MemoryConsolidationMixin:
    """
    Mixin for memory consolidation - grouping related memories, summarizing them, etc.
    """

    async def _consolidate_memories(self):
        """Periodically consolidate related memories."""
        try:
            logger.info("Starting memory consolidation process")
            recent_memories = await self._get_recent_memories(days=7)
            if len(recent_memories) < 5:
                logger.info("Not enough recent memories for consolidation")
                return

            clusters = await self._cluster_similar_memories(recent_memories)
            for cluster_id, cluster_mems in clusters.items():
                if len(cluster_mems) < 3:
                    continue
                summary = await self._summarize_memory_cluster(cluster_mems)
                if summary:
                    significance = max(m.get("significance", 0.5) for m in cluster_mems)
                    memory_ids = [m["id"] for m in cluster_mems if "id" in m]
                    await self.store_significant_memory(
                        text=summary,
                        memory_type="consolidated",
                        metadata={
                            "source_count": len(cluster_mems),
                            "source_ids": memory_ids
                        },
                        min_significance=min(significance + 0.1, 0.95)
                    )
            logger.info("Memory consolidation completed")
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def _get_recent_memories(self, days=7) -> List[Dict[str, Any]]:
        """Example method for retrieving recent memories from server."""
        return []

    async def _cluster_similar_memories(self, memories: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster using DBSCAN or similar."""
        if not memories:
            return {}
        # Example placeholder logic
        return {}

    async def _summarize_memory_cluster(self, memories: List[Dict[str, Any]]) -> str:
        """Summarize a cluster of memory texts."""
        if not memories:
            return ""
        return "CONSOLIDATED MEMORY: ..."

```

# emotion.py

```py
# memory_core/emotion.py

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class EmotionMixin:
    """
    Mixin that handles emotion detection and tracking in the memory system.
    Allows for detecting and storing emotional context of conversations.
    """

    def __init__(self):
        # Initialize emotion tracking
        self.emotion_tracking = {
            "current_emotion": "neutral",
            "emotion_history": [],
            "emotional_triggers": {}
        }

    async def detect_emotion(self, text: str) -> str:
        """
        Detect emotion from text. Uses the HPC service for emotion analysis.
        
        Args:
            text: The text to analyze for emotion
            
        Returns:
            Detected emotion as string
        """
        try:
            connection = await self._get_hpc_connection()
            if not connection:
                logger.error("Cannot detect emotion: No HPC connection")
                return "neutral"
                
            # Create request payload
            payload = {
                "type": "emotion",
                "text": text
            }
            
            # Send request
            await connection.send(json.dumps(payload))
            
            # Get response
            response = await connection.recv()
            data = json.loads(response)
            
            if 'emotion' in data:
                emotion = data['emotion']
                
                # Update emotion tracking
                self.emotion_tracking["current_emotion"] = emotion
                self.emotion_tracking["emotion_history"].append({
                    "text": text,
                    "emotion": emotion,
                    "timestamp": self._get_timestamp()
                })
                
                # Keep history at a reasonable size
                if len(self.emotion_tracking["emotion_history"]) > 50:
                    self.emotion_tracking["emotion_history"] = \
                        self.emotion_tracking["emotion_history"][-50:]
                        
                return emotion
            else:
                logger.warning("No emotion data in response")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral"
        
    async def get_emotional_context(self, limit: int = 5) -> Dict[str, Any]:
        """
        Get the emotional context of recent conversations.
        
        Args:
            limit: Number of recent emotions to include
            
        Returns:
            Dict with emotional context information
        """
        recent_emotions = self.emotion_tracking["emotion_history"][-limit:] if \
            self.emotion_tracking["emotion_history"] else []
            
        return {
            "current_emotion": self.emotion_tracking["current_emotion"],
            "recent_emotions": recent_emotions,
            "emotional_triggers": self.emotion_tracking["emotional_triggers"]
        }
        
    async def store_emotional_trigger(self, trigger: str, emotion: str):
        """
        Store a trigger for a specific emotion.
        
        Args:
            trigger: The text/concept that triggered the emotion
            emotion: The emotion that was triggered
        """
        if trigger and emotion:
            # Add or update the trigger
            self.emotion_tracking["emotional_triggers"][trigger] = emotion
            logger.info(f"Stored emotional trigger: {trigger} -> {emotion}")

```

# enhanced_memory_client.py

```py
# memory_core/enhanced_memory_client.py

import logging
from typing import Dict, Any, Optional

from memory_core.base import BaseMemoryClient
from memory_core.connectivity import ConnectivityMixin
from memory_core.emotion import EmotionMixin
from memory_core.tools import ToolsMixin
from memory_core.personal_details import PersonalDetailsMixin

logger = logging.getLogger(__name__)

class EnhancedMemoryClient(BaseMemoryClient, 
                          ConnectivityMixin,
                          EmotionMixin,
                          ToolsMixin,
                          PersonalDetailsMixin):
    """
    Enhanced memory client that combines all mixins to provide a complete memory system.
    
    This class integrates all the functionality from the various mixins:
    - BaseMemoryClient: Core memory functionality and initialization
    - ConnectivityMixin: WebSocket connection handling for tensor and HPC servers
    - EmotionMixin: Emotion detection and tracking
    - ToolsMixin: Memory search and embedding tools
    - PersonalDetailsMixin: Personal information extraction and storage
    """
    
    def __init__(self, tensor_server_url: str, 
                 hpc_server_url: str,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 **kwargs):
        """
        Initialize the enhanced memory client.
        
        Args:
            tensor_server_url: URL for tensor server WebSocket connection
            hpc_server_url: URL for HPC server WebSocket connection
            session_id: Unique session identifier
            user_id: User identifier
            **kwargs: Additional configuration options
        """
        super().__init__(
            tensor_server_url=tensor_server_url,
            hpc_server_url=hpc_server_url,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )
        logger.info(f"Initialized EnhancedMemoryClient with session_id={session_id}")
    
    async def process_message(self, text: str, role: str = "user") -> None:
        """
        Process an incoming message to extract various information.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
        """
        # Only process user messages for personal details
        if role == "user":
            # Process for personal details
            await self.process_message_for_personal_details(text)
            
            # Process for emotions
            await self.analyze_emotions(text)
        
        # Store message in memory with default significance
        await self.store_memory(
            content=text,
            metadata={"role": role, "type": "message"}
        )
        
        logger.debug(f"Processed {role} message: {text[:50]}...")
    
    async def get_memory_tools_for_llm(self) -> Dict[str, Any]:
        """
        Get all memory tools formatted for the LLM.
        
        Returns:
            Dict with all available memory tools
        """
        # Get standard memory tools
        memory_tools = self.get_memory_tools()
        
        # Add personal details tool
        personal_tool = {
            "type": "function",
            "function": {
                "name": "get_personal_details",
                "description": "Retrieve personal details about the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category of personal detail to retrieve (e.g., 'name', 'location')"
                        }
                    }
                }
            }
        }
        
        # Add emotion tool
        emotion_tool = {
            "type": "function",
            "function": {
                "name": "get_emotional_context",
                "description": "Get the current emotional context of the conversation",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        
        memory_tools.append(personal_tool)
        memory_tools.append(emotion_tool)
        
        return memory_tools
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tool calls to the appropriate handlers.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments for the tool
            
        Returns:
            The result of the tool call
        """
        # Map tool names to their handlers
        tool_handlers = {
            "search_memory": self.search_memory_tool,
            "store_important_memory": self.store_important_memory,
            "get_important_memories": self.get_important_memories,
            "get_personal_details": self.get_personal_details_tool,
            "get_emotional_context": self.get_emotional_context_tool
        }
        
        # Get the appropriate handler
        handler = tool_handlers.get(tool_name)
        
        if handler:
            try:
                return await handler(args)
            except Exception as e:
                logger.error(f"Error handling tool call {tool_name}: {e}")
                return {"error": f"Tool execution error: {str(e)}"}
        else:
            logger.warning(f"Unknown tool call: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

```

# hierarchy.py

```py
# memory_core/hierarchy.py

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class HierarchicalMemoryMixin:
    """
    Mixin that provides hierarchical memory organization capabilities.
    
    Note: This is a stub implementation that will be fully implemented later.
    """
    
    def __init__(self):
        # Initialize hierarchical memory structures
        if not hasattr(self, "memory_hierarchies"):
            self.memory_hierarchies = {}
        
        logger.info("Initialized HierarchicalMemoryMixin (stub)")
    
    async def add_to_hierarchy(self, memory_id: str, category: str = None) -> bool:
        """
        Add a memory to the hierarchy.
        
        Args:
            memory_id: The ID of the memory to add
            category: Optional category to add the memory to
            
        Returns:
            bool: Success status
        """
        # Stub implementation
        logger.debug(f"Would add memory {memory_id} to hierarchy category {category}")
        return True
    
    async def get_category_memories(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories from a specific category.
        
        Args:
            category: The category to get memories from
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in the category
        """
        # Stub implementation
        logger.debug(f"Would retrieve memories from category {category}")
        return []
    
    async def suggest_categories(self, query: str) -> List[str]:
        """
        Suggest relevant categories based on a query.
        
        Args:
            query: The query to suggest categories for
            
        Returns:
            List of suggested categories
        """
        # Stub implementation
        logger.debug(f"Would suggest categories for query: {query}")
        return []

```

# memory_manager.py

```py
# memory_core/memory_manager.py

import logging
import asyncio
from typing import Dict, Any, Optional, List

from memory_core.enhanced_memory_client import EnhancedMemoryClient

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

```

# personal_details.py

```py
# memory_core/personal_details.py

import logging
import re
from typing import Dict, Any, List, Optional, Set
import time

logger = logging.getLogger(__name__)

class PersonalDetailsMixin:
    """
    Mixin that handles personal details extraction and storage.
    Automatically detects and stores personal information with high significance.
    """
    
    def __init__(self):
        # Initialize personal details storage if not exists
        if not hasattr(self, "personal_details"):
            self.personal_details = {}
        
        # Initialize common patterns
        self._name_patterns = [
            r"(?:my name is|i am|i'm|call me) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})",
            r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,2}) (?:is my name|here)",
        ]
        
        # Known personal detail categories
        self._personal_categories = {
            "name": {"patterns": self._name_patterns, "significance": 0.9},
            "birthday": {"patterns": [r"(?:my birthday is|i was born on) (.+?)[.\n]?"], "significance": 0.85},
            "location": {"patterns": [r"(?:i live in|i'm from|i am from) (.+?)[.\n,]?"], "significance": 0.8},
            "job": {"patterns": [r"(?:i work as a|my job is|i am a) (.+?)[.\n,]?"], "significance": 0.75},
            "family": {"patterns": [r"(?:my (?:wife|husband|partner|son|daughter|child|children) (?:is|are)) (.+?)[.\n,]?"], "significance": 0.85},
        }
        
        # Initialize list of detected names
        self._detected_names: Set[str] = set()
    
    async def detect_personal_details(self, text: str) -> Dict[str, Any]:
        """
        Detect personal details in text using pattern matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict of detected personal details
        """
        found_details = {}
        
        # Check all personal categories
        for category, config in self._personal_categories.items():
            for pattern in config["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches[0].strip()
                    found_details[category] = {
                        "value": value,
                        "confidence": 0.9,  # High confidence for direct pattern matches
                        "significance": config["significance"]
                    }
                    
                    # If we found a name, add to detected names
                    if category == "name":
                        self._detected_names.add(value.lower())
        
        return found_details
    
    async def check_for_name_references(self, text: str) -> List[str]:
        """
        Check if text refers to previously detected names.
        
        Args:
            text: The text to check
            
        Returns:
            List of detected names in the text
        """
        found_names = []
        
        # Check for each detected name in the text
        for name in self._detected_names:
            # Split the name to handle first names vs. full names
            name_parts = name.split()
            
            for part in name_parts:
                # Only check parts with length > 2 to avoid false positives
                if len(part) > 2:
                    # Look for the name with word boundaries
                    pattern = r'\b' + re.escape(part) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        found_names.append(name)
                        break
        
        return found_names
    
    async def store_personal_detail(self, category: str, value: str, significance: float = 0.8) -> bool:
        """
        Store a personal detail with high significance.
        
        Args:
            category: The type of personal detail (e.g., 'name', 'location')
            value: The value of the personal detail
            significance: Significance score (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        try:
            # Store in personal details dict
            self.personal_details[category] = {
                "value": value,
                "timestamp": time.time(),
                "significance": significance
            }
            
            # Also store as a high-significance memory
            memory_content = f"User {category}: {value}"
            await self.store_memory(
                content=memory_content,
                significance=significance,
                metadata={
                    "type": "personal_detail",
                    "category": category,
                    "value": value
                }
            )
            
            # If it's a name, add to detected names
            if category == "name":
                self._detected_names.add(value.lower())
            
            logger.info(f"Stored personal detail: {category}={value}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing personal detail: {e}")
            return False
    
    async def get_personal_detail(self, category: str) -> Optional[str]:
        """
        Retrieve a personal detail by category.
        
        Args:
            category: The type of personal detail to retrieve
            
        Returns:
            The value or None if not found
        """
        detail = self.personal_details.get(category)
        if detail:
            return detail.get("value")
        return None
    
    async def process_message_for_personal_details(self, text: str) -> None:
        """
        Process an incoming message to extract and store personal details.
        
        Args:
            text: The message text to process
        """
        # Detect personal details
        details = await self.detect_personal_details(text)
        
        # Store each detected detail
        for category, detail in details.items():
            await self.store_personal_detail(
                category=category,
                value=detail["value"],
                significance=detail["significance"]
            )
        
        # Check for references to known names
        name_references = await self.check_for_name_references(text)
        
        # If names were referenced, boost their significance in memory
        if name_references:
            for name in name_references:
                # Create a memory about the name reference
                memory_content = f"User mentioned name: {name}"
                await self.store_memory(
                    content=memory_content,
                    significance=0.75,  # Slightly lower than initial detection
                    metadata={
                        "type": "name_reference",
                        "name": name
                    }
                )
    
    async def get_personal_details_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to get personal details.
        
        Args:
            args: Tool arguments from LLM
            
        Returns:
            Dict with personal details
        """
        category = args.get("category", None)
        
        # If specific category requested
        if category:
            value = await self.get_personal_detail(category)
            return {
                "found": value is not None,
                "category": category,
                "value": value
            }
        
        # Return all personal details
        formatted_details = {}
        for cat, detail in self.personal_details.items():
            formatted_details[cat] = detail.get("value")
        
        return {
            "personal_details": formatted_details,
            "count": len(formatted_details)
        }

```

# proactive.py

```py
# memory_client/proactive.py

import logging
from typing import Dict, Any, List
import time

logger = logging.getLogger(__name__)

class ProactiveRetrievalMixin:
    """
    Mixin that handles prediction of relevant memories for the conversation context.
    """

    def __init__(self):
        self._proactive_memory_context = []
        self._prediction_weights = {
            "recency": 0.3,
            "relevance": 0.5,
            "importance": 0.2
        }

    async def predict_relevant_memories(self, current_context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Return a list of memory objects likely relevant to the current context.
        """
        # Example placeholder
        return []

    async def is_topic_repetitive(self, text: str) -> bool:
        """
        Check if text is too similar to recently discussed topics (avoid repetition).
        """
        return False

```

# rag_context.py

```py
# memory_client/rag_context.py

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RAGContextMixin:
    """
    Mixin for advanced context generation: RAG, hierarchical context, dynamic context, etc.
    """

    async def get_enhanced_rag_context(self, query: str, context_type: str = "auto", max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Return a dict with 'context' and 'metadata' for the given query.
        """
        return {"context": "", "metadata": {}}

    async def get_hierarchical_context(self, query: str, max_tokens: int = 1024) -> str:
        return ""

    async def generate_dynamic_context(self, query: str, conversation_history: list, max_tokens: int = 1024) -> str:
        return ""

    async def boost_context_quality(self, query: str, context_text: str, feedback: Dict[str, Any] = None) -> str:
        return context_text

    async def get_rag_context(self, query: str, max_tokens: int = 1024) -> str:
        return ""

```

# tools.py

```py
# memory_core/tools.py

import logging
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple

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
        Send text to HPC for embeddings, etc. 
        
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
            
            # Send embedding request
            payload = {"type": "embed", "text": text}
            await connection.send(json.dumps(payload))
            
            # Get embedding response
            response = await connection.recv()
            data = json.loads(response)
            
            # Extract embedding and normalize
            if 'embedding' in data:
                embedding = np.array(data['embedding'])
                
                # Process embedding with HPC for significance
                hpc_connection = await self._get_hpc_connection()
                if not hpc_connection:
                    logger.error("Failed to get HPC connection for significance")
                    return embedding, 0.5  # Default significance
                
                # Send to HPC for significance
                hpc_payload = {"type": "process", "embeddings": embedding.tolist()}
                await hpc_connection.send(json.dumps(hpc_payload))
                
                # Get significance
                hpc_response = await hpc_connection.recv()
                hpc_data = json.loads(hpc_response)
                
                significance = hpc_data.get('significance', 0.5)
                
                return embedding, significance
            else:
                logger.error("No embedding in response")
                return None, 0.0
                
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
            
            # Send search request
            payload = {"type": "search", "text": query, "limit": limit * 2}  # Request more to filter
            await connection.send(json.dumps(payload))
            
            # Get search response
            response = await connection.recv()
            data = json.loads(response)
            
            if 'results' in data:
                results = data['results']
                
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
            else:
                logger.error("No results in search response")
                return []
                
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
            if embedding is None:
                logger.error("Failed to create embedding for memory")
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
    
    async def search_memory_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation for memory search.
        
        Args:
            args: Tool arguments from LLM
            
        Returns:
            Dict with search results
        """
        query = args.get("query", "")
        limit = args.get("limit", 5)
        min_significance = args.get("min_significance", 0.0)
        
        if not query:
            return {"error": "No query provided", "memories": []}
        
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
    
    async def store_important_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory.
        
        Args:
            args: Tool arguments from LLM
            
        Returns:
            Dict with status
        """
        content = args.get("content", "")
        significance = args.get("significance", 0.8)
        
        if not content:
            return {"success": False, "error": "No content provided"}
        
        success = await self.store_memory(
            content=content,
            significance=significance,
            metadata={"source": "llm", "is_important": True}
        )
        
        return {"success": success}
    
    async def get_important_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to get important memories.
        
        Args:
            args: Tool arguments from LLM
            
        Returns:
            Dict with important memories
        """
        limit = args.get("limit", 5)
        min_significance = args.get("min_significance", 0.7)
        
        async with self._memory_lock:
            # Filter memories by significance
            important_memories = [
                mem for mem in self.memories 
                if mem.get("significance", 0.0) >= min_significance
            ]
            
            # Sort by significance (descending)
            sorted_memories = sorted(
                important_memories,
                key=lambda x: x.get("significance", 0.0),
                reverse=True
            )[:limit]
            
            # Format for LLM consumption
            formatted_memories = [
                {
                    "content": mem.get("content", ""),
                    "significance": mem.get("significance", 0.0),
                    "timestamp": mem.get("timestamp", 0)
                } for mem in sorted_memories
            ]
            
            return {
                "memories": formatted_memories,
                "count": len(formatted_memories)
            }

```

