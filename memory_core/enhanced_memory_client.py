# memory_core/enhanced_memory_client.py

import logging
from typing import Dict, Any, Optional, List, Union
import re

from memory_core.base import BaseMemoryClient
from memory_core.connectivity import ConnectivityMixin
from memory_core.emotion import EmotionMixin
from memory_core.tools import ToolsMixin
from memory_core.personal_details import PersonalDetailsMixin
from memory_core.rag_context import RAGContextMixin

logger = logging.getLogger(__name__)

class EnhancedMemoryClient(BaseMemoryClient, 
                          ConnectivityMixin,
                          EmotionMixin,
                          ToolsMixin,
                          PersonalDetailsMixin,
                          RAGContextMixin):
    """
    Enhanced memory client that combines all mixins to provide a complete memory system.
    
    This class integrates all the functionality from the various mixins:
    - BaseMemoryClient: Core memory functionality and initialization
    - ConnectivityMixin: WebSocket connection handling for tensor and HPC servers
    - EmotionMixin: Emotion detection and tracking
    - ToolsMixin: Memory search and embedding tools
    - PersonalDetailsMixin: Personal information extraction and storage
    - RAGContextMixin: Advanced context generation for RAG
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
        # Initialize the base class
        super().__init__(
            tensor_server_url=tensor_server_url,
            hpc_server_url=hpc_server_url,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )
        
        # Explicitly initialize all mixins
        PersonalDetailsMixin.__init__(self)
        EmotionMixin.__init__(self)
        ConnectivityMixin.__init__(self)
        ToolsMixin.__init__(self)
        RAGContextMixin.__init__(self)
        
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
            await self.detect_and_store_personal_details(text, role)
            
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
                # Check handler signature to determine if it expects named parameters or a dictionary
                if tool_name == "search_memory":
                    query = args.get("query", "")
                    limit = args.get("limit", 5)
                    min_significance = args.get("min_significance", 0.0)
                    return await handler(query=query, max_results=limit, min_significance=min_significance)
                elif tool_name == "store_important_memory":
                    content = args.get("content", "")
                    significance = args.get("significance", 0.8)
                    return await handler(content=content, significance=significance)
                elif tool_name == "get_important_memories":
                    limit = args.get("limit", 5)
                    min_significance = args.get("min_significance", 0.7)
                    return await handler(limit=limit, min_significance=min_significance)
                else:
                    # For other tools that still expect a dictionary
                    return await handler(args)
            except Exception as e:
                logger.error(f"Error handling tool call {tool_name}: {e}")
                return {"error": f"Tool execution error: {str(e)}"}
        else:
            logger.warning(f"Unknown tool call: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

    async def store_transcript(self, text: str, sender: str = "user") -> bool:
        """
        Store a transcript in memory with appropriate significance.
        
        Args:
            text: The transcript text
            sender: The sender (user or assistant)
            
        Returns:
            bool: Success status
        """
        try:
            # Check if this is a user message
            is_user = sender.lower() == "user"
            
            # Detect personal details first for user messages
            if is_user:
                # Process for personal details
                details_found = await self.detect_and_store_personal_details(text, role=sender)
                if details_found:
                    logger.info("Personal details detected and stored from transcript")
            
            # Calculate base significance based on sender
            # User messages are generally more significant as they contain user intent and information
            base_significance = 0.7 if is_user else 0.6
            
            # Check for important information patterns in the text
            importance_indicators = [
                # Questions and commands indicate user intent
                r"\?$",
                r"^(what|who|where|when|why|how)",
                r"^(can you|could you|please|help)",
                
                # Statements of fact about the user
                r"^i am", r"^i'm", r"^my", r"^i have", r"^i've",
                
                # Explicit memory requests
                r"remember", r"don't forget", r"note that", r"keep in mind",
                
                # Personal information
                r"name is", r"call me", r"address", r"phone", r"email",
                r"birthday", r"born", r"age", r"family", r"children",
                
                # Preferences and opinions
                r"i (like|love|hate|prefer|enjoy|dislike)",
                r"favorite", r"best", r"worst",
                
                # Future events
                r"tomorrow", r"next week", r"upcoming", r"planning", r"schedule"
            ]
            
            # Increase significance for important information
            for pattern in importance_indicators:
                if re.search(pattern, text, re.IGNORECASE):
                    base_significance = min(base_significance + 0.1, 0.9)
                    break
            
            # Store the transcript with calculated significance
            result = await self.store_memory(
                content=text,
                significance=base_significance,
                metadata={
                    "type": "transcript",
                    "sender": sender,
                    "timestamp": self._get_timestamp()
                }
            )
            
            if result:
                logger.info(f"Stored transcript with significance {base_significance:.2f}")
            else:
                logger.warning("Failed to store transcript")
                
            return result
            
        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return False

    async def detect_and_store_personal_details(self, text: str, role: str = "user") -> bool:
        """
        Detect and store personal details from text.
        
        This method analyzes the provided text for personal information such as names,
        locations, jobs, and family details. It uses pattern matching from the PersonalDetailsMixin
        to identify these details and stores them with high significance scores.
        
        Args:
            text: The text to analyze for personal details
            role: The role of the sender (user or assistant)
            
        Returns:
            bool: True if personal details were detected and stored, False otherwise
        """
        try:
            if role != "user":
                # Only process user messages for personal details
                logger.debug("Skipping personal details detection for non-user message")
                return False
                
            # Check for explicit name introduction patterns that might be missed by the standard patterns
            name_intro_patterns = [
                r"(?:I'm|I am|call me|my name'?s) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})",
                r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,2}) (?:is my name)",
                r"name(?:'s| is) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})"
            ]
            
            for pattern in name_intro_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    name = matches[0].strip()
                    logger.info(f"Detected explicit name introduction: {name}")
                    await self.store_personal_detail("name", name, 0.95)  # Higher significance for explicit intros
                    
                    # Also create a special memory for this name introduction
                    await self.store_memory(
                        content=f"User explicitly stated their name is {name}",
                        significance=0.95,
                        metadata={
                            "type": "name_introduction",
                            "name": name
                        }
                    )
            
            # Process the message to extract personal details using standard patterns
            await self.process_message_for_personal_details(text)
            
            # Check if we found any personal details
            details_found = len(self.personal_details) > 0
            
            if details_found:
                logger.info(f"Detected and stored personal details from text: {list(self.personal_details.keys())}")
                
                # For names specifically, create additional memories to improve retrieval
                if "name" in self.personal_details:
                    name = self.personal_details["name"]["value"]
                    await self.store_memory(
                        content=f"The user's name is {name}",
                        significance=0.9,
                        metadata={
                            "type": "name_reference",
                            "name": name
                        }
                    )
            else:
                logger.debug("No personal details detected in text")
                
            return details_found
            
        except Exception as e:
            logger.error(f"Error detecting and storing personal details: {e}")
            return False

    async def get_rag_context(self, query: str = None, limit: int = 5) -> str:
        """
        Get memory context for LLM RAG (Retrieval-Augmented Generation).
        
        Args:
            query: Optional query to filter memories
            limit: Maximum number of memories to include
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            # Use the RAGContextMixin implementation
            if query:
                return await super().get_rag_context(query, max_tokens=limit*100)
            else:
                # If no query is provided, get recent important memories
                memories = await self.get_important_memories(limit=limit)
                if not memories:
                    return ""
                    
                # Format memories into a context string
                context_parts = []
                for memory in memories:
                    content = memory.get("content", "")
                    metadata = memory.get("metadata", {})
                    timestamp = metadata.get("timestamp", "unknown time")
                    context_parts.append(f"[{timestamp}] {content}")
                    
                return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return ""
            
    async def store_conversation(self, text: str, role: str = "assistant") -> bool:
        """
        Store a conversation message in memory.
        
        Args:
            text: The message text
            role: The role of the sender (user or assistant)
            
        Returns:
            bool: Success status
        """
        try:
            # Store as a memory with conversation metadata
            return await self.store_memory(
                content=text,
                metadata={
                    "type": "conversation",
                    "role": role,
                    "session_id": self.session_id,
                    "timestamp": self._get_timestamp()
                }
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return False
            
    async def mark_topic_discussed(self, topic: Union[str, List[str]]) -> bool:
        """
        Mark a topic or list of topics as discussed in the current session.
        
        Args:
            topic: Topic or list of topics to mark as discussed
            
        Returns:
            bool: Success status
        """
        try:
            if isinstance(topic, str):
                topics = [topic]
            else:
                topics = topic
                
            success = True
            for t in topics:
                # Store a memory indicating the topic was discussed
                result = await self.store_memory(
                    content=f"Topic '{t}' was discussed",
                    metadata={
                        "type": "topic_discussed",
                        "topic": t,
                        "session_id": self.session_id,
                        "timestamp": self._get_timestamp()
                    }
                )
                success = success and result
                
            return success
        except Exception as e:
            logger.error(f"Error marking topic as discussed: {e}")
            return False

    async def get_emotional_context_tool(self) -> Dict[str, Any]:
        """
        Tool implementation to get emotional context.
        
        Returns:
            Dict with emotional context information
        """
        try:
            # Call the emotion mixin's get_emotional_context method
            return await self.get_emotional_context(limit=5)
        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {
                "error": str(e),
                "current_emotion": None,
                "recent_emotions": [],
                "emotional_triggers": {}
            }
