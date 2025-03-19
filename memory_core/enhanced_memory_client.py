# memory_core/enhanced_memory_client.py

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import re
import time
import asyncio
import json
import uuid
import copy
import os
import shutil
import numpy as np
import torch
import datetime

from memory_core.base import BaseMemoryClient
from memory_core.tools import ToolsMixin
from memory_core.emotion import EmotionMixin
from memory_core.connectivity import ConnectivityMixin
from memory_core.personal_details import PersonalDetailsMixin
from memory_core.rag_context import RAGContextMixin
from memory_core.lucidia_memory import LucidiaMemorySystemMixin

# Configure logger
logger = logging.getLogger(__name__)

class EnhancedMemoryClient(BaseMemoryClient,
                           ToolsMixin,
                           EmotionMixin,
                           ConnectivityMixin,
                           PersonalDetailsMixin,
                           RAGContextMixin,
                           LucidiaMemorySystemMixin):
    """
    Enhanced memory client that combines all mixins to provide a complete memory system.
    
    This class integrates all the functionality from the various mixins:
    - BaseMemoryClient: Core memory functionality and initialization
    - ConnectivityMixin: WebSocket connection handling for tensor and HPC servers
    - EmotionMixin: Emotion detection and tracking
    - ToolsMixin: Memory search and embedding tools
    - PersonalDetailsMixin: Personal information extraction and storage
    - RAGContextMixin: Advanced context generation for RAG
    - LucidiaMemorySystemMixin: Advanced memory system with knowledge graph, world model and self model
    """
    
    def __init__(self, tensor_server_url: str, 
                 hpc_server_url: str,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 ping_interval: float = 20.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 connection_timeout: float = 10.0,
                 **kwargs):
        """
        Initialize the enhanced memory client.
        
        Args:
            tensor_server_url: URL for tensor server WebSocket connection
            hpc_server_url: URL for HPC server WebSocket connection
            session_id: Unique session identifier
            user_id: User identifier
            ping_interval: Interval in seconds to send ping messages to servers
            max_retries: Maximum number of retries for failed connections
            retry_delay: Base delay in seconds between retries
            connection_timeout: Timeout in seconds for establishing connections
            **kwargs: Additional configuration options
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Set default emotion analyzer configuration
        self.emotion_analyzer_host = kwargs.get('emotion_analyzer_host', 
                                             os.getenv('EMOTION_ANALYZER_HOST', 'localhost'))
        self.emotion_analyzer_port = kwargs.get('emotion_analyzer_port', 
                                             os.getenv('EMOTION_ANALYZER_PORT', '5007'))
        self.emotion_analyzer_endpoint = f"ws://{self.emotion_analyzer_host}:{self.emotion_analyzer_port}/ws"
        self.logger.info(f"Configured emotion analyzer at: {self.emotion_analyzer_endpoint}")
        
        # Initialize the base class
        super().__init__(
            tensor_server_url=tensor_server_url,
            hpc_server_url=hpc_server_url,
            session_id=session_id,
            user_id=user_id,
            ping_interval=ping_interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
            connection_timeout=connection_timeout,
            **kwargs
        )
        
        # Explicitly initialize all mixins
        self._connected = False
        
        PersonalDetailsMixin.__init__(self)
        EmotionMixin.__init__(self)
        ToolsMixin.__init__(self)
        RAGContextMixin.__init__(self)
        LucidiaMemorySystemMixin.__init__(self)
        
        # Initialize topic suppression settings
        self._topic_suppression = {
            "enabled": True,
            "suppression_time": 3600,  # Default 1 hour in seconds
            "suppressed_topics": {}
        }
        
        # User context tracking
        self._user_preferences = {}
        self._conversation_history = []
        self._max_history_items = 50
        
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
            # Personal details
            await self.detect_and_store_personal_details(text, role)
            # Emotions
            await self.analyze_emotions(text)
        
        # Process through Lucidia's memory system
        await self.process_for_lucidia_memory(text, role)
        
        # Store message in memory
        await self.store_memory(
            content=text,
            metadata={"role": role, "type": "message", "timestamp": time.time()}
        )
        
        # Add to conversation history
        self._conversation_history.append({
            "role": role,
            "content": text,
            "timestamp": time.time()
        })
        
        # Prune history if needed
        if len(self._conversation_history) > self._max_history_items:
            self._conversation_history = self._conversation_history[-self._max_history_items:]
        
        logger.debug(f"Processed {role} message: {text[:50]}...")
    
    async def get_memory_tools_for_llm(self) -> Dict[str, Any]:
        """
        Get all memory tools formatted for the LLM.
        
        Returns:
            Dict with all available memory tools
        """
        tools = {
            "create_memory": self.store_memory,
            "search_memory": self.retrieve_memories,
            "search_information": self.retrieve_information,
            "get_emotional_context": self.get_emotional_context_tool,
            "get_personal_details": self.get_personal_details_tool,
            "track_topic": self.track_conversation_topic,
            "store_important_memory": self.store_important_memory,
            "retrieve_emotional_memories": self.retrieve_emotional_memories_tool,
        }
        
        # Add Lucidia Memory System tools if available
        if hasattr(self, 'get_lucidia_memory_tools'):
            lucidia_tools = self.get_lucidia_memory_tools()
            tools.update(lucidia_tools)
            
        return tools
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tool calls to the appropriate handlers.
        
        Args:
            tool_name: The name of the tool to call
            args: The arguments for the tool
            
        Returns:
            The result of the tool call
        """
        # Prepare arguments object with proper validation
        validated_args = self._validate_tool_args(tool_name, args)
        if "error" in validated_args:
            return {"error": validated_args["error"], "success": False}
        
        # Set up retry parameters
        max_retries = 2
        retry_count = 0
        backoff_factor = 1.5
        retry_delay = 1.0
        
        while retry_count <= max_retries:
            try:
                # Start time for performance tracking
                start_time = time.time()
                
                # Map tool names to their handlers
                result = await self._dispatch_tool_call(tool_name, validated_args)
                
                # Log performance metrics
                elapsed_time = time.time() - start_time
                logger.debug(f"Tool call {tool_name} completed in {elapsed_time:.3f}s")
                
                # Add performance metrics to result if successful
                if isinstance(result, dict) and "error" not in result:
                    result["_metadata"] = {
                        "execution_time": elapsed_time,
                        "tool_name": tool_name
                    }
                
                return result
                
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Tool call {tool_name} timed out after {max_retries} retries")
                    return {"error": f"Tool execution timed out", "success": False}
                
                # Exponential backoff
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(f"Tool call {tool_name} timed out, retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}", exc_info=True)
                retry_count += 1
                
                # Determine if error is retryable
                if retry_count > max_retries or not self._is_retryable_error(e):
                    return {"error": f"Tool execution error: {str(e)}", "success": False}
                
                # Exponential backoff for retryable errors
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(f"Retrying tool call {tool_name} in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
    
    async def _dispatch_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call to the appropriate handler method."""
        # Check if this is a Lucidia memory tool
        lucidia_tools = [
            "query_knowledge_graph", 
            "self_model_reflection", 
            "world_model_insight",
            "narrative_identity_insight"
        ]
        if tool_name in lucidia_tools:
            return await self.handle_lucidia_tool_call(tool_name, args)
            
        # Check if this is a narrative identity specific tool
        narrative_identity_tools = [
            "record_identity_experience",
            "get_identity_narrative",
            "get_autobiographical_timeline"
        ]
        if tool_name in narrative_identity_tools:
            if tool_name == "record_identity_experience":
                return await self.record_identity_experience(args)
            elif tool_name == "get_identity_narrative":
                return await self.get_identity_narrative(
                    narrative_type=args.get("narrative_type", "complete"),
                    style=args.get("style", "neutral")
                )
            elif tool_name == "get_autobiographical_timeline":
                return await self.get_autobiographical_timeline(
                    limit=args.get("limit", 10)
                )
        
        # Tool dispatcher mapping
        tool_handlers = {
            "search_memory": self.search_memory_tool,
            "store_important_memory": self.store_important_memory,
            "get_important_memories": self.get_important_memories,
            "get_personal_details": self.get_personal_details_tool,
            "get_emotional_context": self.get_emotional_context_tool,
            "track_conversation_topic": self.track_conversation_topic,
            "retrieve_memories": self.retrieve_memories,
            "store_memory": self.store_memory,
            "browse_memories": self.browse_memories,
            "search_memories": self.search_memories,
            "retrieve_emotional_memories": self.retrieve_emotional_memories_tool
        }
        
        # Get the appropriate handler
        handler = tool_handlers.get(tool_name)
        
        if not handler:
            logger.warning(f"Unknown tool call: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}", "success": False}
        
        # Handle search_memory tool with updated min_quickrecal_score
        if tool_name == "search_memory":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            min_quickrecal_score = args.get("min_quickrecal_score", 0.0)
            # For backward compatibility
            if "min_significance" in args and "min_quickrecal_score" not in args:
                min_quickrecal_score = args.get("min_significance", 0.0)
            return await handler(query=query, max_results=limit, min_quickrecal_score=min_quickrecal_score)
        
        # Handle store_important_memory tool
        elif tool_name == "store_important_memory":
            content = args.get("content", "")
            quickrecal_score = args.get("quickrecal_score", 0.8)
            return await handler(content=content, quickrecal_score=quickrecal_score)
        
        # Handle get_important_memories tool
        elif tool_name == "get_important_memories":
            limit = args.get("limit", 5)
            min_quickrecal_score = args.get("min_quickrecal_score", 0.7)
            return await handler(limit=limit, min_quickrecal_score=min_quickrecal_score)
        
        # Handle retrieve_memories tool
        elif tool_name == "retrieve_memories":
            query = args.get("query", "")
            limit = int(args.get("limit", 5))
            min_quickrecal_score = args.get("min_quickrecal_score", 0.0)
            return await handler(query=query, max_results=limit, min_quickrecal_score=min_quickrecal_score)
        
        # Handle store_memory tool
        elif tool_name == "store_memory":
            content = args.get("content", "")
            quickrecal_score = args.get("quickrecal_score", 0.8)
            return await handler(content=content, quickrecal_score=quickrecal_score)
        
        # Handle browse_memories tool
        elif tool_name == "browse_memories":
            limit = int(args.get("limit", 10))
            min_quickrecal_score = args.get("min_quickrecal_score", 0.7)
            return await handler(limit=limit, min_quickrecal_score=min_quickrecal_score)
        
        # Handle search_memories tool
        elif tool_name == "search_memories":
            query = args.get("query", "")
            limit = int(args.get("limit", 5))
            return await handler(query=query, limit=limit)
        
        # For other tools, pass args as a dictionary
        return await handler(args)
    
    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool arguments."""
        if not isinstance(args, dict):
            return {"error": "Arguments must be a dictionary"}
        
        # Tool-specific validation
        if tool_name == "search_memory":
            if "query" not in args or not args["query"]:
                return {"error": "Query is required for search_memory tool"}
            
            # Sanitize limit
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))  # Limit between 1-20
                except (ValueError, TypeError):
                    args["limit"] = 5  # Default if invalid
            
            # Sanitize min_quickrecal_score (and handle backward compatibility with min_significance)
            if "min_quickrecal_score" in args:
                try:
                    args["min_quickrecal_score"] = max(0.0, min(float(args["min_quickrecal_score"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["min_quickrecal_score"] = 0.0  # Default if invalid
            elif "min_significance" in args:
                # For backward compatibility
                try:
                    args["min_quickrecal_score"] = max(0.0, min(float(args["min_significance"]), 1.0))  # Range 0-1
                    # Remove min_significance to avoid confusion
                    args.pop("min_significance")
                except (ValueError, TypeError):
                    args["min_quickrecal_score"] = 0.0  # Default if invalid
        
        elif tool_name == "store_important_memory":
            if "content" not in args or not args["content"]:
                return {"error": "Content is required for store_important_memory tool"}
            
            # Sanitize quickrecal_score
            if "quickrecal_score" in args:
                try:
                    args["quickrecal_score"] = max(0.0, min(float(args["quickrecal_score"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["quickrecal_score"] = 0.8  # Default if invalid
        
        elif tool_name == "track_conversation_topic":
            if "topic" not in args or not args["topic"]:
                return {"error": "Topic is required for track_conversation_topic tool"}
            
            # Sanitize importance
            if "importance" in args:
                try:
                    args["importance"] = max(0.0, min(float(args["importance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["importance"] = 0.7  # Default if invalid
            else:
                args["importance"] = 0.7  # Default importance
        
        elif tool_name == "retrieve_memories":
            if "query" not in args or not args["query"]:
                return {"error": "Query is required for retrieve_memories tool"}
            
            # Sanitize limit
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))  # Limit between 1-20
                except (ValueError, TypeError):
                    args["limit"] = 5  # Default if invalid
            
            # Sanitize min_quickrecal_score
            if "min_quickrecal_score" in args:
                try:
                    args["min_quickrecal_score"] = max(0.0, min(float(args["min_quickrecal_score"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["min_quickrecal_score"] = 0.0  # Default if invalid
        
        elif tool_name == "store_memory":
            if "content" not in args or not args["content"]:
                return {"error": "Content is required for store_memory tool"}
            
            # Sanitize quickrecal_score
            if "quickrecal_score" in args:
                try:
                    args["quickrecal_score"] = max(0.0, min(float(args["quickrecal_score"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["quickrecal_score"] = 0.8  # Default if invalid
        
        elif tool_name == "browse_memories":
            # Sanitize limit
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))  # Limit between 1-20
                except (ValueError, TypeError):
                    args["limit"] = 10  # Default if invalid
            
            # Sanitize min_quickrecal_score
            if "min_quickrecal_score" in args:
                try:
                    args["min_quickrecal_score"] = max(0.0, min(float(args["min_quickrecal_score"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["min_quickrecal_score"] = 0.7  # Default if invalid
        
        elif tool_name == "search_memories":
            if "query" not in args or not args["query"]:
                return {"error": "Query is required for search_memories tool"}
            
            # Sanitize limit
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))  # Limit between 1-20
                except (ValueError, TypeError):
                    args["limit"] = 5  # Default if invalid
        
        return args
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        # Network-related errors are retryable
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            json.JSONDecodeError
        )
        
        # Also check error strings for network-related issues
        error_str = str(error).lower()
        network_keywords = ["connection", "timeout", "network", "socket", "unavailable"]
        
        return isinstance(error, retryable_errors) or any(keyword in error_str for keyword in network_keywords)

    async def store_transcript(self, text: str, sender: str = "user", quickrecal_score: float = None, role: str = None) -> bool:
        """
        Store a transcript entry in memory.
        
        This method stores conversation transcripts with appropriate metadata
        and automatically calculates quickrecal_score if not provided.
        
        Args:
            text: The transcript text to store
            sender: Who sent the message (user or assistant)
            quickrecal_score: Optional pre-calculated quickrecal_score value
            role: Alternative name for sender parameter (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not text or not text.strip():
            logger.warning("Empty transcript text provided")
            return False
            
        try:
            # Handle role parameter for backward compatibility
            if role is not None:
                sender = role
                
            # Calculate quickrecal_score if not provided
            if quickrecal_score is None:
                # Use a higher base quickrecal_score for user messages
                base_quickrecal_score = 0.6 if sender.lower() == "user" else 0.4
                
                # Adjust based on text length (longer messages often have more content)
                length_factor = min(len(text) / 100, 0.3)  # Cap at 0.3
                
                # Check for question marks (questions are often important)
                question_factor = 0.15 if "?" in text else 0.0
                
                # Check for exclamation marks (emotional content)
                emotion_factor = 0.1 if "!" in text else 0.0
                
                # Check for personal information markers
                personal_info_markers = ["my name", "I live", "my address", "my phone", "my email", "my birthday"]
                personal_factor = 0.2 if any(marker in text.lower() for marker in personal_info_markers) else 0.0
                
                # Final quickrecal_score calculation (capped at 0.95)
                quickrecal_score = min(base_quickrecal_score + length_factor + question_factor + emotion_factor + personal_factor, 0.95)
            
            # Create metadata
            metadata = {
                "type": "transcript",
                "sender": sender,
                "timestamp": time.time(),
                "session_id": self.session_id
            }
            
            # Store in memory system
            success = await self.store_memory(
                content=text,
                metadata=metadata,
                quickrecal_score=quickrecal_score
            )
            
            if success:
                logger.info(f"Stored transcript from {sender} with quickrecal_score {quickrecal_score:.2f}")
            else:
                logger.warning(f"Failed to store transcript from {sender}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return False

    async def detect_and_store_personal_details(self, text: str, role: str = "user") -> bool:
        """
        Detect and store personal details from text.
        
        This method analyzes text for personal details like name, location, etc.,
        and stores them in the personal details dictionary.
        
        Args:
            text: The text to analyze
            role: The role of the speaker (user or assistant)
            
        Returns:
            bool: True if any details were detected and stored
        """
        # Only process user messages
        if role.lower() != "user":
            return False
            
        try:
            details_found = False
            
            patterns = {
                "name": [
                    r"(?:my name is|i am|i'm|call me|they call me) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
                    r"([A-Z][a-z]+(?: [A-Z][a-z]+){0,3}) (?:is my name|here|speaking)",
                    r"(?:name's|names) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})",
                    r"(?:known as|goes by) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})"
                ],
                "location": [
                    r"i live (?:in|at) ([\w\s,]+)",
                    r"i(?:'m| am) from ([\w\s,]+)",
                    r"my address is ([\w\s,]+)",
                    r"my location is ([\w\s,]+)",
                    r"i (?:reside|stay) (?:in|at) ([\w\s,]+)",
                    r"(?:living|residing) in ([\w\s,]+)",
                    r"(?:based in|located in) ([\w\s,]+)"
                ],
                "birthday": [
                    r"my birthday is ([\w\s,]+)",
                    r"i was born on ([\w\s,]+)",
                    r"born in ([\w\s,]+)",
                    r"my birth date is ([\w\s,]+)",
                    r"my date of birth is ([\w\s,]+)",
                    r"i was born in ([\w\s,]+)"
                ],
                "job": [
                    r"i work as (?:an?|the) ([\w\s]+)",
                    r"i am (?:an?|the) ([\w\s]+)(?: by profession| by trade)?",
                    r"my job is ([\w\s]+)",
                    r"i'm (?:an?|the) ([\w\s]+)(?: by profession| by trade)?",
                    r"my profession is ([\w\s]+)",
                    r"i (?:do|practice) ([\w\s]+)(?: for (?:a|my) living)?",
                    r"i'm (?:employed as|working as) (?:an?|the) ([\w\s]+)",
                    r"my (?:career|occupation) is (?:in|as) ([\w\s]+)"
                ],
                "email": [
                    r"my email (?:is|address is) ([\w.+-]+@[\w-]+\.[\w.-]+)",
                    r"(?:reach|contact) me at ([\w.+-]+@[\w-]+\.[\w.-]+)",
                    r"([\w.+-]+@[\w-]+\.[\w.-]+) is my email"
                ],
                "phone": [
                    r"my (?:phone|number|phone number|cell|mobile) is ((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})",
                    r"(?:reach|contact|call) me at ((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4})",
                    r"((?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}) is my (?:phone|number|phone number|cell|mobile)"
                ],
                "age": [
                    r"i(?:'m| am) (\d+)(?: years old| years of age)?",
                    r"my age is (\d+)",
                    r"i turned (\d+) (?:recently|last year|this year)",
                    r"i'll be (\d+) (?:soon|next year|this year)"
                ]
            }
            
            family_patterns = {
                "spouse": [
                    r"my (wife|husband|spouse|partner) (?:is|'s) ([\w\s]+)",
                    r"i(?:'m| am) married to ([\w\s]+)",
                    r"([\w\s]+) is my (wife|husband|spouse|partner)"
                ],
                "child": [
                    r"my (son|daughter|child) (?:is|'s) ([\w\s]+)",
                    r"i have a (son|daughter|child) (?:named|called) ([\w\s]+)",
                    r"([\w\s]+) is my (son|daughter|child)"
                ],
                "parent": [
                    r"my (mother|father|mom|dad|parent) (?:is|'s) ([\w\s]+)",
                    r"([\w\s]+) is my (mother|father|mom|dad|parent)"
                ],
                "sibling": [
                    r"my (brother|sister|sibling) (?:is|'s) ([\w\s]+)",
                    r"i have a (brother|sister|sibling) (?:named|called) ([\w\s]+)",
                    r"([\w\s]+) is my (brother|sister|sibling)"
                ]
            }
            
            if hasattr(self, "personal_details") and "family" not in self.personal_details:
                self.personal_details["family"] = {}
            
            # Process standard patterns
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        value = matches[0].strip().rstrip('.,:;!?')
                        
                        if len(value) < 2 or value.lower() in ["a", "an", "the", "me", "i", "my", "he", "she", "they"]:
                            continue
                            
                        if hasattr(self, "personal_details"):
                            confidence = 0.9
                            self.personal_details.setdefault(category, []).append({
                                "value": value,
                                "confidence": confidence,
                                "source": "pattern_detection",
                                "timestamp": time.time()
                            })
                            
                            info = f"User's {category}: {value}"
                            await self.store_memory(
                                content=info, 
                                metadata={
                                    "type": "personal_detail",
                                    "category": category,
                                    "detection_method": "pattern",
                                    "confidence": confidence
                                },
                                quickrecal_score=0.9,
                            )
                            
                            logger.info(f"Detected personal detail - {category}: {value}")
                            details_found = True
                        
                        break
            
            # Process family patterns
            for relation, pattern_list in family_patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if len(match) >= 2:
                                if match[1].lower() in ["wife", "husband", "spouse", "partner", "son", "daughter", 
                                                        "child", "mother", "father", "mom", "dad", "parent",
                                                        "brother", "sister", "sibling"]:
                                    name = match[0].strip().rstrip('.,:;!?')
                                    relation_type = match[1].lower()
                                else:
                                    relation_type = match[0].lower()
                                    name = match[1].strip().rstrip('.,:;!?')
                                
                                if len(name) < 2 or name.lower() in ["a", "an", "the", "me", "i", "my"]:
                                    continue
                                    
                                if hasattr(self, "personal_details") and "family" in self.personal_details:
                                    confidence = 0.9
                                    self.personal_details["family"][relation_type] = {
                                        "name": name,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "pattern_detection"
                                    }
                                    logger.info(f"Detected family detail - {relation_type}: {name}")
                                    details_found = True
                                
                                await self.store_memory(
                                    content=f"User's {relation_type}: {name}",
                                    significance=0.85,  # For backward compatibility in this code
                                    metadata={
                                        "type": "personal_detail",
                                        "category": "family",
                                        "relation_type": relation_type,
                                        "value": name,
                                        "confidence": 0.9,
                                        "source": "pattern_detection"
                                    }
                                )
            
            return details_found
            
        except Exception as e:
            logger.error(f"Error detecting personal details: {e}")
            return False

    async def get_rag_context(self, query: str = None, limit: int = 5, min_quickrecal_score: float = None, max_tokens: int = None) -> str:
        """
        Get memory context for LLM RAG (Retrieval-Augmented Generation).
        
        Enhanced version with better categorization, formatting, and relevance.
        
        Args:
            query: Optional query to filter memories
            limit: Maximum number of memories to include
            min_quickrecal_score: Minimum quickrecal_score threshold for memories
            max_tokens: Maximum number of tokens to include (approximate)
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            quickrecal_threshold = min_quickrecal_score if min_quickrecal_score is not None else 0.0
            max_tokens = max_tokens or limit * 100  # Default token limit
            
            is_personal_query = await self._is_personal_query(query) if query else False
            is_memory_query = await self._is_memory_query(query) if query else False
            
            lucidia_context = ""
            try:
                lucidia_context = await self.get_lucidia_rag_context(query, max_tokens=max_tokens // 3)
            except Exception as e:
                logger.error(f"Error getting Lucidia insights: {e}")
            
            context_sections = {
                "personal": "",
                "memory": "",
                "emotional": "",
                "standard": "",
                "lucidia": lucidia_context
            }
            
            context_parts = []
            
            if is_personal_query:
                personal_context = await self._generate_personal_context(query)
                if personal_context:
                    context_sections["personal"] = personal_context
                    context_parts.append("### User Personal Information")
                    context_parts.append(personal_context)
            
            if is_memory_query:
                memory_limit = limit * 2
                memory_context = await self._generate_memory_context(query, memory_limit, quickrecal_threshold)
                if memory_context:
                    context_sections["memory"] = memory_context
                    context_parts.append("### Memory Recall")
                    context_parts.append(memory_context)
            
            if is_personal_query or (query and any(k in query.lower() for k in ["feel", "emotion", "mood", "happy", "sad", "angry"])):
                emotional_context = await self._generate_emotional_context()
                if emotional_context:
                    context_sections["emotional"] = emotional_context
                    context_parts.append("### Recent Emotional States")
                    context_parts.append(emotional_context)
            
            if not context_parts or (not is_memory_query and not is_personal_query):
                standard_context = await self._generate_standard_context(query, limit, quickrecal_threshold)
                if standard_context:
                    context_sections["standard"] = standard_context
                    if not context_parts:
                        context_parts.append("### Relevant Memory Context")
                    context_parts.append(standard_context)
            
            all_context = ""
            
            if context_sections.get("lucidia"):
                all_context += context_sections["lucidia"]
                all_context += "\n\n"
            
            if context_parts:
                all_context += "\n\n".join(context_parts)
            
            if max_tokens and len(all_context) > max_tokens * 4:
                all_context = all_context[: max_tokens * 4]
                
            return all_context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            return ""
    
    async def _is_personal_query(self, query: str) -> bool:
        personal_keywords = [
            "my name", "who am i", "where do i live", "where am i from",
            "how old am i", "what's my age", "what's my birthday", "when was i born",
            "what do i do", "what's my job", "what's my profession", "who is my",
            "my family", "my spouse", "my partner", "my children", "my parents",
            "my email", "my phone", "my number", "my address"
        ]
        if not query:
            return False
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in personal_keywords)
    
    async def _is_memory_query(self, query: str) -> bool:
        memory_keywords = [
            "remember", "recall", "forget", "memory", "memories", "mentioned",
            "said earlier", "talked about", "told me about", "what did i say",
            "what did you say", "earlier", "before", "previously", "last time",
            "yesterday", "last week", "last month", "last year", "in the past"
        ]
        if not query:
            return False
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in memory_keywords)
    
    async def _generate_personal_context(self, query: str) -> str:
        if not hasattr(self, "personal_details") or not self.personal_details:
            return ""
        
        parts = []
        parts.append("# USER PERSONAL INFORMATION")
        parts.append("# The following information is about the USER, not about the assistant")
        parts.append("# The assistant's name is Lucidia")
        
        personal_categories = {
            "name": ["name", "call me", "who am i"],
            "location": ["live", "from", "address", "location", "home"],
            "birthday": ["birthday", "born", "birth date", "age"],
            "job": ["job", "work", "profession", "career", "occupation"],
            "family": ["family", "spouse", "partner", "husband", "wife", "child", "children", "kid", "kids", "parent", "mother", "father"],
            "email": ["email", "e-mail", "mail"],
            "phone": ["phone", "number", "mobile", "cell"]
        }
        
        target_category = None
        query_lower = query.lower()
        for category, keywords in personal_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                target_category = category
                break
        
        if target_category:
            if target_category == "family":
                if "family" in self.personal_details and self.personal_details["family"]:
                    parts.append("USER's family information:")
                    for relation, data in self.personal_details["family"].items():
                        if isinstance(data, dict) and "name" in data:
                            parts.append(f"• USER's {relation}: {data['name']}")
                        else:
                            parts.append(f"• USER's {relation}: {data}")
            else:
                if target_category in self.personal_details:
                    detail = self.personal_details[target_category]
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {target_category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {target_category}: {detail}")
        else:
            for category, detail in self.personal_details.items():
                if category == "family":
                    if detail:
                        parts.append("USER's family information:")
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                parts.append(f"• USER's {relation}: {data['name']}")
                            else:
                                parts.append(f"• USER's {relation}: {data}")
                else:
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {category}: {detail}")
        
        parts.append("\n# IMPORTANT: The above information is about the USER, not the assistant")
        parts.append("# The assistant's name is Lucidia\n")
        
        return "\n".join(parts)
    
    async def _generate_memory_context(self, query: str, limit: int, min_quickrecal_score: float) -> str:
        """Generate context about past memories."""
        try:
            # Use higher quickrecal_score threshold for memory queries
            min_quickrecal_score = max(min_quickrecal_score, 0.3)
            
            # Search for memories related to the query
            memories = await self.search_memory_tool(
                query=query,
                max_results=limit,
                min_quickrecal_score=min_quickrecal_score
            )
            
            if not memories or not memories.get("memories"):
                return ""
            
            parts = []
            results = memories["memories"]
            
            # Format each memory with timestamp and content
            for i, memory in enumerate(results):
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                quickrecal_score = memory.get("quickrecal_score", 0.0)
                
                # Format timestamp as readable date
                date_str = ""
                if timestamp:
                    try:
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                
                # Add quickrecal_score indicators
                qr_indicator = ""
                if quickrecal_score > 0.8:
                    qr_indicator = " [IMPORTANT]"
                elif quickrecal_score > 0.6:
                    qr_indicator = " [Significant]"
                
                parts.append(f"• Memory from {date_str}{qr_indicator}: {content}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating memory context: {e}")
            return ""
    
    async def _generate_emotional_context(self) -> str:
        """Generate context about emotional states."""
        try:
            if hasattr(self, "get_emotional_history"):
                emotional_history = await self.get_emotional_history(limit=3)
                if emotional_history:
                    return emotional_history
            
            if hasattr(self, "emotions") and self.emotions:
                parts = []
                sorted_emotions = sorted(
                    self.emotions.items(),
                    key=lambda x: float(x[0]),
                    reverse=True
                )[:3]
                
                for timestamp, data in sorted_emotions:
                    sentiment = data.get("sentiment", 0)
                    emotions = data.get("emotions", {})
                    
                    try:
                        date_str = datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                    
                    if sentiment > 0.5:
                        sentiment_desc = "very positive"
                    elif sentiment > 0.1:
                        sentiment_desc = "positive"
                    elif sentiment > -0.1:
                        sentiment_desc = "neutral"
                    elif sentiment > -0.5:
                        sentiment_desc = "negative"
                    else:
                        sentiment_desc = "very negative"
                    
                    emotion_str = ""
                    if emotions:
                        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        emotion_str = ", ".join(f"{emotion}" for emotion, _ in top_emotions)
                        emotion_str = f" with {emotion_str}"
                    
                    parts.append(f"• {date_str}: User exhibited {sentiment_desc} sentiment{emotion_str}")
                
                return "\n".join(parts)
            
            return ""
        except Exception as e:
            logger.error(f"Error generating emotional context: {e}")
            return ""
    
    async def _generate_standard_context(self, query: str, limit: int, min_quickrecal_score: float) -> str:
        """Generate standard memory context for general queries."""
        try:
            if query:
                memories = await self.search_memory(
                    query=query,
                    limit=limit,
                    min_significance=min_quickrecal_score  # For backward compatibility in this method
                )
                
                if not memories:
                    return ""
                
                parts = []
                # Check if memories is a list or a dict with a "memories" key
                results = memories.get("memories") if isinstance(memories, dict) else memories
            else:
                memories = await self.get_important_memories(
                    limit=limit,
                    min_quickrecal_score=min_quickrecal_score
                )
                
                if not memories:
                    return ""
                
                parts = []
                # Check if memories is a list or a dict with a "memories" key
                results = memories.get("memories") if isinstance(memories, dict) else memories
            
            for memory in results:
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                
                date_str = ""
                if timestamp:
                    try:
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                        date_str = f"[{date_str}]"
                    except:
                        pass
                
                parts.append(f"• {date_str} {content}")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Error generating standard context: {e}")
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
            # Calculate appropriate quickrecal_score based on content
            base_quickrecal_score = 0.5 if role == "assistant" else 0.6
            
            # Adjust for content length
            length_factor = min(len(text) / 200, 0.2)  # Cap at 0.2
            
            # Check for question marks (questions are often important)
            question_factor = 0.1 if "?" in text else 0.0
            
            # Check for personal information indicators
            personal_indicators = ["name", "email", "phone", "address", "live", "age", "birthday", "family"]
            personal_factor = 0.2 if any(indicator in text.lower() for indicator in personal_indicators) else 0.0
            
            # Calculate final quickrecal_score
            quickrecal_score = min(base_quickrecal_score + length_factor + question_factor + personal_factor, 0.95)
            
            return await self.store_memory(
                content=text,
                quickrecal_score=quickrecal_score,
                metadata={
                    "type": "conversation",
                    "role": role,
                    "session_id": self.session_id,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            return False
            
    async def mark_topic_discussed(self, topic: Union[str, List[str]], importance: float = 0.7) -> bool:
        """
        Mark a topic or list of topics as discussed in the current session.
        
        Args:
            topic: Topic or list of topics to mark as discussed
            importance: Importance of the topic (0.0-1.0)
            
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
                t = t.strip().lower()
                if not t:
                    continue
                
                result = await self.store_memory(
                    content=f"Topic '{t}' was discussed",
                    significance=importance,  # For backward compatibility
                    metadata={
                        "type": "topic_discussed",
                        "topic": t,
                        "session_id": self.session_id,
                        "timestamp": time.time(),
                        "importance": importance
                    }
                )
                
                if self._topic_suppression["enabled"]:
                    expiration = time.time() + self._topic_suppression["suppression_time"]
                    self._topic_suppression["suppressed_topics"][t] = {
                        "expiration": expiration,
                        "importance": importance
                    }
                
                success = success and result
                
            return success
        except Exception as e:
            logger.error(f"Error marking topic as discussed: {e}")
            return False
    
    async def track_conversation_topic(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to track conversation topics for suppression.
        
        Args:
            args: Dictionary containing:
                - topic: The topic being discussed
                - importance: Importance of this topic (0-1)
            
        Returns:
            Dict with status information
        """
        try:
            topic = args.get("topic", "").strip()
            importance = args.get("importance", 0.7)
            
            if not topic:
                return {"success": False, "error": "No topic provided"}
            
            try:
                importance = float(importance)
                importance = max(0.0, min(1.0, importance))
            except (ValueError, TypeError):
                importance = 0.7
            
            success = await self.mark_topic_discussed(topic, importance)
            
            if success:
                return {
                    "success": True,
                    "topic": topic,
                    "importance": importance,
                    "suppression_time": self._topic_suppression["suppression_time"] if self._topic_suppression["enabled"] else 0
                }
            else:
                return {"success": False, "error": "Failed to store topic"}
            
        except Exception as e:
            logger.error(f"Error tracking conversation topic: {e}")
            return {"success": False, "error": str(e)}
    
    async def is_topic_suppressed(self, topic: str) -> Tuple[bool, float]:
        """
        Check if a topic is currently suppressed.
        
        Args:
            topic: The topic to check
            
        Returns:
            Tuple of (is_suppressed, importance)
        """
        if not self._topic_suppression["enabled"]:
            return False, 0.0
            
        topic = topic.strip().lower()
        
        if topic in self._topic_suppression["suppressed_topics"]:
            data = self._topic_suppression["suppressed_topics"][topic]
            expiration = data.get("expiration", 0)
            importance = data.get("importance", 0.5)
            
            if expiration > time.time():
                return True, importance
            else:
                del self._topic_suppression["suppressed_topics"][topic]
        
        return False, 0.0
    
    async def configure_topic_suppression(self, enable: bool = True, suppression_time: int = None) -> None:
        self._topic_suppression["enabled"] = enable
        
        if suppression_time is not None:
            try:
                suppression_time = int(suppression_time)
                if suppression_time > 0:
                    self._topic_suppression["suppression_time"] = suppression_time
            except (ValueError, TypeError):
                pass
        
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
    
    async def reset_topic_suppression(self, topic: str = None) -> None:
        if topic:
            topic = topic.strip().lower()
            if topic in self._topic_suppression["suppressed_topics"]:
                del self._topic_suppression["suppressed_topics"][topic]
        else:
            self._topic_suppression["suppressed_topics"] = {}
    
    async def get_topic_suppression_status(self) -> Dict[str, Any]:
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
        
        active_topics = {}
        for topic, data in self._topic_suppression["suppressed_topics"].items():
            expiration = data.get("expiration", 0)
            if expiration > current_time:
                time_left = int(expiration - current_time)
                active_topics[topic] = {
                    "time_left": time_left,
                    "importance": data.get("importance", 0.5)
                }
        
        return {
            "enabled": self._topic_suppression["enabled"],
            "suppression_time": self._topic_suppression["suppression_time"],
            "active_count": len(active_topics),
            "active_topics": active_topics
        }

    async def get_emotional_context_tool(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Tool implementation to get emotional context.
        
        Args:
            args: Optional arguments (unused)
            
        Returns:
            Dict with emotional context information
        """
        try:
            context = await self.get_emotional_context()
            
            if context:
                if len(context.get("recent_emotions", [])) >= 2:
                    emotions = context.get("recent_emotions", [])
                    if emotions:
                        trend = "steady"
                        sentiment_values = [e.get("sentiment", 0) for e in emotions]
                        
                        if len(sentiment_values) >= 2:
                            if sentiment_values[0] > sentiment_values[-1] + 0.2:
                                trend = "improving"
                            elif sentiment_values[0] < sentiment_values[-1] - 0.2:
                                trend = "declining"
                        
                        context["sentiment_trend"] = trend
                
                all_emotions = {}
                for emotion_data in context.get("recent_emotions", []):
                    for emotion, score in emotion_data.get("emotions", {}).items():
                        all_emotions[emotion] = all_emotions.get(emotion, 0) + score
                
                if all_emotions:
                    dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
                    context["dominant_emotion"] = dominant_emotion
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting emotional context: {e}")
            return {
                "error": str(e),
                "current_emotion": None,
                "recent_emotions": [],
                "emotional_triggers": {}
            }

    async def get_personal_details_tool(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Tool implementation to get personal details.
        
        Enhanced version with better category handling and confidence scores.
        
        Args:
            args: Arguments including:
                - category: Optional category to retrieve
            
        Returns:
            Dict with personal details
        """
        try:
            category = None
            if args and isinstance(args, dict):
                category = args.get("category")
            
            response = {
                "found": False,
                "details": {}
            }
            
            if not hasattr(self, "personal_details") or not self.personal_details:
                return response
            
            if category and category.lower() == "name":
                value = None
                confidence = 0
                if "name" in self.personal_details:
                    detail = self.personal_details["name"]
                    if isinstance(detail, dict) and "value" in detail:
                        value = detail["value"]
                        confidence = detail.get("confidence", 0.9)
                    else:
                        value = detail
                        confidence = 0.9
                
                if not value:
                    try:
                        name_memories = await self.search_memory(
                            "user name", 
                            limit=3, 
                            min_significance=0.8  # For backward compat
                        )
                        
                        for memory in name_memories:
                            content = memory.get("content", "")
                            patterns = [
                                r"User name: ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"User's name is ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"([A-Za-z]+(?: [A-Za-z]+){0,3}) is my name"
                            ]
                            
                            for pattern in patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    value = matches[0].strip()
                                    confidence = 0.85
                                    self.personal_details["name"] = {
                                        "value": value,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "memory_retrieval"
                                    }
                                    break
                            if value:
                                break
                    except Exception as e:
                        logger.error(f"Error searching memory for name: {e}")
                
                if value:
                    return {
                        "found": True,
                        "category": "name",
                        "value": value,
                        "confidence": confidence
                    }
            
            if category:
                category = category.lower()
                if category == "family":
                    if "family" in self.personal_details and self.personal_details["family"]:
                        family_data = {}
                        for relation, data in self.personal_details["family"].items():
                            if isinstance(data, dict) and "name" in data:
                                family_data[relation] = {
                                    "name": data["name"],
                                    "confidence": data.get("confidence", 0.85)
                                }
                            else:
                                family_data[relation] = {
                                    "name": data,
                                    "confidence": 0.85
                                }
                        return {
                            "found": True,
                            "category": "family",
                            "value": family_data,
                            "confidence": 0.9
                        }
                else:
                    if category in self.personal_details:
                        detail = self.personal_details[category]
                        if isinstance(detail, dict) and "value" in detail:
                            return {
                                "found": True,
                                "category": category,
                                "value": detail["value"],
                                "confidence": detail.get("confidence", 0.85)
                            }
                        else:
                            return {
                                "found": True,
                                "category": category,
                                "value": detail,
                                "confidence": 0.85
                            }
            
            formatted_details = {}
            for cat, detail in self.personal_details.items():
                if cat == "family":
                    family_data = {}
                    if detail:
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                family_data[relation] = data["name"]
                            else:
                                family_data[relation] = data
                    formatted_details[cat] = family_data
                else:
                    if isinstance(detail, dict) and "value" in detail:
                        formatted_details[cat] = detail["value"]
                    else:
                        formatted_details[cat] = detail
            
            response["found"] = len(formatted_details) > 0
            response["details"] = formatted_details
            response["count"] = len(formatted_details)
            
            return response
        except Exception as e:
            logger.error(f"Error getting personal details: {e}")
            return {
                "found": False,
                "error": str(e),
                "details": {}
            }

    def _get_timestamp(self) -> float:
        return time.time()

    async def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze and record emotions from text content"""
        # In a real implementation, you might integrate with an external service
        # or a local model. For now, we do a simple pass.
        return {}

    async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
        """
        Detect and analyze emotional context from text.
        This is a wrapper around the EmotionMixin functionality for the voice agent.
        
        Args:
            text: The text to analyze for emotional context
            
        Returns:
            Dict with emotional context information
        """
        if hasattr(self, "detect_emotion") and callable(getattr(self, "detect_emotion")):
            try:
                timestamp = time.time()
                emotional_context = {
                    "timestamp": timestamp,
                    "text": text,
                    "emotions": {},
                    "sentiment": 0.0,
                    "emotional_state": "neutral"
                }
                emotion = await self.detect_emotion(text)
                emotional_context["emotional_state"] = emotion
                emotional_context["emotions"][emotion] = 1.0
                
                if hasattr(self, "_emotional_history"):
                    self._emotional_history.append(emotional_context)
                    if hasattr(self, "_max_emotional_history") and len(self._emotional_history) > self._max_emotional_history:
                        self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                    
                logger.info(f"Detected emotion: {emotion} for text: {text[:30]}...")
                return emotional_context
            except Exception as e:
                logger.error(f"Error in detect_emotion: {e}")
                return {
                    "timestamp": time.time(),
                    "text": text,
                    "emotions": {"neutral": 1.0},
                    "sentiment": 0.0,
                    "emotional_state": "neutral",
                    "error": str(e)
                }
        
        return {
            "timestamp": time.time(),
            "text": text,
            "emotions": {"neutral": 1.0},
            "sentiment": 0.0,
            "emotional_state": "neutral"
        }

    async def store_memory(self,
                           content: str,
                           metadata: Dict[str, Any] = None,
                           quickrecal_score: float = None,
                           importance: float = None) -> bool:
        """
        Store a new memory with semantic embedding and ensure proper persistence.
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            quickrecal_score: Optional override for quickrecal_score
            importance: Alternate name for quickrecal_score (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Handle both quickrecal_score and importance for backward compatibility
            if quickrecal_score is None and importance is not None:
                quickrecal_score = importance
            
            # Get emotional context for the memory content
            emotional_context = None
            try:
                emotional_context = await self.detect_emotional_context(content)
                logger.info(f"Detected emotional context for memory: {emotional_context['emotional_state'] if emotional_context else 'None'}")
            except Exception as e:
                logger.warning(f"Error detecting emotional context: {e}")
            
            try:
                embedding, memory_quickrecal_score = await self.process_embedding(content)
                if quickrecal_score is not None:
                    try:
                        memory_quickrecal_score = float(quickrecal_score)
                        memory_quickrecal_score = max(0.0, min(1.0, memory_quickrecal_score))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid quickrecal_score value: {quickrecal_score}, using calculated value: {memory_quickrecal_score}")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                if hasattr(self, 'embedding_dim'):
                    embedding = torch.zeros(self.embedding_dim)
                else:
                    embedding = torch.zeros(384)
                memory_quickrecal_score = 0.5 if quickrecal_score is None else quickrecal_score
            
            memory_id = str(uuid.uuid4())
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add emotional context to metadata if available
            if emotional_context:
                metadata['emotional_context'] = {
                    'emotional_state': emotional_context.get('emotional_state', 'neutral'),
                    'sentiment': emotional_context.get('sentiment', 0.0),
                    'emotions': emotional_context.get('emotions', {})
                }
                
                # Adjust the quickrecal_score based on emotional intensity
                if 'sentiment' in emotional_context:
                    sentiment_intensity = abs(emotional_context['sentiment'])
                    # Boost quickrecal_score for emotionally intense memories
                    memory_quickrecal_score = max(memory_quickrecal_score, memory_quickrecal_score * (1 + sentiment_intensity * 0.3))
                    memory_quickrecal_score = min(1.0, memory_quickrecal_score)  # Cap at 1.0
            
            mem = {
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "timestamp": time.time(),
                "quickrecal_score": memory_quickrecal_score,
                "metadata": metadata
            }
            
            async with self._memory_lock:
                self.memories.append(mem)
                
            logger.info(f"Stored new memory with ID {memory_id} and quickrecal_score {memory_quickrecal_score:.2f}")
            
            if (memory_quickrecal_score >= 0.7 and
                hasattr(self, 'persistence_enabled') and
                self.persistence_enabled):
                logger.info(f"Forcing immediate persistence for high-quickrecal_score memory: {memory_id}")
                await self._persist_single_memory(mem)
                    
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False

    async def _persist_single_memory(self, memory: Dict[str, Any]) -> bool:
        if not hasattr(self, 'persistence_enabled') or not self.persistence_enabled:
            return False
            
        if not hasattr(self, 'storage_path'):
            logger.error("No storage path configured")
            return False
        try:
            memory_id = memory.get('id')
            if not memory_id:
                logger.warning("Cannot persist memory without ID")
                return False
                
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path, exist_ok=True)
                logger.info(f"Created storage directory: {self.storage_path}")
                
            file_path = self.storage_path / f"{memory_id}.json"
            temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
            backup_file_path = self.storage_path / f"{memory_id}.json.bak"
            
            memory_copy = copy.deepcopy(memory)
            
            if 'embedding' in memory_copy:
                embedding_data = memory_copy['embedding']
                if hasattr(embedding_data, 'tolist'):
                    memory_copy['embedding'] = embedding_data.tolist()
                elif hasattr(embedding_data, 'detach'):
                    memory_copy['embedding'] = embedding_data.detach().cpu().numpy().tolist()
                elif isinstance(embedding_data, np.ndarray):
                    memory_copy['embedding'] = embedding_data.tolist()
            
            memory_copy = self._convert_numpy_to_python(memory_copy)
            
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_copy, f, ensure_ascii=False, indent=2)
                
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_file_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
            
            os.replace(temp_file_path, file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    _ = json.load(f)
                logger.info(f"Successfully persisted memory {memory_id}")
                if backup_file_path.exists():
                    os.remove(backup_file_path)
                return True
            except json.JSONDecodeError:
                logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                if backup_file_path.exists():
                    try:
                        os.replace(backup_file_path, file_path)
                        logger.info(f"Restored memory {memory_id} from backup after verification failure")
                    except Exception as e:
                        logger.error(f"Failed to restore backup for memory {memory_id}: {e}")
                return False
        except Exception as e:
            logger.error(f"Error persisting single memory: {e}")
            return False

    #
    # Renamed Method (store_significant_memory -> store_quickrecal_memory)
    #
    async def store_quickrecal_memory(self, 
                                      text: str, 
                                      memory_type: str = "important", 
                                      metadata: Optional[Dict[str, Any]] = None,
                                      min_quickrecal_score: float = 0.7) -> Dict[str, Any]:
        """
        Store a memory with high quickrecal_score and guaranteed persistence.
        
        Args:
            text: The memory content
            memory_type: Type of memory to store
            metadata: Additional metadata
            min_quickrecal_score: Minimum quickrecal_score value
            
        Returns:
            Dict with memory information
        """
        if not text or not text.strip():
            logger.warning("Cannot store empty memory content")
            return {"success": False, "error": "Empty memory content"}
            
        try:
            embedding, auto_quickrecal_score = await self.process_embedding(text)
            quickrecal_score = max(auto_quickrecal_score, min_quickrecal_score)
            
            full_metadata = {
                "type": memory_type,
                "timestamp": time.time(),
                **(metadata or {})
            }
            
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": text,
                "embedding": embedding,
                "timestamp": time.time(),
                "quickrecal_score": quickrecal_score,
                "metadata": full_metadata
            }
            
            async with self._memory_lock:
                self.memories.append(memory)
                
            success = await self._persist_single_memory(memory)
            
            if success:
                logger.info(f"Stored quickrecal memory {memory_id} with quickrecal_score {quickrecal_score:.2f}")
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "quickrecal_score": quickrecal_score,
                    "type": memory_type
                }
            else:
                logger.error("Failed to persist quickrecal memory")
                return {"success": False, "error": "Persistence failed"}
        except Exception as e:
            logger.error(f"Error storing quickrecal memory: {e}")
            return {"success": False, "error": str(e)}

    #
    # Backward compatibility wrapper for store_significant_memory
    #
    async def store_significant_memory(self,
                                       text: str,
                                       memory_type: str = "important",
                                       metadata: Optional[Dict[str, Any]] = None,
                                       min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Store a significant memory with guaranteed persistence.
        
        Deprecated: Use store_quickrecal_memory instead.
        
        Args:
            text: The memory content
            memory_type: Type of memory to store
            metadata: Additional metadata
            min_significance: Minimum significance value (renamed to min_quickrecal_score)
            
        Returns:
            Dict with memory information
        """
        return await self.store_quickrecal_memory(
            text=text,
            memory_type=memory_type,
            metadata=metadata,
            min_quickrecal_score=min_significance
        )

    async def store_important_memory(self, content: str = "", quickrecal_score: float = 0.8) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory with guaranteed persistence.
        
        Args:
            content: The memory content to store
            quickrecal_score: Memory quickrecal_score (0.0-1.0)
            
        Returns:
            Dict with status
        """
        if not content:
            return {"success": False, "error": "No content provided"}
        
        try:
            # Re-use store_quickrecal_memory logic with the chosen quickrecal_score
            result = await self.store_quickrecal_memory(
                text=content,
                memory_type="important",
                min_quickrecal_score=quickrecal_score
            )
            return {
                "success": result.get("success", False),
                "memory_id": result.get("memory_id"),
                "error": result.get("error")
            }
        except Exception as e:
            logger.error(f"Error storing important memory: {e}")
            return {"success": False, "error": str(e)}

    async def classify_query(self, query: str) -> str:
        try:
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to classify_query")
                return "other"
            query = query.lower().strip()
            
            if re.search(r'\b(remember|recall|what did (i|you) say|previous|earlier|before)\b', query):
                return "recall"
            if re.search(r'\b(who|what|where|when|why|how|explain|tell me about|information)\b', query):
                return "information"
            if re.search(r'\b(remember this|note this|save this|store this|keep this|remember that)\b', query):
                return "new_learning"
            if re.search(r'\b(feel|feeling|sad|happy|angry|upset|depressed|worried|anxious|excited)\b', query):
                return "emotional"
            if re.search(r'\b(what do you mean|clarify|explain again|confused|didn\'t understand)\b', query):
                return "clarification"
            if re.search(r'\b(do this|perform|execute|run|start|stop|create|make|build)\b', query):
                return "task"
            if re.search(r'\b(hello|hi|hey|good morning|good afternoon|good evening|how are you)\b', query):
                return "greeting"
            
            try:
                if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                    llm_classification = await self._classify_with_llm(query)
                    if llm_classification in [
                        "recall", "information", "new_learning", "emotional",
                        "clarification", "task", "greeting", "other"
                    ]:
                        logger.info(f"LLM classified query as: {llm_classification}")
                        return llm_classification
            except Exception as e:
                logger.warning(f"Error during LLM classification, falling back to heuristics: {e}")
            
            return "information"
        except Exception as e:
            logger.error(f"Error classifying query: {e}", exc_info=True)
            return "other"
    
    async def _classify_with_llm(self, query: str) -> str:
        try:
            prompt = f"""Classify the following user query into exactly one of these categories: 
            recall, information, new_learning, emotional, clarification, task, greeting, other.
            Respond with only the category name, nothing else.
            
            Query: "{query}"
            
            Classification:"""
            
            response = await self.llm_pipeline.agenerate_text(
                prompt=prompt,
                max_tokens=10,
                temperature=0.2
            )
            
            if response and isinstance(response, str):
                response = response.lower().strip()
                valid_types = ["recall", "information", "new_learning", "emotional", 
                               "clarification", "task", "greeting", "other"]
                if response in valid_types:
                    return response
                    
            return "information"
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}", exc_info=True)
            return "information"
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_quickrecal_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the given query using semantic search.
        
        Args:
            query: The query text
            limit: Maximum number of memories to return
            min_quickrecal_score: Minimum quickrecal_score threshold
            
        Returns:
            List of dictionaries containing memory entries
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided to retrieve_memories")
                return []
                
            # Get memory embeddings
            try:
                embedding, _ = await self.process_embedding(query)
                if not embedding.any():  # Check if embedding is all zeros
                    logger.warning("Empty embedding returned for query")
                    return []
            except Exception as e:
                logger.error(f"Error getting embedding for query: {e}")
                return []
                
            # Retrieve memories by embedding similarity
            try:
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=embedding,
                    limit=limit,
                    min_quickrecal_score=min_quickrecal_score
                )
                logger.info(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
                return memories
            except Exception as e:
                logger.error(f"Error in standard memory retrieval: {e}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
    
    async def _retrieve_memories_by_embedding(self, query_embedding: List[float], limit: int = 5, 
                                              min_quickrecal_score: float = 0.3) -> List[Any]:
        try:
            if not hasattr(self, 'memory_db') or not self.memory_db:
                logger.warning("Memory database not initialized")
                return []
                
            memories = await self.memory_db.get_by_vector(
                vector=query_embedding,
                limit=limit * 2,
                collection="memories"
            )
            
            filtered_memories = []
            for memory in memories:
                if hasattr(memory, 'quickrecal_score') and memory.quickrecal_score >= min_quickrecal_score:
                    filtered_memories.append(memory)
                elif not hasattr(memory, 'quickrecal_score'):
                    filtered_memories.append(memory)
            
            sorted_memories = sorted(
                filtered_memories, 
                key=lambda x: x.similarity if hasattr(x, 'similarity') else 0.0,
                reverse=True
            )[:limit]
            
            return sorted_memories
        except Exception as e:
            logger.error(f"Error retrieving memories by embedding: {e}", exc_info=True)
            return []
    
    async def retrieve_information(self, query: str, limit: int = 5, min_quickrecal_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve factual information relevant to the query.
        Optimized for informational queries rather than personal memories.
        
        Args:
            query: The query text
            limit: Maximum number of information pieces to retrieve
            min_quickrecal_score: Minimum quickrecal_score threshold
            
        Returns:
            List of dictionaries containing information entries
        """
        
        try:
            logger.info(f"Retrieving information for query: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"Error in retrieve_information: {e}", exc_info=True)
            return []
            
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    info_results = await self.memory_integration.retrieve_information(
                        query=query,
                        limit=limit,
                        min_quickrecal_score=min_quickrecal_score
                    )
                    logger.info(f"Retrieved {len(info_results)} information entries from hierarchical memory")
                    return info_results
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical information store: {e}", exc_info=True)
            
            if hasattr(self, 'get_rag_context'):
                try:
                    context_data = await self.get_rag_context(
                        query=query,
                        limit=limit,
                        min_quickrecal_score=min_quickrecal_score,
                        max_tokens=1000  # example
                    )
                    if context_data and isinstance(context_data, str):
                        info_entries = self._parse_information_context(context_data)
                        if info_entries:
                            logger.info(f"Retrieved {len(info_entries)} information entries from RAG context")
                            return info_entries
                except Exception as e:
                    logger.error(f"Error retrieving RAG context for information: {e}", exc_info=True)
            
            try:
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit * 2,
                    min_quickrecal_score=min_quickrecal_score
                )
                
                result = []
                for memory in memories:
                    is_info_type = False
                    if hasattr(memory, 'memory_type'):
                        info_types = ["fact", "concept", "definition", "information", "knowledge"]
                        memory_type_value = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                        is_info_type = any(info_type in memory_type_value.lower() for info_type in info_types)
                    elif hasattr(memory, 'metadata') and memory.metadata:
                        if memory.metadata.get('type') in ['fact', 'information', 'knowledge']:
                            is_info_type = True
                    
                    if is_info_type or not hasattr(memory, 'memory_type'):
                        result.append({
                            "id": memory.id if hasattr(memory, 'id') else str(uuid.uuid4()),
                            "content": memory.content,
                            "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                            "quickrecal_score": getattr(memory, 'quickrecal_score', 0.5),
                            "memory_type": memory.memory_type.value if (hasattr(memory, 'memory_type') and hasattr(memory.memory_type, 'value')) else "information",
                            "metadata": getattr(memory, 'metadata', {})
                        })
                
                logger.info(f"Retrieved {len(result)} information entries from standard memory system")
                return result[:limit]
            except Exception as e:
                logger.error(f"Error in standard information retrieval: {e}", exc_info=True)
                return []
    
    def _parse_information_context(self, context_data: str) -> List[Dict[str, Any]]:
        try:
            info_entries = []
            sections = re.split(r'\n(?=Fact|Information|Knowledge|Definition|Concept)\s*\d*:', context_data)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                content = section.strip()
                entry_id = str(uuid.uuid4())
                entry_type = "information"
                confidence = 0.7
                
                type_match = re.search(r'^(Fact|Information|Knowledge|Definition|Concept)', content)
                if type_match:
                    entry_type = type_match.group(1).lower()
                
                confidence_match = re.search(r'confidence:\s*(\d+(\.\d+)?)', content, re.IGNORECASE)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        if confidence > 1:
                            confidence /= 100
                    except ValueError:
                        pass
                
                info_entries.append({
                    "id": entry_id,
                    "content": content,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "quickrecal_score": confidence,
                    "memory_type": entry_type,
                    "metadata": {
                        "source": "rag_context",
                        "confidence": confidence,
                        "index": i
                    }
                })
            
            return info_entries
        except Exception as e:
            logger.error(f"Error parsing information context: {e}", exc_info=True)
            return []
    
    async def store_and_retrieve(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Storing and retrieving context for: '{text[:50]}...'")
            
            if not text or not isinstance(text, str):
                logger.warning("Empty or invalid text provided to store_and_retrieve")
                return []
            
            if metadata is None:
                metadata = {
                    "source": "user_input",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "new_learning"
                }
            
            store_success = False
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    memory_id = await self.memory_integration.store(
                        content=text,
                        metadata=metadata
                    )
                    logger.info(f"Stored new information with ID {memory_id} in hierarchical memory")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in hierarchical memory: {e}", exc_info=True)
            
            if not store_success:
                try:
                    memory_id = await self.store_memory(
                        text=text,
                        metadata=metadata,
                        importance=0.7  # For backward compatibility
                    )
                    logger.info(f"Stored new information with ID {memory_id} in base memory system")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in base memory: {e}", exc_info=True)
            
            if not store_success:
                logger.warning("Failed to store new information")
                return []
            
            try:
                query_type = await self.classify_query(text)
                if query_type == "information":
                    related_memories = await self.retrieve_information(text, limit=3, min_quickrecal_score=0.2)
                else:
                    related_memories = await self.retrieve_memories(text, limit=3, min_quickrecal_score=0.2)
                
                logger.info(f"Retrieved {len(related_memories)} related memories")
                return related_memories
            except Exception as e:
                logger.error(f"Error retrieving related memories: {e}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error in store_and_retrieve: {e}", exc_info=True)
            return []
    
    async def store_emotional_context(self, emotional_context: Dict[str, Any]) -> bool:
        try:
            if not emotional_context or not isinstance(emotional_context, dict) or 'emotion' not in emotional_context:
                logger.warning("Invalid emotional context data provided")
                return False
            
            quickrecal_score = emotional_context.get('intensity', 0.5)
            metadata = {
                "type": "emotional_context",
                "primary_emotion": emotional_context.get('emotion'),
                "intensity": emotional_context.get('intensity', 0.5),
                "secondary_emotions": emotional_context.get('secondary_emotions', []),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if hasattr(self, '_emotional_history'):
                self._emotional_history.append(metadata)
                if hasattr(self, '_max_emotional_history') and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
            
            if quickrecal_score >= 0.4:
                text = emotional_context.get('text', f"User expressed {metadata['primary_emotion']} with intensity {metadata['intensity']}")
                store_success = False
                if hasattr(self, 'memory_integration') and self.memory_integration:
                    try:
                        memory_id = await self.memory_integration.store(
                            content=text,
                            metadata=metadata,
                            importance=quickrecal_score
                        )
                        logger.info(f"Stored emotional context with ID {memory_id} in hierarchical memory")
                        store_success = True
                    except Exception as e:
                        logger.error(f"Error storing emotional context in hierarchical memory: {e}", exc_info=True)
                
                if not store_success:
                    try:
                        memory_id = await self.store_memory(
                            text=text,
                            metadata=metadata,
                            quickrecal_score=quickrecal_score
                        )
                        logger.info(f"Stored emotional context with ID {memory_id} in base memory system")
                        store_success = True
                    except Exception as e:
                        logger.error(f"Error storing emotional context in base memory: {e}", exc_info=True)
            
            logger.info(f"Stored emotional context: {metadata['primary_emotion']} (intensity: {metadata['intensity']})")
            return True
        except Exception as e:
            logger.error(f"Error storing emotional context: {e}", exc_info=True)
            return False
    
    async def compare_texts(self, text1: str, text2: str) -> float:
        try:
            embedding1, _ = await self.process_embedding(text1)
            embedding2, _ = await self.process_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Failed to generate embeddings for similarity comparison")
                return 0.0
            
            if isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
                embedding1 = embedding1 / embedding1.norm()
                embedding2 = embedding2 / embedding2.norm()
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                if not isinstance(embedding1, np.ndarray):
                    embedding1 = np.array(embedding1)
                if not isinstance(embedding2, np.ndarray):
                    embedding2 = np.array(embedding2)
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
                similarity = np.dot(embedding1, embedding2)
            
            similarity = float(max(0.0, min(1.0, similarity)))
            return similarity
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0

    async def _handle_knowledge_graph_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.handle_lucidia_tool_call("query_knowledge_graph", args)
            return result
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return {"error": str(e), "success": False}
    
    async def _handle_self_model_reflection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.handle_lucidia_tool_call("self_model_reflection", args)
            return result
        except Exception as e:
            logger.error(f"Error in self model reflection: {e}")
            return {"error": str(e), "success": False}
    
    async def _handle_world_model_insight(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.handle_lucidia_tool_call("world_model_insight", args)
            return result
        except Exception as e:
            logger.error(f"Error retrieving world model insight: {e}")
            return {"error": str(e), "success": False}
    
    async def _handle_narrative_identity_insight(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await self.handle_lucidia_tool_call("narrative_identity_insight", args)
            return result
        except Exception as e:
            logger.error(f"Error retrieving narrative identity insight: {e}")
            return {"error": str(e), "success": False}
    
    async def integrate_lucidia_insights(self, prompt: str) -> Dict[str, Any]:
        try:
            insights = await self.get_lucidia_insights(prompt)
            concepts = prompt.split()
            kg_results = []
            for concept in concepts[:3]:
                if len(concept) > 4:
                    try:
                        kg_result = await self.handle_lucidia_tool_call(
                            "query_knowledge_graph", 
                            {"query": concept, "max_results": 2}
                        )
                        if kg_result.get("result"):
                            kg_results.append(kg_result["result"])
                    except Exception as e:
                        logger.warning(f"Error querying knowledge graph for concept '{concept}': {e}")
            
            if kg_results:
                insights["knowledge_graph_queries"] = kg_results
                
            return insights
        except Exception as e:
            logger.error(f"Error integrating Lucidia insights: {e}")
            return {"error": str(e)}
    
    async def get_lucidia_memory_tools(self) -> List[Dict[str, Any]]:
        tools = []
        
        knowledge_graph_tool = {
            "type": "function",
            "function": {
                "name": "query_knowledge_graph",
                "description": "Query the knowledge graph for concepts, relationships, and insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The concept or query string to search for in the knowledge graph"
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Optional type of relationship to filter by"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        self_model_tool = {
            "type": "function",
            "function": {
                "name": "self_model_reflection",
                "description": "Access the self model for introspection and reflection capabilities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection_type": {
                            "type": "string",
                            "description": "Type of reflection to perform (identity, goals, beliefs, emotions) or similar",
                            "enum": ["identity", "goals", "beliefs", "emotions", "values", "capabilities"]
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional specific query or aspect to reflect upon"
                        }
                    },
                    "required": ["reflection_type"]
                }
            }
        }
        
        world_model_tool = {
            "type": "function",
            "function": {
                "name": "world_model_insight",
                "description": "Get insights from the world model about concepts, domains, and relationships",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "The concept, entity, or topic to explore"
                        },
                        "perspective": {
                            "type": "string",
                            "description": "Optional perspective to view the concept from (e.g., 'historical', 'cultural', 'scientific')"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Optional depth of insight (1-5, where 5 is most detailed)",
                            "enum": [1, 2, 3, 4, 5],
                            "default": 3
                        }
                    },
                    "required": ["concept"]
                }
            }
        }
        
        narrative_identity_tool = {
            "type": "function",
            "function": {
                "name": "narrative_identity_insight",
                "description": "Get insights from the narrative identity system about the user's identity and experiences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query or prompt to analyze for narrative identity insights"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        record_experience_tool = {
            "type": "function",
            "function": {
                "name": "record_identity_experience",
                "description": "Record a significant experience in the narrative identity system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the experience to record"
                        },
                        "significance": {
                            "type": "number",
                            "description": "Importance of this experience (0-1)",
                            "default": 0.7
                        },
                        "emotion": {
                            "type": "string",
                            "description": "Emotional tone of the experience",
                            "default": "neutral"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional contextual information",
                            "default": {}
                        }
                    },
                    "required": ["content"]
                }
            }
        }
        
        get_narrative_tool = {
            "type": "function",
            "function": {
                "name": "get_identity_narrative",
                "description": "Generate a narrative about the identity based on autobiographical memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "narrative_type": {
                            "type": "string",
                            "description": "Type of narrative to generate",
                            "enum": ["complete", "core", "growth", "personal"],
                            "default": "complete"
                        },
                        "style": {
                            "type": "string",
                            "description": "Style of the narrative",
                            "enum": ["neutral", "reflective", "conversational"],
                            "default": "neutral"
                        }
                    }
                }
            }
        }
        
        get_timeline_tool = {
            "type": "function",
            "function": {
                "name": "get_autobiographical_timeline",
                "description": "Retrieve the autobiographical timeline of experiences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of experiences to return",
                            "default": 10
                        }
                    }
                }
            }
        }
        
        tools.append(knowledge_graph_tool)
        tools.append(self_model_tool)
        tools.append(world_model_tool)
        tools.append(narrative_identity_tool)
        tools.append(record_experience_tool)
        tools.append(get_narrative_tool)
        tools.append(get_timeline_tool)
        
        return tools
    
    async def handle_lucidia_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            if tool_name == "query_knowledge_graph":
                query = args.get("query", "")
                relation_type = args.get("relation_type", None)
                max_results = args.get("max_results", 5)
                if not query:
                    return {"error": "Query parameter is required", "success": False}
                result = await asyncio.to_thread(
                    self.knowledge_graph.query,
                    query=query,
                    relation_type=relation_type,
                    max_results=max_results
                )
                return {
                    "result": result,
                    "success": True,
                    "execution_time": time.time() - start_time
                }
            elif tool_name == "self_model_reflection":
                reflection_type = args.get("reflection_type", "")
                query = args.get("query", "")
                if not reflection_type:
                    return {"error": "Reflection type parameter is required", "success": False}
                result = await asyncio.to_thread(
                    self.self_model.reflect,
                    reflection_type=reflection_type,
                    query=query
                )
                return {
                    "result": result,
                    "success": True,
                    "execution_time": time.time() - start_time
                }
            elif tool_name == "world_model_insight":
                concept = args.get("concept", "")
                perspective = args.get("perspective", None)
                depth = args.get("depth", 3)
                if not concept:
                    return {"error": "Concept parameter is required", "success": False}
                result = await asyncio.to_thread(
                    self.world_model.get_insight,
                    concept=concept,
                    perspective=perspective,
                    depth=depth
                )
                return {
                    "result": result,
                    "success": True,
                    "execution_time": time.time() - start_time
                }
            elif tool_name == "narrative_identity_insight":
                query = args.get("query", "")
                if not query:
                    return {"error": "Query parameter is required", "success": False}
                result = await asyncio.to_thread(
                    self.narrative_identity.get_insight,
                    query=query
                )
                return {
                    "result": result,
                    "success": True,
                    "execution_time": time.time() - start_time
                }
            else:
                return {"error": f"Unknown Lucidia tool: {tool_name}", "success": False}
        except AttributeError as e:
            logger.error(f"Lucidia component not properly initialized: {e}")
            return {
                "error": "Lucidia memory system component not properly initialized",
                "details": str(e),
                "success": False
            }
        except Exception as e:
            logger.error(f"Error handling Lucidia tool call {tool_name}: {e}", exc_info=True)
            return {
                "error": f"Error executing Lucidia tool: {str(e)}",
                "success": False
            }
    
    async def get_enhanced_rag_context(self, query: str, context_type: str = "comprehensive", max_tokens: int = 2048) -> Dict[str, Any]:
        start_time = time.time()
        
        result = {
            "context": "",
            "metadata": {
                "context_type": context_type,
                "sources": [],
                "token_count": 0,
                "generation_time": 0
            }
        }
        try:
            if context_type == "auto":
                if await self._is_personal_query(query):
                    context_type = "personal"
                elif await self._is_memory_query(query):
                    context_type = "comprehensive"
                else:
                    context_type = "factual"
            
            result["metadata"]["context_type"] = context_type
            
            token_allocation = {}
            if context_type == "comprehensive":
                token_allocation = {
                    "standard_memory": 0.25,
                    "lucidia_knowledge": 0.20,
                    "lucidia_self": 0.15,
                    "lucidia_world": 0.15,
                    "narrative_identity": 0.15,
                    "emotional": 0.10
                }
            elif context_type == "personal":
                token_allocation = {
                    "standard_memory": 0.20,
                    "lucidia_knowledge": 0.10,
                    "lucidia_self": 0.25,
                    "lucidia_world": 0.10,
                    "narrative_identity": 0.25,
                    "emotional": 0.10
                }
            elif context_type == "factual":
                token_allocation = {
                    "standard_memory": 0.20,
                    "lucidia_knowledge": 0.25,
                    "lucidia_self": 0.10,
                    "lucidia_world": 0.30,
                    "narrative_identity": 0.10,
                    "emotional": 0.05
                }
            
            tasks = []
            
            standard_token_budget = int(max_tokens * token_allocation.get("standard_memory", 0.2))
            tasks.append(self.get_rag_context(query=query, max_tokens=standard_token_budget))
            
            if hasattr(self.knowledge_graph, 'generate_context'):
                kg_token_budget = int(max_tokens * token_allocation.get("lucidia_knowledge", 0.2))
                tasks.append(asyncio.to_thread(
                    self.knowledge_graph.generate_context,
                    query=query,
                    max_tokens=kg_token_budget
                ))
            
            if hasattr(self.self_model, 'generate_context'):
                self_token_budget = int(max_tokens * token_allocation.get("lucidia_self", 0.15))
                tasks.append(asyncio.to_thread(
                    self.self_model.generate_context,
                    query=query,
                    max_tokens=self_token_budget
                ))
            
            if hasattr(self.world_model, 'generate_context'):
                world_token_budget = int(max_tokens * token_allocation.get("lucidia_world", 0.15))
                tasks.append(asyncio.to_thread(
                    self.world_model.generate_context,
                    query=query,
                    max_tokens=world_token_budget
                ))
            
            if hasattr(self.narrative_identity, 'generate_context'):
                narrative_token_budget = int(max_tokens * token_allocation.get("narrative_identity", 0.15))
                tasks.append(asyncio.to_thread(
                    self.narrative_identity.generate_context,
                    query=query,
                    max_tokens=narrative_token_budget
                ))
            
            context_components = await asyncio.gather(*tasks, return_exceptions=True)
            
            context_parts = []
            for i, component in enumerate(context_components):
                if isinstance(component, Exception):
                    logger.error(f"Error generating context component {i}: {component}")
                    continue
                
                if component:
                    if i == 0:
                        context_parts.append("### Memory Context")
                        context_parts.append(component)
                        result["metadata"]["sources"].append("standard_memory")
                    elif i == 1:
                        context_parts.append("### Knowledge Graph Insights")
                        context_parts.append(component)
                        result["metadata"]["sources"].append("lucidia_knowledge_graph")
                    elif i == 2:
                        context_parts.append("### Self-Awareness Context")
                        context_parts.append(component)
                        result["metadata"]["sources"].append("lucidia_self_model")
                    elif i == 3:
                        context_parts.append("### World Knowledge")
                        context_parts.append(component)
                        result["metadata"]["sources"].append("lucidia_world_model")
                    elif i == 4:
                        context_parts.append("### Narrative Identity")
                        context_parts.append(component)
                        result["metadata"]["sources"].append("lucidia_narrative_identity")
            
            result["context"] = "\n\n".join(context_parts)
            
            result["metadata"]["token_count"] = len(result["context"]) // 4
            result["metadata"]["generation_time"] = time.time() - start_time
            
            logger.info(f"Enhanced RAG context generated in {result['metadata']['generation_time']:.2f}s with {len(result['metadata']['sources'])} sources")
        except Exception as e:
            logger.error(f"Error generating enhanced RAG context: {e}")
            result["context"] = ""
            result["metadata"]["error"] = str(e)
        
        return result
    
    async def record_identity_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "content" not in experience_data:
                return {
                    "success": False,
                    "error": "Experience content is required"
                }
            
            experience = {
                "content": experience_data["content"],
                "significance": experience_data.get("significance", 0.7),
                "emotion": experience_data.get("emotion", "neutral"),
                "metadata": experience_data.get("metadata", {}),
                "timestamp": time.time()
            }
            
            result = await asyncio.to_thread(
                self.narrative_identity.add_experience,
                experience=experience
            )
            
            return {
                "success": True,
                "experience_id": result.get("id"),
                "message": "Experience successfully recorded in narrative identity"
            }
        except Exception as e:
            logger.error(f"Error recording identity experience: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_identity_narrative(self, narrative_type: str = "complete", style: str = "neutral") -> Dict[str, Any]:
        try:
            narrative = await asyncio.to_thread(
                self.narrative_identity.generate_narrative,
                narrative_type=narrative_type,
                style=style
            )
            
            return {
                "success": True,
                "narrative": narrative.get("text", ""),
                "metadata": {
                    "type": narrative_type,
                    "style": style,
                    "word_count": len(narrative.get("text", "").split()),
                    "generation_time": narrative.get("generation_time", 0),
                    "key_themes": narrative.get("themes", [])
                }
            }
        except Exception as e:
            logger.error(f"Error generating identity narrative: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_autobiographical_timeline(self, limit: int = 10) -> Dict[str, Any]:
        try:
            timeline = await asyncio.to_thread(
                self.narrative_identity.get_timeline,
                limit=limit
            )
            
            return {
                "success": True,
                "timeline": timeline,
                "count": len(timeline)
            }
        except Exception as e:
            logger.error(f"Error retrieving autobiographical timeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "timeline": []
            }

    async def retrieve_emotional_memories_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool implementation to retrieve memories by emotional context.
        
        Args:
            args: Dictionary containing:
                - emotion: Optional specific emotion to filter by
                - sentiment_threshold: Optional minimum sentiment value
                - sentiment_direction: Optional direction of sentiment ('positive' or 'negative')
                - limit: Optional maximum number of memories to return
                - min_quickrecal_score: Optional minimum quickrecal_score threshold
                
        Returns:
            Dict with retrieved memories and status information
        """
        try:
            # Extract and validate parameters
            emotion = args.get('emotion')
            sentiment_threshold = args.get('sentiment_threshold')
            sentiment_direction = args.get('sentiment_direction')
            limit = int(args.get('limit', 5))
            min_quickrecal_score = float(args.get('min_quickrecal_score', 0.0))
            
            # Call the retrieve_memories_by_emotion method
            memories = await self.retrieve_memories_by_emotion(
                emotion=emotion,
                sentiment_threshold=sentiment_threshold,
                sentiment_direction=sentiment_direction,
                limit=limit,
                min_quickrecal_score=min_quickrecal_score
            )
            
            # Format the results
            results = []
            for mem in memories:
                emotional_context = mem.get('metadata', {}).get('emotional_context', {})
                results.append({
                    "content": mem.get('content'),
                    "quickrecal_score": mem.get('quickrecal_score'),
                    "timestamp": mem.get('timestamp'),
                    "emotional_state": emotional_context.get('emotional_state', 'neutral'),
                    "sentiment": emotional_context.get('sentiment', 0.0),
                    "emotions": emotional_context.get('emotions', {})
                })
            
            return {
                "status": "success",
                "count": len(results),
                "emotion_filter": emotion,
                "sentiment_threshold": sentiment_threshold,
                "sentiment_direction": sentiment_direction,
                "memories": results
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_emotional_memories_tool: {e}")
            return {
                "status": "error",
                "error": str(e),
                "count": 0,
                "memories": []
            }

    async def retrieve_memories_by_emotion(self, 
                                 emotion: str = None, 
                                 sentiment_threshold: float = None,
                                 sentiment_direction: str = None,
                                 limit: int = 5, 
                                 min_quickrecal_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on emotional context.
        
        Args:
            emotion: Specific emotion to filter by (e.g., 'joy', 'anger')
            sentiment_threshold: Minimum absolute sentiment value to filter by
            sentiment_direction: Direction of sentiment ('positive', 'negative', or None for both)
            limit: Maximum number of memories to return
            min_quickrecal_score: Minimum quickrecal_score threshold
            
        Returns:
            List of dictionaries containing memory entries with matching emotional context
        """
        try:
            results = []
            async with self._memory_lock:
                memories = copy.deepcopy(self.memories)
                
            # Filter memories that have emotional context metadata
            filtered_memories = [
                mem for mem in memories 
                if (mem.get('metadata', {}).get('emotional_context') is not None and
                    mem.get('quickrecal_score', 0) >= min_quickrecal_score)
            ]
            
            # Further filter by specific emotion if provided
            if emotion:
                filtered_memories = [
                    mem for mem in filtered_memories
                    if (mem.get('metadata', {}).get('emotional_context', {}).get('emotional_state') == emotion or
                        emotion in mem.get('metadata', {}).get('emotional_context', {}).get('emotions', {}))
                ]
            
            # Filter by sentiment threshold and direction if provided
            if sentiment_threshold is not None:
                threshold = abs(float(sentiment_threshold))
                
                if sentiment_direction == 'positive':
                    filtered_memories = [
                        mem for mem in filtered_memories
                        if mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0) >= threshold
                    ]
                elif sentiment_direction == 'negative':
                    filtered_memories = [
                        mem for mem in filtered_memories
                        if mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0) <= -threshold
                    ]
                else:
                    # If no direction specified, filter by absolute sentiment value
                    filtered_memories = [
                        mem for mem in filtered_memories
                        if abs(mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0)) >= threshold
                    ]
            
            # Sort by quickrecal_score and timestamp for consistent results
            filtered_memories.sort(key=lambda x: (x.get('quickrecal_score', 0), x.get('timestamp', 0)), reverse=True)
            
            # Return the top memories based on limit
            results = filtered_memories[:limit]
            
            logger.info(f"Retrieved {len(results)} memories with emotion filter: {emotion} and sentiment threshold: {sentiment_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories by emotion: {e}")
            return []

    async def store_system_event(self, event_name: str, data: dict) -> None:
        """Store a system-level event (e.g., interrupts, state changes) in memory.
        
        This method logs significant system events that might be useful for analysis,
        debugging, or measuring system behavior over time.
        
        Args:
            event_name: The name/type of the event being logged
            data: A dictionary containing event-specific data
        """
        self.logger.info(f"System event: {event_name} - {data}")
        
        # Create the event record
        event = {
            "event_name": event_name,
            "data": data,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        
        # Store in event collection if using a database backend
        # For now, just log it - we can add persistence in a future update
        # await self._store_event(event)  # This would be implemented if persistence is needed
