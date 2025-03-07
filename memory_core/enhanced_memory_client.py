# memory_core/enhanced_memory_client.py

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import re
import time
import asyncio
import json
import uuid
from memory_core.base import BaseMemoryClient
from memory_core.tools import ToolsMixin
from memory_core.emotion import EmotionMixin
from memory_core.connectivity import ConnectivityMixin
from memory_core.personal_details import PersonalDetailsMixin
from memory_core.rag_context import RAGContextMixin

# Configure logger
logger = logging.getLogger(__name__)

class EnhancedMemoryClient(BaseMemoryClient,
                           ToolsMixin,
                           EmotionMixin,
                           ConnectivityMixin,
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
        # This is important to ensure each mixin initializes its own state
        # We need to initialize before calling __init__ on the mixins to avoid redundant initialization
        self._connected = False
        
        # Now initialize mixins
        PersonalDetailsMixin.__init__(self)
        EmotionMixin.__init__(self)
        # ConnectivityMixin.__init__(self)  # This is done implicitly since it doesn't have an __init__
        ToolsMixin.__init__(self)
        RAGContextMixin.__init__(self)
        
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
            # Process for personal details
            await self.detect_and_store_personal_details(text, role)
            
            # Process for emotions
            await self.analyze_emotions(text)
        
        # Store message in memory with default significance
        await self.store_memory(
            content=text,
            metadata={"role": role, "type": "message", "timestamp": time.time()}
        )
        
        # Add to conversation history with pruning
        self._conversation_history.append({
            "role": role,
            "content": text,
            "timestamp": time.time()
        })
        
        # Prune history if needed
        if len(self._conversation_history) > self._max_history_items:
            self._conversation_history = self._conversation_history[-self._max_history_items:]
        
        logger.debug(f"Processed {role} message: {text[:50]}...")
    
    async def get_memory_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get all memory tools formatted for the LLM.
        
        Returns:
            Dict with all available memory tools
        """
        # Get standard memory tools
        memory_tools = await self.get_memory_tools()
        
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
                            "description": "Optional category of personal detail to retrieve (e.g., 'name', 'location', 'birthday', 'job', 'family')"
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
        
        # Add topic tracking tool
        topic_tool = {
            "type": "function",
            "function": {
                "name": "track_conversation_topic",
                "description": "Track the current conversation topic to prevent repetition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic being discussed"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance of this topic (0-1)",
                            "default": 0.7
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
        
        memory_tools.append(personal_tool)
        memory_tools.append(emotion_tool)
        memory_tools.append(topic_tool)
        
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
        # Tool dispatcher mapping
        tool_handlers = {
            "search_memory": self.search_memory_tool,
            "store_important_memory": self.store_important_memory,
            "get_important_memories": self.get_important_memories,
            "get_personal_details": self.get_personal_details_tool,
            "get_emotional_context": self.get_emotional_context_tool,
            "track_conversation_topic": self.track_conversation_topic
        }
        
        # Get the appropriate handler
        handler = tool_handlers.get(tool_name)
        
        if not handler:
            logger.warning(f"Unknown tool call: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}", "success": False}
        
        # Handle search_memory tool specially due to its different parameter signature
        if tool_name == "search_memory":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            min_significance = args.get("min_significance", 0.0)
            return await handler(query=query, max_results=limit, min_significance=min_significance)
        
        # Handle store_important_memory tool
        elif tool_name == "store_important_memory":
            content = args.get("content", "")
            significance = args.get("significance", 0.8)
            return await handler(content=content, significance=significance)
        
        # Handle get_important_memories tool
        elif tool_name == "get_important_memories":
            limit = args.get("limit", 5)
            min_significance = args.get("min_significance", 0.7)
            return await handler(limit=limit, min_significance=min_significance)
        
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
            
            # Sanitize min_significance
            if "min_significance" in args:
                try:
                    args["min_significance"] = max(0.0, min(float(args["min_significance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["min_significance"] = 0.0  # Default if invalid
        
        elif tool_name == "store_important_memory":
            if "content" not in args or not args["content"]:
                return {"error": "Content is required for store_important_memory tool"}
            
            # Sanitize significance
            if "significance" in args:
                try:
                    args["significance"] = max(0.0, min(float(args["significance"]), 1.0))  # Range 0-1
                except (ValueError, TypeError):
                    args["significance"] = 0.8  # Default if invalid
        
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

    async def store_transcript(self, text: str, sender: str = "user", significance: float = None, role: str = None) -> bool:
        """
        Store a transcript entry in memory.
        
        This method stores conversation transcripts with appropriate metadata
        and automatically calculates significance if not provided.
        
        Args:
            text: The transcript text to store
            sender: Who sent the message (user or assistant)
            significance: Optional pre-calculated significance value
            role: Alternative name for sender parameter (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not text or not text.strip():
            logger.warning("Empty transcript text provided")
            return False
            
        try:
            # Handle role parameter for backward compatibility
            if role is not None and sender == "user":
                sender = role
                
            # Calculate significance if not provided
            if significance is None:
                # Use a higher base significance for user messages
                base_significance = 0.6 if sender.lower() == "user" else 0.4
                
                # Adjust based on text length (longer messages often have more content)
                length_factor = min(len(text) / 100, 0.3)  # Cap at 0.3
                
                # Check for question marks (questions are often important)
                question_factor = 0.15 if "?" in text else 0.0
                
                # Check for exclamation marks (emotional content)
                emotion_factor = 0.1 if "!" in text else 0.0
                
                # Check for personal information markers
                personal_info_markers = ["my name", "I live", "my address", "my phone", "my email", "my birthday"]
                personal_factor = 0.2 if any(marker in text.lower() for marker in personal_info_markers) else 0.0
                
                # Final significance calculation (capped at 0.95)
                significance = min(base_significance + length_factor + question_factor + emotion_factor + personal_factor, 0.95)
            
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
                significance=significance
            )
            
            if success:
                logger.info(f"Stored transcript from {sender} with significance {significance:.2f}")
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
            # Initialize result flag
            details_found = False
            
            # Define comprehensive patterns for different personal details using improved regex patterns
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
            
            # Family patterns need special handling
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
            
            # Initialize family data if not already present
            if hasattr(self, "personal_details") and "family" not in self.personal_details:
                self.personal_details["family"] = {}
            
            # Process standard patterns
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Take the first match and clean it
                        value = matches[0].strip().rstrip('.,:;!?')
                        
                        # Skip very short or likely invalid values
                        if len(value) < 2 or value.lower() in ["a", "an", "the", "me", "i", "my", "he", "she", "they"]:
                            continue
                            
                        # Validate based on category
                        if category == "email" and not re.match(r"[\w.+-]+@[\w-]+\.[\w.-]+", value):
                            continue
                        
                        if category == "age" and (not value.isdigit() or int(value) > 120 or int(value) < 1):
                            continue
                            
                        # Store the detail with confidence score
                        if hasattr(self, "personal_details"):
                            confidence = 0.9  # High confidence for clear pattern matches
                            
                            # Store with appropriate metadata
                            self.personal_details[category] = {
                                "value": value,
                                "confidence": confidence,
                                "timestamp": time.time(),
                                "source": "explicit_mention"
                            }
                            
                            logger.info(f"Stored personal detail: {category}={value} (confidence: {confidence:.2f})")
                            details_found = True
                        
                        # Also store as a high-significance memory
                        await self.store_memory(
                            content=f"User {category}: {value}",
                            significance=0.9,
                            metadata={
                                "type": "personal_detail",
                                "category": category,
                                "value": value,
                                "confidence": 0.9,
                                "timestamp": time.time()
                            }
                        )
                        
                        # Only process one match per category
                        break
            
            # Process family patterns (they have a nested structure)
            for relation, pattern_list in family_patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            # Family patterns can return different structures based on the specific regex
                            if len(match) >= 2:
                                # Handle reversed patterns like "(name) is my (relation)"
                                if match[1] in ["wife", "husband", "spouse", "partner", "son", "daughter", 
                                               "child", "mother", "father", "mom", "dad", "parent",
                                               "brother", "sister", "sibling"]:
                                    name = match[0].strip().rstrip('.,:;!?')
                                    relation_type = match[1].lower()
                                else:
                                    relation_type = match[0].lower()
                                    name = match[1].strip().rstrip('.,:;!?')
                                
                                # Skip very short or likely invalid values
                                if len(name) < 2 or name.lower() in ["a", "an", "the", "me", "i", "my"]:
                                    continue
                                    
                                # Store in family dictionary
                                if hasattr(self, "personal_details") and "family" in self.personal_details:
                                    confidence = 0.9  # High confidence for clear pattern matches
                                    
                                    # Use a consistent structure for family entries
                                    self.personal_details["family"][relation_type] = {
                                        "name": name,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "explicit_mention"
                                    }
                                    
                                    logger.info(f"Stored family detail: {relation_type}={name} (confidence: {confidence:.2f})")
                                    details_found = True
                                
                                # Also store as a memory
                                await self.store_memory(
                                    content=f"User's {relation_type}: {name}",
                                    significance=0.85,
                                    metadata={
                                        "type": "personal_detail",
                                        "category": "family",
                                        "relation_type": relation_type,
                                        "value": name,
                                        "confidence": 0.9,
                                        "timestamp": time.time()
                                    }
                                )
            
            return details_found
            
        except Exception as e:
            logger.error(f"Error detecting personal details: {e}")
            return False

    async def get_rag_context(self, query: str = None, limit: int = 5, min_significance: float = 0.0, max_tokens: int = None) -> str:
        """
        Get memory context for LLM RAG (Retrieval-Augmented Generation).
        
        Enhanced version with better categorization, formatting, and relevance.
        
        Args:
            query: Optional query to filter memories
            limit: Maximum number of memories to include
            min_significance: Minimum significance threshold for memories
            max_tokens: Maximum number of tokens to include (approximate)
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            max_tokens = max_tokens or limit * 100  # Default token limit based on memory count
            
            # Determine if this is a personal query
            is_personal_query = await self._is_personal_query(query) if query else False
            
            # Determine if this is a memory recall or history query
            is_memory_query = await self._is_memory_query(query) if query else False
            
            # Initialize context parts
            context_parts = []
            context_sections = {}
            
            # Generate different types of context based on query type
            if is_personal_query:
                # For personal queries, prioritize user information
                personal_context = await self._generate_personal_context(query)
                if personal_context:
                    context_sections["personal"] = personal_context
                    context_parts.append("### User Personal Information")
                    context_parts.append(personal_context)
            
            if is_memory_query:
                # For memory queries, provide more comprehensive memory context
                memory_limit = limit * 2  # Double the limit for memory-specific queries
                memory_context = await self._generate_memory_context(query, memory_limit, min_significance)
                if memory_context:
                    context_sections["memory"] = memory_context
                    context_parts.append("### Memory Recall")
                    context_parts.append(memory_context)
            
            # Add emotional context if appropriate (for emotional or personal queries)
            if is_personal_query or (query and any(keyword in query.lower() for keyword in ["feel", "emotion", "mood", "happy", "sad", "angry"])):
                emotional_context = await self._generate_emotional_context()
                if emotional_context:
                    context_sections["emotional"] = emotional_context
                    context_parts.append("### Recent Emotional States")
                    context_parts.append(emotional_context)
            
            # Add standard memory context if no specialized context was added or as supplement
            if not context_parts or (not is_memory_query and not is_personal_query):
                standard_context = await self._generate_standard_context(query, limit, min_significance)
                if standard_context:
                    context_sections["standard"] = standard_context
                    if not context_parts:  # If no sections yet, add a general header
                        context_parts.append("### Relevant Memory Context")
                    context_parts.append(standard_context)
            
            # Combine context parts
            context = "\n\n".join(context_parts)
            
            # Truncate if too long (approximate token count based on characters)
            char_limit = max_tokens * 4  # Rough estimate: 1 token ≈ 4 characters
            if len(context) > char_limit:
                # Try to preserve complete sections
                truncated_context = []
                current_length = 0
                
                for part in context_parts:
                    if current_length + len(part) <= char_limit:
                        truncated_context.append(part)
                        current_length += len(part) + 2  # +2 for newlines
                    else:
                        # For the last section, include as much as possible
                        if not truncated_context:  # Ensure at least one section
                            truncation_point = char_limit - current_length
                            truncated_part = part[:truncation_point] + "...\n[Context truncated due to length]"
                            truncated_context.append(truncated_part)
                        else:
                            truncated_context.append("[Additional context truncated due to length]")
                        break
                
                context = "\n\n".join(truncated_context)
            
            logger.info(f"Generated RAG context with {sum(1 for v in context_sections.values() for _ in v.split('•') if '•' in v)} memories")
            return context
            
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            # Return empty string on error rather than propagating the exception
            return ""
    
    async def _is_personal_query(self, query: str) -> bool:
        """Determine if a query is asking for personal information."""
        personal_keywords = [
            "my name", "who am i", "where do i live", "where am i from",
            "how old am i", "what's my age", "what's my birthday", "when was i born",
            "what do i do", "what's my job", "what's my profession", "who is my",
            "my family", "my spouse", "my partner", "my children", "my parents",
            "my email", "my phone", "my number", "my address"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in personal_keywords)
    
    async def _is_memory_query(self, query: str) -> bool:
        """Determine if a query is asking about past memories or history."""
        memory_keywords = [
            "remember", "recall", "forget", "memory", "memories", "mentioned",
            "said earlier", "talked about", "discussed", "told me about",
            "what did i say", "what did you say", "what did we discuss",
            "earlier", "before", "previously", "last time", "yesterday",
            "last week", "last month", "last year", "in the past"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in memory_keywords)
    
    async def _generate_personal_context(self, query: str) -> str:
        """Generate context about personal details."""
        if not hasattr(self, "personal_details") or not self.personal_details:
            return ""
        
        parts = []
        
        # Add clear header to indicate whose information this is
        parts.append("# USER PERSONAL INFORMATION")
        parts.append("# The following information is about the USER, not about the assistant")
        parts.append("# The assistant's name is Lucidia")
        
        # Determine specific personal detail being asked about
        personal_categories = {
            "name": ["name", "call me", "who am i"],
            "location": ["live", "from", "address", "location", "home"],
            "birthday": ["birthday", "born", "birth date", "age"],
            "job": ["job", "work", "profession", "career", "occupation"],
            "family": ["family", "spouse", "partner", "husband", "wife", "child", "children", "kid", "kids", "parent", "mother", "father"],
            "email": ["email", "e-mail", "mail"],
            "phone": ["phone", "number", "mobile", "cell"]
        }
        
        # Check if query targets a specific category
        target_category = None
        query_lower = query.lower()
        for category, keywords in personal_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                target_category = category
                break
        
        # If we have a specific target, focus on that
        if target_category:
            if target_category == "family":
                # Handle family data specially
                if "family" in self.personal_details and self.personal_details["family"]:
                    parts.append("USER's family information:")
                    for relation, data in self.personal_details["family"].items():
                        if isinstance(data, dict) and "name" in data:
                            parts.append(f"• USER's {relation}: {data['name']}")
                        else:
                            parts.append(f"• USER's {relation}: {data}")
            else:
                # Handle standard categories
                if target_category in self.personal_details:
                    detail = self.personal_details[target_category]
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {target_category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {target_category}: {detail}")
        else:
            # No specific target, include all personal details
            for category, detail in self.personal_details.items():
                if category == "family":
                    # Handle family data specially
                    if detail:
                        parts.append("USER's family information:")
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                parts.append(f"• USER's {relation}: {data['name']}")
                            else:
                                parts.append(f"• USER's {relation}: {data}")
                else:
                    # Handle standard categories
                    if isinstance(detail, dict) and "value" in detail:
                        parts.append(f"USER's {category}: {detail['value']}")
                    else:
                        parts.append(f"USER's {category}: {detail}")
        
        # Add clarity reminder at the end
        parts.append("\n# IMPORTANT: The above information is about the USER, not the assistant")
        parts.append("# The assistant's name is Lucidia\n")
        
        return "\n".join(parts)
    
    async def _generate_memory_context(self, query: str, limit: int, min_significance: float) -> str:
        """Generate context about past memories."""
        try:
            # Use higher significance threshold for memory queries
            min_significance = max(min_significance, 0.3)
            
            # Search for memories related to the query
            memories = await self.search_memory_tool(
                query=query,
                max_results=limit,
                min_significance=min_significance
            )
            
            if not memories or not memories.get("memories"):
                return ""
            
            parts = []
            results = memories["memories"]
            
            # Format each memory with timestamp and content
            for i, memory in enumerate(results):
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                significance = memory.get("significance", 0.0)
                
                # Format timestamp as readable date
                date_str = ""
                if timestamp:
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                
                # Add significance indicators for highly significant memories
                sig_indicator = ""
                if significance > 0.8:
                    sig_indicator = " [IMPORTANT]"
                elif significance > 0.6:
                    sig_indicator = " [Significant]"
                
                parts.append(f"• Memory from {date_str}{sig_indicator}: {content}")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating memory context: {e}")
            return ""
    
    async def _generate_emotional_context(self) -> str:
        """Generate context about emotional states."""
        try:
            # Get emotional context if available
            if hasattr(self, "get_emotional_history"):
                emotional_history = await self.get_emotional_history(limit=3)
                if emotional_history:
                    return emotional_history
            
            # Fallback to checking if we have emotions data
            if hasattr(self, "emotions") and self.emotions:
                parts = []
                
                # Sort by timestamp (newest first)
                sorted_emotions = sorted(
                    self.emotions.items(),
                    key=lambda x: float(x[0]),
                    reverse=True
                )[:3]  # Limit to 3 most recent
                
                for timestamp, data in sorted_emotions:
                    sentiment = data.get("sentiment", 0)
                    emotions = data.get("emotions", {})
                    
                    # Format timestamp
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                    
                    # Determine sentiment description
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
                    
                    # Format emotions if available
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
    
    async def _generate_standard_context(self, query: str, limit: int, min_significance: float) -> str:
        """Generate standard context from memories."""
        try:
            if query:
                # Search for memories related to the query
                memories = await self.search_memory_tool(
                    query=query,
                    max_results=limit,
                    min_significance=min_significance
                )
                
                if not memories or not memories.get("memories"):
                    return ""
                
                parts = []
                results = memories["memories"]
            else:
                # Get recent important memories
                memories = await self.get_important_memories(
                    limit=limit,
                    min_significance=min_significance or 0.5
                )
                
                if not memories or not memories.get("memories"):
                    return ""
                
                parts = []
                results = memories["memories"]
            
            # Format each memory with timestamp and content
            for memory in results:
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                
                # Format timestamp as readable date
                date_str = ""
                if timestamp:
                    try:
                        import datetime
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
            # Calculate appropriate significance based on content
            base_significance = 0.5 if role == "assistant" else 0.6
            
            # Adjust for content length
            length_factor = min(len(text) / 200, 0.2)  # Cap at 0.2
            
            # Check for question marks (questions are often important)
            question_factor = 0.1 if "?" in text else 0.0
            
            # Check for personal information indicators
            personal_indicators = ["name", "email", "phone", "address", "live", "age", "birthday", "family"]
            personal_factor = 0.2 if any(indicator in text.lower() for indicator in personal_indicators) else 0.0
            
            # Calculate final significance
            significance = min(base_significance + length_factor + question_factor + personal_factor, 0.95)
            
            # Store as a memory with conversation metadata
            return await self.store_memory(
                content=text,
                significance=significance,
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
                # Clean up topic string
                t = t.strip().lower()
                if not t:
                    continue
                
                # Store a memory indicating the topic was discussed
                result = await self.store_memory(
                    content=f"Topic '{t}' was discussed",
                    significance=importance,
                    metadata={
                        "type": "topic_discussed",
                        "topic": t,
                        "session_id": self.session_id,
                        "timestamp": time.time(),
                        "importance": importance
                    }
                )
                
                # Add to suppressed topics if enabled
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
            # Extract parameters
            topic = args.get("topic", "").strip()
            importance = args.get("importance", 0.7)
            
            if not topic:
                return {"success": False, "error": "No topic provided"}
            
            # Validate importance
            try:
                importance = float(importance)
                importance = max(0.0, min(1.0, importance))
            except (ValueError, TypeError):
                importance = 0.7  # Default if invalid
            
            # Mark topic as discussed
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
            
        # Clean topic
        topic = topic.strip().lower()
        
        # Check if topic is suppressed
        if topic in self._topic_suppression["suppressed_topics"]:
            data = self._topic_suppression["suppressed_topics"][topic]
            expiration = data.get("expiration", 0)
            importance = data.get("importance", 0.5)
            
            # Check if suppression has expired
            if expiration > time.time():
                return True, importance
            else:
                # Remove expired suppression
                del self._topic_suppression["suppressed_topics"][topic]
        
        return False, 0.0
    
    async def configure_topic_suppression(self, enable: bool = True, suppression_time: int = None) -> None:
        """
        Configure topic suppression settings.
        
        Args:
            enable: Whether to enable topic suppression
            suppression_time: Time in seconds to suppress repetitive topics
        """
        self._topic_suppression["enabled"] = enable
        
        if suppression_time is not None:
            try:
                suppression_time = int(suppression_time)
                if suppression_time > 0:
                    self._topic_suppression["suppression_time"] = suppression_time
            except (ValueError, TypeError):
                pass
        
        # Clean up expired suppressions
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
    
    async def reset_topic_suppression(self, topic: str = None) -> None:
        """
        Reset topic suppression for a specific topic or all topics.
        
        Args:
            topic: Specific topic to reset, or None to reset all topics
        """
        if topic:
            # Reset specific topic
            topic = topic.strip().lower()
            if topic in self._topic_suppression["suppressed_topics"]:
                del self._topic_suppression["suppressed_topics"][topic]
        else:
            # Reset all topics
            self._topic_suppression["suppressed_topics"] = {}
    
    async def get_topic_suppression_status(self) -> Dict[str, Any]:
        """
        Get the current status of topic suppression.
        
        Returns:
            Dictionary containing topic suppression status information
        """
        # Clean up expired suppressions
        current_time = time.time()
        expired_topics = [
            topic for topic, data in self._topic_suppression["suppressed_topics"].items()
            if data.get("expiration", 0) <= current_time
        ]
        
        for topic in expired_topics:
            del self._topic_suppression["suppressed_topics"][topic]
        
        # Build status dictionary
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
            # Call the emotion mixin's get_emotional_context method
            context = await self.get_emotional_context()
            
            # Enhanced the response with more helpful summary
            if context:
                # Add sentiment trend analysis if we have enough history
                if len(context.get("recent_emotions", [])) >= 2:
                    emotions = context.get("recent_emotions", [])
                    if emotions:
                        # Calculate sentiment trend
                        trend = "steady"
                        sentiment_values = [e.get("sentiment", 0) for e in emotions]
                        
                        if len(sentiment_values) >= 2:
                            if sentiment_values[0] > sentiment_values[-1] + 0.2:
                                trend = "improving"
                            elif sentiment_values[0] < sentiment_values[-1] - 0.2:
                                trend = "declining"
                        
                        context["sentiment_trend"] = trend
                
                # Add dominant emotion for the entire conversation
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
            # Extract category if provided
            category = None
            if args and isinstance(args, dict):
                category = args.get("category")
            
            # Initialize response
            response = {
                "found": False,
                "details": {}
            }
            
            # If we don't have personal details, return empty
            if not hasattr(self, "personal_details") or not self.personal_details:
                return response
            
            # Handle special case for name queries
            if category and category.lower() == "name":
                value = None
                confidence = 0
                
                # First check in personal details
                if "name" in self.personal_details:
                    detail = self.personal_details["name"]
                    if isinstance(detail, dict) and "value" in detail:
                        value = detail["value"]
                        confidence = detail.get("confidence", 0.9)
                    else:
                        value = detail
                        confidence = 0.9
                
                # If not found, search memories
                if not value:
                    try:
                        name_memories = await self.search_memory(
                            "user name", 
                            limit=3, 
                            min_significance=0.8
                        )
                        
                        # Extract name from memory content
                        for memory in name_memories:
                            content = memory.get("content", "")
                            
                            # Look for explicit name mentions
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
                                    
                                    # Store for future use
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
            
            # Handle specific category request
            if category:
                category = category.lower()
                
                # Family category needs special handling
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
                    # Standard category
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
            
            # No specific category requested or not found, return all details
            formatted_details = {}
            for cat, detail in self.personal_details.items():
                if cat == "family":
                    # Format family data
                    family_data = {}
                    if detail:
                        for relation, data in detail.items():
                            if isinstance(data, dict) and "name" in data:
                                family_data[relation] = data["name"]
                            else:
                                family_data[relation] = data
                        
                        formatted_details[cat] = family_data
                else:
                    # Format standard data
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
        """
        Get current timestamp in seconds since epoch.
        
        Returns:
            float: Current timestamp
        """
        return time.time()

    async def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze and record emotions from text content"""
async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
    """
    Detect and analyze emotional context from text.
    This is a wrapper around the EmotionMixin functionality for the voice agent.
    
    Args:
        text: The text to analyze for emotional context
        
    Returns:
        Dict with emotional context information
    """
    # Use the EmotionMixin's method if available
    if hasattr(self, "detect_emotion") and callable(getattr(self, "detect_emotion")):
        try:
            # First, create a simple emotional context structure
            timestamp = time.time()
            emotional_context = {
                "timestamp": timestamp,
                "text": text,
                "emotions": {},
                "sentiment": 0.0,
                "emotional_state": "neutral"
            }
            
            # Try to detect emotion using the mixin method
            emotion = await self.detect_emotion(text)
            emotional_context["emotional_state"] = emotion
            emotional_context["emotions"][emotion] = 1.0
            
            # Store the emotional context for future reference
            if hasattr(self, "_emotional_history"):
                self._emotional_history.append(emotional_context)
                
                # Trim emotional history if needed
                if hasattr(self, "_max_emotional_history") and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                    
                logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
            
            logger.info(f"Detected emotion: {emotion} for text: {text[:30]}...")
            return emotional_context
        except Exception as e:
            logger.error(f"Error in detect_emotion: {e}")
            # Fallback to empty emotional context
            return {
                "timestamp": time.time(),
                "text": text,
                "emotions": {"neutral": 1.0},
                "sentiment": 0.0,
                "emotional_state": "neutral",
                "error": str(e)
            }
    
    # Fallback if emotion detection isn't available
    return {
        "timestamp": time.time(),
        "text": text,
        "emotions": {"neutral": 1.0},
        "sentiment": 0.0,
        "emotional_state": "neutral"
    }

    async def store_memory(self, content: str, metadata: Dict[str, Any] = None, significance: float = None, importance: float = None) -> bool:
        """
        Store a new memory with semantic embedding and ensure proper persistence.
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            significance: Optional override for significance
            importance: Alternate name for significance (for backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Handle both significance and importance parameters for backward compatibility
            if significance is None and importance is not None:
                significance = importance
            
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
                await self._persist_single_memory(memory)
                    
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False

    async def _persist_single_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Persist a single memory to disk with robust error handling.
        
        Args:
            memory: The memory object to persist
            
        Returns:
            bool: Success status
        """
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
                
            # Ensure storage directory exists
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path, exist_ok=True)
                logger.info(f"Created storage directory: {self.storage_path}")
                
            file_path = self.storage_path / f"{memory_id}.json"
            temp_file_path = self.storage_path / f"{memory_id}.json.tmp"
            backup_file_path = self.storage_path / f"{memory_id}.json.bak"
            
            # Create a deep copy to avoid modifying the original
            memory_copy = copy.deepcopy(memory)
            
            # Convert embedding to list if it's a tensor or numpy array
            if 'embedding' in memory_copy:
                embedding_data = memory_copy['embedding']
                
                if hasattr(embedding_data, 'tolist'):
                    memory_copy['embedding'] = embedding_data.tolist()
                elif hasattr(embedding_data, 'detach'):
                    # PyTorch tensor with grad
                    memory_copy['embedding'] = embedding_data.detach().cpu().numpy().tolist()
                elif isinstance(embedding_data, np.ndarray):
                    memory_copy['embedding'] = embedding_data.tolist()
                    
            # Convert any other complex types
            memory_copy = self._convert_numpy_to_python(memory_copy)
            
            # Write to temporary file first
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_copy, f, ensure_ascii=False, indent=2)
                
            # If the file exists, create a backup before overwriting
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_file_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup for memory {memory_id}: {e}")
            
            # Atomic rename
            os.replace(temp_file_path, file_path)
            
            # Verify file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    _ = json.load(f)  # Just load to verify it's valid JSON
                logger.info(f"Successfully persisted memory {memory_id}")
                
                # Remove backup if everything succeeded
                if backup_file_path.exists():
                    os.remove(backup_file_path)
                    
                return True
            except json.JSONDecodeError:
                logger.error(f"Memory file {file_path} contains invalid JSON after writing")
                # Restore from backup if verification failed
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

    async def store_significant_memory(self, 
                                     text: str, 
                                     memory_type: str = "important", 
                                     metadata: Optional[Dict[str, Any]] = None,
                                     min_significance: float = 0.7) -> Dict[str, Any]:
        """
        Store a significant memory with guaranteed persistence.
        
        Args:
            text: The memory content
            memory_type: Type of memory to store
            metadata: Additional metadata
            min_significance: Minimum significance value
            
        Returns:
            Dict with memory information
        """
        if not text or not text.strip():
            logger.warning("Cannot store empty significant memory")
            return {"success": False, "error": "Empty memory content"}
            
        try:
            # Generate embedding and calculate significance
            embedding, auto_significance = await self.process_embedding(text)
            
            # Use the higher of calculated significance or minimum
            significance = max(auto_significance, min_significance)
            
            # Prepare metadata
            full_metadata = {
                "type": memory_type,
                "timestamp": time.time(),
                **(metadata or {})
            }
            
            # Store the memory
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": text,
                "embedding": embedding,
                "timestamp": time.time(),
                "significance": significance,
                "metadata": full_metadata
            }
            
            # Add to memory collection
            async with self._memory_lock:
                self.memories.append(memory)
                
            # Force immediate persistence
            success = await self._persist_single_memory(memory)
            
            if success:
                logger.info(f"Stored significant memory {memory_id} with significance {significance:.2f}")
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "significance": significance,
                    "type": memory_type
                }
            else:
                logger.error("Failed to persist significant memory")
                return {"success": False, "error": "Persistence failed"}
                
        except Exception as e:
            logger.error(f"Error storing significant memory: {e}")
            return {"success": False, "error": str(e)}

    async def store_important_memory(self, content: str = "", significance: float = 0.8) -> Dict[str, Any]:
        """
        Tool implementation to store an important memory with guaranteed persistence.
        
        Args:
            content: The memory content to store
            significance: Memory significance score (0.0-1.0)
            
        Returns:
            Dict with status
        """
        if not content:
            return {"success": False, "error": "No content provided"}
        
        try:
            # Use the store_significant_memory method
            result = await self.store_significant_memory(
                text=content,
                memory_type="important",
                min_significance=significance
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
        """Classify a user query into a specific query type to determine memory retrieval strategy.
        
        Types include:
        - recall: User is asking to recall a specific memory or conversation
        - information: User is asking for factual information
        - new_learning: User is providing new information to be stored
        - emotional: User is expressing emotions or seeking emotional support
        - clarification: User is asking for clarification on something previously said
        - task: User is requesting a specific task to be performed
        - greeting: User is greeting or making small talk
        - other: Default category for queries that don't fit elsewhere
        
        Args:
            query (str): The user's query text
            
        Returns:
            str: The classified query type
        """
        try:
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to classify_query")
                return "other"
                
            # Normalize query
            query = query.lower().strip()
            
            # Quick pattern matching for common cases
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
                
            # More sophisticated classification with LLM if available
            try:
                if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                    llm_classification = await self._classify_with_llm(query)
                    if llm_classification in ["recall", "information", "new_learning", "emotional", 
                                            "clarification", "task", "greeting", "other"]:
                        logger.info(f"LLM classified query as: {llm_classification}")
                        return llm_classification
            except Exception as e:
                logger.warning(f"Error during LLM classification, falling back to heuristics: {e}")
            
            # Default to information search if nothing else matched
            return "information"
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}", exc_info=True)
            return "other"
            
    async def _classify_with_llm(self, query: str) -> str:
        """Use LLM to classify the query type more accurately.
        
        Args:
            query (str): The user query to classify
            
        Returns:
            str: The classified query type
        """
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
            
            # Clean and validate response
            if response and isinstance(response, str):
                response = response.lower().strip()
                valid_types = ["recall", "information", "new_learning", "emotional", 
                              "clarification", "task", "greeting", "other"]
                if response in valid_types:
                    return response
                    
            return "information"  # Default fallback
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}", exc_info=True)
            return "information"  # Default fallback on error
    
    async def retrieve_memories(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the given query using semantic search.
        
        Args:
            query: The query text to search for related memories
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold for retrieved memories
            
        Returns:
            List of dictionaries containing memory entries
        """
        try:
            logger.info(f"Retrieving memories for query: '{query[:50]}...'")
            
            # Input validation
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_memories")
                return []
                
            if limit <= 0:
                logger.warning(f"Invalid limit parameter: {limit}, using default of 5")
                limit = 5
                
            # Check which memory system to use
            if hasattr(self, 'memory_integration') and self.memory_integration:
                # Use hierarchical memory system
                try:
                    memories = await self.memory_integration.retrieve_memories(
                        query=query,
                        limit=limit,
                        min_significance=min_significance
                    )
                    logger.info(f"Retrieved {len(memories)} memories from hierarchical memory system")
                    return memories
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical memory: {e}", exc_info=True)
                    # Fall back to standard retrieval
            
            # Standard memory retrieval using embedding similarity
            try:
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                    
                # Retrieve memories from database
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit,
                    min_significance=min_significance
                )
                
                # Format results
                result = []
                for memory in memories:
                    result.append({
                        "id": memory.id,
                        "content": memory.content,
                        "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                        "significance": memory.significance,
                        "memory_type": memory.memory_type.value if hasattr(memory, 'memory_type') else "transcript",
                        "metadata": memory.metadata
                    })
                    
                logger.info(f"Retrieved {len(result)} memories from standard memory system")
                return result
                    
            except Exception as e:
                logger.error(f"Error in standard memory retrieval: {e}", exc_info=True)
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return []
            
    async def _retrieve_memories_by_embedding(self, query_embedding: List[float], limit: int = 5, 
                                           min_significance: float = 0.3) -> List[Any]:
        """
        Retrieve memories by comparing the query embedding against stored memory embeddings.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of memories to return
            min_significance: Minimum significance threshold
            
        Returns:
            List of memory objects
        """
        try:
            if not hasattr(self, 'memory_db') or not self.memory_db:
                logger.warning("Memory database not initialized")
                return []
                
            # Retrieve memories from database with similarity search
            memories = await self.memory_db.get_by_vector(
                vector=query_embedding,
                limit=limit * 2,  # Get more than needed to filter by significance
                collection="memories"
            )
            
            # Filter by significance and sort by relevance
            filtered_memories = []
            for memory in memories:
                if hasattr(memory, 'significance') and memory.significance >= min_significance:
                    filtered_memories.append(memory)
                elif not hasattr(memory, 'significance'):
                    # If memory doesn't have significance attribute, include it anyway
                    filtered_memories.append(memory)
            
            # Sort by relevance and limit results
            sorted_memories = sorted(
                filtered_memories, 
                key=lambda x: x.similarity if hasattr(x, 'similarity') else 0.0,
                reverse=True
            )[:limit]
            
            return sorted_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories by embedding: {e}", exc_info=True)
            return []
    
    async def retrieve_information(self, query: str, limit: int = 5, min_significance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve factual information relevant to the query.
        Optimized for informational queries rather than personal memories.
        
        Args:
            query: The query text
            limit: Maximum number of information pieces to retrieve
            min_significance: Minimum significance threshold
            
        Returns:
            List of dictionaries containing information entries
        """
        try:
            logger.info(f"Retrieving information for query: '{query[:50]}...'")
            
            # Input validation
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_information")
                return []
                
            # Check which memory system to use
            if hasattr(self, 'memory_integration') and self.memory_integration:
                # Use hierarchical memory system's information retrieval
                try:
                    info_results = await self.memory_integration.retrieve_information(
                        query=query,
                        limit=limit,
                        min_significance=min_significance
                    )
                    logger.info(f"Retrieved {len(info_results)} information entries from hierarchical memory")
                    return info_results
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical information store: {e}", exc_info=True)
                    # Fall back to standard retrieval
            
            # Use RAG context if available
            if hasattr(self, 'get_rag_context'):
                try:
                    # Generate RAG context for informational queries
                    context_data = await self.get_rag_context(
                        query=query,
                        limit=limit,
                        min_significance=min_significance,
                        context_type="information"
                    )
                    
                    # Parse the context into structured information
                    if context_data and isinstance(context_data, str):
                        info_entries = self._parse_information_context(context_data)
                        if info_entries:
                            logger.info(f"Retrieved {len(info_entries)} information entries from RAG context")
                            return info_entries
                except Exception as e:
                    logger.error(f"Error retrieving RAG context for information: {e}", exc_info=True)
            
            # Fallback: Standard retrieval focused on information memory types
            try:
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                
                # Retrieve information memory types from database
                # This uses the same method but with a filter for information memory types
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit * 2,  # Get more to filter
                    min_significance=min_significance
                )
                
                # Filter for information memory types and format results
                result = []
                for memory in memories:
                    # Check if memory has the appropriate type for information
                    is_info_type = False
                    if hasattr(memory, 'memory_type'):
                        info_types = ["fact", "concept", "definition", "information", "knowledge"]
                        memory_type_value = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                        is_info_type = any(info_type in memory_type_value.lower() for info_type in info_types)
                    
                    # For standard memories, check metadata or content for clues
                    elif hasattr(memory, 'metadata') and memory.metadata:
                        if memory.metadata.get('type') in ['fact', 'information', 'knowledge']:
                            is_info_type = True
                    
                    # Include if it's an information type or if we can't determine (better to include than exclude)
                    if is_info_type or not hasattr(memory, 'memory_type'):
                        result.append({
                            "id": memory.id if hasattr(memory, 'id') else str(uuid.uuid4()),
                            "content": memory.content,
                            "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                            "significance": getattr(memory, 'significance', 0.5),
                            "memory_type": memory.memory_type.value if hasattr(memory, 'memory_type') and hasattr(memory.memory_type, 'value') else "information",
                            "metadata": getattr(memory, 'metadata', {})
                        })
                
                logger.info(f"Retrieved {len(result)} information entries from standard memory system")
                return result[:limit]  # Limit to requested number
                
            except Exception as e:
                logger.error(f"Error in standard information retrieval: {e}", exc_info=True)
                return []
        
        except Exception as e:
            logger.error(f"Error retrieving information: {e}", exc_info=True)
            return []
            
    def _parse_information_context(self, context_data: str) -> List[Dict[str, Any]]:
        """
        Parse RAG context data into structured information entries.
        
        Args:
            context_data: The RAG context string
            
        Returns:
            List of dictionaries containing parsed information
        """
        try:
            info_entries = []
            
            # Split context into sections (each fact or piece of information)
            sections = re.split(r'\n(?=Fact|Information|Knowledge|Definition|Concept)\s*\d*:', context_data)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                # Extract content
                content = section.strip()
                
                # Generate a unique ID
                entry_id = str(uuid.uuid4())
                
                # Determine type and confidence
                entry_type = "information"
                confidence = 0.7  # Default confidence
                
                # Look for type indicators in the content
                type_match = re.search(r'^(Fact|Information|Knowledge|Definition|Concept)', content)
                if type_match:
                    entry_type = type_match.group(1).lower()
                
                # Look for confidence indicators
                confidence_match = re.search(r'confidence:\s*(\d+(\.\d+)?)', content, re.IGNORECASE)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        # Normalize to 0-1 range if it's expressed as percentage
                        if confidence > 1:
                            confidence /= 100
                    except ValueError:
                        pass
                
                info_entries.append({
                    "id": entry_id,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "significance": confidence,
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
        """
        Store new information and retrieve contextually related memories.
        This is useful for learning new information while providing context.
        
        Args:
            text: The text to store
            metadata: Optional metadata to attach to the memory
            
        Returns:
            List of related memory objects
        """
        try:
            logger.info(f"Storing and retrieving context for: '{text[:50]}...'")
            
            # Input validation
            if not text or not isinstance(text, str):
                logger.warning("Empty or invalid text provided to store_and_retrieve")
                return []
            
            # Default metadata if none provided
            if metadata is None:
                metadata = {
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "type": "new_learning"
                }
            
            # Store the new information
            store_success = False
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    # Use hierarchical memory system
                    memory_id = await self.memory_integration.store(
                        content=text,
                        metadata=metadata
                    )
                    logger.info(f"Stored new information with ID {memory_id} in hierarchical memory")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in hierarchical memory: {e}", exc_info=True)
                    # Fall back to base memory storage
            
            # Fallback: Use base memory storage if hierarchical failed or not available
            if not store_success:
                try:
                    memory_id = await self.store_memory(
                        text=text,
                        memory_type="user_input",
                        metadata=metadata
                    )
                    logger.info(f"Stored new information with ID {memory_id} in base memory system")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in base memory: {e}", exc_info=True)
            
            # Return immediately if storage failed
            if not store_success:
                logger.warning("Failed to store new information")
                return []
            
            # Retrieve related memories
            try:
                # First try to classify the text to determine retrieval strategy
                query_type = await self.classify_query(text)
                
                # Use appropriate retrieval method based on classification
                if query_type == "information":
                    related_memories = await self.retrieve_information(text, limit=3, min_significance=0.2)
                else:
                    related_memories = await self.retrieve_memories(text, limit=3, min_significance=0.2)
                
                logger.info(f"Retrieved {len(related_memories)} related memories")
                return related_memories
                
            except Exception as e:
                logger.error(f"Error retrieving related memories: {e}", exc_info=True)
                return []
            
        except Exception as e:
            logger.error(f"Error in store_and_retrieve: {e}", exc_info=True)
            return []
    
    async def store_emotional_context(self, emotional_context: Dict[str, Any]) -> bool:
        """
        Store emotional context information for tracking user emotional states over time.
        
        Args:
            emotional_context: Dictionary containing emotional context data
                - emotion: Primary emotion detected
                - intensity: Intensity score (0.0-1.0)
                - secondary_emotions: List of secondary emotions
                - text: Original text that generated this emotional context
                
        Returns:
            bool: Success status
        """
        try:
            # Input validation
            if not emotional_context or not isinstance(emotional_context, dict) or 'emotion' not in emotional_context:
                logger.warning("Invalid emotional context data provided")
                return False
            
            # Calculate significance based on emotional intensity
            significance = emotional_context.get('intensity', 0.5)
            
            # Prepare metadata
            metadata = {
                "type": "emotional_context",
                "primary_emotion": emotional_context.get('emotion'),
                "intensity": emotional_context.get('intensity', 0.5),
                "secondary_emotions": emotional_context.get('secondary_emotions', []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in emotional history for EmotionMixin
            if hasattr(self, '_emotional_history'):
                self._emotional_history.append(metadata)
                
                # Trim emotional history if needed
                if hasattr(self, '_max_emotional_history') and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                    
                logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
            
            # Also store as a memory if significant enough
            if significance >= 0.4:  # Lower threshold to capture more emotional context
                text = emotional_context.get('text', f"User expressed {metadata['primary_emotion']} with intensity {metadata['intensity']}")
                
                store_success = False
                if hasattr(self, 'memory_integration') and self.memory_integration:
                    try:
                        # Use hierarchical memory system
                        memory_id = await self.memory_integration.store(
                            content=text,
                            metadata=metadata,
                            importance=significance
                        )
                        logger.info(f"Stored emotional context with ID {memory_id} in hierarchical memory")
                        store_success = True
                    except Exception as e:
                        logger.error(f"Error storing emotional context in hierarchical memory: {e}", exc_info=True)
                        # Fall back to base memory storage
                
                # Fallback: Use base memory storage if hierarchical failed or not available
                if not store_success:
                    try:
                        memory_id = await self.store_memory(
                            text=text,
                            memory_type="emotional",
                            significance=significance,
                            metadata=metadata
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
        """
        Compare two texts for semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            # Generate embeddings for both texts
            embedding1, _ = await self.process_embedding(text1)
            embedding2, _ = await self.process_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Failed to generate embeddings for similarity comparison")
                return 0.0
            
            # Calculate cosine similarity
            if isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
                # Normalize embeddings
                embedding1 = embedding1 / embedding1.norm()
                embedding2 = embedding2 / embedding2.norm()
                # Calculate dot product of normalized vectors (cosine similarity)
                similarity = torch.dot(embedding1, embedding2).item()
            else:
                # Convert to numpy arrays if needed
                if not isinstance(embedding1, np.ndarray):
                    embedding1 = np.array(embedding1)
                if not isinstance(embedding2, np.ndarray):
                    embedding2 = np.array(embedding2)
                
                # Normalize embeddings
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
                
                # Calculate dot product (cosine similarity)
                similarity = np.dot(embedding1, embedding2)
            
            # Ensure similarity is between 0 and 1
            similarity = float(max(0.0, min(1.0, similarity)))
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return 0.0