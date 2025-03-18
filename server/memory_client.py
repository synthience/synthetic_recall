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

class EnhancedMemoryClient(
    BaseMemoryClient,
    ToolsMixin,
    EmotionMixin,
    ConnectivityMixin,
    PersonalDetailsMixin,
    RAGContextMixin
):
    """
    Enhanced memory client that combines all mixins to provide a complete memory system.
    
    This class integrates all functionality from the various mixins:
    - BaseMemoryClient: Core memory functionality and initialization
    - ConnectivityMixin: WebSocket connection handling for tensor and HPC servers
    - EmotionMixin: Emotion detection and tracking
    - ToolsMixin: Memory search and embedding tools
    - PersonalDetailsMixin: Personal information extraction and storage
    - RAGContextMixin: Advanced context generation for RAG
    """
    
    def __init__(
        self,
        tensor_server_url: str, 
        hpc_server_url: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ping_interval: float = 20.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        connection_timeout: float = 10.0,
        **kwargs
    ):
        """
        Initialize the enhanced memory client.
        
        Args:
            tensor_server_url: URL for tensor server WebSocket connection
            hpc_server_url: URL for HPC server WebSocket connection
            session_id: Unique session identifier
            user_id: User identifier
            ping_interval: Interval in seconds to send ping messages
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
        self._connected = False
        
        # Now initialize mixins
        PersonalDetailsMixin.__init__(self)
        EmotionMixin.__init__(self)
        ToolsMixin.__init__(self)
        RAGContextMixin.__init__(self)
        
        # Topic suppression settings
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
    
    async def get_memory_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get all memory tools formatted for the LLM.
        
        Returns:
            A list of tools in dictionary form.
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
                            "description": (
                                "Optional category of personal detail to retrieve "
                                "(e.g., 'name', 'location', 'birthday', 'job', 'family')"
                            )
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
        
        max_retries = 2
        retry_count = 0
        backoff_factor = 1.5
        retry_delay = 1.0
        
        while retry_count <= max_retries:
            try:
                start_time = time.time()
                
                # Map tool names to their handlers
                tool_handlers = {
                    "search_memory": self.search_memory_tool,
                    "get_personal_details": self.get_personal_details_tool,
                    "get_emotional_context": self.get_emotional_context_tool,
                    "track_conversation_topic": self.track_conversation_topic
                }
                
                handler = tool_handlers.get(tool_name)
                if not handler:
                    logger.warning(f"Unknown tool call: {tool_name}")
                    return {"error": f"Unknown tool: {tool_name}", "success": False}
                
                # Special handling for search_memory
                if tool_name == "search_memory":
                    query = validated_args.get("query", "")
                    limit = validated_args.get("limit", 5)
                    return await handler(query=query, max_results=limit)
                
                # For other tools, pass args directly
                result = await handler(validated_args)
                
                elapsed_time = time.time() - start_time
                logger.debug(f"Tool call {tool_name} completed in {elapsed_time:.3f}s")
                
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
                
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(
                    f"Tool call {tool_name} timed out, retrying in {wait_time:.2f}s "
                    f"(attempt {retry_count}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}", exc_info=True)
                retry_count += 1
                if retry_count > max_retries or not self._is_retryable_error(e):
                    return {"error": f"Tool execution error: {str(e)}", "success": False}
                
                wait_time = retry_delay * (backoff_factor ** retry_count)
                logger.warning(
                    f"Retrying tool call {tool_name} in {wait_time:.2f}s "
                    f"(attempt {retry_count}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
    
    def _validate_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool arguments."""
        if not isinstance(args, dict):
            return {"error": "Arguments must be a dictionary"}
        
        if tool_name == "search_memory":
            if "query" not in args or not args["query"]:
                return {"error": "Query is required for search_memory tool"}
            
            if "limit" in args:
                try:
                    args["limit"] = max(1, min(int(args["limit"]), 20))
                except (ValueError, TypeError):
                    args["limit"] = 5
        
        elif tool_name == "track_conversation_topic":
            if "topic" not in args or not args["topic"]:
                return {"error": "Topic is required for track_conversation_topic tool"}
            if "importance" in args:
                try:
                    args["importance"] = max(0.0, min(float(args["importance"]), 1.0))
                except (ValueError, TypeError):
                    args["importance"] = 0.7
            else:
                args["importance"] = 0.7
        
        return args
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable (e.g., network-related).
        """
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            json.JSONDecodeError
        )
        error_str = str(error).lower()
        network_keywords = ["connection", "timeout", "network", "socket", "unavailable"]
        
        return isinstance(error, retryable_errors) or any(keyword in error_str for keyword in network_keywords)

    async def store_transcript(
        self, 
        text: str, 
        sender: str = "user", 
        role: str = None
    ) -> bool:
        """
        Store a transcript entry in memory without any significance weighting.
        
        Args:
            text: The transcript text to store
            sender: Who sent the message (user or assistant)
            role: (Optional) alternative name for sender (backward compatibility)
            
        Returns:
            bool: Success status
        """
        if not text or not text.strip():
            logger.warning("Empty transcript text provided")
            return False
            
        try:
            # FIXED: Previously only updated sender if sender="user", now properly handles all cases
            if role is not None:
                sender = role
            
            # Store in memory with no significance parameter
            metadata = {
                "type": "transcript",
                "sender": sender,
                "timestamp": time.time(),
                "session_id": self.session_id
            }
            
            success = await self.store_memory(
                content=text,
                metadata=metadata
            )
            if success:
                logger.info(f"Stored transcript from {sender}")
            else:
                logger.warning(f"Failed to store transcript from {sender}")
            return success
        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return False

    async def detect_and_store_personal_details(self, text: str, role: str = "user") -> bool:
        """
        Detect and store personal details from text (e.g., name, location).
        
        Args:
            text: The text to analyze
            role: The role of the speaker (user or assistant)
            
        Returns:
            bool: True if any details were detected and stored
        """
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
                        
                        if len(value) < 2 or value.lower() in [
                            "a","an","the","me","i","my","he","she","they"
                        ]:
                            continue
                        
                        if category == "email" and not re.match(r"[\w.+-]+@[\w-]+\.[\w.-]+", value):
                            continue
                        if category == "age":
                            if not value.isdigit() or int(value) > 120 or int(value) < 1:
                                continue
                        
                        if hasattr(self, "personal_details"):
                            confidence = 0.9
                            self.personal_details[category] = {
                                "value": value,
                                "confidence": confidence,
                                "timestamp": time.time(),
                                "source": "explicit_mention"
                            }
                            
                            logger.info(
                                f"Stored personal detail: {category}={value} "
                                f"(confidence: {confidence:.2f})"
                            )
                            details_found = True
                        
                        # Also store as memory
                        await self.store_memory(
                            content=f"User {category}: {value}",
                            metadata={
                                "type": "personal_detail",
                                "category": category,
                                "value": value,
                                "confidence": 0.9,
                                "timestamp": time.time()
                            }
                        )
                        
                        break
            
            # Process family patterns
            for relation, pattern_list in family_patterns.items():
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if len(match) >= 2:
                                if match[1] in [
                                    "wife", "husband", "spouse", "partner",
                                    "son", "daughter", "child", "mother",
                                    "father", "mom", "dad", "parent",
                                    "brother", "sister", "sibling"
                                ]:
                                    name = match[0].strip().rstrip('.,:;!?')
                                    relation_type = match[1].lower()
                                else:
                                    relation_type = match[0].lower()
                                    name = match[1].strip().rstrip('.,:;!?')
                                
                                if len(name) < 2 or name.lower() in [
                                    "a", "an", "the", "me", "i", "my"
                                ]:
                                    continue
                                
                                if hasattr(self, "personal_details") and "family" in self.personal_details:
                                    confidence = 0.9
                                    self.personal_details["family"][relation_type] = {
                                        "name": name,
                                        "confidence": confidence,
                                        "timestamp": time.time(),
                                        "source": "explicit_mention"
                                    }
                                    
                                    logger.info(
                                        f"Stored family detail: {relation_type}={name} "
                                        f"(confidence: {confidence:.2f})"
                                    )
                                    details_found = True
                                
                                # Also store as memory
                                await self.store_memory(
                                    content=f"User's {relation_type}: {name}",
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

    async def get_rag_context(
        self, 
        query: str = None, 
        limit: int = 5, 
        max_tokens: int = None,
        min_quickrecal_score: float = 0.0,
        min_significance: float = None
    ) -> str:
        """
        Get memory context for LLM RAG (Retrieval-Augmented Generation),
        minus any significance-based filtering.
        
        Args:
            query: Optional query to filter memories
            limit: Maximum number of memories to include
            max_tokens: Maximum number of tokens to include (approximate)
            min_quickrecal_score: Minimum quickrecal score threshold for memories
            min_significance: Legacy parameter for backward compatibility (deprecated)
            
        Returns:
            str: Formatted memory context for RAG
        """
        try:
            max_tokens = max_tokens or limit * 100  
            
            # Handle legacy parameter for backward compatibility
            if min_significance is not None and min_quickrecal_score == 0.0:
                min_quickrecal_score = min_significance
            is_personal_query = await self._is_personal_query(query) if query else False
            is_memory_query = await self._is_memory_query(query) if query else False
            
            context_parts = []
            context_sections = {}
            
            if is_personal_query:
                personal_context = await self._generate_personal_context(query)
                if personal_context:
                    context_sections["personal"] = personal_context
                    context_parts.append("### User Personal Information")
                    context_parts.append(personal_context)
            
            if is_memory_query:
                memory_limit = limit * 2
                memory_context = await self._generate_memory_context(query, memory_limit)
                if memory_context:
                    context_sections["memory"] = memory_context
                    context_parts.append("### Memory Recall")
                    context_parts.append(memory_context)
            
            if is_personal_query or (
                query and any(
                    keyword in query.lower()
                    for keyword in ["feel","emotion","mood","happy","sad","angry"]
                )
            ):
                emotional_context = await self._generate_emotional_context()
                if emotional_context:
                    context_sections["emotional"] = emotional_context
                    context_parts.append("### Recent Emotional States")
                    context_parts.append(emotional_context)
            
            if not context_parts or (not is_memory_query and not is_personal_query):
                standard_context = await self._generate_standard_context(query, limit)
                if standard_context:
                    context_sections["standard"] = standard_context
                    if not context_parts:
                        context_parts.append("### Relevant Memory Context")
                    context_parts.append(standard_context)
            
            context = "\n\n".join(context_parts)
            
            char_limit = max_tokens * 4  # approximate
            if len(context) > char_limit:
                truncated_context = []
                current_length = 0
                for part in context_parts:
                    if current_length + len(part) <= char_limit:
                        truncated_context.append(part)
                        current_length += len(part) + 2
                    else:
                        if not truncated_context:
                            truncation_point = char_limit - current_length
                            truncated_part = (
                                part[:truncation_point]
                                + "...\n[Context truncated due to length]"
                            )
                            truncated_context.append(truncated_part)
                        else:
                            truncated_context.append("[Additional context truncated due to length]")
                        break
                context = "\n\n".join(truncated_context)
            
            # Rough count of bullet points from memory
            bullet_count = sum(
                1
                for v in context_sections.values()
                for _ in v.split('•')
                if '•' in v
            )
            logger.info(f"Generated RAG context with {bullet_count} memories")
            return context
        except Exception as e:
            logger.error(f"Error generating RAG context: {e}")
            return ""
    
    async def _is_personal_query(self, query: str) -> bool:
        personal_keywords = [
            "my name","who am i","where do i live","where am i from","how old am i",
            "what's my age","what's my birthday","when was i born","what do i do",
            "what's my job","what's my profession","who is my","my family",
            "my spouse","my partner","my children","my parents","my email",
            "my phone","my number","my address"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in personal_keywords)
    
    async def _is_memory_query(self, query: str) -> bool:
        memory_keywords = [
            "remember","recall","forget","memory","memories","mentioned","said earlier",
            "talked about","told me about","what did i say","what did you say",
            "what did we discuss","earlier","before","previously","last time",
            "yesterday","last week","last month","last year","in the past"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in memory_keywords)
    
    async def _generate_personal_context(self, query: str) -> str:
        if not hasattr(self, "personal_details") or not self.personal_details:
            return ""
        
        parts = []
        parts.append("# USER PERSONAL INFORMATION")
        parts.append("# The following information is about the USER, not the assistant")
        parts.append("# The assistant's name is Lucidia")
        
        personal_categories = {
            "name": ["name","call me","who am i"],
            "location": ["live","from","address","location","home"],
            "birthday": ["birthday","born","birth date","age"],
            "job": ["job","work","profession","career","occupation"],
            "family": ["family","spouse","partner","husband","wife","child","children","kid",
                       "kids","parent","mother","father"],
            "email": ["email","e-mail","mail"],
            "phone": ["phone","number","mobile","cell"]
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
    
    async def _generate_memory_context(self, query: str, limit: int) -> str:
        """
        Generate context about past memories (no significance filtering).
        """
        try:
            memories = await self.search_memory_tool(query=query, max_results=limit)
            if not memories or not memories.get("memories"):
                return ""
            
            parts = []
            results = memories["memories"]
            
            for i, memory in enumerate(results):
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
                # For readability
                date_str = ""
                if timestamp:
                    try:
                        import datetime
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = f"timestamp: {timestamp}"
                
                parts.append(f"• Memory from {date_str}: {content}")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Error generating memory context: {e}")
            return ""
    
    async def _generate_emotional_context(self) -> str:
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
                        import datetime
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
    
    async def _generate_standard_context(self, query: str, limit: int) -> str:
        """
        Generate standard context from memories (no significance filtering).
        """
        try:
            if query:
                memories = await self.search_memory_tool(query=query, max_results=limit)
                if not memories or not memories.get("memories"):
                    return ""
                parts = []
                results = memories["memories"]
            else:
                # If no query, just retrieve some recent or relevant memories
                memories = await self.search_memory_tool(query="", max_results=limit)
                if not memories or not memories.get("memories"):
                    return ""
                parts = []
                results = memories["memories"]
            
            for memory in results:
                content = memory.get("content", "").strip()
                timestamp = memory.get("timestamp", 0)
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
        Store a conversation message in memory (no significance).
        
        Args:
            text: The message text
            role: The role of the sender
        """
        try:
            metadata = {
                "type": "conversation",
                "role": role,
                "session_id": self.session_id,
                "timestamp": time.time()
            }
            return await self.store_memory(content=text, metadata=metadata)
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
                
                # Store a memory indicating topic was discussed
                result = await self.store_memory(
                    content=f"Topic '{t}' was discussed",
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
                    "suppression_time": self._topic_suppression["suppression_time"]
                    if self._topic_suppression["enabled"] else 0
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
            topic: Specific topic to reset, or None to reset all
        """
        if topic:
            topic = topic.strip().lower()
            if topic in self._topic_suppression["suppressed_topics"]:
                del self._topic_suppression["suppressed_topics"][topic]
        else:
            self._topic_suppression["suppressed_topics"] = {}
    
    async def get_topic_suppression_status(self) -> Dict[str, Any]:
        """
        Get the current status of topic suppression.
        
        Returns:
            Dictionary with topic suppression status
        """
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
                        name_memories = await self.search_memory("user name", limit=3)
                        for memory in name_memories:
                            content = memory.get("content", "")
                            patterns = [
                                r"User name: ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"User's name is ([A-Za-z]+(?: [A-Za-z]+){0,3})",
                                r"([A-Za-z]+(?: [A-Za-z]+){0,3}) is my name"
                            ]
                            for pat in patterns:
                                matches = re.findall(pat, content, re.IGNORECASE)
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
        """Analyze and record emotions from text content."""
        # Implementation or direct call to emotion detection
        return {}

async def detect_emotional_context(self, text: str) -> Dict[str, Any]:
    """
    Detect and analyze emotional context from text.
    This is a wrapper around the EmotionMixin functionality for the voice agent.
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
                    
            logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
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

    async def store_memory(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store a new memory with semantic embedding (no significance).
        
        Args:
            content: The memory content to store
            metadata: Additional metadata for the memory
            
        Returns:
            bool: Success status
        """
        if not content or not content.strip():
            logger.warning("Attempted to store empty memory content")
            return False
            
        try:
            # Generate embedding
            try:
                embedding, _ = await self.process_embedding(content)
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                if hasattr(self, 'embedding_dim'):
                    embedding = torch.zeros(self.embedding_dim)
                else:
                    embedding = torch.zeros(384)
            
            memory_id = str(uuid.uuid4())
            memory = {
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            async with self._memory_lock:
                self.memories.append(memory)
            
            logger.info(f"Stored new memory with ID {memory_id}")
            
            # You may optionally persist memory immediately
            # if your application design requires it
            # (omitted here since significance is removed)
            
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False

    async def _persist_single_memory(self, memory: Dict[str, Any]) -> bool:
        """
        (Optional) Persist a single memory to disk. 
        Significance references are removed, but 
        you may still want to store the memory’s content, embedding, etc.
        """
        if not hasattr(self, 'persistence_enabled') or not self.persistence_enabled:
            return False
        if not hasattr(self, 'storage_path'):
            logger.error("No storage path configured")
            return False
        
        try:
            import os, shutil, copy, json
            from pathlib import Path
            import numpy as np
            
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
                    memory_copy['embedding'] = (
                        embedding_data.detach().cpu().numpy().tolist()
                    )
                elif isinstance(embedding_data, np.ndarray):
                    memory_copy['embedding'] = embedding_data.tolist()
            
            def convert_numpy_to_python(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_python(x) for x in obj]
                elif isinstance(obj, np.number):
                    return float(obj)
                return obj
            
            memory_copy = convert_numpy_to_python(memory_copy)
            
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
                logger.error(f"Memory file {file_path} has invalid JSON after writing")
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

    async def store_and_retrieve(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Store new information and retrieve contextually related memories, 
        without using significance logic.
        
        Args:
            text: The text to store
            metadata: Optional metadata to attach
        
        Returns:
            List of related memory objects
        """
        try:
            logger.info(f"Storing and retrieving context for: '{text[:50]}...'")
            if not text or not isinstance(text, str):
                logger.warning("Empty or invalid text provided to store_and_retrieve")
                return []
            
            if metadata is None:
                from datetime import datetime
                metadata = {
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "type": "new_learning"
                }
            
            store_success = False
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    memory_id = await self.memory_integration.store(content=text, metadata=metadata)
                    logger.info(f"Stored new information with ID {memory_id} in hierarchical memory")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in hierarchical memory: {e}", exc_info=True)
            
            if not store_success:
                try:
                    memory_id = await self.store_memory(content=text, metadata=metadata)
                    logger.info(f"Stored new information in base memory system")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing in base memory: {e}", exc_info=True)
            
            if not store_success:
                logger.warning("Failed to store new information")
                return []
            
            try:
                query_type = await self.classify_query(text)
                if query_type == "information":
                    related_memories = await self.retrieve_information(text, limit=3)
                else:
                    related_memories = await self.retrieve_memories(text, limit=3)
                
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
        Store emotional context information (no significance).
        
        Args:
            emotional_context: Dictionary containing emotional context data
        
        Returns:
            bool: Success status
        """
        try:
            if not emotional_context or not isinstance(emotional_context, dict) or 'emotion' not in emotional_context:
                logger.warning("Invalid emotional context data provided")
                return False
            
            metadata = {
                "type": "emotional_context",
                "primary_emotion": emotional_context.get('emotion'),
                "intensity": emotional_context.get('intensity', 0.5),
                "secondary_emotions": emotional_context.get('secondary_emotions', []),
                "timestamp": time.time()
            }
            
            if hasattr(self, '_emotional_history'):
                self._emotional_history.append(metadata)
                if hasattr(self, '_max_emotional_history') and len(self._emotional_history) > self._max_emotional_history:
                    self._emotional_history = self._emotional_history[-self._max_emotional_history:]
                logger.debug(f"Added to emotional history (total: {len(self._emotional_history)})")
            
            text = emotional_context.get(
                'text',
                f"User expressed {metadata['primary_emotion']} with intensity {metadata['intensity']}"
            )
            
            store_success = False
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    memory_id = await self.memory_integration.store(content=text, metadata=metadata)
                    logger.info(f"Stored emotional context with ID {memory_id} in hierarchical memory")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing emotional context in hierarchical memory: {e}", exc_info=True)
            
            if not store_success:
                try:
                    _ = await self.store_memory(content=text, metadata=metadata)
                    logger.info("Stored emotional context in base memory system")
                    store_success = True
                except Exception as e:
                    logger.error(f"Error storing emotional context in base memory: {e}", exc_info=True)
            
            logger.info(
                f"Stored emotional context: {metadata['primary_emotion']} "
                f"(intensity: {metadata['intensity']})"
            )
            return True
        except Exception as e:
            logger.error(f"Error storing emotional context: {e}", exc_info=True)
            return False
    
    async def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts for semantic similarity using embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            embedding1, _ = await self.process_embedding(text1)
            embedding2, _ = await self.process_embedding(text2)
            
            if embedding1 is None or embedding2 is None:
                logger.warning("Failed to generate embeddings for similarity comparison")
                return 0.0
            
            import torch
            import numpy as np
            
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
    
    async def classify_query(self, query: str) -> str:
        """
        Classify a user query into a specific type for retrieval strategy (no significance usage).
        """
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
                    valid_types = [
                        "recall","information","new_learning","emotional",
                        "clarification","task","greeting","other"
                    ]
                    if llm_classification in valid_types:
                        logger.info(f"LLM classified query as: {llm_classification}")
                        return llm_classification
            except Exception as e:
                logger.warning(f"Error during LLM classification, fallback: {e}")
            
            return "information"
        except Exception as e:
            logger.error(f"Error classifying query: {e}", exc_info=True)
            return "other"
    
    async def _classify_with_llm(self, query: str) -> str:
        """
        Use LLM to classify the query type more accurately (no significance logic).
        """
        try:
            prompt = (
                "Classify the following user query into exactly one of these categories: \n"
                "recall, information, new_learning, emotional, clarification, task, greeting, other.\n"
                "Respond with only the category name, nothing else.\n\n"
                f"Query: \"{query}\"\n\nClassification:"
            )
            response = await self.llm_pipeline.agenerate_text(
                prompt=prompt,
                max_tokens=10,
                temperature=0.2
            )
            if response and isinstance(response, str):
                response = response.lower().strip()
                valid_types = [
                    "recall","information","new_learning","emotional",
                    "clarification","task","greeting","other"
                ]
                if response in valid_types:
                    return response
            return "information"
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}", exc_info=True)
            return "information"
    
    async def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories related to the query.
        
        Args:
            query: The query text
            limit: Max number of memories to return
            
        Returns:
            List of memory entries
        """
        try:
            # Safety check for query
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_memories")
                return []
                
            logger.info(f"Retrieving memories for query: '{query[:50]}...'" if len(query) > 50 else f"Retrieving memories for query: '{query}'")
            
            # Try hierarchical memory integration first
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    memories = await self.memory_integration.retrieve_memories(
                        query=query,
                        limit=limit
                    )
                    logger.info(f"Retrieved {len(memories)} memories from hierarchical memory system")
                    return memories
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical memory: {e}", exc_info=True)
            
            # Fall back to standard memory retrieval
            try:
                query_embedding = await self._generate_embedding(query)
                if not query_embedding:
                    logger.warning("Failed to generate embedding for query")
                    return []
                    
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit
                )
                
                result = []
                for memory in memories:
                    result.append({
                        "id": memory.id if hasattr(memory, 'id') else str(uuid.uuid4()),
                        "content": memory.content,
                        "timestamp": memory.timestamp.isoformat()
                        if hasattr(memory, 'timestamp') else None,
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
    
    async def _retrieve_memories_by_embedding(self, query_embedding: List[float], limit: int = 5) -> List[Any]:
        """
        Retrieve memories by comparing the query embedding against stored memory embeddings (no significance).
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Max number of memories to return
        
        Returns:
            List of memory objects
        """
        try:
            if not hasattr(self, 'memory_db') or not self.memory_db:
                logger.warning("Memory database not initialized")
                return []
            
            memories = await self.memory_db.get_by_vector(
                vector=query_embedding,
                limit=limit * 2,
                collection="memories"
            )
            
            # Sort purely by relevance
            sorted_memories = sorted(
                memories,
                key=lambda x: x.similarity if hasattr(x, 'similarity') else 0.0,
                reverse=True
            )[:limit]
            return sorted_memories
        except Exception as e:
            logger.error(f"Error retrieving memories by embedding: {e}", exc_info=True)
            return []
    
    async def retrieve_information(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve factual information relevant to the query (no significance).
        
        Args:
            query: The query text
            limit: Max number of information pieces to return
            
        Returns:
            List of dictionaries containing information entries
        """
        try:
            logger.info(f"Retrieving information for query: '{query[:50]}...'" if query else "Empty query")
            if not query or not isinstance(query, str):
                logger.warning("Empty or invalid query provided to retrieve_information")
                return []
            
            if hasattr(self, 'memory_integration') and self.memory_integration:
                try:
                    info_results = await self.memory_integration.retrieve_information(
                        query=query,
                        limit=limit
                    )
                    logger.info(f"Retrieved {len(info_results)} information entries from hierarchical memory")
                    return info_results
                except Exception as e:
                    logger.error(f"Error retrieving from hierarchical information store: {e}", exc_info=True)
            
            if hasattr(self, 'get_rag_context'):
                try:
                    context_data = await self.get_rag_context(
                        query=query,
                        limit=limit
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
                    logger.warning("Failed to generate embedding for information query")
                    return []
                memories = await self._retrieve_memories_by_embedding(
                    query_embedding=query_embedding,
                    limit=limit * 2
                )
                result = []
                for memory in memories:
                    # Check if memory might be "information"-type
                    is_info_type = False
                    if hasattr(memory, 'memory_type'):
                        info_types = ["fact","concept","definition","information","knowledge"]
                        memory_type_value = memory.memory_type.value if (
                            hasattr(memory.memory_type, 'value')
                        ) else str(memory.memory_type)
                        is_info_type = any(info_type in memory_type_value.lower() for info_type in info_types)
                    elif hasattr(memory, 'metadata') and memory.metadata:
                        if memory.metadata.get('type') in ['fact','information','knowledge']:
                            is_info_type = True
                    
                    if is_info_type or not hasattr(memory, 'memory_type'):
                        result.append({
                            "id": memory.id if hasattr(memory, 'id') else str(uuid.uuid4()),
                            "content": memory.content,
                            "timestamp": memory.timestamp.isoformat()
                            if hasattr(memory, 'timestamp') else None,
                            "metadata": getattr(memory, 'metadata', {})
                        })
                
                logger.info(f"Retrieved {len(result)} information entries from standard memory system")
                return result[:limit]
            except Exception as e:
                logger.error(f"Error in standard information retrieval: {e}", exc_info=True)
                return []
        except Exception as e:
            logger.error(f"Error retrieving information: {e}", exc_info=True)
            return []
    
    def _parse_information_context(self, context_data: str) -> List[Dict[str, Any]]:
        """
        Parse RAG context data into structured information entries (no significance).
        """
        try:
            import re
            import uuid
            from datetime import datetime
            
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
                    "timestamp": datetime.now().isoformat(),
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
