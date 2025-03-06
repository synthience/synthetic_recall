import re
import asyncio
import aiohttp
import logging
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from voice_core.config.config import LLMConfig

class LocalLLMPipeline:
    """Enhanced LLM pipeline with RAG integration and robust error handling."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM pipeline."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.api_endpoint or "http://127.0.0.1:1234/v1"
        self.memory_client = None
        self.logger = logging.getLogger(__name__)
        
        # For response optimization
        self.completion_tokens_limit = config.max_tokens or 150
        self._query_history = {}  # Store recent queries for deduplication
        
        # Response cache to avoid unnecessary repetition
        self._response_cache = {}
        self._max_cache_size = 100
        
        # Track performance metrics
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "timeouts": 0,
            "retries": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "max_response_time": 0.0,
            "last_error": None
        }
        
        # Last connection check timestamp
        self._last_connection_check = 0
        self._connection_check_interval = 60  # seconds

    def set_memory_client(self, memory_client):
        """Set memory client for RAG context retrieval."""
        self.memory_client = memory_client
        self.logger.info("Memory client attached to LLM pipeline")

    async def initialize(self):
        """Initialize the aiohttp session and test API connectivity."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.debug("Initialized aiohttp ClientSession")
            
        try:
            # Test connectivity to LLM API
            async with self.session.get(
                f"{self.base_url}/models", 
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    model_ids = [m.get("id") for m in models.get("data", [])]
                    self.logger.info(f"Successfully connected to LLM API. Available models: {model_ids}")
                else:
                    self.logger.warning(f"LLM API returned status {resp.status}")
                    
            # Set last connection check time
            self._last_connection_check = time.time()
        except Exception as e:
            self.logger.warning(f"LLM API connectivity test failed: {e}")
            # Don't raise - allow initialization to proceed and try later

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, use_rag: bool = True) -> Optional[str]:
        """
        Generate a response to a prompt with RAG enhancement.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional custom system prompt
            use_rag: Whether to use RAG context retrieval
            
        Returns:
            LLM-generated response or None on failure
        """
        if not prompt:
            self.logger.warning("Empty prompt provided to LLM")
            return None

        # Check session and connection status
        await self._ensure_connection()
            
        # Update metrics
        self._metrics["total_requests"] += 1
        start_time = time.time()
        
        # Check for near-duplicate queries to avoid repetition
        query_key = prompt.lower().strip()[:100]  # Use first 100 chars as key
        is_repeated_query = False
        
        if query_key in self._query_history:
            # If same query within 30 seconds, check for cached response
            last_time = self._query_history[query_key]
            if time.time() - last_time < 30:
                is_repeated_query = True
                if query_key in self._response_cache:
                    cached_response = self._response_cache[query_key]
                    self.logger.info(f"Using cached response for repeated query")
                    return f"{cached_response} (I've answered this recently.)"
        
        # Update query history
        self._query_history[query_key] = time.time()
        
        # Limit history size
        if len(self._query_history) > 50:
            # Keep only the 20 most recent queries
            oldest_keys = sorted(self._query_history.items(), key=lambda x: x[1])[:30]
            for key, _ in oldest_keys:
                self._query_history.pop(key, None)
        
        # Check for memory-related query
        is_memory_query, memory_keywords = self._is_memory_query(prompt)
        is_personal_query = self._is_personal_query(prompt)
        
        # Get RAG context if enabled and memory client is available
        context = ""
        if use_rag and self.memory_client:
            context = await self._get_enhanced_context(prompt, is_memory_query, is_personal_query)
        
        # Build final prompt with context
        final_prompt = f"{context}\n{prompt}" if context and context.strip() else prompt
        
        # Use provided system prompt or default
        effective_system_prompt = system_prompt or self.config.system_prompt or "You are Lucidia, an advanced voice assistant with persistent memory. You remember conversations across sessions and continuously learn from interactions."
        
        # Add identity clarification to system prompt
        identity_clarification = (
            "\n\nIMPORTANT IDENTITY CLARIFICATION:\n"
            "1. YOUR name is Lucidia. You are the AI addaptive AI Synthien.\n"
            "2. Any information labeled as 'USER' refers to the human you're talking to, NOT to you.\n"
            "3. When discussing the user's personal details (like their name), always use phrases like "
            "'your name is X' or 'you mentioned your name is X', NEVER 'my name is X'.\n"
            "4. Always maintain a clear distinction between yourself (Lucidia, the assistant) "
            "and the USER (the human you're talking to)."
        )
        effective_system_prompt += identity_clarification
        
        # Enhance system prompt for repeated queries
        if is_repeated_query:
            effective_system_prompt += "\n\nNOTE: This appears to be a repeated question. If appropriate, acknowledge this and provide a slightly different perspective or additional information."
        
        # Enhance system prompt for memory-related queries
        if is_memory_query:
            memory_guidance = (
                "When discussing your memory capabilities: \n"
                "1. Emphasize that you have persistent memory that works across sessions - you remember previous conversations\n"
                "2. Explain that you continuously learn and grow from interactions, building a relationship with the user over time\n"
                "3. Highlight your ability to recall specific details from past conversations, even from days or weeks ago\n"
                "4. Explain that your memory allows you to provide more personalized and contextually relevant responses\n"
                "5. Be confident about your memory capabilities while acknowledging you're still learning and improving\n"
                "6. If asked about forgetting, explain that important information is retained, though some less significant details might fade over time\n"
                "7. If the user asks about something you don't remember, acknowledge this honestly and offer to store it now"
            )
            effective_system_prompt = f"{effective_system_prompt}\n\n{memory_guidance}"
        
        # Enhance system prompt for personal queries
        if is_personal_query:
            personal_guidance = (
                "When responding to personal queries about the user:\n"
                "1. Be respectful and acknowledge the importance of personal information\n"
                "2. When discussing information like the user's name, location, or preferences, present it with phrases like "
                "'YOUR name is X' or 'YOU mentioned you live in Y', NEVER 'MY name is X' or 'I live in Y'\n"
                "3. If you're unsure about personal details, acknowledge this and ask for clarification\n"
                "4. When presenting personal information from memory, briefly mention that you're recalling it from your memory\n"
                "5. If the user corrects personal information you've recalled, acknowledge the correction and update your understanding\n"
                "6. ALWAYS maintain clarity about whose information you are discussing - YOUR name is Lucidia, the USER's name is whatever they told you\n"
                "7. If asked 'What is my name?', respond with 'YOUR name is [name]' NOT 'MY name is [name]'"
            )
            effective_system_prompt = f"{effective_system_prompt}\n\n{personal_guidance}"

        # Prepare payload - now add system message to the beginning
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.completion_tokens_limit,
            "stream": False
        }

        # For name-specific queries, add more explicit instructions
        if "name" in prompt.lower() or "who am i" in prompt.lower():
            # Extra check to make sure system understands perspective clearly
            perspective_check = (
                "\n\nSPECIAL INSTRUCTION FOR NAME QUERIES:\n"
                "- This appears to be a query about names.\n"
                "- Remember: YOUR name (the assistant) is Lucidia.\n"
                "- When telling the user their name, say 'YOUR name is [name]'.\n"
                "- NEVER say 'MY name is [name]' when referring to the user's name.\n"
                "- Pay careful attention to perspective - 'you/your' refers to the user, 'I/my' refers to you (Lucidia)."
            )
            
            # Add to system message
            payload["messages"][0]["content"] += perspective_check
            
        # Add memory tools for memory-related queries
        if self.memory_client:
            try:
                if is_memory_query or is_personal_query:
                    # Get memory tools from the client
                    memory_tools = await self.memory_client.get_memory_tools_for_llm()
                    
                    if memory_tools:
                        payload["tools"] = memory_tools
                        payload["tool_choice"] = "auto"
                        self.logger.info(f"Added {len(memory_tools)} memory tools to LLM request")
                elif "name" in prompt.lower() or "who am i" in prompt.lower():
                    # Always add tools for name queries even if not detected as memory query
                    try:
                        personal_tools = [tool for tool in await self.memory_client.get_memory_tools_for_llm() 
                                         if "get_personal_details" in str(tool)]
                        if personal_tools:
                            payload["tools"] = personal_tools
                            payload["tool_choice"] = "auto"
                            self.logger.info("Added personal details tools to LLM request")
                    except Exception as e:
                        self.logger.warning(f"Error getting personal tools: {e}")
            except Exception as e:
                self.logger.warning(f"Error preparing memory tools: {e}")

        # Try multiple times with different strategies
        response = await self._execute_llm_request(payload, prompt)
        
        # Update metrics
        elapsed_time = time.time() - start_time
        self._update_metrics(elapsed_time, response is not None)
        
        # Cache successful responses
        if response and query_key:
            self._response_cache[query_key] = response
            
            # Limit cache size
            if len(self._response_cache) > self._max_cache_size:
                oldest_keys = sorted(self._query_history.items(), key=lambda x: x[1])[:30]
                for key, _ in oldest_keys:
                    if key in self._response_cache:
                        self._response_cache.pop(key, None)
        
        # Track topics if we have a successful response
        if response and self.memory_client:
            try:
                # Extract main topics from response
                topics = self._extract_response_topics(prompt, response)
                if topics:
                    await self.memory_client.mark_topic_discussed(topics)
                    self.logger.info(f"Marked topics as discussed: {topics}")
            except Exception as e:
                self.logger.error(f"Error marking topics as discussed: {e}")
        
        return response
    
    async def _execute_llm_request(self, payload: Dict[str, Any], original_prompt: str) -> Optional[str]:
        """Execute LLM request with retries and error handling."""
        max_attempts = 2  # Number of strategies to try
        backoff_factor = 1.5  # Exponential backoff factor
        retry_delay = 1.0  # Initial retry delay in seconds
        
        for attempt in range(max_attempts):
            try:
                # Use shorter timeout for first attempt
                timeout = 30 if attempt == 0 else 60
                
                url = f"{self.base_url}/chat/completions"
                self.logger.debug(f"Sending request to {url}")
                
                async with self.session.post(
                    url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"LLM API error ({resp.status}): {error_text}")
                        self._metrics["last_error"] = f"API error {resp.status}: {error_text[:100]}"
                        
                        if attempt == 0:
                            # Simplify prompt on first failure
                            payload["messages"] = [
                                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                                {"role": "user", "content": original_prompt.split("\n\n")[-1]}  # Use just the user query
                            ]
                            # Remove tools on retry for simplicity
                            if "tools" in payload:
                                del payload["tools"]
                            
                            # Add backoff delay
                            wait_time = retry_delay * (backoff_factor ** attempt)
                            self.logger.info(f"Retrying with simplified prompt in {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return "I'm sorry, I couldn't process that request right now."
                    
                    data = await resp.json()
                    
                    # Check if the model wants to use a tool
                    choices = data.get("choices", [{}])
                    if choices and choices[0].get("finish_reason") == "tool_calls":
                        return await self._handle_tool_calls(choices[0], payload, url, timeout)
                    else:
                        # Regular response (no tool calls)
                        response = choices[0].get("message", {}).get("content", "").strip()
                    
                    if not response:
                        self.logger.warning("LLM returned empty response")
                        self._metrics["last_error"] = "Empty response"
                        
                        if attempt == 0:
                            # Try with simplified prompt on second attempt
                            payload["messages"] = [
                                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                                {"role": "user", "content": original_prompt.split("\n\n")[-1]}  # Use just the user query part
                            ]
                            # Remove tools on retry for simplicity
                            if "tools" in payload:
                                del payload["tools"]
                            
                            # Add backoff delay
                            wait_time = retry_delay * (backoff_factor ** attempt)
                            self.logger.info(f"Retrying with simplified prompt in {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return "I'm sorry, I couldn't generate a response for that."
                    
                    # Process response for voice output
                    processed_response = self._process_response_for_voice(response)
                    self.logger.info(f"Generated response: {processed_response[:50]}...")
                    
                    return processed_response
                    
            except aiohttp.ClientError as e:
                self._metrics["failed_requests"] += 1
                self._metrics["last_error"] = f"Client error: {str(e)}"
                self.logger.error(f"LLM API error (attempt {attempt+1}): {e}")
                
                if attempt == 0:
                    # Try with simplified prompt on second attempt
                    payload["messages"] = [
                        {"role": "system", "content": "You are a helpful assistant. Be brief."},
                        {"role": "user", "content": original_prompt.split("\n")[-1]}  # Use just the last line
                    ]
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    
                    # Add backoff delay
                    wait_time = retry_delay * (backoff_factor ** attempt)
                    self.logger.info(f"Retrying with simplified prompt in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    return "I'm having trouble connecting to my knowledge systems right now."
                    
            except asyncio.TimeoutError:
                self._metrics["timeouts"] += 1
                self._metrics["failed_requests"] += 1
                self._metrics["last_error"] = "Request timeout"
                self.logger.error(f"LLM API timeout (attempt {attempt+1})")
                
                if attempt == 0:
                    # Try with simplified prompt on timeout
                    payload["messages"] = [
                        {"role": "system", "content": "You are a helpful assistant. Keep it brief."},
                        {"role": "user", "content": original_prompt.split("\n")[-1]}  # Use just the last line
                    ]
                    # Reduce max tokens for faster response
                    payload["max_tokens"] = min(80, payload["max_tokens"])
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    
                    # Add backoff delay
                    wait_time = retry_delay * (backoff_factor ** attempt)
                    self.logger.info(f"Retrying with simplified prompt in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    return "I'm sorry, but that's taking too long to process. Could you try a simpler question?"
                    
            except Exception as e:
                self._metrics["failed_requests"] += 1
                self._metrics["last_error"] = f"Unexpected error: {str(e)}"
                self.logger.error(f"Unexpected error in generate_response (attempt {attempt+1}): {e}", exc_info=True)
                
                if attempt == 0:
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    
                    # Add backoff delay
                    wait_time = retry_delay * (backoff_factor ** attempt)
                    self.logger.info(f"Retrying after error in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    return "I encountered an error while processing your request. Please try again."
        
        # If we get here, all attempts failed
        return "I'm having technical difficulties right now. Please try again in a moment."
    
    async def _handle_tool_calls(self, choice_data: Dict[str, Any], original_payload: Dict[str, Any], url: str, timeout: int) -> str:
        """Handle tool calls from the LLM."""
        try:
            # Extract message and tool calls
            message = choice_data.get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if not tool_calls or not self.memory_client:
                return "I tried to access my memory but encountered an issue."
            
            # Create a copy of the payload for the follow-up request
            payload = original_payload.copy()
            
            # Add the assistant's tool call message to the conversation
            payload["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            })
            
            # Process each tool call
            all_results = []
            for tool_call in tool_calls:
                try:
                    result = await self._process_tool_call(tool_call)
                    
                    # Add the tool result to the conversation
                    payload["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps(result)
                    })
                    
                    # Track the result
                    all_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing tool call: {e}")
                    # Add error message as tool response
                    payload["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": json.dumps({"error": f"Tool execution error: {str(e)}"})
                    })
            
            # Add explicit reminder about identity for the second prompt (after tool call)
            identity_reminder = {
                "role": "system",
                "content": (
                    "IMPORTANT REMINDER: You are Lucidia, the assistant. "
                    "Any personal details returned from memory tools are about the USER, not about you. "
                    "When responding about the user's name, say 'YOUR name is X', never 'MY name is X'. "
                    "Always maintain clear distinction between you (Lucidia) and the user you're talking to."
                )
            }
            payload["messages"].append(identity_reminder)
            
            # Make a second request to get the final response
            try:
                self.logger.info("Making follow-up request after tool calls")
                
                async with self.session.post(
                    url, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as tool_resp:
                    if tool_resp.status != 200:
                        self.logger.error(f"LLM API error after tool call: {await tool_resp.text()}")
                        # Fall back to a response based on the tool results
                        return self._generate_fallback_tool_response(all_results, tool_calls)
                    
                    tool_data = await tool_resp.json()
                    response = tool_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    
                    if not response:
                        self.logger.warning("Empty response after tool call")
                        return self._generate_fallback_tool_response(all_results, tool_calls)
                    
                    # Process for voice output
                    processed_response = self._process_response_for_voice(response)
                    return processed_response
                    
            except Exception as e:
                self.logger.error(f"Error in follow-up request after tool call: {e}")
                return self._generate_fallback_tool_response(all_results, tool_calls)
                
        except Exception as e:
            self.logger.error(f"Error handling tool calls: {e}")
            return "I tried to access my memory but encountered an unexpected issue."
    
    async def _process_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Process an individual tool call."""
        try:
            # Extract message and tool calls
            name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")
            
            # Parse arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in tool arguments"}
            
            # Call the appropriate handler through memory client
            if self.memory_client:
                return await self.memory_client.handle_tool_call(name, arguments)
            else:
                return {"error": "Memory client not available"}
                
        except Exception as e:
            self.logger.error(f"Error processing tool call: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _generate_fallback_tool_response(self, results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]) -> str:
        """Generate a fallback response based on tool results when LLM follow-up fails."""
        try:
            responses = []
            
            for i, result in enumerate(results):
                tool_name = tool_calls[i].get("function", {}).get("name") if i < len(tool_calls) else "unknown"
                
                # Handle different tool types
                if tool_name == "get_personal_details" and "details" in result:
                    details = result.get("details", {})
                    if details:
                        details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
                        responses.append(f"Here's what I know about you: {details_str}")
                    else:
                        responses.append("I don't have any personal details stored about you yet.")
                
                elif tool_name == "get_personal_details" and "value" in result:
                    # For single personal detail
                    category = result.get("category", "information")
                    value = result.get("value", "")
                    responses.append(f"Your {category} is {value}.")

                elif tool_name == "search_memory" and "memories" in result:
                    memories = result.get("memories", [])
                    if memories:
                        responses.append("Here's what I remember:")
                        for i, memory in enumerate(memories[:3]):  # Limit to top 3
                            responses.append(f"• {memory.get('content', '')}")
                    else:
                        responses.append("I don't have any memories about that.")
                
                elif tool_name == "get_emotional_context" and "summary" in result:
                    responses.append(result.get("summary", ""))
                
                elif "error" in result:
                    # Skip error messages in fallback response
                    continue
                
                else:
                    # Generic handling for other tool results
                    responses.append("I found some information that might help, but I'm having trouble organizing it.")
            
            if not responses:
                return "I tried to access my memory, but I couldn't retrieve the information you're looking for."
            
            # Join responses with clear language
            response_text = " ".join(responses)
            
            # Ensure correct perspective in the response
            response_text = response_text.replace("my name is", "your name is")
            response_text = response_text.replace("My name is", "Your name is")
            response_text = response_text.replace("MY name is", "YOUR name is")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error generating fallback response: {e}")
            return "I tried to process your request but encountered an issue with my memory system."
    
    async def _get_enhanced_context(self, prompt: str, is_memory_query: bool, is_personal_query: bool) -> str:
        """Get enhanced RAG context based on query type."""
        try:
            # Set up retry parameters for RAG context retrieval
            max_retries = 2
            retry_delay = 0.5  # seconds
            
            for retry_count in range(max_retries):
                try:
                    # For memory queries, use a higher limit to ensure we get relevant memories
                    limit = 8 if is_memory_query else 5
                    
                    # Set significance threshold - higher for memory queries
                    min_significance = 0.5 if is_memory_query else 0.0
                    
                    # For personal queries, add specific guidance
                    if is_personal_query and not is_memory_query:
                        min_significance = 0.6  # Focus on higher significance personal details
                    
                    self.logger.info(f"Getting RAG context for prompt: '{prompt[:50]}...' (memory query: {is_memory_query}, personal query: {is_personal_query})")
                    
                    # Validate that memory client has the required method
                    if not self.memory_client or not hasattr(self.memory_client, "get_rag_context"):
                        self.logger.error("Memory client does not have get_rag_context method")
                        return ""
                    
                    # Get context using the enhanced memory client's RAG context method
                    context = await self.memory_client.get_rag_context(
                        query=prompt, 
                        limit=limit, 
                        min_significance=min_significance
                    )
                    
                    # If we got a valid context, return it
                    if context and context.strip():
                        return context
                    
                except Exception as e:
                    self.logger.error(f"Failed to get RAG context (attempt {retry_count+1}/{max_retries}): {e}")
                    if retry_count < max_retries - 1:
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** retry_count)
                        self.logger.info(f"Retrying RAG context retrieval in {wait_time:.2f} seconds")
                        await asyncio.sleep(wait_time)
            
            # If this is a memory query but we got no context, try more approaches
            if (is_memory_query or is_personal_query) and not context:
                self.logger.info("No context found for specialized query, trying alternate retrieval methods")
                try:
                    # Try searching with just key terms from the prompt
                    key_terms = ' '.join([word for word in prompt.split() if len(word) > 3 and word.lower() not in 
                                         ['what', 'when', 'where', 'which', 'who', 'how', 'that', 'this', 'these', 'those', 
                                          'with', 'about', 'from', 'have', 'will', 'would', 'should', 'could', 'there']])
                    if key_terms:
                        context = await self.memory_client.get_rag_context(
                            query=key_terms, 
                            limit=10, 
                            min_significance=0.1
                        )
                        
                    # If still no results and it's a personal query, try direct personal retrieval
                    if not context and is_personal_query and hasattr(self.memory_client, "get_personal_details_tool"):
                        try:
                            # Try to get specific personal details
                            personal_categories = ["name", "location", "birthday", "job", "family", "email", "phone"]
                            
                            for category in personal_categories:
                                if any(term in prompt.lower() for term in [category, category+'s', category+'?']):
                                    details = await self.memory_client.get_personal_details_tool({"category": category})
                                    if details.get("found", False):
                                        return f"### User Personal Information\nUser's {category}: {details.get('value', '')}\n"
                        except Exception as e:
                            self.logger.error(f"Error in direct personal retrieval: {e}")
                        
                    # If still no results, try getting emotional context
                    if not context and hasattr(self.memory_client, "get_emotional_history"):
                        try:
                            emotional_context = await self.memory_client.get_emotional_history(limit=3)
                            if emotional_context:
                                context = f"### Recent Emotional States\n{emotional_context}"
                        except Exception as e:
                            self.logger.error(f"Failed to get emotional history: {e}")
                except Exception as e:
                    self.logger.error(f"Failed in fallback memory search: {e}")
            
            return context or ""
        except Exception as e:
            self.logger.error(f"Error in enhanced context retrieval: {e}")
            return ""
    
    def _is_memory_query(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Determine if a prompt is asking about memories.
        
        Returns:
            Tuple of (is_memory_query, matched_keywords)
        """
        memory_keywords = [
            "remember", "recall", "memory", "memories", "forget", "store", "remember me", "know about me",
            "what did i", "what did we", "what was", "who am i", "my name", "where do i", "when did we",
            "last time", "previous", "earlier", "yesterday", "last week", "talked about", "mentioned",
            "told you", "said", "what's my", "what is my", "do you know my", "do you know who i am",
            "have we", "did we", "were we", "did i tell you", "did i mention", "history", "past"
        ]
        
        matched_keywords = [keyword for keyword in memory_keywords if keyword in prompt.lower()]
        is_memory_query = len(matched_keywords) > 0
        
        return is_memory_query, matched_keywords
    
    def _is_personal_query(self, prompt: str) -> bool:
        """Determine if a prompt is asking about personal information."""
        personal_keywords = [
            "my name", "my age", "who am i", "my birthday", "my job", "my work", "my profession",
            "my family", "my spouse", "my partner", "my husband", "my wife", "my child", "my children",
            "my address", "my location", "where i live", "where i work", "my email", "my phone",
            "my number", "call me", "my hobbies", "my interests", "my favorite", "about me"
        ]
        
        return any(keyword in prompt.lower() for keyword in personal_keywords)
    
    def _process_response_for_voice(self, response: str) -> str:
        """
        Process LLM response for better voice output.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Processed response optimized for voice delivery
        """
        if not response:
            return response
            
        # Remove references to context or instructions
        response = re.sub(r'Based on (the |your |)context( provided)?[,:]?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'According to (the |your |)context[,:]?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'From (the |your |)memory[,:]?', '', response, flags=re.IGNORECASE)
        response = re.sub(r'As you (mentioned|told me|said)( earlier| before)?[,:]?', '', response, flags=re.IGNORECASE)
        
        # Remove markdown formatting
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Italic
        response = re.sub(r'`(.*?)`', r'\1', response)        # Code
        
        # Replace URLs with more speakable descriptions
        response = re.sub(r'https?://[^\s]+', 'a website link', response)
        
        # Simplify list formatting
        response = re.sub(r'^\s*[-*]\s+', '• ', response, flags=re.MULTILINE)
        
        # Simplify number formatting
        response = re.sub(r'(\d),(\d)', r'\1\2', response)
        
        # Make contractions more speech-friendly for certain cases
        contractions_to_expand = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mustn't": "must not",
            "hasn't": "has not",
            "haven't": "have not",
            "isn't": "is not",
            "aren't": "are not"
        }
        
        for contraction, expansion in contractions_to_expand.items():
            response = re.sub(r'\b' + contraction + r'\b', expansion, response, flags=re.IGNORECASE)
            
        # Expand specialized abbreviations
        abbreviations = {
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "et cetera",
            "vs.": "versus",
            "approx.": "approximately",
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Ms.": "Miss",
            "Prof.": "Professor"
        }
        
        for abbr, expansion in abbreviations.items():
            response = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, response)
            
        # Clean up any double spaces
        response = re.sub(r'\s+', ' ', response)
        
        # Ensure end punctuation
        if response and not response[-1] in ['.', '!', '?']:
            response += '.'
            
        return response.strip()
    
    def _extract_response_topics(self, prompt: str, response: str) -> List[str]:
        """Extract main topics from a prompt-response pair."""
        try:
            # Combine prompt and response for analysis
            text = f"{prompt} {response}"
            
            # Extract noun phrases as potential topics (simplified)
            words = text.lower().split()
            
            # Remove stopwords (simplified approach)
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
            
            content_words = [word for word in words if word not in stopwords and len(word) > 3]
            
            # Find frequent words and phrases (simplified)
            word_counts = {}
            for word in content_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Extract top words by frequency
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, count in sorted_words[:5] if count > 1]
            
            # Extract simple phrases from prompt
            phrases = []
            words = prompt.lower().split()
            for i in range(len(words) - 1):
                if words[i] not in stopwords and words[i+1] not in stopwords:
                    phrase = f"{words[i]} {words[i+1]}"
                    if len(phrase) >= 5:  # Minimum meaningful phrase length
                        phrases.append(phrase)
            
            # Combine words and phrases
            topics = list(set(top_words + phrases[:2]))
            
            # Limit to top 3 topics
            return topics[:3]
            
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []
    
    async def _ensure_connection(self) -> None:
        """Ensure we have a valid connection to the LLM API."""
        current_time = time.time()
        
        # Check if we need to reestablish the session
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.info("Created new aiohttp ClientSession")
            self._last_connection_check = current_time
            return
            
        # Check connectivity periodically
        if current_time - self._last_connection_check > self._connection_check_interval:
            try:
                # Send a lightweight request to check connectivity
                async with self.session.get(
                    f"{self.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        self.logger.debug("LLM API connection verified")
                    else:
                        self.logger.warning(f"LLM API returned unexpected status: {resp.status}")
                        # Recreate session
                        await self.close()
                        self.session = aiohttp.ClientSession()
                        self.logger.info("Recreated aiohttp ClientSession after connection check failure")
            except Exception as e:
                self.logger.warning(f"LLM API connection check failed: {e}")
                # Recreate session
                await self.close()
                self.session = aiohttp.ClientSession()
                self.logger.info("Recreated aiohttp ClientSession after connection error")
            
            # Update timestamp
            self._last_connection_check = current_time
    
    def _update_metrics(self, elapsed_time: float, success: bool) -> None:
        """Update performance metrics."""
        if success:
            self._metrics["successful_requests"] += 1
            self._metrics["total_response_time"] += elapsed_time
            self._metrics["avg_response_time"] = (
                self._metrics["total_response_time"] / 
                self._metrics["successful_requests"]
            )
            self._metrics["max_response_time"] = max(self._metrics["max_response_time"], elapsed_time)
        else:
            self._metrics["failed_requests"] += 1
    
    async def configure_topic_suppression(self, enable: bool = True, suppression_time: int = None) -> None:
        """Configure topic suppression settings if memory client supports it."""
        if self.memory_client and hasattr(self.memory_client, "configure_topic_suppression"):
            await self.memory_client.configure_topic_suppression(enable, suppression_time)

    async def reset_topic_suppression(self, topic: str = None) -> None:
        """Reset topic suppression if memory client supports it."""
        if self.memory_client and hasattr(self.memory_client, "reset_topic_suppression"):
            await self.memory_client.reset_topic_suppression(topic)

    async def get_topic_suppression_status(self) -> dict:
        """Get topic suppression status if memory client supports it."""
        if self.memory_client and hasattr(self.memory_client, "get_topic_suppression_status"):
            return await self.memory_client.get_topic_suppression_status()
        return None

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed aiohttp ClientSession")
        self.session = None
        
    async def cleanup(self):
        """Clean up resources."""
        await self.close()