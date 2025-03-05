import re
import asyncio
import aiohttp
import logging
import time
import json
from typing import Optional, Dict, Any
from voice_core.config.config import LLMConfig

class LocalLLMPipeline:
    """Enhanced LLM pipeline with RAG integration."""

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

        if not self.session or self.session.closed:
            await self.initialize()
            
        # Check if this is a memory-related query with more comprehensive patterns
        memory_keywords = [
            "remember", "memory", "memories", "recall", "forget", "store", "remember me", "know about me",
            "what did i", "what did we", "what was", "who am i", "my name", "where do i", "when did we",
            "last time", "previous", "earlier", "yesterday", "last week", "talked about", "mentioned",
            "told you", "said", "what's my", "what is my", "do you know my", "do you know who i am"
        ]
        is_memory_query = any(keyword in prompt.lower() for keyword in memory_keywords)
        
        # Get RAG context if enabled and memory client is available
        context = ""
        if use_rag and self.memory_client:
            try:
                # For memory queries, use a higher limit to ensure we get relevant memories
                limit = 8 if is_memory_query else 5
                min_significance = 0.6 if is_memory_query else 0.0
                
                self.logger.info(f"Getting RAG context for prompt: '{prompt[:50]}...' (memory query: {is_memory_query})")
                context = await self.memory_client.get_rag_context(prompt, limit=limit)
                
                # If this is a memory query but we got no context, try a more aggressive search
                if is_memory_query and not context.strip():
                    self.logger.info("No context found for memory query, trying more aggressive search")
                    # Try searching with just key terms from the prompt
                    key_terms = ' '.join([word for word in prompt.split() if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'who', 'how', 'that', 'this', 'these', 'those', 'with', 'about']])
                    if key_terms:
                        context = await self.memory_client.get_rag_context(key_terms, limit=10)
            except Exception as e:
                self.logger.error(f"Failed to get RAG context: {e}")

        # Build final prompt with context
        final_prompt = f"{context}\n{prompt}" if context else prompt
        
        # Use provided system prompt or default
        effective_system_prompt = system_prompt or self.config.system_prompt or "You are Lucidia, an advanced voice assistant with persistent memory. You remember conversations across sessions and continuously learn from interactions."
        
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
                "7. If the user is asking about personal details like their name, location, etc., make sure to directly answer their question first before explaining your memory capabilities\n"
            )
            effective_system_prompt = f"{effective_system_prompt}\n{memory_guidance}"
        
        # Check for near-duplicate queries to avoid repetition
        query_key = prompt.lower().strip()[:100]  # Use first 100 chars as key
        if query_key in self._query_history:
            # If same query within 10 seconds, add variation marker
            last_time = self._query_history[query_key]
            if time.time() - last_time < 10:
                effective_system_prompt += " Note: This appears to be a repeated question, so try to provide an alternative perspective or additional information."
                
        # Update query history
        self._query_history[query_key] = time.time()
        # Limit history size
        if len(self._query_history) > 50:
            # Keep only the 20 most recent queries
            oldest_keys = sorted(self._query_history.items(), key=lambda x: x[1])[:30]
            for key, _ in oldest_keys:
                self._query_history.pop(key, None)

        # Prepare payload
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
        
        # Always include memory tools for memory-related queries
        if self.memory_client:
            if is_memory_query:
                payload["tools"] = self.memory_client.get_memory_tools()
                self.logger.info("Added memory tools to LLM request for memory query")
            elif "name" in prompt.lower() or "who am i" in prompt.lower():
                # Always add tools for name queries even if not detected as memory query
                payload["tools"] = self.memory_client.get_memory_tools()
                self.logger.info("Added memory tools to LLM request for name query")

        # Add response optimization based on prompt complexity
        word_count = len(final_prompt.split())
        if word_count < 10:
            # For short queries, lower temperature slightly for more consistent responses
            payload["temperature"] = max(0.1, self.config.temperature - 0.2)
        elif word_count > 50:
            # For long complex prompts, increase max tokens
            payload["max_tokens"] = min(300, int(self.completion_tokens_limit * 1.5))

        # Try multiple times with different strategies
        for attempt in range(2):
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
                        
                        if attempt == 0:
                            # Simplify prompt on first failure
                            payload["messages"] = [
                                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                                {"role": "user", "content": final_prompt.split("\n\n")[-1]}  # Use just the last part of the prompt
                            ]
                            # Remove tools on retry for simplicity
                            if "tools" in payload:
                                del payload["tools"]
                            await asyncio.sleep(1)  # Brief pause before retry
                            continue
                        else:
                            return "I'm sorry, I couldn't process that request right now."
                    
                    data = await resp.json()
                    
                    # Check if the model wants to use a tool
                    choices = data.get("choices", [{}])
                    if choices and choices[0].get("finish_reason") == "tool_calls":
                        # Handle tool calls
                        message = choices[0].get("message", {})
                        tool_calls = message.get("tool_calls", [])
                        
                        # Process each tool call
                        if tool_calls:
                            # Add the assistant's tool call message to the conversation
                            payload["messages"].append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls
                            })
                            
                            # Process each tool call
                            for tool_call in tool_calls:
                                result = await self._handle_tool_call(tool_call, self.memory_client)
                                # Add the tool result to the conversation
                                payload["messages"].append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.get("id"),
                                    "content": json.dumps(result)
                                })
                            
                            # Make a second request to get the final response
                            async with self.session.post(
                                url, 
                                json=payload, 
                                timeout=aiohttp.ClientTimeout(total=timeout)
                            ) as tool_resp:
                                if tool_resp.status != 200:
                                    self.logger.error(f"LLM API error after tool call: {await tool_resp.text()}")
                                    return "I'm sorry, but that's taking too long to process. Could you try a simpler question?"
                                
                                tool_data = await tool_resp.json()
                                response = tool_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        else:
                            response = "I tried to access my memory but encountered an issue."
                    else:
                        # Regular response (no tool calls)
                        response = choices[0].get("message", {}).get("content", "").strip()
                    
                    if not response:
                        self.logger.warning("LLM returned empty response")
                        if attempt == 0:
                            # Try with simplified prompt on second attempt
                            payload["messages"] = [
                                {"role": "system", "content": "You are a helpful assistant. Be brief."},
                                {"role": "user", "content": final_prompt.split("\n\n")[-1]}  # Use just the user query part
                            ]
                            # Remove tools on retry for simplicity
                            if "tools" in payload:
                                del payload["tools"]
                            continue
                        else:
                            return "I'm sorry, I couldn't generate a response for that."
                    
                    # Process response for voice output
                    processed_response = self._process_response_for_voice(response)
                    self.logger.info(f"Generated response: {processed_response[:50]}...")
                    
                    # Mark both the prompt and response as discussed topics to prevent repetition
                    if self.memory_client:
                        try:
                            # Mark the user's prompt as discussed
                            await self.memory_client.mark_topic_discussed(prompt)
                            
                            # Mark the assistant's response as discussed
                            await self.memory_client.mark_topic_discussed(processed_response)
                            
                            self.logger.info("Marked conversation topics as discussed to prevent repetition")
                        except Exception as e:
                            self.logger.error(f"Error marking topics as discussed: {e}")
                    
                    return processed_response
                    
            except aiohttp.ClientError as e:
                self.logger.error(f"LLM API error (attempt {attempt+1}): {e}")
                if attempt == 0:
                    # Try with simplified prompt on second attempt
                    payload["messages"] = [
                        {"role": "system", "content": "You are a helpful assistant. Be brief."},
                        {"role": "user", "content": final_prompt.split("\n")[-1]}  # Use just the last line
                    ]
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    return "I'm having trouble connecting to my knowledge systems right now."
                    
            except asyncio.TimeoutError:
                self.logger.error(f"LLM API timeout (attempt {attempt+1})")
                if attempt == 0:
                    # Try with simplified prompt on timeout
                    payload["messages"] = [
                        {"role": "system", "content": "You are a helpful assistant. Keep it brief."},
                        {"role": "user", "content": final_prompt.split("\n")[-1]}  # Use just the last line
                    ]
                    # Reduce max tokens for faster response
                    payload["max_tokens"] = min(80, payload["max_tokens"])
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    return "I'm sorry, but that's taking too long to process. Could you try a simpler question?"
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in generate_response (attempt {attempt+1}): {e}", exc_info=True)
                if attempt == 0:
                    # Remove tools on retry for simplicity
                    if "tools" in payload:
                        del payload["tools"]
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    return "I encountered an error while processing your request. Please try again."
        
        # If we get here, all attempts failed
        return "I'm having technical difficulties right now. Please try again in a moment."

    async def _handle_tool_call(self, tool_call, memory_client):
        """Handle a tool call from the LLM.
        
        Args:
            tool_call: Tool call from LLM
            memory_client: Memory client instance
            
        Returns:
            Tool call result
        """
        try:
            name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")
            
            # Parse arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in tool arguments"}
            
            # Check for personal detail queries in search_memory tool calls
            if name == "search_memory":
                query = arguments.get("query", "")
                
                # Check if this is a personal detail query
                personal_detail_patterns = {
                    "name": [r"what.*name", r"who am i", r"call me", r"my name"],
                    "location": [r"where.*live", r"where.*from", r"my location", r"my address"],
                    "birthday": [r"when.*born", r"my birthday", r"my birth date"],
                    "job": [r"what.*do for (a )?living", r"my (job|profession|occupation)"],
                    "family": [r"my (family|wife|husband|partner|child|children|son|daughter)"],
                }
                
                # Check if query matches any personal detail patterns
                for category, patterns in personal_detail_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, query, re.IGNORECASE):
                            self.logger.info(f"Detected personal detail query for {category} in search_memory tool call")
                            
                            # Redirect to get_personal_details tool
                            try:
                                result = await memory_client.get_personal_details_tool(category=category)
                                if result.get("found", False):
                                    self.logger.info(f"Successfully retrieved {category} from personal details")
                                    return {
                                        "results": [
                                            {
                                                "content": f"User {category}: {result.get('value')}",
                                                "significance": 0.95,
                                                "timestamp": time.time()
                                            }
                                        ],
                                        "count": 1
                                    }
                            except Exception as e:
                                self.logger.error(f"Error redirecting to personal details: {e}")
                                # Continue with normal search_memory handling
                
                # Regular search_memory handling
                limit = arguments.get("limit", 5)
                min_significance = arguments.get("min_significance", 0.0)
                
                results = await memory_client.search_memory_tool(
                    query=query,
                    max_results=limit,  # Use limit as max_results
                    min_significance=min_significance
                )
                
                return {"results": results}
                
            # Handle store important memory tool
            elif name == "store_important_memory":
                content = arguments.get("content", "")
                significance = arguments.get("significance", 0.8)
                
                result = await memory_client.store_important_memory(
                    content=content,
                    significance=significance
                )
                
                return result
                
            # Handle get important memories tool
            elif name == "get_important_memories":
                limit = arguments.get("limit", 5)
                min_significance = arguments.get("min_significance", 0.7)
                
                results = await memory_client.get_important_memories(
                    limit=limit,
                    min_significance=min_significance
                )
                
                return results
                
            # Handle get personal details tool
            elif name == "get_personal_details":
                category = arguments.get("category", None)
                
                results = await memory_client.get_personal_details_tool(
                    category=category
                )
                
                return results
                
            # Handle get emotional context tool
            elif name == "get_emotional_context":
                results = await memory_client.get_emotional_context_tool()
                
                return results
                
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            self.logger.error(f"Error handling tool call: {str(e)}")
            return {"error": f"Tool call failed: {str(e)}"}

    async def configure_topic_suppression(self, enable: bool = True, suppression_time: int = None) -> None:
        """Configure topic suppression settings.
        
        Args:
            enable: Whether to enable topic suppression
            suppression_time: Time in seconds to suppress repetitive topics
        """
        if self.memory_client:
            await self.memory_client.configure_topic_suppression(enable, suppression_time)

    async def reset_topic_suppression(self, topic: str = None) -> None:
        """Reset topic suppression for a specific topic or all topics.
        
        Args:
            topic: Specific topic to reset, or None to reset all topics
        """
        if self.memory_client:
            await self.memory_client.reset_topic_suppression(topic)

    async def get_topic_suppression_status(self) -> dict:
        """Get the current status of topic suppression.
        
        Returns:
            Dictionary containing topic suppression status information or None if memory client is not available
        """
        if self.memory_client:
            return await self.memory_client.get_topic_suppression_status()
        return None

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
        response = re.sub(r'Based on (the |your |)context( provided)?[,:]?', '', response)
        response = re.sub(r'According to (the |your |)context[,:]?', '', response)
        
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
        
        # Make contractions more speech-friendly
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",  # Generic handling for most contractions
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'m": " am",
            "'d": " would"
        }
        for contraction, expansion in contractions.items():
            response = response.replace(contraction, expansion)
            
        # Clean up any double spaces
        response = re.sub(r'\s+', ' ', response)
        
        # Ensure end punctuation
        if response and not response[-1] in ['.', '!', '?']:
            response += '.'
            
        return response.strip()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed aiohttp ClientSession")
        self.session = None
        
    async def cleanup(self):
        """Clean up resources."""
        await self.close()