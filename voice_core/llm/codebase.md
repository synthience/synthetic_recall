# __init__.py

```py
"""Local LLM service for voice agent responses"""

import logging
import asyncio
import json
import aiohttp
from typing import Optional, Dict, List

from .llm_pipeline import LocalLLMPipeline
from ..config.config import LLMConfig

logger = logging.getLogger(__name__)

__all__ = ['LocalLLMPipeline', 'LLMConfig', 'LocalLLMService']

class LocalLLMService:
    """Local LLM service using Qwen 2.5 7B model"""
    
    def __init__(self, api_url: str = "http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url
        self.session = None
        self.initialized = False
        self.system_prompt = """You are Lucidia, a helpful voice assistant. Keep your responses natural, concise, and conversational. 
        Avoid long explanations unless asked. Respond in a way that sounds natural when spoken."""
        
    async def initialize(self):
        """Initialize the LLM service"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Test connection to API
            async with self.session.get("http://localhost:1234/v1/models") as response:
                if response.status != 200:
                    raise ConnectionError(f"Failed to connect to LLM API: {response.status}")
                    
            self.initialized = True
            logger.info("✅ LLM service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            if self.session:
                await self.session.close()
            raise
            
    async def generate_response(self, text: str) -> Optional[str]:
        """Generate a response to user input"""
        try:
            if not self.initialized or not self.session:
                logger.warning("⚠️ LLM not initialized")
                return None
                
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Make API request
            async with self.session.post(
                self.api_url,
                json={
                    "messages": messages,
                    "model": "qwen2.5-7b",
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ LLM API error: {response.status}")
                    return None
                    
                data = await response.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return None
            
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False

```

# llm_communication.py

```py
import aiohttp
import json
import logging
import asyncio
from typing import List, Dict, Optional, Union, Generator, Any
from voice_core.config.config import LucidiaConfig

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
config = LucidiaConfig()
LLM_SERVER_URL = config.llm_server_url
MAX_HISTORY_LENGTH = 10
DEFAULT_TIMEOUT = 60

class LLMCommunicationError(Exception):
    """Custom exception for LLM communication errors."""
    pass

async def verify_llm_server() -> bool:
    """Verify LM Studio server connection."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLM_SERVER_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    logger.info("LM Studio server connection verified")
                    return True
                logger.warning(f"LM Studio health check failed: {response.status}")
                return False
    except Exception as e:
        logger.warning(f"Could not connect to LM Studio: {e}")
        return False

class LocalLLM:
    """LM Studio LLM wrapper with OpenAI-compatible interface."""
    def __init__(self):
        self.chat_history = [
            {"role": "system", "content": "You are Lucidia, a helpful and engaging voice assistant."}
        ]
        self.capabilities = type("Capabilities", (), {"streaming": False})()

    async def generate(self, text: str) -> str:
        """Generate a response using LM Studio with OpenAI-compatible payload."""
        try:
            self.chat_history.append({"role": "user", "content": text})
            if len(self.chat_history) > MAX_HISTORY_LENGTH + 1:
                self.chat_history = [self.chat_history[0]] + self.chat_history[-MAX_HISTORY_LENGTH:]

            payload = {
                "model": config.llm.model_name,
                "messages": self.chat_history,
                "temperature": config.llm.temperature,
                "max_tokens": 150,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LLM_SERVER_URL,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        raise LLMCommunicationError(f"LM Studio error: {error_msg}")
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    self.chat_history.append({"role": "assistant", "content": content})
                    return content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I couldn’t generate a response."
```

# llm_pipeline.py

```py
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
            
        # Check if this is a memory-related query
        memory_keywords = ["remember", "memory", "memories", "recall", "forget", "store", "remember me", "know about me"]
        is_memory_query = any(keyword in prompt.lower() for keyword in memory_keywords)
            
        # Get RAG context if enabled and memory client is available
        context = ""
        if use_rag and self.memory_client:
            try:
                context = await self.memory_client.get_rag_context(prompt)
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
        
        # Add memory tools if memory client is available and this is a memory-related query
        if self.memory_client and is_memory_query:
            payload["tools"] = self.memory_client.get_memory_tools()

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
            
            # Handle memory search tool
            if name == "search_memory":
                query = arguments.get("query")
                memory_type = arguments.get("memory_type")
                max_results = arguments.get("max_results", 5)
                min_significance = arguments.get("min_significance", 0.0)
                time_range = arguments.get("time_range")
                
                results = await memory_client.search_memory_tool(
                    query=query,
                    memory_type=memory_type,
                    max_results=max_results,
                    min_significance=min_significance,
                    time_range=time_range
                )
                
                return {"results": results}
                
            # Handle store significant memory tool
            elif name == "store_significant_memory":
                text = arguments.get("text")
                memory_type = arguments.get("memory_type", "important")
                min_significance = arguments.get("min_significance", 0.8)
                
                memory_id = await memory_client.store_significant_memory(
                    text=text,
                    memory_type=memory_type,
                    min_significance=min_significance
                )
                
                return {"memory_id": memory_id, "success": bool(memory_id)}
                
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
```

