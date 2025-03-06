"""Local LLM service for voice agent responses"""

import logging
import asyncio
import json
import aiohttp
from typing import Optional, Dict, List, Any

from .llm_pipeline import LocalLLMPipeline
from ..config.config import LLMConfig

logger = logging.getLogger(__name__)

__all__ = ['LocalLLMPipeline', 'LLMConfig', 'LocalLLMService']

class LocalLLMService:
    """Local LLM service using local LLM models"""
    
    def __init__(self, 
                 api_url: str = "http://localhost:1234/v1/chat/completions",
                 memory_client = None):
        """
        Initialize the LLM service.
        
        Args:
            api_url: URL for the LLM API
            memory_client: Optional memory client for context retrieval
        """
        self.api_url = api_url
        self.session = None
        self.initialized = False
        self.system_prompt = """You are Lucidia, a helpful voice assistant with persistent memory. Keep your responses natural, concise, and conversational. 
        Respond in a way that sounds natural when spoken. Remember past conversations to provide personalized experiences."""
        self.memory_client = memory_client
        
        # Metrics tracking
        self._request_count = 0
        self._error_count = 0
        self._avg_response_time = 0
        self._last_error = None
        
        # Create LLM pipeline if memory client is provided
        if memory_client:
            self.pipeline = LocalLLMPipeline(LLMConfig(
                api_endpoint=api_url,
                system_prompt=self.system_prompt,
                model="qwen2.5-7b",
                temperature=0.7,
                max_tokens=150,
                timeout=30
            ))
            self.pipeline.set_memory_client(memory_client)
        else:
            self.pipeline = None
            
    async def initialize(self):
        """Initialize the LLM service"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Test connection to API
            async with self.session.get(self.api_url.replace("/chat/completions", "/models")) as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to LLM API: {response.status}")
                    if self.session:
                        await self.session.close()
                        self.session = None
                    raise ConnectionError(f"Failed to connect to LLM API: {response.status}")
                    
                # Get available models
                models_data = await response.json()
                available_models = [model.get("id") for model in models_data.get("data", [])]
                logger.info(f"Available LLM models: {available_models}")
            
            # Initialize pipeline if available
            if self.pipeline:
                await self.pipeline.initialize()
                
            self.initialized = True
            logger.info("✅ LLM service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
            
    async def generate_response(self, text: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Generate a response to user input.
        
        Args:
            text: User input text
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated response or None on failure
        """
        start_time = asyncio.get_event_loop().time()
        self._request_count += 1
        
        try:
            # Check if initialized
            if not self.initialized:
                logger.warning("⚠️ LLM not initialized")
                return self._fallback_response("I'm still initializing my knowledge systems. Please try again in a moment.")
            
            # Use pipeline if available (preferred for memory integration)
            if self.pipeline:
                response = await self.pipeline.generate_response(
                    prompt=text,
                    system_prompt=system_prompt or self.system_prompt,
                    use_rag=True
                )
                
                # Update metrics
                elapsed = asyncio.get_event_loop().time() - start_time
                self._update_metrics(elapsed, error=False)
                
                return response
            
            # Fall back to direct API call if pipeline not available
            if not self.session or self.session.closed:
                self.session = aiohttp.ClientSession()
                
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Make API request with retry
            max_retries = 2
            retry_count = 0
            retry_delay = 1.0
            
            while retry_count <= max_retries:
                try:
                    # Set timeout
                    timeout = aiohttp.ClientTimeout(total=30)
                    
                    async with self.session.post(
                        self.api_url,
                        json={
                            "messages": messages,
                            "model": "qwen2.5-7b",
                            "temperature": 0.7,
                            "max_tokens": 150
                        },
                        timeout=timeout
                    ) as response:
                        if response.status != 200:
                            error_body = await response.text()
                            logger.error(f"❌ LLM API error: {response.status} - {error_body}")
                            
                            # Handle recoverable errors
                            if response.status == 429 or response.status >= 500:
                                retry_count += 1
                                if retry_count > max_retries:
                                    self._last_error = f"API error {response.status}: {error_body[:100]}"
                                    self._error_count += 1
                                    return self._fallback_response("I'm experiencing high demand right now. Please try again in a moment.")
                                    
                                # Exponential backoff
                                wait_time = retry_delay * (2 ** (retry_count - 1))
                                logger.info(f"Retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                # Non-recoverable error
                                self._last_error = f"API error {response.status}: {error_body[:100]}"
                                self._error_count += 1
                                return self._fallback_response("I'm having trouble connecting to my knowledge systems.")
                        
                        data = await response.json()
                        response_text = data["choices"][0]["message"]["content"]
                        
                        # Update metrics
                        elapsed = asyncio.get_event_loop().time() - start_time
                        self._update_metrics(elapsed, error=False)
                        
                        return response_text
                
                except asyncio.TimeoutError:
                    retry_count += 1
                    logger.warning(f"LLM request timed out (attempt {retry_count}/{max_retries})")
                    
                    if retry_count > max_retries:
                        self._last_error = "Request timeout"
                        self._error_count += 1
                        return self._fallback_response("I'm taking too long to think. Let me give you a simpler answer.")
                        
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {wait_time:.2f}s after timeout")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"❌ Error generating response: {e}")
                    
                    if retry_count > max_retries:
                        self._last_error = str(e)
                        self._error_count += 1
                        return self._fallback_response("I encountered a technical issue while processing your request.")
                        
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {wait_time:.2f}s after error: {e}")
                    await asyncio.sleep(wait_time)
                    
        except Exception as e:
            # Update metrics
            self._last_error = str(e)
            self._error_count += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            self._update_metrics(elapsed, error=True)
            
            logger.error(f"❌ Unhandled error generating response: {e}")
            return self._fallback_response("I'm sorry, I'm having trouble responding right now.")
    
    def _fallback_response(self, message: str) -> str:
        """Generate a fallback response when normal processing fails."""
        fallback_responses = [
            message,
            "I'm having some technical difficulties at the moment. Let's try again in a bit.",
            "My thinking process seems to be running slower than usual. Could you try a simpler question?",
            "I'm still learning and sometimes I get stuck. Let's try a different approach."
        ]
        
        # Use primary message for most cases, but occasionally use alternatives to sound more natural
        import random
        if random.random() < 0.2:  # 20% chance of alternative message
            return random.choice(fallback_responses[1:])
        return fallback_responses[0]
    
    def _update_metrics(self, elapsed_time: float, error: bool = False) -> None:
        """Update performance metrics."""
        if not error:
            # Update average response time (weighted average)
            if self._request_count > 1:
                self._avg_response_time = (0.9 * self._avg_response_time) + (0.1 * elapsed_time)
            else:
                self._avg_response_time = elapsed_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "avg_response_time": self._avg_response_time,
            "error_rate": self._error_count / max(1, self._request_count),
            "last_error": self._last_error
        }
            
    async def cleanup(self):
        """Clean up resources"""
        # Close pipeline if available
        if self.pipeline:
            await self.pipeline.close()
            
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
            
        self.initialized = False
        logger.info("LLM service cleaned up")