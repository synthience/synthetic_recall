"""LLM Pipeline for Lucidia Dream Processing

This module provides a minimal implementation of the LLM pipeline interface
used by the dream processing system to generate text and process prompts.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger("LocalLLMPipeline")

class LocalLLMPipeline:
    """Enhanced LLM pipeline with memory integration and robust error handling."""

    def __init__(self, api_endpoint: str = None, model: str = "auto", config: Optional[Any] = None):
        """Initialize the LLM pipeline with the provided configuration.
        
        Args:
            api_endpoint: API endpoint for LLM service
            model: Model identifier to use for generation
            config: LLMConfig object with configuration parameters
        """
        self.config = config
        self.session = None
        
        # Access LLMConfig attributes directly instead of using .get()
        if self.config:
            self.base_url = self.config.api_endpoint
            self.model = self.config.model
            self.completion_tokens_limit = self.config.max_tokens
            self.temperature = self.config.temperature
        else:
            self.base_url = api_endpoint or os.environ.get("LLM_API_ENDPOINT") or "http://127.0.0.1:1234/v1"
            self.model = model
            self.completion_tokens_limit = 1000
            self.temperature = 0.7
            
        self.memory_client = None
        self.logger = logging.getLogger("LocalLLMPipeline")
        self._response_cache = {}
        self._max_cache_size = 100
        
        self._last_connection_check = 0
        self._connection_check_interval = 60  # seconds
        
        self.logger.info(f"Initialized LLM Pipeline with model={model} endpoint={self.base_url}")

    def set_memory_client(self, memory_client):
        """Set memory client for memory system integration."""
        self.memory_client = memory_client
        self.logger.info("Memory client attached to LLM pipeline")

    async def initialize(self):
        """Initialize aiohttp session and test API connectivity."""
        if not self.session or self.session.closed:
            import aiohttp
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(f"{self.base_url}/models", timeout=5) as resp:
                if resp.status == 200:
                    self.logger.info("Successfully connected to LLM API.")
            self._last_connection_check = time.time()
            return True
        except Exception as e:
            self.logger.warning(f"LLM API connectivity test failed: {e}")
            return False

    async def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Temperature parameter (overrides default)
            
        Returns:
            Generated text
        """
        await self._ensure_connection()
        
        effective_max_tokens = max_tokens or self.completion_tokens_limit
        effective_temperature = temperature or self.temperature
        
        # Use the OpenAI-compatible API format
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are Lucidia, an advanced AI assistant with strong reasoning abilities."},
                {"role": "user", "content": prompt}
            ],
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            "stream": False
        }
        
        response = await self._execute_llm_request(payload)
        return response or "I'm not sure how to respond to that."

    async def _execute_llm_request(self, payload: Dict[str, Any]) -> Optional[str]:
        """Execute LLM request with retries and error handling."""
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        self.logger.error(f"LLM API error: {await resp.text()}")
                        continue
                    data = await resp.json()
                    response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    return response if response else "I'm not sure how to answer that."
            except Exception as e:
                self.logger.error(f"Error in LLM request: {e}")
                if attempt < max_attempts - 1:
                    import asyncio
                    await asyncio.sleep(1)  # Brief pause before retry
                    
        return None

    async def _ensure_connection(self) -> bool:
        """Ensure a valid connection to the LLM API."""
        # Check if we need to initialize or if it's been a while since our last check
        import time
        current_time = time.time()
        if not self.session or self.session.closed or current_time - self._last_connection_check > self._connection_check_interval:
            return await self.initialize()
        return True

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the API or tensor service.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        await self._ensure_connection()
        
        try:
            # Use the OpenAI-compatible embeddings endpoint
            payload = {
                "input": text,
                "model": "text-embedding-ada-002"  # Model name for compatibility
            }
            
            async with self.session.post(f"{self.base_url}/embeddings", json=payload, timeout=10) as resp:
                if resp.status != 200:
                    self.logger.error(f"Embedding API error: {await resp.text()}")
                    return [0.0] * 384  # Default embedding dimension
                
                data = await resp.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                return embedding
                
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return [0.0] * 384  # Default embedding dimension

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for sentiment, topics, and entities.
        
        Args:
            text: Input text
            
        Returns:
            Analysis results
        """
        results = {
            "sentiment": "neutral",
            "topics": [],
            "entities": []
        }
        
        # Use the LLM to analyze the text
        prompt = (
            "Analyze the following text and extract:\n"
            "1. Overall sentiment (positive, negative, or neutral)\n"
            "2. Key topics (up to 3)\n"
            "3. Named entities (people, places, organizations)\n\n"
            "Format your response as JSON with keys 'sentiment', 'topics', and 'entities'.\n\n"
            "Text to analyze: " + text
        )
        
        try:
            response = await self.generate(prompt)
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                try:
                    analysis = json.loads(json_match.group(0))
                    # Update results with extracted analysis
                    results.update(analysis)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse analysis JSON")
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            
        return results

    async def get_status(self) -> Dict[str, Any]:
        """Get status of the LLM service.
        
        Returns:
            Status information
        """
        is_connected = await self._ensure_connection()
        
        return {
            "status": "operational" if is_connected else "disconnected",
            "model": self.model,
            "api_endpoint": self.base_url,
            "initialized": is_connected
        }
        
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
