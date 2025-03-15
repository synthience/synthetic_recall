import re
import asyncio
import aiohttp
import logging
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from voice_core.config.config import LLMConfig

class LocalLLMPipeline:
    """Enhanced LLM pipeline with hierarchical memory integration and robust error handling."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM pipeline."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.api_endpoint or "http://127.0.0.1:1234/v1"
        self.memory_client = None
        self.logger = logging.getLogger(__name__)
        
        self.completion_tokens_limit = config.max_tokens or 150
        self._query_history = {}  # Store recent queries for deduplication
        self._response_cache = {}
        self._max_cache_size = 100
        
        self._last_connection_check = 0
        self._connection_check_interval = 60  # seconds
        
        # Memory system configuration
        self._memory_config = {
            'stm_priority': 0.8,        # Priority for Short-Term Memory
            'ltm_priority': 0.5,        # Priority for Long-Term Memory
            'min_significance': 0.3,    # Minimum significance threshold
            'max_context_tokens': 1024,  # Maximum tokens for memory context
            'similarity_threshold': 0.75, # Threshold for similar query detection
            'enable_embedding_cache': True, # Cache embeddings for faster retrieval
            'enable_query_optimization': True, # Optimize queries before processing
        }

    def set_memory_client(self, memory_client):
        """Set memory client for hierarchical memory system integration."""
        self.memory_client = memory_client
        self.logger.info("Memory client attached to LLM pipeline with hierarchical memory support")

    async def initialize(self):
        """Initialize aiohttp session and test API connectivity."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(f"{self.base_url}/models", timeout=5) as resp:
                if resp.status == 200:
                    self.logger.info("Successfully connected to LLM API.")
            self._last_connection_check = time.time()
        except Exception as e:
            self.logger.warning(f"LLM API connectivity test failed: {e}")

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate a response with hierarchical memory enhancement."""
        if not prompt:
            return None

        await self._ensure_connection()
        
        # Check for similar past queries using memory embeddings
        similar_response = await self._check_similar_queries(prompt)
        if similar_response:
            return f"{similar_response} (I've answered this recently.)"
        
        # Get memory context through the Memory Prioritization Layer
        context = await self._get_hierarchical_memory_context(prompt)
        final_prompt = f"{context}\n{prompt}" if context else prompt
        
        effective_system_prompt = system_prompt or self.config.system_prompt or "You are Lucidia, an advanced AI assistant."
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
        
        response = await self._execute_llm_request(payload)
        if response:
            # Store the response in cache
            query_key = prompt.lower().strip()[:100]
            self._response_cache[query_key] = response
            if len(self._response_cache) > self._max_cache_size:
                self._response_cache.pop(next(iter(self._response_cache)))
            
            # Store the response in memory for future retrieval
            if self.memory_client:
                try:
                    await self.memory_client.store_memory(
                        content=response,
                        metadata={
                            "type": "assistant_response",
                            "query": prompt,
                            "timestamp": time.time(),
                            "model": self.config.model
                        },
                        importance=0.7  # Default importance for assistant responses
                    )
                except Exception as e:
                    self.logger.error(f"Error storing response in memory: {e}")
        
        return response
    
    async def _check_similar_queries(self, prompt: str) -> Optional[str]:
        """Check if we've answered a similar query recently using embedding similarity."""
        if not self.memory_client or not self._memory_config['enable_embedding_cache']:
            # Fall back to simple string matching if memory client not available
            query_key = prompt.lower().strip()[:100]
            return self._response_cache.get(query_key)
        
        try:
            # Use embedding comparison for semantic similarity
            recent_queries = list(self._response_cache.items())[-10:]  # Last 10 queries
            
            for query, response in recent_queries:
                try:
                    # Compare current prompt with cached query
                    similarity = await self.memory_client.compare_texts(prompt, query)
                    
                    if similarity >= self._memory_config['similarity_threshold']:
                        self.logger.info(f"Found similar query with similarity {similarity:.2f}")
                        return response
                except Exception as e:
                    self.logger.warning(f"Error comparing query embeddings: {e}")
            
            return None
        except Exception as e:
            self.logger.error(f"Error in similar query check: {e}")
            # Fall back to simple string matching
            query_key = prompt.lower().strip()[:100]
            return self._response_cache.get(query_key)
    
    async def _get_hierarchical_memory_context(self, prompt: str) -> str:
        """Retrieve memory context through the hierarchical memory system."""
        if not self.memory_client:
            return ""
        
        try:
            # First, try to get context from the new hierarchical memory system
            if hasattr(self.memory_client, 'route_query'):
                # Use MPL directly if available
                result = await self.memory_client.route_query(prompt)
                if result and 'context' in result:
                    return result['context']
            
            # Next, try the enhanced RAG context method
            if hasattr(self.memory_client, 'get_rag_context'):
                context = await self.memory_client.get_rag_context(
                    query=prompt, 
                    limit=5, 
                    min_significance=self._memory_config['min_significance'],
                    max_tokens=self._memory_config['max_context_tokens']
                )
                return context if context else ""
            
            # Fall back to legacy method if needed
            return await self._get_implicit_context(prompt)
            
        except Exception as e:
            self.logger.error(f"Error retrieving hierarchical memory context: {e}")
            # Fall back to legacy method
            return await self._get_implicit_context(prompt)
    
    async def _get_implicit_context(self, prompt: str) -> str:
        """Legacy method to retrieve implicit context for the query."""
        if not self.memory_client:
            return ""
        
        try:
            context = await self.memory_client.get_rag_context(
                query=prompt, 
                limit=5, 
                min_significance=self._memory_config['min_significance']
            )
            return context if context else ""
        except Exception as e:
            self.logger.error(f"Error retrieving RAG context: {e}")
            return ""

    async def _execute_llm_request(self, payload: Dict[str, Any]) -> Optional[str]:
        """Execute LLM request with retries and error handling."""
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                # Log the request attempt
                self.logger.info(f"Sending LLM request attempt {attempt+1}/{max_attempts} to {self.base_url}/chat/completions")
                
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload, timeout=30) as resp:
                    # Log the response status and headers for debugging
                    self.logger.info(f"LLM API response status: {resp.status}, content-type: {resp.headers.get('content-type')}")
                    
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"LLM API error (status {resp.status}): {error_text}")
                        if attempt == max_attempts - 1:  # Last attempt
                            return f"I'm sorry, I encountered a service error (HTTP {resp.status})."
                        continue
                    
                    # Try to parse the response as JSON
                    try:
                        data = await resp.json()
                    except Exception as json_error:
                        response_text = await resp.text()
                        self.logger.error(f"Failed to parse LLM API response as JSON: {json_error}")
                        self.logger.error(f"Raw response: {response_text[:500]}..." if len(response_text) > 500 else f"Raw response: {response_text}")
                        return "I'm sorry, I received an invalid response format."
                    
                    # Extract the response text
                    response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    
                    # Log if response is empty
                    if not response:
                        self.logger.warning(f"LLM returned empty content. Full API response: {data}")
                        return "I'm not sure how to respond to that. Could you rephrase your question?"
                    
                    self.logger.info(f"Successful LLM response: {response[:100]}..." if len(response) > 100 else f"Successful LLM response: {response}")
                    return response
                    
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout waiting for LLM API response (attempt {attempt+1}/{max_attempts})")
                await asyncio.sleep(1.5 * (attempt + 1))
            except Exception as e:
                self.logger.error(f"Error in LLM request (attempt {attempt+1}/{max_attempts}): {e}")
                self.logger.error(f"Request payload: {json.dumps(payload)[:500]}..." if len(json.dumps(payload)) > 500 else f"Request payload: {json.dumps(payload)}")
                await asyncio.sleep(1.5 * (attempt + 1))
        
        return "I'm having trouble processing that request right now. Please try again later."
    
    async def _ensure_connection(self):
        """Ensure a valid connection to the LLM API."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            return
        if time.time() - self._last_connection_check > self._connection_check_interval:
            try:
                async with self.session.get(f"{self.base_url}/models", timeout=3) as resp:
                    if resp.status != 200:
                        self.logger.warning("Reinitializing session due to connectivity issue.")
                        await self.close()
                        self.session = aiohttp.ClientSession()
            except Exception as e:
                self.logger.warning(f"LLM API connection check failed: {e}")
                await self.close()
                self.session = aiohttp.ClientSession()
            self._last_connection_check = time.time()
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
