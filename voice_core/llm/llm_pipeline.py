from __future__ import annotations
import aiohttp
import asyncio
import logging
from typing import Optional
from voice_core.config.config import LLMConfig

logger = logging.getLogger(__name__)

class LocalLLMPipeline:
    """Pipeline for local LLM integration using LM Studio."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "http://127.0.0.1:1234/v1"  # LM Studio default
        self.logger = logging.getLogger(__name__)
        self.memory_client = None

    def set_memory_client(self, memory_client):
        """Set the memory client for RAG context retrieval."""
        self.memory_client = memory_client

    async def initialize(self):
        """Initialize the aiohttp session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.debug("Initialized aiohttp ClientSession")
        try:
            # Test connectivity to LM Studio
            async with self.session.get(f"{self.base_url}/models") as resp:
                resp.raise_for_status()
                self.logger.info("Successfully connected to LM Studio")
        except Exception as e:
            self.logger.warning(f"LM Studio connectivity test failed: {e}")

    async def generate_response(self, prompt: str, use_rag: bool = True) -> str:
        """Generate a complete response from LM Studio with optional RAG context."""
        if not prompt:
            self.logger.warning("Empty prompt provided, returning empty response")
            return ""

        if not self.session or self.session.closed:
            await self.initialize()

        # Get RAG context if enabled and memory client is available
        context = ""
        if use_rag and self.memory_client:
            try:
                context = await self.memory_client.get_rag_context(prompt)
            except Exception as e:
                self.logger.error(f"Failed to get RAG context: {e}")

        # Build final prompt with context
        final_prompt = f"{context}\n{prompt}" if context else prompt

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are Lucidia, a helpful voice assistant. Keep responses natural and conversational."},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response = data["choices"][0]["message"]["content"].strip()
                self.logger.info(f"Generated response: {response[:50]}...")
                return response
        except aiohttp.ClientError as e:
            self.logger.error(f"LM Studio API error: {e}")
            return "Sorry, I couldn’t process that right now."
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_response: {e}", exc_info=True)
            return "An error occurred, please try again."

    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        if hasattr(self, 'model'):
            # Clean up any model resources
            self.model = None

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("Closed aiohttp ClientSession")
        self.session = None