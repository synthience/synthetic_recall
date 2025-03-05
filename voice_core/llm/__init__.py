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
