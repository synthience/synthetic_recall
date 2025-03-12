import logging
import aiohttp
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMManager:
    """Manager for handling LLM (Language Model) interactions.
    
    This class provides methods for generating text completions and chat responses,
    abstracting away the details of specific LLM implementations.
    """
    
    def __init__(self, llm_config: Dict[str, Any] = None):
        """Initialize the LLM Manager.
        
        Args:
            llm_config: Configuration for the language model service
        """
        self.config = llm_config or {}
        self.api_base_url = self.config.get("api_base_url", "http://localhost:8000")
        self.local_model = self.config.get("local_model", False)
        self.api_key = self.config.get("api_key", "")
        self.default_model = self.config.get("default_model", "gpt-3.5-turbo")
        self.allow_simulation = self.config.get("allow_simulation", True)
        
        # Connection client for API requests
        self.session = None
        logger.info(f"Initialized LLM Manager with API base: {self.api_base_url}, local model: {self.local_model}, simulation: {'enabled' if self.allow_simulation else 'disabled'}")
    
    async def initialize(self):
        """Initialize the LLM manager, creating API session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return True
    
    async def shutdown(self):
        """Shut down the LLM manager, closing any connections."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate_chat_completion(self, messages: List[Dict[str, str]], model: str = None, 
                                       temperature: float = 0.7, max_tokens: int = 1000,
                                       stream: bool = False) -> Dict[str, Any]:
        """Generate a chat completion from the provided messages.
        
        Args:
            messages: List of message objects with role and content
            model: The model to use for generation
            temperature: Creativity temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            The model's response as a dictionary
        """
        if not self.session:
            await self.initialize()
        
        if stream:
            logger.warning("Streaming not supported yet, falling back to non-streaming")
        
        try:
            # Default response format
            default_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'm sorry, I'm having trouble generating a response right now."
                        }
                    }
                ],
                "usage": {"total_tokens": 0}
            }
            
            # Try direct LLM call or local LLM
            if self.local_model or self.config.get("use_local_llm", False):
                # First try LM Studio-compatible API
                try:
                    # Use the configured API base URL instead of hardcoded localhost
                    async with self.session.post(
                        f"{self.api_base_url}/v1/chat/completions",
                        json={
                            "model": "local-model",
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        },
                        timeout=60  # Local models might take longer
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        logger.warning(f"Local LLM returned status {response.status}")
                except Exception as e:
                    logger.error(f"Error calling local LLM: {e}")
                
                # If that fails, try simulating a response for testing
                if self.allow_simulation:
                    logger.warning("Simulating LLM response for testing")
                    user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                    default_response["choices"][0]["message"]["content"] = (
                        f"This is a simulated response to your message: '{user_message}'. "
                        f"Real LLM integration is not available."
                    )
                    return default_response
                else:
                    raise Exception("LLM simulation is disabled and no valid LLM response was received.")
            
            # Use remote API if not local
            model = model or self.default_model
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with self.session.post(
                f"{self.api_base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            ) as response:
                if response.status == 200:
                    return await response.json()
                
                error_text = await response.text()
                logger.error(f"Error from LLM API: {response.status} - {error_text}")
                return default_response
        
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            return default_response
            
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a text completion from the provided prompt.
        For compatibility with older code that expects a generate() method.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum tokens to generate
            temperature: Creativity temperature (0.0-1.0)
            
        Returns:
            Dictionary with response text
        """
        # Convert the prompt to a chat message format
        messages = [{"role": "user", "content": prompt}]
        
        # Use the chat completion method
        response = await self.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the content from the response and return in expected format
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"text": content, "usage": response.get("usage", {})}
