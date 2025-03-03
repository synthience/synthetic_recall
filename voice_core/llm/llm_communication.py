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