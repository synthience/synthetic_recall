import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LucidiaConfig:
    """Configuration class for Lucidia voice assistant."""
    llm_server_url: str = os.getenv('LLM_SERVER_URL', 'http://localhost:1234/v1/chat/completions')
    llm_model: str = os.getenv('LLM_MODEL', 'local-model')  # Default model name
    max_tokens: int = int(os.getenv('MAX_TOKENS', '2048'))
    temperature: float = float(os.getenv('TEMPERATURE', '0.7'))
    top_p: float = float(os.getenv('TOP_P', '0.95'))
    context_length: int = int(os.getenv('CONTEXT_LENGTH', '4096'))
    max_history: int = int(os.getenv('MAX_HISTORY', '10'))  # Number of messages to keep in history
