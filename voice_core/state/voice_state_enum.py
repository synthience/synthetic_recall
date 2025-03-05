# voice_state_enum.py

from enum import Enum

class VoiceState(Enum):
    """Possible states for the voice assistant pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    INTERRUPTED = "interrupted"
    ERROR = "error"
