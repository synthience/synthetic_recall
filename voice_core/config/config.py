# voice_core/config/config.py
"""Configuration classes for the voice assistant."""

import os
import torch
from dotenv import load_dotenv
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class WhisperConfig:
    """Configuration for Whisper STT"""
    model_name: str = field(default_factory=lambda: os.getenv('WHISPER_MODEL', 'small'))
    language: str = field(default_factory=lambda: os.getenv('WHISPER_LANGUAGE', 'en'))
    sample_rate: int = 16000  # Fixed at 16kHz for Whisper
    num_channels: int = 1
    vad_threshold: float = 0.15  # Reduced from 0.25 for less sensitivity
    chunk_duration_ms: int = 1000
    silence_duration_ms: int = 1000  # Increased from 500 for more silence tolerance
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    min_audio_length: float = 0.5
    max_audio_length: float = 30.0
    initial_silence_threshold: float = -45.0
    silence_duration: float = 1.0  # Increased from 0.5
    noise_floor: float = -65.0
    max_buffer_length: int = 30000
    beam_size: int = 5
    best_of: int = 3
    temperature: float = 0.0
    patience: float = 1.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.8  # Increased from 0.6 for less sensitivity
    condition_on_previous: bool = True
    initial_prompt: str = None
    fp16: bool = True
    verbose: bool = False
    bandpass_low: float = 100.0
    bandpass_high: float = 4000.0
    min_speech_duration: float = 0.8  # Increased from 0.5
    heartbeat_interval: int = 10
    max_init_retries: int = 3
    speech_confidence_threshold: float = 0.2  # Reduced from 0.3 for less sensitivity
    speech_confidence_decay: float = 0.05  # Reduced from 0.1 for slower confidence decay
    speech_confidence_boost: float = 0.3
    max_low_energy_frames: int = 8  # Increased from 5
    energy_threshold_end: float = 15.0  # Reduced from 20.0
    fine_tuned_model_path: str = field(
        default_factory=lambda: os.path.join(
            'voice_core', 'models', 'whisper', 'whisper-small-personal-voice.pt'
        )
    )
    use_fine_tuned_model: bool = True  # Flag to use the fine-tuned model

    def __post_init__(self):
        cuda_available = torch.cuda.is_available()
        if self.device == "cuda" and not cuda_available:
            logger.warning("CUDA is not available, defaulting to CPU")
            self.device = "cpu"
        
        logger.info(f"🎙️ Whisper config: model={self.model_name}, device={self.device}, sr={self.sample_rate}Hz")

@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    model: str = "qweniversal_studios-1m-7b"
    api_endpoint: str = "http://localhost:1234/v1"
    temperature: float = 0.7
    max_tokens: int = 150
    system_prompt: str = "You are Lucidia, a helpful voice assistant. Keep your responses concise and natural for spoken conversation."
    timeout: float = 15.0
    stream: bool = True
    
    def __post_init__(self):
        if os.getenv("LLM_MODEL"):
            self.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_ENDPOINT"):
            self.api_endpoint = os.getenv("LLM_API_ENDPOINT")
        if os.getenv("LLM_TEMPERATURE"):
            self.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        if os.getenv("LLM_SYSTEM_PROMPT"):
            self.system_prompt = os.getenv("LLM_SYSTEM_PROMPT")
        
        logger.info(f"🤖 LLM config: model={self.model}, temp={self.temperature}, max_tokens={self.max_tokens}")

@dataclass
class VoskConfig:
    """Configuration for Vosk STT"""
    model_name: str = field(default_factory=lambda: os.getenv('VOSK_MODEL', 'small'))
    model_path: str = field(default_factory=lambda: os.path.join('voice_core', 'models', 'vosk', 'en-us-0-22'))
    sample_rate: int = 16000
    num_channels: int = 1

    def __post_init__(self):
        if os.getenv("VOSK_MODEL_PATH"):
            self.model_path = os.getenv("VOSK_MODEL_PATH")
        logger.info(f"🎙️ Vosk config: model={self.model_name}, path={self.model_path}")

@dataclass
class TTSConfig:
    """Configuration for TTS service."""
    voice: str = "en-US-AvaMultilingualNeural"
    sample_rate: int = 24000
    channels: int = 1
    ssml_enabled: bool = True
    cache_enabled: bool = True
    cache_size: int = 50
    
    def __post_init__(self):
        if os.getenv("EDGE_TTS_VOICE"):
            self.voice = os.getenv("EDGE_TTS_VOICE")
        if os.getenv("TTS_SAMPLE_RATE"):
            self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE"))
        if os.getenv("TTS_CHANNELS"):
            self.channels = int(os.getenv("TTS_CHANNELS"))
        if os.getenv("TTS_SSML_ENABLED"):
            self.ssml_enabled = os.getenv("TTS_SSML_ENABLED").lower() in ["true", "1", "yes"]
        if os.getenv("TTS_CACHE_ENABLED"):
            self.cache_enabled = os.getenv("TTS_CACHE_ENABLED").lower() in ["true", "1", "yes"]
        if os.getenv("TTS_CACHE_SIZE"):
            self.cache_size = int(os.getenv("TTS_CACHE_SIZE"))

@dataclass
class StateConfig:
    """Configuration for state management."""
    processing_timeout: float = 30.0
    speaking_timeout: float = 120.0
    vad_silence_threshold: float = 1.0
    debug: bool = False
    
    def __post_init__(self):
        if os.getenv("PROCESSING_TIMEOUT"):
            self.processing_timeout = float(os.getenv("PROCESSING_TIMEOUT"))
        if os.getenv("SPEAKING_TIMEOUT"):
            self.speaking_timeout = float(os.getenv("SPEAKING_TIMEOUT"))
        if os.getenv("VAD_SILENCE_THRESHOLD"):
            self.vad_silence_threshold = float(os.getenv("VAD_SILENCE_THRESHOLD"))
        if os.getenv("STATE_DEBUG"):
            self.debug = os.getenv("STATE_DEBUG").lower() in ["true", "1", "yes"]

@dataclass
class RoomConfig:
    """Configuration for LiveKit room."""
    url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "secret"
    room_name: str = "playground"
    sample_rate: int = 48000  # LiveKit standard
    channels: int = 1
    chunk_size: int = 480  # 10ms at 48kHz
    buffer_size: int = 4800  # 100ms buffer
    
    def __post_init__(self):
        if os.getenv("LIVEKIT_URL"):
            self.url = os.getenv("LIVEKIT_URL")
        if os.getenv("LIVEKIT_API_KEY"):
            self.api_key = os.getenv("LIVEKIT_API_KEY")
        if os.getenv("LIVEKIT_API_SECRET"):
            self.api_secret = os.getenv("LIVEKIT_API_SECRET")
        if os.getenv("ROOM_NAME"):
            self.room_name = os.getenv("ROOM_NAME")

@dataclass
class LucidiaConfig:
    """Main configuration for the voice assistant."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    vosk: VoskConfig = field(default_factory=VoskConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    state: StateConfig = field(default_factory=StateConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    initial_greeting: str = "Hello! I'm Lucidia, your voice assistant. How can I help you today?"
    
    def __post_init__(self):
        if os.getenv("INITIAL_GREETING"):
            self.initial_greeting = os.getenv("INITIAL_GREETING")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "model": self.llm.model,
                "api_endpoint": self.llm.api_endpoint,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "system_prompt": self.llm.system_prompt,
                "timeout": self.llm.timeout,
                "stream": self.llm.stream
            },
            "whisper": {
                "model_name": self.whisper.model_name,
                "device": self.whisper.device,
                "language": self.whisper.language,
                "sample_rate": self.whisper.sample_rate,
                "vad_threshold": self.whisper.vad_threshold,
                "min_speech_duration": self.whisper.min_speech_duration,
                "max_audio_length": self.whisper.max_audio_length,
                "speech_confidence_threshold": self.whisper.speech_confidence_threshold,
                "fine_tuned_model_path": self.whisper.fine_tuned_model_path,
                "use_fine_tuned_model": self.whisper.use_fine_tuned_model
            },
            "vosk": {
                "model_name": self.vosk.model_name,
                "model_path": self.vosk.model_path,
                "sample_rate": self.vosk.sample_rate,
                "num_channels": self.vosk.num_channels
            },
            "tts": {
                "voice": self.tts.voice,
                "sample_rate": self.tts.sample_rate,
                "channels": self.tts.channels,
                "ssml_enabled": self.tts.ssml_enabled,
                "cache_enabled": self.tts.cache_enabled,
                "cache_size": self.tts.cache_size
            },
            "state": {
                "processing_timeout": self.state.processing_timeout,
                "speaking_timeout": self.state.speaking_timeout,
                "vad_silence_threshold": self.state.vad_silence_threshold,
                "debug": self.state.debug
            },
            "room": {
                "url": self.room.url,
                "api_key": self.room.api_key,
                "api_secret": "[REDACTED]",
                "room_name": self.room.room_name,
                "sample_rate": self.room.sample_rate,
                "channels": self.room.channels,
                "chunk_size": self.room.chunk_size,
                "buffer_size": self.room.buffer_size
            },
            "initial_greeting": self.initial_greeting
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LucidiaConfig':
        """Create configuration from dictionary."""
        llm_config = LLMConfig(
            model=config_dict.get("llm", {}).get("model", "qwen2.5-7b-instruct-1m"),
            api_endpoint=config_dict.get("llm", {}).get("api_endpoint", "http://localhost:1234/v1"),
            temperature=config_dict.get("llm", {}).get("temperature", 0.7),
            max_tokens=config_dict.get("llm", {}).get("max_tokens", 150),
            system_prompt=config_dict.get("llm", {}).get("system_prompt", "You are Lucidia, a helpful voice assistant."),
            timeout=config_dict.get("llm", {}).get("timeout", 15.0),
            stream=config_dict.get("llm", {}).get("stream", True)
        )
        
        whisper_config = WhisperConfig(
            model_name=config_dict.get("whisper", {}).get("model_name", "base"),
            device=config_dict.get("whisper", {}).get("device", "cpu"),
            language=config_dict.get("whisper", {}).get("language", "en"),
            sample_rate=config_dict.get("whisper", {}).get("sample_rate", 16000),
            vad_threshold=config_dict.get("whisper", {}).get("vad_threshold", 0.25),
            min_speech_duration=config_dict.get("whisper", {}).get("min_speech_duration", 0.5),
            max_audio_length=config_dict.get("whisper", {}).get("max_audio_length", 30.0),
            speech_confidence_threshold=config_dict.get("whisper", {}).get("speech_confidence_threshold", 0.3),
            fine_tuned_model_path=config_dict.get("whisper", {}).get("fine_tuned_model_path", ""),
            use_fine_tuned_model=config_dict.get("whisper", {}).get("use_fine_tuned_model", False)
        )
        
        vosk_config = VoskConfig(
            model_name=config_dict.get("vosk", {}).get("model_name", "small"),
            model_path=config_dict.get("vosk", {}).get("model_path", os.path.join('voice_core', 'models', 'vosk', 'en-us-0-22')),
            sample_rate=config_dict.get("vosk", {}).get("sample_rate", 16000),
            num_channels=config_dict.get("vosk", {}).get("num_channels", 1)
        )
        
        tts_config = TTSConfig(
            voice=config_dict.get("tts", {}).get("voice", "en-US-AvaMultilingualNeural"),
            sample_rate=config_dict.get("tts", {}).get("sample_rate", 24000),
            channels=config_dict.get("tts", {}).get("channels", 1),
            ssml_enabled=config_dict.get("tts", {}).get("ssml_enabled", True),
            cache_enabled=config_dict.get("tts", {}).get("cache_enabled", True),
            cache_size=config_dict.get("tts", {}).get("cache_size", 50)
        )
        
        state_config = StateConfig(
            processing_timeout=config_dict.get("state", {}).get("processing_timeout", 30.0),
            speaking_timeout=config_dict.get("state", {}).get("speaking_timeout", 120.0),
            vad_silence_threshold=config_dict.get("state", {}).get("vad_silence_threshold", 1.0),
            debug=config_dict.get("state", {}).get("debug", False)
        )
        
        room_config = RoomConfig(
            url=config_dict.get("room", {}).get("url", "ws://localhost:7880"),
            api_key=config_dict.get("room", {}).get("api_key", "devkey"),
            api_secret=config_dict.get("room", {}).get("api_secret", "secret"),
            room_name=config_dict.get("room", {}).get("room_name", "lucidia_room"),
            sample_rate=config_dict.get("room", {}).get("sample_rate", 48000),
            channels=config_dict.get("room", {}).get("channels", 1),
            chunk_size=config_dict.get("room", {}).get("chunk_size", 480),
            buffer_size=config_dict.get("room", {}).get("buffer_size", 4800)
        )
        
        return cls(
            llm=llm_config,
            whisper=whisper_config,
            vosk=vosk_config,
            tts=tts_config,
            state=state_config,
            room=room_config,
            initial_greeting=config_dict.get("initial_greeting", "Hello! I'm Lucidia, your voice assistant. How can I help you today?")
        )