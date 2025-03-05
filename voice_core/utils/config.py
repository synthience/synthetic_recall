class LucidiaConfig:
    def __init__(self):
        self.tts = {
            "voice": "en-US-AvaMultilingualNeural",
            "sample_rate": 48000,
            "num_channels": 1,
        }

class LLMConfig:
    def __init__(self):
        self.server_url = "http://localhost:1234/v1/chat/completions"
        self.model_name = "local-model"
        self.temperature = 0.7
        self.max_tokens = 300

class WhisperConfig:
    def __init__(self):
        self.model_name = "small"
        self.language = "en"
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.min_audio_length = 0.5
        self.max_audio_length = 3.0
