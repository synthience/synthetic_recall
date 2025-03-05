# Lucidia Voice Assistant

A powerful voice assistant with real-time speech-to-text, text-to-speech, and natural language understanding capabilities.

## Project Structure

```
project_root/
├── voice_core/           # Main source code
│   ├── config/          # Configuration management
│   ├── stt/            # Speech-to-text services
│   ├── tts/            # Text-to-speech services
│   ├── llm/            # Language model integration
│   ├── livekit/        # LiveKit integration
│   ├── audio/          # Audio processing utilities
│   ├── conversation/   # Conversation management
│   ├── pipeline/       # Voice pipeline components
│   ├── handlers/       # Message handlers
│   ├── custom/         # Custom components
│   └── utils/          # Shared utilities
├── scripts/            # Utility scripts
├── tests/             # Test cases
└── docs/              # Documentation
```

## Key Features

- Real-time speech recognition with multiple STT backends (Vosk, Whisper)
- High-quality text-to-speech using Edge TTS
- LiveKit integration for real-time audio streaming
- Configurable audio pipeline with 16kHz/48kHz sample rate conversion
- Robust conversation management and memory system
- Extensible architecture for custom components

## Technical Specifications

- Input Sample Rate: 16kHz
- Output Sample Rate: 48kHz (LiveKit requirement)
- Audio Format: Float32 [-1, 1] normalized
- Frame Size: 1024 samples
- Voice: en-US-AvaMultilingualNeural (default)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
LIVEKIT_SAMPLE_RATE=48000
LIVEKIT_CHANNELS=1
FRAME_SIZE=1024
VOICE_NAME=en-US-AvaMultilingualNeural
STT_SAMPLE_RATE=16000
VAD_THRESHOLD=-40.0
VAD_FRAME_DURATION=0.03
```

3. Run the voice assistant:
```bash
python main.py
```

## Development

- Follow the modular architecture when adding new features
- Maintain backward compatibility with existing components
- Add tests for new functionality
- Document changes in the appropriate docs/ files

## License

[Your License Here]
