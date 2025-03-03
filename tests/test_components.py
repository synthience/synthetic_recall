"""Test individual components of the voice pipeline."""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from voice_core.stt import WhisperSTTService
from voice_core.tts import EdgeTTSTTS
from voice_core.utils.config import WhisperConfig
from livekit import rtc

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_whisper():
    """Test Whisper STT initialization."""
    logger.info("Testing Whisper STT...")
    config = WhisperConfig(model_name="base", device="cpu")
    stt = WhisperSTTService(config=config)
    
    try:
        await asyncio.wait_for(stt.initialize(), timeout=30.0)
        logger.info("✅ Whisper test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Whisper test failed: {e}")
        return False

async def test_edge_tts():
    """Test Edge TTS."""
    logger.info("Testing Edge TTS...")
    tts = EdgeTTSTTS()
    
    try:
        await asyncio.wait_for(
            tts.speak("This is a test message."),
            timeout=10.0
        )
        logger.info("✅ Edge TTS test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Edge TTS test failed: {e}")
        return False

async def test_livekit():
    """Test LiveKit connection."""
    logger.info("Testing LiveKit connection...")
    try:
        room = await rtc.Room.connect(
            "ws://localhost:7880",
            "test-token",
            connect_options=rtc.ConnectOptions(auto_subscribe=True)
        )
        logger.info(f"Connected to room: {room.name}")
        await asyncio.sleep(2)  # Wait for connection to stabilize
        await room.disconnect()
        logger.info("✅ LiveKit test passed")
        return True
    except Exception as e:
        logger.error(f"❌ LiveKit test failed: {e}")
        return False

async def main():
    """Run all component tests."""
    logger.info("Starting component tests...")
    
    # Test each component
    whisper_ok = await test_whisper()
    edge_tts_ok = await test_edge_tts()
    livekit_ok = await test_livekit()
    
    # Report results
    logger.info("\nTest Results:")
    logger.info(f"Whisper STT: {'✅' if whisper_ok else '❌'}")
    logger.info(f"Edge TTS: {'✅' if edge_tts_ok else '❌'}")
    logger.info(f"LiveKit: {'✅' if livekit_ok else '❌'}")
    
    if not all([whisper_ok, edge_tts_ok, livekit_ok]):
        logger.error("Some tests failed!")
        sys.exit(1)
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    # Use Windows-compatible event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Tests failed with error: {e}")
        sys.exit(1)
