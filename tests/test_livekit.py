"""Test LiveKit connection independently."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from livekit import rtc
from voice_core.livekit_integration.livekit_service import generate_token

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LIVEKIT_URL = "ws://localhost:7880"

async def test_livekit_connection():
    """Test basic LiveKit connection."""
    logger.info(f"Testing LiveKit connection to {LIVEKIT_URL}")
    
    try:
        # Create room and connect
        room = rtc.Room()
        token = generate_token("playground")
        
        logger.info("Connecting to room...")
        await asyncio.wait_for(
            room.connect(LIVEKIT_URL, token),
            timeout=10.0
        )
        
        logger.info(f"Connected successfully. Room state: {room.connection_state}")
        logger.info(f"Local participant: {room.local_participant.identity}")
        
        # Wait a bit to ensure connection is stable
        logger.info("Waiting to verify connection stability...")
        await asyncio.sleep(5)
        
        # Check final state
        if room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            logger.error(f"Room not in connected state: {room.connection_state}")
            return False
            
        logger.info("Connection test passed!")
        return True
        
    except asyncio.TimeoutError:
        logger.error("Connection timed out")
        return False
    except Exception as e:
        logger.error(f"Connection failed: {e}", exc_info=True)
        return False
    finally:
        if 'room' in locals():
            await room.disconnect()
            logger.info("Disconnected from room")

async def main():
    """Run the test."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    success = await test_livekit_connection()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)
