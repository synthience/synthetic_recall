import asyncio
import inspect
import logging
from state.voice_state_manager import VoiceStateManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Print the method source directly without instantiating
print("Checking setup_tts_track implementation...")
setup_tts_track_source = inspect.getsource(VoiceStateManager.setup_tts_track)
print(f"setup_tts_track implementation:\n{setup_tts_track_source}")

# Check if the method includes the required parameters
if "sample_rate" in setup_tts_track_source and "num_channels" in setup_tts_track_source:
    print("✓ setup_tts_track includes required AudioSource parameters")
else:
    print("✗ setup_tts_track is missing required AudioSource parameters")

print("Test completed successfully")