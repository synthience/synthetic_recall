import asyncio
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
import livekit.rtc as rtc

class MockTrackPublication:
    def __init__(self, sid="mock-track-sid", kind=rtc.TrackKind.KIND_AUDIO):
        self.sid = sid
        self.kind = kind

class MockLocalParticipant:
    def __init__(self, identity="mock-participant"):
        self.identity = identity
        self.track_publications = {
            "track1": MockTrackPublication()
        }
        self.publish_transcription = AsyncMock()
        self.publish_data = AsyncMock(return_value=True)

class MockRoom:
    def __init__(self, participant_identity="mock-participant"):
        self.local_participant = MockLocalParticipant(identity=participant_identity)
        self.connection_state = rtc.ConnectionState.CONN_CONNECTED

async def test_transcript_identity():
    """Test transcript identity attribution with various scenarios"""
    logger = logging.getLogger("test_transcript_identity")
    logger.info("Creating VoiceStateManager")
    
    # Create the voice state manager
    mgr = VoiceStateManager(debug=True)
    
    # Set up mock room with mock local participant
    mock_room = MockRoom(participant_identity="test-room-participant")
    mgr._room = mock_room
    
    # Test 1: With explicit participant identity
    logger.info("Test 1: Explicit participant identity")
    await mgr.publish_transcription(
        "Test transcript with explicit identity", 
        "user", 
        True, 
        participant_identity="test-user-explicit"
    )
    
    # Verify explicit identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Transcription API - participant_identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-user-explicit", f"Expected 'test-user-explicit', got '{trans.participant_identity}'"
    
    # Verify explicit identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel - participant_identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-user-explicit", f"Expected 'test-user-explicit', got '{data_json['participant_identity']}'"
    
    # Reset mocks
    mock_room.local_participant.publish_transcription.reset_mock()
    mock_room.local_participant.publish_data.reset_mock()
    
    # Test 2: Without explicit participant identity (should use local participant identity)
    logger.info("Test 2: Implicit identity from local participant")
    await mgr.publish_transcription(
        "Test transcript without explicit identity", 
        "user", 
        True
    )
    
    # Verify local participant identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Transcription API - implicit identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-room-participant", f"Expected 'test-room-participant', got '{trans.participant_identity}'"
    
    # Verify local participant identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel - implicit identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-room-participant", f"Expected 'test-room-participant', got '{data_json['participant_identity']}'"
    
    # Reset mocks
    mock_room.local_participant.publish_transcription.reset_mock()
    mock_room.local_participant.publish_data.reset_mock()
    
    # Test 3: Different sender types (assistant vs user)
    logger.info("Test 3: Different sender types (assistant)")
    await mgr.publish_transcription(
        "Test transcript from assistant", 
        "assistant", 
        True, 
        participant_identity="test-assistant"
    )
    
    # Verify assistant identity was used (Transcription API)
    trans_api_args, _ = mock_room.local_participant.publish_transcription.call_args_list[0]
    trans = trans_api_args[0]
    logger.info(f"Assistant transcript - identity: {trans.participant_identity}")
    assert trans.participant_identity == "test-assistant", f"Expected 'test-assistant', got '{trans.participant_identity}'"
    
    # Verify assistant identity was used (Data Channel)
    data_channel_args, _ = mock_room.local_participant.publish_data.call_args_list[0]
    data_json = json.loads(data_channel_args[0].decode())
    logger.info(f"Data Channel (assistant) - identity: {data_json['participant_identity']}")
    assert data_json['participant_identity'] == "test-assistant", f"Expected 'test-assistant', got '{data_json['participant_identity']}'"
    assert data_json['sender'] == "assistant", f"Expected sender 'assistant', got '{data_json['sender']}'"
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_transcript_identity())
