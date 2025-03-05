"""
Test script to verify fixes for voice assistant pipeline issues.
Tests the Transcription API usage, multi-turn flow, and task management.
"""

import asyncio
import unittest
import uuid
import json
import time
from unittest.mock import MagicMock, patch, AsyncMock

import livekit.rtc as rtc
from voice_core.state.voice_state_manager import VoiceStateManager, VoiceState
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.agent2 import LucidiaVoiceAgent


class TestVoiceFixesIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for voice assistant pipeline fixes."""
    
    async def asyncSetUp(self):
        """Set up test environment with mocked components."""
        # Create mocked room and participant
        self.room = MagicMock()
        self.room.connection_state = rtc.ConnectionState.CONN_CONNECTED
        self.local_participant = MagicMock()
        self.room.local_participant = self.local_participant
        self.local_participant.identity = "test-user"
        
        # Mock track publications
        self.track_pub = MagicMock()
        self.track_pub.kind = rtc.TrackKind.KIND_AUDIO
        self.track_pub.sid = "test-track-sid"
        self.local_participant.track_publications = {"audio": self.track_pub}
        
        # Set up publish methods as AsyncMocks
        self.local_participant.publish_data = AsyncMock()
        self.local_participant.publish_transcription = AsyncMock()
        
        # Create state manager
        self.state_manager = VoiceStateManager()
        await self.state_manager.set_room(self.room)
        
        # Create job context mock
        self.job_context = MagicMock()
        self.job_context.room = self.room
        
        # Create agent with mocked LLM service
        self.agent = LucidiaVoiceAgent(self.job_context, "Hello, I'm a test assistant.")
        self.agent.llm_service = MagicMock()
        self.agent.llm_service.generate_response = AsyncMock(return_value="This is a test response")
        self.agent.llm_service.initialize = AsyncMock()
        
        # Replace agent's state manager with our test one
        self.agent.state_manager = self.state_manager
        
        # Create mocked STT and TTS services
        self.agent.stt_service = MagicMock()
        self.agent.stt_service.initialize = AsyncMock()
        self.agent.stt_service.clear_buffer = AsyncMock()
        self.agent.stt_service.set_room = MagicMock()
        
        self.agent.tts_service = MagicMock()
        self.agent.tts_service.initialize = AsyncMock()
        self.agent.tts_service.set_room = AsyncMock()
        self.agent.tts_service.speak = AsyncMock()
        
        # Initialize agent
        await self.agent.initialize()
    
    async def test_transcription_api_usage(self):
        """Test that Transcription API is used correctly with segments."""
        # Simulate transcript handling
        text = "Test transcript"
        await self.state_manager.handle_stt_transcript(text)
        
        # Verify Transcription API was called with correct structure
        self.local_participant.publish_transcription.assert_called_once()
        call_args = self.local_participant.publish_transcription.call_args[0][0]
        
        # Check Transcription object structure
        self.assertEqual(call_args.participant_identity, "test-user")
        self.assertEqual(call_args.track_sid, "test-track-sid")
        self.assertEqual(len(call_args.segments), 1)
        self.assertEqual(call_args.segments[0].text, "Test transcript")
        self.assertTrue(call_args.segments[0].final)
        self.assertEqual(call_args.segments[0].language, "en")
        
        # Verify data channel was also used with sequence number
        self.local_participant.publish_data.assert_called()
        data_calls = [call for call in self.local_participant.publish_data.call_args_list 
                     if b'"type":"transcript"' in call[0][0]]
        self.assertTrue(len(data_calls) > 0)
        
        # Check sequence number is included
        data = json.loads(data_calls[0][0][0].decode())
        self.assertIn("sequence", data)
        self.assertEqual(data["sender"], "user")
    
    async def test_multi_turn_flow(self):
        """Test that multi-turn conversation flow works without hanging."""
        # Start the agent
        await self.agent.start()
        
        # Verify greeting was sent
        self.agent.tts_service.speak.assert_called_once_with("Hello, I'm a test assistant.")
        self.agent.tts_service.speak.reset_mock()
        
        # Simulate first user input
        await self.agent._handle_transcript("First test input")
        
        # Verify LLM was called and response was spoken
        self.agent.llm_service.generate_response.assert_called_once_with("First test input")
        self.agent.tts_service.speak.assert_called_once_with("This is a test response")
        self.agent.stt_service.clear_buffer.assert_called()
        
        # Reset mocks for second turn
        self.agent.llm_service.generate_response.reset_mock()
        self.agent.tts_service.speak.reset_mock()
        self.agent.stt_service.clear_buffer.reset_mock()
        
        # Simulate second user input
        await self.agent._handle_transcript("Second test input")
        
        # Verify second turn was processed
        self.agent.llm_service.generate_response.assert_called_once_with("Second test input")
        self.agent.tts_service.speak.assert_called_once_with("This is a test response")
        self.agent.stt_service.clear_buffer.assert_called()
        
        # Verify state is back to LISTENING
        self.assertEqual(self.state_manager.current_state, VoiceState.LISTENING)
    
    async def test_llm_timeout_handling(self):
        """Test that LLM timeouts are handled properly."""
        # Mock LLM to simulate timeout
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow response
            return "Slow response"
            
        self.agent.llm_service.generate_response = AsyncMock(side_effect=slow_response)
        
        # Patch asyncio.wait_for to simulate timeout
        original_wait_for = asyncio.wait_for
        
        async def mock_wait_for(coro, timeout):
            if "generate_response" in str(coro):
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout)
            
        with patch('asyncio.wait_for', side_effect=mock_wait_for):
            # Simulate user input
            await self.agent._handle_transcript("Test input")
            
            # Verify error message was spoken
            self.agent.tts_service.speak.assert_called_once()
            call_args = self.agent.tts_service.speak.call_args[0][0]
            self.assertIn("took too long", call_args)
            
            # Verify state is back to LISTENING
            self.assertEqual(self.state_manager.current_state, VoiceState.LISTENING)
    
    async def test_tts_interruption(self):
        """Test that TTS interruption works correctly."""
        # Start speaking
        tts_task = asyncio.create_task(asyncio.sleep(1))  # Mock TTS task
        await self.state_manager.start_speaking(tts_task)
        
        # Verify state is SPEAKING
        self.assertEqual(self.state_manager.current_state, VoiceState.SPEAKING)
        
        # Simulate interruption
        await self.state_manager.handle_user_speech_detected("Interrupt text")
        
        # Verify state transitions
        self.assertEqual(self.state_manager.current_state, VoiceState.PROCESSING)
        
        # Clean up
        if not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    unittest.main()