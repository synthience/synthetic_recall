import os
import sys
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voice_core.stt.dolphining_stt_corrector import DolphiningSttCorrector
from voice_core.stt.dolphining_integration import DolphiningSTTIntegrator
from memory_core.enhanced_memory_client import EnhancedMemoryClient


@pytest.fixture
def mock_memory_client():
    """Create a mock EnhancedMemoryClient for testing."""
    memory_client = AsyncMock(spec=EnhancedMemoryClient)
    
    # Mock detect_emotional_context
    memory_client.detect_emotional_context = AsyncMock(return_value={
        "current_emotion": "neutral",
        "sentiment": 0.2,
        "confidence": 0.8
    })
    
    # Mock get_recent_entities
    memory_client.get_recent_conversation_entities = AsyncMock(return_value={
        "Lucidia": 0.9,
        "voice recognition": 0.85
    })
    
    # Mock get_conversation_context
    memory_client.get_conversation_context = AsyncMock(return_value={
        "topics": ["voice assistant", "speech recognition", "AI"],
        "sentiment": 0.3,
        "entities": {"Lucidia": 0.9, "voice recognition": 0.85}
    })
    
    # Mock narrative identity methods
    memory_client.narrative_identity_insight = AsyncMock(return_value={
        "insights": ["User values accuracy in communication"],
        "preferences": ["Prefers technical terms to be recognized correctly"]
    })
    
    memory_client.record_identity_experience = AsyncMock()
    
    return memory_client


class MockWebSocketClient:
    """Mock WebSocket client for testing."""
    
    def __init__(self, response_data):
        self.response_data = response_data
        self.connected = False
    
    async def __aenter__(self):
        self.connected = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
    
    async def send(self, data):
        pass
    
    async def receive(self):
        # Simulate receiving JSON response from WebSocket
        return self.response_data


@pytest.mark.asyncio
async def test_dolphining_phases(mock_memory_client):
    """Test all phases of the Dolphining correction framework."""
    # Setup domain dictionary with technical terms
    domain_dict = {
        "Lucidia": 0.9,
        "neural network": 0.85,
        "machine learning": 0.8,
        "artificial intelligence": 0.9,
        "NLP": 0.75,
        "STT": 0.8
    }
    
    # Create corrector
    corrector = DolphiningSttCorrector(
        memory_client=mock_memory_client,
        domain_dictionary=domain_dict,
        confidence_threshold=0.7
    )
    
    # Test with a transcript containing errors related to technical terms
    transcript = "lucydia uses neural networks and machine lorning for artificial inteligence"
    
    # Apply correction
    result = await corrector.correct_transcript(transcript)
    
    # Verify each phase worked correctly
    assert result["original"] == transcript
    assert result["corrected"] == "Lucidia uses neural networks and machine learning for artificial intelligence"
    assert result["changed"] is True
    assert result["confidence"] > 0.7  # Should have high confidence
    assert len(result["candidates"]) > 0  # Should have generated multiple candidates
    
    # Verify that memory context was used
    mock_memory_client.get_conversation_context.assert_called_once()
    
    # Test feedback mechanism
    corrector.feedback_correction(transcript, result["corrected"], True)
    
    # Verify statistics
    stats = corrector.get_correction_statistics()
    assert stats["total_corrections"] == 1
    assert stats["accepted_corrections"] == 1


@pytest.mark.asyncio
async def test_integrator_callbacks(mock_memory_client):
    """Test the DolphiningSTTIntegrator callback system."""
    # Create integrator
    integrator = DolphiningSTTIntegrator(
        memory_client=mock_memory_client,
        domain_dictionary={"Lucidia": 0.9}
    )
    
    # Create callback trackers
    correction_called = False
    clarification_called = False
    emotion_called = False
    
    # Define callbacks
    async def on_correction(data):
        nonlocal correction_called
        correction_called = True
        assert "original" in data
        assert "corrected" in data
    
    async def on_clarification(data):
        nonlocal clarification_called
        clarification_called = True
        assert "options" in data
    
    async def on_emotion(data):
        nonlocal emotion_called
        emotion_called = True
        assert "current_emotion" in data
    
    # Register callbacks
    integrator.register_callback("on_correction", on_correction)
    integrator.register_callback("on_clarification_needed", on_clarification)
    integrator.register_callback("on_emotion_detected", on_emotion)
    
    # Simulate transcription data
    transcription_data = {"text": "lucydia uses neural networks"}
    
    # Apply the Dolphining correction via the integrator
    with patch.object(DolphiningSttCorrector, "correct_transcript", new_callable=AsyncMock) as mock_correct:
        # Simulate a correction that changes the text
        mock_correct.return_value = {
            "original": "lucydia uses neural networks",
            "corrected": "Lucidia uses neural networks",
            "changed": True,
            "confidence": 0.8,
            "candidates": ["Lucidia uses neural networks", "Lucy uses neural networks"],
            "needs_clarification": False
        }
        
        await integrator._handle_nemo_transcription(transcription_data)
    
    # Verify callbacks were called
    assert correction_called is True
    assert clarification_called is False  # No clarification needed
    
    # Verify emotion detection was called
    mock_memory_client.detect_emotional_context.assert_called_once()
    assert emotion_called is True


@pytest.mark.asyncio
async def test_websocket_processing(mock_memory_client):
    """Test the WebSocket STT processing with Dolphining correction."""
    # Create corrector
    corrector = DolphiningSttCorrector(
        memory_client=mock_memory_client,
        domain_dictionary={"Lucidia": 0.9},
        confidence_threshold=0.7
    )
    
    # Mock WebSocket response
    mock_response = '{"text": "lucydia voice assistant", "confidence": 0.8}'
    
    # Patch the WebSocketClientProtocol to return our mock response
    with patch('voice_core.stt.dolphining_stt_corrector.websockets.connect', 
               new=lambda url: MockWebSocketClient(mock_response)):
        
        # Test processing with WebSocket STT
        audio_bytes = b'test audio data'
        stt_url = 'ws://test.example.com/stt'
        
        # Mock the _score_candidates method to ensure deterministic results
        with patch.object(corrector, '_score_candidates', new_callable=AsyncMock) as mock_score:
            mock_score.return_value = [
                {"text": "Lucidia voice assistant", "score": 0.9},
                {"text": "lucydia voice assistant", "score": 0.5}
            ]
            
            result = await corrector.process_with_websocket_stt(audio_bytes, stt_url)
    
    # Verify results
    assert result["original"] == "lucydia voice assistant"
    assert result["corrected"] == "Lucidia voice assistant"
    assert result["changed"] is True
    assert result["confidence"] > 0.7


@pytest.mark.asyncio
async def test_memory_integration(mock_memory_client):
    """Test integration with EnhancedMemoryClient, particularly narrative identity."""
    # Create corrector
    corrector = DolphiningSttCorrector(
        memory_client=mock_memory_client,
        domain_dictionary={"Lucidia": 0.9},
        confidence_threshold=0.7
    )
    
    # Test a transcript requiring emotional context
    transcript = "I'm feeling frustrated with the voice recognition"
    
    # Mock the _score_candidates method to ensure deterministic results
    with patch.object(corrector, '_score_candidates', new_callable=AsyncMock) as mock_score:
        mock_score.return_value = [
            {"text": "I'm feeling frustrated with the voice recognition", "score": 0.9}, 
            {"text": "I'm feeling frustrated with the Boys recognition", "score": 0.4}
        ]
        
        result = await corrector.correct_transcript(transcript)
    
    # Verify that emotional context detection was called
    mock_memory_client.detect_emotional_context.assert_called()
    
    # Verify that narrative identity insight was used
    mock_memory_client.narrative_identity_insight.assert_called()
    
    # Verify no change was made because the top candidate is the same as original
    assert result["changed"] is False
    assert result["original"] == transcript
    assert result["corrected"] == transcript
