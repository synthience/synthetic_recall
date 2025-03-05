# tests/test_personal_details.py

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from memory_core.enhanced_memory_client import EnhancedMemoryClient


@pytest.fixture
def memory_client():
    """Create a memory client with mocked WebSocket connections."""
    with patch('memory_core.connectivity.websockets.connect') as mock_connect:
        # Mock the WebSocket connection
        mock_ws = MagicMock()
        mock_ws.__aenter__.return_value = mock_ws
        mock_ws.send.return_value = None
        mock_ws.recv.return_value = '{"status": "success"}'
        mock_connect.return_value = mock_ws
        
        # Create the client
        client = EnhancedMemoryClient(
            tensor_server_url="ws://localhost:8765",
            hpc_server_url="ws://localhost:8766",
            session_id="test_session",
            user_id="test_user"
        )
        
        yield client


@pytest.mark.asyncio
async def test_detect_and_store_personal_details(memory_client):
    """Test that personal details are correctly detected and stored."""
    # Test with a message containing a name
    text_with_name = "My name is John Doe and I live in New York."
    
    # Mock the store_memory method to avoid actual storage
    with patch.object(memory_client, 'store_memory', return_value=True) as mock_store:
        # Call the method
        result = await memory_client.detect_and_store_personal_details(text_with_name)
        
        # Check the result
        assert result is True
        
        # Verify personal details were detected
        assert "name" in memory_client.personal_details
        assert memory_client.personal_details["name"]["value"] == "John Doe"
        
        assert "location" in memory_client.personal_details
        assert memory_client.personal_details["location"]["value"] == "New York"
        
        # Verify store_memory was called with appropriate arguments
        mock_store.assert_any_call(
            content="User name: John Doe",
            significance=0.9,  # High significance for names
            metadata={
                "type": "personal_detail",
                "category": "name",
                "value": "John Doe"
            }
        )


@pytest.mark.asyncio
async def test_detect_and_store_personal_details_non_user(memory_client):
    """Test that personal details are not processed for non-user messages."""
    # Test with assistant message
    text = "My name is Assistant and I'm here to help."
    
    # Call the method with role="assistant"
    result = await memory_client.detect_and_store_personal_details(text, role="assistant")
    
    # Check that processing was skipped
    assert result is False
    
    # Verify no personal details were stored
    assert len(memory_client.personal_details) == 0


@pytest.mark.asyncio
async def test_detect_and_store_personal_details_no_details(memory_client):
    """Test behavior when no personal details are found."""
    # Test with message containing no personal details
    text = "The weather is nice today."
    
    # Call the method
    result = await memory_client.detect_and_store_personal_details(text)
    
    # Check that no details were found
    assert result is False
    
    # Verify no personal details were stored
    assert len(memory_client.personal_details) == 0


@pytest.mark.asyncio
async def test_detect_and_store_personal_details_lucidia_filter(memory_client):
    """Test that 'Lucidia' is not detected as a name."""
    # Test with message containing 'Lucidia'
    text = "Lucidia, what's the weather today?"
    
    # Call the method
    result = await memory_client.detect_and_store_personal_details(text)
    
    # Check that no details were found (Lucidia should be filtered)
    assert result is False
    
    # Verify no personal details were stored
    assert len(memory_client.personal_details) == 0


@pytest.mark.asyncio
async def test_process_message_integration(memory_client):
    """Test integration with process_message method."""
    # Mock the methods that would be called
    with patch.object(memory_client, 'detect_and_store_personal_details') as mock_detect, \
         patch.object(memory_client, 'analyze_emotions') as mock_emotions, \
         patch.object(memory_client, 'store_memory') as mock_store:
        
        # Set up return values
        mock_detect.return_value = True
        mock_emotions.return_value = {"emotion": "happy", "confidence": 0.8}
        mock_store.return_value = True
        
        # Call process_message
        await memory_client.process_message("My name is Jane and I'm feeling great!")
        
        # Verify all methods were called with correct arguments
        mock_detect.assert_called_once_with("My name is Jane and I'm feeling great!", "user")
        mock_emotions.assert_called_once_with("My name is Jane and I'm feeling great!")
        mock_store.assert_called_once_with(
            content="My name is Jane and I'm feeling great!",
            metadata={"role": "user", "type": "message"}
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main(['-xvs', __file__])
