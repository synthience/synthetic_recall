# synthians_memory_core/orchestrator/tests/test_context_cascade_engine.py

import pytest
import numpy as np
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any

from ..context_cascade_engine import ContextCascadeEngine
from synthians_memory_core.geometry_manager import GeometryManager


@pytest.fixture
def geometry_manager():
    """Test fixture for GeometryManager."""
    return GeometryManager({
        'embedding_dim': 768,
        'geometry_type': 'euclidean',
    })


@pytest.fixture
def engine(geometry_manager):
    """Test fixture for ContextCascadeEngine with mock URLs."""
    return ContextCascadeEngine(
        memory_core_url="http://memory-core-test",
        trainer_url="http://trainer-test",
        geometry_manager=geometry_manager
    )


@pytest.fixture
def mock_response():
    """Create a mock for aiohttp ClientResponse."""
    mock = MagicMock()
    mock.status = 200
    mock.json = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_process_new_memory(engine, mock_response):
    """Test the complete flow of processing a new memory."""
    # Mock embeddings and memory data
    test_content = "This is a test memory"
    test_embedding = np.random.randn(768).tolist()
    test_memory_id = "test-memory-123"
    
    # Mock memory core response
    memory_response = {
        "id": test_memory_id,
        "embedding": test_embedding,
        "quickrecal_score": 0.8
    }
    mock_response.json.return_value = memory_response
    
    # Mock trainer response
    trainer_response = {
        "predicted_embedding": np.random.randn(768).tolist(),
        "surprise_score": 0.3,
        "memory_state": {
            "sequence": [test_embedding],
            "surprise_history": [0.3],
            "momentum": np.random.randn(768).tolist()
        }
    }
    mock_trainer_response = MagicMock()
    mock_trainer_response.status = 200
    mock_trainer_response.json = AsyncMock(return_value=trainer_response)
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post, \
         patch('aiohttp.ClientSession.get') as mock_get:
            
        # Configure mock to return different responses for different URLs
        mock_post.side_effect = lambda url, **kwargs: \
            mock_response if "memory-core-test" in url else mock_trainer_response
        
        # Call the method under test
        result = await engine.process_new_memory(
            content=test_content,
            embedding=test_embedding
        )
        
        # Verify memory core was called
        assert mock_post.call_count >= 1
        # Verify memory_id is present in result
        assert result["memory_id"] == test_memory_id
        # Verify prediction data is present
        assert "prediction" in result
        # Verify last_predicted_embedding was updated
        assert engine.last_predicted_embedding is not None


@pytest.mark.asyncio
async def test_retrieve_memories(engine, mock_response):
    """Test retrieving memories through the engine."""
    # Mock query and response
    query = "test query"
    memories = [
        {"id": "mem1", "content": "Memory 1", "similarity": 0.9},
        {"id": "mem2", "content": "Memory 2", "similarity": 0.8}
    ]
    
    mock_response.json.return_value = {"memories": memories}
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Call the method under test
        result = await engine.retrieve_memories(query=query, limit=2)
        
        # Verify memory core was called
        mock_post.assert_called_once()
        # Verify results
        assert len(result["memories"]) == 2
        assert result["memories"][0]["id"] == "mem1"


@pytest.mark.asyncio
async def test_error_handling(engine):
    """Test error handling for HTTP responses."""
    # Mock error response
    error_response = MagicMock()
    error_response.status = 500
    error_response.text = AsyncMock(return_value="Internal server error")
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value = error_response
        
        # Call the method under test and expect error handling
        result = await engine.process_new_memory(content="Error test")
        
        # Verify error is captured
        assert "error" in result
        assert result["status"] == "error"


@pytest.mark.asyncio
async def test_surprise_detection(engine, mock_response):
    """Test surprise detection when actual embedding differs from predicted."""
    # Setup initial state with a predicted embedding
    engine.last_predicted_embedding = np.random.randn(768).tolist()
    
    # Create actual embedding with high difference
    actual_embedding = np.random.randn(768).tolist()  # Will be different due to randomness
    
    # Mock memory core response
    memory_response = {
        "id": "test-memory-456",
        "embedding": actual_embedding,
        "quickrecal_score": 0.7
    }
    mock_response.json.return_value = memory_response
    
    # Mock trainer response with high surprise
    trainer_response = {
        "predicted_embedding": np.random.randn(768).tolist(),
        "surprise_score": 0.8,  # High surprise
        "memory_state": {
            "sequence": [actual_embedding],
            "surprise_history": [0.8],
            "momentum": np.random.randn(768).tolist()
        }
    }
    mock_trainer_response = MagicMock()
    mock_trainer_response.status = 200
    mock_trainer_response.json = AsyncMock(return_value=trainer_response)
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        # Configure mock to return different responses for different URLs
        mock_post.side_effect = lambda url, **kwargs: \
            mock_response if "memory-core-test" in url else mock_trainer_response
        
        # Call the method under test
        result = await engine.process_new_memory(
            content="Surprise test",
            embedding=actual_embedding
        )
        
        # Verify surprise was detected
        assert "surprise" in result
        assert result["surprise"]["score"] > 0.7  # High surprise threshold
