import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from ...geometry_manager import GeometryManager
from ..http_server import app
from ..neural_memory import NeuralMemoryModule


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_neural_memory():
    """Create a mock NeuralMemoryModule instance."""
    with patch('synthians_memory_core.synthians_trainer_server.http_server.get_neural_memory', autospec=True) as mock_get:
        mock_instance = MagicMock(spec=NeuralMemoryModule)
        mock_get.return_value = mock_instance
        
        # Configure mocked methods for new Neural Memory API
        mock_instance.retrieve.return_value = np.random.randn(768)
        mock_instance.update_memory.return_value = (0.1, 0.2)  # loss, grad_norm
        
        yield mock_instance


def test_health_endpoint(test_client):
    """Test that the health endpoint returns a 200 status code."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_retrieve_endpoint(test_client, mock_neural_memory):
    """Test the retrieve endpoint of the Neural Memory API."""
    # Prepare request data
    input_embedding = np.random.randn(768).tolist()
    
    request_data = {
        "input_embedding": input_embedding
    }
    
    # Make the request
    response = test_client.post("/retrieve", json=request_data)
    
    # Verify the response
    assert response.status_code == 200
    result = response.json()
    assert "retrieved_embedding" in result
    assert len(result["retrieved_embedding"]) == 768
    
    # Verify the mock was called correctly
    mock_neural_memory.retrieve.assert_called_once()
    # First positional arg should be the input embedding (as numpy array)
    call_args = mock_neural_memory.retrieve.call_args[0]
    assert len(call_args) >= 1
    np.testing.assert_array_almost_equal(call_args[0], input_embedding)


def test_update_memory_endpoint(test_client, mock_neural_memory):
    """Test the update_memory endpoint of the Neural Memory API."""
    # Prepare request data
    input_embedding = np.random.randn(768).tolist()
    
    request_data = {
        "input_embedding": input_embedding
    }
    
    # Make the request
    response = test_client.post("/update_memory", json=request_data)
    
    # Verify the response
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert result["status"] == "success"
    assert "loss" in result
    assert "grad_norm" in result
    
    # Verify the mock was called correctly
    mock_neural_memory.update_memory.assert_called_once()
    # First positional arg should be the input embedding (as numpy array)
    call_args = mock_neural_memory.update_memory.call_args[0]
    assert len(call_args) >= 1
    np.testing.assert_array_almost_equal(call_args[0], input_embedding)