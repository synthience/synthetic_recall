import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from ...geometry_manager import GeometryManager
from ..http_server import app, SynthiansTrainer
from ..models import PredictNextEmbeddingRequest, TrainSequenceRequest


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_trainer():
    """Create a mock SynthiansTrainer instance."""
    with patch('synthians_memory_core.synthians_trainer_server.http_server.SynthiansTrainer', autospec=True) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        # Configure mocked methods
        mock_instance.predict_next.return_value = np.random.randn(768)
        mock_instance.train_sequence.return_value = True
        yield mock_instance


def test_health_endpoint(test_client):
    """Test that the health endpoint returns a 200 status code."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_next_embedding_stateless(test_client, mock_trainer):
    """Test the predict_next_embedding endpoint with explicit previous state."""
    # Prepare request data
    embeddings = [np.random.randn(768).tolist() for _ in range(3)]
    previous_memory_state = {
        "sequence": [np.random.randn(768).tolist() for _ in range(2)],
        "surprise_history": [0.1, 0.2],
        "momentum": np.random.randn(768).tolist()
    }
    
    request_data = {
        "embeddings": embeddings,
        "previous_memory_state": previous_memory_state
    }
    
    # Send request
    response = test_client.post("/predict_next_embedding", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert "predicted_embedding" in response.json()
    assert "memory_state" in response.json()
    assert "surprise_score" in response.json()
    
    # Verify trainer was called correctly
    mock_trainer.predict_next.assert_called_once()
    args, _ = mock_trainer.predict_next.call_args
    assert len(args) >= 2  # At least embeddings and previous state
    

def test_train_sequence(test_client, mock_trainer):
    """Test the train_sequence endpoint."""
    # Prepare request data
    embeddings = [np.random.randn(768).tolist() for _ in range(5)]
    
    request_data = {
        "embeddings": embeddings,
    }
    
    # Send request
    response = test_client.post("/train_sequence", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Verify trainer was called correctly
    mock_trainer.train_sequence.assert_called_once()
    args, _ = mock_trainer.train_sequence.call_args
    assert len(args[0]) == 5  # Embeddings length


def test_predict_next_embedding_errors(test_client):
    """Test error handling in predict_next_embedding endpoint."""
    # Test with empty embeddings
    request_data = {
        "embeddings": []
    }
    response = test_client.post("/predict_next_embedding", json=request_data)
    assert response.status_code == 400
    
    # Test with malformed embeddings (wrong dimension)
    request_data = {
        "embeddings": [np.random.randn(10).tolist() for _ in range(3)]
    }
    response = test_client.post("/predict_next_embedding", json=request_data)
    assert response.status_code == 400