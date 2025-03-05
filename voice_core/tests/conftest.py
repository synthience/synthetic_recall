import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for all tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture(autouse=True)
def mock_cuda_available():
    """Force CPU usage for consistent testing"""
    import torch
    torch.cuda.is_available = lambda: False
