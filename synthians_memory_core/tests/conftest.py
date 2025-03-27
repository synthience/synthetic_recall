import os
import pytest
import asyncio
import shutil
import tempfile
from datetime import datetime

# Define common test fixtures

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test data that's removed after tests finish."""
    test_dir = tempfile.mkdtemp(prefix="synthians_test_")
    print(f"Created temporary test directory: {test_dir}")
    yield test_dir
    # Clean up after tests
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"Removed temporary test directory: {test_dir}")

@pytest.fixture(scope="session")
def test_server_url():
    """Return the URL of the test server."""
    # Default to localhost:5010, but allow override through environment variable
    return os.environ.get("SYNTHIANS_TEST_URL", "http://localhost:5010")

# Add markers for categorizing tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test (basic functionality)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow test"
    )
    config.addinivalue_line(
        "markers", "emotion: mark test as testing emotion analysis"
    )
    config.addinivalue_line(
        "markers", "retrieval: mark test as testing memory retrieval"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test"
    )

# Helper functions for test utilities

async def create_test_memories(client, count=5, prefix="Test memory"):
    """Create a batch of test memories and return their IDs."""
    memory_ids = []
    timestamp = datetime.now().isoformat()
    
    for i in range(count):
        content = f"{prefix} {i+1} created at {timestamp}"
        metadata = {"test_batch": timestamp, "index": i}
        
        response = await client.process_memory(content=content, metadata=metadata)
        if response.get("success"):
            memory_ids.append(response.get("memory_id"))
    
    return memory_ids, timestamp
