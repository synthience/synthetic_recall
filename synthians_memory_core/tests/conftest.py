import os
import pytest
import asyncio
import shutil
import tempfile
import time
from datetime import datetime

# Configure pytest-asyncio
# Custom event_loop fixture removed to avoid deprecation warnings

# Define common test fixtures
@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test data that's removed after tests finish."""
    test_dir = tempfile.mkdtemp(prefix="synthians_test_")
    print(f"\nCreated temporary test directory: {test_dir}")
    yield test_dir
    # Clean up after tests
    # Add retry logic for Windows file locking issues
    attempts = 3
    while attempts > 0:
        try:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=False)
                print(f"Removed temporary test directory: {test_dir}")
            else:
                print(f"Temporary test directory already removed: {test_dir}")
            break # Success
        except OSError as e:
            print(f"Warning: Error removing temp directory (attempt {4-attempts}): {e}")
            attempts -= 1
            if attempts == 0:
                print(f"ERROR: Failed to remove temp directory {test_dir} after multiple attempts.")
            else:
                time.sleep(0.5) # Wait before retrying

@pytest.fixture(scope="session")
def test_server_url():
    """Return the URL of the test server."""
    # Default to localhost:5010, but allow override through environment variable
    return os.environ.get("SYNTHIANS_TEST_URL", "http://localhost:5010")

# Configure markers for test categories
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
    """Create a batch of test memories for testing."""
    memory_ids = []
    for i in range(count):
        content = f"{prefix} {i}"
        memory_id = f"test_memory_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
        
        # Create a test memory with random embedding
        embedding = [float(j) / 100 for j in range(384)]  # 384-dimensional embedding
        
        # Use the API to create the memory
        memory_entry = {
            "content": content,
            "embedding": embedding,
            "metadata": {
                "source": "test",
                "test_id": i,
                "test_batch": prefix
            }
        }
        
        # Store the memory in the database
        await client.process_memory(memory_entry, memory_id)
        memory_ids.append(memory_id)
    
    return memory_ids
