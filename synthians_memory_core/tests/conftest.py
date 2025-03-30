import os
import pytest
import pytest_asyncio
import asyncio
import aiohttp
import httpx  # Using httpx for health checks
import shutil
import tempfile
import time
from datetime import datetime

# Import the Memory Core client
from synthians_memory_core.api.client.client import SynthiansClient

# --- Configuration for variant integration tests ---
MC_URL = "http://localhost:5010"
NM_URL = "http://localhost:8001"
CCE_URL = "http://localhost:8002"

# --- Health Check Fixture (Function-Scoped) ---
@pytest_asyncio.fixture(autouse=False)  # Not auto-using by default to avoid affecting other tests
async def check_services_responsive(request):
    """
    Quickly check if core services are responsive before each test function.
    Uses httpx for simple async requests. Skips if test is marked 'skip_health_check'.
    """
    if "skip_health_check" in request.keywords:
        print("\nSkipping health check for this test.")
        yield
        return

    # Short timeout for quick check
    async with httpx.AsyncClient(timeout=3.0) as client:
        service_endpoints = {
            "Memory Core": f"{MC_URL}/health",
            "Neural Memory": f"{NM_URL}/health",
            "CCE": f"{CCE_URL}/"  # Basic root check for CCE
        }
        tasks = []
        for name, url in service_endpoints.items():
            tasks.append(client.get(url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (name, url), result in zip(service_endpoints.items(), results):
            if isinstance(result, Exception):
                pytest.fail(f"Health check failed: Service '{name}' at {url} unreachable. Error: {result}", pytrace=False)
            elif not result.is_success:
                pytest.fail(f"Health check failed: Service '{name}' at {url} returned status {result.status_code}", pytrace=False)
    yield  # Let the test run

# --- API Client Fixture (Function-Scoped) ---
@pytest_asyncio.fixture
async def api_clients():
    """
    Provides an aiohttp session and an initialized SynthiansClient (for MC).
    This fixture is used by variant integration tests to interact with the running Docker services.
    """
    # Provides aiohttp session for CCE/NM and dedicated client for MC
    async with aiohttp.ClientSession() as session, \
               SynthiansClient(base_url=MC_URL) as mc_client:
        yield session, mc_client  # Yield clients for the test function

# Helper function for variant tests to create test memories
async def create_test_memories(client, count=5, prefix="Test memory"):
    """Create a batch of test memories for testing."""
    memory_ids = []
    for i in range(count):
        content = f"{prefix} {i}"
        memory_id = f"test_variant_{os.environ.get('TITANS_VARIANT', 'UNKNOWN')}_{i}"
        
        # Create a test memory with random embedding
        embedding = [float(j) / 100 for j in range(384)]  # 384-dimensional embedding
        
        # Use the API to create the memory
        memory_entry = {
            "content": content,
            "embedding": embedding,
            "metadata": {
                "source": "test",
                "test_id": i,
                "test_batch": prefix,
                "variant": os.environ.get('TITANS_VARIANT', 'UNKNOWN')
            }
        }
        
        # Store the memory in the database
        await client.process_memory(memory_entry, memory_id)
        memory_ids.append(memory_id)
    
    return memory_ids

# --- Original fixtures for local testing ---
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
    config.addinivalue_line(
        "markers", "skip_health_check: skip the services health check"
    )
    config.addinivalue_line(
        "markers", "variant: mark test as a Titans variant test (MAC, MAG, MAL)"
    )
