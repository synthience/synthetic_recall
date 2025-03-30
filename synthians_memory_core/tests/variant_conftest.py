# tests/variant_conftest.py

import pytest
import pytest_asyncio
import asyncio
import aiohttp
import os
import httpx # Using httpx for simpler async requests in checks

# Assuming SynthiansClient is available and targets the Memory Core API (e.g., port 5010)
from synthians_memory_core.api.client.client import SynthiansClient

MC_URL = "http://localhost:5010"
NM_URL = "http://localhost:8001"
CCE_URL = "http://localhost:8002"

# Fixture to provide API clients to tests
@pytest_asyncio.fixture
async def api_clients():
    # Provides aiohttp session for CCE/NM and dedicated client for MC
    async with aiohttp.ClientSession() as session, \
               SynthiansClient(base_url=MC_URL) as mc_client:
        # Optional: Add a quick health check *here* before yielding if desired
        # await check_services_quick(session)
        yield session, mc_client

# Optional: Quick check before each test function
@pytest_asyncio.fixture(autouse=True)
async def check_services_quick_fixture():
    """Quickly check if services seem responsive before each test"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp_mc = await client.get(f"{MC_URL}/health")
            resp_nm = await client.get(f"{NM_URL}/health")
            resp_cce = await client.get(f"{CCE_URL}/") # Basic check
            resp_mc.raise_for_status()
            resp_nm.raise_for_status()
            resp_cce.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            pytest.fail(f"Service unresponsive before test: {e}")
        except Exception as e:
             pytest.fail(f"Unexpected error checking services: {e}")

# Keep other useful fixtures for variant tests
# Helper functions for test utilities
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
