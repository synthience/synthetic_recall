# synthians_memory_core/tests/test_assembly_sync.py

import os
import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta

from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.memory_structures import MemoryEntry, MemoryAssembly
from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.custom_logger import logger

# Constants for testing
EMBEDDING_DIM = 256
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')

# Create test directory if it doesn't exist
os.makedirs(TEST_DIR, exist_ok=True)

# Utility functions
def clear_test_directory():
    """Remove test directory and recreate it"""
    import shutil
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)

def create_random_embedding(dim=EMBEDDING_DIM):
    """Create a random normalized embedding"""
    vector = np.random.random(dim).astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def create_test_memory(idx, gm):
    """Create a test memory with random embedding"""
    embedding = create_random_embedding()
    memory = MemoryEntry(
        content=f"Test memory {idx}",
        embedding=embedding,
        metadata={"test": True, "idx": idx}
    )
    return memory

def create_test_assembly(idx, gm, memories=None, with_timestamp=True):
    """Create a test assembly with provided memories"""
    assembly = MemoryAssembly(
        assembly_id=f"test_assembly_{idx}",
        name=f"Test Assembly {idx}",
        geometry_manager=gm
    )
    
    if memories:
        for memory in memories:
            assembly.add_memory(memory)
    
    # Set vector_index_updated_at if requested
    if with_timestamp:
        assembly.vector_index_updated_at = datetime.now(timezone.utc)
    
    return assembly

@pytest.mark.asyncio
async def test_activate_assemblies_filter():
    """Test that _activate_assemblies correctly filters unsynchronized assemblies."""
    clear_test_directory()
    
    # Initialize a test memory core
    core = SynthiansMemoryCore({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'memory_core'),
        'assembly_threshold': 0.7  # Set threshold for testing
    })
    await core.initialize()
    
    # Create a geometry manager for test assemblies
    gm = GeometryManager()
    
    # Create test memories
    test_memories = [create_test_memory(i, gm) for i in range(10)]
    
    # Create two assemblies - one synchronized, one not
    synced_assembly = create_test_assembly(1, gm, test_memories[:5], with_timestamp=True)
    unsynced_assembly = create_test_assembly(2, gm, test_memories[5:], with_timestamp=False)
    
    # Add assemblies to memory core
    async with core._lock:
        core.assemblies[synced_assembly.assembly_id] = synced_assembly
        core.assemblies[unsynced_assembly.assembly_id] = unsynced_assembly
    
    # Create a test query embedding
    query_embedding = create_random_embedding()
    
    # Force both assemblies to have high similarity for testing
    synced_assembly.get_similarity = lambda x: 0.9  # Mock to return high similarity
    unsynced_assembly.get_similarity = lambda x: 0.9  # Mock to return high similarity
    
    # Call _activate_assemblies with the test query
    activated = await core._activate_assemblies(query_embedding)
    
    # Verify only the synchronized assembly is activated
    assert len(activated) == 1, "Only the synchronized assembly should be activated"
    assert activated[0][0].assembly_id == synced_assembly.assembly_id, "The activated assembly should be the synchronized one"
    assert unsynced_assembly.assembly_id not in [a[0].assembly_id for a in activated], "Unsynchronized assembly should not be activated"
    
    # Clean up
    await core.shutdown()

@pytest.mark.asyncio
async def test_retrieve_memories_no_boost_for_unsynced():
    """Test that retrieve_memories does not boost scores from unsynchronized assemblies."""
    clear_test_directory()
    
    # Initialize a test memory core
    core = SynthiansMemoryCore({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'memory_core'),
        'assembly_threshold': 0.1  # Low threshold to ensure activation for testing
    })
    await core.initialize()
    
    # Create unique test content that we can search for
    content_synced = "This is a unique memory that should be boosted when in a synchronized assembly"
    content_unsynced = "This is another unique memory that should not be boosted when in an unsynchronized assembly"
    
    # Create two test memories with the test content
    gm = GeometryManager()
    memory_synced = MemoryEntry(
        content=content_synced,
        embedding=create_random_embedding(),
        metadata={"test": True, "boosted": True}
    )
    memory_unsynced = MemoryEntry(
        content=content_unsynced,
        embedding=create_random_embedding(),
        metadata={"test": True, "boosted": False}
    )
    
    # Process these memories to add them to the memory core
    await core.process_new_memory(content_synced, embedding=memory_synced.embedding, metadata=memory_synced.metadata)
    await core.process_new_memory(content_unsynced, embedding=memory_unsynced.embedding, metadata=memory_unsynced.metadata)
    
    # Create two assemblies - one synchronized, one not
    synced_assembly = create_test_assembly(1, gm, [memory_synced], with_timestamp=True)
    unsynced_assembly = create_test_assembly(2, gm, [memory_unsynced], with_timestamp=False)
    
    # Add assemblies to memory core
    async with core._lock:
        core.assemblies[synced_assembly.assembly_id] = synced_assembly
        core.assemblies[unsynced_assembly.assembly_id] = unsynced_assembly
    
    # Retrieve memories with both memories' content in the query
    results = await core.retrieve_memories(f"{content_synced} {content_unsynced}", top_k=10)
    
    # Find the memories in the results
    synced_result = None
    unsynced_result = None
    for memory in results["memories"]:
        if content_synced in memory["content"]:
            synced_result = memory
        elif content_unsynced in memory["content"]:
            unsynced_result = memory
    
    # Verify results - both memories should be found, but only the synced one should have a boost
    assert synced_result is not None, "Synced memory should be retrieved"
    assert unsynced_result is not None, "Unsynced memory should be retrieved"
    
    # The memory in the synced assembly should have a higher score than its base similarity
    # due to assembly boost, while the unsynced one should not
    assert synced_result["boost_contribution"] > 0, "Synced memory should have a boost contribution"
    assert unsynced_result.get("boost_contribution", 0) == 0, "Unsynced memory should not have a boost contribution"
    
    # Clean up
    await core.shutdown()

@pytest.mark.asyncio
async def test_api_sync_diagnostics(aiohttp_client):
    """Test API endpoints correctly report synchronization status."""
    from fastapi import FastAPI
    from synthians_memory_core.api.server import app as core_app, lifespan
    
    # Setup test app
    app = FastAPI(lifespan=lifespan)
    app.mount("/", core_app)
    client = await aiohttp_client(app)
    
    # Get stats endpoint - should include assembly_sync field
    response = await client.get("/stats")
    assert response.status == 200
    stats = await response.json()
    
    # Verify vector_index and assembly_sync fields exist in stats
    assert "vector_index" in stats, "Stats should include vector_index information"
    assert "assembly_sync" in stats, "Stats should include assembly_sync information"
    
    # Create test assemblies - one synchronized, one not
    gm = GeometryManager()
    synced_assembly = create_test_assembly(1, gm, with_timestamp=True)
    unsynced_assembly = create_test_assembly(2, gm, with_timestamp=False)
    
    # Add assemblies to memory core
    async with app.state.memory_core._lock:
        app.state.memory_core.assemblies[synced_assembly.assembly_id] = synced_assembly
        app.state.memory_core.assemblies[unsynced_assembly.assembly_id] = unsynced_assembly
    
    # Check /assemblies/{id} for synchronized assembly
    response = await client.get(f"/assemblies/{synced_assembly.assembly_id}")
    assert response.status == 200
    synced_result = await response.json()
    
    # Check /assemblies/{id} for unsynchronized assembly
    response = await client.get(f"/assemblies/{unsynced_assembly.assembly_id}")
    assert response.status == 200
    unsynced_result = await response.json()
    
    # Verify synchronization fields are present and correct
    assert "vector_index_updated_at" in synced_result, "Synced assembly should have vector_index_updated_at field"
    assert "is_synchronized" in synced_result, "Synced assembly should have is_synchronized field"
    assert synced_result["is_synchronized"] is True, "Synced assembly should be marked as synchronized"
    
    assert "vector_index_updated_at" in unsynced_result, "Unsynced assembly should have vector_index_updated_at field"
    assert "is_synchronized" in unsynced_result, "Unsynced assembly should have is_synchronized field"
    assert unsynced_result["is_synchronized"] is False, "Unsynced assembly should be marked as not synchronized"

if __name__ == "__main__":
    """Run the tests directly for debugging"""
    asyncio.run(test_activate_assemblies_filter())
    asyncio.run(test_retrieve_memories_no_boost_for_unsynced())
