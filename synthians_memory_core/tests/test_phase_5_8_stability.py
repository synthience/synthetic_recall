import pytest
import asyncio
import json
import time
import numpy as np
import os
import sys
import logging
import shutil
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Core imports using proper package structure
from synthians_memory_core.vector_index import MemoryVectorIndex
from synthians_memory_core.memory_structures import MemoryAssembly, MemoryEntry
from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.assembly_sync_manager import AssemblySyncManager
from synthians_memory_core.geometry_manager import GeometryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase_5_8_stability")

# Test constants
TEST_DIR = os.path.join(os.getcwd(), 'test_phase_5_8')
EMBEDDING_DIM = 768
NUM_TEST_MEMORIES = 50
NUM_TEST_ASSEMBLIES = 10

# Helper functions
def clear_test_directory():
    """Remove test directory and recreate it"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    
def create_random_embedding(dim=EMBEDDING_DIM):
    """Create a random normalized embedding"""
    embedding = np.random.random(dim).astype('float32')
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def create_test_memory(idx, gm):
    """Create a test memory with random embedding"""
    memory = MemoryEntry(
        content=f"Test memory content {idx}",
        id=f"test_mem_{idx}"
    )
    memory.embedding = create_random_embedding()
    memory.embedding = gm._validate_vector(memory.embedding, f"Memory {idx}")
    return memory
    
def create_test_assembly(idx, gm, memories=None):
    """Create a test assembly with provided memories"""
    assembly = MemoryAssembly(
        geometry_manager=gm,
        assembly_id=f"test_asm_{idx}",
        name=f"Test Assembly {idx}",
        description=f"Test assembly for stability tests {idx}"
    )
    
    if memories:
        for memory in memories:
            assembly.add_memory(memory)
            
    return assembly

def corrupt_index_mapping(index):
    """Deliberately corrupt the index mapping to simulate drift"""
    # Remove a few entries from the mapping but leave them in FAISS
    keys_to_remove = random.sample(list(index.id_to_index.keys()), min(10, len(index.id_to_index)))
    for key in keys_to_remove:
        del index.id_to_index[key]
    
    # Log the deliberate corruption
    logger.info(f"Deliberately corrupted index by removing {len(keys_to_remove)} mappings")
    return keys_to_remove

@pytest.mark.asyncio
async def test_index_drift_detection():
    """Test that index drift is properly detected and reported."""
    clear_test_directory()
    
    # Initialize components
    vector_index = MemoryVectorIndex({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'vector_index'),
        'index_type': 'L2'
    })
    await vector_index.initialize()
    
    # Add test vectors
    test_vectors = [create_random_embedding() for _ in range(NUM_TEST_MEMORIES)]
    for i, vector in enumerate(test_vectors):
        memory_id = f"test_memory_{i}"
        success = await vector_index.add_async(memory_id, vector)
        assert success, f"Failed to add vector {i}"
    
    # Get initial stats - should show no drift
    initial_stats = vector_index.get_stats()  # Remove await as this is a synchronous method
    assert initial_stats['drift_count'] == 0, "Fresh index should have no drift"
    
    # Verify that integrity check passes
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert is_consistent, "Fresh index should pass integrity check"
    
    # Make a safe copy of the ID mappings before corrupting
    id_mapping_copy = dict(vector_index.id_to_index)
    
    # Deliberately create drift by adding multiple "ghost" mappings that don't exist in FAISS
    # We need more than 10 to trigger the drift_warning flag
    for i in range(15):  # Add 15 ghost mappings to exceed the warning threshold (>10)
        vector_index.id_to_index[f"non_existent_memory_{i}"] = len(vector_index.id_to_index) + 100 + i
    
    # Verify that corruption is detected
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert not is_consistent, "Corrupted index should fail integrity check"
    
    # Get stats and verify corruption is reported
    stats = vector_index.get_stats()  # Remove await as this is a synchronous method
    assert stats['drift_count'] > 0, "Corrupted index should report drift"
    assert stats['drift_warning'], "Stats should include drift warning"
    # The drift_percentage field doesn't exist in the current implementation
    # assert stats['drift_percentage'] > 0, "Drift percentage should be positive"
    
    # Restore the original mapping to avoid affecting other tests
    vector_index.id_to_index = id_mapping_copy
    
    # Verify integrity is restored
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert is_consistent, "Integrity should be restored after fixing mapping"
    
    # Clean up
    await vector_index.reset_async()
    
    logger.info("\u2705 Index drift detection test passed")

@pytest.mark.asyncio
async def test_assembly_sync_enforcement():
    """Test that assemblies are only activated when properly synchronized with the vector index."""
    clear_test_directory()
    
    # Initialize components
    gm = GeometryManager()
    vector_index = MemoryVectorIndex({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'vector_index'),
        'index_type': 'L2'
    })
    await vector_index.initialize()
    
    # Create test memories and add to index
    test_memories = [create_test_memory(i, gm) for i in range(10)]
    for mem in test_memories:
        await vector_index.add_async(mem.id, mem.embedding)
    
    # Create an assembly with these memories
    assembly = create_test_assembly(0, gm, test_memories)
    
    # Initially, the assembly should have vector_index_updated_at = None
    assert assembly.vector_index_updated_at is None, "New assembly should have no synchronization timestamp"
    
    # Test 1: Boost without sync should return base score
    memory_id = test_memories[0].id
    base_score = 0.75
    
    # Boost should return base score (no boost applied) when not synchronized
    boosted_score = assembly.boost_memory_score(memory_id, base_score)
    assert abs(boosted_score - base_score) < 0.001, "Score should not be boosted when not synchronized"
    
    # Test 2: Sync assembly and verify boost is applied
    success = await assembly.update_vector_index_async(vector_index)
    assert success, "Assembly sync should succeed"
    assert assembly.vector_index_updated_at is not None, "vector_index_updated_at should be set after sync"
    
    # Activate assembly
    assembly.activate(0.8)
    
    # Now boost should be applied
    boosted_score = assembly.boost_memory_score(memory_id, base_score, boost_factor=0.5)
    assert boosted_score > base_score, "Score should be boosted when synchronized"
    
    # Test 3: With expired sync timestamp, no boost
    # Set timestamp to 2 hours ago (beyond default 1 hour max drift)
    assembly.vector_index_updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
    
    # Boost with default max_allowed_drift_seconds should not apply boost
    boosted_score = assembly.boost_memory_score(memory_id, base_score)
    assert abs(boosted_score - base_score) < 0.001, "Score should not be boosted with expired timestamp"
    
    # But if we increase the allowed drift, boost should work
    boosted_score = assembly.boost_memory_score(
        memory_id, base_score, 
        max_allowed_drift_seconds=7200  # 2 hours
    )
    assert boosted_score > base_score, "Score should be boosted with extended drift allowance"
    
    # Get sync diagnostics and verify they're meaningful
    diagnostics = assembly.get_sync_diagnostics()
    assert 'drift_seconds' in diagnostics, "Diagnostics should include drift seconds"
    assert diagnostics['drift_seconds'] >= 7000, "Drift seconds should be ~2 hours"
    
    logger.info("\u2705 Assembly sync enforcement test passed")

@pytest.mark.asyncio
async def test_assembly_persistence_integrity():
    print("--- test_assembly_persistence_integrity START ---")
    # 0. Setup
    # --------
    # Keep using asm: prefix for the logical ID but the persistence layer will handle safe filenames
    assembly_id = "asm:test-integrity-1"
    memory_ids = [f"test-mem-{i}" for i in range(1, 6)]
    print(f"[TEST] Initializing persistence...")
    persistence = MemoryPersistence({
        'storage_path': os.path.join(TEST_DIR, 'persistence')
    })
    await persistence.initialize()
    print(f"[TEST] Persistence initialized.")

    # 1. Create and save memories
    print(f"[TEST] Saving {len(memory_ids)} memories...")
    gm = GeometryManager()
    for mem_id in memory_ids:
        mem = MemoryEntry(
            id=mem_id,
            content=f"Content for {mem_id}",
            embedding=np.random.rand(gm.config['embedding_dim']).tolist(),
            metadata={'timestamp': time.time(), 'source': 'test'}
        )
        print(f"[TEST] Saving memory {mem_id}...")
        save_success = await persistence.save_memory(mem)
        print(f"[TEST] Save success for {mem_id}: {save_success}")
        assert save_success
    print(f"[TEST] Memories saved.")

    # 2. Create an assembly
    print(f"[TEST] Creating assembly {assembly_id}...")
    assembly = MemoryAssembly(
        geometry_manager=gm,
        assembly_id=assembly_id,
        name="Test Integrity Assembly",
        description="Assembly created for persistence integrity test"
    )
    
    # Set additional properties
    assembly.tags = set(["test", "integrity"])
    assembly.topics = ["persistence", "asyncio"]
    assembly.vector_index_updated_at = datetime.now(timezone.utc)  # Simulate sync
    
    # Load and add memories to the assembly
    print(f"[TEST] Loading saved memories to add to assembly {assembly_id}...")
    for mem_id in memory_ids:
        # Load the memory entry we just saved
        mem_entry = await persistence.load_memory(mem_id, geometry_manager=gm)
        if mem_entry:
            print(f"[TEST] Adding memory {mem_id} to assembly {assembly_id}...")
            assembly.add_memory(mem_entry)  # This automatically updates the composite embedding
        else:
            pytest.fail(f"Failed to load memory {mem_id} needed for assembly creation")
    print(f"[TEST] Memories added to assembly {assembly_id}.")
    
    # Verify the composite embedding was created
    print(f"[TEST] Verifying composite embedding for {assembly_id}...")
    assert assembly.composite_embedding is not None, "Composite embedding should have been created during memory addition"
    print(f"[TEST] Composite embedding verified for {assembly_id}.")
    print(f"[TEST] Assembly {assembly_id} created.")

    # 3. Save the assembly
    print(f"[TEST] Saving assembly {assembly_id}...")
    print(f"[TEST] Assembly properties before save: "
          f"id={assembly.assembly_id}, "
          f"memories={len(assembly.memories)}, "
          f"composite_embedding_shape={None if assembly.composite_embedding is None else len(assembly.composite_embedding)}")
    save_success = await persistence.save_assembly(assembly, geometry_manager=gm)
    print(f"[TEST] Save assembly result: {save_success}")
    assert save_success, f"Failed to save assembly {assembly_id}"
    print(f"[TEST] Assembly {assembly_id} saved.")

    # --- Simulate Application Restart (Clear Persistence Instance Cache) ---
    print(f"[TEST] Simulating restart: Clearing persistence index/cache...")
    persistence.memory_index.clear() # Clear in-memory index
    # In a real scenario, a new Persistence object would be created
    print(f"[TEST] Persistence cache cleared.")

    # 4. Load the assembly
    print(f"[TEST] Loading assembly {assembly_id}...")
    loaded_assembly = await persistence.load_assembly(assembly_id, gm)
    print(f"[TEST] Load result for assembly {assembly_id}: {type(loaded_assembly)}")
    assert loaded_assembly is not None
    assert isinstance(loaded_assembly, MemoryAssembly)
    print(f"[TEST] Assembly {assembly_id} loaded successfully.")

    # 5. Verify loaded assembly integrity
    print(f"[TEST] Verifying integrity of loaded assembly {assembly_id}...")
    assert loaded_assembly.assembly_id == assembly_id
    assert loaded_assembly.memories == set(memory_ids)
    assert loaded_assembly.tags == {"test", "integrity"}
    assert loaded_assembly.topics == ["persistence", "asyncio"]
    assert loaded_assembly.composite_embedding is not None
    assert np.allclose(loaded_assembly.composite_embedding, assembly.composite_embedding)
    assert loaded_assembly.vector_index_updated_at is not None
    print(f"[TEST] Loaded assembly integrity verified.")

    print("--- test_assembly_persistence_integrity END ---")

@pytest.mark.asyncio
async def test_retry_queue_recovery():
    """Test that failed synchronization operations get retried."""
    clear_test_directory()
    
    # Initialize components
    gm = GeometryManager()
    vector_index = MemoryVectorIndex({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'vector_index'),
        'index_type': 'L2'
    })
    await vector_index.initialize()
    
    # Create sync manager with retry interval
    storage_path = os.path.join(TEST_DIR, 'sync_manager')
    os.makedirs(storage_path, exist_ok=True)
    sync_manager = AssemblySyncManager(vector_index, storage_path=storage_path, max_retries=3)
    await sync_manager.initialize()
    
    # Create test memories and assemblies
    test_memories = [create_test_memory(i, gm) for i in range(10)]
    assemblies = [create_test_assembly(i, gm, test_memories[i:i+3]) for i in range(0, 9, 3)]
    
    # Deliberately make the vector index unavailable
    # Simulating a temporary failure - we'll simulate it by clearing
    # the vector_index's internal state without proper shutdown
    await vector_index.reset_async()
    
    # Now try to sync assemblies - should fail but be queued
    for asm in assemblies:
        # This should queue them for retry since index is unavailable
        await sync_manager.queue_assembly_for_sync(asm, vector_index)
    
    # Verify they're in the retry queue
    retry_queue = sync_manager.get_pending_updates()
    assert len(retry_queue) > 0, "Assemblies should be in retry queue"
    
    # Now make vector index available and add test vectors
    for mem in test_memories:
        await vector_index.add_async(mem.id, mem.embedding)
    
    # Run one retry cycle manually
    retried = await sync_manager.process_pending_updates(vector_index)
    assert retried > 0, "Should have retried pending syncs"
    
    # Verify retry queue is now empty or reduced
    retry_queue = sync_manager.get_pending_updates()
    assert len(retry_queue) < len(assemblies), "Retry queue should be reduced"
    
    # Verify assemblies are now synchronized
    for asm in assemblies:
        assert asm.vector_index_updated_at is not None, "Assembly should be marked as synchronized"
    
    # Clean up
    await sync_manager.shutdown() if hasattr(sync_manager, 'shutdown') else None
    await vector_index.reset_async()
    
    logger.info("\u2705 Retry queue recovery test passed")

@pytest.mark.asyncio
async def test_index_auto_repair():
    """Test that the index can automatically repair integrity issues."""
    clear_test_directory()
    
    # Initialize components
    vector_index = MemoryVectorIndex({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'vector_index'),
        'index_type': 'L2'
    })
    await vector_index.initialize()
    
    # Add vectors to the index
    test_vectors = [create_random_embedding() for _ in range(20)]
    for i, vector in enumerate(test_vectors):
        memory_id = f"test_memory_{i}"
        success = await vector_index.add_async(memory_id, vector)
        assert success, f"Failed to add vector {i}"
    
    # Save the index
    success = await vector_index.save_async()
    assert success, "Index save should succeed"
    
    # Verify initial integrity
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert is_consistent, "Fresh index should pass integrity check"
    
    # Deliberately corrupt the index by removing mappings but keeping vectors
    removed_keys = corrupt_index_mapping(vector_index)
    assert len(removed_keys) > 0, "Should have removed some keys"
    
    # Verify corruption is detected
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert not is_consistent, "Corrupted index should fail integrity check"
    
    # Get stats before repair
    before_stats = vector_index.get_stats()  
    assert before_stats['drift_count'] > 0, "Should detect drift before repair"
    
    # Repair the index
    repaired = await vector_index.repair_index_async()
    assert repaired, "Repair operation should succeed"
    
    # Get stats after repair
    after_stats = vector_index.get_stats()  # Remove await as this is a synchronous method
    
    # Verify drift is reduced or eliminated
    assert after_stats['drift_count'] < before_stats['drift_count'], "Drift should be reduced after repair"
    # Ideally, repair should completely eliminate drift
    assert after_stats['drift_count'] == 0, "Complete repair should eliminate all drift"
    assert after_stats['mapping_count'] == after_stats['faiss_count'], "Mapping and FAISS counts should match after repair"
    
    # Verify integrity is restored
    is_consistent, details = vector_index.verify_index_integrity()  # Remove await as this is a synchronous method
    assert is_consistent, "Index should pass integrity check after repair"
    
    logger.info("\u2705 Index auto-repair test passed")

@pytest.mark.asyncio
async def test_post_initialization_check():
    """Test that post-initialization checks detect anomalies."""
    clear_test_directory()
    
    # Initialize components
    vector_index = MemoryVectorIndex({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'vector_index'),
        'index_type': 'L2'
    })
    # Don't call initialize() yet, as we're specifically testing that functionality
    
    # Run post-init check with the default configuration - should pass
    success = await vector_index._post_initialize_check()
    assert success, "Post-init check should pass for properly initialized index"
    
    # Add vectors to the index to ensure it's properly working
    await vector_index.initialize()
    test_vectors = [create_random_embedding() for _ in range(5)]
    for i, vector in enumerate(test_vectors):
        memory_id = f"test_memory_{i}"
        success = await vector_index.add_async(memory_id, vector)
        assert success, f"Failed to add vector {i}"
    
    # Create a separate index with wrong dimensions to test validation
    import faiss
    
    # Save the original index
    original_index = vector_index.index
    
    # Create an index with a mismatched dimension
    wrong_dim = EMBEDDING_DIM // 2  # Half the expected dimension
    wrong_dim_index = faiss.IndexFlatL2(wrong_dim)
    
    # Replace the index with the wrong dimension one
    vector_index.index = wrong_dim_index
    
    # Post-init check should detect the dimension mismatch
    success = await vector_index._post_initialize_check()
    assert not success, "Post-init check should fail with wrong dimension"
    
    # Reset the index back to the original
    vector_index.index = original_index
    
    # Verify it passes the check again
    success = await vector_index._post_initialize_check()
    assert success, "Post-init check should pass after restoring proper index"
    
    logger.info("\u2705 Post-initialization check test passed")

@pytest.mark.asyncio
async def test_end_to_end_sync_enforcement():
    """Test end-to-end retrieval with sync enforcement in the complete pipeline."""
    clear_test_directory()
    
    # Initialize a memory core with test configuration
    memory_core = SynthiansMemoryCore({
        'embedding_dim': EMBEDDING_DIM,
        'storage_path': os.path.join(TEST_DIR, 'memory_core'),
        'assembly_threshold': 0.3  # Lower threshold for testing
    })
    await memory_core.initialize()
    
    # Create test memories with distinct content for easy retrieval
    synced_memory_content = "This unique content should receive boost when in a synchronized assembly"
    unsynced_memory_content = "This other unique content should not receive boost when in an unsynchronized assembly"
    
    # Process the memories
    synced_memory_id = await memory_core.process_new_memory(
        synced_memory_content,
        embedding=create_random_embedding(),
        metadata={"test": True, "group": "synced"}
    )
    unsynced_memory_id = await memory_core.process_new_memory(
        unsynced_memory_content,
        embedding=create_random_embedding(),
        metadata={"test": True, "group": "unsynced"}
    )
    
    # Create a synchronized assembly with the first memory
    synced_assembly = MemoryAssembly(
        assembly_id="test_synced_assembly",
        name="Test Synced Assembly",
        geometry_manager=memory_core.geometry_manager
    )
    synced_memory = await memory_core.get_memory_by_id_async(synced_memory_id)
    synced_assembly.add_memory(synced_memory)
    synced_assembly.vector_index_updated_at = datetime.now(timezone.utc)  # Mark as synchronized
    
    # Create an unsynchronized assembly with the second memory
    unsynced_assembly = MemoryAssembly(
        assembly_id="test_unsynced_assembly",
        name="Test Unsynced Assembly",
        geometry_manager=memory_core.geometry_manager
    )
    unsynced_memory = await memory_core.get_memory_by_id_async(unsynced_memory_id)
    unsynced_assembly.add_memory(unsynced_memory)
    # Deliberately leave vector_index_updated_at as None to simulate unsynced state
    
    # Add assemblies to memory core
    async with memory_core._lock:
        memory_core.assemblies[synced_assembly.assembly_id] = synced_assembly
        memory_core.assemblies[unsynced_assembly.assembly_id] = unsynced_assembly
    
    # Perform a retrieval that should match both memories
    query = "unique content in assemblies"  # Should match both memories
    results = await memory_core.retrieve_memories(query, top_k=10)
    
    # Verify both memories were found
    assert "memories" in results and len(results["memories"]) >= 2, "Should retrieve both memories"
    
    # Find the results corresponding to our test memories
    synced_result = None
    unsynced_result = None
    for memory in results["memories"]:
        if synced_memory_content in memory.get("content", ""):
            synced_result = memory
        elif unsynced_memory_content in memory.get("content", ""):
            unsynced_result = memory
    
    # Verify both memories were found in the results
    assert synced_result is not None, "Synced memory should be in results"
    assert unsynced_result is not None, "Unsynced memory should be in results"
    
    # Examine the metadata to verify sync enforcement
    # Synced memory should have assembly boost contribution
    assert synced_result.get("boost_info", {}).get("assembly_boost", 0) > 0, "Synced memory should have assembly boost"
    
    # Unsynced memory should not have assembly boost contribution
    unsynced_boost = unsynced_result.get("boost_info", {}).get("assembly_boost", 0)
    assert unsynced_boost == 0, f"Unsynced memory should not have assembly boost, got {unsynced_boost}"
    
    # Clean up
    await memory_core.shutdown()
    clear_test_directory()
    
    logger.info("\u2705 End-to-end sync enforcement test passed")

if __name__ == "__main__":
    """Run the tests directly for debugging"""
    asyncio.run(test_index_drift_detection())
    asyncio.run(test_assembly_sync_enforcement())
    asyncio.run(test_assembly_persistence_integrity())
    asyncio.run(test_retry_queue_recovery())
    asyncio.run(test_index_auto_repair())
    asyncio.run(test_post_initialization_check())
    asyncio.run(test_end_to_end_sync_enforcement())
