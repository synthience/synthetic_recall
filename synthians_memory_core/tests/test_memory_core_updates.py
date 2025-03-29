# tests/test_memory_core_updates.py

import pytest
import pytest_asyncio  # Import the decorator
import asyncio
import json
import time
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
import numpy as np
from typing import Dict, Any, List, Optional
import aiofiles # Ensure aiofiles is imported

# Import the necessary components from the core
from synthians_memory_core import (
    SynthiansMemoryCore,
    MemoryEntry,
    GeometryManager,
    MemoryPersistence
)
from synthians_memory_core.vector_index import MemoryVectorIndex

# --- Dummy Async Lock for Testing ---
class DummyAsyncLock:
    """A dummy lock that doesn't block, for testing purposes."""
    async def __aenter__(self):
        # print("DEBUG: Entering DummyAsyncLock") # Optional debug print
        pass
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # print("DEBUG: Exiting DummyAsyncLock") # Optional debug print
        pass

# Helper function for removing test directories
async def _remove_directory_with_retry(directory, max_attempts=5, delay=0.5):
    """Helper function to remove a directory with retry logic"""
    attempts = 0
    while attempts < max_attempts:
        try:
            shutil.rmtree(directory, ignore_errors=True)
            if not os.path.exists(directory):
                print(f"  - Successfully removed {directory}")
                return True
            raise OSError(f"Directory still exists after removal: {directory}")
        except OSError as e:
            attempts += 1
            print(f"  - Error removing directory (attempt {attempts}/{max_attempts}): {e}")
            if attempts < max_attempts:
                await asyncio.sleep(delay)
    
    print(f"  - Failed to remove directory after {max_attempts} attempts: {directory}")
    return False

# --- REVISED FIXTURE ---
@pytest_asyncio.fixture
async def memory_core(temp_test_dir, request):
    """Provides a properly configured SynthiansMemoryCore instance with cleanup."""
    # Create a test-specific directory within the temp directory
    test_name = request.node.name
    test_dir = os.path.join(temp_test_dir, test_name.replace('/', '_').replace(':', '_')) # Added replace for ':'
    os.makedirs(test_dir, exist_ok=True)
    print(f"\nSetting up memory_core fixture for test: {test_name} in {test_dir}")

    # Create components for testing
    geometry_manager = GeometryManager(
        config={
            'embedding_dim': 384,
            'geometry_type': 'euclidean',
        }
    )

    vector_index = MemoryVectorIndex(
        config={
            'embedding_dim': 384,
            'storage_path': test_dir,
            'index_type': 'L2',
            'use_gpu': False
        }
    )

    # Use the original MemoryPersistence, but we'll override its lock later
    persistence = MemoryPersistence(
        config={
            'storage_path': test_dir,
            # 'auto_save': True # Removed, rely on explicit saves/shutdown
        }
    )
    # Initialize persistence explicitly before creating core
    await persistence.initialize()

    # Create the memory core instance with only a config
    core = SynthiansMemoryCore(
        config={
            'embedding_dim': 384,
            'storage_path': test_dir,
            'vector_index_type': 'L2',
            'use_gpu': False,
            # Disable background tasks for unit testing updates
            'persistence_interval': 3600 * 24,
            'decay_interval': 3600 * 24,
            'prune_check_interval': 3600 * 24,
        }
    )

    # Manually replace the components for testing
    core.vector_index = vector_index
    core.persistence = persistence
    core.geometry_manager = geometry_manager

    # Replace the locks with dummy locks
    dummy_lock = DummyAsyncLock()
    core._lock = dummy_lock
    persistence._lock = dummy_lock

    # Initialize core - crucial step to load state and start components
    # Background tasks are disabled by high intervals in config
    await core.initialize()

    # Setup cleanup function
    async def async_finalizer():
        """Async cleanup function that properly awaits shutdown"""
        print(f"\n==== Cleaning up memory_core for test: {test_name} ====")
        
        # First, set the shutdown signal to stop background loops
        if hasattr(core, '_shutdown_signal'):
            core._shutdown_signal.set()
            print("- Set shutdown signal")
        
        # Wait a moment for tasks to observe the signal
        await asyncio.sleep(0.2)
        
        # Explicitly get all tasks that might be associated with this test
        # to ensure we don't leave anything hanging
        all_tasks = asyncio.all_tasks()
        tasks_to_cancel = [
            t for t in all_tasks 
            if not t.done() and 
               t is not asyncio.current_task() and
               'test_' in t.get_name()  # Only care about test-related tasks
        ]
        
        # Cancel all background tasks associated with the test
        if tasks_to_cancel:
            print(f"- Found {len(tasks_to_cancel)} tasks to cancel")
            for task in tasks_to_cancel:
                if not task.done() and not task.cancelled():
                    task.cancel()
                    print(f"  - Cancelled task: {task.get_name()}")
        
            # Wait for the tasks to finish cancelling
            try:
                await asyncio.wait(tasks_to_cancel, timeout=2)
                print("- Waited for tasks to cancel")
            except Exception as e:
                print(f"- Error waiting for tasks: {e}")
        
        # Now run the core's shutdown method
        if hasattr(core, 'shutdown'):
            print("- Running core.shutdown()...")
            try:
                await asyncio.wait_for(core.shutdown(), timeout=3)
                print("- Shutdown completed")
            except asyncio.TimeoutError:
                print("- Warning: Shutdown timed out")
            except Exception as e:
                print(f"- Error during shutdown: {e}")
        
        # Finally, remove the test directory
        if os.path.exists(test_dir):
            print(f"- Removing test directory: {test_dir}")
            await _remove_directory_with_retry(test_dir, max_attempts=3)
            
        print(f"==== Cleanup finished for test: {test_name} ====")
            
    def finalizer():
        """Sync wrapper for the async finalizer"""
        loop = asyncio.get_event_loop_policy().get_event_loop()
        
        # If we're in an event loop running from pytest_asyncio
        if loop.is_running():
            task = asyncio.create_task(
                async_finalizer(), 
                name=f"finalizer_{test_name}"
            )
            
            # We need to ensure this task completes, but we can't await directly
            # Create a shared event for signaling completion
            done_event = threading.Event()
            
            def _on_task_done(task):
                # Signal that the task is done, regardless of result
                done_event.set()
                
            task.add_done_callback(_on_task_done)
            
            # Wait for the task to complete with a timeout
            # This is a blocking wait, but it's necessary for cleanup
            if not done_event.wait(timeout=5):
                print("Warning: Cleanup task timed out!")
        else:
            # If we're not in a running loop (shouldn't happen with pytest_asyncio)
            loop.run_until_complete(async_finalizer())
    
    # Register the cleanup function
    request.addfinalizer(finalizer)
    
    # Return the memory core for use in tests
    return core

# --- Tests ---

@pytest.mark.asyncio
async def test_get_memory_by_id(memory_core: SynthiansMemoryCore):
    """Test retrieving a memory by ID."""
    print("\n--- Running test_get_memory_by_id ---")
    # Create a test memory using the correct method
    timestamp = datetime.now(timezone.utc)
    content = f"Test memory for retrieval at {timestamp.isoformat()}"
    original_embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32) # Use configured dim

    # Normalize the original embedding *before* sending, as the core will normalize it.
    normalized_original_embedding = memory_core.geometry_manager.normalize_embedding(original_embedding)

    print(f"Creating memory '{content[:20]}...'")
    # Use process_new_memory to store, passing the original (will be normalized inside)
    memory_entry = await memory_core.process_new_memory(
        content=content,
        embedding=original_embedding, # Send the original random one
        metadata={"source": "test_get_by_id", "importance": 0.75}
    )
    assert memory_entry is not None, "Failed to create Memory A"
    memory_id = memory_entry.id
    print(f"Memory created with ID: {memory_id}")

    # Retrieve the memory by ID using the core method
    print(f"Retrieving memory {memory_id}...")
    # Use the core's synchronous method directly (as it accesses internal dict)
    # Ensure the lock is the dummy one to avoid blocking
    assert isinstance(memory_core._lock, DummyAsyncLock)
    retrieved_memory = memory_core.get_memory_by_id(memory_id)

    # Assert memory was retrieved
    assert retrieved_memory is not None, f"Memory with ID {memory_id} was not found"
    assert isinstance(retrieved_memory, MemoryEntry), "get_memory_by_id did not return a MemoryEntry object"
    print("Memory retrieved successfully.")

    # Verify memory contents using object attributes
    assert retrieved_memory.id == memory_id
    assert retrieved_memory.content == content, "Retrieved memory content does not match original"
    assert retrieved_memory.embedding is not None, "Retrieved memory embedding is None"

    # Compare the *retrieved* embedding with the *normalized version* of the original
    print("Comparing embeddings...")
    assert np.allclose(retrieved_memory.embedding, normalized_original_embedding, atol=1e-6), \
        f"Retrieved memory embedding does not match the normalized original.\nRetrieved (first 5): {retrieved_memory.embedding[:5]}\nExpected (first 5): {normalized_original_embedding[:5]}"
    print("Embeddings match.")

    assert retrieved_memory.metadata.get("source") == "test_get_by_id", "Retrieved memory metadata does not match original"

    # Test retrieving non-existent memory
    print("Testing retrieval of non-existent memory...")
    non_existent_id = "non_existent_id_12345"
    non_existent_memory = memory_core.get_memory_by_id(non_existent_id)
    assert non_existent_memory is None, f"Memory with non-existent ID {non_existent_id} was found"
    print("Non-existent memory test passed.")
    print("--- test_get_memory_by_id PASSED ---")


@pytest.mark.asyncio
async def test_update_quickrecal_score(memory_core: SynthiansMemoryCore):
    """Test updating the QuickRecal score of a memory."""
    print("\n--- Running test_update_quickrecal_score ---")
    # Create a test memory
    timestamp = datetime.now(timezone.utc)
    content = f"Test memory for QuickRecal update at {timestamp.isoformat()}"
    embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32)

    # Use process_new_memory
    print("Creating memory...")
    try:
        async with asyncio.timeout(10):  # 10 second timeout for memory creation
            memory_entry = await memory_core.process_new_memory(
                content=content,
                embedding=embedding,
                metadata={"source": "test_update_quickrecal"}
            )
        assert memory_entry is not None, "Failed to create memory"
        memory_id = memory_entry.id
        initial_score_actual = memory_entry.quickrecal_score # Get the actual initial score
        print(f"Memory created (ID: {memory_id}), initial score: {initial_score_actual:.6f}")
    except asyncio.TimeoutError:
        pytest.fail("Timed out waiting for process_new_memory to complete - possible deadlock")

    # Give time for any background tasks to complete (though they shouldn't run)
    await asyncio.sleep(0.1)

    # Verify initial score
    # Use synchronous get_memory_by_id
    assert isinstance(memory_core._lock, DummyAsyncLock)
    memory_before = memory_core.get_memory_by_id(memory_id)
    assert memory_before is not None, f"Memory {memory_id} not found"
    assert abs(memory_before.quickrecal_score - initial_score_actual) < 1e-6

    # Update QuickRecal score
    new_score = 0.9
    print(f"Updating score to {new_score}...")

    # Use a timeout to prevent indefinite waiting if deadlock occurs
    try:
        async with asyncio.timeout(5):  # 5 seconds should be plenty
            updated = await memory_core.update_memory(
                memory_id=memory_id,
                updates={"quickrecal_score": new_score}
            )
        assert updated is True, "Memory update failed"
        print("Update successful.")
    except asyncio.TimeoutError:
        pytest.fail("Timed out waiting for update_memory to complete - possible deadlock")

    # Give time for persistence (even though loop is disabled, update_memory calls save)
    await asyncio.sleep(0.1)

    # Verify updated score
    # Use synchronous get_memory_by_id
    memory_after = memory_core.get_memory_by_id(memory_id)
    assert memory_after is not None, f"Memory {memory_id} not found after update"
    assert abs(memory_after.quickrecal_score - new_score) < 1e-6, \
        f"QuickRecal score was not updated correctly (Expected: {new_score}, Found: {memory_after.quickrecal_score})"
    print(f"Score updated to: {memory_after.quickrecal_score:.6f}")

    # Test update clamping (high)
    print("Testing high score clamping...")
    # Use a timeout to prevent indefinite waiting if deadlock occurs
    try:
        async with asyncio.timeout(5):  # 5 seconds timeout
            updated = await memory_core.update_memory(
                memory_id=memory_id,
                updates={"quickrecal_score": 1.5}  # Should be clamped to 1.0
            )
        assert updated is True, "Memory update (high score) failed"
    except asyncio.TimeoutError:
        pytest.fail("Timed out waiting for update_memory (high score) to complete - possible deadlock")

    # Verify clamped score
    await asyncio.sleep(0.1)
    memory_after_high = memory_core.get_memory_by_id(memory_id)
    assert memory_after_high is not None
    assert abs(memory_after_high.quickrecal_score - 1.0) < 1e-6, \
        f"Score was not properly clamped (Expected: 1.0, Found: {memory_after_high.quickrecal_score})"
    print(f"Score clamped to: {memory_after_high.quickrecal_score:.6f}")

    # Test update clamping (low)
    print("Testing low score clamping...")
    try:
        async with asyncio.timeout(5):  # 5 seconds timeout
            updated = await memory_core.update_memory(
                memory_id=memory_id,
                updates={"quickrecal_score": -0.5}  # Should be clamped to 0.0
            )
        assert updated is True, "Memory update (low score) failed"
    except asyncio.TimeoutError:
        pytest.fail("Timed out waiting for update_memory (low score) to complete - possible deadlock")

    # Verify clamped score
    await asyncio.sleep(0.1)
    memory_after_low = memory_core.get_memory_by_id(memory_id)
    assert memory_after_low is not None
    assert abs(memory_after_low.quickrecal_score - 0.0) < 1e-6, \
        f"Score was not properly clamped (Expected: 0.0, Found: {memory_after_low.quickrecal_score})"
    print(f"Score clamped to: {memory_after_low.quickrecal_score:.6f}")
    print("--- test_update_quickrecal_score PASSED ---")


@pytest.mark.asyncio
async def test_update_metadata(memory_core: SynthiansMemoryCore):
    """Test updating metadata of a memory."""
    print("\n--- Running test_update_metadata ---")
    # Create a test memory
    timestamp = datetime.now(timezone.utc)
    content = f"Test memory for metadata update at {timestamp.isoformat()}"
    embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32)
    initial_metadata = {
        "source": "test_update_metadata",
        "tags": ["test", "metadata"],
        "nested": {"key1": "value1", "key2": "value2"}
    }
    memory_entry = await memory_core.process_new_memory(
        content=content,
        embedding=embedding,
        metadata=initial_metadata.copy()
    )
    assert memory_entry is not None, "Failed to create memory"
    memory_id = memory_entry.id
    print(f"Memory created (ID: {memory_id})")

    # Verify initial custom metadata persisted (synthesized data also exists)
    memory_before = memory_core.get_memory_by_id(memory_id)
    assert memory_before is not None
    assert memory_before.metadata.get("source") == initial_metadata["source"]
    assert set(memory_before.metadata.get("tags", [])) == set(initial_metadata["tags"]) # Use set for order independence
    assert memory_before.metadata.get("nested") == initial_metadata["nested"]
    print("Initial metadata verified.")

    # Update metadata
    metadata_updates = {
        "category": "tested",
        "tags": ["test", "metadata", "updated"], # Replace list
        "nested": {"key1": "updated_value1", "key3": "new_value3"}, # Merge dict
        "another_new_field": 123
    }
    print(f"Updating metadata with: {metadata_updates}")
    updated = await memory_core.update_memory(
        memory_id=memory_id,
        updates={"metadata": metadata_updates}
    )
    assert updated is True, "Memory metadata update failed"
    print("Metadata update successful.")

    # Verify updated metadata
    await asyncio.sleep(0.1) # Allow persistence
    memory_after = memory_core.get_memory_by_id(memory_id)
    assert memory_after is not None
    final_metadata = memory_after.metadata
    print(f"Final Metadata: {json.dumps(final_metadata, indent=2)}")

    # Check updated and added fields
    assert final_metadata.get("category") == "tested"
    assert set(final_metadata.get("tags", [])) == set(["test", "metadata", "updated"])
    assert final_metadata.get("another_new_field") == 123

    # Check merged nested field updates
    assert final_metadata.get("nested", {}).get("key1") == "updated_value1"
    assert final_metadata.get("nested", {}).get("key3") == "new_value3"
    # Check original nested field persisted
    assert final_metadata.get("nested", {}).get("key2") == "value2"

    # Check original top-level field persisted
    assert final_metadata.get("source") == "test_update_metadata"
    print("--- test_update_metadata PASSED ---")


@pytest.mark.asyncio
async def test_update_invalid_fields(memory_core: SynthiansMemoryCore):
    """Test updating with invalid/non-existent fields."""
    print("\n--- Running test_update_invalid_fields ---")
    timestamp = datetime.now(timezone.utc)
    content = f"Test memory for invalid field update at {timestamp.isoformat()}"
    embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32)
    memory_entry = await memory_core.process_new_memory(
        content=content,
        embedding=embedding,
        metadata={"source": "test_invalid_fields"}
    )
    assert memory_entry is not None, "Failed to create memory"
    memory_id = memory_entry.id

    # Try to update with invalid field
    print("Attempting update with invalid field 'invalid_field_xyz'...")
    updated_invalid = await memory_core.update_memory(
        memory_id=memory_id,
        updates={"invalid_field_xyz": "some_value"}
    )
    # Update should still likely return True if it ignores bad fields
    print(f"Update call returned: {updated_invalid}")
    await asyncio.sleep(0.1)
    memory_after_invalid = memory_core.get_memory_by_id(memory_id)
    assert memory_after_invalid is not None
    assert not hasattr(memory_after_invalid, "invalid_field_xyz"), "Invalid field was added to memory object"
    assert "invalid_field_xyz" not in memory_after_invalid.metadata, "Invalid field was added to metadata"
    print("Verified invalid field was ignored.")

    # Try to update with a valid field and an invalid field
    initial_score = memory_after_invalid.quickrecal_score
    print(f"Attempting update with valid score and invalid field ('another_invalid_field'). Initial score: {initial_score}")
    updated_mixed = await memory_core.update_memory(
        memory_id=memory_id,
        updates={
            "quickrecal_score": 0.77,
            "another_invalid_field": "another_value"
        }
    )
    assert updated_mixed is True, "Mixed update failed"
    print("Mixed update successful.")

    await asyncio.sleep(0.1)
    memory_after_mixed = memory_core.get_memory_by_id(memory_id)
    assert memory_after_mixed is not None
    assert abs(memory_after_mixed.quickrecal_score - 0.77) < 1e-6, "Valid field 'quickrecal_score' was not updated during mixed update"
    assert not hasattr(memory_after_mixed, "another_invalid_field"), "Invalid field was added to memory object during mixed update"
    assert "another_invalid_field" not in memory_after_mixed.metadata, "Invalid field was added to metadata during mixed update"
    print("Verified mixed update handled correctly.")
    print("--- test_update_invalid_fields PASSED ---")


@pytest.mark.asyncio
async def test_update_nonexistent_memory(memory_core: SynthiansMemoryCore):
    """Test updating a memory that doesn't exist."""
    print("\n--- Running test_update_nonexistent_memory ---")
    non_existent_id = "non_existent_id_98765"
    print(f"Attempting update for non-existent ID: {non_existent_id}")
    updated = await memory_core.update_memory(
        memory_id=non_existent_id,
        updates={"quickrecal_score": 0.8}
    )
    assert updated is False, "Update to non-existent memory reported success"
    print("Verified update returned False for non-existent memory.")
    print("--- test_update_nonexistent_memory PASSED ---")


@pytest.mark.asyncio
async def test_update_persistence(memory_core: SynthiansMemoryCore, temp_test_dir, request):
    """Test that updates are persisted properly by reloading."""
    print("\n--- Running test_update_persistence ---")
    test_name = request.node.name
    test_dir = os.path.join(temp_test_dir, test_name.replace('/', '_').replace(':', '_'))
    print(f"Using test directory: {test_dir}")

    timestamp = datetime.now(timezone.utc)
    content = f"Test memory for persistence at {timestamp.isoformat()}"
    embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32)
    print("Creating initial memory...")
    memory_entry = await memory_core.process_new_memory(
        content=content,
        embedding=embedding,
        metadata={"source": "test_persistence"},
    )
    assert memory_entry is not None, "Failed to create memory"
    memory_id = memory_entry.id
    print(f"Memory created (ID: {memory_id})")

    # Update the memory
    new_score = 0.88
    new_meta_value = "updated_value"
    update_timestamp_iso = datetime.now(timezone.utc).isoformat()
    updates_dict = {
        "quickrecal_score": new_score,
        "metadata": {"update_status": new_meta_value, "last_update_iso": update_timestamp_iso }
    }
    print(f"Updating memory {memory_id} with: {updates_dict}")
    updated = await memory_core.update_memory(
        memory_id=memory_id,
        updates=updates_dict
    )
    assert updated is True, "Memory update failed"
    print("Memory update successful.")

    # Ensure persistence happens - explicitly call save if loops disabled
    await memory_core.persistence.save_memory(memory_core.get_memory_by_id(memory_id))
    await memory_core.persistence._save_index() # Force save index
    print("Explicit save performed.")
    await asyncio.sleep(0.2) # Small delay

    # --- Simulate Restart ---
    print("Shutting down original memory core...")
    # Need to shut down cleanly to ensure files are closed
    # await memory_core.shutdown() # Shutdown might cause issues with fixture cleanup

    # Create a NEW memory core instance using the SAME config
    config = memory_core.config # Reuse config dict
    print(f"Re-initializing Memory Core with storage path: {config['storage_path']}")
    new_memory_core = SynthiansMemoryCore(config=config)
    # Replace locks with dummy locks for the new instance too
    new_memory_core._lock = DummyAsyncLock()
    new_memory_core.persistence._lock = DummyAsyncLock()
    # Initialize the new core, which loads from persistence
    await new_memory_core.initialize()
    print("New memory core initialized, loading from persistence.")

    # Retrieve the memory from the new instance
    print(f"Retrieving memory {memory_id} from reloaded core...")
    memory_after_reload = new_memory_core.get_memory_by_id(memory_id)

    # Verify the updated values were loaded from persistence
    assert memory_after_reload is not None, f"Memory with ID {memory_id} was not found after reload"
    assert isinstance(memory_after_reload, MemoryEntry), "Did not get MemoryEntry object after reload"
    print("Memory retrieved after reload.")

    assert abs(memory_after_reload.quickrecal_score - new_score) < 1e-6, \
        f"Updated QuickRecal score was not persisted (Expected: {new_score}, Found: {memory_after_reload.quickrecal_score})"
    assert memory_after_reload.metadata.get("update_status") == new_meta_value, \
        "Updated metadata field 'update_status' was not persisted"
    assert memory_after_reload.metadata.get("last_update_iso") == update_timestamp_iso, \
        "Added metadata field 'last_update_iso' was not persisted"
    assert memory_after_reload.metadata.get("source") == "test_persistence", \
        "Original metadata field 'source' was lost during update/persistence"
    print("Verified persisted updates.")

    # await new_memory_core.shutdown() # Shutdown the new core instance
    print("--- test_update_persistence PASSED ---")


@pytest.mark.asyncio
async def test_quickrecal_updated_timestamp(memory_core: SynthiansMemoryCore):
    """Test that quickrecal_updated_at timestamp is set correctly in metadata."""
    print("\n--- Running test_quickrecal_updated_timestamp ---")
    content = "Test memory for quickrecal timestamp"
    embedding = np.random.rand(memory_core.config['embedding_dim']).astype(np.float32)

    print("Creating memory...")
    memory_entry = await memory_core.process_new_memory(content=content, embedding=embedding)
    assert memory_entry is not None, "Failed to create memory"
    memory_id = memory_entry.id

    memory_before = memory_core.get_memory_by_id(memory_id)
    assert memory_before.metadata.get('quickrecal_updated_at') is None, \
        "quickrecal_updated_at should be None initially in metadata"
    print("Initial state verified (no quickrecal_updated_at).")

    # Update score
    await asyncio.sleep(0.1)
    time_before_update = datetime.now(timezone.utc)
    await asyncio.sleep(0.1)

    print("Updating quickrecal_score...")
    updated = await memory_core.update_memory(
        memory_id=memory_id,
        updates={"quickrecal_score": 0.9}
    )
    assert updated is True

    await asyncio.sleep(0.1)
    time_after_update = datetime.now(timezone.utc)
    await asyncio.sleep(0.1) # Allow persistence

    memory_after = memory_core.get_memory_by_id(memory_id)
    assert memory_after is not None

    updated_at_str = memory_after.metadata.get('quickrecal_updated_at')
    assert updated_at_str is not None, "quickrecal_updated_at was not set in metadata"
    print(f"Found quickrecal_updated_at: {updated_at_str}")

    # Parse and compare timestamp
    try:
        if updated_at_str.endswith('Z'): updated_at_str = updated_at_str[:-1] + '+00:00'
        updated_at_dt = datetime.fromisoformat(updated_at_str)
        if updated_at_dt.tzinfo is None: updated_at_dt = updated_at_dt.replace(tzinfo=timezone.utc)
        if time_before_update.tzinfo is None: time_before_update = time_before_update.replace(tzinfo=timezone.utc)
        if time_after_update.tzinfo is None: time_after_update = time_after_update.replace(tzinfo=timezone.utc)

        assert time_before_update <= updated_at_dt <= time_after_update, \
            f"quickrecal_updated_at timestamp ({updated_at_dt}) is outside the expected update window ({time_before_update} - {time_after_update})"
        print("Timestamp is within expected range.")
    except ValueError:
        pytest.fail(f"Could not parse quickrecal_updated_at timestamp: {updated_at_str}")

    # Update metadata only, timestamp should NOT change
    await asyncio.sleep(0.1)
    print("Updating metadata only...")
    updated_meta = await memory_core.update_memory(
        memory_id=memory_id,
        updates={"metadata": {"another_field": "value"}}
    )
    assert updated_meta is True
    await asyncio.sleep(0.1) # Allow persistence
    memory_after_meta = memory_core.get_memory_by_id(memory_id)
    assert memory_after_meta.metadata.get('quickrecal_updated_at') == updated_at_str, \
        "quickrecal_updated_at changed when only metadata was updated"
    print("Verified quickrecal_updated_at unchanged after metadata-only update.")
    print("--- test_quickrecal_updated_timestamp PASSED ---")