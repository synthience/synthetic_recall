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
from unittest.mock import patch, AsyncMock, MagicMock
import uuid

# Core imports using proper package structure
from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.memory_structures import MemoryAssembly, MemoryEntry
from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.metrics.merge_tracker import MergeTracker
from synthians_memory_core.explainability.activation import generate_activation_explanation
from synthians_memory_core.explainability.merge import generate_merge_explanation
from synthians_memory_core.explainability.lineage import trace_lineage

# API routes for client-side testing
from fastapi.testclient import TestClient
from synthians_memory_core.api.server import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase_5_9_explainability")

# Test constants
TEST_DIR = os.path.join(os.getcwd(), 'test_phase_5_9')
EMBEDDING_DIM = 768
NUM_TEST_MEMORIES = 20
NUM_TEST_ASSEMBLIES = 5

# Helper functions
async def _remove_directory_with_retry(directory, max_attempts=3, delay=0.5):
    """Remove directory with retries to handle file locking issues."""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=False)
                if not os.path.exists(directory):
                    logger.info(f"Removed test directory: {directory}")
                    return True
                else:
                    logger.warning(f"Attempt {attempt + 1}: shutil.rmtree completed but directory still exists.")
            else:
                logger.info(f"Test directory already removed: {directory}")
                return True
        except PermissionError as e:
            logger.warning(f"Attempt {attempt + 1} failed: PermissionError removing {directory}: {e}")
        except OSError as e:
            logger.warning(f"Attempt {attempt + 1} failed: OSError removing {directory}: {e}")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: Unexpected error removing {directory}: {e}")
        
        if attempt < max_attempts - 1:
            logger.info(f"Retrying removal in {delay} seconds...")
            await asyncio.sleep(delay)
            # Increase delay for next attempt
            delay *= 2
    
    logger.error(f"ERROR: Failed to remove test directory {directory} after {max_attempts} attempts.")
    return False

def clear_test_directory():
    """Remove test directory and recreate it"""
    if os.path.exists(TEST_DIR):
        try:
            # Simple non-async removal attempt first
            shutil.rmtree(TEST_DIR)
        except (PermissionError, OSError):
            # If that fails, run the async version with retries
            logger.warning("Initial directory removal failed, trying with retry mechanism")
            asyncio.run(_remove_directory_with_retry(TEST_DIR))
            
    # Create fresh directories
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, 'stats'), exist_ok=True)
    
def create_random_embedding(dim=EMBEDDING_DIM):
    """Create a random normalized embedding"""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

def create_test_memory(idx, gm):
    """Create a test memory with random embedding"""
    memory_id = f"mem_{idx}"
    embedding = create_random_embedding()
    memory = MemoryEntry(
        id=memory_id,
        content=f"Test memory content {idx}",
        embedding=embedding,
        metadata={"test_key": f"test_value_{idx}"}
    )
    return memory

def create_test_assembly(idx, gm, memories=None):
    """Create a test assembly with provided memories"""
    assembly_id = f"asm_{idx}"
    assembly = MemoryAssembly(
        assembly_id=assembly_id,
        name=f"Test Assembly {idx}",
        description=f"Test assembly description {idx}",
        geometry_manager=gm
    )
    assembly.tags = {"test", "phase_5_9"}
    assembly.keywords = {"test", "explain", "phase_5_9"}
    
    if memories:
        for memory in memories:
            valid_embedding = gm._validate_vector(memory.embedding)
            if valid_embedding is not None:  # Check validation result
                assembly.add_memory(memory, valid_embedding)
            else:
                logger.warning(f"Skipping memory {memory.id} in test assembly {idx} due to invalid embedding.")
            
    # Make sure the assembly has a composite embedding
    if len(assembly.memories) > 0:
        # The composite embedding should be created by add_memory automatically
        # Verification check to ensure it exists
        assert assembly.composite_embedding is not None, \
            f"Composite embedding missing in {assembly.assembly_id} after adding {len(assembly.memories)} memories."
        
    return assembly

@pytest.fixture
def memory_core():
    """Create a test memory core instance with test data"""
    # Setup test environment
    clear_test_directory()
    
    # Create memory core with test config
    config = {
        'storage_path': os.path.join(TEST_DIR, 'storage'),
        'embedding_dim': EMBEDDING_DIM,
        'index_path': os.path.join(TEST_DIR, 'index'),
        'logs_path': os.path.join(TEST_DIR, 'logs'),
        'stats_path': os.path.join(TEST_DIR, 'stats'),
        'assembly_activation_threshold': 0.7,
        'assembly_boost_factor': 0.2,
        'ENABLE_EXPLAINABILITY': True,
        'merge_log_max_entries': 100,
        'max_lineage_depth': 10
    }
    
    # Create the core with mock vector index
    core = SynthiansMemoryCore(config=config)
    
    # Replace the vector_index with a mock to avoid index errors
    mock_vector_index = AsyncMock()
    # Configure async method return values
    mock_vector_index.initialize.return_value = True
    mock_vector_index.update_vector_async.return_value = True
    mock_vector_index.add_async.return_value = True  
    mock_vector_index.search_knn_async.return_value = (["test_id"], [0.9])
    # Make verify_index_integrity return the expected tuple format
    mock_vector_index.verify_index_integrity.return_value = (True, {"faiss_count": 0, "id_mapping_count": 0, "is_consistent": True})
    
    # Apply the mock
    core.vector_index = mock_vector_index
    
    # Skip initializing the core to avoid await issues with the mock
    # Instead, we'll set up the needed test data directly
    
    # Create test assemblies manually (with mocked vector functionality)
    assemblies = []
    memories_by_assembly = {}
    
    # Create memory entries for assemblies
    for i in range(NUM_TEST_ASSEMBLIES):
        # Create an assembly
        assembly_id = f"asm_{i}"
        assembly = MemoryAssembly(
            assembly_id=assembly_id,
            name=f"Test Assembly {i}",
            description=f"Test assembly description {i}",
            geometry_manager=core.geometry_manager
        )
        assembly.tags = {"test", "phase_5_9"}
        assembly.keywords = {"test", "explain", "phase_5_9"}
        
        # Create test memories for this assembly
        assembly_memories = []
        for j in range(3):  # 3 memories per assembly
            memory_id = f"mem_{i}_{j}"
            embedding = create_random_embedding()
            memory = MemoryEntry(
                id=memory_id,
                content=f"Test memory content for assembly {i}, memory {j}",
                embedding=embedding,
                metadata={"assembly": f"asm_{i}", "test_key": f"test_value_{j}"}
            )
            
            # Manually add memory to assembly
            assembly.add_memory(memory, memory.embedding)
            assembly_memories.append(memory)
        
        # Add assembly directly to core.assemblies dictionary
        core.assemblies[assembly_id] = assembly
        memories_by_assembly[assembly_id] = assembly_memories
        assemblies.append(assembly)
        
        # Save assembly to persistence
        loop = asyncio.get_event_loop()
        loop.run_until_complete(core.persistence.save_assembly(assembly, core.geometry_manager))
    
    # Create a merged assembly
    merged_id = "asm_merged"
    merged_assembly = MemoryAssembly(
        assembly_id=merged_id,
        name="Merged Test Assembly",
        description="Assembly formed by merging",
        geometry_manager=core.geometry_manager
    )
    merged_assembly.tags = {"test", "merged", "phase_5_9"}
    merged_assembly.keywords = {"test", "merged", "explain"}
    
    # Set the merged_from field to track lineage
    merged_assembly.merged_from = ["asm_0", "asm_1"]
    
    # Add memories to the merged assembly
    for i in range(2):
        # Reuse some memories from the source assemblies
        if memories_by_assembly.get(f"asm_{i}"):
            for memory in memories_by_assembly[f"asm_{i}"][:2]:  # Use first 2 memories
                merged_assembly.add_memory(memory, memory.embedding)
    
    # Add the merged assembly to core.assemblies
    core.assemblies[merged_id] = merged_assembly
    loop.run_until_complete(core.persistence.save_assembly(merged_assembly, core.geometry_manager))
    
    # Log a merge event
    merge_event_id = loop.run_until_complete(core.merge_tracker.log_merge_creation_event(
        source_assembly_ids=["asm_0", "asm_1"],
        target_assembly_id=merged_id,
        similarity_at_merge=0.85,
        merge_threshold=0.80
    ))
    
    # Log successful cleanup
    loop.run_until_complete(core.merge_tracker.log_cleanup_status_event(
        merge_event_id=merge_event_id,
        new_status="completed"
    ))
    
    # Return the initialized core with test data
    yield core
    
    # Cleanup with improved error handling
    try:
        logger.info(f"=== Cleaning up memory_core for test ====")
        # First, shutdown the core properly
        loop = asyncio.get_event_loop()
        logger.info("Shutting down memory core...")
        loop.run_until_complete(core.shutdown())
        
        # Sleep briefly to ensure resources are released
        logger.info("Waiting for resources to be released...")
        time.sleep(0.5)
        
        # Force close any open file handles
        if hasattr(core.persistence, '_close_file_handles'):
            logger.info("Explicitly closing persistence file handles...")
            loop.run_until_complete(core.persistence._close_file_handles())
            
        # Now attempt to clear the directories with the retry mechanism
        logger.info("Clearing test directory...")
        loop.run_until_complete(_remove_directory_with_retry(TEST_DIR))
        
        logger.info("Memory core cleanup completed successfully")
    except PermissionError as e:
        logger.warning(f"Permission error during teardown: {e}. Some files may still be in use.")
    except Exception as e:
        logger.error(f"Error during teardown: {e}")

@pytest.fixture
def create_merged_assembly(memory_core):
    """Create a merged assembly for testing merge explanations and lineage"""
    # Select two existing assemblies for merging
    source_ids = [f"asm_0", f"asm_1"]
    target_id = f"asm_merged"

    # Create memories for the merged assembly
    merged_memories = []
    for i in range(3):
        memory = create_test_memory(i + 100, memory_core.geometry_manager)
        loop = asyncio.get_event_loop()
        # Use process_new_memory instead of add_memory
        result = loop.run_until_complete(memory_core.process_new_memory(
            content=memory.content,
            embedding=memory.embedding,
            metadata=memory.metadata
        ))
        # Process the result which should be a MemoryEntry object
        if result and isinstance(result, MemoryEntry):
            merged_memories.append(result)  # Append the actual MemoryEntry object
        else:
            pytest.fail(f"Failed to process memory {i} in create_merged_assembly setup")
    
    # Create the merged assembly with MemoryEntry objects
    merged_assembly = create_test_assembly("merged", memory_core.geometry_manager, merged_memories)
    merged_assembly.merged_from = source_ids.copy()  # Set the merged_from field
    
    # Save the merged assembly to persistence
    loop = asyncio.get_event_loop()
    loop.run_until_complete(memory_core.persistence.save_assembly(merged_assembly, memory_core.geometry_manager))
    
    # Retrieve the merged assembly ID for tracing and verification
    memory_core.assemblies[merged_assembly.assembly_id] = merged_assembly
    
    # Log a merge event for testing merge explanations
    merge_event_id = loop.run_until_complete(memory_core.merge_tracker.log_merge_creation_event(
        source_assembly_ids=source_ids,
        target_assembly_id=merged_assembly.assembly_id,
        similarity_at_merge=0.85,
        merge_threshold=0.80
    ))
    
    # Log a successful cleanup
    loop.run_until_complete(memory_core.merge_tracker.log_cleanup_status_event(
        merge_event_id=merge_event_id,
        new_status="completed"
    ))
    
    return merged_assembly.assembly_id, merge_event_id

@pytest.fixture
def test_client(memory_core):
    """Create a test client with explainability features enabled."""
    # Ensure explainability is enabled for tests
    memory_core.config["ENABLE_EXPLAINABILITY"] = True
    
    # Create a new app instance with routers mounted
    from synthians_memory_core.api.server import app
    from fastapi.testclient import TestClient
    from synthians_memory_core.api.explainability_routes import router as explainability_router
    from synthians_memory_core.api.diagnostics_routes import router as diagnostics_router
    
    # Set the memory_core in the app state
    app.state.memory_core = memory_core
    
    # Mount the routers manually to ensure they're available
    app.include_router(explainability_router)
    app.include_router(diagnostics_router)
    
    # Create and return the test client
    client = TestClient(app)
    return client

# ---- CORE FUNCTION UNIT TESTS ----

def test_activation_explanation(memory_core):
    """Test that activation explanation works correctly"""
    # We'll use the first assembly and memory we find for the test
    assemblies = memory_core.assemblies
    
    assert len(assemblies) > 0, "No assemblies found"
    assembly_id = next(iter(assemblies.keys()))
    memory_id = None
    
    assembly = assemblies[assembly_id]
    assert assembly is not None, "Test assembly not found"
    
    if assembly.memories:
        memory_id = next(iter(assembly.memories))
    
    assert memory_id is not None, "No memories found in test assembly"
    
    # Generate activation explanation
    loop = asyncio.get_event_loop()
    explanation = loop.run_until_complete(generate_activation_explanation(
        assembly_id=assembly_id,
        memory_id=memory_id,
        trigger_context="test_context_activation",
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager,
        config=memory_core.config
    ))
    
    # Verify the explanation contains the expected fields
    assert explanation is not None
    assert explanation["assembly_id"] == assembly_id
    assert explanation["memory_id"] == memory_id
    assert "calculated_similarity" in explanation
    assert "activation_threshold" in explanation
    assert "passed_threshold" in explanation
    assert "check_timestamp" in explanation

def test_merge_explanation(memory_core, create_merged_assembly):
    """Test that merge explanation works correctly"""
    merged_id, merge_event_id = create_merged_assembly
    
    # Generate merge explanation
    loop = asyncio.get_event_loop()
    explanation = loop.run_until_complete(generate_merge_explanation(
        assembly_id=merged_id,
        merge_tracker=memory_core.merge_tracker,
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager
    ))
    
    # Verify the explanation contains the expected fields
    assert explanation is not None
    assert explanation["target_assembly_id"] == merged_id
    assert explanation["merge_event_id"] is not None
    assert explanation["source_assembly_ids"] is not None
    assert len(explanation["source_assembly_ids"]) > 0
    assert explanation["similarity_at_merge"] is not None
    assert explanation["threshold_at_merge"] is not None
    assert explanation["reconciled_cleanup_status"] == "completed"

def test_lineage_tracing(memory_core, create_merged_assembly):
    """Test that lineage tracing works correctly"""
    merged_id, _ = create_merged_assembly
    
    # Trace lineage
    loop = asyncio.get_event_loop()
    lineage = loop.run_until_complete(trace_lineage(
        assembly_id=merged_id,
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager,
        max_depth=5
    ))
    
    # Verify the lineage contains the expected entries
    assert lineage is not None
    assert len(lineage) > 0
    
    # Check the root node
    root = next((entry for entry in lineage if entry["depth"] == 0), None)
    assert root is not None
    assert root["assembly_id"] == merged_id
    
    # Check that source assemblies are included
    source_assemblies = [entry for entry in lineage if entry["depth"] == 1]
    assert len(source_assemblies) > 0

def test_merge_log_reconciliation(memory_core, create_merged_assembly):
    """Test that merge log reconciliation works correctly"""
    _, merge_event_id = create_merged_assembly
    
    # Get reconciled merge events
    loop = asyncio.get_event_loop()
    reconciled_events = loop.run_until_complete(memory_core.merge_tracker.reconcile_merge_events(limit=10))
    
    # Verify reconciled events
    assert reconciled_events is not None
    assert len(reconciled_events) > 0
    
    # Find our test event
    test_event = next((event for event in reconciled_events 
                       if event["merge_event_id"] == merge_event_id), None)
    
    assert test_event is not None
    assert test_event["final_cleanup_status"] == "completed"

def test_runtime_config_sanitization(memory_core):
    """Test that runtime configuration is properly sanitized"""
    # Add some sensitive keys to config
    memory_core.config["SECRET_KEY"] = "super_secret"
    memory_core.config["DB_PASSWORD"] = "db_password"
    memory_core.config["embedding_dim"] = EMBEDDING_DIM
    
    # Define safe keys (same as in diagnostics_routes.py)
    safe_keys = [
        "embedding_dim", "assembly_activation_threshold",
        "assembly_boost_factor", "ENABLE_EXPLAINABILITY",
        "merge_log_max_entries", "max_lineage_depth"
    ]
    
    # Get sanitized config (manually implementing similar logic to the route)
    sanitized_config = {k: v for k, v in memory_core.config.items() if k in safe_keys}
    
    # Verify sanitization
    assert "embedding_dim" in sanitized_config
    assert "SECRET_KEY" not in sanitized_config
    assert "DB_PASSWORD" not in sanitized_config

# ---- API ENDPOINT INTEGRATION TESTS ----

def test_explain_activation_endpoint(test_client, memory_core):
    """Test the explain activation endpoint"""
    # Get test data
    assembly = next(iter(memory_core.assemblies.values()))
    memory = next(iter(assembly.memories))
    assembly.memory_activation_reason = {memory.id: "Test activation reason"}
    
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(memory_core.persistence.save_assembly(assembly, memory_core.geometry_manager))

    # Construct URL with memory_id as a string parameter
    memory_id_str = memory.id # Get the string ID
    url = f"/assemblies/{assembly.assembly_id}/explain_activation?memory_id={memory_id_str}"
    # Make the request
    response = test_client.get(url)

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    explanation = data["explanation"]
    assert explanation["target_assembly_id"] == assembly.assembly_id
    assert explanation["memory_id"] == memory.id

def test_explain_merge_endpoint(test_client, memory_core, create_merged_assembly):
    """Test the explain merge endpoint"""
    # Get the merged assembly ID from the fixture
    merged_id, _ = create_merged_assembly
    
    # Test the endpoint
    response = test_client.get(f"/assemblies/{merged_id}/explain_merge")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    explanation = data["explanation"]
    assert explanation["target_assembly_id"] == merged_id

def test_lineage_endpoint(test_client, memory_core, create_merged_assembly):
    """Test the lineage endpoint"""
    # Get the merged assembly ID from the fixture
    merged_id, _ = create_merged_assembly
    
    # Test the endpoint
    response = test_client.get(f"/assemblies/{merged_id}/lineage?max_depth=5")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["target_assembly_id"] == merged_id
    assert len(data["lineage"]) > 0
    
    # Test caching by calling again
    response2 = test_client.get(f"/assemblies/{merged_id}/lineage?max_depth=5")
    assert response2.status_code == 200

def test_merge_log_endpoint(test_client, memory_core):
    """Test the merge log endpoint"""
    # Test the endpoint
    response = test_client.get("/diagnostics/merge_log?limit=10")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "reconciled_log_entries" in data
    assert isinstance(data["reconciled_log_entries"], list)
    assert data["count"] == len(data["reconciled_log_entries"])

def test_runtime_config_endpoint(test_client, memory_core):
    """Test the runtime config endpoint"""
    # Add a test key to config
    memory_core.config["embedding_dim"] = 512
    memory_core.config["SECRET_KEY"] = "this_should_not_appear"
    
    # Test the endpoint
    response = test_client.get("/diagnostics/runtime/config/memory-core")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "config" in data
    assert "embedding_dim" in data["config"]
    # Sensitive data should be filtered out
    assert "SECRET_KEY" not in data["config"]

def test_lineage_max_depth_limiting(test_client, memory_core, create_merged_assembly):
    """Test the lineage endpoint properly limits depth and sets the max_depth_reached flag"""
    # Get the merged assembly ID from the fixture (not used directly, but fixture creates base assemblies)
    merged_id, _ = create_merged_assembly
    
    # Create a deeper lineage chain artificially
    depth_chain = ["asm_0", "asm_1", "asm_2", "asm_3", "asm_4"] # IDs of existing assemblies
    
    # Verify all assemblies exist before proceeding
    for asm_id in depth_chain:
        assert asm_id in memory_core.assemblies, f"Assembly {asm_id} not found in memory_core. Available: {list(memory_core.assemblies.keys())}"
    
    loop = asyncio.get_event_loop() # Get loop once
    
    # Set up a chain of merges and persist them
    for i in range(len(depth_chain) - 1):
        parent_id = depth_chain[i]
        child_id = depth_chain[i+1]
        
        # Get the assembly
        assembly = memory_core.assemblies[parent_id]
        
        # Set merged_from to create the lineage chain
        assembly.merged_from = [child_id]
        logger.info(f"Setting assembly {parent_id}.merged_from = [{child_id}]")
        
        # Save each modified assembly back to persistence
        loop.run_until_complete(memory_core.persistence.save_assembly(assembly, memory_core.geometry_manager))
        logger.info(f"Saved modified assembly {assembly.assembly_id} with merged_from={assembly.merged_from}")
        
        # Double-check that the assembly was saved correctly by loading it back from disk
        saved_assembly = loop.run_until_complete(memory_core.persistence.load_assembly(parent_id, memory_core.geometry_manager))
        assert saved_assembly.merged_from == [child_id], f"Assembly {parent_id} was not saved correctly. Expected merged_from=[{child_id}], got {saved_assembly.merged_from}"
    
    # Set a very low max_depth to ensure we hit the limit
    max_depth = 1
    
    # First, test the direct function call to verify it works
    direct_lineage = loop.run_until_complete(
        trace_lineage(
            assembly_id=depth_chain[0],
            persistence=memory_core.persistence,
            geometry_manager=memory_core.geometry_manager,
            max_depth=max_depth
        )
    )
    
    # Verify max depth limiting in direct function call
    direct_max_depth_reached = any(entry.get("status") == "depth_limit_reached" for entry in direct_lineage)
    logger.info(f"Direct trace - max_depth_reached: {direct_max_depth_reached}")
    logger.info(f"Direct lineage entries: {json.dumps(direct_lineage, indent=2)}")
    assert direct_max_depth_reached, "Max depth limiting was not applied in direct function call"
    
    # Now test the API endpoint
    response = test_client.get(f"/assemblies/{depth_chain[0]}/lineage?max_depth={max_depth}")
    
    # Verify API response
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    
    # Log API response for debugging
    logger.info(f"API response: {json.dumps(data, indent=2)}")
    
    # Check that max_depth_reached flag is set to True in API response
    assert data["max_depth_reached"] is True, "Max depth limiting was not applied in API response"
    
    # Verify there's at least one entry with status="depth_limit_reached" in API response
    depth_limited_entries = [entry for entry in data["lineage"] if entry.get("status") == "depth_limit_reached"]
    assert len(depth_limited_entries) > 0, "No entries marked with depth_limit_reached status"

# ---- EDGE CASES AND ERROR HANDLING TESTS ----

def test_activation_explanation_nonexistent_assembly(memory_core):
    """Test activation explanation with a nonexistent assembly"""
    loop = asyncio.get_event_loop()
    explanation = loop.run_until_complete(generate_activation_explanation(
        assembly_id="nonexistent_assembly",
        memory_id="mem_0",
        trigger_context="test_context_nonexistent",
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager,
        config=memory_core.config
    ))
    
    assert explanation is not None
    assert "notes" in explanation
    assert "not found" in explanation["notes"]

def test_merge_explanation_nonmerged_assembly(memory_core):
    """Test merge explanation with an assembly not formed through merging"""
    # Use a regular assembly that wasn't created through merging
    assembly_id = "asm_0"
    
    loop = asyncio.get_event_loop()
    explanation = loop.run_until_complete(generate_merge_explanation(
        assembly_id=assembly_id,
        merge_tracker=memory_core.merge_tracker,
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager
    ))
    
    assert explanation is not None
    assert "notes" in explanation
    assert "not formed by a merge" in explanation["notes"]

def test_lineage_with_cycles(memory_core):
    """Test lineage tracing with potential cycles"""
    # Create a cycle in merged_from (this is an artificial test case)
    assembly_id = "asm_0"
    assembly = memory_core.assemblies[assembly_id] if assembly_id in memory_core.assemblies else None
    
    if assembly:
        # Create a fake cycle - asm_0 -> asm_1 -> asm_0
        assembly.merged_from = ["asm_1"]
        
        # Save the modified assembly to persistence
        loop = asyncio.get_event_loop()
        loop.run_until_complete(memory_core.persistence.save_assembly(assembly, memory_core.geometry_manager))
        
        # Add the cycle from asm_1 back to asm_0
        if "asm_1" in memory_core.assemblies:
            memory_core.assemblies["asm_1"].merged_from = [assembly_id]
            loop.run_until_complete(memory_core.persistence.save_assembly(memory_core.assemblies["asm_1"], memory_core.geometry_manager))
            
            # Now trace the lineage
            lineage = loop.run_until_complete(trace_lineage(
                assembly_id=assembly_id,
                persistence=memory_core.persistence,
                geometry_manager=memory_core.geometry_manager,
                max_depth=10
            ))
            
            # Verify cycle detection
            cycle_detected = any(entry["status"] == "cycle_detected" for entry in lineage)
            assert cycle_detected, "Cycle was not detected in lineage. Lineage entries: " + str(lineage)

def test_lineage_max_depth_limiting(memory_core, create_merged_assembly):
    """Test that lineage tracing respects max_depth"""
    merged_id, _ = create_merged_assembly
    
    # Create a deeper lineage chain artificially
    depth_chain = ["asm_0", "asm_1", "asm_2", "asm_3", "asm_4"]
    
    # Set up a chain of merges
    for i in range(len(depth_chain) - 1):
        assembly = memory_core.assemblies[depth_chain[i]] if depth_chain[i] in memory_core.assemblies else None
        if assembly:
            assembly.merged_from = [depth_chain[i+1]]
            # CRITICAL: Persist the modified assembly to storage
            logger.info(f"Persisting assembly {depth_chain[i]} with merged_from={assembly.merged_from}")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(memory_core.persistence.save_assembly(assembly, memory_core.geometry_manager))
            
    # Verify the lineage chain was properly saved
    for i in range(len(depth_chain) - 1):
        # Load from persistence to verify
        saved_assembly = loop.run_until_complete(
            memory_core.persistence.load_assembly(
                depth_chain[i],
                memory_core.geometry_manager
            )
        )
        logger.info(f"Verified assembly {depth_chain[i]} has merged_from={saved_assembly.merged_from}")
        assert saved_assembly.merged_from == [depth_chain[i+1]], f"Assembly {depth_chain[i]} merged_from not saved correctly"

    # Test with low max_depth
    loop = asyncio.get_event_loop()
    lineage = loop.run_until_complete(trace_lineage(
        assembly_id=depth_chain[0],
        persistence=memory_core.persistence,
        geometry_manager=memory_core.geometry_manager,
        max_depth=1  # Set a low max depth
    ))
    
    # Verify max depth limiting
    max_depth_reached = any(entry["status"] == "depth_limit_reached" for entry in lineage)
    assert max_depth_reached, "Max depth limiting was not applied"

def test_feature_flag_disabling(test_client, memory_core):
    """Test that explainability feature flag disables endpoints"""
    # Temporarily disable explainability
    orig_config = memory_core.config.get("ENABLE_EXPLAINABILITY")
    memory_core.config["ENABLE_EXPLAINABILITY"] = False
    
    try:
        # Test an endpoint
        response = test_client.get("/diagnostics/merge_log")
        
        # Should be forbidden when disabled
        assert response.status_code == 403
        
        # Try another endpoint
        response = test_client.get("/assemblies/asm_0/explain_activation?memory_id=mem_0")
        assert response.status_code == 403
    finally:
        # Restore original config
        memory_core.config["ENABLE_EXPLAINABILITY"] = orig_config

# Run tests directly for debugging
if __name__ == "__main__":
    """Run the tests directly for debugging"""
    asyncio.run(test_activation_explanation(None))
    asyncio.run(test_merge_explanation(None, None))
    asyncio.run(test_lineage_tracing(None, None))
