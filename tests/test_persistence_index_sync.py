import os
import pytest
import asyncio
import shutil
import tempfile
import numpy as np
from pathlib import Path
import pytest_asyncio

# Import the classes we need to test
from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.memory_structures import MemoryEntry, MemoryAssembly
from synthians_memory_core.geometry_manager import GeometryManager

@pytest.fixture
def temp_storage_path():
    """Provide a temporary storage path for testing."""
    temp_dir = tempfile.mkdtemp(prefix="synthians_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def geometry_manager():
    """Provide a geometry manager for testing."""
    return GeometryManager({
        'embedding_dim': 768,
        'geometry_type': 'hyperbolic',
        'curvature': -1.0
    })

@pytest_asyncio.fixture
async def persistence(temp_storage_path):
    """Create a MemoryPersistence instance for testing."""
    persistence = MemoryPersistence({
        'storage_path': str(temp_storage_path)
    })
    await persistence.initialize()
    return persistence

def create_test_memory():
    """Create a test memory entry."""
    test_embedding = np.random.randn(768).astype(np.float32)  # Random embedding
    memory = MemoryEntry(
        id=f"mem:{os.urandom(4).hex()}",  # Generate a unique ID
        content="Test memory content",
        embedding=test_embedding.tolist(),
        metadata={
            "source": "test",
            "created_at": "2025-04-04T14:08:00Z"
        }
    )
    return memory

def create_test_assembly(geometry_manager):
    """Create a test assembly."""
    # Create assembly with required parameters
    assembly_id = f"asm:{os.urandom(4).hex()}"  # Generate a unique ID
    assembly = MemoryAssembly(
        geometry_manager=geometry_manager,
        assembly_id=assembly_id,
        name=f"Test Assembly {assembly_id[-6:]}",
        description="Assembly created for testing index sync"
    )
    
    # Add tags, topics
    assembly.tags = {"test", "index_sync"}
    assembly.topics = ["persistence", "testing"]
    
    # Create a composite embedding
    composite_embedding = np.random.randn(768).astype(np.float32)  # Random embedding
    # Validate embedding with geometry manager
    validated_embedding = geometry_manager._validate_vector(composite_embedding, "test")
    assembly.composite_embedding = validated_embedding
    
    # Add dummy memory IDs
    for _ in range(3):
        assembly.memories.add(f"mem:{os.urandom(4).hex()}")  
    
    return assembly

@pytest.mark.asyncio
async def test_memory_index_update_on_save(persistence):
    """Test that memory index is correctly updated when saving a memory."""
    # Create a test memory
    memory = create_test_memory()
    
    # Check initial state of index
    initial_index = persistence.memory_index.copy()
    assert memory.id not in initial_index, f"Memory {memory.id} should not be in index before saving"
    
    # Save the memory
    save_result = await persistence.save_memory(memory)
    assert save_result, "Memory save should succeed"
    
    # Check state after save
    assert memory.id in persistence.memory_index, f"Memory {memory.id} should be in index after saving"
    
    # Verify index entry fields
    index_entry = persistence.memory_index[memory.id]
    assert index_entry.get('type') == 'memory', "Index entry should have type 'memory'"
    assert index_entry.get('path') is not None, "Index entry should have a path"
    assert index_entry.get('timestamp') is not None, "Index entry should have a timestamp"

@pytest.mark.asyncio
async def test_assembly_index_update_on_save(persistence, geometry_manager):
    """Test that memory index is correctly updated when saving an assembly."""
    # Create a test assembly
    assembly = create_test_assembly(geometry_manager)
    
    # Check initial state of index
    initial_index = persistence.memory_index.copy()
    assert assembly.assembly_id not in initial_index, f"Assembly {assembly.assembly_id} should not be in index before saving"
    
    # Save the assembly
    save_result = await persistence.save_assembly(assembly, geometry_manager)
    assert save_result, "Assembly save should succeed"
    
    # Check state after save
    assert assembly.assembly_id in persistence.memory_index, f"Assembly {assembly.assembly_id} should be in index after saving"
    
    # Verify index entry fields
    index_entry = persistence.memory_index[assembly.assembly_id]
    assert index_entry.get('type') == 'assembly', "Index entry should have type 'assembly'"
    assert index_entry.get('path') is not None, "Index entry should have a path"
    assert index_entry.get('timestamp') is not None, "Index entry should have a timestamp"

@pytest.mark.asyncio
async def test_index_update_on_load_fallback(persistence, temp_storage_path, geometry_manager):
    """Test that memory index is updated when loading an item not in the index but found on disk."""
    # Create a test memory
    memory = create_test_memory()
    
    # Manually save to disk without updating index
    memories_dir = temp_storage_path / "memories"
    os.makedirs(memories_dir, exist_ok=True)
    
    # Use sanitized filename for Windows compatibility
    safe_mem_id = MemoryPersistence.sanitize_id_for_filename(memory.id)
    file_path = memories_dir / f"{safe_mem_id}.json"
    
    with open(file_path, 'w') as f:
        import json
        json.dump(memory.to_dict(), f)
    
    # Verify not in index initially
    assert memory.id not in persistence.memory_index, f"Memory {memory.id} should not be in index before loading"
    
    # Load the memory (should trigger the fallback path that updates index)
    loaded_memory = await persistence.load_memory(memory.id, geometry_manager)
    
    # Check that memory was loaded
    assert loaded_memory is not None, f"Memory {memory.id} should be loaded successfully"
    assert loaded_memory.id == memory.id, "Loaded memory should have the same ID"
    
    # Check that index was updated
    assert memory.id in persistence.memory_index, f"Memory {memory.id} should be in index after loading"
    
    # Verify index entry fields
    index_entry = persistence.memory_index[memory.id]
    assert index_entry.get('type') == 'memory', "Index entry should have type 'memory'"
    assert index_entry.get('path') is not None, "Index entry should have a path"
    assert index_entry.get('timestamp') is not None, "Index entry should have a timestamp"

@pytest.mark.asyncio
async def test_index_file_update_after_save(persistence, temp_storage_path):
    """Test that the memory_index.json file is updated after saving a memory."""
    # Create a test memory
    memory = create_test_memory()
    
    # Save the memory
    save_result = await persistence.save_memory(memory)
    assert save_result, "Memory save should succeed"
    
    # Check that the memory_index.json file exists and contains the entry
    index_path = temp_storage_path / "memory_index.json"
    assert index_path.exists(), "memory_index.json should exist after save"
    
    # Read the index file
    with open(index_path, 'r') as f:
        import json
        index_data = json.load(f)
    
    # Verify memory is in the persisted index
    assert memory.id in index_data, f"Memory {memory.id} should be in persisted index file"
    
    # Verify the stored path uses sanitized filename
    path_in_index = index_data[memory.id].get('path')
    assert path_in_index is not None, "Path should be stored in the index"
    assert ":" not in path_in_index, f"Path in index should not contain colons, got: {path_in_index}"
