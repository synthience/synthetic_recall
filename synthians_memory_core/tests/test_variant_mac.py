# tests/test_variant_mac.py

import pytest
import pytest_asyncio
import asyncio
import json
import os
import time
from typing import Dict, List, Any

# Import our variant testing fixtures
from variant_conftest import api_clients, create_test_memories

# Get current variant for logging and assertions
CURRENT_VARIANT = os.environ.get('TITANS_VARIANT', 'UNKNOWN')
if CURRENT_VARIANT != 'MAC':
    pytest.skip(f"Skipping MAC tests since current variant is {CURRENT_VARIANT}", allow_module_level=True)

# Test functions specifically for MAC variant
@pytest.mark.asyncio
async def test_mac_variant_memory_processing(api_clients):
    """Test MAC variant's basic memory processing capabilities."""
    session, mc_client = api_clients
    
    # 1. Create test memories
    test_content = f"This is a MAC variant test memory created at {time.time()}"
    memory_ids = await create_test_memories(mc_client, count=3, 
                                          prefix=f"MAC-Variant-Test")
    
    # 2. Wait briefly for asynchronous processing
    await asyncio.sleep(1)  # Allow processing to complete
    
    # 3. Call CCE to process a related memory through the MAC variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": f"This is a follow-up to the MAC-Variant-Test memories",
            "embedding": [float(i) / 100 for i in range(384)],  # Simple test embedding
            "metadata": {
                "source": "variant_test",
                "variant": "MAC"
            }
        }
    ) as response:
        assert response.status == 200, f"Failed to process memory via CCE: {await response.text()}"
        result = await response.json()
        assert "memory_id" in result, "No memory_id in response"
        cce_memory_id = result["memory_id"]
    
    # 4. Wait for CCE to process the memory (MAC model needs time to process)
    await asyncio.sleep(3)  # Allow sufficient time for MAC processing
    
    # 5. Verify the CCE-processed memory exists in Memory Core
    retrieved_memory = await mc_client.get_memory(cce_memory_id)
    assert retrieved_memory is not None, f"Could not retrieve memory {cce_memory_id}"
    assert "metadata" in retrieved_memory, "No metadata in retrieved memory"
    
    # MAC specific: Memory should have been processed by MAC variant
    # The processing_info is deeply nested, so we need to handle it carefully
    metadata = retrieved_memory.get("metadata", {})
    processing_info = metadata.get("processing_info", {})
    
    # Check for MAC-specific indicators in the memory
    # Note: The exact structure depends on your implementation
    assert processing_info.get("variant") == "MAC" or \
           processing_info.get("titans_variant") == "MAC" or \
           metadata.get("titans_variant") == "MAC", \
           f"Memory not processed by MAC variant: {metadata}"
    
    # Clean up test memories
    for memory_id in memory_ids + [cce_memory_id]:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mac_variant_retrieval(api_clients):
    """Test MAC variant's retrieval behavior."""
    session, mc_client = api_clients
    
    # 1. Create a series of memories with known semantic relationships
    memory_contents = [
        "Artificial intelligence models require large datasets for training",
        "Neural networks have many interconnected layers of neurons", 
        "Deep learning systems process information similarly to the human brain",
        "Machine learning algorithms improve with more training data",
        "Gradient descent is used to optimize neural network weights"
    ]
    
    memory_ids = []
    for i, content in enumerate(memory_contents):
        memory_entry = {
            "content": content,
            "embedding": [float(j) / 100 for j in range(384)],  # Simple test embedding
            "metadata": {
                "source": "mac_variant_test",
                "test_id": i,
                "variant": "MAC"
            }
        }
        
        result = await mc_client.process_memory(memory_entry)
        memory_ids.append(result["memory_id"])
    
    # 2. Wait for processing
    await asyncio.sleep(1)
    
    # 3. Query through CCE with MAC variant
    query = "How do neural networks process information?"
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": query,
            "max_memories": 3
        }
    ) as response:
        assert response.status == 200, f"Failed to retrieve memories: {await response.text()}"
        result = await response.json()
        
        # 4. Verify MAC-specific retrieval behavior
        # MAC should have specific associative characteristics
        assert "memories" in result, "No memories in response"
        assert len(result["memories"]) > 0, "No memories retrieved"
        
        # Look for memories that mention neural networks
        found_neural = False
        for memory in result["memories"]:
            if "neural" in memory["content"].lower():
                found_neural = True
                break
        
        assert found_neural, "MAC variant did not retrieve relevant neural network memories"
    
    # 5. Verify Memory Core can directly retrieve our test memories
    retrieved = await mc_client.retrieve_memories(
        query=query,
        max_memories=5
    )
    assert len(retrieved["memories"]) > 0, "No memories retrieved directly from Memory Core"
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mac_variant_state(api_clients):
    """Test MAC variant state management and internal memory structure."""
    session, mc_client = api_clients
    
    # 1. Check Neural Memory server state to confirm MAC model is active
    async with session.get("http://localhost:8001/status") as response:
        assert response.status == 200, "Failed to get Neural Memory status"
        nm_status = await response.json()
        
        # The status response format depends on your implementation
        # Look for MAC-specific indicators
        assert "model_info" in nm_status, "No model info in Neural Memory status"
        model_info = nm_status["model_info"]
        
        # Verify it's running MAC variant
        if "variant" in model_info:
            assert model_info["variant"] == "MAC", f"Wrong variant: {model_info['variant']}"
        elif "architecture" in model_info:
            assert "MAC" in model_info["architecture"], f"MAC not in architecture: {model_info['architecture']}"
    
    # 2. Check context-cascade-engine status
    async with session.get("http://localhost:8002/status") as response:
        assert response.status == 200, "Failed to get CCE status"
        cce_status = await response.json()
        
        # Verify CCE is also configured for MAC
        # The exact path depends on your CCE status response format
        titan_config = cce_status.get("config", {}).get("titan", {})
        if titan_config:
            assert titan_config.get("variant") == "MAC" or \
                   titan_config.get("titans_variant") == "MAC", \
                   f"CCE not configured for MAC: {titan_config}"
        
        # Alternative check locations depending on implementation
        titans_variant = cce_status.get("titans_variant") or \
                        cce_status.get("config", {}).get("titans_variant")
        if titans_variant is not None:
            assert titans_variant == "MAC", f"Wrong CCE variant: {titans_variant}"

@pytest.mark.asyncio
async def test_mac_memory_characteristics(api_clients):
    """Test MAC-specific memory characteristics and behaviors."""
    session, mc_client = api_clients
    
    # MAC variant is expected to have specific characteristics:
    # 1. It operates more like traditional associative memory
    # 2. Its QuickRecall values may differ from other variants
    # 3. Its retrievals should show specific patterns
    
    # Create a memory sequence to test associations
    test_sequence = [
        "The capital of France is Paris.",
        "Paris is known for the Eiffel Tower.",
        "The Eiffel Tower was built in 1889.",
        "The year 1889 was in the 19th century."
    ]
    
    # Process these in sequence through CCE
    memory_ids = []
    for content in test_sequence:
        async with session.post(
            "http://localhost:8002/process_memory",
            json={
                "content": content,
                "metadata": {"test": "mac_characteristics"}
            }
        ) as response:
            result = await response.json()
            memory_ids.append(result["memory_id"])
        # Brief pause between memories to ensure sequential processing
        await asyncio.sleep(0.5)
    
    # Allow processing to complete
    await asyncio.sleep(2)
    
    # Test associative retrieval with CCE
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": "What is Paris known for?",
            "max_memories": 2
        }
    ) as response:
        result = await response.json()
        memories = result.get("memories", [])
        
        # MAC should find related memories about Paris
        paris_memory = next((m for m in memories if "paris" in m["content"].lower()), None)
        assert paris_memory is not None, "MAC variant didn't retrieve Paris-related memory"
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)
