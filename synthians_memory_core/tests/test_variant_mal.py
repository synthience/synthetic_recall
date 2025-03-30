# tests/test_variant_mal.py

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
if CURRENT_VARIANT != 'MAL':
    pytest.skip(f"Skipping MAL tests since current variant is {CURRENT_VARIANT}", allow_module_level=True)

# Test functions specifically for MAL variant
@pytest.mark.asyncio
async def test_mal_variant_memory_processing(api_clients):
    """Test MAL variant's basic memory processing capabilities."""
    session, mc_client = api_clients
    
    # 1. Create test memories
    test_content = f"This is a MAL variant test memory created at {time.time()}"
    memory_ids = await create_test_memories(mc_client, count=3, 
                                          prefix=f"MAL-Variant-Test")
    
    # 2. Wait briefly for asynchronous processing
    await asyncio.sleep(1)  # Allow processing to complete
    
    # 3. Call CCE to process a related memory through the MAL variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": f"This is a follow-up to the MAL-Variant-Test memories",
            "embedding": [float(i) / 100 for i in range(384)],  # Simple test embedding
            "metadata": {
                "source": "variant_test",
                "variant": "MAL"
            }
        }
    ) as response:
        assert response.status == 200, f"Failed to process memory via CCE: {await response.text()}"
        result = await response.json()
        assert "memory_id" in result, "No memory_id in response"
        cce_memory_id = result["memory_id"]
    
    # 4. Wait for CCE to process the memory (MAL model needs time for latent memory processing)
    await asyncio.sleep(3)  # Allow sufficient time for MAL processing
    
    # 5. Verify the CCE-processed memory exists in Memory Core
    retrieved_memory = await mc_client.get_memory(cce_memory_id)
    assert retrieved_memory is not None, f"Could not retrieve memory {cce_memory_id}"
    assert "metadata" in retrieved_memory, "No metadata in retrieved memory"
    
    # MAL specific: Memory should have been processed by MAL variant
    # The processing_info is deeply nested, so we need to handle it carefully
    metadata = retrieved_memory.get("metadata", {})
    processing_info = metadata.get("processing_info", {})
    
    # Check for MAL-specific indicators in the memory
    # Note: The exact structure depends on your implementation
    assert processing_info.get("variant") == "MAL" or \
           processing_info.get("titans_variant") == "MAL" or \
           metadata.get("titans_variant") == "MAL", \
           f"Memory not processed by MAL variant: {metadata}"
    
    # Clean up test memories
    for memory_id in memory_ids + [cce_memory_id]:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mal_variant_retrieval(api_clients):
    """Test MAL variant's unique latent memory retrieval behavior."""
    session, mc_client = api_clients
    
    # 1. Create semantically related memories for testing MAL's latent connecting abilities
    memory_contents = [
        "Quantum computing uses qubits instead of classical bits",
        "Superposition allows qubits to be in multiple states simultaneously",
        "Quantum entanglement is a phenomenon where particles become correlated",
        "Einstein called quantum entanglement 'spooky action at a distance'",
        "Richard Feynman was a pioneer in quantum electrodynamics"
    ]
    
    memory_ids = []
    for i, content in enumerate(memory_contents):
        memory_entry = {
            "content": content,
            "embedding": [float(j) / 100 for j in range(384)],  # Simple test embedding
            "metadata": {
                "source": "mal_variant_test",
                "test_id": i,
                "variant": "MAL"
            }
        }
        
        result = await mc_client.process_memory(memory_entry)
        memory_ids.append(result["memory_id"])
    
    # 2. Wait for processing - MAL needs time to develop latent connections
    await asyncio.sleep(3)
    
    # 3. Query with a term related to but not explicitly mentioned in our memories
    query = "What did Einstein think about quantum physics?"
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": query,
            "max_memories": 3
        }
    ) as response:
        assert response.status == 200, f"Failed to retrieve memories: {await response.text()}"
        result = await response.json()
        
        # 4. Verify MAL-specific retrieval behavior (latent connections)
        assert "memories" in result, "No memories in response"
        assert len(result["memories"]) > 0, "No memories retrieved"
        
        # MAL should find the Einstein reference through latent connections
        found_einstein = False
        for memory in result["memories"]:
            if "einstein" in memory["content"].lower():
                found_einstein = True
                break
        
        assert found_einstein, "MAL variant didn't retrieve Einstein-related memory through latent connections"
    
    # 5. Try another query that should benefit from MAL's latent space
    query = "Tell me about quantum phenomena"
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": query,
            "max_memories": 3
        }
    ) as response:
        assert response.status == 200
        result = await response.json()
        
        # Should retrieve memories about superposition or entanglement
        found_quantum_phenomenon = False
        for memory in result.get("memories", []):
            content = memory["content"].lower()
            if "superposition" in content or "entanglement" in content:
                found_quantum_phenomenon = True
                break
                
        assert found_quantum_phenomenon, "MAL didn't retrieve appropriate quantum phenomena memories"
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mal_variant_state(api_clients):
    """Test MAL variant state management and internal memory structure."""
    session, mc_client = api_clients
    
    # 1. Check Neural Memory server state to confirm MAL model is active
    async with session.get("http://localhost:8001/status") as response:
        assert response.status == 200, "Failed to get Neural Memory status"
        nm_status = await response.json()
        
        # The status response format depends on your implementation
        # Look for MAL-specific indicators
        assert "model_info" in nm_status, "No model info in Neural Memory status"
        model_info = nm_status["model_info"]
        
        # Verify it's running MAL variant
        if "variant" in model_info:
            assert model_info["variant"] == "MAL", f"Wrong variant: {model_info['variant']}"
        elif "architecture" in model_info:
            assert "MAL" in model_info["architecture"], f"MAL not in architecture: {model_info['architecture']}"
    
    # 2. Check context-cascade-engine status
    async with session.get("http://localhost:8002/status") as response:
        assert response.status == 200, "Failed to get CCE status"
        cce_status = await response.json()
        
        # Verify CCE is also configured for MAL
        # The exact path depends on your CCE status response format
        titan_config = cce_status.get("config", {}).get("titan", {})
        if titan_config:
            assert titan_config.get("variant") == "MAL" or \
                   titan_config.get("titans_variant") == "MAL", \
                   f"CCE not configured for MAL: {titan_config}"
        
        # Alternative check locations depending on implementation
        titans_variant = cce_status.get("titans_variant") or \
                        cce_status.get("config", {}).get("titans_variant")
        if titans_variant is not None:
            assert titans_variant == "MAL", f"Wrong CCE variant: {titans_variant}"

@pytest.mark.asyncio
async def test_mal_latent_memory_formation(api_clients):
    """Test MAL's ability to form latent memories from sequential inputs."""
    session, mc_client = api_clients
    
    # MAL variant is expected to develop latent representations
    # when processing a sequence of related memories
    
    # 1. Create a sequence of related but indirect memories
    test_sequence = [
        "Machine learning models require training data.",
        "Large datasets help improve model accuracy.",
        "Data preprocessing is an important step in machine learning.",
        "Feature engineering can significantly impact model performance.",
        "Hyperparameter tuning optimizes model configuration."
    ]
    
    # Process these in sequence through CCE to allow MAL to build latent space
    memory_ids = []
    for content in test_sequence:
        async with session.post(
            "http://localhost:8002/process_memory",
            json={
                "content": content,
                "metadata": {"test": "mal_latent_formation"}
            }
        ) as response:
            result = await response.json()
            memory_ids.append(result["memory_id"])
        # MAL may need more time between memories to form latent connections
        await asyncio.sleep(1)
    
    # 2. Allow processing to complete and latent space to develop
    await asyncio.sleep(5)
    
    # 3. Query with a concept not directly mentioned but latently related
    query = "How can we improve AI models?"
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": query,
            "max_memories": 3
        }
    ) as response:
        result = await response.json()
        memories = result.get("memories", [])
        
        # 4. MAL should retrieve memories about data, preprocessing, or tuning
        # even though "AI models" wasn't explicitly mentioned
        assert len(memories) > 0, "MAL variant didn't retrieve any memories"
        
        # Check if retrieved memories are relevant to improving models
        relevant_count = 0
        for memory in memories:
            content = memory["content"].lower()
            if any(term in content for term in ["data", "accuracy", "performance", "tuning", "preprocessing"]):
                relevant_count += 1
        
        # At least one memory should be relevant through latent connections
        assert relevant_count > 0, "MAL didn't form effective latent connections"
    
    # 5. Test Memory Core can directly retrieve the memories we created
    retrieved = await mc_client.retrieve_memories(
        query="Tell me about machine learning",
        max_memories=5
    )
    assert len(retrieved["memories"]) > 0, "No memories retrieved directly from Memory Core"
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)
