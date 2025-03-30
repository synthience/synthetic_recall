# tests/test_variant_mag.py

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
if CURRENT_VARIANT != 'MAG':
    pytest.skip(f"Skipping MAG tests since current variant is {CURRENT_VARIANT}", allow_module_level=True)

# Test functions specifically for MAG variant
@pytest.mark.asyncio
async def test_mag_variant_memory_processing(api_clients):
    """Test MAG variant's basic memory processing capabilities."""
    session, mc_client = api_clients
    
    # 1. Create test memories
    test_content = f"This is a MAG variant test memory created at {time.time()}"
    memory_ids = await create_test_memories(mc_client, count=3, 
                                          prefix=f"MAG-Variant-Test")
    
    # 2. Wait briefly for asynchronous processing
    await asyncio.sleep(1)  # Allow processing to complete
    
    # 3. Call CCE to process a related memory through the MAG variant
    async with session.post(
        "http://localhost:8002/process_memory",
        json={
            "content": f"This is a follow-up to the MAG-Variant-Test memories",
            "embedding": [float(i) / 100 for i in range(384)],  # Simple test embedding
            "metadata": {
                "source": "variant_test",
                "variant": "MAG"
            }
        }
    ) as response:
        assert response.status == 200, f"Failed to process memory via CCE: {await response.text()}"
        result = await response.json()
        assert "memory_id" in result, "No memory_id in response"
        cce_memory_id = result["memory_id"]
    
    # 4. Wait for CCE to process the memory (MAG model might need more time for gating)
    await asyncio.sleep(3)  # Allow sufficient time for MAG processing
    
    # 5. Verify the CCE-processed memory exists in Memory Core
    retrieved_memory = await mc_client.get_memory(cce_memory_id)
    assert retrieved_memory is not None, f"Could not retrieve memory {cce_memory_id}"
    assert "metadata" in retrieved_memory, "No metadata in retrieved memory"
    
    # MAG specific: Memory should have been processed by MAG variant
    # The processing_info is deeply nested, so we need to handle it carefully
    metadata = retrieved_memory.get("metadata", {})
    processing_info = metadata.get("processing_info", {})
    
    # Check for MAG-specific indicators in the memory
    # Note: The exact structure depends on your implementation
    assert processing_info.get("variant") == "MAG" or \
           processing_info.get("titans_variant") == "MAG" or \
           metadata.get("titans_variant") == "MAG", \
           f"Memory not processed by MAG variant: {metadata}"
    
    # Clean up test memories
    for memory_id in memory_ids + [cce_memory_id]:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mag_variant_retrieval(api_clients):
    """Test MAG variant's retrieval behavior with gating characteristics."""
    session, mc_client = api_clients
    
    # 1. Create memories for testing MAG's gating behavior
    # Include some high emotional memories and some neutral ones
    memory_contents = [
        {"content": "Today was an amazing day with perfect weather!", "emotion": "joy", "intensity": 0.9},
        {"content": "I learned about neural network architecture today", "emotion": "neutral", "intensity": 0.2},
        {"content": "The accident on the highway was terrible", "emotion": "sadness", "intensity": 0.8},
        {"content": "The conference presentation was informative", "emotion": "neutral", "intensity": 0.3},
        {"content": "I'm extremely frustrated with the software bugs", "emotion": "anger", "intensity": 0.75}
    ]
    
    memory_ids = []
    for i, memory_data in enumerate(memory_contents):
        memory_entry = {
            "content": memory_data["content"],
            "embedding": [float(j) / 100 for j in range(384)],  # Simple test embedding
            "metadata": {
                "source": "mag_variant_test",
                "test_id": i,
                "variant": "MAG",
                "dominant_emotion": memory_data["emotion"],
                "emotion_intensity": memory_data["intensity"]
            }
        }
        
        result = await mc_client.process_memory(memory_entry)
        memory_ids.append(result["memory_id"])
    
    # 2. Wait for processing
    await asyncio.sleep(1)
    
    # 3. Query through CCE with different emotional states
    # First with emotional query that should activate gating
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": "Tell me about emotional experiences",
            "max_memories": 3,
            "query_metadata": {
                "current_emotion": "joy",  # Current emotional state
                "emotion_intensity": 0.7    # High intensity
            }
        }
    ) as response:
        assert response.status == 200, f"Failed to retrieve memories: {await response.text()}"
        result = await response.json()
        
        # 4. Verify MAG-specific retrieval behavior (emotional gating)
        assert "memories" in result, "No memories in response"
        assert len(result["memories"]) > 0, "No memories retrieved"
        
        # MAG should prioritize emotionally congruent memories (joy in this case)
        # At least one high-joy memory should be present in the results
        found_joy = False
        for memory in result["memories"]:
            memory_emotion = memory.get("metadata", {}).get("dominant_emotion")
            if memory_emotion == "joy":
                found_joy = True
                break
        
        assert found_joy, "MAG variant did not retrieve emotionally congruent memories"
    
    # 5. Now query with neutral state - MAG should show different behavior
    async with session.post(
        "http://localhost:8002/retrieve_memories",
        json={
            "query": "Tell me about informative content",
            "max_memories": 3,
            "query_metadata": {
                "current_emotion": "neutral",  # Neutral emotional state
                "emotion_intensity": 0.2       # Low intensity
            }
        }
    ) as response:
        assert response.status == 200
        result = await response.json()
        
        # MAG should prioritize neutral memories with low emotional content
        neutral_memories = []
        for memory in result.get("memories", []):
            memory_emotion = memory.get("metadata", {}).get("dominant_emotion")
            if memory_emotion == "neutral":
                neutral_memories.append(memory)
                
        assert len(neutral_memories) > 0, "MAG didn't retrieve neutral memories with neutral query"
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)

@pytest.mark.asyncio
async def test_mag_variant_state(api_clients):
    """Test MAG variant state management and internal gating structure."""
    session, mc_client = api_clients
    
    # 1. Check Neural Memory server state to confirm MAG model is active
    async with session.get("http://localhost:8001/status") as response:
        assert response.status == 200, "Failed to get Neural Memory status"
        nm_status = await response.json()
        
        # Look for MAG-specific indicators
        assert "model_info" in nm_status, "No model info in Neural Memory status"
        model_info = nm_status["model_info"]
        
        # Verify it's running MAG variant
        if "variant" in model_info:
            assert model_info["variant"] == "MAG", f"Wrong variant: {model_info['variant']}"
        elif "architecture" in model_info:
            assert "MAG" in model_info["architecture"], f"MAG not in architecture: {model_info['architecture']}"
    
    # 2. Check context-cascade-engine status
    async with session.get("http://localhost:8002/status") as response:
        assert response.status == 200, "Failed to get CCE status"
        cce_status = await response.json()
        
        # Verify CCE is also configured for MAG
        titan_config = cce_status.get("config", {}).get("titan", {})
        if titan_config:
            assert titan_config.get("variant") == "MAG" or \
                   titan_config.get("titans_variant") == "MAG", \
                   f"CCE not configured for MAG: {titan_config}"
        
        # Alternative check locations depending on implementation
        titans_variant = cce_status.get("titans_variant") or \
                        cce_status.get("config", {}).get("titans_variant")
        if titans_variant is not None:
            assert titans_variant == "MAG", f"Wrong CCE variant: {titans_variant}"

@pytest.mark.asyncio
async def test_mag_emotion_gating_influence(api_clients):
    """Test the impact of MAG's emotion gating mechanism on memory retrieval."""
    session, mc_client = api_clients
    
    # Create emotion-tagged memories
    emotions = ["joy", "anger", "sadness", "fear", "neutral"]
    memory_ids = []
    
    # Create memories with different emotions
    for emotion in emotions:
        for i in range(2):  # 2 memories per emotion
            content = f"This is a {emotion} memory {i} for testing MAG's emotion gating"
            memory_entry = {
                "content": content,
                "embedding": [float(j) / 100 for j in range(384)],  # Simple test embedding
                "metadata": {
                    "source": "mag_emotion_test",
                    "dominant_emotion": emotion,
                    "emotion_intensity": 0.8 if emotion != "neutral" else 0.2
                }
            }
            
            result = await mc_client.process_memory(memory_entry)
            memory_ids.append(result["memory_id"])
    
    # Allow time for processing
    await asyncio.sleep(2)
    
    # Test the gating effect with different emotional contexts
    emotion_queries = [
        {"emotion": "joy", "query": "Tell me about happy memories"},
        {"emotion": "anger", "query": "What makes people upset?"},
        {"emotion": "neutral", "query": "Give me factual information"}
    ]
    
    for eq in emotion_queries:
        # Query with specific emotional context
        async with session.post(
            "http://localhost:8002/retrieve_memories",
            json={
                "query": eq["query"],
                "max_memories": 4,
                "query_metadata": {
                    "current_emotion": eq["emotion"],
                    "emotion_intensity": 0.7 if eq["emotion"] != "neutral" else 0.2
                }
            }
        ) as response:
            result = await response.json()
            memories = result.get("memories", [])
            
            # Count emotion matches in retrieved memories
            matching_emotions = 0
            for memory in memories:
                memory_emotion = memory.get("metadata", {}).get("dominant_emotion")
                if memory_emotion == eq["emotion"]:
                    matching_emotions += 1
            
            # MAG should prioritize emotion-congruent memories
            # At least 50% of retrieved memories should match the query emotion
            assert matching_emotions >= len(memories) * 0.5, \
                   f"MAG gating not working for {eq['emotion']} emotion. " \
                   f"Only {matching_emotions}/{len(memories)} memories matched."
    
    # Clean up
    for memory_id in memory_ids:
        await mc_client.delete_memory(memory_id)
