import pytest
import asyncio
import json
import time
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_basic_memory_flow():
    """Test the basic memory creation, retrieval, and feedback flow."""
    async with SynthiansClient() as client:
        # Step 1: Create a unique memory with a timestamp
        current_time = datetime.now().isoformat()
        content = f"Testing memory processing lifecycle at {current_time}"
        memory_resp = await client.process_memory(
            content=content,
            metadata={"source": "test_suite", "importance": 0.8}
        )
        
        # Assert successful creation
        assert memory_resp.get("success") is True, f"Memory creation failed: {memory_resp.get('error')}"
        memory_id = memory_resp.get("memory_id")
        assert memory_id is not None, "No memory ID returned"
        
        # Print for debugging
        print(f"Memory created with ID: {memory_id}")
        print(f"Memory response: {json.dumps(memory_resp, indent=2)}")
        
        # Step 2: Retrieve the memory
        # Use a unique portion of the content to ensure we get this specific memory
        query = f"memory processing lifecycle at {current_time}"
        # Add a lower threshold to ensure retrieval works
        retrieval_resp = await client.retrieve_memories(query, top_k=3, threshold=0.2)
        
        # Assert successful retrieval
        assert retrieval_resp.get("success") is True, f"Memory retrieval failed: {retrieval_resp.get('error')}"
        memories = retrieval_resp.get("memories", [])
        assert len(memories) > 0, "No memories retrieved"
        
        # Check if our specific memory was retrieved
        retrieved_ids = [m.get("id") for m in memories]
        assert memory_id in retrieved_ids, f"Created memory {memory_id} not found in retrieved memories: {retrieved_ids}"
        
        # Print for debugging
        print(f"Retrieved {len(memories)} memories")
        print(f"Retrieved memory IDs: {retrieved_ids}")
        
        # Step 3: Provide feedback
        feedback_resp = await client.provide_feedback(
            memory_id=memory_id,
            similarity_score=0.85,
            was_relevant=True
        )
        
        # Assert successful feedback
        assert feedback_resp.get("success") is True, f"Feedback submission failed: {feedback_resp.get('error')}"
        assert "new_threshold" in feedback_resp, "No threshold adjustment information returned"
        
        # Print for debugging
        print(f"Feedback response: {json.dumps(feedback_resp, indent=2)}")

@pytest.mark.asyncio
async def test_memory_persistence_roundtrip():
    """Test that memories persist and can be retrieved after creation."""
    async with SynthiansClient() as client:
        # Create a unique memory
        unique_id = int(time.time() * 1000)
        content = f"Persistence test memory with unique ID: {unique_id}"
        
        # Create the memory
        creation_resp = await client.process_memory(content=content)
        assert creation_resp.get("success") is True, "Memory creation failed"
        memory_id = creation_resp.get("memory_id")
        
        # Wait briefly to ensure persistence
        await asyncio.sleep(0.5)
        
        # Retrieve the memory with the unique identifier
        retrieval_resp = await client.retrieve_memories(f"unique ID: {unique_id}", top_k=5, threshold=0.2)
        print(f"\nRetrieval response: {json.dumps(retrieval_resp, indent=2)}")
        assert retrieval_resp.get("success") is True, f"Memory retrieval failed: {retrieval_resp.get('error', 'No error specified')}"
        
        # Verify the memory was retrieved
        memories = retrieval_resp.get("memories", [])
        retrieved_ids = [m.get("id") for m in memories]
        assert memory_id in retrieved_ids, f"Memory {memory_id} not persisted/retrieved"

@pytest.mark.asyncio
async def test_metadata_enrichment_on_store():
    """Test that metadata is properly enriched when storing memories."""
    async with SynthiansClient() as client:
        # Create a memory with minimal metadata
        content = "Test memory for metadata enrichment"
        metadata = {"source": "test_suite", "custom_field": "custom_value"}
        
        response = await client.process_memory(content=content, metadata=metadata)
        assert response.get("success") is True, "Memory creation failed"
        
        # Verify metadata enrichment
        returned_metadata = response.get("metadata", {})
        
        # Check that our custom metadata was preserved
        assert returned_metadata.get("source") == "test_suite"
        assert returned_metadata.get("custom_field") == "custom_value"
        
        # Check that system metadata was added
        assert "timestamp" in returned_metadata, "Timestamp metadata missing"
        assert "length" in returned_metadata, "Length metadata missing"
        assert "uuid" in returned_metadata, "UUID metadata missing"
        
        # Optional checks for more advanced metadata
        if "cognitive_complexity" in returned_metadata:
            assert isinstance(returned_metadata["cognitive_complexity"], (int, float))
        
        print(f"Enriched metadata: {json.dumps(returned_metadata, indent=2)}")

@pytest.mark.asyncio
async def test_delete_memory_by_id():
    """Test memory deletion functionality."""
    async with SynthiansClient() as client:
        # Create a memory
        content = f"Memory to be deleted at {datetime.now().isoformat()}"
        creation_resp = await client.process_memory(content=content)
        assert creation_resp.get("success") is True, "Memory creation failed"
        memory_id = creation_resp.get("memory_id")
        
        # TODO: Implement actual delete endpoint call once available
        # This is a placeholder for when the delete endpoint is implemented
        
        # Example of how delete might be implemented:
        # delete_resp = await client.delete_memory(memory_id=memory_id)
        # assert delete_resp.get("success") is True, "Memory deletion failed"
        
        # After implementing deletion, verify the memory is gone:
        # retrieval_resp = await client.retrieve_memories(content, top_k=1)  
        # memories = retrieval_resp.get("memories", [])
        # assert memory_id not in [m.get("id") for m in memories], "Memory still exists after deletion"
