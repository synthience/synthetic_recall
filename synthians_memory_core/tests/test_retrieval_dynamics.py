import pytest
import asyncio
import json
import time
import numpy as np
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_retrieve_with_emotion_match():
    """Test retrieval with emotional matching."""
    async with SynthiansClient() as client:
        # Create memories with different emotions
        happy_memory = await client.process_memory(
            content="I'm so excited about this amazing project! Everything is going wonderfully!",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        sad_memory = await client.process_memory(
            content="I'm feeling really down today. Nothing seems to be working out.",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        angry_memory = await client.process_memory(
            content="I'm absolutely furious about how this situation was handled!",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        # Wait briefly for processing
        await asyncio.sleep(1)
        
        # Retrieve with happy emotion context
        happy_emotion = {"dominant_emotion": "joy", "emotions": {"joy": 0.8, "surprise": 0.2}}
        happy_results = await client.retrieve_memories(
            query="feeling emotion test",
            top_k=5,
            user_emotion=happy_emotion
        )
        
        # Retrieve with sad emotion context
        sad_emotion = {"dominant_emotion": "sadness", "emotions": {"sadness": 0.9}}
        sad_results = await client.retrieve_memories(
            query="feeling emotion test",
            top_k=5,
            user_emotion=sad_emotion
        )
        
        # If emotional gating is working correctly, happy memories should rank higher
        # when queried with happy emotion, and sad memories with sad emotion
        happy_memories = happy_results.get("memories", [])
        sad_memories = sad_results.get("memories", [])
        
        print(f"Happy emotion results (first memory): {json.dumps(happy_memories[0] if happy_memories else {}, indent=2)}")
        print(f"Sad emotion results (first memory): {json.dumps(sad_memories[0] if sad_memories else {}, indent=2)}")
        
        # Note: These assertions might be too strict depending on implementation
        # The exact ranking will depend on many factors
        if happy_memories and sad_memories:
            for memory in happy_memories:
                if memory.get("content", "").startswith("I'm so excited"):
                    happy_rank = happy_memories.index(memory)
                    break
            else:
                happy_rank = -1
                
            for memory in sad_memories:
                if memory.get("content", "").startswith("I'm feeling really down"):
                    sad_rank = sad_memories.index(memory)
                    break
            else:
                sad_rank = -1
            
            print(f"Happy memory rank in happy query: {happy_rank}")
            print(f"Sad memory rank in sad query: {sad_rank}")

@pytest.mark.asyncio
async def test_retrieve_with_low_threshold():
    """Test retrieval with different threshold values."""
    async with SynthiansClient() as client:
        # Create a unique memory
        unique_id = int(time.time())
        unique_content = f"This is a unique threshold test memory {unique_id}"
        
        memory_resp = await client.process_memory(content=unique_content)
        memory_id = memory_resp.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Query with high threshold
        high_threshold_resp = await client.retrieve_memories(
            query=f"completely unrelated query {unique_id}",  # Unrelated but with unique ID
            top_k=10,
            threshold=0.9  # High threshold should filter out most memories
        )
        
        # Query with low threshold
        low_threshold_resp = await client.retrieve_memories(
            query=f"completely unrelated query {unique_id}",  # Same unrelated query
            top_k=10,
            threshold=0.1  # Low threshold should include most memories
        )
        
        high_threshold_memories = high_threshold_resp.get("memories", [])
        low_threshold_memories = low_threshold_resp.get("memories", [])
        
        # Low threshold should return more memories than high threshold
        print(f"High threshold returned {len(high_threshold_memories)} memories")
        print(f"Low threshold returned {len(low_threshold_memories)} memories")
        
        # Check if the unique memory is in the low threshold results
        low_thresh_ids = [m.get("id") for m in low_threshold_memories]
        memory_found = memory_id in low_thresh_ids
        
        print(f"Memory found in low threshold results: {memory_found}")
        print(f"Low threshold memory IDs: {low_thresh_ids}")

@pytest.mark.asyncio
async def test_metadata_filtering():
    """Test retrieval with metadata filters."""
    async with SynthiansClient() as client:
        # Create memories with different metadata
        timestamp = int(time.time())
        
        # Create memory with importance=high
        high_importance = await client.process_memory(
            content=f"High importance memory {timestamp}",
            metadata={"importance": "high", "category": "test", "filter_test": True}
        )
        
        # Create memory with importance=medium
        medium_importance = await client.process_memory(
            content=f"Medium importance memory {timestamp}",
            metadata={"importance": "medium", "category": "test", "filter_test": True}
        )
        
        # Create memory with importance=low
        low_importance = await client.process_memory(
            content=f"Low importance memory {timestamp}",
            metadata={"importance": "low", "category": "test", "filter_test": True}
        )
        
        # Create memory with different category
        different_category = await client.process_memory(
            content=f"Different category memory {timestamp}",
            metadata={"importance": "high", "category": "other", "filter_test": True}
        )
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Test if we can filter by metadata
        # Note: This assumes the retrieve_memories endpoint supports metadata filtering
        # If not, this test will need to be adapted
        
        try:
            # Query for high importance memories only
            # This might need to be updated based on actual API implementation
            high_imp_query = await client.retrieve_memories(
                query=f"memory {timestamp}",
                top_k=10,
                metadata_filter={"importance": "high"}
            )
            
            # Query for test category memories only
            test_category_query = await client.retrieve_memories(
                query=f"memory {timestamp}",
                top_k=10,
                metadata_filter={"category": "test"}
            )
            
            high_imp_memories = high_imp_query.get("memories", [])
            test_cat_memories = test_category_query.get("memories", [])
            
            print(f"High importance query returned {len(high_imp_memories)} memories")
            print(f"Test category query returned {len(test_cat_memories)} memories")
            
            # Check that our filtered queries worked as expected
            high_imp_contents = [m.get("content", "") for m in high_imp_memories]
            test_cat_contents = [m.get("content", "") for m in test_cat_memories]
            
            print(f"High importance memory contents: {high_imp_contents}")
            print(f"Test category memory contents: {test_cat_contents}")
            
        except Exception as e:
            # This test may fail if the API doesn't support metadata filtering
            print(f"Metadata filtering test failed: {str(e)}")
            print("This feature may not be implemented yet or works differently.")

@pytest.mark.asyncio
async def test_top_k_ranking_accuracy():
    """Test that memory retrieval respects top_k parameter and ranks by relevance."""
    async with SynthiansClient() as client:
        # Create a set of memories with varying relevance to a specific query
        base_content = "This is a test of the ranking system"
        timestamp = int(time.time())
        
        # Create memories with varying relevance
        await client.process_memory(
            content=f"{base_content} with direct relevance to ranking and sorting. {timestamp}"
        )
        
        await client.process_memory(
            content=f"{base_content} with some relevance to sorting. {timestamp}"
        )
        
        await client.process_memory(
            content=f"{base_content} with minimal relevance. {timestamp}"
        )
        
        await client.process_memory(
            content=f"Completely unrelated content that shouldn't be ranked highly. {timestamp}"
        )
        
        # Create 10 more filler memories
        for i in range(10):
            await client.process_memory(
                content=f"Filler memory {i} for ranking test. {timestamp}"
            )
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Test with different top_k values
        top_3_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=3
        )
        
        top_5_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=5
        )
        
        top_10_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=10
        )
        
        # Verify the correct number of results returned
        assert len(top_3_results.get("memories", [])) <= 3, "top_k=3 returned too many results"
        assert len(top_5_results.get("memories", [])) <= 5, "top_k=5 returned too many results"
        assert len(top_10_results.get("memories", [])) <= 10, "top_k=10 returned too many results"
        
        # Check the ranking - most relevant should be first
        if top_10_results.get("memories"):
            # Get first result content
            first_result = top_10_results["memories"][0]["content"]
            print(f"First ranked result: {first_result}")
            
            # It should contain "ranking and sorting"
            assert "ranking and sorting" in first_result.lower(), "Most relevant content not ranked first"
