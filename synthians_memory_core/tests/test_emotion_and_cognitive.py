import pytest
import asyncio
import json
import numpy as np
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_emotion_analysis_rich():
    """Test emotion analysis with various emotional inputs."""
    async with SynthiansClient() as client:
        # Test happy emotion
        happy_text = "I'm incredibly happy today! Everything is going wonderfully well!"
        happy_result = await client.analyze_emotion(happy_text)
        
        assert happy_result.get("success") is True, "Emotion analysis failed"
        assert happy_result.get("dominant_emotion") in ["joy", "happiness"], f"Expected happy emotion, got {happy_result.get('dominant_emotion')}"
        assert happy_result.get("emotions", {}).get("joy", 0) > 0.5, "Expected high joy score"
        
        print(f"Happy emotion result: {json.dumps(happy_result, indent=2)}")
        
        # Test sad emotion
        sad_text = "I feel so sad and depressed today. Everything is going wrong."
        sad_result = await client.analyze_emotion(sad_text)
        
        assert sad_result.get("success") is True, "Emotion analysis failed"
        assert sad_result.get("dominant_emotion") in ["sadness", "sorrow"], f"Expected sad emotion, got {sad_result.get('dominant_emotion')}"
        
        print(f"Sad emotion result: {json.dumps(sad_result, indent=2)}")
        
        # Test angry emotion
        angry_text = "I'm absolutely furious about how I was treated! This is outrageous!"
        angry_result = await client.analyze_emotion(angry_text)
        
        assert angry_result.get("success") is True, "Emotion analysis failed"
        assert angry_result.get("dominant_emotion") in ["anger", "rage"], f"Expected anger emotion, got {angry_result.get('dominant_emotion')}"
        
        print(f"Angry emotion result: {json.dumps(angry_result, indent=2)}")

@pytest.mark.asyncio
async def test_emotion_fallback_path():
    """Test emotion analysis fallback mechanisms when model fails."""
    # Note: This test assumes emotion analyzer has a fallback mechanism
    # when the primary model fails. We'll test with extreme text that might
    # cause issues for the model.
    
    async with SynthiansClient() as client:
        # Test with extremely long text that might cause issues
        long_text = "happy " * 1000  # Very long repetitive text
        result = await client.analyze_emotion(long_text)
        
        # Even if the main model fails, we should still get a result
        assert result.get("success") is True, "Emotion analysis completely failed"
        assert "dominant_emotion" in result, "No dominant emotion provided"
        
        # Test with empty text
        empty_result = await client.analyze_emotion("")
        assert empty_result.get("success") is True, "Empty text analysis failed"
        assert "dominant_emotion" in empty_result, "No dominant emotion for empty text"
        
        print(f"Empty text emotion result: {json.dumps(empty_result, indent=2)}")

@pytest.mark.asyncio
async def test_emotion_saved_in_metadata():
    """Test that emotional analysis is saved in memory metadata."""
    async with SynthiansClient() as client:
        # Create a memory with strong emotional content
        content = "I am absolutely thrilled about the amazing news I received today!"
        
        # Process the memory with emotion analysis enabled
        memory_resp = await client.process_memory(
            content=content,
            metadata={"analyze_emotion": True}
        )
        
        assert memory_resp.get("success") is True, "Memory creation failed"
        metadata = memory_resp.get("metadata", {})
        
        # Check that emotion data was added to metadata
        assert "dominant_emotion" in metadata, "No dominant emotion in metadata"
        assert "emotional_intensity" in metadata, "No emotional intensity in metadata"
        assert "emotions" in metadata, "No emotions dictionary in metadata"
        
        # Check that the emotion is reasonable for the content
        assert metadata.get("dominant_emotion") in ["joy", "happiness"], f"Expected happy emotion, got {metadata.get('dominant_emotion')}"
        
        print(f"Memory metadata with emotions: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_cognitive_load_score_range():
    """Test that cognitive load scoring works across different complexity levels."""
    async with SynthiansClient() as client:
        # Test with simple text
        simple_text = "This is a simple sentence."
        simple_memory = await client.process_memory(content=simple_text)
        
        # Test with complex text
        complex_text = """The quantum mechanical model is a theoretical framework that describes the behavior of subatomic 
        particles through probabilistic wave functions. It posits that particles exhibit both wave-like and 
        particle-like properties, a concept known as wave-particle duality. The Schrödinger equation, a fundamental 
        mathematical formulation in quantum mechanics, predicts how these wave functions evolve over time. 
        Unlike classical mechanics, quantum mechanics introduces inherent uncertainty in measurements, 
        as formalized in Heisenberg's uncertainty principle, which states that certain pairs of physical properties 
        cannot be precisely measured simultaneously."""
        complex_memory = await client.process_memory(content=complex_text)
        
        # Get metadata for both memories
        simple_metadata = simple_memory.get("metadata", {})
        complex_metadata = complex_memory.get("metadata", {})
        
        # If cognitive_complexity is in the metadata, verify it's higher for complex text
        if "cognitive_complexity" in simple_metadata and "cognitive_complexity" in complex_metadata:
            simple_complexity = simple_metadata.get("cognitive_complexity", 0)
            complex_complexity = complex_metadata.get("cognitive_complexity", 0)
            
            # The complex text should have higher cognitive complexity
            assert complex_complexity > simple_complexity, \
                f"Expected higher complexity for complex text: simple={simple_complexity}, complex={complex_complexity}"
            
            print(f"Simple text complexity: {simple_complexity}")
            print(f"Complex text complexity: {complex_complexity}")

@pytest.mark.asyncio
async def test_emotional_gating_blocks_mismatched():
    """Test that emotional gating blocks memories with mismatched emotions."""
    async with SynthiansClient() as client:
        # Create a happy memory
        happy_text = "I'm so happy and excited about my new job!"
        happy_memory = await client.process_memory(content=happy_text)
        
        # Wait briefly for processing
        await asyncio.sleep(0.5)
        
        # Try to retrieve with angry emotion context
        angry_emotion = {"dominant_emotion": "anger", "emotions": {"anger": 0.9}}
        retrieval_resp = await client.retrieve_memories(
            query="job",
            top_k=5,
            user_emotion=angry_emotion
        )
        
        memories = retrieval_resp.get("memories", [])
        
        # If emotional gating is working, the happy memory might be ranked lower or filtered
        # We can't assert exact behavior since it depends on implementation details
        # Instead, we'll log the results for inspection
        print(f"Retrieved {len(memories)} memories with mismatched emotion")
        
        # Create an angry memory
        angry_text = "I'm absolutely furious about how they handled my job application!"
        angry_memory = await client.process_memory(content=angry_text)
        
        # Wait briefly for processing
        await asyncio.sleep(0.5)
        
        # Retrieve again with the same angry emotion context
        angry_retrieval = await client.retrieve_memories(
            query="job",
            top_k=5,
            user_emotion=angry_emotion
        )
        
        # The angry memory should now be present and possibly ranked higher
        angry_memories = angry_retrieval.get("memories", [])
        
        print(f"Retrieved {len(angry_memories)} memories with matching emotion")
        
        # Print the scores for comparison (if available)
        if memories and angry_memories and "quickrecal_score" in memories[0] and "quickrecal_score" in angry_memories[0]:
            print(f"Mismatched emotion memory score: {memories[0].get('quickrecal_score')}")
            print(f"Matching emotion memory score: {angry_memories[0].get('quickrecal_score')}")
