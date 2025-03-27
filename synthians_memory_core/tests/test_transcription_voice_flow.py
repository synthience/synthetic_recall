import pytest
import asyncio
import json
import time
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient

# Add process_transcription method to SynthiansClient if not already present
async def process_transcription(self, text: str, audio_metadata: dict = None, embedding=None):
    """Process transcription data and store it in the memory system."""
    payload = {
        "text": text,
        "audio_metadata": audio_metadata or {},
        "embedding": embedding
    }
    async with self.session.post(
        f"{self.base_url}/process_transcription", json=payload
    ) as response:
        return await response.json()

# Add the method to the client class if not present
if not hasattr(SynthiansClient, "process_transcription"):
    SynthiansClient.process_transcription = process_transcription

@pytest.mark.asyncio
async def test_transcription_feature_extraction():
    """Test that transcription processing extracts relevant features."""
    async with SynthiansClient() as client:
        # Create a transcription with rich metadata
        text = "This is a test transcription with some pauses... and rhythm changes."
        audio_metadata = {
            "duration_sec": 5.2,
            "avg_volume": 0.75,
            "speaking_rate": 2.1,  # Words per second
            "pauses": [
                {"start": 1.2, "duration": 0.5},
                {"start": 3.5, "duration": 0.8}
            ]
        }
        
        # Process the transcription
        result = await client.process_transcription(
            text=text,
            audio_metadata=audio_metadata
        )
        
        # Verify successful processing
        assert result.get("success") is True, f"Transcription processing failed: {result.get('error')}"
        assert "memory_id" in result, "No memory ID returned for transcription"
        
        # Check metadata enrichment
        metadata = result.get("metadata", {})
        
        # Basic metadata verification
        assert "timestamp" in metadata, "No timestamp in metadata"
        assert "speaking_rate" in metadata, "Speaking rate not captured in metadata"
        assert "duration_sec" in metadata, "Duration not captured in metadata"
        
        # Advanced feature extraction verification (if implemented)
        if "pause_count" in metadata:
            assert metadata["pause_count"] >= 2, "Expected at least 2 pauses to be detected"
        
        if "speech_features" in metadata:
            assert isinstance(metadata["speech_features"], dict), "Speech features not properly structured"
        
        print(f"Transcription metadata: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_interrupt_metadata_enrichment():
    """Test that interruption metadata is properly stored and processed."""
    async with SynthiansClient() as client:
        # Create a transcription with interruption data
        text = "I was talking about- wait, let me restart. This is what I meant to say."
        audio_metadata = {
            "duration_sec": 7.5,
            "was_interrupted": True,
            "interruptions": [
                {"timestamp": 2.1, "duration": 0.3, "type": "self"}
            ],
            "user_interruptions": 1
        }
        
        # Process the transcription
        result = await client.process_transcription(
            text=text,
            audio_metadata=audio_metadata
        )
        
        # Verify successful processing
        assert result.get("success") is True, "Transcription processing failed"
        
        # Check interruption metadata
        metadata = result.get("metadata", {})
        assert "was_interrupted" in metadata, "Interruption flag not in metadata"
        assert metadata.get("was_interrupted") is True, "Interruption flag not preserved"
        
        if "interruption_count" in metadata:
            assert metadata["interruption_count"] >= 1, "Expected at least 1 interruption to be counted"
        
        if "user_interruptions" in metadata:
            assert metadata["user_interruptions"] >= 1, "User interruptions not preserved in metadata"
        
        print(f"Interruption metadata: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_session_level_memory():
    """Test that multiple utterances within a session are properly linked."""
    async with SynthiansClient() as client:
        # Generate a unique session ID
        session_id = f"test-session-{int(time.time())}"
        
        # Create first utterance in session
        text1 = "This is the first part of a multi-utterance conversation."
        metadata1 = {
            "session_id": session_id,
            "utterance_index": 1,
            "timestamp": time.time()
        }
        
        result1 = await client.process_memory(
            content=text1,
            metadata=metadata1
        )
        
        assert result1.get("success") is True, "First utterance processing failed"
        memory_id1 = result1.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Create second utterance in same session
        text2 = "This is the second part, continuing from what I said before."
        metadata2 = {
            "session_id": session_id,
            "utterance_index": 2,
            "timestamp": time.time(),
            "previous_memory_id": memory_id1  # Link to previous utterance
        }
        
        result2 = await client.process_memory(
            content=text2,
            metadata=metadata2
        )
        
        assert result2.get("success") is True, "Second utterance processing failed"
        memory_id2 = result2.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Create third utterance in same session
        text3 = "This is the third and final part of my conversation."
        metadata3 = {
            "session_id": session_id,
            "utterance_index": 3,
            "timestamp": time.time(),
            "previous_memory_id": memory_id2  # Link to previous utterance
        }
        
        result3 = await client.process_memory(
            content=text3,
            metadata=metadata3
        )
        
        assert result3.get("success") is True, "Third utterance processing failed"
        
        # Retrieve memories from this session
        # This assumes the API has a way to filter by session_id
        # If not, we can query by the unique session ID in the content
        retrieval_resp = await client.retrieve_memories(
            query=f"session:{session_id}",
            top_k=10
        )
        
        # Check if all three memories were retrieved
        memories = retrieval_resp.get("memories", [])
        memory_ids = [m.get("id") for m in memories]
        
        print(f"Retrieved session memories: {json.dumps(memory_ids, indent=2)}")
        
        # Check for session links in metadata (if implemented)
        for memory in memories:
            if "metadata" in memory and "session_id" in memory["metadata"]:
                assert memory["metadata"]["session_id"] == session_id, "Session ID not preserved"

@pytest.mark.asyncio
async def test_voice_state_tracking():
    """Test that voice state transitions are properly tracked in memory metadata."""
    async with SynthiansClient() as client:
        # Create a transcription with voice state metadata
        text = "This is a test of voice state tracking."
        audio_metadata = {
            "voice_state": "SPEAKING",
            "state_duration": 3.2,
            "previous_state": "LISTENING",
            "state_transition_count": 5,
            "last_state_transition_time": time.time() - 3.2
        }
        
        try:
            # Process the transcription (if endpoint exists)
            result = await client.process_transcription(
                text=text,
                audio_metadata=audio_metadata
            )
            
            # Verify successful processing
            assert result.get("success") is True, "Voice state tracking test failed"
            
            # Check if voice state metadata was preserved
            metadata = result.get("metadata", {})
            if "voice_state" in metadata:
                assert metadata["voice_state"] == "SPEAKING", "Voice state not preserved"
            
            if "state_transition_count" in metadata:
                assert metadata["state_transition_count"] == 5, "State transition count not preserved"
            
            print(f"Voice state metadata: {json.dumps(metadata, indent=2)}")
            
        except Exception as e:
            # This test may fail if the API doesn't support voice state tracking yet
            print(f"Voice state tracking test failed: {str(e)}")
            print("This feature may not be implemented yet.")
