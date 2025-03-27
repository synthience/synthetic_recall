import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import the client class directly from client module
from synthians_memory_core.api.client.client import SynthiansClient

async def test_metadata_synthesis():
    """Test the metadata synthesis capabilities of the memory system."""
    print("\n=== Testing Metadata Synthesis ===\n")
    
    async with SynthiansClient() as client:
        # 1. Process a memory with specific emotional content
        print("\n1. Creating memory with emotional content...")
        happy_memory = await client.process_memory(
            content="I am feeling incredibly happy and joyful today. It's a wonderful day and everything is going great!",
            metadata={
                "source": "metadata_test",
                "importance": 0.9,
                "test_type": "positive_emotion"
            }
        )
        print(f"Happy memory result: {json.dumps(happy_memory, indent=2)}")
        
        # 2. Process a memory with negative emotional content
        print("\n2. Creating memory with negative emotional content...")
        sad_memory = await client.process_memory(
            content="I'm feeling quite sad and disappointed today. Things aren't going well and I'm frustrated.",
            metadata={
                "source": "metadata_test",
                "importance": 0.7,
                "test_type": "negative_emotion"
            }
        )
        print(f"Sad memory result: {json.dumps(sad_memory, indent=2)}")
        
        # 3. Process a memory with technical content
        print("\n3. Creating memory with technical/complex content...")
        tech_memory = await client.process_memory(
            content="The quantum computational paradigm leverages superposition and entanglement to perform calculations that would be infeasible on classical computers. The fundamental unit is the qubit, which can exist in multiple states simultaneously.",
            metadata={
                "source": "metadata_test",
                "importance": 0.8,
                "test_type": "complex_content"
            }
        )
        print(f"Technical memory result: {json.dumps(tech_memory, indent=2)}")
        
        # 4. Retrieve memories and check if metadata is preserved
        print("\n4. Retrieving memories to verify metadata...")
        # First try with default parameters
        retrieve_resp = await client.retrieve_memories(
            "test metadata synthesis", 
            top_k=5
        )
        print(f"Default retrieval results: {json.dumps(retrieve_resp, indent=2)}")
        
        # Try again with a lowered threshold to bypass ThresholdCalibrator
        print("\n4b. Retrieving with lowered threshold...")
        retrieve_with_threshold = await client.retrieve_memories(
            "test metadata synthesis", 
            top_k=5,
            threshold=0.4  # Explicitly lower the threshold well below our ~0.66 scores
        )
        print(f"Retrieval with threshold=0.4: {json.dumps(retrieve_with_threshold, indent=2)}")
        
        # Try with exact memory IDs to force retrieval
        print("\n4c. Retrieving by exact memory IDs...")
        memory_ids = [
            happy_memory.get("memory_id"),
            sad_memory.get("memory_id"),
            tech_memory.get("memory_id")
        ]
        # Filter out any None values
        memory_ids = [mid for mid in memory_ids if mid]
        
        if memory_ids:
            memory_by_id = await client.retrieve_memory_by_id(memory_ids[0])
            print(f"Retrieved by ID: {json.dumps(memory_by_id, indent=2)}")
            
            # Try direct query of each test type
            print("\n4d. Retrieving with direct test type queries...")
            for test_type in ["positive_emotion", "negative_emotion", "complex_content"]:
                test_query = await client.retrieve_memories(
                    test_type,  # Use the test_type as the query
                    top_k=1,
                    threshold=0.4,
                    user_emotion=None  # Bypass emotional gating
                )
                print(f"Query '{test_type}' results: {json.dumps(test_query, indent=2)}")
        
        # 5. Verify key metadata fields in each memory
        print("\n5. Validating metadata fields...")
        memories = retrieve_resp.get("memories", [])
        
        validation_results = []
        for memory in memories:
            metadata = memory.get("metadata", {})
            validation = {
                "id": memory.get("id"),
                "metadata_schema_version": metadata.get("metadata_schema_version"),
                "has_timestamp": "timestamp" in metadata,
                "has_timestamp_iso": "timestamp_iso" in metadata,
                "has_time_of_day": "time_of_day" in metadata,
                "has_dominant_emotion": "dominant_emotion" in metadata,
                "has_emotional_intensity": "emotional_intensity" in metadata,
                "has_complexity_estimate": "complexity_estimate" in metadata,
                "has_embedding_metadata": all(key in metadata for key in ["embedding_valid", "embedding_dim"])
            }
            validation_results.append(validation)
        
        print(f"Validation results: {json.dumps(validation_results, indent=2)}")
        
        # Summary
        print("\n=== Metadata Synthesis Test Summary ===\n")
        if validation_results:
            success = all(result.get("has_timestamp") and 
                         result.get("has_dominant_emotion") and 
                         result.get("has_complexity_estimate") 
                         for result in validation_results)
            if success:
                print("✅ SUCCESS: All memories have proper metadata synthesis")
            else:
                print("❌ FAILURE: Some memories are missing key metadata fields")
        else:
            print("❓ INCONCLUSIVE: No memories were retrieved for validation")

def main():
    """Run the metadata synthesis test."""
    asyncio.run(test_metadata_synthesis())

if __name__ == "__main__":
    main()
