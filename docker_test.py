#!/usr/bin/env python

import logging
import numpy as np
from datetime import datetime
import asyncio
import uuid
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_test")

# Test with different dimension vectors to verify dimension mismatch handling
async def test_faiss_vector_index():
    """Test the FAISS vector index implementation directly."""
    try:
        from synthians_memory_core.vector_index import MemoryVectorIndex
        
        logger.info("Creating FAISS vector index...")
        index = MemoryVectorIndex({
            'embedding_dim': 768,
            'storage_path': '/tmp/test_index',
            'index_type': 'L2',
            'use_gpu': True  # Enable GPU usage for testing
        })
        
        # Check if GPU is being used
        logger.info(f"Using GPU: {index.is_using_gpu}")
        
        # Test 1: Basic functionality with normal embeddings
        memory_id = str(uuid.uuid4())
        embedding = np.random.random(768).astype('float32')  # Normal dimension
        
        logger.info(f"Adding memory {memory_id} with 768-dim embedding")
        index.add(memory_id, embedding)
        
        results = index.search(embedding, k=1)
        if results and results[0][0] == memory_id:
            logger.info("✓ Basic functionality test passed")
        else:
            logger.error("✗ Basic functionality test failed")
            return False
        
        # Test 2: Different dimension handling
        memory_id2 = str(uuid.uuid4())
        embedding2 = np.random.random(384).astype('float32')  # Different dimension
        
        logger.info(f"Adding memory {memory_id2} with 384-dim embedding")
        index.add(memory_id2, embedding2)
        
        results = index.search(embedding2, k=2)
        found_memory2 = any(mem_id == memory_id2 for mem_id, _ in results)
        
        if found_memory2:
            logger.info("✓ Dimension mismatch handling test passed")
        else:
            logger.error("✗ Dimension mismatch handling test failed")
            return False
        
        # Test 3: NaN/Inf handling
        memory_id3 = str(uuid.uuid4())
        embedding3 = np.ones(768).astype('float32')
        embedding3[0] = np.nan  # Add a NaN
        embedding3[1] = np.inf  # Add an Inf
        
        logger.info(f"Adding memory {memory_id3} with NaN/Inf values")
        index.add(memory_id3, embedding3)
        
        # Create a query with NaN/Inf
        query = np.ones(768).astype('float32')
        query[0] = np.nan
        
        logger.info("Searching with NaN values in query")
        try:
            results = index.search(query, k=3)
            logger.info("✓ NaN/Inf handling test passed")
        except Exception as e:
            logger.error(f"✗ NaN/Inf handling test failed: {str(e)}")
            return False
        
        logger.info(f"Vector index has {index.count()} vectors")
        logger.info("All direct vector index tests passed!")
        return True
    except Exception as e:
        logger.error(f"Error during vector index test: {str(e)}")
        return False

# Test the memory system E2E
async def test_memory_system():
    """Test the full memory system API endpoints."""
    # Use the following depending on where the test is running:
    # - 'localhost:5010' when running test from host machine
    # - 'http://127.0.0.1:5010' when running test from host machine
    # - 'http://synthians_core:5010' when running from another container
    base_url = "http://127.0.0.1:5010"  # For direct host testing
    process_endpoint = f"{base_url}/process_memory"
    retrieve_endpoint = f"{base_url}/retrieve_memories"
    
    logger.info("--- Starting Memory System API Test ---")
    try:
        # Generate a consistent dummy embedding for the test
        # Use a fixed seed for reproducibility if needed, e.g., np.random.seed(42)
        dummy_embedding = np.random.rand(768).astype(np.float32)

        # Wait briefly for the server to be ready
        await asyncio.sleep(5)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Process new memory
            payload = {
                "content": "Test content",
                # Pass a dummy embedding to match Pydantic model
                "embedding": dummy_embedding.tolist(), 
                "metadata": {"source": "test_script", "timestamp": datetime.now().isoformat()}
            }
            logger.info(f"--- TEST: Sending request to {process_endpoint} ---")
            response = await client.post(process_endpoint, json=payload)
            logger.info(f"--- TEST: Received response from {process_endpoint}: {response.status_code} ---")
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            assert result["success"], f"API call failed: {result.get('error', 'Unknown error')}"
            memory_id = result.get("memory_id")
            assert memory_id
            logger.info(f"Successfully processed new memory: {memory_id}")

            # Wait a bit for processing
            await asyncio.sleep(2)

            # 2. Retrieve memories
            retrieve_payload = {
                "query": "test memory",  # More specific query that should match our content
                "top_k": 5,
                "threshold": 0.5  # Explicitly set a lower threshold for testing
            }
            logger.info(f"--- TEST: Sending request to {retrieve_endpoint} ---")
            response = await client.post(retrieve_endpoint, json=retrieve_payload)
            logger.info(f"--- TEST: Received response from {retrieve_endpoint}: {response.status_code} ---")

            response.raise_for_status()
            result = response.json()
            
            # Print the full response for debugging
            logger.info(f"RESPONSE DETAILS: success={result.get('success')}, memories_count={len(result.get('memories', []))}")
            if result.get("memories"):
                for i, mem in enumerate(result.get("memories", [])):
                    logger.info(f"Memory {i}: id={mem.get('id')}, similarity={mem.get('similarity', 0.0):.4f}")
            else:
                logger.info("NO MEMORIES RETURNED IN RESPONSE")
                
            assert result["success"], f"API call failed: {result.get('error', 'Unknown error')}"
            # We expect at least the memory we added to be retrieved
            assert len(result.get("memories", [])) > 0, "Expected at least one memory to be retrieved"
            logger.info(f"Retrieved {len(result.get('memories', []))} memories.")

            logger.info("✓ Memory System API Test Passed")
            return True

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Request error occurred: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    # Run the tests
    vector_index_result = await test_faiss_vector_index()
    memory_system_result = await test_memory_system()
    
    if vector_index_result and memory_system_result:
        logger.info("\n✓ ALL TESTS PASSED")
        return 0
    else:
        failed_tests = []
        if not vector_index_result:
            failed_tests.append("FAISS Vector Index")
        if not memory_system_result:
            failed_tests.append("Memory System")
            
        logger.error(f"\n✗ TESTS FAILED: {', '.join(failed_tests)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
