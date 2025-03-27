#!/usr/bin/env python

import logging
import numpy as np
from datetime import datetime
import asyncio
import uuid

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
            'use_gpu': True
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
    """Test the full memory system directly."""
    try:
        from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore
        
        logger.info("Creating memory core instance...")
        memory_core = SynthiansMemoryCore()
        await memory_core.initialize()
        
        # Create a unique test memory
        timestamp = datetime.now().isoformat()
        content = f"Test memory created at {timestamp}"
        metadata = {"test_type": "direct_test", "timestamp": timestamp}
        
        logger.info(f"Creating memory: {content}")
        memory_id = await memory_core.process_new_memory(content, metadata)
        logger.info(f"Created memory with ID: {memory_id}")
        
        # Wait for indexing
        logger.info("Waiting for indexing...")
        await asyncio.sleep(1)
        
        # Retrieve memory
        query = f"test {timestamp}"
        logger.info(f"Retrieving memory with query: '{query}'")
        
        # Use the lower threshold (0.3) that we know improves recall
        memories = await memory_core.retrieve_memories(query, 5, threshold=0.3)
        logger.info(f"Retrieved {len(memories)} memories")
        
        # Check if our memory was found
        found = any(memory.id == memory_id for memory in memories)
        if found:
            logger.info("✓ Memory retrieval test passed")
            return True
        else:
            logger.error("✗ Memory retrieval test failed")
            return False
    except Exception as e:
        logger.error(f"Error during memory system test: {str(e)}")
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
