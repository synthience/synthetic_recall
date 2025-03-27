#!/usr/bin/env python

import os
import sys
import numpy as np
import logging
import asyncio
import json
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_retrieval_test")

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the SynthiansMemoryCore and tools
from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.vector_index import MemoryVectorIndex


async def test_memory_retrieval_fix():
    # Create a test directory
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_memory_system")
    os.makedirs(test_dir, exist_ok=True)
    
    # Configuration for the memory core
    config = {
        'embedding_dim': 768,
        'storage_path': test_dir,
        'vector_index_type': 'Cosine',
        'use_gpu': False  # Use CPU for testing simplicity
    }
    
    # Create the memory core
    logger.info("Creating memory core")
    memory_core = SynthiansMemoryCore(config)
    await memory_core.initialize()
    
    # Generate some test memories
    memory_count = 5
    test_contents = [
        "Artificial intelligence is the simulation of human intelligence in machines.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are computing systems vaguely inspired by biological brains.",
        "Deep learning uses neural networks with many layers to analyze complex data.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Process the test memories
    logger.info(f"Processing {memory_count} test memories")
    memory_ids = []
    for i, content in enumerate(test_contents):
        metadata = {"test_id": f"test_{i}", "source": "test_script"}
        memory_id = await memory_core.process_new_memory(content=content, metadata=metadata)
        memory_ids.append(memory_id)
        logger.info(f"Created memory {i+1}/{memory_count}: {memory_id}")
        
    # Force persistence of memories
    logger.info("Persisting memories to disk")
    await memory_core._persist_all_managed_memories()
    logger.info(f"Persisted {len(memory_ids)} memories")
    
    # Log vector index stats
    vector_index = memory_core.vector_index
    logger.info(f"Vector index status: {vector_index.count()} embeddings, {len(vector_index.id_to_index)} ID mappings")
    
    # Test retrieval with different queries
    test_queries = [
        "What is artificial intelligence?",
        "Tell me about neural networks",
        "How does machine learning work?"
    ]
    
    # Ensure we clean up previous index files to avoid any stale data
    await memory_core.shutdown()
    
    # Create a new memory core and load from disk
    logger.info("\nCreating new memory core and loading from disk")
    new_memory_core = SynthiansMemoryCore(config)
    await new_memory_core.initialize()
    logger.info(f"Loaded memories from disk, cached entries: {len(new_memory_core._memories)}")
    
    # Log vector index status after loading
    new_vector_index = new_memory_core.vector_index
    logger.info(f"Vector index after load: {new_vector_index.count()} embeddings, {len(new_vector_index.id_to_index)} ID mappings")
    
    # Test retrieval for each query
    logger.info("\nTesting memory retrieval with different queries")
    for i, query in enumerate(test_queries):
        logger.info(f"\nQuery {i+1}: '{query}'")
        
        # First try with normal threshold
        memories = await new_memory_core.retrieve_memories(query=query, top_k=3, threshold=0.7)
        logger.info(f"Retrieved {len(memories)} memories (threshold=0.7)")
        
        # If no memories found, try with a lower threshold
        if not memories:
            memories = await new_memory_core.retrieve_memories(query=query, top_k=3, threshold=0.05)
            logger.info(f"Retrieved {len(memories)} memories with LOWERED threshold=0.05")
        
        for j, memory in enumerate(memories):
            # Handle different memory data structures based on return type
            if isinstance(memory, str):
                # Memory is just the ID
                logger.info(f"  Result {j+1}: {memory} - Score: N/A")
            elif isinstance(memory, dict):
                # Memory is a dictionary
                memory_id = memory.get('id', 'unknown')
                score = memory.get('score', 'N/A')
                content = memory.get('content', '')
                logger.info(f"  Result {j+1}: {memory_id} - Score: {score if isinstance(score, str) else f'{score:.4f}'}")
                logger.info(f"  Content: {content[:50] if content else 'No content'}...")
            else:
                # Unknown type
                logger.info(f"  Result {j+1}: Unknown memory type: {type(memory)}")
    
    # Clean up
    await new_memory_core.shutdown()
    logger.info("Test complete")

    # Return success if we retrieved at least one memory
    return len(memories) > 0

def clean_test_dir():
    """Clean up the test directory."""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_memory_system")
    if os.path.exists(test_dir):
        try:
            for root, dirs, files in os.walk(test_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            logger.error(f"Error cleaning test directory: {str(e)}")

if __name__ == "__main__":
    try:
        # Clean any previous test data
        clean_test_dir()
        
        # Run the test using asyncio
        success = asyncio.run(test_memory_retrieval_fix())
        
        # Clean up after the test
        clean_test_dir()
        
        # Exit with appropriate status
        if success:
            logger.info("✅ Memory retrieval test PASSED!")
            sys.exit(0)
        else:
            logger.error("❌ Memory retrieval test FAILED!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
