#!/usr/bin/env python

import os
import sys
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_index_test")

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def random_unit_vector(dim: int) -> np.ndarray:
    """Generate a random unit vector of the specified dimension."""
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)

def test_vector_index_persistence():
    """Test that the vector index can persist id_to_index mappings."""
    try:
        # Import the MemoryVectorIndex class - importing here to catch import errors
        logger.info("Importing MemoryVectorIndex class...")
        from synthians_memory_core.vector_index import MemoryVectorIndex
        
        # Create a test directory
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_index")
        os.makedirs(test_dir, exist_ok=True)
        
        # Configuration
        config = {
            'embedding_dim': 768,
            'storage_path': test_dir,
            'index_type': 'Cosine',
            'use_gpu': False  # Use CPU for testing simplicity
        }
        
        # Create the index
        logger.info("Creating new vector index")
        index = MemoryVectorIndex(config)
        
        # Generate some test memories
        memory_count = 10
        memory_ids = [f"test_mem_{i}" for i in range(memory_count)]
        embeddings = [random_unit_vector(config['embedding_dim']) for _ in range(memory_count)]
        
        # Add the memories to the index
        logger.info(f"Adding {memory_count} test memories to index")
        for i, (mem_id, embedding) in enumerate(zip(memory_ids, embeddings)):
            success = index.add(mem_id, embedding)
            logger.info(f"Added memory {i+1}/{memory_count}: {mem_id} - Success: {success}")
        
        # Save the index
        logger.info("Saving index with id_to_index mappings")
        save_success = index.save()
        logger.info(f"Save success: {save_success}")
        
        if not save_success:
            logger.error("Failed to save index, aborting test")
            return False
        
        # Create a new index and load from the saved file
        logger.info("Creating new index and loading from saved file")
        new_index = MemoryVectorIndex(config)
        load_success = new_index.load()
        logger.info(f"Load success: {load_success}, Loaded {len(new_index.id_to_index)} id_to_index mappings")
        
        if not load_success:
            logger.error("Failed to load index, aborting test")
            return False
        
        # Test retrieval using the original embeddings as queries
        logger.info("Testing retrieval with original embeddings")
        all_retrieved = True
        for i, (mem_id, embedding) in enumerate(zip(memory_ids, embeddings)):
            results = new_index.search(embedding, k=1, threshold=0.8)
            if results and results[0][0] == mem_id:
                logger.info(f"Successfully retrieved memory {i+1}/{memory_count}: {mem_id} (score: {results[0][1]:.4f})")
            else:
                logger.error(f"Failed to retrieve memory {i+1}/{memory_count}: {mem_id}")
                if results:
                    logger.error(f"Got {results[0][0]} instead with score {results[0][1]:.4f}")
                else:
                    logger.error("No results returned")
                all_retrieved = False
        
        # Final result
        if all_retrieved:
            logger.info("✅ All memories successfully retrieved - PERSISTENCE FIX WORKS!")
        else:
            logger.error("❌ Some memories were not retrieved correctly - fix needs more work")
        
        return all_retrieved
    
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("This test requires the FAISS library to be installed correctly.")
        return False
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def clean_test_dir():
    """Clean up the test directory."""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_index")
    if os.path.exists(test_dir):
        try:
            for f in os.listdir(test_dir):
                file_path = os.path.join(test_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")
        except Exception as e:
            logger.error(f"Error cleaning test directory: {str(e)}")

if __name__ == "__main__":
    try:
        # Clean any previous test data
        clean_test_dir()
        
        # Run the test
        success = test_vector_index_persistence()
        
        # Clean up after the test
        clean_test_dir()
        
        # Exit with appropriate status
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
