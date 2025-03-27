import pytest
import asyncio
import json
import time
import numpy as np
import os
import sys
import logging
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient
from synthians_memory_core.vector_index import MemoryVectorIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_vector_index")

@pytest.mark.asyncio
async def test_faiss_vector_index_creation():
    """Test the creation and basic functionality of the FAISS vector index."""
    # Create a test vector index with a specific dimension
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2',
        'use_gpu': True  # This will use GPU if available, otherwise fall back to CPU
    })
    
    # Verify the index was created with the right parameters
    assert index.dimension == dimension, f"Expected dimension {dimension}, got {index.dimension}"
    logger.info(f"Created vector index with dimension {index.dimension}, GPU usage: {index.is_using_gpu}")
    
    # Create some test embeddings
    num_vectors = 100
    test_vectors = np.random.random((num_vectors, dimension)).astype('float32')
    
    # Add vectors to the index
    for i in range(num_vectors):
        memory_id = f"test_memory_{i}"
        index.add(memory_id, test_vectors[i])
    
    # Verify the index contains the expected number of vectors
    assert index.count() == num_vectors, f"Expected {num_vectors} vectors in index, got {index.count()}"
    
    # Test search functionality
    query_vector = np.random.random(dimension).astype('float32')
    k = 10
    results = index.search(query_vector, k)
    
    # Verify search results format
    assert len(results) <= k, f"Expected at most {k} results, got {len(results)}"
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Results should be (memory_id, score) tuples"
    
    # Test index persistence
    index_path = os.path.join(index.storage_path, 'test_index.faiss')
    index.save(index_path)
    assert os.path.exists(index_path), f"Index file not found at {index_path}"
    
    # Test index loading
    new_index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    new_index.load(index_path)
    
    # Verify loaded index has the same vectors
    assert new_index.count() == index.count(), "Loaded index has different vector count"
    
    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)
    logger.info("Vector index creation and persistence test completed successfully")

@pytest.mark.asyncio
async def test_dimension_mismatch_handling():
    """Test the handling of embedding dimension mismatches."""
    # Create a vector index with specific dimension
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    
    # Create vectors with different dimensions
    smaller_dim = 384
    larger_dim = 1024
    
    standard_vector = np.random.random(dimension).astype('float32')
    smaller_vector = np.random.random(smaller_dim).astype('float32')
    larger_vector = np.random.random(larger_dim).astype('float32')
    
    # Add vectors with different dimensions
    index.add("standard_vector", standard_vector)
    index.add("smaller_vector", smaller_vector)  # Should be padded
    index.add("larger_vector", larger_vector)    # Should be truncated
    
    # Verify all vectors were added
    assert index.count() == 3, f"Expected 3 vectors in index, got {index.count()}"
    
    # Test search with different dimension vectors
    standard_results = index.search(standard_vector, 3)
    smaller_results = index.search(smaller_vector, 3)
    larger_results = index.search(larger_vector, 3)
    
    # Verify search results contain expected entries
    assert any(r[0] == "standard_vector" for r in standard_results), "Standard vector not found in results"
    assert any(r[0] == "smaller_vector" for r in smaller_results), "Smaller vector not found in results"
    assert any(r[0] == "larger_vector" for r in larger_results), "Larger vector not found in results"
    
    logger.info("Dimension mismatch handling test completed successfully")

@pytest.mark.asyncio
async def test_malformed_embedding_handling():
    """Test the handling of malformed embeddings (NaN/Inf values)."""
    # Create a vector index
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    
    # Create a normal vector and malformed vectors
    normal_vector = np.random.random(dimension).astype('float32')
    
    # Vector with NaN values
    nan_vector = np.random.random(dimension).astype('float32')
    nan_vector[10:20] = np.nan
    
    # Vector with Inf values
    inf_vector = np.random.random(dimension).astype('float32')
    inf_vector[30:40] = np.inf
    
    # Add vectors - the malformed ones should be handled gracefully
    index.add("normal_vector", normal_vector)
    
    # These should be handled by replacing with zeros or normalized vectors
    index.add("nan_vector", nan_vector)
    index.add("inf_vector", inf_vector)
    
    # Verify we can search without errors
    results = index.search(normal_vector, 3)
    assert len(results) > 0, "No results returned from search"
    
    # Search with malformed query vectors should also work
    nan_query = np.random.random(dimension).astype('float32')
    nan_query[5:15] = np.nan
    
    nan_results = index.search(nan_query, 3)
    assert len(nan_results) > 0, "No results returned from search with NaN query"
    
    logger.info("Malformed embedding handling test completed successfully")

@pytest.mark.asyncio
async def test_end_to_end_vector_retrieval():
    """End-to-end test of vector indexing and retrieval through the API."""
    async with SynthiansClient() as client:
        # Step 1: Create distinct test memories
        timestamp = datetime.now().isoformat()
        
        memory1 = await client.process_memory(
            content=f"FAISS vector index test memory Alpha at {timestamp}",
            metadata={"test_group": "vector_index", "category": "alpha"}
        )
        
        memory2 = await client.process_memory(
            content=f"FAISS vector index test memory Beta at {timestamp}",
            metadata={"test_group": "vector_index", "category": "beta"}
        )
        
        memory3 = await client.process_memory(
            content=f"FAISS vector index test memory Gamma at {timestamp}",
            metadata={"test_group": "vector_index", "category": "gamma"}
        )
        
        # Allow time for processing and indexing
        await asyncio.sleep(1)
        
        # Step 2: Retrieve with exact match
        alpha_query = f"Alpha at {timestamp}"
        alpha_results = await client.retrieve_memories(alpha_query, top_k=3)
        
        # Verify retrieval accuracy
        assert alpha_results.get("success") is True, "Retrieval failed"
        alpha_memories = alpha_results.get("memories", [])
        alpha_ids = [m.get("id") for m in alpha_memories]
        
        # Memory1 should be retrieved
        assert memory1.get("memory_id") in alpha_ids, "Alpha memory not found in retrieval results"
        
        # Step 3: Test with lower threshold to ensure retrieval works
        general_query = f"vector index test at {timestamp}"
        low_threshold_results = await client.retrieve_memories(
            general_query, 
            top_k=10, 
            threshold=0.3  # Lower threshold as per the memory improvement
        )
        
        all_memories = low_threshold_results.get("memories", [])
        all_ids = [m.get("id") for m in all_memories]
        
        # All memories should be retrieved with a lower threshold
        assert memory1.get("memory_id") in all_ids, "Memory 1 not found with low threshold"
        assert memory2.get("memory_id") in all_ids, "Memory 2 not found with low threshold"
        assert memory3.get("memory_id") in all_ids, "Memory 3 not found with low threshold"
        
        logger.info(f"Retrieved {len(all_memories)} memories with low threshold")
        logger.info("End-to-end vector retrieval test completed successfully")

if __name__ == "__main__":
    # For manual test execution
    asyncio.run(test_faiss_vector_index_creation())
    asyncio.run(test_dimension_mismatch_handling())
    asyncio.run(test_malformed_embedding_handling())
    asyncio.run(test_end_to_end_vector_retrieval())
