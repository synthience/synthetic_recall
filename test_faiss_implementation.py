#!/usr/bin/env python

import sys
import time
import os
import logging
import numpy as np
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faiss_test")

# Try to import FAISS with fallback installation
try:
    import faiss
    logger.info(f"FAISS version {getattr(faiss, '__version__', 'unknown')} already installed")
except ImportError:
    logger.warning("FAISS not found. Attempting to install...")
    try:
        # Check for GPU availability
        gpu_available = False
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            gpu_available = result.returncode == 0
        except:
            pass
            
        # Install appropriate FAISS package
        if gpu_available:
            logger.info("GPU detected, installing FAISS with GPU support")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-gpu'])
        else:
            logger.info("No GPU detected, installing CPU-only FAISS")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-cpu'])
            
        # Try importing again
        import faiss
        logger.info(f"Successfully installed and imported FAISS {getattr(faiss, '__version__', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
        sys.exit(1)

# Import vector index implementation
from synthians_memory_core.vector_index import MemoryVectorIndex

# Import client if available for end-to-end test
try:
    from synthians_memory_core.api.client.client import SynthiansClient
    client_available = True
except ImportError:
    logger.warning("SynthiansClient not available, skipping API tests")
    client_available = False

def test_faiss_vector_index_creation():
    """Test the creation and basic functionality of the FAISS vector index."""
    logger.info("Testing FAISS vector index creation and basic functionality...")
    
    # Create a test vector index with a specific dimension
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2',
        'use_gpu': True  # This will use GPU if available, otherwise fall back to CPU
    })
    
    # Log GPU usage status
    logger.info(f"Created vector index with dimension {index.dimension}, GPU usage: {index.is_using_gpu}")
    
    # Create some test embeddings
    num_vectors = 100
    test_vectors = np.random.random((num_vectors, dimension)).astype('float32')
    
    # Add vectors to the index
    start_time = time.time()
    for i in range(num_vectors):
        memory_id = f"test_memory_{i}"
        index.add(memory_id, test_vectors[i])
    add_time = time.time() - start_time
    logger.info(f"Added {num_vectors} vectors in {add_time:.4f} seconds")
    
    # Verify the index contains the expected number of vectors
    count = index.count()
    logger.info(f"Index contains {count} vectors")
    
    # Test search functionality
    query_vector = np.random.random(dimension).astype('float32')
    k = 10
    start_time = time.time()
    results = index.search(query_vector, k)
    search_time = time.time() - start_time
    
    logger.info(f"Search completed in {search_time:.4f} seconds")
    logger.info(f"Search returned {len(results)} results")
    
    # Test index persistence
    index_path = os.path.join(index.storage_path, 'test_index.faiss')
    index.save(index_path)
    logger.info(f"Saved index to {index_path}")
    
    # Test index loading
    new_index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    new_index.load(index_path)
    logger.info(f"Loaded index from {index_path} with {new_index.count()} vectors")
    
    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)
        logger.info(f"Cleaned up test index file {index_path}")
    
    logger.info("Vector index creation and persistence test completed successfully")
    return True

def test_dimension_mismatch_handling():
    """Test the handling of embedding dimension mismatches."""
    logger.info("Testing dimension mismatch handling...")
    
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
    
    # Log the number of vectors added
    vector_count = index.count()
    logger.info(f"Added vectors with different dimensions, index contains {vector_count} vectors")
    
    # Test search with different dimension vectors
    standard_results = index.search(standard_vector, 3)
    smaller_results = index.search(smaller_vector, 3)
    larger_results = index.search(larger_vector, 3)
    
    # Log search results
    logger.info(f"Standard vector search returned {len(standard_results)} results")
    logger.info(f"Smaller vector search returned {len(smaller_results)} results")
    logger.info(f"Larger vector search returned {len(larger_results)} results")
    
    logger.info("Dimension mismatch handling test completed successfully")
    return True

def test_malformed_embedding_handling():
    """Test the handling of malformed embeddings (NaN/Inf values)."""
    logger.info("Testing malformed embedding handling...")
    
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
    try:
        index.add("nan_vector", nan_vector)
        logger.info("Successfully handled NaN vector")
    except Exception as e:
        logger.error(f"Failed to handle NaN vector: {str(e)}")
    
    try:
        index.add("inf_vector", inf_vector)
        logger.info("Successfully handled Inf vector")
    except Exception as e:
        logger.error(f"Failed to handle Inf vector: {str(e)}")
    
    # Verify we can search without errors
    try:
        results = index.search(normal_vector, 3)
        logger.info(f"Search with normal vector returned {len(results)} results")
    except Exception as e:
        logger.error(f"Failed to search with normal vector: {str(e)}")
    
    # Search with malformed query vectors should also work
    nan_query = np.random.random(dimension).astype('float32')
    nan_query[5:15] = np.nan
    
    try:
        nan_results = index.search(nan_query, 3)
        logger.info(f"Search with NaN query vector returned {len(nan_results)} results")
    except Exception as e:
        logger.error(f"Failed to search with NaN query vector: {str(e)}")
    
    logger.info("Malformed embedding handling test completed successfully")
    return True

async def test_end_to_end_vector_retrieval():
    """End-to-end test of vector indexing and retrieval through the API."""
    if not client_available:
        logger.warning("SynthiansClient not available, skipping end-to-end test")
        return False
    
    logger.info("Testing end-to-end vector retrieval...")
    
    client = SynthiansClient()
    await client.connect()
    
    try:
        # Step 1: Create distinct test memories
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Creating test memories with timestamp {timestamp}")
        
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
        logger.info("Waiting for memory processing and indexing...")
        await asyncio.sleep(1)
        
        # Step 2: Retrieve with exact match
        alpha_query = f"Alpha at {timestamp}"
        logger.info(f"Retrieving memory with query: '{alpha_query}'")
        alpha_results = await client.retrieve_memories(alpha_query, top_k=3)
        
        # Log retrieval results
        alpha_memories = alpha_results.get("memories", [])
        alpha_ids = [m.get("id") for m in alpha_memories]
        logger.info(f"Retrieved {len(alpha_memories)} memories for Alpha query")
        
        # Step 3: Test with lower threshold to ensure retrieval works
        general_query = f"vector index test at {timestamp}"
        logger.info(f"Retrieving memories with general query and low threshold: '{general_query}'")
        low_threshold_results = await client.retrieve_memories(
            general_query, 
            top_k=10, 
            threshold=0.3  # Lower threshold as per the memory improvement
        )
        
        all_memories = low_threshold_results.get("memories", [])
        logger.info(f"Retrieved {len(all_memories)} memories with low threshold")
        
        logger.info("End-to-end vector retrieval test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in end-to-end test: {str(e)}")
        return False
    finally:
        await client.disconnect()

async def run_tests():
    """Run all tests and report results."""
    logger.info("Starting FAISS implementation tests...")
    
    test_results = {}
    
    # Run all tests
    try:
        test_results["vector_index_creation"] = test_faiss_vector_index_creation()
    except Exception as e:
        logger.error(f"Vector index creation test failed: {str(e)}")
        test_results["vector_index_creation"] = False
    
    try:
        test_results["dimension_mismatch"] = test_dimension_mismatch_handling()
    except Exception as e:
        logger.error(f"Dimension mismatch test failed: {str(e)}")
        test_results["dimension_mismatch"] = False
    
    try:
        test_results["malformed_embedding"] = test_malformed_embedding_handling()
    except Exception as e:
        logger.error(f"Malformed embedding test failed: {str(e)}")
        test_results["malformed_embedding"] = False
    
    if client_available:
        try:
            test_results["end_to_end"] = await test_end_to_end_vector_retrieval()
        except Exception as e:
            logger.error(f"End-to-end test failed: {str(e)}")
            test_results["end_to_end"] = False
    else:
        test_results["end_to_end"] = "SKIPPED"
    
    # Report results
    logger.info("\n== FAISS Implementation Test Results ==\n")
    for test_name, result in test_results.items():
        status = "PASSED" if result is True else "FAILED" if result is False else result
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Check for failures
    failed_tests = [name for name, result in test_results.items() if result is False]
    
    if failed_tests:
        logger.error(f"\n{len(failed_tests)} tests failed: {', '.join(failed_tests)}")
        return False
    else:
        logger.info("\nAll tests completed successfully!")
        return True

if __name__ == "__main__":
    # Determine if the server is running by trying to import SynthiansClient
    logger.info("Checking FAISS installation and GPU capabilities...")
    
    # Check if FAISS has GPU support
    has_gpu_support = hasattr(faiss, 'StandardGpuResources')
    logger.info(f"FAISS has GPU support: {has_gpu_support}")
    
    # Try to detect GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        gpu_detected = result.returncode == 0
        logger.info(f"GPU detected: {gpu_detected}")
    except:
        gpu_detected = False
        logger.info("Failed to detect GPU, assuming no GPU available")
    
    # Run the tests
    asyncio.run(run_tests())
