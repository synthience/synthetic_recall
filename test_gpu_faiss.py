import os
import sys
import logging
import numpy as np
import time
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FAISS-GPU-Test")

def test_faiss_gpu_support():
    """Test FAISS GPU support by running vector searches with and without GPU."""
    # First, run the GPU setup script to ensure the right FAISS package is installed
    try:
        logger.info("Running GPU setup script...")
        # Import the setup module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from synthians_memory_core.gpu_setup import check_gpu_and_install_faiss
        
        # Run the setup
        gpu_available, faiss_type = check_gpu_and_install_faiss()
        logger.info(f"GPU availability: {gpu_available}, FAISS type: {faiss_type}")
    except Exception as e:
        logger.error(f"Error during GPU setup: {str(e)}")
        return
    
    # Try to import FAISS
    try:
        import faiss
        logger.info(f"Successfully imported FAISS: {faiss.__version__}")
    except ImportError:
        logger.error("Failed to import FAISS. Please install it using pip.")
        return
    
    # Create a test dataset
    dim = 128    # Embedding dimension
    n_vectors = 10000  # Number of vectors to index
    n_queries = 100    # Number of queries to run
    k = 10        # Number of nearest neighbors to retrieve
    
    # Generate random vectors for testing
    logger.info(f"Generating {n_vectors} random vectors with dimension {dim}...")
    vectors = np.random.random((n_vectors, dim)).astype('float32')
    queries = np.random.random((n_queries, dim)).astype('float32')
    
    # Test CPU index
    logger.info("Testing CPU index...")
    cpu_start = time.time()
    
    # Create and build CPU index
    cpu_index = faiss.IndexFlatL2(dim)
    cpu_index.add(vectors)
    
    # Search using CPU index
    cpu_search_start = time.time()
    cpu_distances, cpu_indices = cpu_index.search(queries, k)
    cpu_end = time.time()
    
    cpu_total_time = cpu_end - cpu_start
    cpu_search_time = cpu_end - cpu_search_start
    logger.info(f"CPU index build+search time: {cpu_total_time:.4f}s")
    logger.info(f"CPU search time only: {cpu_search_time:.4f}s")
    
    # Test GPU index if available
    if gpu_available:
        try:
            logger.info("Testing GPU index...")
            gpu_start = time.time()
            
            # Create GPU resources
            gpu_res = faiss.StandardGpuResources()
            
            # Create and build GPU index
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss.IndexFlatL2(dim))
            gpu_index.add(vectors)
            
            # Search using GPU index
            gpu_search_start = time.time()
            gpu_distances, gpu_indices = gpu_index.search(queries, k)
            gpu_end = time.time()
            
            gpu_total_time = gpu_end - gpu_start
            gpu_search_time = gpu_end - gpu_search_start
            logger.info(f"GPU index build+search time: {gpu_total_time:.4f}s")
            logger.info(f"GPU search time only: {gpu_search_time:.4f}s")
            
            # Calculate speedup
            total_speedup = cpu_total_time / gpu_total_time
            search_speedup = cpu_search_time / gpu_search_time
            logger.info(f"GPU speedup (total): {total_speedup:.2f}x")
            logger.info(f"GPU speedup (search only): {search_speedup:.2f}x")
            
            # Verify results are similar
            if np.array_equal(cpu_indices, gpu_indices):
                logger.info("CPU and GPU indices match exactly")
            else:
                # Check if the top results are at least similar (FAISS GPU can have slight differences)
                top_matches = sum(len(set(cpu_indices[i][:3]).intersection(set(gpu_indices[i][:3]))) for i in range(n_queries))
                match_percentage = top_matches / (n_queries * 3) * 100
                logger.info(f"Top-3 result match percentage: {match_percentage:.2f}%")
        except Exception as e:
            logger.error(f"Error during GPU index testing: {str(e)}")
    else:
        logger.info("Skipping GPU index test as GPU is not available")
    
    # Import and test the MemoryVectorIndex from our implementation
    try:
        logger.info("Testing MemoryVectorIndex class...")
        from synthians_memory_core.vector_index import MemoryVectorIndex
        
        # Create a new index with GPU support if available
        vector_index = MemoryVectorIndex(dimension=dim, use_gpu=gpu_available)
        
        # Add some test vectors
        test_size = 1000
        test_vectors = np.random.random((test_size, dim)).astype('float32')
        for i in range(test_size):
            vector_index.add(f"memory_{i}", test_vectors[i])
        
        logger.info(f"Added {test_size} vectors to MemoryVectorIndex")
        
        # Run some test queries
        test_queries = np.random.random((10, dim)).astype('float32')
        for i, query in enumerate(test_queries):
            results = vector_index.search(query, 5)
            logger.info(f"Query {i}: Found {len(results)} results")
        
        # Check if GPU resources were used
        logger.info(f"MemoryVectorIndex GPU usage: {vector_index.is_using_gpu}")
        
    except Exception as e:
        logger.error(f"Error testing MemoryVectorIndex: {str(e)}")

if __name__ == "__main__":
    test_faiss_gpu_support()
