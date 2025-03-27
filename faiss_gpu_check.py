#!/usr/bin/env python

import logging
import numpy as np
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("faiss_gpu_check")

def main():
    logger.info("=== FAISS GPU Support Verification ===\n")
    
    # Step 1: Check if FAISS is installed
    logger.info("Checking FAISS installation...")
    try:
        import faiss
        logger.info(f"FAISS version {getattr(faiss, '__version__', 'unknown')} installed successfully")
    except ImportError:
        logger.error("FAISS is not installed. Please install it with pip install faiss-gpu or faiss-cpu")
        return 1
    
    # Step 2: Check if FAISS was built with GPU support
    has_gpu_support = hasattr(faiss, 'StandardGpuResources')
    logger.info(f"FAISS has GPU support: {has_gpu_support}")
    
    if not has_gpu_support:
        logger.warning("This FAISS installation does not have GPU support")
        return 0
    
    # Step 3: Try to create a simple GPU index
    logger.info("\nTesting GPU index creation...")
    try:
        # Create a small random dataset
        dimension = 64  # Small dimension for quick test
        num_vectors = 1000
        logger.info(f"Creating random dataset with {num_vectors} vectors of dimension {dimension}")
        
        # Generate random vectors
        vectors = np.random.random((num_vectors, dimension)).astype('float32')
        logger.info("Dataset created successfully")
        
        # Create CPU index for comparison
        cpu_index = faiss.IndexFlatL2(dimension)
        cpu_start = time.time()
        cpu_index.add(vectors)
        cpu_time = time.time() - cpu_start
        logger.info(f"Added vectors to CPU index in {cpu_time:.4f} seconds")
        
        # Try to create a GPU index
        logger.info("Creating GPU resources...")
        gpu_res = faiss.StandardGpuResources()
        
        logger.info("Creating GPU index...")
        gpu_index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, gpu_index)
        
        logger.info("Adding vectors to GPU index...")
        gpu_start = time.time()
        gpu_index.add(vectors)
        gpu_time = time.time() - gpu_start
        logger.info(f"Added vectors to GPU index in {gpu_time:.4f} seconds")
        
        # Compare performance
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        logger.info(f"\nPerformance comparison:\n  CPU time: {cpu_time:.4f}s\n  GPU time: {gpu_time:.4f}s\n  Speedup: {speedup:.2f}x")
        
        # Test search
        logger.info("\nTesting search functionality...")
        query = np.random.random((1, dimension)).astype('float32')
        
        # CPU search
        cpu_search_start = time.time()
        cpu_distances, cpu_indices = cpu_index.search(query, 5)
        cpu_search_time = time.time() - cpu_search_start
        logger.info(f"CPU search completed in {cpu_search_time:.4f}s")
        
        # GPU search
        gpu_search_start = time.time()
        gpu_distances, gpu_indices = gpu_index.search(query, 5)
        gpu_search_time = time.time() - gpu_search_start
        logger.info(f"GPU search completed in {gpu_search_time:.4f}s")
        
        # Compare search performance
        search_speedup = cpu_search_time / gpu_search_time if gpu_search_time > 0 else 0
        logger.info(f"\nSearch performance comparison:\n  CPU time: {cpu_search_time:.4f}s\n  GPU time: {gpu_search_time:.4f}s\n  Speedup: {search_speedup:.2f}x")
        
        logger.info("\n=== GPU TEST SUCCESSFUL ===\n")
        logger.info("FAISS GPU support is working correctly")
        
        return 0
    except Exception as e:
        logger.error(f"GPU index creation failed: {str(e)}")
        logger.error("FAISS GPU support is not working correctly")
        
        # Print the full error traceback for debugging
        import traceback
        logger.error("Error traceback:")
        logger.error(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
