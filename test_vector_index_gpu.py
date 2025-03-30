import numpy as np
import time

# Test function to check if MemoryVectorIndex is using GPU
def test_faiss_gpu_integration():
    print("===== Testing FAISS GPU Integration with MemoryVectorIndex =====")
    from synthians_memory_core.vector_index import MemoryVectorIndex
    
    # Create a test index with GPU enabled
    config = {
        'embedding_dim': 768,
        'storage_path': '/tmp/test_faiss',
        'index_type': 'L2',
        'use_gpu': True
    }
    
    # Initialize the index
    print("Initializing vector index with GPU support...")
    index = MemoryVectorIndex(config)
    
    # Check if GPU was initialized
    print(f"Index is using GPU: {index.is_using_gpu}")
    
    # Generate random test data
    num_vectors = 10000
    dimension = 768
    print(f"Generating {num_vectors} random test vectors...")
    test_vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    
    # Add vectors to the index
    print("Adding vectors to index...")
    start_time = time.time()
    for i in range(num_vectors):
        index.add(f"test_id_{i}", test_vectors[i])
    add_time = time.time() - start_time
    print(f"Added {num_vectors} vectors in {add_time:.2f} seconds")
    
    # Perform search
    print("Performing search test...")
    query = np.random.random(dimension).astype(np.float32)
    start_time = time.time()
    results = index.search(query, k=10)
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.4f} seconds")
    print(f"Found {len(results)} results")
    
    print("===== Test Completed =====")

if __name__ == "__main__":
    test_faiss_gpu_integration()
