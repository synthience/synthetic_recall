#!/usr/bin/env python

import os
import sys
import time
import logging
import numpy as np
import asyncio
import signal
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faiss_integration_test")

# Set a timeout for operations that might hang
DEFAULT_TIMEOUT = 30  # seconds

# Define timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Import FAISS
try:
    import faiss
    logger.info(f"FAISS version {getattr(faiss, '__version__', 'unknown')} loaded successfully")
    logger.info(f"FAISS has GPU support: {hasattr(faiss, 'StandardGpuResources')}")
except ImportError:
    logger.error("FAISS not found. Tests cannot proceed.")
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


class FAISSIntegrationTest:
    """Test suite for FAISS vector index implementation"""
    
    def __init__(self, use_gpu=True):
        self.test_results = {}
        self.test_dir = os.path.join(os.getcwd(), 'test_index')
        os.makedirs(self.test_dir, exist_ok=True)
        self.use_gpu = use_gpu
        logger.info(f"Test initialized with use_gpu={use_gpu}")
    
    def run_tests(self):
        """Run all tests and report results"""
        logger.info("\n===== STARTING FAISS INTEGRATION TESTS =====")
        
        # Run all tests
        self.test_results["basic_functionality"] = self.test_basic_functionality()
        self.test_results["dimension_mismatch"] = self.test_dimension_mismatch()
        self.test_results["malformed_embeddings"] = self.test_malformed_embeddings()
        self.test_results["persistence"] = self.test_persistence()
        
        # Report results
        logger.info("\n===== TEST RESULTS =====")
        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        # Final status
        if all(self.test_results.values()):
            logger.info("\nu2705 ALL TESTS PASSED u2705")
            return True
        else:
            failed = [name for name, result in self.test_results.items() if not result]
            logger.error(f"\nu274c {len(failed)} TESTS FAILED: {', '.join(failed)} u274c")
            return False
    
    def test_basic_functionality(self):
        """Test basic FAISS vector index functionality"""
        logger.info("\n----- Testing Basic Functionality -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create vector index
            dimension = 768
            logger.info("Creating vector index...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            logger.info(f"Created index with dimension {dimension}, GPU usage: {index.is_using_gpu}")
            
            # Add vectors
            vectors_to_add = 50  # Reduced from 100 to speed up tests
            logger.info(f"Adding {vectors_to_add} vectors to index...")
            start_time = time.time()
            for i in range(vectors_to_add):
                memory_id = f"test_{i}"
                vector = np.random.random(dimension).astype('float32')
                index.add(memory_id, vector)
                # Log progress for every 10 vectors
                if i % 10 == 0 and i > 0:
                    logger.info(f"Added {i} vectors so far...")
            
            add_time = time.time() - start_time
            logger.info(f"Added {vectors_to_add} vectors in {add_time:.4f}s ({vectors_to_add/add_time:.2f} vectors/s)")
            
            # Search vectors
            logger.info("Searching for similar vectors...")
            query = np.random.random(dimension).astype('float32')
            search_start = time.time()
            results = index.search(query, 5)  # Reduced from 10
            search_time = time.time() - search_start
            
            logger.info(f"Search completed in {search_time:.4f}s, returned {len(results)} results")
            if results:
                logger.info(f"First result: {results[0]}")
            
            # Verify count
            logger.info("Verifying vector count...")
            count = index.count()
            logger.info(f"Index count: {count}, expected: {vectors_to_add}")
            assert count == vectors_to_add, f"Expected {vectors_to_add} vectors, got {count}"
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Basic functionality test passed")
            return True
        except TimeoutError:
            logger.error("Basic functionality test timed out")
            return False
        except Exception as e:
            logger.error(f"Basic functionality test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_dimension_mismatch(self):
        """Test handling of vectors with different dimensions"""
        logger.info("\n----- Testing Dimension Mismatch Handling -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create index with specific dimension
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Test vectors with different dimensions
            dimensions = {
                'smaller': 384,   # Common dimension mismatch case
                'standard': dimension,
                'larger': 1024
            }
            
            # Add vectors with different dimensions
            for name, dim in dimensions.items():
                logger.info(f"Testing {name} vector with dimension {dim}...")
                vector = np.random.random(dim).astype('float32')
                try:
                    index.add(f"vector_{name}", vector)
                    logger.info(f"Successfully added {name} vector with dimension {dim}")
                except Exception as e:
                    logger.error(f"Failed to add {name} vector: {str(e)}")
                    signal.alarm(0)
                    return False
            
            # Search with different dimension vectors
            for name, dim in dimensions.items():
                logger.info(f"Searching with {name} vector ({dim} dimensions)...")
                query = np.random.random(dim).astype('float32')
                try:
                    results = index.search(query, 3)
                    logger.info(f"Successfully searched with {name} vector, got {len(results)} results")
                except Exception as e:
                    logger.error(f"Failed to search with {name} vector: {str(e)}")
                    signal.alarm(0)
                    return False
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Dimension mismatch test passed")
            return True
        except TimeoutError:
            logger.error("Dimension mismatch test timed out")
            return False
        except Exception as e:
            logger.error(f"Dimension mismatch test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_malformed_embeddings(self):
        """Test handling of malformed embeddings (NaN/Inf)"""
        logger.info("\n----- Testing Malformed Embedding Handling -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create index
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Create test vectors
            normal = np.random.random(dimension).astype('float32')
            
            # Vector with NaN values
            nan_vector = np.random.random(dimension).astype('float32')
            nan_vector[10:20] = np.nan
            
            # Vector with Inf values
            inf_vector = np.random.random(dimension).astype('float32')
            inf_vector[30:40] = np.inf
            
            # Mixed vector
            mixed_vector = np.random.random(dimension).astype('float32')
            mixed_vector[5:10] = np.nan
            mixed_vector[50:55] = np.inf
            
            # Add vectors
            test_vectors = {
                'normal': normal,
                'nan': nan_vector,
                'inf': inf_vector,
                'mixed': mixed_vector
            }
            
            for name, vector in test_vectors.items():
                logger.info(f"Testing {name} vector...")
                try:
                    index.add(f"vector_{name}", vector)
                    logger.info(f"Successfully added {name} vector")
                except Exception as e:
                    logger.error(f"Failed to add {name} vector: {str(e)}")
                    if name == 'normal':  # Normal vectors must be added successfully
                        signal.alarm(0)
                        return False
            
            # Search with malformed query vectors
            for name, vector in test_vectors.items():
                logger.info(f"Searching with {name} vector...")
                try:
                    results = index.search(vector, 3)
                    logger.info(f"Successfully searched with {name} vector, got {len(results)} results")
                except Exception as e:
                    logger.error(f"Failed to search with {name} vector: {str(e)}")
                    if name == 'normal':  # Normal vectors must be searchable
                        signal.alarm(0)
                        return False
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Malformed embedding test passed")
            return True
        except TimeoutError:
            logger.error("Malformed embedding test timed out")
            return False
        except Exception as e:
            logger.error(f"Malformed embedding test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_persistence(self):
        """Test index persistence (save/load)"""
        logger.info("\n----- Testing Index Persistence -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create and populate index
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Add vectors with known IDs
            vectors_to_add = 20  # Reduced from 50
            known_ids = []
            logger.info(f"Adding {vectors_to_add} vectors to index...")
            
            for i in range(vectors_to_add):
                memory_id = f"persistent_{i}"
                known_ids.append(memory_id)
                vector = np.random.random(dimension).astype('float32')
                index.add(memory_id, vector)
            
            # Save index
            index_path = os.path.join(self.test_dir, 'persistence_test.faiss')
            logger.info(f"Saving index to {index_path}...")
            index.save(index_path)
            logger.info(f"Saved index to {index_path}")
            
            # Create new index and load
            logger.info("Creating new index and loading saved data...")
            new_index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            new_index.load(index_path)
            logger.info(f"Loaded index with {new_index.count()} vectors")
            
            # Verify counts match
            logger.info("Verifying vector counts match...")
            assert new_index.count() == index.count(), "Vector counts don't match after loading"
            
            # Clean up
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"Cleaned up test index file {index_path}")
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Persistence test passed")
            return True
        except TimeoutError:
            logger.error("Persistence test timed out")
            return False
        except Exception as e:
            logger.error(f"Persistence test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False

async def test_api_integration():
    """Test integration with the memory API"""
    logger.info("\n----- Testing API Integration -----")
    
    if not client_available:
        logger.warning("SynthiansClient not available, skipping API test")
        return False
    
    try:
        # Set timeout for operations
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(DEFAULT_TIMEOUT)
        
        logger.info("Connecting to API...")
        client = SynthiansClient()
        await client.connect()
        
        # Create unique test memories
        timestamp = datetime.now().isoformat()
        unique_prefix = f"faiss_test_{timestamp}"
        
        logger.info(f"Creating test memories with prefix: {unique_prefix}")
        
        # Create memories
        memories = []
        for i in range(3):
            content = f"{unique_prefix} Memory {i}: This is a test memory for FAISS integration testing"
            logger.info(f"Creating memory {i}...")
            response = await client.process_memory(
                content=content,
                metadata={"test_type": "faiss_integration", "memory_number": i}
            )
            
            if response.get("success"):
                memory_id = response.get("memory_id")
                memories.append((memory_id, content))
                logger.info(f"Created memory {i} with ID: {memory_id}")
            else:
                logger.error(f"Failed to create memory {i}: {response.get('error')}")
        
        # Wait for indexing
        logger.info("Waiting for memories to be indexed...")
        await asyncio.sleep(1)
        
        # Retrieve memories
        query = unique_prefix
        logger.info(f"Retrieving memories with query: '{query}'")
        
        response = await client.retrieve_memories(query, top_k=5, threshold=0.2)
        
        if not response.get("success"):
            logger.error(f"Retrieval failed: {response.get('error')}")
            signal.alarm(0)
            return False
        
        results = response.get("memories", [])
        retrieved_ids = [m.get("id") for m in results]
        
        logger.info(f"Retrieved {len(results)} memories")
        
        # Verify that our memories were retrieved
        success = True
        for memory_id, _ in memories:
            if memory_id not in retrieved_ids:
                logger.error(f"Memory {memory_id} was not retrieved")
                success = False
        
        # Display similarity scores
        if results:
            logger.info("Similarity scores:")
            for memory in results:
                logger.info(f"  {memory.get('id')}: {memory.get('similarity_score', 'N/A')}")
        
        # Test with lower threshold
        logger.info("Testing with lower threshold (0.3)...")
        low_threshold_response = await client.retrieve_memories(
            query, top_k=5, threshold=0.3
        )
        
        low_results = low_threshold_response.get("memories", [])
        logger.info(f"Retrieved {len(low_results)} memories with lower threshold")
        
        await client.disconnect()
        
        # Cancel timeout
        signal.alarm(0)
        
        if success:
            logger.info("API integration test passed")
        else:
            logger.error("API integration test failed - not all memories were retrieved")
        
        return success
    except TimeoutError:
        logger.error("API integration test timed out")
        return False
    except Exception as e:
        logger.error(f"API integration test failed: {str(e)}")
        # Cancel timeout in case of exception
        signal.alarm(0)
        return False

async def main():
    # Run tests with and without GPU
    logger.info("\n===== FIRST RUNNING TESTS WITH CPU ONLY =====\n")
    cpu_test_suite = FAISSIntegrationTest(use_gpu=False)
    cpu_success = cpu_test_suite.run_tests()
    
    # Only try GPU if CPU tests pass
    if cpu_success:
        logger.info("\n===== NOW RUNNING TESTS WITH GPU =====\n")
        gpu_test_suite = FAISSIntegrationTest(use_gpu=True)
        gpu_success = gpu_test_suite.run_tests()
    else:
        logger.warning("Skipping GPU tests because CPU tests failed")
        gpu_success = False
    
    # Run API integration test
    api_success = await test_api_integration()
    
    if cpu_success and gpu_success and api_success:
        logger.info("\u2705 ALL TESTS PASSED INCLUDING GPU AND API INTEGRATION \u2705")
        return 0
    elif cpu_success and api_success:
        logger.warning("\u26a0ufe0f CPU AND API TESTS PASSED BUT GPU TESTS FAILED \u26a0ufe0f")
        return 1
    elif cpu_success:
        logger.warning("\u26a0ufe0f CPU TESTS PASSED BUT GPU AND API TESTS FAILED \u26a0ufe0f")
        return 2
    else:
        logger.error("\u274c ALL TESTS FAILED \u274c")
        return 3

if __name__ == "__main__":
    # Try to fix SIGALRM not available on Windows
    if sys.platform == "win32":
        logger.warning("Timeout functionality not available on Windows, disabling timeouts")
        # Define dummy functions
        def timeout_handler(signum, frame):
            pass
        signal.SIGALRM = signal.SIGTERM  # Just a placeholder
        signal.alarm = lambda x: None    # No-op function
    
    sys.exit(asyncio.run(main()))
