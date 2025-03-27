#!/usr/bin/env python

import logging
import numpy as np
import time
import uuid
import sys
import json
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_validation")

# Try to import the client
try:
    from synthians_memory_core.api.client.client import SynthiansClient
    client_available = True
except ImportError:
    logger.warning("SynthiansClient not available, some tests will be skipped")
    client_available = False

class MemorySystemValidator:
    """End-to-end validation of the memory system with FAISS integration."""
    
    def __init__(self):
        self.client = None
        self.test_memories = []
        
    async def connect(self):
        """Connect to the memory system API."""
        if not client_available:
            logger.error("SynthiansClient not available")
            return False
            
        try:
            self.client = SynthiansClient()
            await self.client.connect()
            logger.info("Connected to memory system API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to memory system API: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from the memory system API."""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from memory system API")
    
    async def test_basic_memory_flow(self):
        """Test the basic memory creation and retrieval flow."""
        logger.info("\n=== Testing Basic Memory Flow ===")
        
        try:
            # Create a unique memory with timestamp
            timestamp = datetime.now().isoformat()
            content = f"Testing memory system with FAISS integration at {timestamp}"
            
            logger.info(f"Creating memory: {content}")
            response = await self.client.process_memory(
                content=content,
                metadata={"test_type": "faiss_integration", "timestamp": timestamp}
            )
            
            if not response.get("success"):
                logger.error(f"Failed to create memory: {response.get('error')}")
                return False
                
            memory_id = response.get("memory_id")
            self.test_memories.append((memory_id, content))
            logger.info(f"Created memory with ID: {memory_id}")
            
            # Wait for indexing
            logger.info("Waiting for memory to be indexed...")
            await asyncio.sleep(1)
            
            # Retrieve the memory
            query = f"FAISS integration at {timestamp}"
            logger.info(f"Retrieving memory with query: '{query}'")
            
            retrieval_response = await self.client.retrieve_memories(
                query=query, 
                top_k=5,
                threshold=0.3  # Uses the lower threshold as per memory improvements
            )
            
            if not retrieval_response.get("success"):
                logger.error(f"Failed to retrieve memory: {retrieval_response.get('error')}")
                return False
                
            memories = retrieval_response.get("memories", [])
            logger.info(f"Retrieved {len(memories)} memories")
            
            # Check if our memory was retrieved
            retrieved_ids = [m.get("id") for m in memories]
            if memory_id not in retrieved_ids:
                logger.error(f"Memory {memory_id} was not retrieved")
                return False
                
            logger.info("Basic memory flow test passed")
            return True
        except Exception as e:
            logger.error(f"Basic memory flow test failed: {str(e)}")
            return False
    
    async def test_dimension_mismatch_handling(self):
        """Test handling of embeddings with different dimensions."""
        logger.info("\n=== Testing Dimension Mismatch Handling ===")
        
        try:
            # We'll create memories that will result in different embedding dimensions
            # The system should handle this gracefully
            
            # Create a unique identifier
            timestamp = datetime.now().isoformat()
            
            # Create test memories with content that will produce different embeddings
            memories_to_create = [
                # Short text (likely to produce smaller embedding)
                f"Short test {timestamp}",
                
                # Medium text
                f"This is a medium length test for dimension mismatch handling at {timestamp}",
                
                # Long text (likely to produce larger embedding)
                f"This is a very long test text for dimension mismatch handling. It contains multiple sentences and should result in a different embedding dimension compared to the shorter texts. The memory system should handle this gracefully by aligning vectors as needed. Testing at {timestamp}."
            ]
            
            # Create memories
            for i, content in enumerate(memories_to_create):
                logger.info(f"Creating test memory {i+1} with different length...")
                response = await self.client.process_memory(
                    content=content,
                    metadata={"test_type": "dimension_mismatch", "memory_number": i, "timestamp": timestamp}
                )
                
                if not response.get("success"):
                    logger.error(f"Failed to create memory {i+1}: {response.get('error')}")
                    return False
                    
                memory_id = response.get("memory_id")
                self.test_memories.append((memory_id, content))
                logger.info(f"Created memory {i+1} with ID: {memory_id}")
            
            # Wait for indexing
            logger.info("Waiting for memories to be indexed...")
            await asyncio.sleep(1)
            
            # Retrieve memories with a common query term
            query = f"test {timestamp}"
            logger.info(f"Retrieving memories with query: '{query}'")
            
            retrieval_response = await self.client.retrieve_memories(
                query=query, 
                top_k=5,
                threshold=0.3
            )
            
            if not retrieval_response.get("success"):
                logger.error(f"Failed to retrieve memories: {retrieval_response.get('error')}")
                return False
                
            memories = retrieval_response.get("memories", [])
            logger.info(f"Retrieved {len(memories)} memories")
            
            # Check if our memories were retrieved
            retrieved_ids = [m.get("id") for m in memories]
            success = True
            
            for memory_id, content in self.test_memories[-3:]:  # The last 3 memories we created
                if memory_id not in retrieved_ids:
                    logger.error(f"Memory {memory_id} was not retrieved")
                    success = False
            
            if success:
                logger.info("Dimension mismatch handling test passed")
            return success
        except Exception as e:
            logger.error(f"Dimension mismatch handling test failed: {str(e)}")
            return False
    
    async def test_malformed_embedding_handling(self):
        """Test handling of malformed embeddings."""
        logger.info("\n=== Testing Malformed Embedding Handling ===")
        
        try:
            # We can't directly create malformed embeddings through the API
            # But we can test that the API doesn't crash when processing potentially problematic text
            
            # Create a unique identifier
            timestamp = datetime.now().isoformat()
            
            # Create unusual test memories that might cause embedding issues
            memories_to_create = [
                # Memory with unusual Unicode characters
                f"Unicode test with special chars: \u2620\u2764\u263A\u2639 at {timestamp}",
                
                # Memory with repeated content (might cause normalization issues)
                f"{'A' * 100} repeated text test at {timestamp}",
                
                # Memory with numeric content
                f"Numeric content test: {' '.join([str(i) for i in range(100)])} at {timestamp}"
            ]
            
            # Create memories
            for i, content in enumerate(memories_to_create):
                logger.info(f"Creating test memory {i+1} with unusual content...")
                response = await self.client.process_memory(
                    content=content,
                    metadata={"test_type": "malformed_embedding", "memory_number": i, "timestamp": timestamp}
                )
                
                if not response.get("success"):
                    logger.error(f"Failed to create memory {i+1}: {response.get('error')}")
                    return False
                    
                memory_id = response.get("memory_id")
                self.test_memories.append((memory_id, content))
                logger.info(f"Created memory {i+1} with ID: {memory_id}")
            
            # Wait for indexing
            logger.info("Waiting for memories to be indexed...")
            await asyncio.sleep(1)
            
            # Retrieve memories with a common query term
            query = f"test at {timestamp}"
            logger.info(f"Retrieving memories with query: '{query}'")
            
            retrieval_response = await self.client.retrieve_memories(
                query=query, 
                top_k=5,
                threshold=0.3
            )
            
            if not retrieval_response.get("success"):
                logger.error(f"Failed to retrieve memories: {retrieval_response.get('error')}")
                return False
                
            memories = retrieval_response.get("memories", [])
            logger.info(f"Retrieved {len(memories)} memories")
            
            # Success criteria: the API didn't crash and returned a valid response
            logger.info("Malformed embedding handling test passed")
            return True
        except Exception as e:
            logger.error(f"Malformed embedding handling test failed: {str(e)}")
            return False
    
    async def test_retrieval_with_threshold(self):
        """Test memory retrieval with different thresholds."""
        logger.info("\n=== Testing Retrieval with Different Thresholds ===")
        
        try:
            # Create a unique memory
            timestamp = datetime.now().isoformat()
            unique_id = str(uuid.uuid4())[:8]
            content = f"Unique threshold test memory {unique_id} at {timestamp}"
            
            logger.info(f"Creating memory: {content}")
            response = await self.client.process_memory(
                content=content,
                metadata={"test_type": "threshold_test", "timestamp": timestamp}
            )
            
            if not response.get("success"):
                logger.error(f"Failed to create memory: {response.get('error')}")
                return False
                
            memory_id = response.get("memory_id")
            self.test_memories.append((memory_id, content))
            logger.info(f"Created memory with ID: {memory_id}")
            
            # Wait for indexing
            logger.info("Waiting for memory to be indexed...")
            await asyncio.sleep(1)
            
            # First retrieval with high threshold
            query = f"partially related query with {unique_id}"
            logger.info(f"Retrieving with high threshold (0.9) using query: '{query}'")
            
            high_threshold_response = await self.client.retrieve_memories(
                query=query, 
                top_k=5,
                threshold=0.9  # High threshold should filter out most memories
            )
            
            if not high_threshold_response.get("success"):
                logger.error(f"High threshold retrieval failed: {high_threshold_response.get('error')}")
                return False
                
            high_threshold_memories = high_threshold_response.get("memories", [])
            logger.info(f"High threshold (0.9) retrieved {len(high_threshold_memories)} memories")
            
            # Second retrieval with low threshold
            logger.info(f"Retrieving with low threshold (0.3) using same query")
            
            low_threshold_response = await self.client.retrieve_memories(
                query=query, 
                top_k=5,
                threshold=0.3  # Low threshold should include more memories
            )
            
            if not low_threshold_response.get("success"):
                logger.error(f"Low threshold retrieval failed: {low_threshold_response.get('error')}")
                return False
                
            low_threshold_memories = low_threshold_response.get("memories", [])
            logger.info(f"Low threshold (0.3) retrieved {len(low_threshold_memories)} memories")
            
            # Check that low threshold returns more (or equal) memories
            if len(low_threshold_memories) < len(high_threshold_memories):
                logger.error("Lower threshold unexpectedly returned fewer memories")
                return False
                
            logger.info("Threshold test passed")
            return True
        except Exception as e:
            logger.error(f"Threshold test failed: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("\n===== STARTING MEMORY SYSTEM VALIDATION =====\n")
        
        # Connect to the memory system
        if not await self.connect():
            logger.error("Failed to connect to memory system")
            return False
        
        try:
            # Run all tests
            test_results = {
                "basic_memory_flow": await self.test_basic_memory_flow(),
                "dimension_mismatch": await self.test_dimension_mismatch_handling(),
                "malformed_embedding": await self.test_malformed_embedding_handling(),
                "threshold_retrieval": await self.test_retrieval_with_threshold()
            }
            
            # Report results
            logger.info("\n===== TEST RESULTS =====")
            for test_name, result in test_results.items():
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            
            # Final status
            if all(test_results.values()):
                logger.info("\nALL TESTS PASSED")
                return True
            else:
                failed = [name for name, result in test_results.items() if not result]
                logger.error(f"\n{len(failed)} TESTS FAILED: {', '.join(failed)}")
                return False
        finally:
            # Always disconnect
            await self.disconnect()

async def check_faiss_gpu_support():
    """Check if FAISS is installed with GPU support."""
    logger.info("=== Checking FAISS Installation and GPU Support ===\n")
    
    try:
        # Import FAISS
        import faiss
        logger.info(f"FAISS version {getattr(faiss, '__version__', 'unknown')} is installed")
        
        # Check if FAISS has GPU support
        has_gpu_support = hasattr(faiss, 'StandardGpuResources')
        logger.info(f"FAISS has GPU support: {has_gpu_support}")
        
        if not has_gpu_support:
            logger.warning("This FAISS installation does not have GPU support")
            return False
        
        # Try to create a small GPU index to verify
        logger.info("Creating a test GPU index...")
        dimension = 64  # Small dimension for quick test
        gpu_res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        
        # Add a single vector to verify it works
        test_vector = np.random.random((1, dimension)).astype('float32')
        gpu_index.add(test_vector)
        
        logger.info("Successfully created GPU index and added a test vector")
        logger.info("FAISS GPU support is working correctly")
        return True
    except ImportError:
        logger.error("FAISS is not installed")
        return False
    except Exception as e:
        logger.error(f"Error testing FAISS GPU support: {str(e)}")
        return False

async def main():
    """Run all validation tests."""
    # First check if FAISS is installed with GPU support
    faiss_gpu_working = await check_faiss_gpu_support()
    
    if not faiss_gpu_working:
        logger.warning("FAISS GPU support test failed, but continuing with memory system tests")
    
    # Run memory system tests
    validator = MemorySystemValidator()
    success = await validator.run_all_tests()
    
    if success and faiss_gpu_working:
        logger.info("\n===== ALL TESTS PASSED INCLUDING FAISS GPU SUPPORT =====")
        return 0
    elif success:
        logger.warning("\n===== MEMORY TESTS PASSED BUT FAISS GPU SUPPORT TEST FAILED =====")
        return 1
    elif faiss_gpu_working:
        logger.error("\n===== MEMORY TESTS FAILED BUT FAISS GPU SUPPORT TEST PASSED =====")
        return 2
    else:
        logger.error("\n===== ALL TESTS FAILED =====")
        return 3

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
