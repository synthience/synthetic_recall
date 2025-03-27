#!/usr/bin/env python
# tests/test_memory_retrieval_api.py

import os
import sys
import asyncio
import logging
import time
import json
import random
from typing import Dict, Any, List, Optional

# Add the project root to the path so we can import the client
sys.path.append("/app")

# Import the client
from synthians_memory_core.api.client.client import SynthiansClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory_api_test")

# Constants
API_BASE_URL = os.environ.get("API_URL", "http://localhost:5010")
TEST_MEMORIES = [
    "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
    "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
    "Deep Learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to analyze various factors of data.",
    "Natural Language Processing (NLP) is a field of AI that gives computers the ability to understand text and spoken words in the same way humans can.",
    "Computer Vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos."
]
TEST_QUERIES = [
    "What is AI?", 
    "Explain machine learning", 
    "How does deep learning work?", 
    "Tell me about NLP", 
    "What is computer vision?"
]

# GPU/FAISS Diagnostic Function
def check_gpu_status():
    """Check GPU and FAISS status"""
    try:
        import torch
        import faiss
        
        gpu_info = {
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "torch_cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "faiss_gpu_count": faiss.get_num_gpus(),
        }
        
        logger.info("🔍 GPU Status Check")
        logger.info(f"PyTorch CUDA available: {gpu_info['torch_cuda_available']}")
        logger.info(f"CUDA device count: {gpu_info['torch_cuda_device_count']}")
        if gpu_info['torch_cuda_available']:
            logger.info(f"CUDA device name: {gpu_info['torch_cuda_device_name']}")
        logger.info(f"FAISS GPU count: {gpu_info['faiss_gpu_count']}")
        
        # Try to create a small FAISS index on GPU
        if gpu_info['faiss_gpu_count'] > 0:
            try:
                d = 128  # Dimension
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatL2(res, d)
                logger.info(f"✅ Successfully created test GPU FAISS index")
            except Exception as e:
                logger.warning(f"Failed to create test GPU FAISS index: {str(e)}")
        
        return gpu_info
    except ImportError as e:
        logger.warning(f"Could not import GPU libraries: {str(e)}")
        return {"error": str(e)}

# Test Functions
async def test_health_and_stats(client: SynthiansClient):
    """Test health check and stats endpoints"""
    try:
        health = await client.health_check()
        logger.info(f"Health check: {health['status']}")
        
        stats = await client.get_stats()
        logger.info(f"Memory count: {stats.get('memory_count', 'N/A')}")
        logger.info(f"Vector index info: {stats.get('vector_index_info', 'N/A')}")
        
        return True
    except Exception as e:
        logger.error(f"Failed health/stats check: {str(e)}")
        return False

async def test_memory_creation(client: SynthiansClient):
    """Test memory creation via API"""
    success_count = 0
    
    logger.info("📝 Testing Memory Creation")
    for i, content in enumerate(TEST_MEMORIES):
        try:
            # Add unique identifier to track memories
            test_id = f"test_{int(time.time())}_{i}"
            metadata = {
                "test_id": test_id,
                "test_group": "api_test",
                "timestamp": time.time()
            }
            
            response = await client.process_memory(content=content, metadata=metadata)
            
            if response.get("success") and response.get("memory_id"):
                logger.info(f"✅ Memory {i+1}/{len(TEST_MEMORIES)} created successfully: {response.get('memory_id')}")
                success_count += 1
            else:
                logger.error(f"❌ Failed to create memory {i+1}: {response}")
            
            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error creating memory {i+1}: {str(e)}")
    
    success_rate = (success_count / len(TEST_MEMORIES)) * 100 if TEST_MEMORIES else 0
    logger.info(f"Memory creation success rate: {success_rate:.1f}% ({success_count}/{len(TEST_MEMORIES)})")
    
    # Get updated stats
    try:
        stats = await client.get_stats()
        logger.info(f"Updated memory count: {stats.get('memory_count', 'N/A')}")
        logger.info(f"Updated vector index info: {stats.get('vector_index_info', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to get updated stats: {str(e)}")
    
    return success_count > 0

async def test_memory_retrieval(client: SynthiansClient):
    """Test memory retrieval via API"""
    success_count = 0
    total_memories_found = 0
    
    logger.info("🔍 Testing Memory Retrieval")
    for i, query in enumerate(TEST_QUERIES):
        try:
            # Try with both regular and low threshold
            for test_run, threshold in [("Standard", None), ("Low threshold", 0.2)]:
                response = await client.retrieve_memories(
                    query=query, 
                    top_k=5,
                    threshold=threshold
                )
                
                memories = response.get("memories", [])
                memory_count = len(memories)
                total_memories_found += memory_count
                
                if response.get("success"):
                    logger.info(f"✅ {test_run} query {i+1}/{len(TEST_QUERIES)} successful: {memory_count} memories found")
                    
                    # Display similarity scores
                    if memories:
                        logger.info("Top similarity scores:")
                        for j, memory in enumerate(memories[:3]):
                            score = memory.get("metadata", {}).get("similarity_score", "N/A")
                            logger.info(f"  [{j+1}] Score: {score:.4f} - {memory.get('content', '')[:50]}...")
                    
                    if memory_count > 0:
                        success_count += 1
                else:
                    logger.error(f"❌ {test_run} query {i+1} failed: {response}")
                
                # Small delay between queries
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error during query {i+1}: {str(e)}")
    
    # Calculate metrics (2 test runs per query)
    total_runs = len(TEST_QUERIES) * 2
    success_rate = (success_count / total_runs) * 100 if total_runs else 0
    avg_memories = total_memories_found / total_runs if total_runs else 0
    
    logger.info(f"Memory retrieval success rate: {success_rate:.1f}% ({success_count}/{total_runs})")
    logger.info(f"Average memories per query: {avg_memories:.1f}")
    
    return success_count > 0

async def run_full_test_suite():
    """Run all API tests"""
    logger.info("🧠 Starting Synthians Memory API Test Suite")
    logger.info(f"Connecting to API at: {API_BASE_URL}")
    
    # Check GPU status first
    gpu_info = check_gpu_status()
    
    start_time = time.time()
    
    # Run tests through the API client
    async with SynthiansClient(base_url=API_BASE_URL) as client:
        try:
            # Test health and stats
            health_ok = await test_health_and_stats(client)
            if not health_ok:
                logger.error("Health check failed, aborting tests")
                return False
            
            # Test memory creation
            creation_ok = await test_memory_creation(client)
            if not creation_ok:
                logger.error("Memory creation failed, aborting retrieval test")
                return False
            
            # Short pause to ensure memories are indexed
            logger.info("Pausing to allow indexing to complete...")
            await asyncio.sleep(2)
            
            # Test memory retrieval
            retrieval_ok = await test_memory_retrieval(client)
            
            # Overall result
            elapsed = time.time() - start_time
            if creation_ok and retrieval_ok:
                logger.info(f"✅ ALL TESTS PASSED in {elapsed:.2f} seconds! The memory system is working correctly.")
                return True
            else:
                logger.warning(f"⚠️ TESTS PARTIALLY PASSED in {elapsed:.2f} seconds. Some components may not be working correctly.")
                return False
                
        except Exception as e:
            logger.error(f"Test suite failed with error: {str(e)}")
            return False

if __name__ == "__main__":
    # Run the test suite
    result = asyncio.run(run_full_test_suite())
    # Set exit code based on test result
    sys.exit(0 if result else 1)
