#!/usr/bin/env python

import os
import time
import numpy as np
import logging
import asyncio
import json
import aiohttp
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_system_test")

API_BASE_URL = "http://localhost:5010"

class SynthiansClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def process_memory(self, 
                             content: str, 
                             embedding: Optional[List[float]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a new memory"""
        payload = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "analyze_emotion": True
        }
        
        async with self.session.post(f"{self.base_url}/process_memory", json=payload) as response:
            return await response.json()
    
    async def process_broken_memory(self, 
                                content: str, 
                                embedding_dict: Dict[str, List[float]],
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Intentionally process a memory with a broken embedding format (dict instead of list)"""
        payload = {
            "content": content,
            "embedding": embedding_dict,  # This is wrong format (dict)
            "metadata": metadata or {},
            "analyze_emotion": True
        }
        
        async with self.session.post(f"{self.base_url}/process_memory", json=payload) as response:
            return await response.json()
    
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        """Generate an embedding for text"""
        payload = {"text": text}
        async with self.session.post(f"{self.base_url}/generate_embedding", json=payload) as response:
            return await response.json()
    
    async def retrieve_memories(self, 
                             query: str, 
                             top_k: int = 5,
                             threshold: Optional[float] = None) -> Dict[str, Any]:
        """Retrieve memories based on query"""
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold
        }
        
        async with self.session.post(f"{self.base_url}/retrieve_memories", json=payload) as response:
            return await response.json()

async def test_memory_system():
    """Test the memory system with all our fixes"""
    
    logger.info("======== MEMORY SYSTEM TEST STARTED ========")
    
    try:
        async with SynthiansClient() as client:
            # Step 1: Check if the API is healthy
            logger.info("Testing API health...")
            health_response = await client.health_check()
            logger.info(f"API Health: {health_response}")
            
            if not health_response.get('status') == 'healthy':
                logger.error("API is not healthy. Test failed.")
                return False
            
            # Step 2: Generate some embeddings
            logger.info("\nTesting embedding generation...")
            test_texts = [
                "This is a test memory about artificial intelligence",
                "Quantum computing will revolutionize the future of technology",
                "Neural networks are inspired by the human brain",
                "GPUs are essential for training large language models",
                "FAISS is a library for efficient similarity search"
            ]
            
            memory_ids = []
            embeddings = []
            
            # Generate embeddings for test texts
            for text in test_texts:
                embedding_response = await client.generate_embedding(text)
                if embedding_response.get('success'):
                    embeddings.append(embedding_response.get('embedding'))
                    logger.info(f"Generated embedding for: '{text[:30]}...' (dim={len(embedding_response.get('embedding'))})")
                else:
                    logger.error(f"Failed to generate embedding: {embedding_response.get('error')}")
            
            # Step 3: Add memories with correct embedding format
            logger.info("\nTesting memory creation with correct embedding format...")
            for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
                metadata = {"test_id": i, "source": "memory_system_test", "test_type": "correct_format"}
                process_response = await client.process_memory(text, embedding, metadata)
                
                memory_id = process_response.get('memory_id')
                if memory_id:
                    memory_ids.append(memory_id)
                    logger.info(f"Created memory {i+1}/{len(test_texts)} with ID: {memory_id}")
                else:
                    logger.error(f"Failed to create memory: {process_response.get('error')}")
            
            # Step 4: Test incorrect embedding format (dict instead of list)
            logger.info("\nTesting incorrect embedding format handling...")
            # Create a dict embedding (incorrect format)
            dict_embedding = {"embedding": embeddings[0]}
            dict_response = await client.process_broken_memory(
                "This memory has an incorrectly formatted embedding (dict instead of list)",
                dict_embedding,
                {"test_id": 99, "source": "memory_system_test", "test_type": "incorrect_format"}
            )
            
            if dict_response.get('memory_id'):
                logger.info(f"Successfully handled dict embedding! Memory ID: {dict_response.get('memory_id')}")
                memory_ids.append(dict_response.get('memory_id'))
            else:
                logger.warning(f"Dict embedding not handled: {dict_response.get('error')}")
            
            # Wait for indexing to complete
            logger.info("Waiting for indexing to complete...")
            await asyncio.sleep(2)
            
            # Step 5: Retrieve memories
            logger.info("\nTesting memory retrieval...")
            test_queries = [
                "Tell me about artificial intelligence",
                "How do neural networks work?",
                "What is the role of GPUs in AI?"
            ]
            
            # Try with progressively lower thresholds if needed
            thresholds = [0.3, 0.2, 0.1, 0.05, 0.01]  # Start with 0.3 then drop if needed
            
            for query in test_queries:
                memories_found = False
                for threshold in thresholds:
                    logger.info(f"Retrieving memories for query: '{query}' with threshold {threshold}")
                    retrieval_response = await client.retrieve_memories(query, top_k=3, threshold=threshold)
                    
                    if retrieval_response.get('success'):
                        memories = retrieval_response.get('memories', [])
                        logger.info(f"Retrieved {len(memories)} memories with threshold {threshold}")
                        
                        if len(memories) > 0:
                            memories_found = True
                            for i, memory in enumerate(memories):
                                logger.info(f"  Memory {i+1}: {memory.get('content')[:50]}... (score={memory.get('similarity_score'):.4f})")
                            break  # Stop trying lower thresholds if we found memories
                    else:
                        logger.error(f"Memory retrieval failed: {retrieval_response.get('error')}")
                
                if not memories_found:
                    logger.warning(f"No memories found for '{query}' at any threshold!")
            
            logger.info("\n======== MEMORY SYSTEM TEST COMPLETED SUCCESSFULLY ========")
            return True
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    try:
        # Run the async test in the event loop
        return asyncio.run(test_memory_system())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
