#!/usr/bin/env python

import asyncio
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quick_test")

async def test_memory_system():
    try:
        # Import the client
        from synthians_memory_core.api.client.client import SynthiansClient
        
        # Determine correct API endpoint - use internal service name within Docker
        # When running in Docker, hostname should be the service name (or 0.0.0.0) and port 5010
        hostname = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5010))
        base_url = f"http://{hostname}:{port}"
        
        logger.info(f"Connecting to memory system at {base_url}...")
        client = SynthiansClient(base_url)
        
        # Using async context manager to handle session creation/cleanup
        async with client:
            logger.info("Successfully connected to memory system")
            
            # Create a unique test memory
            timestamp = datetime.now().isoformat()
            content = f"Test memory created at {timestamp}"
            
            logger.info(f"Creating test memory: {content}")
            response = await client.process_memory(
                content=content,
                metadata={"test_type": "quick_test", "timestamp": timestamp}
            )
            
            if not response.get("success"):
                logger.error(f"Failed to create memory: {response}")
                return False
            
            memory_id = response.get("memory_id")
            logger.info(f"Successfully created memory with ID: {memory_id}")
            
            # Wait a moment for indexing
            logger.info("Waiting for indexing...")
            await asyncio.sleep(1)
            
            # Retrieve the memory
            query = f"test {timestamp}"
            logger.info(f"Retrieving memory with query: '{query}'")
            
            retrieval_response = await client.retrieve_memories(
                query=query,
                top_k=5,
                threshold=0.3
            )
            
            if not retrieval_response.get("success"):
                logger.error(f"Failed to retrieve memory: {retrieval_response}")
                return False
            
            memories = retrieval_response.get("memories", [])
            logger.info(f"Retrieved {len(memories)} memories")
            
            # Check if our memory was found
            for memory in memories:
                logger.info(f"Memory ID: {memory.get('id')}, Score: {memory.get('similarity_score')}")
                if memory.get("id") == memory_id:
                    logger.info("✓ Test memory successfully retrieved!")
                    return True
            
            logger.error("✗ Test memory was not retrieved")
            return False
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False

async def main():
    result = await test_memory_system()
    if result:
        logger.info("✓ FAISS Memory System Test: PASSED")
        return 0
    else:
        logger.error("✗ FAISS Memory System Test: FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
