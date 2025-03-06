# test_memory_persistence.py

import asyncio
import logging
import os
from pathlib import Path
from memory_core.enhanced_memory_client import EnhancedMemoryClient
from server.memory_system import MemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_client():
    # Create memory client
    client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:8765",
        hpc_server_url="ws://localhost:8766",
        session_id="test_session",
        user_id="test_user"
    )
    
    # Test memory storage
    memory_text = "This is a test memory to verify persistence."
    
    logger.info(f"Storing memory: {memory_text}")
    success = await client.store_memory(
        content=memory_text,
        metadata={"type": "test"},
        significance=0.8
    )
    
    logger.info(f"Memory storage result: {success}")
    logger.info(f"Memory storage path: {client.storage_path}")
    logger.info(f"Number of memories: {len(client.memories)}")
    
    # Force persist memories
    logger.info("Forcing memory persistence...")
    await client._persist_memories()
    
    # Check if files were created
    memory_files = list(client.storage_path.glob("*.json"))
    logger.info(f"Memory files found: {len(memory_files)}")
    for file in memory_files:
        logger.info(f"  - {file}")
    
    # Clean up
    await client.cleanup()

async def test_memory_system():
    # Create memory system
    memory_system = MemorySystem()
    
    # Test memory storage
    memory_text = "This is a test memory for the server system."
    import torch
    dummy_embedding = torch.zeros(384)  # Create a dummy embedding
    
    logger.info(f"Storing memory in system: {memory_text}")
    memory = await memory_system.add_memory(
        text=memory_text,
        embedding=dummy_embedding,
        significance=0.9
    )
    
    logger.info(f"Memory system storage path: {memory_system.storage_path}")
    logger.info(f"Number of memories in system: {len(memory_system.memories)}")
    
    # Check if files were created
    memory_files = list(memory_system.storage_path.glob("*.json"))
    logger.info(f"Memory system files found: {len(memory_files)}")
    for file in memory_files:
        logger.info(f"  - {file}")

async def main():
    logger.info("=== Testing Memory Client Persistence ===")
    await test_memory_client()
    
    logger.info("\n=== Testing Memory System Persistence ===")
    await test_memory_system()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())
