# test_memory_system.py

import asyncio
import logging
import torch
from pathlib import Path
from server.memory_system import MemorySystem

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Create memory system with explicit storage path
    storage_path = Path.cwd() / "memory/stored"
    logger.info(f"Using storage path: {storage_path.absolute()}")
    
    memory_system = MemorySystem({
        'storage_path': storage_path
    })
    
    # Test memory storage
    memory_text = "This is a test memory for the server system."
    dummy_embedding = torch.zeros(384)  # Create a dummy embedding
    
    logger.info(f"Storing memory in system: {memory_text}")
    memory = await memory_system.add_memory(
        text=memory_text,
        embedding=dummy_embedding,
        significance=0.9
    )
    
    logger.info(f"Memory ID: {memory['id']}")
    logger.info(f"Memory system storage path: {memory_system.storage_path}")
    logger.info(f"Number of memories in system: {len(memory_system.memories)}")
    
    # Check if files were created
    memory_files = list(memory_system.storage_path.glob("*.json"))
    logger.info(f"Memory system files found: {len(memory_files)}")
    for file in memory_files:
        logger.info(f"  - {file.name}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
