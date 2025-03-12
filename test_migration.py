#!/usr/bin/env python3
"""
Test script to verify memory migration from flat to hierarchical memory system
"""

import asyncio
import logging
from server.memory_bridge import MemoryBridge
from server.memory_system import MemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_migration')

async def test_migration():
    """Test memory migration between flat and hierarchical systems"""
    logger.info("Initializing memory systems...")
    memory_system = MemorySystem()
    memory_bridge = MemoryBridge(memory_system)
    
    # Log initial stats
    logger.info(f"Initial memory counts: {memory_bridge.stats}")
    
    # Perform migration
    logger.info("Starting memory migration...")
    await memory_bridge.migrate_memories()
    
    # Log final stats
    logger.info(f"Final memory counts: {memory_bridge.stats}")
    logger.info("Migration test complete!")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_migration())
