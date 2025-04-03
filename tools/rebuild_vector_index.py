#!/usr/bin/env python

"""
Direct utility script to rebuild the vector index without going through the API.
This is useful for fixing vector index inconsistencies directly.
"""

import os
import sys
import asyncio
import logging

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.custom_logger import setup_logger

logger = setup_logger()

async def rebuild_vector_index():
    # Create a temporary memory core instance
    memory_core = SynthiansMemoryCore()
    
    # Initialize it
    logger.info("Initializing memory core...")
    await memory_core.initialize()
    
    # Check index integrity
    logger.info("Checking vector index integrity...")
    is_consistent, diagnostics = await memory_core.check_index_integrity()
    
    logger.info(f"Current index integrity: {is_consistent}")
    logger.info(f"Diagnostics: {diagnostics}")
    
    # Force rebuild regardless of integrity check result
    logger.info("Rebuilding vector index from persistence...")
    success = await memory_core.repair_index(repair_type="rebuild_from_persistence")
    
    # Verify the index after rebuild
    if success:
        logger.info("Vector index rebuilt successfully!")
        is_consistent, diagnostics = await memory_core.check_index_integrity()
        logger.info(f"New index integrity: {is_consistent}")
        logger.info(f"New diagnostics: {diagnostics}")
    else:
        logger.error("Failed to rebuild vector index!")
    
    # Shutdown core
    await memory_core.shutdown()
    
    return success

def main():
    logger.info("Starting vector index rebuild utility")
    success = asyncio.run(rebuild_vector_index())
    
    if success:
        logger.info("Vector index rebuild completed successfully")
        return 0
    else:
        logger.error("Vector index rebuild failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
