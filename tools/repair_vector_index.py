#!/usr/bin/env python

"""
Utility script to repair vector index inconsistencies.

This script provides direct access to the vector index repair functions
without going through the API, allowing for maintenance operations
even when the server is not running.
"""

import sys
import argparse
import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VectorIndexRepair')

# Local imports - adjust these paths as needed based on your project structure
sys.path.append('./')
try:
    from synthians_memory_core.vector_index import MemoryVectorIndex
    from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure to run this script from the project root directory")
    sys.exit(1)

async def repair_index(storage_path, repair_type="auto", use_gpu=False):
    """
    Repair the vector index at the specified storage path.
    
    Args:
        storage_path: Path to the memory storage directory
        repair_type: Type of repair to perform (auto, recreate_mapping, rebuild)
        use_gpu: Whether to use GPU for the index
        
    Returns:
        dict: Repair results
    """
    # Initialize vector index
    vector_index = MemoryVectorIndex({
        'embedding_dim': 768,  # Standard dimension
        'storage_path': storage_path,
        'index_type': 'Cosine',  # Most common type
        'use_gpu': use_gpu
    })
    
    # Check integrity before repair
    logger.info("Checking index integrity before repair...")
    initial_integrity = await vector_index.check_integrity()
    logger.info(f"Initial integrity: FAISS count: {initial_integrity['faiss_count']}, "
               f"Mapping count: {initial_integrity['mapping_count']}")
    logger.info(f"Consistent: {initial_integrity['consistent']}")
    
    if initial_integrity['consistent']:
        logger.info("Vector index is already consistent. No repair needed.")
        return {'success': True, 'message': 'No repair needed'}
    
    # Perform repair
    logger.info(f"Performing {repair_type} repair...")
    
    if repair_type == "auto":
        if initial_integrity['faiss_count'] < initial_integrity['mapping_count']:
            # More IDs in mapping than vectors in FAISS - recreate mapping
            logger.info("Auto-selected 'recreate_mapping' strategy")
            repair_type = "recreate_mapping"
        else:
            # More vectors in FAISS than IDs in mapping - default to recreate_mapping too
            # as a safer option (rebuild would be more invasive)
            logger.info("Auto-selected 'recreate_mapping' strategy (default)")
            repair_type = "recreate_mapping"
    
    if repair_type == "recreate_mapping":
        success = await vector_index.rebuild_mapping()
    elif repair_type == "rebuild":
        logger.warning("Full rebuild requires a populated memory core. Not implemented in standalone tool.")
        logger.warning("Falling back to recreate_mapping...")
        success = await vector_index.rebuild_mapping()
    else:
        logger.error(f"Unknown repair type: {repair_type}")
        return {'success': False, 'message': f"Unknown repair type: {repair_type}"}
    
    # Check integrity after repair
    logger.info("Checking index integrity after repair...")
    final_integrity = await vector_index.check_integrity()
    logger.info(f"Final integrity: FAISS count: {final_integrity['faiss_count']}, "
              f"Mapping count: {final_integrity['mapping_count']}")
    logger.info(f"Consistent: {final_integrity['consistent']}")
    
    if final_integrity['consistent']:
        logger.info("Repair successful!")
        return {'success': True, 'message': 'Repair successful'}
    else:
        logger.warning("Repair did not fully resolve inconsistency.")
        return {
            'success': False, 
            'message': 'Repair did not fully resolve inconsistency',
            'before': initial_integrity,
            'after': final_integrity
        }

async def main():
    parser = argparse.ArgumentParser(description="Repair vector index inconsistencies.")
    parser.add_argument("--storage-path", type=str, default="/app/memory/stored/synthians",
                      help="Path to the memory storage directory")
    parser.add_argument("--repair-type", type=str, default="auto", 
                      choices=["auto", "recreate_mapping", "rebuild"],
                      help="Type of repair to perform")
    parser.add_argument("--use-gpu", action="store_true",
                      help="Use GPU for the index (if available)")
    args = parser.parse_args()
    
    # Check if storage path exists
    storage_path = Path(args.storage_path)
    if not storage_path.exists():
        logger.error(f"Storage path {storage_path} does not exist.")
        return 1
    
    # Perform repair
    result = await repair_index(str(storage_path), args.repair_type, args.use_gpu)
    
    if result['success']:
        logger.info(f"Repair completed: {result['message']}")
        return 0
    else:
        logger.error(f"Repair failed: {result['message']}")
        return 1

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    sys.exit(loop.run_until_complete(main()))
