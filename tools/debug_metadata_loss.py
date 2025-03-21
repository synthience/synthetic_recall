#!/usr/bin/env python3
# debug_metadata_loss.py - Investigating why emotion metadata is being lost

import os
import sys
import json
import logging
from pathlib import Path
import asyncio
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import memory system
from server.memory_system import MemorySystem

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_metadata_loss.log')
    ]
)
logger = logging.getLogger('debug_metadata_loss')

async def debug_memory_system():
    """Debug the memory system's handling of metadata."""
    try:
        # Initialize memory system with debug path
        debug_path = Path('memory/debug_test')
        debug_path.mkdir(parents=True, exist_ok=True)
        
        memory_system = MemorySystem({
            'storage_path': debug_path,
            'embedding_dim': 768
        })
        
        # Create test metadata with emotion data
        test_metadata = {
            'emotions': {
                'joy': 0.8,
                'sadness': 0.1,
                'anger': 0.05,
                'fear': 0.05
            },
            'dominant_emotion': 'joy',
            'source': 'debug_test',
            'test_field': 'test_value'
        }
        
        logger.info(f"Original metadata: {test_metadata}")
        
        # Create a test embedding
        test_embedding = [0.1] * 768
        
        # Add a memory with this metadata
        logger.info("Adding memory with emotion metadata...")
        memory = await memory_system.add_memory(
            text="This is a test memory with emotion metadata.",
            embedding=test_embedding,
            quickrecal_score=0.75,
            metadata=test_metadata
        )
        
        logger.info(f"Memory added with ID: {memory['id']}")
        
        # Check if memory was saved correctly
        memory_file = debug_path / f"{memory['id']}.json"
        logger.info(f"Checking saved memory file: {memory_file}")
        
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                saved_memory = json.load(f)
                
            logger.info(f"Saved metadata: {saved_memory.get('metadata', {})}")
            
            # Compare original and saved metadata
            if test_metadata == saved_memory.get('metadata', {}):
                logger.info("SUCCESS: Metadata saved correctly")
            else:
                logger.error("FAILURE: Metadata was modified during saving")
                logger.error(f"Original: {test_metadata}")
                logger.error(f"Saved: {saved_memory.get('metadata', {})}")
                
                # Check which keys were lost
                orig_keys = set(test_metadata.keys())
                saved_keys = set(saved_memory.get('metadata', {}).keys())
                
                missing_keys = orig_keys - saved_keys
                if missing_keys:
                    logger.error(f"Missing keys: {missing_keys}")
                
                # Check specifically for emotion data
                if 'emotions' not in saved_memory.get('metadata', {}):
                    logger.error("The 'emotions' key is missing from the saved metadata")
                
                if 'dominant_emotion' not in saved_memory.get('metadata', {}):
                    logger.error("The 'dominant_emotion' key is missing from the saved metadata")
        else:
            logger.error(f"Memory file not found: {memory_file}")
            
        # Create another memory with metadata inside a deep copy
        # This is to test if any accidental mutation of the metadata is occurring
        import copy
        test_metadata_copy = copy.deepcopy(test_metadata)
        
        logger.info("Adding memory with deep copied emotion metadata...")
        memory2 = await memory_system.add_memory(
            text="This is a second test memory with copied emotion metadata.",
            embedding=test_embedding,
            quickrecal_score=0.75,
            metadata=test_metadata_copy
        )
        
        memory_file2 = debug_path / f"{memory2['id']}.json"
        if memory_file2.exists():
            with open(memory_file2, 'r') as f:
                saved_memory2 = json.load(f)
                
            logger.info(f"Saved metadata (copy): {saved_memory2.get('metadata', {})}")
            
            if test_metadata_copy == saved_memory2.get('metadata', {}):
                logger.info("SUCCESS: Deep copied metadata saved correctly")
            else:
                logger.error("FAILURE: Deep copied metadata was modified during saving")

    except Exception as e:
        logger.error(f"Error in debug function: {str(e)}")
        logger.error(traceback.format_exc())

async def main():
    logger.info("Starting metadata loss debug script")
    await debug_memory_system()
    logger.info("Debug complete")

if __name__ == '__main__':
    asyncio.run(main())
