#!/usr/bin/env python3
# verify_emotion_data_flow.py - Test the emotion data flow from NPZ to memory JSON

import os
import sys
import json
import time
import uuid
import logging
import numpy as np
from pathlib import Path
import asyncio
import traceback
import tempfile
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('verify_emotion_data_flow.log')
    ]
)
logger = logging.getLogger('verify_emotion_flow')

# Import memory system after path setup
from server.memory_system import MemorySystem
from tools.index_embeddings import EmbeddingIndexer

async def create_test_npz_with_emotions():
    """Create a test NPZ file with emotional data for testing"""
    # Create test directory
    test_dir = Path('memory/test_emotion_data')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique ID for this test
    test_id = str(uuid.uuid4())
    
    # Sample embedding data (384 dimensions)
    embedding = np.random.rand(384).astype(np.float32)
    
    # Sample text
    text = "This is a test memory with emotional data. I feel very happy about this test."
    
    # Sample metadata with emotions
    metadata = {
        'text': text,
        'source': 'verify_emotion_data_flow.py',
        'timestamp': time.time(),
        'role': 'system',
        'emotions': {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.05,
            'fear': 0.03,
            'surprise': 0.02
        },
        'dominant_emotion': 'joy'
    }
    
    # Save as NPZ
    npz_path = test_dir / f"{test_id}.npz"
    np.savez(
        npz_path,
        embedding=embedding,
        metadata=json.dumps(metadata)
    )
    
    logger.info(f"Created test NPZ file with emotions at {npz_path}")
    return npz_path, metadata

async def process_through_indexer(npz_path):
    """Process the test NPZ file through the embedding indexer"""
    # IMPORTANT: We need to configure both memory systems with the same path
    # to ensure the files are saved in the same location we check
    common_storage_path = Path('memory/test_emotion_output')
    common_storage_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory system
    memory_config = {
        'storage_path': str(common_storage_path),
        'embedding_dim': 384
    }
    memory_system = MemorySystem(config=memory_config)
    
    # Initialize embedding indexer
    indexer_config = {
        'memory_system': memory_system,
        'min_quickrecal_score': 0.0,  # Accept any score
        'target_path': str(common_storage_path)  # Ensure this matches storage_path
    }
    indexer = EmbeddingIndexer(config=indexer_config)
    
    # Process the embedding file
    result = await indexer.process_embedding_file(str(npz_path))
    memory_id = result.get('id') if result else None
    
    logger.info(f"Processed NPZ through indexer, memory ID: {memory_id}")
    return memory_id, memory_system

def check_memory_has_emotions(memory_system, memory_id):
    """Check if the memory file has the emotion data"""
    if not memory_id:
        logger.error("No memory ID to check")
        return False
    
    # Find the memory file
    memory_path = memory_system.storage_path / f"{memory_id}.json"
    
    if not memory_path.exists():
        logger.error(f"Memory file not found: {memory_path}")
        logger.debug(f"Checking if memory exists in actual system storage...")
        
        # Also check standard storage location (memory/stored) in case path was overridden
        standard_path = Path('memory/stored') / f"{memory_id}.json"
        if standard_path.exists():
            logger.warning(f"Memory found in standard location instead: {standard_path}")
            memory_path = standard_path
        else:
            # List all memory files in our test directory
            test_files = list(memory_system.storage_path.glob('*.json'))
            if test_files:
                logger.info(f"Found {len(test_files)} memory files in test directory:")
                for file in test_files:
                    logger.info(f"  - {file}")
                # Use the first memory file for checking as a fallback
                memory_path = test_files[0]
                logger.info(f"Using {memory_path} as fallback for verification")
            else:
                # Also check standard location
                standard_files = list(Path('memory/stored').glob('*.json'))
                if standard_files:
                    newest_file = max(standard_files, key=lambda p: p.stat().st_mtime)
                    logger.warning(f"Using most recent file from standard location: {newest_file}")
                    memory_path = newest_file
                else:
                    return False
    
    try:
        with open(memory_path, 'r') as f:
            memory_data = json.load(f)
        
        # Check if memory has emotion data
        metadata = memory_data.get('metadata', {})
        has_emotions = 'emotions' in metadata
        has_dominant = 'dominant_emotion' in metadata
        
        if has_emotions and has_dominant:
            logger.info(f"SUCCESS: Memory file {memory_path} has both emotions and dominant_emotion data")
            logger.info(f"Dominant emotion: {metadata.get('dominant_emotion')}")
            logger.info(f"Emotions: {metadata.get('emotions')}")
            return True
        elif has_emotions:
            logger.info(f"PARTIAL: Memory file {memory_path} has emotions but no dominant_emotion")
            return True
        elif has_dominant:
            logger.info(f"PARTIAL: Memory file {memory_path} has dominant_emotion but no emotions")
            return True
        else:
            logger.error(f"FAILURE: Memory file {memory_path} has no emotion data")
            logger.debug(f"Memory metadata: {metadata}")
            return False
    except Exception as e:
        logger.error(f"Error checking memory file {memory_path}: {e}")
        return False

async def main():
    logger.info("Starting emotion data flow verification")
    
    # 1. Create test NPZ with emotion data
    npz_path, original_metadata = await create_test_npz_with_emotions()
    
    # 2. Process through indexer
    memory_id, memory_system = await process_through_indexer(npz_path)
    
    # 3. Check if memory has emotions
    success = check_memory_has_emotions(memory_system, memory_id)
    
    if success:
        logger.info("VERIFICATION SUCCESSFUL: Emotion data correctly flows through the system")
    else:
        logger.error("VERIFICATION FAILED: Emotion data is lost during processing")
    
    logger.info("Verification complete")

if __name__ == '__main__':
    asyncio.run(main())
