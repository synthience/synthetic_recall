#!/usr/bin/env python3
# test_memory_params.py - Test memory retrieval parameters

import sys
import os
import logging
import asyncio
import torch
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory system components
from memory.lucidia_memory_system.core.long_term_memory import LongTermMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('test_memory_params')

async def test_quickrecal_parameters():
    """Test min_quickrecal and min_quickrecal_score parameters"""
    logger.info("Initializing memory system for parameter testing")
    
    # Initialize long-term memory
    ltm = LongTermMemory({
        'storage_path': os.path.join('memory', 'stored'),
        'embedding_dim': 768
    })
    
    # List of threshold values to test
    thresholds = [0.0, 0.1, 0.2, 0.5]
    
    query = "What is memory?"
    
    logger.info(f"Testing with query: '{query}'")
    
    # Test with min_quickrecal_score parameter
    logger.info("Testing min_quickrecal_score parameter:")
    for threshold in thresholds:
        try:
            memories = await ltm.search_memory(
                query=query,
                limit=3,
                min_quickrecal_score=threshold
            )
            logger.info(f"  min_quickrecal_score={threshold}: Found {len(memories)} memories")
            
            # Log the scores of the first few memories
            if memories:
                logger.info(f"  First memory scores:")
                for i, mem in enumerate(memories[:min(3, len(memories))]):
                    logger.info(f"    Memory {i+1}: QR={mem.get('quickrecal_score', 'N/A')}, Content: {mem.get('content', '')[:50]}...")
                    
        except Exception as e:
            logger.error(f"  min_quickrecal_score={threshold}: Error: {e}")

async def test_emotion_data_extraction():
    """Test extraction of emotion data from memories"""
    logger.info("Initializing memory system for emotion data testing")
    
    # Initialize long-term memory
    ltm = LongTermMemory({
        'storage_path': os.path.join('memory', 'stored'),
        'embedding_dim': 768
    })
    
    query = "emotion"
    
    logger.info(f"Retrieving memories with query: '{query}'")
    
    # Retrieve some memories
    memories = await ltm.search_memory(
        query=query,
        limit=5,
        min_quickrecal_score=0.0
    )
    
    logger.info(f"Retrieved {len(memories)} memories")
    
    # Extract and print emotion data
    for i, memory in enumerate(memories):
        logger.info(f"Memory {i+1}:")
        
        # Check for emotion data in different possible locations
        metadata = memory.get('metadata', {})
        
        # Try various possible locations
        emotion_data = None
        if 'emotions' in metadata:
            emotion_data = {'format': 'direct', 'data': metadata['emotions']}
        elif 'emotional_data' in metadata:
            emotion_data = {'format': 'emotional_data', 'data': metadata['emotional_data']}
        elif 'emotional_context' in metadata and isinstance(metadata['emotional_context'], dict):
            if 'emotions' in metadata['emotional_context']:
                emotion_data = {'format': 'emotional_context.emotions', 'data': metadata['emotional_context']['emotions']}
        
        # Check for dominant emotion
        dominant_emotion = 'unknown'
        if 'dominant_emotion' in metadata:
            dominant_emotion = metadata['dominant_emotion']
        elif 'emotional_context' in metadata and isinstance(metadata['emotional_context'], dict):
            if 'dominant_emotion' in metadata['emotional_context']:
                dominant_emotion = metadata['emotional_context']['dominant_emotion']
            elif 'emotional_state' in metadata['emotional_context']:
                dominant_emotion = metadata['emotional_context']['emotional_state']
        
        # Log the findings
        logger.info(f"  Content sample: {memory.get('content', '')[:50]}...")
        logger.info(f"  Dominant emotion: {dominant_emotion}")
        logger.info(f"  Emotion data: {emotion_data if emotion_data else 'None'}")
        logger.info(f"  Raw metadata: {metadata}")

async def main():
    """Run all tests"""
    logger.info("Starting memory parameter tests")
    
    # Test min_quickrecal parameters
    await test_quickrecal_parameters()
    
    logger.info("\n" + "=" * 80 + "\n")
    
    # Test emotion data extraction
    await test_emotion_data_extraction()
    
    logger.info("All tests completed")

if __name__ == '__main__':
    asyncio.run(main())
