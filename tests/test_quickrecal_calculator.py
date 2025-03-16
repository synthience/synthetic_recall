#!/usr/bin/env python3
"""
Test script for the unified QuickRecal calculator integration.
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.memory_adapter import MemoryAdapter

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_quickrecal_calculator():
    """Test the unified QuickRecal calculator."""
    # Initialize memory adapter
    adapter = MemoryAdapter({
        'prefer_unified': True,
        'adapter_mode': 'redirect',  # Use only the unified implementation
        'log_performance': True
    })
    
    # Get QuickRecal calculator
    calculator = await adapter.get_quickrecal_calculator()
    
    # Test with text
    text_samples = [
        "This is a normal message with no special significance.",
        "IMPORTANT: Remember to backup your data before upgrading!",
        "My name is John and I live in New York.",
        "I'm feeling really happy today because I got a promotion!",
        "The capital of France is Paris and it has a population of about 2.2 million."
    ]
    
    for text in text_samples:
        quickrecal_score = await calculator.calculate(text=text)
        logger.info(f"Text: '{text}'\nQuickRecal Score: {quickrecal_score:.4f}\n")
    
    # Test with embedding
    import numpy as np
    
    # Create some random embeddings
    embedding1 = np.random.rand(768).astype(np.float32)  # Common embedding size
    embedding2 = np.random.rand(768).astype(np.float32)
    
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    
    # Create context with embedding history
    context = {
        "embedding_history": [embedding1],
        "timestamp": 1646006400  # March 1, 2022
    }
    
    # Calculate QuickRecal score for embedding2 with context
    quickrecal_score = await calculator.calculate(embedding=embedding2, context=context)
    logger.info(f"Embedding QuickRecal score: {quickrecal_score:.4f}")
    
    # Get calculator stats
    stats = calculator.get_stats()
    logger.info(f"QuickRecal calculator stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_quickrecal_calculator())
