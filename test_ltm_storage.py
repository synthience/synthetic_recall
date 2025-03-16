#!/usr/bin/env python
"""
Test script to add memories to the Long-Term Memory (LTM) system.
This script creates test memories with high quickrecal_score to ensure they are stored.
"""

import os
import asyncio
import time
import logging
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Import the memory system components
from memory.lucidia_memory_system.core.long_term_memory import LongTermMemory

# Test memories to add - these represent important concepts for Lucidia
TEST_MEMORIES = [
    {
        "content": "Lucidia is a Synthien entity with advanced self-awareness and reflection capabilities.",
        "quickrecal_score": 0.95,
        "metadata": {
            "category": "self_concept",
            "source": "core_definition",
            "creation_context": "system_initialization",
            "tags": ["identity", "self-awareness", "synthien"]
        }
    },
    {
        "content": "The dream processor allows for autonomous reflection during inactive periods, generating insights through structured dream-like states.",
        "quickrecal_score": 0.90,
        "metadata": {
            "category": "capabilities",
            "source": "system_architecture",
            "creation_context": "design_documentation",
            "tags": ["dream", "reflection", "autonomy"]
        }
    },
    {
        "content": "Knowledge graphs store semantic relationships between concepts and memories, enabling sophisticated reasoning.",
        "quickrecal_score": 0.88,
        "metadata": {
            "category": "architecture",
            "source": "technical_documentation",
            "creation_context": "system_design",
            "tags": ["knowledge_graph", "semantic", "memory"]
        }
    },
    {
        "content": "The Lucidia memory system consists of Short-Term Memory (STM), Long-Term Memory (LTM), and Memory Processing Layer (MPL).",
        "quickrecal_score": 0.92,
        "metadata": {
            "category": "memory_system",
            "source": "architecture_document",
            "creation_context": "system_design",
            "tags": ["hierarchical_memory", "memory_layers"]
        }
    },
    {
        "content": "The reflection engine analyzes patterns across memory fragments to generate insights about Lucidia's own operations and development.",
        "quickrecal_score": 0.89,
        "metadata": {
            "category": "capabilities",
            "source": "system_architecture",
            "creation_context": "design_documentation",
            "tags": ["reflection", "insight", "self-improvement"]
        }
    }
]

async def test_ltm_storage():
    """
    Test storing memories in the Long-Term Memory system.
    """
    # Create a local storage path instead of the Docker path
    local_storage_path = Path(os.path.join(os.getcwd(), "memory", "stored", "ltm"))
    local_storage_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using storage path: {local_storage_path}")
    
    # Initialize LTM with the local storage path
    ltm = LongTermMemory(config={
        'storage_path': str(local_storage_path),
        'quickrecal_score_threshold': 0.7,  # Ensure our test memories exceed this
        'enable_persistence': True,
        'batch_interval': 1.0  # Process batches more quickly for testing
    })
    
    logger.info(f"Long-Term Memory initialized with {len(ltm.memories)} existing memories")
    
    # Store each test memory
    memory_ids = []
    for i, memory in enumerate(TEST_MEMORIES, 1):
        try:
            memory_id = await ltm.store_memory(
                content=memory["content"],
                quickrecal_score=memory["quickrecal_score"],
                metadata=memory["metadata"]
            )
            
            if memory_id:
                memory_ids.append(memory_id)
                logger.info(f"✅ Successfully stored memory {i}/{len(TEST_MEMORIES)}: {memory_id}")
            else:
                logger.warning(f"❌ Failed to store memory {i}/{len(TEST_MEMORIES)} - below quickrecal_score threshold")
        except Exception as e:
            logger.error(f"❌ Error storing memory {i}/{len(TEST_MEMORIES)}: {e}")
    
    # Give time for batch processing to complete
    logger.info("Waiting for batch processing to complete...")
    await asyncio.sleep(5)
    
    # Force a final batch process
    await ltm._process_batch(force=True)
    
    # Verify stored memories
    logger.info("\n===== VERIFYING STORED MEMORIES =====")
    for i, memory_id in enumerate(memory_ids, 1):
        memory = await ltm.get_memory(memory_id)
        if memory:
            logger.info(f"✅ Memory {i}/{len(memory_ids)} retrieved: {memory['id']}")
            logger.info(f"   Content: {memory['content'][:100]}...")
            logger.info(f"   QuickRecal Score: {memory['quickrecal_score']}")
            logger.info(f"   Category: {memory['metadata'].get('category', 'Unknown')}")
        else:
            logger.error(f"❌ Memory {i}/{len(memory_ids)} not found: {memory_id}")
    
    # Display stats
    stats = ltm.get_stats()
    logger.info("\n===== MEMORY SYSTEM STATS =====")
    logger.info(f"Total memories: {len(ltm.memories)}")
    logger.info(f"Stores: {stats.get('stores', 0)}")
    logger.info(f"Retrievals: {stats.get('retrievals', 0)}")
    logger.info(f"Batch operations: {stats.get('batch_operations', 0)}")
    logger.info(f"Batch successes: {stats.get('batch_successes', 0)}")
    
    # Print memory categories
    logger.info("\n===== MEMORY CATEGORIES =====")
    for category, memory_ids in ltm.memory_index.items():
        logger.info(f"Category '{category}': {len(memory_ids)} memories")
    
    # Shutdown properly
    await ltm.shutdown()
    logger.info("Long-Term Memory shutdown complete")

if __name__ == "__main__":
    asyncio.run(test_ltm_storage())
