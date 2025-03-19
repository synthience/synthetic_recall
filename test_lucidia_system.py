#!/usr/bin/env python
# test_lucidia_system.py

import asyncio
import logging
import os

from voice_core.config.config import LucidiaConfig
from memory_core.enhanced_memory_client import EnhancedMemoryClient
from voice_core.llm.llm_pipeline import LocalLLMPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Test the enhanced Lucidia system with self-model and world-model integration."""
    # Initialize configuration
    config = LucidiaConfig()
    
    # Print the system prompt
    logger.info(f"Using system prompt: {config.llm.system_prompt}")
    
    # Initialize memory client
    memory_client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:5005",
        hpc_server_url="ws://localhost:5006",
        session_id="test_session",
        user_id="test_user"
    )
    
    # Initialize LLM pipeline
    llm_pipeline = LocalLLMPipeline(config.llm)
    await llm_pipeline.initialize()
    llm_pipeline.set_memory_client(memory_client)
    
    # Test queries
    test_queries = [
        "What are your capabilities?",
        "What do you know about me?",
        "Tell me about AI technology"
    ]
    
    for query in test_queries:
        logger.info(f"\n\n=== Testing query: {query} ===")
        
        # Get context
        context = await llm_pipeline._get_hierarchical_memory_context(query)
        logger.info(f"Context length: {len(context)}")
        logger.info(f"Context preview: {context[:200]}...")
        
        # Generate response
        response = await llm_pipeline.generate_response(query)
        logger.info(f"Response: {response}")
    
    # Close connections
    await llm_pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
