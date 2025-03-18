#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the emotion-integrated memory system.
This script tests the enhanced memory system with emotion integration.
"""

import asyncio
import logging
import json
import argparse
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.enhanced_memory_client import EnhancedMemoryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_emotion_memory_integration():
    """Test the integration of emotion detection with memory storage and retrieval."""
    logger.info("Testing emotion memory integration...")
    
    # Initialize a testing EnhancedMemoryClient
    client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:5001/ws",
        hpc_server_url="ws://localhost:5002/ws",
        session_id="test_emotion_integration",
        user_id="test_user"
    )
    
    # Connect to the servers
    await client.connect()
    
    logger.info("Connected to memory servers. Testing emotional memory storage...")
    
    # Test sentences with different emotions
    test_sentences = [
        "I'm so excited about this new feature! It's going to be amazing!",  # joy/excitement
        "I can't believe this keeps failing. It's so frustrating!",  # anger/frustration
        "I'm worried that we won't finish this project on time.",  # fear/worry
        "I'm really sad about the news I received today.",  # sadness
        "This is just a neutral statement about the weather.",  # neutral
    ]
    
    # Store each sentence as a memory
    memory_ids = []
    for i, sentence in enumerate(test_sentences):
        logger.info(f"\n[Test {i+1}] Storing memory with text: {sentence}")
        
        # Store the memory
        success = await client.store_memory(
            content=sentence,
            metadata={"test_id": i}
        )
        
        if success:
            logger.info(f"Successfully stored memory for test {i+1}")
            memory_ids.append(i)
        else:
            logger.error(f"Failed to store memory for test {i+1}")
    
    # Give some time for processing
    await asyncio.sleep(2)
    
    # Test retrieving memories by emotion
    logger.info("\nTesting memory retrieval by emotion...")
    
    # Test retrieving by positive emotions
    logger.info("\n[Retrieval Test 1] Positive emotions (sentiment > 0.3)")
    positive_memories = await client.retrieve_memories_by_emotion(
        sentiment_threshold=0.3,
        sentiment_direction="positive"
    )
    
    logger.info(f"Retrieved {len(positive_memories)} positive memories:")
    for mem in positive_memories:
        emotion = mem.get('metadata', {}).get('emotional_context', {}).get('emotional_state', 'unknown')
        sentiment = mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0)
        logger.info(f"Content: {mem.get('content')}")
        logger.info(f"Emotion: {emotion}, Sentiment: {sentiment:.2f}")
        logger.info(f"QuickRecal Score: {mem.get('quickrecal_score', 0):.2f}\n")
    
    # Test retrieving by negative emotions
    logger.info("\n[Retrieval Test 2] Negative emotions (sentiment < -0.2)")
    negative_memories = await client.retrieve_memories_by_emotion(
        sentiment_threshold=0.2,
        sentiment_direction="negative"
    )
    
    logger.info(f"Retrieved {len(negative_memories)} negative memories:")
    for mem in negative_memories:
        emotion = mem.get('metadata', {}).get('emotional_context', {}).get('emotional_state', 'unknown')
        sentiment = mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0)
        logger.info(f"Content: {mem.get('content')}")
        logger.info(f"Emotion: {emotion}, Sentiment: {sentiment:.2f}")
        logger.info(f"QuickRecal Score: {mem.get('quickrecal_score', 0):.2f}\n")
    
    # Test retrieving by specific emotion
    logger.info("\n[Retrieval Test 3] Specific emotion (joy)")
    joy_memories = await client.retrieve_memories_by_emotion(
        emotion="joy"
    )
    
    logger.info(f"Retrieved {len(joy_memories)} joy memories:")
    for mem in joy_memories:
        emotion = mem.get('metadata', {}).get('emotional_context', {}).get('emotional_state', 'unknown')
        sentiment = mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0)
        logger.info(f"Content: {mem.get('content')}")
        logger.info(f"Emotion: {emotion}, Sentiment: {sentiment:.2f}")
        logger.info(f"QuickRecal Score: {mem.get('quickrecal_score', 0):.2f}\n")
    
    # Test enhanced quickrecal scores based on emotion intensity
    logger.info("\n[Retrieval Test 4] Memory boost from emotional intensity")
    all_memories = await client.retrieve_memories_by_emotion()
    
    logger.info("All memories sorted by quickrecal_score:")
    # Sort by quickrecal_score for display
    all_memories.sort(key=lambda x: x.get('quickrecal_score', 0), reverse=True)
    
    for mem in all_memories:
        emotion = mem.get('metadata', {}).get('emotional_context', {}).get('emotional_state', 'unknown')
        sentiment = mem.get('metadata', {}).get('emotional_context', {}).get('sentiment', 0)
        logger.info(f"Content: {mem.get('content')}")
        logger.info(f"Emotion: {emotion}, Sentiment: {abs(sentiment):.2f} (intensity)")
        logger.info(f"QuickRecal Score: {mem.get('quickrecal_score', 0):.2f}\n")
    
    # Disconnect from servers
    await client.disconnect()
    logger.info("Test completed and disconnected from servers.")


async def test_emotional_memories_tool():
    """Test the LLM tool implementation for retrieving emotional memories."""
    logger.info("Testing emotional memories tool...")
    
    # Initialize a testing EnhancedMemoryClient
    client = EnhancedMemoryClient(
        tensor_server_url="ws://localhost:5001/ws",
        hpc_server_url="ws://localhost:5002/ws",
        session_id="test_emotion_tool",
        user_id="test_user"
    )
    
    # Connect to the servers
    await client.connect()
    
    # Store some test memories
    test_sentences = [
        "I'm so excited about this new feature! It's going to be amazing!",  # joy/excitement
        "I can't believe this keeps failing. It's so frustrating!",  # anger/frustration
        "I'm worried that we won't finish this project on time.",  # fear/worry
        "I'm really sad about the news I received today.",  # sadness
        "This is just a neutral statement about the weather.",  # neutral
    ]
    
    # Store each sentence as a memory
    for i, sentence in enumerate(test_sentences):
        await client.store_memory(
            content=sentence,
            metadata={"test_id": i}
        )
    
    # Give some time for processing
    await asyncio.sleep(2)
    
    # Test the tool
    logger.info("\nTesting retrieve_emotional_memories tool with positive sentiment")
    result = await client.retrieve_emotional_memories_tool({
        "sentiment_direction": "positive",
        "sentiment_threshold": 0.3
    })
    
    logger.info(f"Tool results: {json.dumps(result, indent=2)}")
    
    # Test with specific emotion
    logger.info("\nTesting retrieve_emotional_memories tool with specific emotion")
    result = await client.retrieve_emotional_memories_tool({
        "emotion": "anger"
    })
    
    logger.info(f"Tool results: {json.dumps(result, indent=2)}")
    
    # Disconnect from servers
    await client.disconnect()
    logger.info("Tool test completed and disconnected from servers.")


def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test emotional memory integration")
    parser.add_argument("--tool-test", action="store_true", help="Test the LLM tool for emotional memories")
    
    args = parser.parse_args()
    
    if args.tool_test:
        asyncio.run(test_emotional_memories_tool())
    else:
        asyncio.run(test_emotion_memory_integration())


if __name__ == "__main__":
    main()
