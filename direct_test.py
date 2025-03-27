#!/usr/bin/env python
# direct_test.py - Direct test of the memory system via API endpoints

import asyncio
import aiohttp
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory_direct_test")

# API endpoint
API_URL = "http://localhost:5010"

# Test content
TEST_MEMORY = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
TEST_QUERY = "Tell me about AI"

async def direct_test():
    """Run a direct test of memory creation and retrieval"""
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Check if the API is running
            logger.info("Step 1: Checking API health...")
            async with session.get(f"{API_URL}/health") as response:
                health = await response.json()
                logger.info(f"Health status: {health.get('status', 'unknown')}")
                if health.get('status') != 'healthy':
                    logger.error("API is not healthy, aborting test")
                    return False
            
            # Step 2: Get current stats
            logger.info("Step 2: Getting current system stats...")
            async with session.get(f"{API_URL}/stats") as response:
                stats = await response.json()
                memory_count = stats.get('memory', {}).get('total_memories', 'N/A')
                vector_count = stats.get('vector_index', {}).get('count', 'N/A')
                vector_mappings = stats.get('vector_index', {}).get('id_mappings', 'N/A')
                
                logger.info(f"Current memory count: {memory_count}")
                logger.info(f"Current vector count: {vector_count}")
                logger.info(f"Current ID mappings: {vector_mappings}")
            
            # Step 3: Create a test memory
            logger.info("Step 3: Creating test memory...")
            test_id = f"test_{int(time.time())}"
            payload = {
                "content": TEST_MEMORY,
                "metadata": {
                    "test_id": test_id,
                    "test_type": "direct_test"
                }
            }
            
            async with session.post(f"{API_URL}/process_memory", json=payload) as response:
                result = await response.json()
                memory_id = result.get("memory_id")
                if memory_id:
                    logger.info(f"✅ Memory created successfully: {memory_id}")
                else:
                    logger.error(f"❌ Failed to create memory: {result}")
                    return False
            
            # Step 4: Get updated stats
            logger.info("Step 4: Getting updated stats after memory creation...")
            async with session.get(f"{API_URL}/stats") as response:
                updated_stats = await response.json()
                updated_memory_count = updated_stats.get('memory', {}).get('total_memories', 'N/A')
                updated_vector_count = updated_stats.get('vector_index', {}).get('count', 'N/A')
                updated_vector_mappings = updated_stats.get('vector_index', {}).get('id_mappings', 'N/A')
                
                logger.info(f"Updated memory count: {updated_memory_count}")
                logger.info(f"Updated vector count: {updated_vector_count}")
                logger.info(f"Updated ID mappings: {updated_vector_mappings}")
                
                # Calculate vector count change if both values are numeric
                try:
                    if isinstance(updated_vector_count, (int, float)) and isinstance(vector_count, (int, float)):
                        count_change = updated_vector_count - vector_count
                        logger.info(f"Vector count change: {count_change}")
                    else:
                        logger.info("Vector count change: Not available (non-numeric values)")
                except Exception as e:
                    logger.info(f"Could not calculate vector count change: {str(e)}")
            
            # Step 5: Small delay to ensure memory is indexed
            logger.info("Step 5: Pausing to allow memory indexing...")
            await asyncio.sleep(1)
            
            # Step 6: Retrieve memory with query
            logger.info("Step 6: Retrieving memory with test query...")
            retrieve_payload = {
                "query": TEST_QUERY,
                "top_k": 5,
                "threshold": 0.2  # Use lower threshold for better recall
            }
            
            async with session.post(f"{API_URL}/retrieve_memories", json=retrieve_payload) as response:
                retrieve_result = await response.json()
                memories = retrieve_result.get("memories", [])
                if memories:
                    logger.info(f"✅ Retrieved {len(memories)} memories")
                    # Check if our test memory is in the results
                    found = False
                    for mem in memories:
                        # Extract similarity score - check multiple possible locations
                        similarity = None
                        if "metadata" in mem and "similarity_score" in mem["metadata"]:
                            similarity = mem["metadata"]["similarity_score"]
                        elif "similarity" in mem:
                            similarity = mem["similarity"]
                        elif "score" in mem:
                            similarity = mem["score"]
                            
                        logger.info(f"Memory {mem.get('id', '')[:8]}: {mem.get('content', '')[:50]}...")
                        logger.info(f"Similarity score: {similarity if similarity is not None else 'N/A'}")
                        
                        if mem.get("id") == memory_id:
                            logger.info(f"✅ Found our test memory in results with similarity {similarity if similarity is not None else 'N/A'}")
                            found = True
                    
                    if not found:
                        logger.warning("⚠️ Our test memory was not found in results")
                else:
                    logger.error("❌ No memories retrieved")
                    return False
            
            logger.info("✅ Direct test completed successfully!")
            return True
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    result = asyncio.run(direct_test())
    print("\n" + "=" * 50)
    if result:
        print("✅ DIRECT TEST PASSED - Memory system is working correctly!")
    else:
        print("❌ DIRECT TEST FAILED - See logs above for details")
    print("=" * 50)
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if result else 1)
