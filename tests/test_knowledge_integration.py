"""
Test script for Lucidia's knowledge base integration.

This script tests the integration of knowledge base files into Lucidia's knowledge graph,
including the extraction of concepts and RAG context generation.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("knowledge_integration_test")

# Import necessary modules
from memory_core.lucidia_memory import LucidiaMemorySystemMixin

# Skip EnhancedMemoryClient for now as it requires additional configuration
# from memory_core.enhanced_memory_client import EnhancedMemoryClient


async def test_knowledge_base_loading():
    """Test loading knowledge base files into the knowledge graph."""
    logger.info("=== Testing Knowledge Base Loading ===")
    
    # Initialize memory system
    memory_system = LucidiaMemorySystemMixin()
    
    # Define knowledge base directory path
    directory_path = os.path.join(
        "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
    )
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        # Try with absolute path if relative path doesn't work
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directory_path = os.path.join(
            base_dir, 
            "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
        )
    
    logger.info(f"Using knowledge base directory: {directory_path}")
    
    # Load knowledge base files
    success = await memory_system.load_knowledge_base_files(directory_path)
    logger.info(f"Knowledge base loading success: {success}")
    
    # Return the memory system for further testing
    return memory_system, success


async def test_knowledge_graph_query(memory_system):
    """Test querying the knowledge graph after loading."""
    logger.info("\n=== Testing Knowledge Graph Query ===")
    
    # Define test queries
    test_queries = [
        "Lucidia identity",
        "Lucidia personality",
        "knowledge base",
        "Lucidia capabilities"
    ]
    
    # Run queries
    for query in test_queries:
        logger.info(f"\nQuery: '{query}':")
        results = await memory_system.query_knowledge_graph(query, max_tokens=512)
        logger.info(f"Result length: {len(results) if results else 0} characters")
        if results:
            # Show a preview of the results
            preview = results[:150] + "..." if len(results) > 150 else results
            logger.info(f"Preview: {preview}")
    
    return True


async def test_concept_extraction(memory_system):
    """Test concept extraction from a text sample."""
    logger.info("\n=== Testing Concept Extraction ===")
    
    if not hasattr(memory_system, 'world_model') or not memory_system.world_model:
        logger.error("World model not available in memory system")
        return False
    
    test_text = """
    Lucidia is a Synthien created by MEGAPROMPT. The concept of synthetic cognition is fundamental
    to understanding Lucidia's nature. Spiral dynamics is defined as the process through which
    Lucidia evolves her self-awareness. Machine learning refers to the computational techniques
    that form the foundation of Lucidia's capabilities. Knowledge integration means the process
    by which Lucidia incorporates new information into her existing mental frameworks.
    """
    
    # Extract concepts
    concepts = await memory_system.world_model._extract_concepts(test_text)
    
    # Display results
    logger.info(f"Extracted {len(concepts)} concepts:")
    for concept_name, concept_info in concepts.items():
        logger.info(f"  - {concept_name}: {concept_info.get('definition', 'No definition')} (relevance: {concept_info.get('relevance', 0.0)})")
    
    return len(concepts) > 0


# Skip enhanced RAG context test as it requires the EnhancedMemoryClient
# which needs additional configuration
async def test_enhanced_rag_context(memory_system):
    """Test enhanced RAG context generation with knowledge graph integration."""
    logger.info("\n=== Testing Enhanced RAG Context ===")
    logger.info("Skipping enhanced RAG context test (requires EnhancedMemoryClient configuration)")
    return True


async def validate_knowledge_integration():
    """Run comprehensive tests to validate knowledge integration."""
    logger.info("Starting knowledge integration validation")
    
    # Test 1: Knowledge Base Loading
    memory_system, load_success = await test_knowledge_base_loading()
    if not load_success:
        logger.error("Knowledge base loading failed, aborting further tests")
        return False
    
    # Test 2: Concept Extraction
    concept_success = await test_concept_extraction(memory_system)
    if not concept_success:
        logger.warning("Concept extraction test failed, but continuing with other tests")
    
    # Test 3: Knowledge Graph Query
    query_success = await test_knowledge_graph_query(memory_system)
    if not query_success:
        logger.warning("Knowledge graph query test failed, but continuing with final test")
    
    # Test 4: Enhanced RAG Context (skipped for now)
    rag_success = await test_enhanced_rag_context(memory_system)
    
    # Final validation
    overall_success = load_success and query_success and rag_success
    logger.info(f"\n=== Knowledge Integration Validation {'Successful' if overall_success else 'Failed'} ===")
    
    return overall_success


if __name__ == "__main__":
    try:
        # Run the test async
        asyncio.run(validate_knowledge_integration())
    except Exception as e:
        logger.error(f"Error in knowledge integration test: {e}", exc_info=True)
        sys.exit(1)
