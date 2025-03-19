#!/usr/bin/env python
"""
Knowledge Base Integration Utility

This command-line tool provides utilities for managing knowledge base files
and validating their integration with Lucidia's knowledge graph.

Usage:
    python knowledge_base_utility.py --validate
    python knowledge_base_utility.py --query "your query here"
    python knowledge_base_utility.py --add-file "path/to/file.md" --description "File description"
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import Dict, Any, List

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("knowledge_base_utility")

# Import necessary modules
from memory_core.lucidia_memory import LucidiaMemorySystemMixin


async def validate_knowledge_base():
    """Validate the knowledge base integration."""
    logger.info("Validating knowledge base integration...")
    
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
    
    # Load knowledge base files
    success = await memory_system.load_knowledge_base_files(directory_path)
    
    if success:
        logger.info("✅ Knowledge base files loaded successfully!")
        
        # Test concept extraction
        await test_concept_extraction(memory_system)
    else:
        logger.error("❌ Failed to load knowledge base files")


async def query_knowledge_graph(query: str):
    """Query the knowledge graph."""
    logger.info(f"Querying knowledge graph with: '{query}'")
    
    # Initialize memory system
    memory_system = LucidiaMemorySystemMixin()
    
    # Load knowledge base files first
    directory_path = os.path.join(
        "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
    )
    
    if not os.path.exists(directory_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directory_path = os.path.join(
            base_dir, 
            "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
        )
    
    await memory_system.load_knowledge_base_files(directory_path)
    
    # Query the knowledge graph
    results = await memory_system.query_knowledge_graph(query, max_tokens=1024)
    
    if results:
        print("\n=== Knowledge Graph Query Results ===")
        print(f"Query: '{query}'")
        print("---\n")
        print(results)
        print("\n---")
    else:
        logger.error(f"No results found for query: '{query}'")


async def add_knowledge_file(file_path: str, description: str = ""):
    """Add a file to the knowledge base."""
    logger.info(f"Adding file to knowledge base: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    # Initialize memory system
    memory_system = LucidiaMemorySystemMixin()
    
    # Get the file content and type
    file_type = os.path.splitext(file_path)[1].lstrip('.')
    file_name = os.path.basename(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add to knowledge graph
    if hasattr(memory_system, 'knowledge_graph') and memory_system.knowledge_graph:
        # Add to knowledge graph with metadata
        node_id = await memory_system.knowledge_graph.add_knowledge_node(
            content=content,
            metadata={
                'source': f"user_added/{file_name}",
                'description': description or f"User-added file: {file_name}",
                'type': file_type,
                'filename': file_name
            }
        )
        
        if node_id:
            logger.info(f"✅ File added to knowledge graph: {file_name} (node_id: {node_id})")
            
            # Extract concepts
            if memory_system.world_model and hasattr(memory_system.world_model, '_extract_concepts'):
                concepts = await memory_system.world_model._extract_concepts(content)
                
                logger.info(f"Extracted {len(concepts)} concepts:")
                for concept_name, concept_info in concepts.items():
                    logger.info(f"  - {concept_name}: {concept_info.get('definition', 'No definition')}")
        else:
            logger.error(f"❌ Failed to add file to knowledge graph: {file_name}")
    else:
        logger.error("❌ Knowledge graph not available in memory system")


async def test_concept_extraction(memory_system):
    """Test concept extraction from a text sample."""
    logger.info("Testing concept extraction...")
    
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
    if concepts:
        logger.info(f"✅ Extracted {len(concepts)} concepts:")
        for concept_name, concept_info in concepts.items():
            logger.info(f"  - {concept_name}: {concept_info.get('definition', 'No definition')}")
    else:
        logger.error("❌ Failed to extract concepts from test text")
    
    return len(concepts) > 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Base Integration Utility")
    
    # Add arguments
    parser.add_argument('--validate', action='store_true', help='Validate knowledge base integration')
    parser.add_argument('--query', type=str, help='Query the knowledge graph')
    parser.add_argument('--add-file', type=str, help='Add a file to the knowledge base')
    parser.add_argument('--description', type=str, default='', help='Description for the added file')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.validate:
        await validate_knowledge_base()
    elif args.query:
        await query_knowledge_graph(args.query)
    elif args.add_file:
        await add_knowledge_file(args.add_file, args.description)
    else:
        # No arguments provided, show help
        print("\nKnowledge Base Integration Utility\n")
        print("Available commands:")
        print("  --validate            Validate knowledge base integration")
        print("  --query "<query>"     Query the knowledge graph")
        print("  --add-file "<path>"   Add a file to the knowledge base")
        print("  --description "<text>" Description for the added file\n")


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
