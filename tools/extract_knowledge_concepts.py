#!/usr/bin/env python
"""
Concept Extraction and Knowledge Graph Builder

This script extracts concepts from the knowledge base files
and builds relationships between them in the knowledge graph.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("concept_extraction")

# Import necessary modules
from memory_core.lucidia_memory import LucidiaMemorySystemMixin


async def extract_and_link_file_concepts(memory_system, file_path: str, metadata: Optional[Dict[str, Any]] = None):
    """Extract concepts from a file and link them in the knowledge graph."""
    logger.info(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract file metadata if not provided
    if metadata is None:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lstrip('.')
        metadata = {
            'source': f"extract_script/{file_name}",
            'description': f"Auto-extracted from {file_name}",
            'type': file_ext,
            'filename': file_name
        }
    
    # Add to knowledge graph
    if hasattr(memory_system, 'knowledge_graph') and memory_system.knowledge_graph:
        # Add to knowledge graph with metadata
        node_id = await memory_system.knowledge_graph.add_knowledge_node(
            content=content,
            metadata=metadata
        )
        
        if node_id:
            logger.info(f"✅ File added to knowledge graph: {metadata['filename']} (node_id: {node_id})")
            return True
        else:
            logger.error(f"❌ Failed to add file to knowledge graph: {metadata['filename']}")
            return False
    else:
        logger.error("❌ Knowledge graph not available in memory system")
        return False


async def process_knowledge_base_directory():
    """Process all files in the knowledge base directory."""
    logger.info("Processing knowledge base directory...")
    
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
    
    # Check for manifest file
    manifest_path = os.path.join(directory_path, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return False
    
    # Process files according to manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    files_data = manifest.get("knowledge_base", {}).get("files", [])
    processed_count = 0
    
    # Process each file
    for file_data in files_data:
        filename = file_data.get("filename")
        if not filename:
            continue
        
        file_path = os.path.join(directory_path, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        # Extract metadata from manifest
        metadata = {
            'source': f"lucidia_files_archive/{filename}",
            'description': file_data.get("description", ""),
            'type': file_data.get("type", ""),
            'filename': filename
        }
        
        # Process file
        success = await extract_and_link_file_concepts(memory_system, file_path, metadata)
        if success:
            processed_count += 1
    
    logger.info(f"Processed {processed_count} out of {len(files_data)} files")
    
    # Test query functionality
    await test_knowledge_graph_queries(memory_system)
    
    return processed_count > 0


async def test_knowledge_graph_queries(memory_system):
    """Test various queries against the knowledge graph."""
    logger.info("\n=== Testing Knowledge Graph Queries ===")
    
    test_queries = [
        "Lucidia identity",
        "synthetic cognition",
        "emotional intelligence",
        "Lucidia capabilities"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = await memory_system.query_knowledge_graph(query, max_tokens=512)
        
        if results:
            preview = results[:150] + "..." if len(results) > 150 else results
            logger.info(f"✅ Got results ({len(results)} chars)")
            logger.info(f"Preview: {preview}")
        else:
            logger.warning(f"❌ No results for query: '{query}'")
    
    # Return True to indicate test completion
    return True


async def main():
    """Main entry point."""
    logger.info("Starting concept extraction and knowledge graph building...")
    
    # Process knowledge base directory
    success = await process_knowledge_base_directory()
    
    if success:
        logger.info("\n✅ Knowledge base processing completed successfully!")
    else:
        logger.error("\n❌ Failed to process knowledge base")


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
