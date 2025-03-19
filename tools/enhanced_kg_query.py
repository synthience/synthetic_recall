#!/usr/bin/env python
"""
Enhanced Knowledge Graph Query Tool

This script provides improved query capabilities for the knowledge graph,
implementing fuzzy matching and semantic search to find relevant information.
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import re
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_kg_query")

# Import necessary modules
from memory_core.lucidia_memory import LucidiaMemorySystemMixin


def preprocess_query(query: str) -> List[str]:
    """Preprocess the query into searchable terms."""
    # Lowercase and remove punctuation
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    
    # Split into words and remove stopwords
    stopwords = {'the', 'and', 'is', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'at', 'by', 'an', 'as'}
    terms = [word for word in query.split() if word not in stopwords and len(word) > 2]
    
    # Add original query and combinations
    result = [query]  # Original full query
    result.extend(terms)  # Individual significant terms
    
    # Add bi-grams if available
    bigrams = []
    for i in range(len(terms) - 1):
        bigrams.append(f"{terms[i]} {terms[i+1]}")
    result.extend(bigrams)
    
    return result


def get_file_descriptors(directory_path: str) -> List[Dict[str, Any]]:
    """Get descriptors for all files in the knowledge base."""
    descriptors = []
    
    # Check for manifest file
    manifest_path = os.path.join(directory_path, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file not found: {manifest_path}")
        return descriptors
    
    # Load manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    files_data = manifest.get("knowledge_base", {}).get("files", [])
    
    for file_data in files_data:
        filename = file_data.get("filename")
        if not filename:
            continue
        
        file_path = os.path.join(directory_path, filename)
        if not os.path.exists(file_path):
            continue
        
        descriptors.append({
            'path': file_path,
            'filename': filename,
            'type': file_data.get("type", ""),
            'description': file_data.get("description", "")
        })
    
    return descriptors


async def search_file_content(file_descriptor: Dict[str, Any], search_terms: List[str]) -> Tuple[str, float]:
    """Search a file for the query terms and return content with relevance score."""
    file_path = file_descriptor['path']
    file_type = file_descriptor['type']
    
    # Read file content
    content = ""
    try:
        if file_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to string for searching
                content = json.dumps(data, indent=2)
        else:  # Text, markdown, etc.
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return "", 0.0
    
    # Calculate relevance score based on term frequency
    score = 0.0
    term_scores = []
    
    # Preprocess content
    content_lower = content.lower()
    
    # Search for each term
    for i, term in enumerate(search_terms):
        term_lower = term.lower()
        count = content_lower.count(term_lower)
        
        # Weight based on term position (original query has higher weight)
        weight = 1.0 if i == 0 else 0.7 if i < len(search_terms) // 2 else 0.4
        
        # Calculate term score
        term_score = min(count * weight * 0.2, 1.0)
        term_scores.append(term_score)
    
    # Overall score is weighted average of term scores
    if term_scores:
        score = sum(term_scores) / len(term_scores)
    
    # Boost score based on metadata matching
    meta_score = 0.0
    for term in search_terms:
        if term.lower() in file_descriptor['description'].lower():
            meta_score += 0.3
        if term.lower() in file_descriptor['filename'].lower():
            meta_score += 0.2
    
    # Cap meta_score at 1.0
    meta_score = min(meta_score, 1.0)
    
    # Combine content and metadata scores
    final_score = (score * 0.7) + (meta_score * 0.3)
    
    # Return empty string if score is too low
    if final_score < 0.2:
        return "", final_score
    
    # Add metadata header to content
    header = f"Source: {file_descriptor['filename']}\nType: {file_descriptor['type']}\nDescription: {file_descriptor['description']}\n\n"
    
    return header + content, final_score


async def enhanced_knowledge_query(query: str):
    """Perform an enhanced search of the knowledge graph."""
    logger.info(f"Performing enhanced query: '{query}'")
    
    # Initialize memory system
    memory_system = LucidiaMemorySystemMixin()
    
    # Define knowledge base directory path
    directory_path = os.path.join(
        "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
    )
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        directory_path = os.path.join(
            base_dir, 
            "memory", "lucidia_memory_system", "core", "Self", "Lucidia_Files_Archive"
        )
    
    # Preprocess query
    search_terms = preprocess_query(query)
    logger.info(f"Search terms: {search_terms}")
    
    # Get file descriptors
    file_descriptors = get_file_descriptors(directory_path)
    logger.info(f"Found {len(file_descriptors)} files in knowledge base")
    
    # Search files for content
    search_tasks = []
    for file_descriptor in file_descriptors:
        task = search_file_content(file_descriptor, search_terms)
        search_tasks.append(task)
    
    search_results = await asyncio.gather(*search_tasks)
    
    # Filter and sort results
    valid_results = []
    for content, score in search_results:
        if content and score > 0.2:
            valid_results.append((content, score))
    
    # Sort by relevance score (descending)
    valid_results.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    if valid_results:
        print(f"\n=== Found {len(valid_results)} relevant results ===\n")
        
        # Combine top results
        combined_content = ""
        for i, (content, score) in enumerate(valid_results[:3]):
            result_header = f"\n--- Result {i+1} (Relevance: {score:.2f}) ---\n"
            combined_content += result_header + content + "\n\n"
        
        print(combined_content)
    else:
        logger.warning(f"No results found for query: '{query}'")
        print(f"\nNo relevant information found for: '{query}'")
    
    # Also try the standard query
    try:
        standard_result = await memory_system.query_knowledge_graph(query, max_tokens=512)
        if standard_result:
            print(f"\n=== Knowledge Graph API Results ===\n")
            print(standard_result)
    except Exception as e:
        logger.error(f"Error in standard query: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Knowledge Graph Query Tool")
    parser.add_argument('query', nargs='*', help='Query to search for in the knowledge graph')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.query:
        query = ' '.join(args.query)
        await enhanced_knowledge_query(query)
    else:
        print("\nEnhanced Knowledge Graph Query Tool\n")
        print("Usage: python enhanced_kg_query.py <your query here>\n")
        print("Example: python enhanced_kg_query.py Lucidia identity\n")


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
