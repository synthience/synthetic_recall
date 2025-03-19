# Knowledge Base Integration Guide

## Overview

This guide documents the implementation of knowledge base file integration into Lucidia's knowledge graph. The integration enables Lucidia to extract concepts and relationships from external knowledge files (JSON, Markdown) and incorporate them into her mental model.

## Implementation Details

### Core Components

1. **`_extract_concepts` Method in LucidiaWorldModel**
   - Added to extract concepts and relationships from text content using pattern matching and NLP techniques
   - Identifies potential concepts, definitions, and extracts keywords
   - Returns a dictionary of concepts with their definitions and relevance scores

2. **Knowledge Graph Integration**
   - The `add_knowledge_node` method in `LucidiaKnowledgeGraph` processes content and extracts concepts
   - Creates nodes for both the knowledge content and extracted concepts
   - Establishes relationships between knowledge nodes and concept nodes

3. **Knowledge Base File Loading**
   - The `load_knowledge_base_files` method in `LucidiaMemorySystemMixin` loads files from a specified directory
   - Verifies directory existence and manifest file structure
   - Processes each file based on its type (JSON, Markdown)
   - Adds content to the knowledge graph with appropriate metadata

### Workflow

1. Knowledge base files are stored in the `/memory/lucidia_memory_system/core/Self/Lucidia_Files_Archive` directory
2. The manifest.json file lists all files with their types and descriptions
3. When `load_knowledge_base_files` is called, it processes each file in the manifest
4. Files are read and stored in the knowledge graph as knowledge nodes
5. The `_extract_concepts` method is called to identify concepts in the content
6. Concept nodes are created and linked to knowledge nodes with "mentions" relationships
7. The knowledge graph can be queried to retrieve relevant information

## Test Results

We implemented and tested the knowledge base integration with the following outcomes:

### Test Scripts

1. **test_knowledge_integration.py**
   - Validates the core functionality of knowledge base loading
   - Tests concept extraction from sample text
   - Verifies knowledge graph queries

2. **knowledge_base_utility.py**
   - Command line tool for interacting with the knowledge base
   - Provides functions to validate, query, and add files to the knowledge base

3. **extract_knowledge_concepts.py**
   - Processes all files in the knowledge base directory
   - Extracts concepts and builds relationships in the knowledge graph
   - Tests queries against the knowledge graph

### Findings

- The concept extraction functionality works as expected, identifying relevant concepts from text
- The knowledge base files are successfully loaded into the knowledge graph
- Query functionality requires further refinement as some expected queries return no results
- The integration with the RAG context generation is in place but needs additional testing

### Next Steps

1. **Improve Query Capability**
   - Enhance the semantic search functionality to better match queries with knowledge content
   - Implement fuzzy matching for concept recognition

2. **Expand Concept Extraction**
   - Improve pattern matching to capture more complex relationships
   - Consider implementing additional NLP techniques for better concept identification

3. **Test RAG Integration**
   - Complete testing of the EnhancedMemoryClient with knowledge graph integration
   - Ensure RAG context includes relevant knowledge base content

4. **User Feedback**
   - Gather feedback on the quality and relevance of extracted concepts
   - Refine the extraction algorithm based on user needs

## Implementation Notes

### Feature: `_extract_concepts` Method

```python
async def _extract_concepts(self, text: str) -> Dict[str, Dict[str, Any]]:
    """Extract concepts and relationships from text content.
    
    Args:
        text: The text content to extract concepts from
        
    Returns:
        Dictionary of concept names with their attributes
    """
    concepts = {}
    
    # Skip empty text
    if not text or len(text.strip()) == 0:
        return concepts
    
    # Clean and prepare text
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Pattern 1: Extract explicit definitions (X is defined as Y)
    definition_patterns = [
        r'([\w\s]+) is defined as ([^.]+)',
        r'([\w\s]+) means ([^.]+)',
        r'([\w\s]+) refers to ([^.]+)',
        r'([\w\s]+) represents ([^.]+)'
    ]
    
    for pattern in definition_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            concept = match.group(1).strip()
            definition = match.group(2).strip()
            
            if len(concept) > 50 or len(concept.split()) > 5:  
                continue  # Skip overly long concepts
                
            # Clean up concept name and make it sentence case
            concept = concept[0].upper() + concept[1:].lower()
            
            # Store concept with its definition and relevance
            concepts[concept] = {
                'definition': definition,
                'relevance': 0.8,  # High relevance for explicit definitions
                'source': 'pattern_match',
                'pattern': 'definition'
            }
    
    # Pattern 2: Extract important concepts that lack explicit definitions
    # Look for capitalized phrases that likely represent key concepts
    capitalized_pattern = r'(?<![.\?!]\s)\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,2})\b'
    matches = re.finditer(capitalized_pattern, text)
    
    for match in matches:
        concept = match.group(1).strip()
        
        # Skip overly long concepts or already defined ones
        if concept in concepts or len(concept) > 50:
            continue
            
        # Determine if concept is mentioned multiple times (relevance)
        mentions = len(re.findall(r'\b' + re.escape(concept) + r'\b', text, re.IGNORECASE))
        relevance = min(0.5 + (mentions * 0.1), 0.9)  # Scale relevance with mentions
        
        concepts[concept] = {
            'definition': f"Concept mentioned in text: {concept.lower()}",
            'relevance': relevance,
            'source': 'capitalization',
            'pattern': 'capitalized'
        }
    
    return concepts
```

## Conclusion

The knowledge base integration implementation successfully enhances Lucidia's world model by incorporating external knowledge files. The concept extraction functionality enables Lucidia to identify and relate concepts from these files, building a more comprehensive knowledge graph. While the basic functionality is in place, further refinement is needed for query capabilities and RAG integration.

The current implementation provides a solid foundation for future enhancements and will enable Lucidia to maintain a more dynamic and comprehensive understanding of her knowledge domain.
