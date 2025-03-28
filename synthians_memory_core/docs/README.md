# Synthians Memory Core Documentation

## Bi-Hemispheric Cognitive Architecture

- [Bi-Hemispheric Architecture Overview](bihemispheric_architecture.md) - Complete design overview and neural pathway flow
- [API Reference](api_reference.md) - Detailed API references for all components
- [Implementation Guide](implementation_guide.md) - Technical implementation and integration guide

## Vector Index and FAISS Integration

- [Memory Vector Index with FAISS](vector_index.md) - Core implementation details and usage
- [FAISS GPU Integration Guide](faiss_gpu_integration.md) - How GPU acceleration is implemented
- [Embedding Handling](embedding_handling.md) - Robust embedding validation and dimension alignment

## Memory System

### Core Features

- Memory storage and retrieval
- Efficient vector similarity search via FAISS
- Automatic embedding validation and dimension alignment
- Metadata synthesis and enrichment
- Emotion analysis integration

### Implementation Details

#### Memory Retrieval

- Improved pre-filter threshold (reduced from 0.5 to 0.3)
- Added NaN/Inf validation for embedding vectors
- Enhanced similarity score logging
- Added explicit threshold parameter support

#### Metadata Enrichment

- MetadataSynthesizer integration in the memory processing workflow
- Automatic addition of UUID and content length to metadata
- Sophisticated metadata extraction and enrichment

#### Emotion Analysis

- Optimized emotion analysis to avoid redundant processing
- Respect for pre-computed emotion data from API
- Fallback mechanisms for handling unavailable services

## Architecture

### Components

1. **SynthiansMemoryCore** - The main memory management system
2. **MemoryVectorIndex** - FAISS-based vector indexing for efficient retrieval
3. **MetadataSynthesizer** - Enriches memory with metadata
4. **EmotionAnalyzer** - Analyzes emotional content of text

### Deployment

- Docker integration with GPU support
- Automatic dependency management
- Robust error handling and fallbacks

## Docker Integration

The system is designed to run in a Docker environment with optional GPU acceleration:

- Automatic detection and installation of appropriate FAISS version
- GPU acceleration when available
- Seamless fallback to CPU processing when necessary

## API

The system exposes a comprehensive API for memory operations:

- Memory processing and storage
- Similarity-based retrieval
- Embedding generation
- Emotion analysis
- Transcription processing

See the API server implementation for detailed endpoint specifications.

## Technologies

- **FAISS** - Facebook AI Similarity Search for efficient vector operations
- **Sentence Transformers** - For generating text embeddings
- **FastAPI** - For the REST API interface
- **Docker** - For containerized deployment
- **CUDA** - For GPU acceleration
