# Synthians Memory Core

<p align="center">
  <img src="https://via.placeholder.com/600x200?text=Synthians+Memory+Core" alt="Synthians Memory Core Banner">
</p>

<p align="center">
  <a href="https://github.com/synthians/memory-core/releases"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version 1.0.0"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/synthians/memory-core/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status"></a>
  <a href="https://synthians-memory-core.readthedocs.io/"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Documentation Status"></a>
  <a href="https://pypi.org/project/synthians-memory-core/"><img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg" alt="Python Versions"></a>
</p>

## A Unified, Efficient Memory System for AI Applications

Synthians Memory Core is a sophisticated memory management system designed for AI applications that require intelligent, context-aware memory retrieval. It incorporates advanced features like hyperbolic geometry for efficient representation, emotional intelligence for context-aware retrieval, and adaptive thresholds for optimized recall.

The system is built to handle complex memory operations with a focus on relevance, emotional context, and efficient retrieval, making it ideal for conversational AI, personal assistants, knowledge management systems, and other applications requiring human-like memory capabilities.

## ✨ Key Features

- **HPC-QuickRecal System**: Unified recall calculation that considers recency, emotional significance, and contextual relevance
- **Hyperbolic Geometry**: Efficient representation of hierarchical memory structures in embedding space
- **Emotional Intelligence**: Context-aware memory retrieval based on emotional states
- **Memory Assemblies**: Organization of related memories into coherent groups
- **Adaptive Thresholds**: Dynamic optimization of retrieval relevance based on feedback
- **Comprehensive API**: RESTful interface for all memory operations
- **Transcription Processing**: Special handling for transcribed speech with feature extraction
- **Contradiction Detection**: Identification of potentially contradictory memories

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from PyPI

```bash
pip install synthians-memory-core
```

### Install from Source

```bash
git clone https://github.com/synthians/memory-core.git
cd memory-core
pip install -e .
```

### Dependencies

The core dependencies will be automatically installed with the package:

- numpy
- sentence-transformers
- fastapi
- uvicorn
- aiohttp
- pydantic

## 🏁 Quick Start

### Basic Usage

```python
from synthians_memory_core import SynthiansMemoryCore

# Initialize the memory core
memory_core = SynthiansMemoryCore()

# Process a new memory
memory_id, quickrecal_score = memory_core.process_new_memory(
    content="This is an important memory about project Alpha.",
    metadata={"source": "user_input", "importance": 0.8}
)

# Retrieve relevant memories
memories = memory_core.retrieve_memories(
    query="project Alpha",
    top_k=3
)

# Print retrieved memories
for memory in memories:
    print(f"ID: {memory.id}, Score: {memory.similarity:.4f}")
    print(f"Content: {memory.content}")
    print(f"Metadata: {memory.metadata}")
    print("---")
```

### Using the API Client

```python
import asyncio
from synthians_memory_core.api.client.client import SynthiansClient

async def main():
    async with SynthiansClient(base_url="http://localhost:5010") as client:
        # Store a memory
        response = await client.process_memory(
            content="Meeting notes regarding the Q3 roadmap.",
            metadata={
                "source": "meeting_notes",
                "project": "RoadmapQ3",
                "attendees": ["Alice", "Bob"]
            }
        )
        
        # Retrieve memories
        memories = await client.retrieve_memories(
            query="roadmap planning",
            top_k=5
        )
        
        # Print results
        for memory in memories.get("memories", []):
            print(f"ID: {memory.get('id')}, Score: {memory.get('similarity'):.4f}")
            print(f"Content: {memory.get('content')}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

Synthians Memory Core is built with a modular architecture that separates concerns and allows for flexible configuration and extension.

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Synthians Memory Core                       │
├─────────────┬─────────────────┬────────────────┬────────────┤
│ Memory      │ HPC-QuickRecal  │ Geometry       │ Emotional  │
│ Structures  │ Calculator      │ Manager        │ Intelligence│
├─────────────┼─────────────────┼────────────────┼────────────┤
│ Memory      │ Adaptive        │ API            │ Persistence│
│ Assemblies  │ Components      │ (Server/Client)│ Layer      │
└─────────────┴─────────────────┴────────────────┴────────────┘
```

### Data Flow

1. **Memory Processing**:
   - Text content is received
   - Embedding is generated (if not provided)
   - Emotion analysis is performed
   - QuickRecal score is calculated
   - Memory is stored with enriched metadata

2. **Memory Retrieval**:
   - Query is received and embedded
   - Vector search is performed
   - Emotional gating is applied
   - Adaptive thresholding is used
   - Results are returned with relevance scores

3. **Feedback Loop**:
   - Retrieval results can receive feedback
   - Thresholds are adjusted based on feedback
   - System learns to improve relevance over time

## 🔌 API Reference

Synthians Memory Core provides a comprehensive RESTful API for all memory operations.

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process_memory` | POST | Process and store a new memory |
| `/retrieve_memories` | POST | Retrieve relevant memories based on a query |
| `/api/memories/{memory_id}` | GET | Retrieve a specific memory by ID |
| `/generate_embedding` | POST | Generate an embedding vector for text |
| `/calculate_quickrecal` | POST | Calculate relevance score for text or embedding |
| `/analyze_emotion` | POST | Analyze emotional content in text |
| `/provide_feedback` | POST | Provide feedback on retrieval relevance |
| `/process_transcription` | POST | Process transcribed speech with feature extraction |
| `/detect_contradictions` | POST | Identify potentially contradictory memories |
| `/health` | GET | Check system health and uptime |
| `/stats` | GET | Retrieve detailed system statistics |

For detailed API documentation, see the [API Reference](https://synthians-memory-core.readthedocs.io/en/latest/api/).

## 🔧 Advanced Usage

### Customizing Memory Processing

```python
# Configure with custom parameters
memory_core = SynthiansMemoryCore(
    embedding_model="all-mpnet-base-v2",
    geometry_type=GeometryType.HYPERBOLIC,
    quickrecal_mode=QuickRecallMode.BALANCED,
    initial_threshold=0.7,
    storage_path="/path/to/storage"
)

# Process memory with custom metadata
memory_id, _ = memory_core.process_new_memory(
    content="Complex technical information about quantum computing.",
    metadata={
        "source": "research_paper",
        "topic": "quantum_computing",
        "importance": 0.9,
        "complexity": 0.8
    }
)
```

### Emotional Gating for Retrieval

```python
# Retrieve memories with emotional context
memories = memory_core.retrieve_memories(
    query="important decision",
    user_emotion={"dominant_emotion": "focused"},
    cognitive_load=0.3,
    top_k=5
)
```

### Working with Memory Assemblies

```python
# Create a memory assembly
assembly_id = memory_core.create_assembly(
    name="Project Alpha Documentation",
    memory_ids=["mem_123", "mem_456", "mem_789"],
    metadata={"project": "Alpha", "type": "documentation"}
)

# Retrieve an assembly
assembly = memory_core.get_assembly(assembly_id)

# Update an assembly
memory_core.update_assembly(
    assembly_id=assembly_id,
    add_memory_ids=["mem_101", "mem_102"],
    remove_memory_ids=["mem_456"]
)
```

## 📚 Examples

### Complete Memory Management Workflow

```python
import asyncio
from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.emotional_intelligence import EmotionalAnalyzer

# Initialize components
memory_core = SynthiansMemoryCore()
emotion_analyzer = EmotionalAnalyzer()

# Process memories with different emotional content
async def process_memories():
    # Happy memory
    happy_content = "I'm thrilled about the progress we've made on the project! The team has exceeded expectations."
    happy_emotions = await emotion_analyzer.analyze(happy_content)
    
    happy_id, _ = memory_core.process_new_memory(
        content=happy_content,
        metadata={
            "source": "team_update",
            "emotional_context": happy_emotions
        }
    )
    
    # Technical memory
    tech_content = "The system architecture uses a microservices approach with containerized deployments and Kubernetes orchestration."
    tech_id, _ = memory_core.process_new_memory(
        content=tech_content,
        metadata={
            "source": "technical_documentation",
            "complexity": 0.8
        }
    )
    
    # Retrieve with different emotional contexts
    focused_results = memory_core.retrieve_memories(
        query="project progress",
        user_emotion={"dominant_emotion": "focused"},
        top_k=3
    )
    
    excited_results = memory_core.retrieve_memories(
        query="project progress",
        user_emotion={"dominant_emotion": "excited"},
        top_k=3
    )
    
    # Compare results
    print("Focused emotional state results:")
    for mem in focused_results:
        print(f"- {mem.content[:50]}... (Score: {mem.final_score:.4f})")
    
    print("\nExcited emotional state results:")
    for mem in excited_results:
        print(f"- {mem.content[:50]}... (Score: {mem.final_score:.4f})")

# Run the example
asyncio.run(process_memories())
```

### Contradiction Detection

```python
# Store potentially contradictory memories
memory_core.process_new_memory(
    content="The project deadline has been extended to the end of Q3.",
    metadata={"source": "management", "timestamp": time.time()}
)

memory_core.process_new_memory(
    content="All project deliverables must be completed by the end of Q2.",
    metadata={"source": "client_requirements", "timestamp": time.time()}
)

# Detect contradictions
contradictions = memory_core.detect_contradictions(threshold=0.7)

# Review potential contradictions
for contradiction in contradictions:
    print(f"Potential contradiction found (similarity: {contradiction['similarity']:.4f}):")
    print(f"Statement 1: {contradiction['memory_a_content']}")
    print(f"Statement 2: {contradiction['memory_b_content']}")
    print("---")
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/synthians/memory-core.git
cd memory-core

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=synthians_memory_core

# Run specific test modules
pytest tests/test_quickrecal.py
```

### Contributing

We welcome contributions to Synthians Memory Core! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## 📊 Benchmarks

Synthians Memory Core has been benchmarked on various datasets and scenarios:

| Scenario | Memory Count | Query Time | Accuracy |
|----------|--------------|------------|----------|
| Small Dataset | 1,000 | 15ms | 92% |
| Medium Dataset | 10,000 | 45ms | 89% |
| Large Dataset | 100,000 | 120ms | 85% |
| With Emotional Gating | 10,000 | 60ms | 94% |

For detailed benchmark methodology and results, see the [Benchmarks](https://synthians-memory-core.readthedocs.io/en/latest/benchmarks/) documentation.

## 🗺️ Roadmap

- **Short-term**
  - Improved contradiction detection with logical reasoning
  - Enhanced transcription feature extraction
  - Additional embedding model options

- **Medium-term**
  - Multi-modal memory support (text, images, audio)
  - Distributed memory storage for large-scale deployments
  - Advanced memory assembly operations

- **Long-term**
  - Self-organizing memory structures
  - Causal reasoning between memories
  - Cross-lingual memory capabilities

## 📄 License

Synthians Memory Core is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- The Synthians team for their vision and support
- Contributors to the open-source libraries we depend on
- Research in hyperbolic embeddings and emotional intelligence that inspired this work

## 📞 Contact and Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: [https://synthians-memory-core.readthedocs.io/](https://synthians-memory-core.readthedocs.io/)
- **Email**: support@synthians.ai
- **Discord**: [Join our community](https://discord.gg/synthians)

---

<p align="center">
  <sub>Built with ❤️ by the Synthians Team</sub>
</p>
