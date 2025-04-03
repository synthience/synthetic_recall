# Synthians Memory Core - Development Guide

This document provides comprehensive guidance for developers working with the Synthians Memory Core project, including setup instructions, coding standards, testing procedures, and contribution guidelines.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/synthians/memory-core.git
   cd memory-core
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Development Dependencies

The development dependencies include:

- **pytest**: For running tests
- **pytest-cov**: For test coverage reporting
- **black**: For code formatting
- **isort**: For import sorting
- **mypy**: For static type checking
- **flake8**: For linting
- **sphinx**: For documentation generation
- **sphinx-rtd-theme**: For documentation theme

## Code Structure

The project follows a modular structure:

```
synthians_memory_core/
├── __init__.py                  # Package exports
├── synthians_memory_core.py     # Main implementation
├── memory_structures.py         # Core data structures
├── hpc_quickrecal.py            # QuickRecal implementation
├── geometry_manager.py          # Geometry handling
├── emotional_intelligence.py    # Emotion analysis and gating
├── memory_persistence.py        # Storage and retrieval
├── adaptive_components.py       # Adaptive thresholds
├── interruption.py              # Interruption handling
├── custom_logger.py             # Logging system
├── api/                         # API implementation
│   ├── __init__.py
│   ├── server.py                # FastAPI server
│   └── client/                  # Client implementation
│       ├── __init__.py
│       └── client.py            # SynthiansClient
└── utils/                       # Utility functions
    ├── __init__.py
    └── transcription_feature_extractor.py
```

## Coding Standards

### Style Guide

The project follows the PEP 8 style guide with some modifications:

- Line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting
- Use type hints for all function signatures

### Code Formatting

Format your code using Black:

```bash
black synthians_memory_core tests
```

Sort imports using isort:

```bash
isort synthians_memory_core tests
```

### Type Checking

Run static type checking with mypy:

```bash
mypy synthians_memory_core
```

### Linting

Run linting with flake8:

```bash
flake8 synthians_memory_core
```

## Testing

### Running Tests

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=synthians_memory_core
```

Generate a coverage report:

```bash
pytest --cov=synthians_memory_core --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use pytest fixtures for common setup
- Aim for at least 80% code coverage

Example test:

```python
import pytest
from synthians_memory_core import SynthiansMemoryCore

@pytest.fixture
def memory_core():
    """Create a memory core instance for testing."""
    return SynthiansMemoryCore()

def test_process_new_memory(memory_core):
    """Test processing a new memory."""
    content = "Test memory content"
    metadata = {"source": "test", "importance": 0.8}
    
    memory_id, quickrecal_score = memory_core.process_new_memory(
        content=content,
        metadata=metadata
    )
    
    assert memory_id is not None
    assert 0.0 <= quickrecal_score <= 1.0
    
    # Verify the memory was stored
    memories = memory_core.retrieve_memories(query="test memory")
    assert len(memories) > 0
    assert memories[0].content == content
```

## Documentation

### Building Documentation

Generate documentation using Sphinx:

```bash
cd docs
make html
```

View the documentation:

```bash
open _build/html/index.html
```

### Documentation Standards

- Use docstrings for all modules, classes, and functions
- Follow Google-style docstring format
- Include type hints in docstrings
- Document parameters, return values, and exceptions
- Provide usage examples for complex functions

Example docstring:

```python
def process_new_memory(
    self, 
    content: str, 
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """Process and store a new memory.
    
    Args:
        content: The text content of the memory.
        metadata: Optional metadata dictionary. If not provided, an empty dict is used.
        embedding: Optional pre-computed embedding. If not provided, an embedding
            is generated from the content.
            
    Returns:
        A tuple containing:
            - memory_id: The unique ID of the stored memory.
            - quickrecal_score: The calculated QuickRecal score (0.0-1.0).
            
    Raises:
        ValueError: If content is empty and no embedding is provided.
        
    Example:
        >>> memory_id, score = memory_core.process_new_memory(
        ...     content="Important memory",
        ...     metadata={"source": "user", "importance": 0.8}
        ... )
        >>> print(f"Stored memory {memory_id} with score {score:.2f}")
        Stored memory mem_12345 with score 0.85
    """
```

## Contribution Guidelines

### Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Format your code
6. Submit a pull request

### Pull Request Guidelines

- Provide a clear description of the changes
- Link to any related issues
- Include tests for new functionality
- Ensure all tests pass
- Follow the coding standards
- Update documentation as needed

### Commit Message Guidelines

Follow the conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code changes that neither fix bugs nor add features
- perf: Performance improvements
- test: Adding or updating tests
- chore: Maintenance tasks

Example:
```
feat(memory): add support for memory tagging

Add ability to tag memories with custom tags for easier filtering.
Includes new API endpoints and client methods.

Closes #123
```

## Release Process

### Version Numbering

The project follows semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible new functionality
- PATCH: Backwards-compatible bug fixes

### Creating a Release

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create a release commit
4. Tag the release
5. Push to GitHub
6. Create a GitHub release
7. Publish to PyPI

```bash
# Update version in __init__.py
# Update CHANGELOG.md

# Commit changes
git add .
git commit -m "chore(release): prepare for v1.2.0"

# Tag the release
git tag -a v1.2.0 -m "Version 1.2.0"

# Push to GitHub
git push origin main
git push origin v1.2.0

# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## Debugging

### Logging

The project uses a custom logger that can be configured for different verbosity levels:

```python
from synthians_memory_core.custom_logger import logger

# Log levels: DEBUG, INFO, WARNING, ERROR
logger.set_level("DEBUG")

# Log messages
logger.debug("component_name", "Debug message", {"extra": "data"})
logger.info("component_name", "Info message", {"extra": "data"})
logger.warning("component_name", "Warning message", {"extra": "data"})
logger.error("component_name", "Error message", {"extra": "data"})
```

### Debugging Tools

- Use the `debug` endpoint on the API server to get detailed system state
- Enable debug mode in the memory core:
  ```python
  memory_core = SynthiansMemoryCore(debug=True)
  ```
- Use the `get_stats()` method to retrieve system statistics

## Performance Optimization

### Memory Usage Optimization

- Use batch processing for large operations
- Implement memory cleanup for unused embeddings
- Configure appropriate vector index parameters

### Speed Optimization

- Use pre-computed embeddings when possible
- Implement caching for frequent operations
- Configure appropriate batch sizes

### Profiling

Profile code performance:

```python
import cProfile
import pstats

# Profile a function
cProfile.run('memory_core.retrieve_memories(query="test")', 'retrieve_stats')

# Analyze results
p = pstats.Stats('retrieve_stats')
p.sort_stats('cumulative').print_stats(10)
```

## Troubleshooting

### Common Issues

1. **Vector Index Errors**
   - Solution: Use the `repair_index` endpoint to fix index issues

2. **Memory Leaks**
   - Solution: Ensure proper cleanup of large objects, especially embeddings

3. **Slow Retrieval**
   - Solution: Optimize vector index parameters, use batch retrieval

4. **Embedding Dimension Mismatch**
   - Solution: Ensure consistent embedding models, use dimension alignment

### Getting Help

- Open an issue on GitHub
- Join the Discord community
- Check the FAQ in the documentation
- Contact the maintainers at support@synthians.ai

## Advanced Development

### Custom Embedding Models

Integrate custom embedding models:

```python
from sentence_transformers import SentenceTransformer
from synthians_memory_core import SynthiansMemoryCore

# Create custom embedding model
custom_model = SentenceTransformer("custom-model-name")

# Initialize memory core with custom model
memory_core = SynthiansMemoryCore(embedding_model=custom_model)
```

### Custom Storage Backends

Implement a custom storage backend:

```python
from synthians_memory_core.memory_persistence import MemoryPersistence

class CustomStorage(MemoryPersistence):
    """Custom storage implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize storage connection
        
    def store_memory(self, memory_entry):
        # Implement storage logic
        
    def retrieve_memory(self, memory_id):
        # Implement retrieval logic
        
    def list_memories(self, filter_criteria=None):
        # Implement listing logic

# Use custom storage
memory_core = SynthiansMemoryCore(
    persistence_provider=CustomStorage("connection-string")
)
```

### Plugin Development

Create plugins for the memory core:

```python
from synthians_memory_core import SynthiansMemoryCore

class MemoryAnalyticsPlugin:
    """Plugin for memory analytics."""
    
    def __init__(self, memory_core: SynthiansMemoryCore):
        self.memory_core = memory_core
        self.register_hooks()
        
    def register_hooks(self):
        # Register hooks for memory events
        self.memory_core.on_memory_created(self.on_memory_created)
        self.memory_core.on_memory_retrieved(self.on_memory_retrieved)
        
    def on_memory_created(self, memory_id: str, memory_data: dict):
        # Handle memory creation event
        
    def on_memory_retrieved(self, memory_id: str, query: str):
        # Handle memory retrieval event

# Use the plugin
memory_core = SynthiansMemoryCore()
analytics_plugin = MemoryAnalyticsPlugin(memory_core)
```

## Appendix

### Glossary

- **QuickRecal**: The system for calculating memory relevance scores
- **Embedding**: Vector representation of text content
- **Memory Assembly**: Group of related memories
- **Emotional Gating**: Filtering memories based on emotional context
- **Threshold Calibration**: Dynamic adjustment of similarity thresholds
- **Vector Index**: Efficient index for similarity search

### References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hyperbolic Embeddings Paper](https://arxiv.org/abs/1705.08039)
- [Emotional Intelligence in AI Systems](https://example.com/emotional-intelligence)
