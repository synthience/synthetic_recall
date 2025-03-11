# Long-Term Memory (LTM) System Guide

This document provides detailed information about the Lucidia Long-Term Memory (LTM) system, its implementation, configuration, and usage.

## Overview

The Long-Term Memory (LTM) component is a critical part of Lucidia's hierarchical memory architecture. It provides persistent, significance-weighted storage that ensures only important memories are retained long-term. The system implements dynamic significance decay, allowing less important memories to fade naturally over time while preserving critical information.

## Key Features

- **Significance-Based Persistence**: Only memories exceeding a configurable significance threshold are stored
- **Asynchronous Batch Processing**: Improves performance by batching persistence operations
- **Dynamic Decay**: Automatically reduces significance of memories over time based on configurable decay rates
- **Category-Based Organization**: Organizes memories by categories for efficient retrieval
- **Flexible Metadata**: Supports arbitrary metadata for rich memory context
- **Memory Boosting**: Increases significance of frequently accessed memories
- **Automatic Purging**: Removes memories that fall below significance thresholds

## Technical Implementation

The LTM system is implemented in `memory/lucidia_memory_system/core/long_term_memory.py`. The primary class is `LongTermMemory`, which manages all aspects of persistent memory storage.

### Core Components

1. **Memory Storage**: In-memory dictionary indexed by unique IDs
2. **Category Index**: Secondary index organizing memories by categories
3. **Batch Queue**: Deque for asynchronous persistence operations
4. **Statistics Tracking**: Comprehensive performance metrics

### Configuration Options

```python
default_config = {
    'storage_path': os.path.join('/app/memory/stored', 'ltm'),  # Storage location
    'significance_threshold': 0.7,  # Minimum significance for storage
    'max_memories': 10000,          # Maximum number of memories to store
    'decay_rate': 0.05,             # Base decay rate (per day)
    'decay_check_interval': 86400,  # Time between decay checks (1 day)
    'min_retention_time': 604800,   # Minimum retention time regardless of decay (1 week)
    'embedding_dim': 384,           # Embedding dimension
    'enable_persistence': True,     # Whether to persist memories to disk
    'purge_threshold': 0.3,         # Memories below this significance get purged
    'batch_size': 50,               # Max operations in a batch
    'batch_interval': 5.0,          # Max seconds between batch processing
    'batch_retries': 3,             # Number of retries for failed batch operations
    'batch_retry_delay': 1.0,       # Delay between retries (seconds)
}
```

## Key Methods

### Memory Storage and Retrieval

```python
async def store_memory(self, content: str, embedding: Optional[torch.Tensor] = None,
                     significance: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]
```

Stores a memory in long-term storage if it meets the significance threshold.

```python
async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]
```

Retrieves a specific memory by ID.

```python
async def search_memory(self, query: str, limit: int = 5, 
                       min_significance: float = 0.0,
                       categories: Optional[List[str]] = None)
```

Searches for memories based on text content, with filtering by significance and categories.

### Memory Maintenance

```python
async def _run_decay_and_purge(self)
```

Runs decay calculations and purges low-significance memories.

```python
async def _purge_memory(self, memory_id: str)
```

Purges a specific memory from storage.

```python
async def backup(self)
```

Creates a backup of all memories.

## Integration with MemoryCore

The LTM system integrates with the broader memory architecture through the `MemoryCore` component, which manages the hierarchical memory system with STM, LTM, and MPL layers. The LTM provides the persistent storage layer for this architecture.

## Docker Integration Notes

When running in a Docker environment, the LTM system uses the `/app/memory/stored/ltm` path for storage. When running outside of Docker, this path should be adjusted to a local directory, as was done in the testing implementation with:

```python
local_storage_path = Path(os.path.join(os.getcwd(), "memory", "stored", "ltm"))
```

## Testing and Validation

The `test_ltm_storage.py` script demonstrates how to properly test the LTM system. It creates test memories with high significance values and various metadata, stores them in the LTM system, and then verifies retrieval.

Key aspects of testing include:

1. **Path Configuration**: Adjusting storage paths for the test environment
2. **Memory Verification**: Confirming that stored memories can be retrieved correctly
3. **Category Organization**: Validating that memories are properly organized by category
4. **Proper Shutdown**: Ensuring that the LTM system shuts down gracefully after batch processing

## Integration with Reflection Engine

The LTM system works in concert with the Reflection Engine, providing persistent storage for significant insights and reflections generated during dream processing. The Reflection Engine uses the LTM to store and retrieve memories that inform its analysis.

## Best Practices

1. **Significance Thresholds**: Configure significance thresholds carefully based on the importance of different memory types
2. **Categorization**: Use clear, consistent categories for memories to aid in retrieval
3. **Metadata Enrichment**: Provide rich metadata to enhance memory context and searchability
4. **Regular Backups**: Implement regular backups of the LTM store, especially before significant system changes
5. **Performance Monitoring**: Track LTM statistics to identify performance bottlenecks

## Common Issues and Solutions

### Memory Not Being Stored

- **Check significance threshold**: Ensure the memory's significance is above the configured threshold
- **Verify persistence settings**: Confirm that `enable_persistence` is set to `True`
- **Check storage path**: Ensure the storage directory exists and is writable

### Slow Performance

- **Adjust batch settings**: Increase `batch_size` or decrease `batch_interval` for more efficient processing
- **Review memory count**: If approaching `max_memories`, consider purging unused memories or increasing the limit
- **Monitor decay operations**: Frequent decay calculations can impact performance

### Memory Leaks

- **Ensure proper shutdown**: Always call `shutdown()` when finished with the LTM system
- **Check for references**: Ensure no external references are keeping memory objects alive
- **Monitor memory growth**: Track memory usage over time to identify potential leaks

## Future Enhancements

1. **Vector Database Integration**: Add support for dedicated vector databases for improved embedding search
2. **Distributed Storage**: Support for distributed memory storage across multiple nodes
3. **Memory Compression**: Implement memory compression techniques for efficient storage
4. **Enhanced Security**: Add encryption and access controls for sensitive memories
5. **Memory Versioning**: Track changes to memories over time with versioning
