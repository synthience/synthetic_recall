# Lucid Recall Memory System

## Ephemeral Memory Implementation

### Overview
The system uses an ephemeral memory store that maintains memories in RAM with time-based decay and significance weighting. This implementation prioritizes speed and recency over persistence.

### Key Features
1. **In-Memory Storage**
   - RAM-based storage
   - No persistence between restarts
   - Fast access and retrieval
   - Automatic cleanup on container restart

2. **Time Decay**
   - Newer memories get higher priority
   - Exponential decay based on time difference
   - Configurable decay rate
   - Formula: `exp(-decay_rate * (current_time - memory_time))`

3. **Significance Weighting**
   - Memories weighted by significance score
   - Combines with similarity for ranking
   - Affects search results ordering
   - Range: 0.0 to 1.0

4. **Memory Structure**
```python
Memory = {
    'id': str,          # Unique identifier
    'embedding': tensor, # Normalized embedding vector
    'timestamp': float, # Unix timestamp
    'significance': float, # Importance score
    'content': str      # Original text content
}
```

### Memory Operations

#### 1. Storage
```python
async def add_memory(memory_id, embedding, timestamp, significance=1.0, content=None):
    # Normalize embedding
    # Add to memory list
    # Rebuild index if threshold reached
```

#### 2. Retrieval
```python
def search(query_embedding, k=5):
    # Normalize query
    # Calculate similarities
    # Apply significance weights
    # Apply time decay
    # Return top-k results
```

#### 3. Index Management
```python
def build_index():
    # Stack embeddings
    # Normalize vectors
    # Build search index
```

### Configuration Parameters
- `embedding_dim`: 384 (matches model output)
- `rebuild_threshold`: 100 memories
- `time_decay`: 0.01
- `min_similarity`: 0.7
- `batch_size`: 32

### Performance Characteristics

#### Memory Usage
- Per Memory: ~1.5KB + embedding size
- Embedding Size: 384 * 4 bytes = 1.5KB
- Index Size: N * 1.5KB (N = number of memories)

#### Time Complexity
- Storage: O(1)
- Search: O(N) where N is number of memories
- Index Rebuild: O(N) on every rebuild_threshold insertions

#### GPU Acceleration
- Embedding operations GPU-accelerated
- Similarity calculations on GPU
- Index operations on GPU when available

### Integration Points

#### 1. Tensor Server
```python
# Store memory
await tensor_ws.send(json.dumps({
    "type": "embed",
    "text": "content"
}))

# Search memories
await tensor_ws.send(json.dumps({
    "type": "search",
    "text": "query",
    "limit": 5
}))
```

#### 2. HPC Server
```python
# Process embedding
await hpc_ws.send(json.dumps({
    "type": "process",
    "embeddings": embedding_data
}))
```

### System Limitations

1. **Memory Constraints**
   - Limited by available RAM
   - No persistence
   - Requires rebuild on restart
   - Memory loss on container shutdown

2. **Scaling Limitations**
   - Single node only
   - No distributed storage
   - Linear search complexity
   - Index rebuild overhead

3. **Operational Considerations**
   - Regular memory monitoring needed
   - Performance degrades with size
   - GPU memory management required
   - Backup strategy needed for important data

### Future Enhancements

1. **Planned Improvements**
   - Persistent storage option
   - Distributed memory support
   - Optimized index structures
   - Automatic pruning

2. **Performance Optimizations**
   - Approximate nearest neighbors
   - Batch processing
   - Index sharding
   - Memory compression

3. **Feature Additions**
   - Memory categories
   - Priority queues
   - Custom decay functions
   - Memory lifecycle management