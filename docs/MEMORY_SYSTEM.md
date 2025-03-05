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

# Lucidia Memory System Architecture

## Overview

The Lucidia Memory System is a sophisticated component designed to provide context-aware, significance-based memory management for the voice assistant. It enables Lucidia to remember important information across conversations, detect personal details, and respond intelligently to memory-related queries.

## Key Components

### 1. Memory Client (`memory_client.py`)

The central interface for all memory operations, responsible for:

- Storing and retrieving memories
- Processing embeddings through the HPC server
- Detecting and storing personal details
- Providing memory tools for LLM integration
- Managing memory significance scoring

### 2. HPC Server (`hpc_server.py`)

A WebSocket server that processes embeddings and calculates significance scores:

- Handles embedding normalization
- Calculates surprise and diversity scores
- Maintains momentum buffer for context comparison
- Provides significance scores for memory storage decisions

### 3. HPC-SIG Flow Manager (`hpc_sig_flow_manager.py`)

Manages the hypersphere processing chain for embeddings:

- Projects embeddings to unit hypersphere
- Calculates surprise scores based on momentum buffer
- Applies shock absorption for high-surprise inputs
- Computes significance scores based on multiple factors

### 4. LLM Pipeline Integration (`llm_pipeline.py`)

Connects the memory system to the LLM service:

- Handles memory-related tool calls
- Processes search queries with significance filtering
- Enables LLM to store significant memories

## Memory Significance Calculation

The system uses a multi-factor approach to calculate memory significance:

1. **Surprise Score (40%)**: How unexpected the embedding is compared to recent context
2. **Magnitude (30%)**: The norm of the embedding vector
3. **Diversity (30%)**: How different the embedding is from others in the momentum buffer

This calculation occurs in the HPC-SIG Flow Manager and determines which memories are worth storing and retrieving.

## Memory Storage Flow

1. Text input is received (user message, detected personal detail, etc.)
2. Memory Client sends text to HPC Server for embedding and significance calculation
3. HPC Server processes the text through the HPC-SIG Flow Manager
4. Significance score is calculated and returned with the processed embedding
5. Memory Client stores the memory if significance exceeds the threshold

## Memory Retrieval Flow

1. LLM makes a tool call to search memory
2. Memory Client processes the query through the HPC Server
3. Query embedding is compared to stored memories for similarity
4. Results are filtered by memory type and minimum significance
5. Matching memories are returned to the LLM

## Personal Detail Detection

The system automatically detects and stores personal details with high significance (0.95):

- User's name from patterns like "My name is John", "I'm John", etc.
- Filters out common non-name phrases
- Future expansion planned for other personal details (age, location, etc.)

## LLM Tool Integration

The memory system provides two primary tools for LLM interaction:

1. **search_memory**: Searches memories with filters for type, significance, and time range
2. **store_significant_memory**: Allows the LLM to explicitly store important information

These tools enable the LLM to actively query and manage the memory system rather than relying solely on RAG context retrieval.

## Configuration Parameters

- **Embedding Dimension**: 384 (matches the all-MiniLM-L6-v2 model)
- **Momentum**: 0.9 (for surprise calculation)
- **Diversity Threshold**: 0.7
- **Surprise Threshold**: 0.8
- **Significance Threshold**: Configurable, defaults to 0.0

## Future Improvements

1. Expand personal detail detection patterns
2. Implement memory consolidation for related memories
3. Add time-based memory decay
4. Develop more granular memory type detection
5. Enhance significance calculation with additional factors

## Technical Implementation Notes

- Uses WebSocket connections for real-time communication
- Supports GPU acceleration when available
- Implements caching for efficiency
- Provides persistence across sessions
- Handles connection retries and error recovery