# Sequence Context Management

**Author:** Lucidia Core Team  
**Date:** 2025-03-30  
**Status:** Implemented

## Overview

The `SequenceContextManager` is responsible for maintaining a history of cognitive operations for use in attention mechanisms within the Titans Architecture variants. It provides a fixed-length buffer of recent processing steps including input embeddings, projections, and outputs, enabling temporal context for attention calculations.

## Implementation Details

The `SequenceContextManager` is implemented in `orchestrator/history.py` and uses a `collections.deque` with a fixed maximum length to efficiently manage the sequence history.

### Context Structure

Each context entry is stored as a tuple with the following components:

```python
ContextTuple = Tuple[float, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)
```

Where:
- `timestamp`: When the entry was processed (float)
- `memory_id`: Unique identifier of the memory (string)
- `x_t`: Original input embedding (numpy array)
- `k_t`: Key projection (numpy array)
- `v_t`: Value projection (numpy array)
- `q_t`: Query projection (numpy array)
- `y_t`: Neural memory output embedding (numpy array)

## API Reference

### Constructor

```python
SequenceContextManager(max_length: int = 50)
```

**Parameters:**
- `max_length`: Maximum number of context tuples to store (default: 50)

### Methods

#### add_context

```python
def add_context(
    self,
    memory_id: str,
    x_t: np.ndarray,
    k_t: np.ndarray,
    v_t: np.ndarray,
    q_t: np.ndarray,
    y_t: np.ndarray,
    timestamp: Optional[float] = None
) -> None
```

Adds a new context element (tuple) to the buffer.

**Parameters:**
- `memory_id`: Identifier for the memory entry
- `x_t`: Input embedding
- `k_t`: Key projection
- `v_t`: Value projection
- `q_t`: Query projection
- `y_t`: Neural memory output embedding
- `timestamp`: Optional timestamp (defaults to current time)

#### update_last_context

```python
def update_last_context(self, y_t: np.ndarray) -> bool
```

Updates the most recent context entry with the y_t value. This is useful when y_t is not available at the time of initial context creation.

**Parameters:**
- `y_t`: The retrieved embedding (output from Neural Memory)

**Returns:**
- `True` if update was successful, `False` otherwise

#### get_recent_history

```python
def get_recent_history(self, count: Optional[int] = None) -> List[ContextTuple]
```

Returns the most recent context tuples.

**Parameters:**
- `count`: Optional number of items to retrieve (defaults to all available)

**Returns:**
- List of context tuples

#### Retrieval Helper Methods

The following methods extract specific components from the history:

```python
def get_recent_keys(self, count: Optional[int] = None) -> List[np.ndarray]
def get_recent_values(self, count: Optional[int] = None) -> List[np.ndarray]
def get_recent_queries(self, count: Optional[int] = None) -> List[np.ndarray]
def get_recent_outputs(self, count: Optional[int] = None) -> List[np.ndarray]
```

Each method returns a list of the specific components (k_t, v_t, q_t, or y_t) from the most recent entries.

#### Convenience Methods for Attention

```python
def get_recent_kv_pairs(self, count: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]
def get_recent_ky_pairs(self, count: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]
```

These methods return pairs of components specifically needed for attention calculations:
- `get_recent_kv_pairs`: Returns (keys, values) for MAL variant
- `get_recent_ky_pairs`: Returns (keys, outputs) for MAC variant

#### Utility Methods

```python
def __len__(self) -> int  # Returns the current number of items in the buffer
def clear(self) -> None    # Clears the context buffer
```

## Integration with Titans Variants

The different Titans variants use the sequence context in different ways:

- **MAC (Memory-Attended Computation):**
  - Uses `get_recent_ky_pairs()` to retrieve historical keys and output embeddings
  - Applies attention between current query and history to enhance the retrieved output

- **MAG (Memory-Attended Gates):**
  - Uses `get_recent_keys()` to retrieve historical keys
  - Applies attention between current query and historical keys to calculate gate values

- **MAL (Memory-Attended Learning):**
  - Uses `get_recent_kv_pairs()` to retrieve historical keys and values
  - Applies attention to modify the value projection before neural memory update

## Usage Example

```python
# Create a sequence context manager with max 100 entries
sequence_manager = SequenceContextManager(max_length=100)

# Add a new context entry after processing
sequence_manager.add_context(
    memory_id="mem_12345",
    x_t=input_embedding,
    k_t=key_projection,
    v_t=value_projection,
    q_t=query_projection,
    y_t=output_embedding
)

# Retrieve historical keys and values for attention
historical_keys, historical_values = sequence_manager.get_recent_kv_pairs(count=10)

# Apply attention between current query and history
attention_weights = calculate_attention(current_query, historical_keys)
attended_value = np.sum(attention_weights[:, np.newaxis] * historical_values, axis=0)
```

## Best Practices

1. **Buffer Size Management:** Choose an appropriate `max_length` value that balances memory usage with sufficient context for attention calculations. The default of 50 is sufficient for most scenarios.

2. **Embedding Validation:** Always ensure that embeddings passed to `add_context()` are valid numpy arrays to prevent issues with attention calculations.

3. **Context Population:** Allow sufficient context to accumulate before relying heavily on attention mechanisms. Variants can handle empty or small history buffers, but their effectiveness improves with more context.

4. **Temporal Relevance:** Consider that older context entries may be less relevant. The deque automatically removes the oldest entries when full, maintaining recency.
