# MultiHeadAttentionModule

**Author:** Lucidia Core Team
**Date:** 2025-03-28
**Status:** Implemented

## Overview

The `MultiHeadAttentionModule` is a core component of Lucidia's Phase 4 implementation, providing a configurable attention mechanism that enables the Titans Architecture Variants (MAC, MAG, MAL) to operate effectively. This document details the implementation, usage, and configuration of this module.

> *"Attention is the lens through which memory gains focus."*

## Implementation Details

The attention module is implemented in `synthians_trainer_server/attention.py` and builds upon TensorFlow's multi-head attention layer with significant enhancements for robustness and debugging.

### Key Features

1. **Robust Embedding Handling**:
   - Validation of input embeddings (checking for NaN/Inf values)
   - Automatic dimension alignment (384D vs 768D handling)
   - Proper batching and reshaping of inputs

2. **Performance Optimizations**:
   - Configurable number of attention heads
   - Per-head dimension control
   - Optional dropout for regularization

3. **Architectural Enhancements**:
   - Optional residual connections
   - Optional layer normalization
   - Configurable activation functions

4. **Metrics and Diagnostics**:
   - Attention score capture
   - Attention entropy calculation
   - Sparsity measurements

## API Reference

### Constructor

```python
MultiHeadAttentionModule(
    num_heads=4,
    key_dim=32,
    value_dim=None,  # Defaults to key_dim
    dropout=0.0,
    use_bias=True,
    use_layer_norm=True,
    use_residual=True,
    activation=None,
    name="multi_head_attention",
    **kwargs
)
```

### Key Methods

#### call

```python
def call(
    self, 
    query,
    key,
    value,
    return_attention_scores=False,
    training=None
)
```

Applies attention mechanism to query, key, and value tensors.

- **Parameters**:
  - `query`: Query tensor of shape `[batch_size, query_length, query_dim]`
  - `key`: Key tensor of shape `[batch_size, key_length, key_dim]`
  - `value`: Value tensor of shape `[batch_size, value_length, value_dim]`
  - `return_attention_scores`: If True, returns attention scores along with output
  - `training`: Boolean indicating whether in training mode

- **Returns**:
  - If `return_attention_scores=False`: Output tensor of shape `[batch_size, query_length, output_dim]`
  - If `return_attention_scores=True`: Tuple of (output tensor, attention scores)

#### process_sequence

```python
def process_sequence(
    self,
    query,
    keys,
    values,
    return_attention_scores=False
)
```

Processes a query against sequences of keys and values.

- **Parameters**:
  - `query`: Query tensor of shape `[query_dim]`
  - `keys`: List of key tensors, each of shape `[key_dim]`
  - `values`: List of value tensors, each of shape `[value_dim]`
  - `return_attention_scores`: If True, returns attention scores along with output

- **Returns**:
  - If `return_attention_scores=False`: Output tensor of shape `[output_dim]`
  - If `return_attention_scores=True`: Tuple of (output tensor, attention scores)

## Usage Examples

### Basic Usage

```python
import tensorflow as tf
from synthians_trainer_server.attention import MultiHeadAttentionModule

# Initialize the attention module
attention = MultiHeadAttentionModule(
    num_heads=4,
    key_dim=32,
    use_layer_norm=True,
    use_residual=True
)

# Create sample inputs
query = tf.random.normal([1, 1, 128])  # Single query
keys = tf.random.normal([1, 10, 128])   # 10 key vectors
values = tf.random.normal([1, 10, 128]) # 10 value vectors

# Apply attention
output, attention_scores = attention(
    query, keys, values, 
    return_attention_scores=True
)

print(f"Output shape: {output.shape}")
print(f"Attention scores shape: {attention_scores.shape}")
```

### Integration with Titans Variants

```python
from orchestrator.titans_variants import MACVariant
from synthians_trainer_server.attention import MultiHeadAttentionModule

# Create attention module
attention_module = MultiHeadAttentionModule(
    num_heads=4,
    key_dim=32,
    use_layer_norm=True,
    use_residual=True
)

# Initialize MAC variant with attention module
mac_variant = MACVariant(
    attention_module=attention_module,
    sequence_context_manager=sequence_context_manager
)

# Process input through variant
result = mac_variant.process_input(
    x_t=current_embedding,
    k_t=key_projection,
    v_t=value_projection,
    q_t=query_projection,
    y_t=retrieved_embedding
)
```

## Configuration

The attention module can be configured in several ways to optimize for different use cases:

### Optimizing for Sequence Handling

```python
# For long-term dependencies
attention = MultiHeadAttentionModule(
    num_heads=8,  # More heads for finer-grained attention
    key_dim=16,   # Smaller per-head dimension
    dropout=0.1   # Add dropout for regularization
)

# For focused, specific attention
attention = MultiHeadAttentionModule(
    num_heads=2,  # Fewer heads for more focused attention
    key_dim=64,   # Larger per-head dimension for more capacity
    use_residual=False  # Disable residual to force attention-based output
)
```

### Dynamic Configuration

The attention module parameters should ideally be aligned with the Neural Memory configuration. Future implementations should fetch these parameters dynamically:

```python
# Get Neural Memory configuration
config_response = await neural_memory_client.get_config()

# Configure attention module based on Neural Memory parameters
attention = MultiHeadAttentionModule(
    num_heads=config_response.attention_heads,
    key_dim=config_response.key_dim // config_response.attention_heads,
    dropout=config_response.attention_dropout
)
```

## Debugging and Metrics

The attention module provides several metrics for debugging and performance analysis:

```python
# Get output and attention scores
output, scores = attention(query, keys, values, return_attention_scores=True)

# Calculate entropy of attention distribution
entropy = -tf.reduce_sum(scores * tf.math.log(scores + 1e-10), axis=-1)
print(f"Attention entropy: {entropy}")  # Higher entropy = more distributed attention

# Calculate sparsity (% of scores below threshold)
sparsity = tf.reduce_mean(tf.cast(scores < 0.01, tf.float32))
print(f"Attention sparsity: {sparsity}")  # Higher sparsity = more focused attention
```

## Limitations and Future Work

1. **Scale Limitations**: The current implementation may have performance issues with very long sequences (1000+). Future versions could implement sparse attention or other efficient attention mechanisms.

2. **Memory Usage**: For long sequences, memory usage can be significant. Consider implementing attention with linear complexity (e.g., Performer, Reformer).

3. **TensorFlow Dependency**: The current implementation relies on TensorFlow. A PyTorch version might be beneficial for some deployment scenarios.

4. **Dynamic Configuration**: Implement a dedicated endpoint in the Neural Memory server to expose configuration parameters for attention.

5. **Attention Visualization**: Add tooling to visualize attention patterns for better interpretability.

---

**Related Documentation:**
- [Phase 4 Implementation](phase_4_implementation.md)
- [Titans Variants Integration](titans_variants_integration.md)
- [API Updates](api_updates.md)
