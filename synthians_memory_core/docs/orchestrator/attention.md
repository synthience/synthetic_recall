# Attention Mechanism in Titans Variants

**Author:** Lucidia Core Team
**Date:** 2025-03-30
**Status:** Implemented

## Overview

The attention mechanism is a core component of Lucidia's Phase 4 implementation, providing the foundation for the Titans Architecture Variants (MAC, MAG, MAL). Each variant directly incorporates TensorFlow's `tf.keras.layers.MultiHeadAttention` layer to enable sophisticated temporal context awareness and enhanced memory operations.

> *"Attention is the lens through which memory gains focus."*

## Implementation Details

The attention mechanism is implemented within each Titans variant class in `orchestrator/titans_variants.py`, utilizing TensorFlow's built-in multi-head attention layer with configuration specific to each variant's needs.

### Key Features

1. **Robust Embedding Handling**:
   - Validation of input embeddings through wrapper methods
   - Automatic dimension alignment (384D vs 768D handling) via the GeometryManager
   - Proper batching and reshaping of inputs before passing to attention mechanism

2. **Performance Optimizations**:
   - Configurable number of attention heads (default: 4)
   - Per-head dimension control (default: 32)
   - Optional dropout for regularization (default: 0.0)

3. **Variant-Specific Applications**:
   - **MAC**: Enhances memory retrieval by attending over historical memory outputs
   - **MAG**: Modifies gate values for neural memory updates by attending over historical keys
   - **MAL**: Modifies value projections by attending over historical values

4. **Integration with Sequence Context**:
   - Maintains history of recent memory operations via SequenceContextManager
   - Provides temporal context for attention operations

## Configuration

The attention mechanism is configured via the `TitansVariantConfig` class with the following parameters:

```python
# Default configuration values
defaults = {
    "variant": TitansVariantType.NONE.value,  # NONE, MAC, MAG, or MAL
    "attention_num_heads": 4,              # Number of attention heads
    "attention_key_dim": 32,               # Dimension per head
    "attention_dropout": 0.0,              # Dropout rate
    "max_context_length": 50,             # Max sequence history length
    "max_dim_mismatch_warnings": 10,      # Rate limiting for warnings
}
```

## Variant-Specific Implementations

### MAC (Memory-Attended Computation)

The MAC variant enhances memory retrieval by attending over historical memory outputs:

```python
# Simplified example from MACVariant.__init__
self.attention_module = tf.keras.layers.MultiHeadAttention(
    num_heads=attention_config["num_heads"],
    key_dim=attention_config["key_dim"],
    dropout=attention_config["dropout"],
    name="MAC_Attention"
)
```

Flow: `q_t -> M -> y_t -> Attend(q_t, K_hist, Y_hist) -> attended_y_t`

### MAG (Memory-Attended Gates)

The MAG variant modifies gate values for neural memory updates:

```python
# Simplified example from MAGVariant.__init__
self.attention_module = tf.keras.layers.MultiHeadAttention(
    num_heads=attention_config["num_heads"],
    key_dim=attention_config["key_dim"],
    dropout=attention_config["dropout"],
    name="MAG_Attention"
)
```

Flow: 
1. `q_t -> Attend(q_t, K_hist, K_hist) -> attention_output`
2. Call Neural Memory's `/calculate_gates` endpoint with attention output
3. Update memory with calculated gates

### MAL (Memory-Augmented Learning)

The MAL variant modifies value projections for neural memory updates:

```python
# Simplified example from MALVariant.__init__
self.attention_module = tf.keras.layers.MultiHeadAttention(
    num_heads=attention_config["num_heads"],
    key_dim=attention_config["key_dim"],
    dropout=attention_config["dropout"],
    name="MAL_Attention"
)
```

Flow: 
1. `q_t, K_hist, V_hist -> Attend(q_t, K_hist, V_hist) -> attended_v_t`
2. Combine `attended_v_t` with `v_t` -> `v_prime_t`
3. Update memory with `k_t` and `v_prime_t`

## Usage Example

The ContextCascadeEngine coordinates the use of attention mechanisms within the appropriate variant:

```python
# Example configuration in ContextCascadeEngine
variant_config = TitansVariantConfig(
    variant="MAC",                # Use Memory-Attended Computation variant
    attention_num_heads=8,       # 8 attention heads
    attention_key_dim=64,        # 64 dimensions per head
    attention_dropout=0.1,       # 10% dropout for regularization
    max_context_length=100       # Remember up to 100 prior interactions
)

# Initialize the engine with this configuration
engine = ContextCascadeEngine(
    memory_core_url="http://localhost:5010",
    neural_memory_url="http://localhost:8001",
    variant_config=variant_config
)
```

## Best Practices

1. **Sequence Length**: Balance history length with computational resources; longer sequences provide more context but require more memory and processing time.

2. **Embedding Dimension**: Ensure the embedding dimension is consistent or properly aligned with the GeometryManager when using multiple embedding models.

3. **Head Configuration**: More attention heads allow finer-grained focus but increase computational cost. The default of 4 heads with 32 dimensions per head works well for most scenarios.

4. **Variant Selection**: 
   - Use MAC for improved retrieval quality when sequence matters
   - Use MAG for dynamic adjustments to memory learning rates based on context
   - Use MAL for directly influencing what is stored in memory
