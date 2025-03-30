# Phase 4 Implementation: Titans Architecture Variants

**Author:** Lucidia (MEGA)
**Date:** 2025-03-28 15:45:00 UTC
**Status:** Complete

## Overview

This document details the implementation of the Titans Architecture Variants (MAC, MAG, MAL) as outlined in Section 4 of the Titans paper. Phase 4 extends Lucidia's cognitive architecture by integrating attention mechanisms with the Neural Memory module, enhancing its adaptive capabilities and contextual awareness.

> *"The blueprint remembers, but attention shapes what is recalled."*

## Implementation Components

The implementation consists of five key components:

1. **MultiHeadAttentionModule**: A robust attention mechanism implemented in `synthians_trainer_server/attention.py`
2. **SequenceContextManager**: A deque-based context buffer in `orchestrator/history.py`
3. **Neural Memory API Extensions**: Enhanced API endpoints in `synthians_trainer_server/http_server.py`
4. **Titans Variant Implementations**: Base class and specific variant implementations in `orchestrator/titans_variants.py`
5. **ContextCascadeEngine Integration**: Connection of variants to the orchestration layer in `orchestrator/context_cascade_engine.py`

## Detailed Implementation

### 1. MultiHeadAttentionModule

Implemented in `synthians_trainer_server/attention.py`, this module provides a configurable multi-head attention mechanism with:

- Dimension validation and standardization (handles the 384D vs 768D embedding mismatch issues)
- Optional residual connections and layer normalization
- Metrics tracking for attention scores, entropy, and sparsity
- Robust error handling for malformed embeddings and NaN/Inf values

```python
class MultiHeadAttentionModule(tf.keras.layers.Layer):
    """Multi-head attention module with dimension validation and metrics tracking."""
    # Implementation details in attention.py
```

### 2. SequenceContextManager

Implemented in `orchestrator/history.py`, this module manages a history of context tuples:

- Stores `(timestamp, memory_id, x_t, k_t, v_t, q_t, y_t)` tuples
- Provides methods for retrieving recent keys, values, and outputs
- Uses a deque with configurable max length to control memory usage

```python
class SequenceContextManager:
    """Manages a sequence of context tuples for attention-based processing."""
    # Implementation details in history.py
```

### 3. Neural Memory API Extensions

Enhanced in `synthians_trainer_server/http_server.py` to expose internal projections:

- Extended `UpdateMemoryResponse` to include `key_projection` and `value_projection`
- Extended `RetrieveResponse` to include `query_projection`
- Modified handlers to calculate projections and include them in responses

```python
class UpdateMemoryResponse(BaseModel):
    status: str
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    key_projection: Optional[List[float]] = None
    value_projection: Optional[List[float]] = None
```

### 4. Titans Variant Implementations

Implemented in `orchestrator/titans_variants.py`, providing three attention-based variants:

#### 4.1 Base Variant Class

```python
class TitansVariantBase:
    """Base class for all Titans architecture variants."""
    # Common functionality and interfaces for all variants
```

#### 4.2 Memory-Attended Computation (MAC)

```python
class MACVariant(TitansVariantBase):
    """Memory-Attended Computation (MAC) variant.
    
    Enhances memory retrieval by attending over historical memory outputs.
    Flow: q_t -> M -> y_t -> Attend(q_t, K_hist, Y_hist) -> attended_y_t
    """
    # Implementation in titans_variants.py
```

MAC enhances output by applying attention over historical memory outputs, providing a more contextually relevant retrieval.

#### 4.3 Memory-Attended Gates (MAG)

```python
class MAGVariant(TitansVariantBase):
    """Memory-Attended Gates (MAG) variant.
    
    Modifies gate values (alpha, theta, eta) for the neural memory update
    by attending over historical key projections.
    """
    # Implementation in titans_variants.py
```

MAG dynamically adjusts memory decay rates based on contextual relevance, allowing for adaptive forgetting.

#### 4.4 Memory-Augmented Learning (MAL)

```python
class MALVariant(TitansVariantBase):
    """Memory-Augmented Learning (MAL) variant.
    
    Modifies value projection for neural memory update by attending over
    historical value projections.
    """
    # Implementation in titans_variants.py
```

MAL enhances learning by augmenting value projections with historically relevant values, facilitating associative connections.

### 5. ContextCascadeEngine Integration

Extended in `orchestrator/context_cascade_engine.py` to activate and utilize the appropriate variant:

- Reads `TITANS_VARIANT` environment variable to determine active variant
- Initializes variant processor with appropriate configuration
- Extracts projections from API responses and populates the context manager
- Processes inputs through the active variant and handles variant-specific outputs

## Configuration

Titans variants can be configured via environment variables and configuration objects:

```python
# Select variant via environment variable
os.environ["TITANS_VARIANT"] = "MAC"  # Options: NONE, MAC, MAG, MAL

# Configure attention parameters
attention_config = {
    'num_heads': 4,
    'key_dim': 32,  # Per head dimension
    'dropout': 0.0,
    'use_layer_norm': True,
    'use_residual': True,
}
```

## Using the Variants

### MAC Variant

The MAC variant enhances memory retrieval by attending over historical memory outputs. It's particularly useful for tasks requiring coherent sequential recall, such as conversation modeling or narrative generation.

### MAG Variant

The MAG variant dynamically adjusts the memory decay rates (alpha, theta, eta) based on contextual relevance. This is beneficial for systems that need to selectively preserve or forget information based on changing contexts.

### MAL Variant

The MAL variant augments the learning process by modifying value projections with historically relevant values. This facilitates richer associations and connections between memories, enhancing conceptual learning.

## Current Limitations & Future Work

1. **MAG and MAL Timing**: The current implementation processes MAG and MAL variants after the `/update_memory` call, whereas ideally they should influence the call itself. Future work will refactor the processing order.

2. **Neural Memory Configuration**: Currently using hardcoded attention parameters. Future implementation could fetch these from a Neural Memory config endpoint.

3. **Integration Testing**: Comprehensive integration tests for each variant in different scenarios are needed.

4. **Documentation**: API reference and usage examples for each variant should be expanded.

## Conclusion

The Phase 4 implementation of Titans Architecture Variants significantly enhances Lucidia's cognitive architecture by introducing contextual attention mechanisms. These variants enable more adaptive, context-aware memory operations, aligning with the core principles of the cognitive architecture:

- "Memory is weighted, not just chronological" (QuickRecal)
- "Emotion shapes recall" (Emotional Gating)
- "Surprise signals significance" (Neural Memory Loss/Grad → QuickRecal Boost)
- "Ideas cluster and connect" (Attention-based context)
- "Presence emerges from adaptive memory" (Variant-specific adaptive mechanisms)

---

**Next Steps:**

1. Refactor processing flow for MAG and MAL to influence the `/update_memory` call
2. Implement integration tests for each variant
3. Enhance configuration options with dynamic parameter loading
4. Expand metrics tracking for variant-specific performance analysis
