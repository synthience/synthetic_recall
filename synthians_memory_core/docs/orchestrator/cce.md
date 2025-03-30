# Context Cascade Engine (CCE)

**Author:** Lucidia Core Team  
**Date:** 2025-03-30  
**Status:** Implemented

## Overview

The Context Cascade Engine (CCE) is the central orchestrator of the Synthians Cognitive Architecture, implementing the refactored cognitive flow between the Memory Core and Neural Memory services. It manages the sequence of operations that constitute the cognitive cycle, including variant-specific steps for MAC, MAG, and MAL implementations.

## Core Functionality

### Cognitive Cycle

The CCE implements the following sequence for processing a new input (`content`, `embedding`, `metadata`):

1. **Store Memory:** CCE sends input to Memory Core (`/process_memory`). Memory Core stores it, generates metadata, calculates initial QuickRecal, and returns the validated embedding (`x_t`), `memory_id`, and `quickrecal_score`.

2. **Get Projections:** CCE sends `x_t` to Neural Memory Server (`/get_projections`). NM Server returns Key (`k_t`), Value (`v_t`), and Query (`q_t`) projections *without* updating its internal weights.

3. **Variant Pre-Update (MAG/MAL):**
   - If **MAG** is active: CCE calculates attention output (using `q_t`, historical keys `K_hist`) and calls NM Server (`/calculate_gates`) to get external gate values (`alpha_t`, `theta_t`, `eta_t`).
   - If **MAL** is active: CCE calculates attention output (using `q_t`, historical keys `K_hist`, historical values `V_hist`), combines it with `v_t` to create a modified value projection (`v'_t`).
   - If **NONE** or **MAC**: This step is skipped.

4. **Update Neural Memory:** CCE calls NM Server (`/update_memory`) providing:
   - Base: `input_embedding` (`x_t`).
   - MAG: External gate values (`external_alpha_gate`, etc.).
   - MAL: Explicit projections (`key_projection=k_t`, `value_projection=v'_t`).
   - NM Server performs the test-time update using the provided parameters and returns `loss` and `grad_norm`.

5. **Apply QuickRecal Boost:** CCE calculates a boost value based on `loss`/`grad_norm`. It calls Memory Core (`/api/memories/update_quickrecal_score`) to apply this boost to the original memory's score.

6. **Retrieve from Neural Memory:** CCE sends `x_t` to NM Server (`/retrieve`). NM Server calculates the query projection `q_t` (may differ slightly from step 2 if weights changed) and retrieves the associated raw embedding (`y_t_raw`) using its internal memory `M(q_t)`. It returns `y_t_raw` and the `query_projection` used.

7. **Variant Post-Retrieval (MAC):**
   - If **MAC** is active: CCE calculates attention output (using `q_t` from step 6, historical keys `K_hist`, historical outputs `Y_hist`), combines it with `y_t_raw` to create an attended output (`y_t_final`).
   - Otherwise, `y_t_final` is set to `y_t_raw`.

8. **Update History:** CCE adds the full context tuple `(timestamp, memory_id, x_t, k_t, v_t, q_t, y_t_final)` to the `SequenceContextManager`.

9. **Finalize:** CCE constructs and returns a response containing the `memory_id`, processing status, surprise metrics, retrieval results (`y_t_final`), QuickRecal feedback status, and variant metrics.

### SequenceContextManager

The `SequenceContextManager` maintains a history of recent cognitive operations for use in attention mechanisms:

- It stores a deque of tuples `(timestamp, memory_id, x, k, v, q, y_final)` representing the history of processed inputs and their projections/outputs.
- It provides methods for retrieving historical keys, values, queries, and outputs needed for attention calculations.
- It manages the history size to prevent memory leaks while maintaining sufficient context for attention.

### Variant Support

The CCE dynamically configures itself based on the selected Titans Architecture Variant:

- **MAC (Memory-Attention-Combined)**: Enhances Neural Memory output using attention over historical outputs.
- **MAG (Memory-Attention-Gated)**: Modulates memory update gates using attention over historical keys.
- **MAL (Memory-Attention-Layer)**: Modifies the value projection using attention over historical keys and values.

The variant can be selected via the `TITANS_VARIANT` environment variable.

## TensorFlow Integration

The CCE implements lazy loading of TensorFlow to avoid NumPy version conflicts:

```python
def _get_tf():
    """Lazily import TensorFlow to avoid early NumPy import."""
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf
```

This approach ensures that `fix_numpy.py` can execute before TensorFlow tries to import NumPy.

## Surprise Feedback Loop

A key responsibility of the CCE is implementing the surprise feedback loop:

1. The Neural Memory Server's `/update_memory` endpoint returns `loss` and `grad_norm` metrics.
2. The CCE calculates a `boost` value based on these metrics (higher surprise → higher boost).
3. The CCE calls the Memory Core's `/api/memories/update_quickrecal_score` endpoint with the `memory_id` and `delta=boost`.
4. The Memory Core updates the memory's QuickRecal score and adds surprise metadata.

This mechanism reinforces memories that contained surprising or hard-to-predict information, implementing the principle that **"Surprise signals significance."**

## Configuration Options

- `memory_core_url`: URL of the Memory Core API
- `neural_memory_url`: URL of the Neural Memory Server API
- `titans_variant`: Selected variant ("MAC", "MAG", "MAL", or "NONE")
- `history_size`: Maximum number of entries in the sequence history
- `attention_temperature`: Scaling factor for attention softmax
- `surprise_boost_factor`: Scaling factor for converting surprise metrics to QuickRecal boosts
