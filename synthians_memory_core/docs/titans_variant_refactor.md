# Titans Variant Refactoring: Fixing MAG/MAL Timing

**Author:** Lucidia Core Team
**Date:** 2025-03-28
**Status:** Completed

## Problem Statement

The current implementation of the Context Cascade Engine (CCE) has a timing issue that prevents the MAG and MAL variants from properly influencing the Neural Memory update process. Specifically, the variant processing occurs *after* the `/update_memory` call they are intended to influence, rendering their modifications ineffective.

> *"The cascade must flow in the right order."*

## Previous Flow

The previous `ContextCascadeEngine.process_new_input` method followed this sequence:

1. Store input in Memory Core → Get `x_t`, `memory_id`
2. Update Neural Memory with `x_t` → Get `k_t`, `v_t`, `q_t`, `loss`, `grad_norm`
3. Update QuickRecal score with `loss`, `grad_norm`
4. Retrieve from Neural Memory → Get `y_t` (raw retrieval)
5. Process variant (MAC/MAG/MAL):
   - For MAC: Override `y_t` with attention-augmented `attended_y_t`
   - For MAG/MAL: Calculate outputs, but **too late** to affect `/update_memory`
6. Add context to history
7. Return final results

This sequence was problematic because:

- MAG is designed to modify the gate values (`alpha_t`, `theta_t`, `eta_t`) that control the Neural Memory update
- MAL is designed to modify the value projection (`v_prime_t`) before it's used in the Neural Memory update
- Both modifications need to happen *before* step 2 (the `/update_memory` call)

## Implemented Refactored Flow

The solution has been implemented by reorganizing the processing flow so that variant-specific modifications occur before the `/update_memory` call:

1. Store input in Memory Core → Get `x_t`, `memory_id`
2. **Get projections from Neural Memory** → Get `k_t`, `v_t`, `q_t` (without updating)
3. **Apply variant-specific preprocessing**:
   - If MAG: Calculate attention-based gates (`alpha_t`, `theta_t`, `eta_t`)
   - If MAL: Calculate modified value (`v_prime_t`)
4. **Update Neural Memory** with appropriate modifications:
   - If MAG: Include gate values in request
   - If MAL: Use modified value projection
   - Get `loss`, `grad_norm` from response
5. Update QuickRecal score
6. Retrieve from Neural Memory → Get `y_t` (raw retrieval)
7. **Apply post-retrieval variant processing**:
   - If MAC: Override `y_t` with attention-augmented `attended_y_t`
8. Add full context to history
9. Return final results

## Implementation Details

### 1. Modular Design

The refactored `ContextCascadeEngine.process_new_input` method now uses a series of specialized helper methods for better readability and maintainability:

```python
async def process_new_input(self, content: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, Any]] = None, intent_id: Optional[str] = None):
    """Orchestrates the refactored cognitive cascade for a single input."""
    async with self.processing_lock:
        # 1. Setup Intent & Metadata
        intent_id, user_emotion = self._setup_intent_and_metadata(intent_id, metadata)
        
        # Initialize context dict for this step
        step_context = {...}  # Contains all processing state
        
        # 2. Store Memory
        store_resp = await self._store_memory(content, embedding, metadata)
        
        # 3. Get Projections (without updating memory)
        proj_resp = await self._get_projections_from_nm(step_context["x_t"])
        
        # 4. Variant Pre-Update Logic (MAG/MAL)
        if self.variant_processor and self.active_variant_type in [TitansVariantType.MAG, TitansVariantType.MAL]:
            variant_pre_result = await self._apply_variant_pre_update(step_context)
        
        # 5. Update Neural Memory
        update_resp = await self._update_neural_memory(step_context)
        
        # 6. Apply QuickRecal Boost
        feedback_resp = await self._apply_quickrecal_boost(step_context, quickrecal_initial)
        
        # 7. Retrieve from Neural Memory
        retrieve_resp = await self._retrieve_from_neural_memory(step_context["x_t"])
        
        # 8. Apply MAC Post-Retrieval Logic
        if self.variant_processor and self.active_variant_type == TitansVariantType.MAC:
            mac_resp = await self._apply_variant_post_retrieval(step_context)
        
        # 9. Update History
        await self._update_history(step_context)
        
        # 10. Finalize Response
        response = self._finalize_response({}, step_context, update_resp, retrieve_resp, feedback_resp)
        
        return response
```

### 2. Robust Error Handling

Each helper method now includes comprehensive error handling and validation:

- Embedding validation to handle NaN/Inf values
- Type checking and conversion between numpy arrays and lists
- Graceful handling of dimension mismatches
- Proper logging of error conditions

### 3. TensorFlow Lazy Loading

To prevent NumPy version conflicts, TensorFlow is now lazy-loaded only when needed:

```python
# Global variable for TensorFlow instance
_tf = None

def _get_tf():
    """Lazy-load TensorFlow only when needed."""
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
            logger.info("TensorFlow loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow: {e}")
    return _tf
```

### 4. MAL Variant Implementation

The MAL variant now includes a `calculate_v_prime` method that modifies the value projection using attention over historical values:

```python
async def calculate_v_prime(self, q_t: np.ndarray, v_t: np.ndarray):
    """Calculate modified value projection using attention over historical values."""
    # Get historical keys and values
    k_hist, v_hist = self.sequence_context.get_recent_kv_pairs()
    
    # Apply attention to generate attended values
    attended_v = self.attention_module(
        query=q_t,
        key=k_hist,
        value=v_hist
    )
    
    # Combine original and attended values
    v_prime = self.combine_values(v_t, attended_v)
    
    return {"v_prime": v_prime, "metrics": {...}}
```

## Testing Results

All four Titans variants (NONE, MAC, MAG, MAL) have been tested and confirmed to function correctly:

- **NONE**: Base functionality works with default processing
- **MAC**: Successfully modifies retrieved memory output with attention
- **MAG**: Properly influences Neural Memory update with calculated gate values
- **MAL**: Correctly modifies value projections before Neural Memory update

## Conclusion

The refactored implementation successfully addresses the timing issues with the MAG and MAL variants while improving code modularity, readability, and maintainability. The additional parameter flexibility provides a solid foundation for further extensions and optimizations of the cognitive architecture.

---

**Related Documentation:**
- [MAG Variant Implementation](mag_variant_implementation.md)
- [Architecture Overview](architecture_overview.md)
- [Embedding Handling](embedding_handling.md)
- [NumPy/TensorFlow Compatibility](numpy_tensorflow_compatibility.md)
