# MAL Variant Implementation Guide

## Overview

The Memory-Attended Learning (MAL) variant is a specialized architecture in the Lucidia Cognitive System that modifies the value projections used in Neural Memory updates through attention mechanisms over historical context. This document details the implementation, integration, and usage of the MAL variant within the refactored Context Cascade Engine.

## Architecture

The MAL variant follows this processing flow:

1. Get projections from Neural Memory (k_t, v_t, q_t) without updating
2. `q_t`, `v_t` + Historical context (K_hist, V_hist) u2192 Attend(q_t, K_hist, V_hist) u2192 Modified value `v_prime`
3. Update Neural Memory using modified value projection `v_prime`

![MAL Architecture](../assets/diagrams/mal_architecture.png)

## Implementation Details

### Core Components

1. **TitansVariantBase**
   - Provides common infrastructure for all variants
   - Handles API client initialization and neural memory URL configuration
   - Manages sequence context and historical context tracking
   - Implements lazy loading for TensorFlow to prevent NumPy version conflicts

2. **MALVariant Class**
   - Implements the Memory-Attended Learning logic
   - Initializes attention modules for value projection modification
   - Processes query and value projections through attention mechanisms
   - Creates enhanced value representations for memory storage

3. **NeuralMemoryModule**
   - Processes input embeddings to calculate key, value, and query projections
   - Supports updates with externally provided value projections
   - Performs memory updates with the modified value projection

4. **ContextCascadeEngine**
   - Orchestrates the variant selection and initialization
   - Routes memory operations through the appropriate variant
   - Invokes MAL processing *before* Neural Memory update
   - Passes the modified value projection to the Neural Memory update

### Key Methods

#### MALVariant

```python
async def calculate_v_prime(self, q_t: np.ndarray, v_t: np.ndarray) -> Dict[str, Any]:
    """Calculate modified value projection using attention over historical values.
    
    Args:
        q_t: Query projection from Neural Memory
        v_t: Original value projection from Neural Memory
    
    Returns:
        Dict containing modified value projection and metrics
    """
    try:
        # Get historical keys and values for attention calculation
        k_hist, v_hist = self.sequence_context.get_recent_kv_pairs()
        
        if not k_hist or len(k_hist) == 0 or not v_hist or len(v_hist) == 0:
            logger.warning("No historical context available for MAL attention")
            return {"status": "error", "error": "No historical context available"}
        
        # Validate inputs and handle dimension mismatches
        q_t = self._validate_embedding(q_t)
        v_t = self._validate_embedding(v_t)
        
        # Apply attention between query and historical keys/values
        tf = _get_tf()  # Lazy load TensorFlow
        
        # Ensure inputs are properly shaped for attention
        query = tf.expand_dims(tf.convert_to_tensor(q_t, dtype=tf.float32), axis=0)  # [1, dim]
        keys = tf.convert_to_tensor(k_hist, dtype=tf.float32)  # [seq_len, dim]
        keys = tf.expand_dims(keys, axis=0)  # [1, seq_len, dim]
        values = tf.convert_to_tensor(v_hist, dtype=tf.float32)  # [seq_len, dim]
        values = tf.expand_dims(values, axis=0)  # [1, seq_len, dim]
        
        # Apply attention to generate attended values
        attended_v = self.attention_module(
            query=query,  # [1, 1, dim]
            key=keys,     # [1, seq_len, dim]
            value=values  # [1, seq_len, dim]
        )
        
        # Remove batch dimension [1, 1, dim] -> [dim]
        attended_v = tf.squeeze(attended_v).numpy()
        
        # Combine original and attended values to create v_prime
        v_prime = self.combine_values(v_t, attended_v)
        
        return {
            "status": "success",
            "v_prime": v_prime,
            "metrics": {
                "attention_magnitude": float(np.linalg.norm(attended_v)),
                "combination_ratio": self.combination_ratio
            }
        }
    except Exception as e:
        logger.error(f"Error in MAL variant processing: {str(e)}")
        return {"status": "error", "error": str(e)}
```

#### Integration with ContextCascadeEngine

The refactored ContextCascadeEngine handles the MAL variant by applying its processing *before* the Neural Memory update, modifying how memories are stored:

```python
async def _apply_variant_pre_update(self, step_context):
    """Apply variant-specific pre-update processing for MAG/MAL variants.
    
    Args:
        step_context: The current processing context
        
    Returns:
        Dict containing variant processing results
    """
    try:
        # ... [MAG variant handling code] ...
        
        elif self.active_variant_type == TitansVariantType.MAL:
            # Process MAL variant: Calculate modified value projection
            mal_result = await self.variant_processor.calculate_v_prime(
                step_context["q_t"], step_context["v_t"]
            )
            
            if "v_prime" in mal_result:
                # Store modified value projection for use in Neural Memory update
                step_context["v_prime"] = mal_result["v_prime"]
                logger.info("MAL variant calculated modified value projection")
            else:
                logger.warning(f"MAL variant processing failed: {mal_result.get('error')}")
            
            return mal_result
            
        return {"status": "not_applicable"}
    except Exception as e:
        logger.error(f"Error in _apply_variant_pre_update: {str(e)}")
        return {"status": "error", "error": str(e)}
```

### Neural Memory Update

The Neural Memory update process accepts and applies the modified value projection calculated by the MAL variant:

```python
async def _update_neural_memory(self, step_context):
    """Update Neural Memory with appropriate modifications based on active variant.
    
    Args:
        step_context: The current processing context
        
    Returns:
        Dict containing update response
    """
    try:
        # Prepare update parameters
        update_params = {"input_embedding": self._to_list(step_context["x_t"])}
        
        # ... [MAG variant handling code] ...
        
        # Add MAL variant modified value if available
        if "v_prime" in step_context and step_context["v_prime"] is not None:
            update_params.update({
                "key_projection": self._to_list(step_context["k_t"]),
                "value_projection": self._to_list(step_context["v_prime"])
            })
            logger.info("Adding MAL modified value projection to Neural Memory update")
        
        # Call Neural Memory update endpoint
        update_resp = await self.neural_memory_client.update_memory(**update_params)
        
        # Update step context with response data
        step_context["loss"] = update_resp.get("loss")
        step_context["grad_norm"] = update_resp.get("grad_norm")
        
        return update_resp
        
    except Exception as e:
        logger.error(f"Error updating Neural Memory: {str(e)}")
        return {"status": "error", "error": str(e)}
```

### Value Combination

The MAL variant combines the original value projection with the attention-based value to create the enhanced `v_prime`:

```python
def combine_values(self, v_t, attended_v):
    """Combine original value projection with attention-based value.
    
    Args:
        v_t: Original value projection
        attended_v: Attention-based value from historical context
    
    Returns:
        Combined value projection (v_prime)
    """
    # Ensure dimensions match
    v_t, attended_v = self._align_vectors(v_t, attended_v)
    
    # Combine using configured ratio
    v_prime = (1 - self.combination_ratio) * v_t + self.combination_ratio * attended_v
    
    return v_prime
```

### Embedding Handling

The MAL variant includes robust handling for embedding dimension mismatches and malformed embeddings:

1. **Dimension Alignment**: Uses the `_align_vectors` method to handle mismatches between 384D and 768D embeddings
2. **Validation**: Uses the `_validate_embedding` method to detect and handle NaN/Inf values
3. **Safe Conversion**: Uses proper tensor conversion with error handling

```python
def _validate_embedding(self, embedding):
    """Validate embedding and replace invalid values with zeros.
    
    Args:
        embedding: Input embedding to validate
    
    Returns:
        Validated embedding with NaN/Inf replaced by zeros
    """
    try:
        # Convert to numpy if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning(f"Found NaN/Inf in embedding, replacing with zeros")
            # Replace NaN/Inf with zeros
            embedding = np.where(np.isnan(embedding) | np.isinf(embedding), 0.0, embedding)
        
        return embedding
    except Exception as e:
        logger.error(f"Error validating embedding: {str(e)}")
        # Return zero vector as fallback
        return np.zeros(768, dtype=np.float32)
```

## Testing the MAL Variant

To test the MAL variant, you can use the `lucidia_think_trace` tool with the appropriate environment variable:

```bash
# Run in Docker container
docker exec -e TITANS_VARIANT=MAL trainer-server python -m synthians_memory_core.tools.lucidia_think_trace --query "Testing MAL variant" --memcore-url "http://host.docker.internal:5010"
```

The output should show:

1. Successful calculation of modified value projection
2. Proper application of modified value during Neural Memory update
3. Expected loss and gradient norm metrics

## Activation

To activate the MAL variant, set the `TITANS_VARIANT` environment variable:

```bash
export TITANS_VARIANT=MAL  # For Linux/macOS
set TITANS_VARIANT=MAL      # For Windows CMD
```

In the Docker setup, you can specify this when starting the container:

```bash
docker run -e TITANS_VARIANT=MAL ...
```

## Common Issues and Troubleshooting

### Insufficient Historical Context

The MAL variant requires historical keys and values to calculate the modified value projection. If there isn't enough historical context, you might see warnings like:

```
No historical context available for MAL attention
```

Solution: Ensure that multiple inputs have been processed through the system before expecting MAL to influence the memory update process.

### TensorFlow Import Errors

If you encounter errors related to TensorFlow imports or NumPy version conflicts, verify that:

1. The lazy loading mechanism is correctly implemented
2. The fix_numpy.py script has run before any TensorFlow imports

### Dimension Mismatch Errors

If you encounter dimension mismatch errors, verify that:

1. The `_align_vectors` method is properly handling dimension differences
2. All inputs are properly validated before processing
3. TensorFlow operations are properly handling tensor shapes

## Conclusion

The MAL variant implementation enhances memory storage by modifying how value projections are calculated before Neural Memory updates. This approach provides several benefits:

1. Improved contextual coherence in stored memories
2. Enhanced learning by incorporating relevant historical values
3. More efficient memory representation through context-aware value projections

By applying attention to modify the value projection *before* the Neural Memory update, MAL influences how memories are stored rather than how they're retrieved, complementing the approaches of the MAC and MAG variants.
