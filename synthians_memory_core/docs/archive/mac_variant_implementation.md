# MAC Variant Implementation Guide

## Overview

The Memory-Attended Content (MAC) variant is a specialized architecture in the Lucidia Cognitive System that enhances retrieved memory embeddings using attention mechanisms over historical context. This document details the implementation, integration, and usage of the MAC variant within the refactored Context Cascade Engine.

## Architecture

The MAC variant follows this processing flow:

1. Retrieve raw embedding from Neural Memory → Get `y_t` (raw retrieval)
2. `q_t`, `y_t` + Historical context (K_hist, Y_hist) → Attend(q_t, K_hist, Y_hist) → `attended_y_t`
3. Return `attended_y_t` as enhanced memory representation

![MAC Architecture](../assets/diagrams/mac_architecture.png)

## Implementation Details

### Core Components

1. **TitansVariantBase**
   - Provides common infrastructure for all variants
   - Handles API client initialization and neural memory URL configuration
   - Manages sequence context and historical context tracking
   - Implements lazy loading for TensorFlow to prevent NumPy version conflicts

2. **MACVariant Class**
   - Implements the Memory-Attended Content logic
   - Initializes attention modules for output enhancement
   - Processes query embeddings and retrieved outputs through attention mechanisms
   - Applies attention over historical keys and values to enhance retrieved memory

3. **ContextCascadeEngine**
   - Orchestrates the variant selection and initialization
   - Routes memory operations through the appropriate variant
   - Invokes MAC processing *after* Neural Memory retrieval
   - Updates sequence history with the enhanced output

### Key Methods

#### MACVariant

```python
async def process_output(self, q_t: np.ndarray, y_t: np.ndarray) -> Dict[str, Any]:
    """Process output through MAC variant logic to enhance retrieved memory.
    
    Args:
        q_t: Query projection from Neural Memory
        y_t: Raw retrieved embedding from Neural Memory
    
    Returns:
        Dict containing attended output and metrics
    """
    try:
        # Get historical keys and values for attention calculation
        k_hist = self.sequence_context.get_recent_keys()
        y_hist = self.sequence_context.get_recent_outputs()
        
        if not k_hist or len(k_hist) == 0 or not y_hist or len(y_hist) == 0:
            logger.warning("No historical context available for MAC attention")
            return {"status": "error", "error": "No historical context available"}
        
        # Apply attention between query and historical keys
        attention_output = self.compute_attention(
            query=q_t,
            keys=k_hist,
            values=y_hist
        )
        
        # Combine retrieved embedding with attention output
        attended_y_t = self.combine_outputs(y_t, attention_output)
        
        return {
            "status": "success",
            "attended_y_t": attended_y_t,
            "metrics": {
                "attention_magnitude": float(np.linalg.norm(attention_output)),
                "combination_ratio": self.combination_ratio
            }
        }
    except Exception as e:
        logger.error(f"Error in MAC variant processing: {str(e)}")
        return {"status": "error", "error": str(e)}
```

#### Integration with ContextCascadeEngine

The refactored ContextCascadeEngine handles the MAC variant by applying its processing *after* Neural Memory retrieval, enhancing the retrieved content before returning it:

```python
async def _apply_variant_post_retrieval(self, step_context):
    """Apply variant-specific post-retrieval processing for MAC variant.
    
    Args:
        step_context: The current processing context
        
    Returns:
        Dict containing variant processing results
    """
    try:
        if self.active_variant_type == TitansVariantType.MAC:
            # Process MAC variant: Enhance retrieved embedding with attention
            mac_result = await self.variant_processor.process_output(
                step_context["q_t"], step_context["y_t"]
            )
            
            if "attended_y_t" in mac_result:
                # Replace retrieved embedding with attention-enhanced version
                step_context["y_t"] = mac_result["attended_y_t"]
                step_context["y_t_list"] = self._to_list(mac_result["attended_y_t"])
                logger.info("MAC variant produced attended output")
            else:
                logger.warning(f"MAC variant processing failed: {mac_result.get('error')}")
                
            return mac_result
            
        return {"status": "not_applicable"}
    except Exception as e:
        logger.error(f"Error in _apply_variant_post_retrieval: {str(e)}")
        return {"status": "error", "error": str(e)}
```

### Attention Mechanism

The MAC variant uses a multi-head attention mechanism to determine the relevance of historical memory embeddings to the current query:

```python
def compute_attention(self, query, keys, values):
    """Compute attention between query and historical keys/values.
    
    Args:
        query: Current query embedding (q_t)
        keys: Historical key embeddings (k_hist)
        values: Historical value or output embeddings (y_hist)
    
    Returns:
        Attention-weighted combination of values
    """
    tf = _get_tf()  # Lazy load TensorFlow
    
    # Ensure inputs are properly shaped for attention
    query = tf.expand_dims(tf.convert_to_tensor(query, dtype=tf.float32), axis=0)  # [1, dim]
    keys = tf.convert_to_tensor(keys, dtype=tf.float32)  # [seq_len, dim]
    keys = tf.expand_dims(keys, axis=0)  # [1, seq_len, dim]
    values = tf.convert_to_tensor(values, dtype=tf.float32)  # [seq_len, dim]
    values = tf.expand_dims(values, axis=0)  # [1, seq_len, dim]
    
    # Apply attention
    attention_output = self.attention_layer(
        query=query,  # [1, 1, dim]
        key=keys,     # [1, seq_len, dim]
        value=values  # [1, seq_len, dim]
    )
    
    # Remove batch dimension [1, 1, dim] -> [dim]
    return tf.squeeze(attention_output).numpy()
```

### Embedding Handling

The MAC variant includes robust handling for embedding dimension mismatches and malformed embeddings:

1. **Dimension Alignment**: Uses the `_align_vectors_for_comparison` method to handle mismatches between 384D and 768D embeddings
2. **Validation**: Validates embeddings to detect and handle NaN/Inf values
3. **Safe Conversion**: Properly handles different tensor types when converting between TensorFlow and NumPy

```python
def _align_vectors(self, vector_a, vector_b):
    """Align vectors to the same dimension for processing.
    
    Handles dimension mismatches by padding smaller vectors with zeros
    or truncating larger vectors.
    
    Args:
        vector_a: First vector
        vector_b: Second vector to align with
        
    Returns:
        Tuple of aligned vectors (a_aligned, b_aligned)
    """
    a_dim = vector_a.shape[-1]
    b_dim = vector_b.shape[-1]
    
    if a_dim == b_dim:
        return vector_a, vector_b
    
    if a_dim < b_dim:
        # Pad vector_a to match vector_b
        padding = np.zeros(b_dim - a_dim)
        a_aligned = np.concatenate([vector_a, padding])
        return a_aligned, vector_b
    else:
        # Truncate vector_a to match vector_b
        return vector_a[:b_dim], vector_b
```

## Testing the MAC Variant

To test the MAC variant, you can use the `lucidia_think_trace` tool with the appropriate environment variable:

```bash
# Run in Docker container
docker exec -e TITANS_VARIANT=MAC trainer-server python -m synthians_memory_core.tools.lucidia_think_trace --query "Testing MAC variant" --memcore-url "http://host.docker.internal:5010"
```

The output should show:

1. Successful Neural Memory retrieval
2. Proper enhancement of retrieved embedding via attention
3. Modified retrieved embedding in the response

## Activation

To activate the MAC variant, set the `TITANS_VARIANT` environment variable:

```bash
export TITANS_VARIANT=MAC  # For Linux/macOS
set TITANS_VARIANT=MAC      # For Windows CMD
```

In the Docker setup, you can specify this when starting the container:

```bash
docker run -e TITANS_VARIANT=MAC ...
```

## Common Issues and Troubleshooting

### Insufficient Historical Context

The MAC variant requires historical keys and values to calculate attention. If there isn't enough historical context, you might see warnings like:

```
No historical context available for MAC attention
```

Solution: Ensure that multiple inputs have been processed through the system before expecting MAC to enhance memory retrieval.

### TensorFlow Import Errors

If you encounter errors related to TensorFlow imports or NumPy version conflicts, verify that:

1. The lazy loading mechanism is correctly implemented
2. The fix_numpy.py script has run before any TensorFlow imports

## Conclusion

The MAC variant implementation enhances memory retrieval by using attention mechanisms to incorporate relevant historical context into retrieved embeddings. This approach provides several benefits:

1. Improved contextual relevance of retrieved memories
2. Enhanced continuity across sequential memory operations
3. Reduced retrieval errors by incorporating complementary information from past retrievals

By applying attention *after* the Neural Memory update and retrieval, MAC focuses on enhancing the usefulness of retrieved content rather than modifying how memories are stored.
