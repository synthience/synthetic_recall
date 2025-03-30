# MAG Variant Implementation Guide

## Overview

The Memory-Attended Gates (MAG) variant is a specialized architecture in the Lucidia Cognitive System that modifies the gate values used in the Neural Memory update process through attention mechanisms. This document details the implementation, integration, and usage of the MAG variant within the refactored Context Cascade Engine.

## Architecture

The MAG variant follows this processing flow:

1. `q_t` → Attend(q_t, K_hist, K_hist) → `attention_output`
2. Call Neural Memory's `/calculate_gates` endpoint with attention output
3. Update memory with calculated gates

![MAG Architecture](../assets/diagrams/mag_architecture.png)

## Implementation Details

### Core Components

1. **TitansVariantBase**
   - Provides common infrastructure for all variants
   - Handles API client initialization and neural memory URL configuration
   - Manages sequence context and historical context tracking
   - Implements lazy loading for TensorFlow to prevent NumPy version conflicts

2. **MAGVariant Class**
   - Implements the Memory-Attended Gates logic
   - Initializes attention modules for gate calculation
   - Processes input embeddings and queries through attention mechanisms
   - Calculates attention-based gate values to influence Neural Memory updates

3. **NeuralMemoryModule**
   - Provides gate calculation capabilities via dedicated projection layers
   - Processes attention outputs to compute optimal gate values
   - Applies external gate values during memory updates
   - Returns loss and gradient norm metrics for QuickRecal boosting

4. **ContextCascadeEngine**
   - Orchestrates the variant selection and initialization
   - Routes memory operations through the appropriate variant
   - Manages the flow of data between components
   - Ensures correct sequencing of operations to maximize variant effectiveness

### Key Methods

#### MAGVariant

```python
async def process_input(self, q_t: np.ndarray):
    """Process input through MAG variant logic to generate gate values.
    
    Args:
        q_t: Query projection from Neural Memory
    
    Returns:
        Dict containing gate values and metrics
    """
    try:
        # Get historical keys for attention calculation
        k_hist = self.sequence_context.get_recent_keys()
        
        if not k_hist or len(k_hist) == 0:
            logger.warning("No historical keys available for MAG attention")
            return {"status": "error", "error": "No historical context available"}
        
        # Use attention to determine gate values
        attention_output = self.compute_attention(q_t, k_hist)
        
        # Call Neural Memory's /calculate_gates endpoint
        response = await self.api_client.calculate_gates(
            attention_output=self._to_list(attention_output)
        )
        
        # Extract the calculated gates
        gates = response.get("gates", {})
        
        return {
            "status": "success",
            "gates": gates,
            "metrics": {
                "attention_magnitude": float(np.linalg.norm(attention_output))
            }
        }
    except Exception as e:
        logger.error(f"Error in MAG variant processing: {str(e)}")
        return {"status": "error", "error": str(e)}
```

#### Integration with ContextCascadeEngine

The refactored ContextCascadeEngine handles the MAG variant by applying its processing *before* the Neural Memory update, ensuring gates can properly influence the memory update process:

```python
async def _apply_variant_pre_update(self, step_context):
    """Apply variant-specific pre-update processing for MAG/MAL variants.
    
    For MAG: Calculates attention-based gates
    For MAL: Calculates modified value projection
    
    Args:
        step_context: The current processing context
        
    Returns:
        Dict containing variant processing results
    """
    try:
        if self.active_variant_type == TitansVariantType.MAG:
            # Process MAG variant
            mag_result = await self.variant_processor.process_input(step_context["q_t"])
            
            if mag_result.get("status") == "success":
                # Store gates for use in Neural Memory update
                step_context["gates"] = mag_result.get("gates", {})
                logger.info(f"MAG variant calculated gates: {step_context['gates']}")
            else:
                logger.warning(f"MAG variant processing failed: {mag_result.get('error')}")
            
            return mag_result
            
        elif self.active_variant_type == TitansVariantType.MAL:
            # Process MAL variant
            # ...
            
    except Exception as e:
        logger.error(f"Error in _apply_variant_pre_update: {str(e)}")
        return {"status": "error", "error": str(e)}
```

### Neural Memory Update

The Neural Memory update process now accepts and applies the gates calculated by the MAG variant:

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
        
        # Add MAG gates if available
        if "gates" in step_context and step_context["gates"]:
            update_params.update({
                "alpha_t": step_context["gates"].get("alpha_t"),
                "theta_t": step_context["gates"].get("theta_t"),
                "eta_t": step_context["gates"].get("eta_t")
            })
            
        # Add MAL modified value if available
        if "v_prime" in step_context and step_context["v_prime"] is not None:
            update_params.update({
                "key_projection": self._to_list(step_context["k_t"]),
                "value_projection": self._to_list(step_context["v_prime"])
            })
            
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

## Testing the MAG Variant

To test the MAG variant, you can use the `lucidia_think_trace` tool with the appropriate environment variable:

```bash
# Run in Docker container
docker exec -e TITANS_VARIANT=MAG trainer-server python -m synthians_memory_core.tools.lucidia_think_trace --query "Testing MAG variant" --memcore-url "http://host.docker.internal:5010"
```

The output should show:

1. Successful calculation of attention-based gates
2. Proper application of gates during Neural Memory update
3. Expected loss and gradient norm metrics

## Activation

To activate the MAG variant, set the `TITANS_VARIANT` environment variable:

```bash
export TITANS_VARIANT=MAG  # For Linux/macOS
set TITANS_VARIANT=MAG      # For Windows CMD
```

In the Docker setup, you can specify this when starting the container:

```bash
docker run -e TITANS_VARIANT=MAG ...
```

## Common Issues and Troubleshooting

### Insufficient Historical Context

The MAG variant requires historical keys to calculate attention-based gates. If there isn't enough historical context, you might see warnings like:

```
No historical keys available for MAG attention
```

Solution: Ensure that multiple inputs have been processed through the system before expecting MAG to influence the memory update process.

### TensorFlow Import Errors

If you encounter errors related to TensorFlow imports or NumPy version conflicts, verify that:

1. The lazy loading mechanism is correctly implemented
2. The fix_numpy.py script has run before any TensorFlow imports

## Conclusion

The refactored MAG variant implementation enables more effective memory-based cognitive processing by:

1. Using attention mechanisms to dynamically adjust Neural Memory update parameters
2. Properly sequencing operations to ensure gates are calculated before the memory update
3. Maintaining a clean and modular architecture with appropriate separation of concerns

This implementation follows the general Lucidia principle: "Memory shapes how we think, and thinking shapes how we remember." By allowing attention over past experiences to modulate how new experiences are stored, the MAG variant enhances the cognitive system's ability to prioritize and integrate information.
