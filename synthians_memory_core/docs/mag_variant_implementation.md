# MAG Variant Implementation Guide

## Overview

The Memory-Attended Gates (MAG) variant is a specialized architecture in the Lucidia Cognitive System that modifies the gate values used in the Neural Memory update process through attention mechanisms. This document details the implementation, integration, and usage of the MAG variant.

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

2. **MAGVariant Class**
   - Implements the Memory-Attended Gates logic
   - Initializes attention modules for gate calculation
   - Processes input embeddings and queries through attention mechanisms

3. **NeuralMemoryModule**
   - Provides gate calculation capabilities via dedicated projection layers
   - Processes attention outputs to compute optimal gate values
   - Applies external gate values during memory updates

4. **ContextCascadeEngine**
   - Orchestrates the variant selection and initialization
   - Routes memory operations through the appropriate variant
   - Manages the flow of data between components

### Key Methods

#### MAGVariant

```python
def process_input(self, memory_id: str, x_t: np.ndarray, k_t: np.ndarray, v_t: np.ndarray, q_t: np.ndarray, y_t: np.ndarray):
    """Process input through MAG variant logic.
    
    Args:
        memory_id: ID of the current memory being processed
        x_t: Original input embedding
        k_t: Key projection
        v_t: Value projection
        q_t: Query projection
        y_t: Retrieved embedding from Neural Memory
    
    Returns:
        Dict containing gate values and metrics
    """
    try:
        # Use attention to determine gate values
        attention_output = self.compute_attention(q_t, k_t)
        
        # Call Neural Memory's /calculate_gates endpoint
        response = self.api_client.calculate_gates(
            attention_output=attention_output.numpy().tolist()
        )
        
        # Extract the calculated gates
        gates = response.get("gates", {})
        
        return {
            "memory_id": memory_id,
            "gates": gates,
            "metrics": {...}
        }
    except Exception as e:
        logger.error(f"Error in MAG variant processing: {str(e)}")
        return {"error": str(e)}
```

#### NeuralMemoryModule

```python
def calculate_gates(self, attention_output):
    """Calculate gate values from attention output.
    
    Args:
        attention_output: Output from attention mechanism
        
    Returns:
        Dict containing alpha_t, theta_t, and eta_t values
    """
    # Convert to tensor if needed
    if not isinstance(attention_output, tf.Tensor):
        attention_output = tf.convert_to_tensor(attention_output, dtype=tf.float32)
    
    # Apply gate projections
    alpha_t = self.attention_to_alpha(attention_output)
    theta_t = self.attention_to_theta(attention_output)
    eta_t = self.attention_to_eta(attention_output)
    
    # Apply activation functions
    alpha_t = tf.nn.sigmoid(alpha_t) * self.alpha_scale
    theta_t = tf.nn.sigmoid(theta_t) * self.theta_scale
    eta_t = tf.nn.sigmoid(eta_t) * self.eta_scale
    
    # Return as dict
    return {
        "alpha_t": float(alpha_t.numpy()),
        "theta_t": float(theta_t.numpy()),
        "eta_t": float(eta_t.numpy())
    }
```

### HTTP Server Integration

The HTTP server provides essential endpoints for MAG variant operation:

1. `/calculate_gates`: Calculates gate values from attention output
2. `/update_memory`: Accepts external gate values for memory updates
3. `/config`: Reports current configuration and capabilities

## Dynamic Capability Detection

The system dynamically determines which capabilities are supported using runtime method signature inspection:

```python
# Check if update_step supports external gates and projections using inspect
update_step_sig = inspect.signature(nm.update_step)
supports_external_gates = any(param in update_step_sig.parameters 
                           for param in ["external_alpha_t", "external_theta_t", "external_eta_t"])
supports_external_projections = any(param in update_step_sig.parameters 
                                for param in ["external_k_t", "external_v_t"])
```

This approach ensures that the system accurately reports its capabilities regardless of the current implementation details.

## Usage

### Activating the MAG Variant

To activate the MAG variant, set the `TITANS_VARIANT` environment variable:

```bash
export TITANS_VARIANT=MAG  # For Linux/macOS
set TITANS_VARIANT=MAG      # For Windows CMD
$env:TITANS_VARIANT="MAG"   # For Windows PowerShell
```

Or when running a container:

```bash
docker exec -e TITANS_VARIANT=MAG trainer-server [command]
```

### Testing the MAG Variant

Use the `lucidia_think_trace` tool to test the MAG variant:

```bash
docker exec -e TITANS_VARIANT=MAG trainer-server python -m synthians_memory_core.tools.lucidia_think_trace --query "Your test query here" --memcore-url "http://host.docker.internal:5010"
```

## Performance Considerations

### Advantages

- **Adaptive Memory Update**: Gate values adapt based on input context
- **Improved Relevance**: Attention mechanisms help focus on relevant historical context
- **Efficient Learning**: Can reduce learning rate for already-familiar inputs

### Limitations

- **Computational Overhead**: Adds additional computation for attention and gate calculation
- **Requires History**: Performance depends on having sufficient historical context
- **Configuration Sensitivity**: May require tuning of attention parameters

## Troubleshooting

### Common Issues

1. **AttributeError for debug_logging**
   - Caused by references to undefined attributes in ContextCascadeEngine
   - Solution: Use logger.debug directly instead of conditional checks

2. **API Client Communication Failures**
   - Caused by missing neural_memory_url in variant processor
   - Solution: Ensure set_neural_memory_url is called during initialization

3. **Incorrect Capabilities Reporting**
   - Caused by hardcoded capability flags instead of dynamic detection
   - Solution: Use inspect.signature to dynamically determine capabilities

## Future Improvements

1. **Performance Optimization**: Reduce computational overhead of attention calculation
2. **Fine-Tuning Interface**: Add tools for tuning MAG variant parameters
3. **Visualization Tools**: Create visualizations of gate value changes over time
4. **Hybrid Approaches**: Explore combinations of MAG with other variants like MAC and MAL
