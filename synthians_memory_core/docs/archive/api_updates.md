# API Updates for Phase 4

**Author:** Lucidia Core Team
**Date:** 2025-03-28
**Status:** Planned

## Overview

This document outlines the necessary API changes to complete Phase 4 of the Lucidia cognitive system. These changes enable the proper functioning of the Titans Architecture Variants (MAC, MAG, MAL) by exposing neural projections, supporting variant-specific parameters, and maintaining backward compatibility.

> *"The interface evolves to support the growing cognitive capabilities."*

## Neural Memory API Changes

### 1. New Endpoint: `/get_projections`

A new endpoint to calculate key, value, and query projections for an input embedding without updating memory.

#### Request Model

```python
class GetProjectionsRequest(BaseModel):
    input_embedding: List[float]
```

#### Response Model

```python
class GetProjectionsResponse(BaseModel):
    status: str
    key_projection: Optional[List[float]] = None
    value_projection: Optional[List[float]] = None
    query_projection: Optional[List[float]] = None
```

#### Implementation

```python
@app.post("/get_projections")
async def get_projections(request: GetProjectionsRequest) -> GetProjectionsResponse:
    """Calculate key, value, and query projections for an input embedding without updating memory."""
    try:
        embedding = request.input_embedding
        embedding_np = np.array(embedding)
        
        # Get projections from neural memory module
        k_t, v_t, q_t = neural_memory_module.get_projections(embedding_np)
        
        return GetProjectionsResponse(
            status="success",
            key_projection=k_t.tolist(),
            value_projection=v_t.tolist(),
            query_projection=q_t.tolist()
        )
    except Exception as e:
        logger.error(f"Error in get_projections: {e}")
        return GetProjectionsResponse(status="error")
```

### 2. New Endpoint: `/calculate_gates`

A new endpoint to calculate gate values based on attention output (for MAG variant).

#### Request Model

```python
class CalculateGatesRequest(BaseModel):
    attention_output: List[float]
```

#### Response Model

```python
class CalculateGatesResponse(BaseModel):
    status: str
    alpha_t: Optional[float] = None
    theta_t: Optional[float] = None
    eta_t: Optional[float] = None
```

#### Implementation

```python
@app.post("/calculate_gates")
async def calculate_gates(request: CalculateGatesRequest) -> CalculateGatesResponse:
    """Calculate gate values based on attention output."""
    try:
        attention_output = np.array(request.attention_output)
        alpha_t, theta_t, eta_t = neural_memory_module.calculate_gates(attention_output)
        
        return CalculateGatesResponse(
            status="success",
            alpha_t=float(alpha_t),
            theta_t=float(theta_t),
            eta_t=float(eta_t)
        )
    except Exception as e:
        logger.error(f"Error in calculate_gates: {e}")
        return CalculateGatesResponse(status="error")
```

### 3. Enhanced `/update_memory` Endpoint

Expand the existing endpoint to accept MAG gates and MAL modified projections.

#### Updated Request Model

```python
class UpdateMemoryRequest(BaseModel):
    input_embedding: List[float]
    # MAG parameters (optional)
    alpha_t: Optional[float] = None
    theta_t: Optional[float] = None
    eta_t: Optional[float] = None
    # MAL parameters (optional)
    key_projection: Optional[List[float]] = None
    value_projection: Optional[List[float]] = None
```

#### Updated Response Model

```python
class UpdateMemoryResponse(BaseModel):
    status: str
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    key_projection: Optional[List[float]] = None
    value_projection: Optional[List[float]] = None
```

#### Implementation Changes

```python
@app.post("/update_memory")
async def update_memory(request: UpdateMemoryRequest) -> UpdateMemoryResponse:
    """Update neural memory with input embedding and optional custom parameters."""
    try:
        embedding = request.input_embedding
        embedding_np = np.array(embedding)
        
        # Handle MAG variant (external gates)
        external_gates = None
        if request.alpha_t is not None and request.theta_t is not None and request.eta_t is not None:
            external_gates = {
                "alpha_t": request.alpha_t,
                "theta_t": request.theta_t,
                "eta_t": request.eta_t
            }
        
        # Handle MAL variant (external key/value projections)
        key_projection = None
        value_projection = None
        if request.key_projection is not None:
            key_projection = np.array(request.key_projection)
        if request.value_projection is not None:
            value_projection = np.array(request.value_projection)
        
        # Update memory with appropriate parameters
        result = neural_memory_module.update_step(
            embedding_np,
            external_gates=external_gates,
            key_projection=key_projection,
            value_projection=value_projection
        )
        
        # Get projections for response
        k_t, v_t, _ = neural_memory_module.get_projections(embedding_np)
        
        return UpdateMemoryResponse(
            status="success",
            loss=float(result["loss"]),
            grad_norm=float(result["grad_norm"]),
            key_projection=k_t.tolist(),
            value_projection=v_t.tolist()
        )
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        return UpdateMemoryResponse(status="error")
```

### 4. Enhanced `/retrieve` Endpoint

Update the existing endpoint to include query projection in the response.

#### Response Model Update

```python
class RetrieveResponse(BaseModel):
    status: str
    retrieved_embedding: Optional[List[float]] = None
    query_projection: Optional[List[float]] = None  # New field
```

#### Implementation Changes

```python
@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve from neural memory using an input embedding."""
    try:
        embedding = request.input_embedding
        embedding_np = np.array(embedding)
        
        # Process query through neural memory module
        retrieved, q_t = neural_memory_module.retrieve(embedding_np, return_query=True)
        
        return RetrieveResponse(
            status="success",
            retrieved_embedding=retrieved.tolist(),
            query_projection=q_t.tolist()  # Include query projection
        )
    except Exception as e:
        logger.error(f"Error retrieving from memory: {e}")
        return RetrieveResponse(status="error")
```

### 5. New Endpoint: `/config`

A new endpoint to retrieve configuration parameters, particularly for attention mechanism setup.

#### Response Model

```python
class ConfigResponse(BaseModel):
    status: str
    key_dim: int
    value_dim: int
    query_dim: int
    recommended_attention_heads: int
    momentum_settings: Dict[str, float]
```

#### Implementation

```python
@app.get("/config")
async def get_config() -> ConfigResponse:
    """Get Neural Memory configuration parameters."""
    try:
        return ConfigResponse(
            status="success",
            key_dim=neural_memory_module.key_dim,
            value_dim=neural_memory_module.value_dim,
            query_dim=neural_memory_module.query_dim,
            recommended_attention_heads=4,  # Default recommendation
            momentum_settings={
                "alpha": neural_memory_module.alpha,
                "theta": neural_memory_module.theta,
                "eta": neural_memory_module.eta
            }
        )
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return ConfigResponse(status="error")
```

## Neural Memory Module Changes

### 1. New Projection Helper Method

```python
def get_projections(self, x_t):
    """Calculate key, value, and query projections for an input embedding.
    
    Args:
        x_t: Input embedding
        
    Returns:
        Tuple of (key_projection, value_projection, query_projection)
    """
    # Ensure input is properly shaped for TensorFlow
    x_t = self._prepare_input(x_t)
    
    # Calculate projections
    k_t = self.key_projection(x_t)
    v_t = self.value_projection(x_t)
    q_t = self.query_projection(x_t)
    
    return k_t.numpy(), v_t.numpy(), q_t.numpy()
```

### 2. Enhanced Update Step Method

```python
def update_step(self, x_t, external_gates=None, key_projection=None, value_projection=None):
    """Update memory with input embedding and optional external parameters.
    
    Args:
        x_t: Input embedding
        external_gates: Dict with keys 'alpha_t', 'theta_t', 'eta_t' for MAG variant
        key_projection: Optional external key projection for MAL variant
        value_projection: Optional external value projection for MAL variant
    
    Returns:
        Dict with loss and grad_norm
    """
    # Use provided projections if available, otherwise calculate them
    if key_projection is None or value_projection is None:
        k_t, v_t, q_t = self.get_projections(x_t)
        
        if key_projection is None:
            key_projection = k_t
        if value_projection is None:
            value_projection = v_t
    else:
        # Ensure query projection for metrics
        _, _, q_t = self.get_projections(x_t)
    
    # Use external gates if provided (MAG variant)
    alpha_t = self.alpha
    theta_t = self.theta  
    eta_t = self.eta
    
    if external_gates is not None:
        alpha_t = external_gates['alpha_t']
        theta_t = external_gates['theta_t']
        eta_t = external_gates['eta_t']
    
    # Perform update with potentially modified parameters
    with tf.GradientTape() as tape:
        # Forward pass through memory MLP
        tape.watch(self.memory_mlp.trainable_variables)
        pred_v = self.memory_mlp(q_t, training=True)
        
        # Calculate loss
        loss = 0.5 * tf.reduce_sum(tf.square(pred_v - value_projection))
    
    # Get gradients and update memory
    grads = tape.gradient(loss, self.memory_mlp.trainable_variables)
    grad_norm = self._calculate_grad_norm(grads)
    
    # Update momentum with gradient scaling and decay
    self._update_momentum(grads, theta_t, eta_t)
    
    # Apply momentum and forgetting to memory weights
    self._update_memory_weights(alpha_t)
    
    return {"loss": loss.numpy(), "grad_norm": grad_norm.numpy()}
```

### 3. Enhanced Retrieve Method

```python
def retrieve(self, x_t, return_query=False):
    """Retrieve from memory using input embedding.
    
    Args:
        x_t: Input embedding
        return_query: Whether to return the query projection
        
    Returns:
        Retrieved embedding or tuple of (retrieved_embedding, query_projection)
    """
    # Ensure input is properly shaped
    x_t = self._prepare_input(x_t)
    
    # Calculate query projection
    q_t = self.query_projection(x_t)
    
    # Forward pass through memory MLP
    y_t = self.memory_mlp(q_t, training=False)
    
    if return_query:
        return y_t.numpy(), q_t.numpy()
    else:
        return y_t.numpy()
```

### 4. New Gate Calculation Method

```python
def calculate_gates(self, attention_output):
    """Calculate gate values based on attention output for MAG variant.
    
    Args:
        attention_output: Output from attention mechanism
        
    Returns:
        Tuple of (alpha_t, theta_t, eta_t)
    """
    # Ensure input is properly shaped
    attention_output = self._prepare_input(attention_output)
    
    # Simple linear transformation and sigmoid activation
    # This is a placeholder implementation - actual gate calculation
    # might be more sophisticated based on specific MAG design
    gate_layer = tf.keras.layers.Dense(3, activation="sigmoid")
    gates = gate_layer(attention_output)
    
    # Extract individual gates (default range 0-1)
    alpha_t = gates[0, 0].numpy()  # Forget rate
    theta_t = gates[0, 1].numpy()  # Inner learning rate
    eta_t = gates[0, 2].numpy()    # Momentum decay
    
    # Scale to appropriate ranges based on default values
    alpha_t = alpha_t * 0.1        # Scale to 0-0.1 range
    theta_t = theta_t * 0.5        # Scale to 0-0.5 range
    eta_t = 0.9 + (eta_t * 0.09)   # Scale to 0.9-0.99 range
    
    return alpha_t, theta_t, eta_t
```

## Neural Memory Client Changes

### 1. New Get Projections Method

```python
async def get_projections(self, embedding):
    """Get key, value, and query projections for an input embedding."""
    try:
        response = await self.post(
            "/get_projections",
            {"input_embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding}
        )
        return (
            np.array(response["key_projection"]),
            np.array(response["value_projection"]),
            np.array(response["query_projection"])
        )
    except Exception as e:
        logger.error(f"Error getting projections: {e}")
        return None, None, None
```

### 2. New Calculate Gates Method

```python
async def calculate_gates(self, attention_output):
    """Calculate gate values based on attention output."""
    try:
        response = await self.post(
            "/calculate_gates",
            {"attention_output": attention_output.tolist() if hasattr(attention_output, 'tolist') else attention_output}
        )
        return {
            "alpha_t": response["alpha_t"],
            "theta_t": response["theta_t"],
            "eta_t": response["eta_t"]
        }
    except Exception as e:
        logger.error(f"Error calculating gates: {e}")
        return None
```

### 3. Enhanced Update Memory Method

```python
async def update_memory(self, params):
    """Update neural memory with input embedding and optional parameters."""
    try:
        response = await self.post("/update_memory", params)
        return response
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        return {"status": "error", "error": str(e)}
```

### 4. New Get Config Method

```python
async def get_config(self):
    """Get Neural Memory configuration parameters."""
    try:
        response = await self.get("/config")
        return response
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return {"status": "error", "error": str(e)}
```

## Context Cascade Engine Changes

### 1. Dynamic Attention Configuration

```python
async def _initialize_attention_module(self):
    """Initialize attention module with dynamic configuration from Neural Memory server."""
    try:
        # Get configuration from Neural Memory server
        config_response = await self.neural_memory_client.get_config()
        
        if config_response["status"] == "success":
            # Calculate appropriate parameters
            key_dim = config_response["key_dim"]
            num_heads = config_response["recommended_attention_heads"]
            
            # Configure per-head dimension
            per_head_dim = max(key_dim // num_heads, 8)  # Ensure at least 8 dimensions per head
            
            # Create attention module
            self.attention_module = MultiHeadAttentionModule(
                num_heads=num_heads,
                key_dim=per_head_dim,
                use_layer_norm=True,
                use_residual=True
            )
            
            logger.info(f"Initialized attention module with {num_heads} heads, "
                       f"{per_head_dim} dimensions per head")
        else:
            # Fallback to default configuration
            logger.warning("Failed to get config from Neural Memory server. "
                          "Using default attention configuration.")
            self.attention_module = MultiHeadAttentionModule(
                num_heads=4,
                key_dim=32,
                use_layer_norm=True,
                use_residual=True
            )
    except Exception as e:
        logger.error(f"Error initializing attention module: {e}")
        # Fallback to default configuration
        self.attention_module = MultiHeadAttentionModule(
            num_heads=4,
            key_dim=32,
            use_layer_norm=True,
            use_residual=True
        )
```

## MetricsStore Fix

### Fix for format_diagnostics_as_table Method

```python
def format_diagnostics_as_table(self):
    """Format diagnostics data as a markdown table for display."""
    if not self.diagnostics:
        return "No diagnostics data available."
    
    # Ensure data_points exists with a default empty list
    if "data_points" not in self.diagnostics:
        self.diagnostics["data_points"] = []
    
    # Process data points
    data_points = self.diagnostics["data_points"]
    if not data_points:
        return "No data points in diagnostics."
    
    # Create table header
    headers = list(data_points[0].keys())
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---" for _ in headers]) + " |\n"
    
    # Add data rows
    for point in data_points:
        table += "| " + " | ".join([str(point.get(h, "")) for h in headers]) + " |\n"
    
    return table
```

## Backward Compatibility Considerations

1. **NONE Variant Support**: The refactored flow must continue to work with the "NONE" variant, which represents the original Phase 3 implementation without attention mechanisms.

2. **Default Gate Values**: When no external gates are provided (non-MAG variants), the system should use the default gate values from the Neural Memory configuration.

3. **Optional Parameters**: All new parameters in API requests should be optional to maintain compatibility with existing clients.

4. **Error Handling**: Enhanced error handling is needed to gracefully handle clients that do not send the expected parameters or handle the enhanced responses.

## Testing Strategy

### 1. API Endpoint Tests

Create comprehensive tests for each new and modified endpoint:

```python
async def test_get_projections_endpoint():
    """Test the /get_projections endpoint."""
    client = NeuralMemoryClient(...)
    
    # Test with valid embedding
    embedding = np.random.random(128).astype(np.float32)
    k_t, v_t, q_t = await client.get_projections(embedding)
    
    assert k_t is not None and len(k_t) > 0
    assert v_t is not None and len(v_t) > 0
    assert q_t is not None and len(q_t) > 0
    
    # Test with invalid embedding (e.g., NaN values)
    embedding_with_nan = np.array([np.nan] * 128).astype(np.float32)
    k_t, v_t, q_t = await client.get_projections(embedding_with_nan)
    
    # Should handle NaN gracefully
    assert k_t is not None
```

### 2. Integration Tests

Create tests that verify the complete flow for each variant:

```python
async def test_mag_variant_integration():
    """Test the complete MAG variant flow."""
    # Set environment variable
    os.environ["TITANS_VARIANT"] = "MAG"
    
    # Initialize components
    memory_client = MemoryCoreClient(...)
    neural_memory_client = NeuralMemoryClient(...)
    cce = ContextCascadeEngine(...)
    
    # Process a sequence of inputs
    embeddings = [np.random.random(128).astype(np.float32) for _ in range(5)]
    results = []
    
    for embedding in embeddings:
        result = await cce.process_new_input(embedding)
        results.append(result)
    
    # Verify gate values are being applied
    # (This would require instrumenting the neural_memory_module to expose actual gate values used)
```

## Conclusion

The API updates outlined in this document provide the necessary foundation for completing Phase 4 of the Lucidia cognitive system. These changes enable the Titans Architecture Variants to function correctly, with proper timing and information flow between components.

The enhanced API maintains backward compatibility while adding the flexibility needed for the attention-based variants. The addition of configuration endpoints and improved diagnostics will facilitate easier integration, monitoring, and debugging.

Implementing these changes will resolve the critical MAG/MAL timing issue identified in the codebase review, allowing all variants to function as designed and completing the Phase 4 implementation.

---

**Related Documentation:**
- [Phase 4 Implementation](phase_4_implementation.md)
- [Attention](attention.md)
- [Titans Variant Refactor](titans_variant_refactor.md)
