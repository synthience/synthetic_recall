# Migration Guide: From Significance-Based to HPC-QuickRecal

## Introduction

This guide outlines the process of migrating from traditional significance-based memory systems to the new HPC-QuickRecal (HPC-QR) framework. The migration involves both conceptual changes and code modifications.

## Conceptual Shifts

### From

- Discrete, separate calculations of significance metrics
- Simple threshold-based memory decisions
- Limited geometric awareness
- No explicit consideration of causal novelty or redundancy

### To

- Unified scoring system that integrates multiple factors
- Geometry-aware distance metrics
- Causal and contextual novelty detection
- Self-organization divergence measurements
- Explicit redundancy penalties

## Migration Steps

### 1. Identify Current Significance Calculations

Locate all significance-related calculations in your existing system. These might include:

- Surprise scores based on vector distances
- Novelty metrics based on time or frequency
- Simple thresholding mechanisms
- Ad-hoc prioritization rules

### 2. Map Existing Components to HPC-QR Factors

Determine how your existing calculations map to the four main HPC-QR factors:

| Old Component | HPC-QR Factor |
|---------------|---------------|
| Vector distance | R_geom(x) |
| Surprise metrics | C_novel(x) |
| Diversity/uniqueness | S_org(x) |
| Similarity checks | O_red(x) |

### 3. Implement Core HPC-QR Calculator

Create or update your calculator class to implement the unified HPC-QR equation:

```python
class HPCQuickRecalCalculator:
    def __init__(self, config):
        self.config = {
            'embedding_dim': 384,
            'alpha': 0.4,  # R_geom weight
            'beta': 0.3,   # C_novel weight
            'gamma': 0.2,  # S_org weight
            'delta': 0.1,  # O_red weight
            **(config or {})
        }
        self.momentum_buffer = None
        
    def compute_hpc_qr(self, embedding):
        """Calculate the unified HPC-QR score."""
        # Calculate individual factors
        r_geom = self._calculate_r_geom(embedding)
        c_novel = self._calculate_c_novel(embedding)
        s_org = self._calculate_s_org(embedding)
        o_red = self._calculate_o_red(embedding)
        
        # Apply HPC-QR equation with weights
        alpha = self.config['alpha']
        beta = self.config['beta']
        gamma = self.config['gamma']
        delta = self.config['delta']
        
        hpc_qr = alpha * r_geom + beta * c_novel + gamma * s_org - delta * o_red
        
        return float(min(1.0, max(0.0, hpc_qr)))  # Clamp to [0,1]
```

### 4. Replace Individual Factor Calculations

Update the implementations of individual factors with more sophisticated versions:

```python
def _calculate_r_geom(self, embedding):
    """Calculate the geometry-aware distance."""
    if self.momentum_buffer is None:
        return 0.5  # Default mid-range value when no buffer exists
    
    # Calculate center of momentum buffer
    if isinstance(self.momentum_buffer, list):
        if not self.momentum_buffer:  # Empty list
            return 0.5
        center = np.mean(self.momentum_buffer, axis=0)
    else:  # Assume numpy array or torch tensor
        center = np.mean(self.momentum_buffer, axis=0) if len(self.momentum_buffer.shape) > 1 else self.momentum_buffer
    
    # Ensure dimensions match
    embedding, center = self._align_vectors_for_comparison(embedding, center)
    
    # Calculate geometric distance (cosine distance is a simple starting point)
    similarity = np.dot(embedding, center) / (np.linalg.norm(embedding) * np.linalg.norm(center) + 1e-9)
    distance = 1.0 - similarity
    
    return float(distance)
```

### 5. Update Memory Storage and Retrieval

Update how memory items are stored and retrieved based on HPC-QR scores:

```python
def process_and_store(self, embedding, content):
    """Process an embedding and decide whether to store based on HPC-QR score."""
    # Calculate QuickRecal score
    qr_score = self.compute_hpc_qr(embedding)
    
    # Determine storage strategy based on score
    if qr_score > self.config.get('storage_threshold', 0.7):
        # High QuickRecal score - store with high priority
        self.memory_store.add(content, embedding, priority=qr_score)
        return True, qr_score
    elif qr_score > self.config.get('reference_threshold', 0.4):
        # Medium QuickRecal score - store but with lower priority
        self.memory_store.add(content, embedding, priority=qr_score)
        return True, qr_score
    else:
        # Low QuickRecal score - don't store
        return False, qr_score
```

### 6. Implement Vector Alignment for Dimension Handling

Ensure proper handling of embeddings with different dimensions:

```python
def _align_vectors_for_comparison(self, v1, v2, log_warnings=True):
    """Align two vectors to the same dimension for comparison operations."""
    v1_np = np.asarray(v1)
    v2_np = np.asarray(v2)
    
    if v1_np.shape != v2_np.shape:
        # Log warning if needed
        if log_warnings:
            logger.warning(f"Vector dimensions don't match: {v1_np.shape} vs {v2_np.shape}")
        
        # Determine target dimension (minimum of the two)
        min_dim = min(v1_np.shape[0], v2_np.shape[0])
        
        # Truncate vectors to the same dimension
        v1_aligned = v1_np[:min_dim]
        v2_aligned = v2_np[:min_dim]
        
        return v1_aligned, v2_aligned
    
    return v1_np, v2_np
```

### 7. Update Asynchronous Processing

Modify async processing flows to incorporate HPC-QR calculations:

```python
async def process_embedding_async(self, embedding):
    """Process an embedding asynchronously and return HPC-QR score."""
    try:
        # Preprocess embedding if needed
        preprocessed = self._preprocess_embedding(embedding)
        
        # Compute HPC-QR score
        qr_score = await self._compute_hpc_qr_async(preprocessed)
        
        # Update momentum buffer
        self._update_momentum(preprocessed)
        
        return preprocessed, qr_score
    except Exception as e:
        logger.error(f"Error in async processing: {e}")
        raise
```

### 8. Update Configuration Parameters

Update your configuration with HPC-QR specific parameters:

```python
default_config = {
    # Embedding parameters
    'embedding_dim': 384,
    'normalize_embeddings': True,
    
    # HPC-QR weights
    'alpha': 0.4,  # R_geom weight
    'beta': 0.3,   # C_novel weight
    'gamma': 0.2,  # S_org weight
    'delta': 0.1,  # O_red weight
    
    # Additional factors
    'use_emotion': False,
    'epsilon': 0.2,  # Emotion weight if used
    
    # Storage thresholds
    'storage_threshold': 0.7,
    'reference_threshold': 0.4,
    
    # Performance settings
    'max_threads': 4,
    'retry_attempts': 3,
    'timeout': 5.0,
    
    # Logging settings
    'max_dim_mismatch_warnings': 10
}
```

### 9. Update Tests

Update your test suite to validate the new HPC-QR system:

```python
def test_hpc_qr_calculation():
    """Test the HPC-QR calculation with known inputs and expected outputs."""
    calculator = HPCQuickRecalCalculator({
        'alpha': 0.5,
        'beta': 0.3,
        'gamma': 0.1,
        'delta': 0.1
    })
    
    # Create a test embedding
    embedding = np.random.random(384)
    
    # Mock the internal factor calculations
    calculator._calculate_r_geom = lambda x: 0.7
    calculator._calculate_c_novel = lambda x: 0.6
    calculator._calculate_s_org = lambda x: 0.5
    calculator._calculate_o_red = lambda x: 0.4
    
    # Calculate expected result: 0.5*0.7 + 0.3*0.6 + 0.1*0.5 - 0.1*0.4 = 0.59
    expected_score = 0.59
    
    # Get actual result
    actual_score = calculator.compute_hpc_qr(embedding)
    
    # Assert approximate equality (floating point comparison)
    assert abs(actual_score - expected_score) < 1e-6
```

## Advanced Migration Options

### Riemannian Geometry Integration

For more sophisticated handling of curved embedding spaces:

```python
def _calculate_r_geom_hyperbolic(self, embedding, center, kappa=-1.0):
    """Calculate distance in hyperbolic space."""
    # Convert to torch tensors for easier calculation
    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
    center_tensor = torch.tensor(center, dtype=torch.float32)
    
    # Calculate hyperbolic distance
    diff = emb_tensor - center_tensor
    norm = torch.norm(diff, p=2)
    distance = 2.0 / torch.sqrt(abs(kappa)) * torch.arcsinh(torch.sqrt(abs(kappa)) * norm / 2.0)
    
    return float(distance)
```

### Integration with Self-Organizing Maps

For more sophisticated self-organization divergence:

```python
class SOMIntegration:
    def __init__(self, input_dim, grid_size=(10, 10)):
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.weights = np.random.random((grid_size[0], grid_size[1], input_dim))
        self.learning_rate = 0.5
        self.sigma = max(grid_size) / 2.0
    
    def find_bmu(self, x):
        """Find the Best Matching Unit for input x."""
        # Ensure x has the right dimension
        if len(x) != self.input_dim:
            # Handle dimension mismatch
            if len(x) > self.input_dim:
                x = x[:self.input_dim]
            else:
                x_padded = np.zeros(self.input_dim)
                x_padded[:len(x)] = x
                x = x_padded
        
        # Calculate distance to all neurons
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        return bmu_idx
    
    def update_weights(self, x, bmu_idx, iteration):
        """Update weights based on input and BMU."""
        # Update learning rate and neighborhood size based on iteration
        learning_rate = self.learning_rate * np.exp(-iteration / 1000.0)
        sigma = self.sigma * np.exp(-iteration / 1000.0)
        
        # Generate grid coordinates
        rows, cols = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]))
        grid_dist = np.sqrt((rows - bmu_idx[0]) ** 2 + (cols - bmu_idx[1]) ** 2)
        
        # Calculate influence
        influence = np.exp(-grid_dist ** 2 / (2 * sigma ** 2))
        influence = influence.reshape(self.grid_size[0], self.grid_size[1], 1)
        
        # Update weights
        self.weights += learning_rate * influence * (x - self.weights)
    
    def compute_divergence(self, x):
        """Compute divergence/energy required to incorporate x."""
        bmu_idx = self.find_bmu(x)
        bmu = self.weights[bmu_idx]
        
        # Quantization error
        quant_error = np.linalg.norm(x - bmu)
        
        # Make a copy of weights
        orig_weights = self.weights.copy()
        
        # Simulate one update step
        self.update_weights(x, bmu_idx, 0)
        
        # Calculate weight change magnitude
        weight_delta = np.linalg.norm(self.weights - orig_weights)
        
        # Restore original weights
        self.weights = orig_weights
        
        # Combine quantization error and weight change
        return float(quant_error + 0.5 * weight_delta)
```

## Troubleshooting Common Migration Issues

### Dimension Mismatches

**Problem**: Errors due to embeddings with different dimensions.

**Solution**: Use the `_align_vectors_for_comparison` method to ensure compatibility, and add logging throttling to prevent excessive warnings:

```python
# Add class-level counters
self.dim_mismatch_warnings = 0
self.max_dim_mismatch_warnings = 10
self.dim_mismatch_logged = False

# Then in the alignment method
if log_warnings and self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
    logger.warning(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
    self.dim_mismatch_warnings += 1
    
    # If this is the last warning we'll show, add a summary message
    if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
        logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
        self.dim_mismatch_logged = True
```

### Parameter Tuning

**Problem**: Difficulty finding the right values for α, β, γ, δ.

**Solution**: Start with defaults then implement a parameter sweep:

```python
def parameter_sweep():
    """Run a parameter sweep to find optimal HPC-QR weights."""
    # Test dataset
    test_data = load_test_embeddings()
    
    best_score = 0
    best_params = {}
    
    # Parameter ranges to test
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    betas = [0.2, 0.3, 0.4, 0.5]
    gammas = [0.1, 0.2, 0.3]
    deltas = [0.05, 0.1, 0.15, 0.2]
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                for delta in deltas:
                    # Skip invalid combinations (should sum to approximately 1)
                    if abs(alpha + beta + gamma - delta - 1.0) > 0.1:
                        continue
                    
                    # Configure calculator with these parameters
                    calc = HPCQuickRecalCalculator({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'delta': delta
                    })
                    
                    # Evaluate performance
                    score = evaluate_performance(calc, test_data)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                            'delta': delta
                        }
    
    return best_params, best_score
```

### Integration with External Services

**Problem**: Errors when integrating with emotion detection or other external services.

**Solution**: Implement robust fallbacks:

```python
async def _calculate_emotion(self, embedding, context):
    """Calculate emotional salience with robust fallbacks."""
    try:
        # Attempt to use the emotion service if text is available
        if 'text' in context and self.emotion_service is not None:
            emotion_score = await self.emotion_service.analyze_text(context['text'])
            return float(emotion_score)
        
        # Fallback to embedding-based emotion if no text or service error
        if self.emotion_detector is not None:
            emotion_score = self.emotion_detector.detect_from_embedding(embedding)
            return float(emotion_score)
        
        # Final fallback: use a static model or heuristic
        return self._fallback_emotion_estimate(embedding)
    
    except Exception as e:
        logger.warning(f"Emotion calculation failed: {e}. Using fallback.")
        return self._fallback_emotion_estimate(embedding)
```

## Conclusion

Migrating from a significance-based approach to HPC-QuickRecal requires architectural changes but offers substantial benefits in memory prioritization accuracy and efficiency. This migration guide outlines the key steps, from basic implementation to advanced extensions.

By following this guide, you can modernize your memory system to better handle different embedding dimensions, incorporate geometric and causal factors, and provide more human-like memory prioritization.

For any questions or additional support, refer to the [HPC-QuickRecal Technical Reference](./hpc_quickrecal_technical_reference.md) or contact the system maintainers.
