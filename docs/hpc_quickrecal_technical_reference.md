# HPC-QuickRecal Technical Reference

## Mathematical Foundation

The HPC-QuickRecal (HPC-QR) system replaces traditional significance-based approaches with a unified equation that integrates multiple cognitive factors:

```
HPC_QR(x) = α·R_geom(x) + β·C_novel(x) + γ·S_org(x) - δ·O_red(x)
```

Where:
- `x` is the embedding to be evaluated
- `α`, `β`, `γ`, `δ` are tunable weights
- Each term represents a distinct cognitive factor

## Factor Definitions

### Geometry-Aware Distance: R_geom(x)

Measures the distance between a new embedding and existing memory representations in an appropriate geometric space.

**Implementation Options:**

```python
# Simple cosine distance
def compute_r_geom_cosine(embedding, center):
    return 1.0 - torch.cosine_similarity(embedding, center, dim=0)

# Hyperbolic distance
def compute_r_geom_hyperbolic(embedding, center, kappa=-1.0):
    diff = embedding - center
    norm = torch.norm(diff, p=2)
    return 2.0 / torch.sqrt(abs(kappa)) * torch.arcsinh(torch.sqrt(abs(kappa)) * norm / 2.0)
```

### Causal/Contextual Novelty: C_novel(x)

Quantifies how surprising or novel an embedding is relative to causal expectations.

**Implementation Options:**

```python
# Simple difference from last embedding
def compute_c_novel_simple(embedding, last_embedding):
    return float(torch.norm(embedding - last_embedding, p=2))

# Causal surprise using a Bayesian framework (pseudocode)
def compute_c_novel_causal(embedding, causal_model):
    # Compute prior probability
    prior_prob = causal_model.get_prior_probability(embedding)
    # Higher surprise for lower probability
    return -torch.log(prior_prob + 1e-9)  # Add epsilon to avoid log(0)
```

### Self-Organization Divergence: S_org(x)

Measures how strongly a new embedding prompts reconfiguration of internal representations.

**Implementation Options:**

```python
# Using Self-Organizing Map (SOM)
def compute_s_org_som(embedding, som):
    # Find best matching unit (BMU)
    bmu_idx = som.find_bmu(embedding)
    # Calculate quantization error
    bmu = som.weights[bmu_idx]
    return float(torch.norm(embedding - bmu, p=2))

# Meta-learning gradient magnitude
def compute_s_org_meta(embedding, meta_learner):
    # Store original parameters
    orig_params = [p.clone() for p in meta_learner.parameters()]
    
    # Forward pass
    loss = meta_learner.compute_loss(embedding)
    
    # Backward pass
    meta_learner.zero_grad()
    loss.backward()
    
    # Calculate gradient magnitude
    grad_magnitude = sum(torch.norm(p.grad) for p in meta_learner.parameters() if p.grad is not None)
    
    # Restore original parameters
    for i, param in enumerate(meta_learner.parameters()):
        param.data = orig_params[i]
    
    return float(grad_magnitude)
```

### Redundancy/Overlap: O_red(x)

Quantifies how much information in a new embedding is already present in memory.

**Implementation Options:**

```python
# Maximum cosine similarity
def compute_o_red_cosine(embedding, memory_buffer):
    # Calculate cosine similarity with all items in memory
    similarities = torch.matmul(memory_buffer, embedding)
    norms = torch.norm(memory_buffer, dim=1) * torch.norm(embedding)
    cos_sims = similarities / (norms + 1e-9)
    return float(torch.max(cos_sims))

# Information-theoretic redundancy
def compute_o_red_info(embedding, memory_buffer, encoder):
    # Encode the new embedding
    code_x = encoder.encode(embedding)
    
    # Encode each memory item
    mem_codes = [encoder.encode(mem) for mem in memory_buffer]
    
    # Calculate conditional entropy reduction
    redundancy = max(encoder.conditional_entropy_reduction(code_x, mem_code) for mem_code in mem_codes)
    
    return float(redundancy)
```

## Integration with UnifiedQuickRecallCalculator

The `UnifiedQuickRecallCalculator` class implements the HPC-QR equation and integrates with various cognitive factors:

```python
async def calculate_quickrecal_score(self, embedding, context=None):
    """Calculate the unified HPC-QR score for an embedding."""
    context = context or {}
    
    # Calculate individual factors
    r_geom = await self._calculate_r_geometry(embedding, context)
    c_novel = await self._calculate_causal_novelty(embedding, context)
    s_org = await self._calculate_diversity(embedding, context)  # Diversity as proxy for self-organization
    o_red = await self._calculate_surprise(embedding, context)  # Inverse of surprise as redundancy
    
    # Calculate emotion if available
    emotion = 0.0
    if self.config.get('use_emotion', False):
        emotion = await self._calculate_emotion(embedding, context)
    
    # Normalize each factor to [0,1] range if needed
    # ...
    
    # Apply HPC-QR equation with weights from config
    alpha = self.factor_weights.get('r_geometry', 0.4)
    beta = self.factor_weights.get('causal_novelty', 0.3)
    gamma = self.factor_weights.get('diversity', 0.2)
    delta = self.factor_weights.get('redundancy', 0.1)
    epsilon = self.factor_weights.get('emotion', 0.2)  # Optional emotion weight
    
    # Base HPC-QR equation
    hpc_qr = alpha * r_geom + beta * c_novel + gamma * s_org - delta * o_red
    
    # Add emotion if available
    if self.config.get('use_emotion', False):
        hpc_qr += epsilon * emotion
    
    return float(min(1.0, max(0.0, hpc_qr)))  # Clamp to [0,1]
```

## Dimension Handling

The HPC-QR system can handle embeddings of different dimensions through vector alignment:

```python
def _align_vectors_for_comparison(self, v1, v2, log_warnings=True):
    """Align two vectors to the same dimension for comparison operations."""
    if v1.shape != v2.shape:
        # Only log warning if under threshold and warnings are enabled
        if log_warnings and self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
            logger.warning(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
            self.dim_mismatch_warnings += 1
            
            # If this is the last warning we'll show, add a summary message
            if self.dim_mismatch_warnings >= self.max_dim_mismatch_warnings and not self.dim_mismatch_logged:
                logger.warning(f"Suppressing further dimension mismatch warnings after {self.max_dim_mismatch_warnings} occurrences")
                self.dim_mismatch_logged = True
        
        # Determine target dimension (minimum of the two)
        min_dim = min(v1.shape[0], v2.shape[0])
        
        # Truncate vectors to the same dimension
        v1_aligned = v1[:min_dim]
        v2_aligned = v2[:min_dim]
        
        return v1_aligned, v2_aligned
    
    return v1, v2
```

## Asynchronous Processing

The system leverages asynchronous processing for efficiency:

```python
async def process_embedding(self, embedding):
    """Asynchronously process a single embedding and return (processed_embedding, quickrecal_score)."""
    start_time = time.time()
    
    try:
        # Preprocess embedding
        normalized = await self._preprocess_embedding_async(embedding)
        
        # Compute HPC-QR score
        qr_score = await self.calculate_quickrecal_score(normalized)
        
        # Update momentum buffer
        await self._update_momentum_async(normalized)
        
        return normalized, qr_score
    
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        # Handle error appropriately
        raise
    finally:
        # Update processing time stats
        processing_time = time.time() - start_time
        self._update_stats(processing_time=processing_time)
```

## Configuration Parameters

Key configuration parameters for the HPC-QR system:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `embedding_dim` | Target embedding dimension | 384 |
| `alpha` | Weight for R_geom factor | 0.4 |
| `beta` | Weight for C_novel factor | 0.3 |
| `gamma` | Weight for S_org factor | 0.2 |
| `delta` | Weight for O_red factor | 0.1 |
| `use_emotion` | Whether to include emotional salience | False |
| `epsilon` | Weight for emotion factor | 0.2 |
| `max_dim_mismatch_warnings` | Threshold for dimension mismatch warnings | 10 |
| `som_grid_size` | Size of self-organizing map grid | (10, 10) |
| `retry_attempts` | Number of processing retry attempts | 3 |
| `timeout` | Processing timeout in seconds | 5.0 |

## Performance Considerations

1. **Vector Operations**: Most computations are vectorized for performance
   
2. **Thread Pool**: CPU-bound operations run in a thread pool to avoid blocking the main event loop
   
3. **Error Handling**: Robust retry mechanisms with exponential backoff
   
4. **Dimension Handling**: Efficient padding/truncation of vectors
   
5. **Logging Optimization**: Throttled warnings to prevent log flooding

## Implementation Recommendations

1. **Start Simple**: Begin with basic implementations of each factor
   
2. **Iterative Enhancement**: Gradually replace simple implementations with more sophisticated ones
   
3. **Weight Tuning**: Experiment with different weight configurations for your specific domain
   
4. **Monitoring**: Implement comprehensive stats tracking to monitor performance
   
5. **Integration Testing**: Validate that the HPC-QR scores lead to appropriate memory prioritization
