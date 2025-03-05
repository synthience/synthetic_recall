
---

# HypersphereTokenFlow Documentation

## 1. Overview

The **HypersphereTokenFlow** module provides a framework for:

1. **Projecting** token embeddings onto or near a hypersphere (enforcing L2 norm constraints).
2. **Applying chunk-based** or partial normalizations for large-scale HPC contexts.
3. **Managing** surprising token updates with partial resets (shock absorbers).
4. **Detecting** “surprise” in token embeddings via a combined MSE + max-deviation metric.
5. **Adapting** token flow using an attention-based iterative update (flowTokens).

By doing so, it aims to **improve stability** (bounding extreme embeddings), **preserve diversity** of tokens, and **enhance** contextual adaptation in HPC or large-scale scenarios.

---

## 2. Installation & Setup

1. **Prerequisites**:
   - Node.js ≥ 16 (16.x or 18.x LTS recommended for GPU support)
   - For Node.js:
     * `@tensorflow/tfjs-node-gpu` (for GPU acceleration)
     * `@tensorflow/tfjs-node` (CPU fallback)
   - For Browser:
     * `@tensorflow/tfjs` (WebGL acceleration)
   - CUDA toolkit and NVIDIA drivers (for native GPU support)
   - TypeScript (recommended)
2. **Install**:
   ```bash
   # For Node.js with GPU support
   npm install @tensorflow/tfjs-node-gpu

   # For Node.js without GPU
   npm install @tensorflow/tfjs-node

   # For browser environments
   npm install @tensorflow/tfjs

   npm install tslog
   ```
3. **Import** in your code:
   ```ts
   import { HypersphereTokenFlow, TokenFlowConfig } from './HypersphereTokenFlow';
   ```

---

## 3. GPU Acceleration & Backend Configuration

The HypersphereTokenFlow module supports multiple acceleration backends:

1. **Node.js Environment**:
   - **Native GPU** (CUDA):
     * Requires Node.js 16.x or 18.x LTS
     * CUDA toolkit and compatible NVIDIA drivers
     * Best performance for large-scale operations
   - **CPU Fallback**:
     * Automatically used if GPU is unavailable
     * Suitable for development and testing

2. **Browser Environment**:
   - **WebGL**:
     * Hardware acceleration through WebGL 2.0
     * Supports large matrix operations (tested up to 2000x2000)
     * Performance metrics:
       - Matrix multiplication (2000x2000): ~389ms
       - Max texture size: 16384
     * Automatic fallback to CPU if WebGL is unavailable

3. **Backend Selection**:
   ```ts
   // In Node.js
   import * as tf from '@tensorflow/tfjs-node-gpu';
   // Or for CPU: import * as tf from '@tensorflow/tfjs-node';

   // In Browser
   import * as tf from '@tensorflow/tfjs';
   await tf.setBackend('webgl');
   await tf.ready();
   ```

---

## 4. Configuration

All HPC/hypersphere settings reside in the **`TokenFlowConfig`** interface. Key fields include:

- **dimension**: The embedding dimension (e.g., 768).
- **numSteps**: Number of flow steps to apply in `flowTokens`.
- **temperature**: Blending factor in the slerp (spherical interpolation).
- **rMin, rMax**: Min/Max radius for soft hypersphere constraints.
- **chunkSize**: Size for chunk-based operations (like partial normalization).
- **normThreshold**: If average norm for a chunk is above this, we clamp or reduce norms.
- **shockThreshold**: Surprise threshold for shock absorber logic.
- **resetRatio**: Fraction of embedding we keep during a partial reset.
- **noiseScale**: Scale of random orthonormal noise used in shock absorber.
- **surpriseAlpha**: Weight for combining mean MSE vs. max deviation in surprise measure.

Example config:
```ts
const config: TokenFlowConfig = {
  dimension: 16,
  diversityThreshold: 0.5,
  numSteps: 10,
  temperature: 0.1,
  adaptiveScaling: true,
  maxHistorySize: 1000,
  rMin: 0.1,
  rMax: 1.0,
  epsilon: 1e-7,
  chunkSize: 32,
  normThreshold: 1.0,
  shockThreshold: 0.5,
  resetRatio: 0.5,
  noiseScale: 0.1,
  surpriseAlpha: 0.5
};
```

---

## 4. Core Classes & Functions

### 4.1 `HypersphereTokenFlow` Class

**Purpose**: The main interface for HPC token flow. Manages chunk-based normalization, partial resets, projection to hypersphere, and iterative flow steps using attention.

| Method                     | Description                                                                          |
|---------------------------|--------------------------------------------------------------------------------------|
| **constructor(config)**   | Initializes flow parameters using the `TokenFlowConfig`.                             |
| **projectToHypersphere**  | Projects tokens to a given `radius`; typically used for normalizing to L2=1.         |
| **slerp**                 | Spherical linear interpolation for two vectors.                                      |
| **computeGeodesicDistance** | Returns the angle between two vectors on the hypersphere.                          |
| **flowTokens**            | Iteratively updates tokens with attention-driven flow. Supports partial logs.        |
| **normalizeEmbedding**    | Single-vector L2 normalization with optional bounding in \([r_{\text{min}}, r_{\text{max}}]\). |
| **normalizeChunk**        | Chunk-based normalization for large sets.                                           |
| **applyShockAbsorber**    | Partial reset logic when surprise is high.                                           |
| **calculateSurprise**     | Weighted MSE + max calculation for novelty detection.                                |
| **processEmbeddingBatch** | Splits embeddings into chunks and normalizes them if needed.                         |
| **dispose**               | Cleans up resources and logs final disposal message.                                 |

---

### 4.2 `TokenFlowMonitor` Class

Used internally for logging token displacement or diversity:

- **logDisplacement**: Compares initial tokens vs. current tokens to show average difference.
- **logDiversity**: (Not fully implemented) Could measure pairwise similarities or angular spreads.

```ts
class TokenFlowMonitor {
  logDisplacement(initialTokens: tf.Tensor, currentTokens: tf.Tensor): void {
    const displacement = tf.mean(tf.sub(currentTokens, initialTokens)).dataSync()[0];
    logger.info(`Average token displacement: ${displacement}`);
  }

  logDiversity(tokens: tf.Tensor): void {
    logger.info("Diversity logging not yet implemented");
  }
}
```

---

## 5. Typical Usage Workflow

1. **Instantiate** with desired config:
   ```ts
   const myFlowHandler = new HypersphereTokenFlow(config);
   ```
2. **Load / Create** a batch of token embeddings:
   ```ts
   // Suppose we have a [batchSize, seqLen, dimension] tensor
   const tokens = tf.randomNormal([4, 8, config.dimension]);
   ```
3. **Generate attention** for flow:
   ```ts
   const attention = tf.randomUniform([4, 8, 8]); // e.g., random for testing
   ```
4. **Apply flow**:
   ```ts
   const updatedTokens = await myFlowHandler.flowTokens(tokens, attention);
   console.log("Flowed token shape:", updatedTokens.shape);
   ```
5. **Cleanup** if desired:
   ```ts
   myFlowHandler.dispose();
   ```

---

## 6. Example: HPC Batch Processing

```ts
// HPC scenario: we have large embedding sets

// Step 1: Build an NxD embedding matrix
const embeddings = tf.randomNormal([10000, config.dimension]) as tf.Tensor2D;

// Step 2: pass them in manageable chunks
const processed = await myFlowHandler.processEmbeddingBatch(embeddings);

// processed now chunk-normalized (only if avg norm > normThreshold).
```

---

## 7. Code Snippets & Explanation

### 7.1 Chunk-Based Normalization

```ts
async normalizeChunk(chunk: tf.Tensor2D): Promise<tf.Tensor2D> {
  return tf.tidy(() => {
    // norms: shape [chunkSize]
    const norms = tf.norm(chunk, 2, 1); 
    // average norm
    const avgNorm = tf.mean(norms).dataSync()[0];

    if (avgNorm > this.normThreshold) {
      const normFactors = tf.reciprocal(tf.add(norms, this.epsilon));
      // Expand dims to broadcast across each row
      return tf.mul(chunk, tf.expandDims(normFactors, 1));
    }
    return chunk;
  });
}
```
- For each chunk, compute row-wise L2 norm.
- Compare the average to `normThreshold`.
- If above threshold, re-scale each row by `1 / \| \mathbf{v} \|`.

### 7.2 Shock Absorber Partial Reset

```ts
async applyShockAbsorber(embedding: tf.Tensor1D): Promise<tf.Tensor1D> {
  return tf.tidy(() => {
    const scaledEmbedding = tf.mul(embedding, this.resetRatio);
    const noise = tf.randomNormal(embedding.shape, 0, this.noiseScale);
    
    // Project out the component parallel to embedding
    const dotProduct = tf.sum(tf.mul(noise, embedding));
    const embNormSquared = tf.square(tf.norm(embedding));
    
    // projection = (dot / norm^2)* embedding
    const projection = tf.mul(embedding, tf.div(dotProduct, embNormSquared));
    const orthogonalNoise = tf.sub(noise, projection);
    
    // combined
    const combined = tf.add(
      scaledEmbedding,
      tf.mul(orthogonalNoise, 1 - this.resetRatio)
    );
    
    // re-normalize
    return tf.div(combined, tf.add(tf.norm(combined), this.epsilon));
  });
}
```
- **Explanation**: Scales the original embedding by `resetRatio`, then adds orthonormal random noise. This “partial reset” helps the model pivot away from an extreme vector without losing all prior structure.

---

## 8. Performance Considerations

- **Chunk Size**: For HPC scenarios, pick a chunk size that balances memory usage and re-scaling overhead.  
- **tf.tidy** usage**: Ensures intermediate Tensors are released.  
- **Check Overheads**: If `numSteps` is large or `flowTokens` is called repeatedly, consider how many logs or slerp operations occur, possibly reduce logs in production.  
- **GPU vs. CPU**: If you run `@tensorflow/tfjs-node-gpu`, ensure you have a suitable environment (CUDA, etc.).  

---

## 9. Common Pitfalls

1. **Forgetting to `await`** the `flowTokens` or chunk-based methods that do asynchronous `tf` ops.  
2. **Over-logging** in HPC contexts, drowning out relevant info.  
3. **Letting partial resets** degrade embeddings to zero if noise or ratio is set incorrectly.  
4. **Misconfiguring** minRadius > maxRadius or chunkSize > total embeddings.  

---

## 10. Extended Example

```ts
import * as tf from '@tensorflow/tfjs-node';
import { HypersphereTokenFlow, TokenFlowConfig } from './HypersphereTokenFlow';

async function runHypersphereDemo() {
  const config: TokenFlowConfig = {
    dimension: 32,
    diversityThreshold: 0.3,
    numSteps: 5,
    temperature: 0.2,
    adaptiveScaling: true,
    maxHistorySize: 500,
    rMin: 0.5,
    rMax: 2.0,
    epsilon: 1e-8,
    chunkSize: 64,
    normThreshold: 1.2,
    shockThreshold: 2.0,
    resetRatio: 0.5,
    noiseScale: 0.05,
    surpriseAlpha: 0.6
  };

  // Step 1: Initialize
  const flowHandler = new HypersphereTokenFlow(config);

  // Step 2: Generate random tokens & attention
  const batchSize = 2, seqLen = 10;
  const tokens = tf.randomNormal([batchSize, seqLen, config.dimension]);
  const attn = tf.randomUniform([batchSize, seqLen, seqLen]);

  // Step 3: Flow tokens
  console.log("Initial tokens shape:", tokens.shape);
  const flowed = await flowHandler.flowTokens(tokens, attn);

  // Step 4: Possibly do HPC chunk-based operations on the final embedding
  const finalMatrix = flowed.reshape([batchSize * seqLen, config.dimension]);
  const processed = await flowHandler.processEmbeddingBatch(finalMatrix);

  console.log("Processed matrix shape:", processed.shape);

  // Cleanup
  flowHandler.dispose();
}

runHypersphereDemo().catch(console.error);
```

---

## 11. Glossary

- **tf.tidy**: Utility from TensorFlow.js to automatically dispose intermediate Tensors.  
- **Slerp**: Spherical Linear Interpolation.  
- **Geodesic Distance**: Angle between two unit vectors on the hypersphere.  
- **Shock Absorber**: A partial reset mechanism that randomizes embeddings away from extremes.  
- **Surprise**: Weighted measure of how unexpected an embedding is relative to a prediction.

---

## 12. Concluding Remarks

The **HypersphereTokenFlow** system merges HPC-friendly chunk-based normalization, robust outlier handling (shock absorbers), and flexible “surprise” scoring. By following the best practices—like using `tf.tidy`, configuring chunk sizes, and carefully balancing partial resets—users can scale from local experiments to large HPC pipelines while preserving embedding stability and diversity.

**Happy coding on the hypersphere!**