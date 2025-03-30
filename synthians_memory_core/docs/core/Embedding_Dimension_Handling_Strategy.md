Okay, here is the specific document detailing the embedding dimension handling strategy currently implemented in the Synthians codebase.

```markdown
# Synthians Cognitive Architecture: Embedding Dimension Handling Strategy

**Version:** 1.0
**Date:** March 29, 2025

## 1. Overview

This document outlines the strategy employed across the Synthians cognitive architecture (Memory Core, Neural Memory Server, Orchestrator) to handle potentially different embedding dimensions (e.g., 384D vs. 768D) and ensure robust processing of vector data.

The core goals of this strategy are:

1.  **Consistency:** Ensure vector operations (similarity, distance) work reliably even with mixed-dimension inputs.
2.  **Configurability:** Allow definition of a primary `embedding_dim` and an `alignment_strategy`.
3.  **Robustness:** Validate embeddings for correctness (e.g., check for NaN/Inf values) and handle invalid data gracefully.
4.  **Compatibility:** Ensure components requiring specific dimensions (like FAISS index, Neural Memory projections) receive correctly dimensioned data.

## 2. Core Strategy: Multi-Layered Validation and Alignment

The system uses a multi-layered approach, primarily centered around the `GeometryManager`, but with validation and alignment steps occurring at different component boundaries:

1.  **Central Authority (`GeometryManager`):**
    *   Defines the system's target `embedding_dim` via its configuration.
    *   Defines the `alignment_strategy` (`'truncate'` or `'pad'`) to use when dimensions mismatch the target.
    *   Provides core methods for validation (`_validate_vector`), alignment (`align_vectors`), and normalization (`normalize_embedding`).

2.  **API Layer Validation (Memory Core API):**
    *   The main API server (`api/server.py`) performs initial validation and alignment of embeddings received in requests (e.g., in `/process_memory`) *before* passing them to the `SynthiansMemoryCore` logic. This acts as a first line of defense.

3.  **Memory Core Internal Processing:**
    *   The `SynthiansMemoryCore` class relies heavily on the `GeometryManager` instance for all internal embedding operations: validating inputs, aligning vectors for comparison (`calculate_similarity`), and normalizing vectors.

4.  **Vector Index Internal Alignment (`MemoryVectorIndex`):**
    *   The `MemoryVectorIndex` (FAISS wrapper) performs its *own* validation and alignment (`_validate_embedding`, `_align_embedding_dimension`) when adding vectors (`add`) or receiving query vectors (`search`).
    *   **Crucially:** It ensures that all vectors *stored within the FAISS index itself* strictly match the index's configured `embedding_dim`. This is achieved by padding or truncating vectors *before* they are added to the FAISS C-level index.

5.  **Neural Memory Server Expectations:**
    *   The Neural Memory module (`neural_memory.py`) expects input tensors matching the dimensions defined in its `NeuralMemoryConfig` (`input_dim`, `key_dim`, etc.).
    *   Validation for the Neural Memory API (`http_server.py`) checks incoming vectors against these expected dimensions using `_validate_vector`.

6.  **Orchestrator (`ContextCascadeEngine`):**
    *   Acts primarily as a conduit, converting numpy arrays to lists for API calls.
    *   Relies on the shared `GeometryManager` for any internal validation or processing needs.

## 3. Key Components and Implementation Details

### 3.1. `GeometryManager`

*   **Configuration:**
    *   `embedding_dim`: Target dimension (e.g., 768).
    *   `alignment_strategy`: `'truncate'` (shorten longer vectors) or `'pad'` (zero-pad shorter vectors). Default appears to be a hybrid based on relative size if not specified, but explicit config is preferred.
    *   `normalization_enabled`: Controls L2 normalization.
*   **`_validate_vector`:** Checks for `None`, converts to `np.float32` array, checks for `NaN`/`Inf` (replaces with zeros and warns).
*   **`align_vectors`:** Takes two vectors, aligns *both* to the configured `embedding_dim` based on the `alignment_strategy`. Logs warnings on dimension mismatch (limited number of warnings).
*   **`normalize_embedding`:** Performs L2 normalization if enabled. Handles zero vectors.
*   **Backward Compatibility:** Includes `_align_vectors` and `_normalize` methods that simply forward calls to the non-underscored versions, ensuring components using older naming still work.

```python
# synthians_memory_core/geometry_manager.py

def align_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # ... validation ...
    target_dim = self.config['embedding_dim']
    strategy = self.config['alignment_strategy']
    # ... logic to pad/truncate vec_a and vec_b to target_dim ...
    if dim_a != target_dim:
        # Apply strategy to align vec_a to target_dim
        aligned_a = self._apply_alignment(vec_a, target_dim, strategy)
    if dim_b != target_dim:
        # Apply strategy to align vec_b to target_dim
        aligned_b = self._apply_alignment(vec_b, target_dim, strategy)
    return aligned_a, aligned_b

def _validate_vector(...):
    # ... checks for None, type, NaN/Inf ...
    if np.isnan(vector).any() or np.isinf(vector).any():
        # ... log warning ...
        return np.zeros_like(vector) # Replace invalid vector with zeros
    return vector
```

### 3.2. `MemoryVectorIndex` (FAISS Wrapper)

*   **Configuration:** Takes `embedding_dim` on initialization, which *must* match the dimension of the internal FAISS index.
*   **`_validate_embedding`:** Internal validation similar to `GeometryManager`, but *also* performs alignment (padding/truncation) to match `self.embedding_dim`. This is crucial because FAISS requires all vectors within an index to have the same dimension.
*   **`add`:** Calls `_validate_embedding` on the input vector. The validated (and potentially aligned) vector is added to the FAISS index.
*   **`search`:** Calls `_validate_embedding` on the query vector to ensure it matches the index dimension before performing the FAISS search.

```python
# synthians_memory_core/vector_index.py

def _validate_embedding(self, embedding: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
    # ... checks for None, type, 1D shape, NaN/Inf ...

    # Check dimension and align to self.embedding_dim
    if len(embedding) != self.embedding_dim:
        logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        if len(embedding) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        else:
            # Truncate
            embedding = embedding[:self.embedding_dim]
    # ... ensure float32 ...
    return embedding

def add(self, memory_id: str, embedding: np.ndarray) -> bool:
    validated_embedding = self._validate_embedding(embedding)
    if validated_embedding is None: return False
    # FAISS expects shape [n, dim]
    self.index.add(np.array([validated_embedding], dtype=np.float32))
    # ... update mapping ...

def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
    validated_query = self._validate_embedding(query_embedding)
    if validated_query is None: return []
    # FAISS expects shape [n, dim]
    distances, indices = self.index.search(np.array([validated_query], dtype=np.float32), k)
    # ... process results ...
```

### 3.3. Memory Core API Server (`api/server.py`)

*   **`process_memory` Endpoint:** Explicitly validates the incoming `embedding` for NaN/Inf and dimension mismatches *before* calling `memory_core.process_new_memory`. It aligns the embedding to the expected dimension (`memory_core.config['embedding_dim']`).
*   **Other Endpoints:** Generally pass embeddings as lists within JSON payloads. Downstream components are responsible for validation and alignment.

```python
# synthians_memory_core/api/server.py - Inside process_memory endpoint

if embedding is not None:
    # ... Check for NaN/Inf ...
    # Ensure correct dimensionality
    expected_dim = app.state.memory_core.config.get('embedding_dim', 768)
    actual_dim = len(embedding)
    if actual_dim != expected_dim:
        logger.warning(...)
        if actual_dim < expected_dim:
            embedding = embedding + [0.0] * (expected_dim - actual_dim) # Pad
        else:
            embedding = embedding[:expected_dim] # Truncate

# Call core processing with potentially aligned embedding
result = await app.state.memory_core.process_new_memory(...)
```

### 3.4. `SynthiansMemoryCore` Class

*   **`process_new_memory`:** Receives embedding (potentially pre-aligned by the API layer), validates again using `geometry_manager._validate_vector`, aligns using `geometry_manager._align_vectors` (often redundant if API pre-aligned, but safe), and normalizes using `geometry_manager._normalize`.
*   **`retrieve_memories` / `_get_candidate_memories`:** Uses `geometry_manager.calculate_similarity` for comparisons *after* retrieving candidates. Candidate retrieval relies on `vector_index.search`, where alignment happens internally.

### 3.5. Neural Memory Server (`synthians_trainer_server/http_server.py`)

*   **`_validate_vector` Helper:** Validates incoming vectors in API requests against the specific dimensions required by the endpoint (e.g., `input_dim` for `/update_memory`, `query_dim` for `/retrieve` queries *after projection*). It raises HTTPExceptions on mismatch. **It does not perform alignment.**
*   **Expectation:** Assumes the caller (CCE) provides correctly dimensioned vectors based on the Neural Memory's configuration.

### 3.6. Orchestrator (`orchestrator/context_cascade_engine.py`)

*   Relies on the shared `GeometryManager` for validation (`_validate_embedding`).
*   Uses helper (`_to_list`) to convert numpy arrays to lists before sending them via API calls to the Memory Core or Neural Memory Server.

## 4. Validation Details

*   **NaN/Inf Handling:** Vectors containing `NaN` or `Inf` are detected by `_validate_vector` (in `GeometryManager` and `MemoryVectorIndex`). These invalid vectors are typically replaced with **zero vectors** of the appropriate dimension, accompanied by a warning log.
*   **Shape:** Validation generally ensures vectors are 1-dimensional.
*   **Type:** Vectors are consistently converted to `np.float32` before being used in FAISS or TensorFlow operations.

## 5. Normalization

*   L2 normalization is typically applied to embeddings before storage, similarity calculation, or use in geometric operations.
*   This is controlled by the `normalization_enabled` flag in `GeometryManager` and implemented in `normalize_embedding`.

## 6. Configuration

*   **`embedding_dim`:** Set consistently across `GeometryManager`, `MemoryVectorIndex`, Memory Core API server (`SynthiansMemoryCore` config), and relevant dimensions in `NeuralMemoryConfig`.
*   **`alignment_strategy`:** Configured in `GeometryManager` (`'truncate'` or `'pad'`).

## 7. Potential Issues & Areas for Improvement

*   **Redundancy:** `MetadataSynthesizer` contains its own `_validate_embedding` and `_align_vectors_for_comparison` methods. These should ideally be removed, and it should use the shared `GeometryManager` instance for consistency.
*   **Consistency Checks:** Add startup checks to verify that `embedding_dim` configurations match across key components (GeometryManager, VectorIndex, NeuralMemory input/output dims where applicable).
*   **Alignment Strategy Default:** The default behavior in `GeometryManager`'s `align_vectors` if `alignment_strategy` isn't explicitly 'pad' or 'truncate' seems to be a mix (truncate if larger, pad if smaller). This should be clarified or made stricter based on the config value.

## 8. Conclusion

The Synthians system employs a robust, multi-layered strategy for handling embedding dimensions and validation. `GeometryManager` serves as the central configuration point, while `MemoryVectorIndex` ensures internal consistency for FAISS. Validation and alignment occur at API boundaries and within core components, aiming for both flexibility and operational reliability. Key features include NaN/Inf replacement, configurable alignment (padding/truncation), and consistent use of L2 normalization.
```