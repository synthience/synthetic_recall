# Geometry Management

The `synthians_memory_core.geometry_manager.GeometryManager` class is responsible for handling the geometric aspects of embedding vectors within the Synthians Memory Core.

## Core Responsibilities

1.  **Dimension Handling & Alignment:**
    *   Ensures that vectors being compared or processed have compatible dimensions, even if the system ingests embeddings of different sizes (e.g., 384 vs. 768).
    *   Uses the configured `alignment_strategy` with a **default of `'truncate'`** (not `'pad'`). This means that by default, when aligning vectors of different dimensions, the larger vector will be truncated to match the smaller one's dimension.
    *   The other available strategies are `'pad'` (which pads the smaller vector with zeros) and `'project'` (reserved for future implementation of dimension reduction techniques).
    *   Implementation of the alignment logic via the `align_vectors` method:
      ```python
      if strategy == 'pad':
          # Pad the smaller vector with zeros
          if dim_a < target_dim:
              aligned_a = np.pad(vec_a, (0, target_dim - dim_a), 'constant')
          if dim_b < target_dim:
              aligned_b = np.pad(vec_b, (0, target_dim - dim_b), 'constant')
      elif strategy == 'truncate':
          # Truncate to smaller dimension
          if dim_a > target_dim:
              aligned_a = vec_a[:target_dim]
          if dim_b > target_dim:
              aligned_b = vec_b[:target_dim]
      ```

2.  **Normalization:**
    *   Provides methods for L2 normalization (`normalize_embedding`), ensuring vectors have unit length, which is crucial for accurate cosine similarity calculations.
    *   Handles edge cases like zero vectors and vectors with NaN/Inf values during normalization.

3.  **Distance & Similarity Calculation:**
    *   Offers functions to compute distances (e.g., Euclidean) and similarities (e.g., Cosine) between vectors.
    *   Abstracts the specific geometric calculations based on configuration.
    *   Supports different similarity metrics:
      ```python
      def calculate_similarity(self, vec_a, vec_b):
          """Calculate similarity between two vectors based on the configured geometry."""
          geometry_type = self.config.get('geometry_type', GeometryType.EUCLIDEAN)
          
          if geometry_type == GeometryType.EUCLIDEAN:
              return self.calculate_cosine_similarity(vec_a, vec_b)
          elif geometry_type == GeometryType.HYPERBOLIC:
              return self.calculate_hyperbolic_similarity(vec_a, vec_b)
          # ... other geometries
      ```

4.  **Geometric Space Management:**
    *   Supports different geometric spaces beyond Euclidean:
      * `EUCLIDEAN`: Standard Euclidean space with cosine similarity
      * `HYPERBOLIC`: Hyperbolic space with custom similarity calculation
      * `SPHERICAL`: Reserved for future implementation
      * `MIXED`: Reserved for future implementation
    *   The `curvature` parameter (default `-1.0`) controls the properties of non-Euclidean spaces.

5.  **Robust Vector Validation:**
    *   Provides the `_validate_vector` method to detect and handle problematic vectors:
      * Checks for NaN/Inf values and replaces them with zeros
      * Handles different input types (lists, numpy arrays, torch tensors)
      * Tracks warning counts to avoid log spamming

## Key Difference from `embedding_handling.md`

While there is some overlap, the key distinction is:

* **GeometryManager (This Document)**: Focuses on the mathematical/geometric operations on vectors - how they are compared, aligned, normalized, and what geometric space they live in. This is the core component that implements the operations.

* **Embedding Handling (embedding_handling.md)**: Focuses on the overall system approach to embedding processing, including the integration points, validation flow, backward compatibility mechanisms, and how the GeometryManager is utilized throughout the system.

## Recent Implementation Improvements

Recent updates to the dimension handling implementation include:

* Unified approach to vector alignment across the system using the central GeometryManager
* Enhanced handling of dimension mismatches in HPC-QR factor calculations
* Improved validation to handle NaN/Inf values consistently
* Added backward compatibility methods to ensure consistent naming conventions

## Configuration

The behavior of the `GeometryManager` is influenced by the main `SynthiansMemoryCore` configuration:

*   `embedding_dim`: The primary embedding dimension used internally (default: `768`).
*   `geometry_type`: Specifies the default geometric space (default: `'euclidean'`).
*   `alignment_strategy`: How to handle dimension mismatches (default: `'truncate'`).
*   `normalization_enabled`: Whether to normalize vectors during operations (default: `True`).
*   `curvature`: Parameter for non-Euclidean geometries (default: `-1.0`).

## Importance

Centralizing geometric operations in `GeometryManager` ensures:

*   **Consistency:** All parts of the system use the same methods for alignment, normalization, and distance calculation.
*   **Robustness:** Handles potential issues like dimension mismatches gracefully.
*   **Flexibility:** Allows easier adaptation to different embedding types or geometric calculations in the future.
