# Vector Index (FAISS)

The `synthians_memory_core.vector_index.MemoryVectorIndex` class manages the storage and efficient retrieval of high-dimensional embedding vectors using the FAISS library.

## Purpose

Vector indexing allows for fast approximate nearest neighbor (ANN) searches, enabling the Memory Core to quickly find memories semantically similar to a given query embedding.

## Key Component: `MemoryVectorIndex`

*   **Functionality:**
    *   Wraps a FAISS index object (e.g., `faiss.IndexFlatL2`, `faiss.IndexFlatIP`).
    *   Crucially, uses `faiss.IndexIDMap` to map the user-facing string `memory_id` (UUID) to the internal 64-bit integer IDs required by FAISS. This allows adding and retrieving vectors using the meaningful string IDs.
    *   Handles adding new vectors (`add_vector`), searching for similar vectors (`search`), removing vectors (`remove_vector`), and updating vectors (`update_vector`).
    *   Manages persistence of the FAISS index to disk (`save_index`, `load_index`).
    *   Provides utilities for verifying index integrity (`verify_index_integrity`) and migrating older index formats (`migrate_to_idmap`).
    *   Supports GPU acceleration if configured and available (`_initialize_gpu`).
*   **Integration:** Used extensively by `SynthiansMemoryCore` for storing embeddings associated with memories and performing similarity searches during retrieval.

## FAISS `IndexIDMap`

*   **Requirement:** Standard FAISS indices operate on sequential integer IDs (0, 1, 2...).
*   **Solution:** `IndexIDMap` acts as a layer on top of a base index (like `IndexFlatL2`). It maintains an internal mapping between arbitrary 64-bit integer IDs (which we derive from the string `memory_id`s) and the sequential IDs used by the base index.
*   **Benefit:** Allows using meaningful, potentially non-sequential IDs directly with `add_with_ids` and interpreting the IDs returned by `search`.
*   **GPU Limitation:** ⚠️ **Important:** When using `IndexIDMap`, the `add_with_ids` operation does not support GPU acceleration. The implementation falls back to CPU for these operations, even if the system is configured to use GPU. This is a limitation of the FAISS library itself, not the Synthians implementation. Search operations with `IndexIDMap` can still benefit from GPU acceleration.

## Persistence

*   The FAISS index itself (vectors and the ID map) is saved to a `.faiss` file (e.g., `storage_path/vector_index/memory_vectors.faiss`).
*   A separate `mapping.json` file is often kept as a backup, storing the `string_memory_id -> int64_faiss_id` mapping.

## Configuration

*   `vector_index_path`: Directory to store the index files.
*   `vector_index_type`: The base FAISS index type (e.g., `'IndexFlatL2'`, `'IndexFlatIP'`).
*   `use_gpu`: Boolean flag to enable GPU usage.
*   `embedding_dim`: Must match the dimension of the stored vectors.

## Importance

The vector index is the foundation of memory retrieval by semantic similarity, which is the core functionality of the Memory Core. An efficient, robust, and scalable vector index implementation is essential for overall system performance.

## Failure Handling

*   **Missing Index:** If the index file is not found on disk, a new one is automatically created.
*   **Index Corruption:** Methods like `verify_index_integrity` and `repair_index` can help diagnose and fix index issues.
*   **ID Mapping Loss:** If the mapping between string IDs and FAISS integer IDs is lost, it can potentially be recreated from the `memory_index.json` file using consistent hashing.
*   **GPU Fallback:** If GPU initialization fails, the system automatically falls back to CPU and logs a warning.

## Migration from Legacy Formats

Older versions might have used FAISS indices without `IndexIDMap`, relying on sequential IDs matching the position in some external memory list. The `migrate_to_idmap` method can convert these legacy indices to the more robust `IndexIDMap` format, ensuring each vector has a stable 64-bit ID derived from its string memory UUID.
