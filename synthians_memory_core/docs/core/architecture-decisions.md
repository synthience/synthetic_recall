# Synthians Cognitive Architecture - Architecture Decision Log (ADL)

**Status:** Active
**Date:** 2025-04-07

This document records significant architectural decisions made during the development of the Synthians Cognitive Architecture, particularly focusing on the Memory Core and its integrations. Its purpose is to provide context and rationale for key design choices.

---

## ADL-001: Unified Memory Core (`synthians_memory_core`)

*   **Date:** ~2025-03-25 (During initial refactor)
*   **Status:** Implemented
*   **Decision:** Consolidate core memory functionality (storage, retrieval, indexing, scoring, assemblies, emotion) into a single, cohesive Python package (`synthians_memory_core`) and primary class (`SynthiansMemoryCore`). Decommission distributed/fragmented components (HPC server, Tensor server, multiple persistence layers, complex KG modules for MVP).
*   **Context:** The previous codebase was highly fragmented, with duplicated logic, unclear component boundaries, and complex inter-service dependencies, making maintenance and development difficult. A local, high-performance MVP was needed.
*   **Rationale:**
    *   **Maintainability:** Centralizing core logic improves code clarity and reduces redundancy.
    *   **Performance (Local):** Eliminates network latency between core memory functions for the MVP.
    *   **Simplicity:** Reduces the number of moving parts and deployment complexity for the initial unified system.
    *   **Focus:** Allows concentrating development effort on core algorithms (QuickRecal, Assemblies, Geometry) before scaling out.
*   **Impact:** Required significant refactoring. Removed components like `MemoryBroker`, `MemoryClientProxy`, `hpc_server`, `tensor_server`. Simplifies local deployment but requires later consideration for distributed scaling if needed.

---

## ADL-002: FAISS `IndexIDMap` for Vector Index

*   **Date:** ~2025-03-27 (During core unification/vector index implementation)
*   **Status:** Implemented
*   **Decision:** Utilize `faiss.IndexIDMap` wrapping a base index (e.g., `IndexFlatIP` for cosine similarity) as the primary vector storage mechanism in `MemoryVectorIndex`. Generate 64-bit integer IDs deterministically from string IDs (e.g., `mem_xyz`, `asm_abc`) using hashing.
*   **Context:** Needed a way to map stable, human-readable/application-level string IDs to FAISS's internal vector indices, allowing efficient removal and updates without relying on potentially unstable sequential indices.
*   **Rationale:**
    *   **Stable IDs:** Allows using UUIDs or other meaningful string identifiers directly.
    *   **Efficient Operations:** `IndexIDMap` provides `add_with_ids` and `remove_ids` for direct manipulation using the mapped 64-bit IDs.
    *   **Decoupling:** Separates the application's ID space from FAISS's internal indexing.
*   **Impact:**
    *   FAISS `IndexIDMap` add/remove operations are CPU-bound, even if the base index uses GPU for search. This limits the speed benefits of GPU for write-heavy workloads but ensures ID stability.
    *   Requires managing the mapping between string IDs and 64-bit integer IDs persistently alongside the FAISS index itself (`.bin` and `.mapping.json`).
    *   Introduces potential inconsistencies between the index and the mapping, necessitating integrity checks (See ADL-004, ADL-005).

---

## ADL-003: Asynchronous Operations & Concurrency Control

*   **Date:** ~2025-03-28 (During core unification and fixing hangs)
*   **Status:** Implemented
*   **Decision:** Implement core I/O operations (persistence, vector index saves/loads) and background tasks asynchronously using `asyncio`. Utilize `asyncio.Lock` to protect shared state modifications (in-memory caches, index mappings) within `SynthiansMemoryCore` and `MemoryPersistence`. Use `asyncio.to_thread` for blocking library calls (like some FAISS operations or standard `json`/`os` calls) within async functions.
*   **Context:** Needed to prevent blocking operations (disk I/O, potentially slow FAISS operations) from halting the main API event loop, ensuring responsiveness. Shared data structures (`_memories`, `assemblies`, `id_to_index`) needed protection against race conditions.
*   **Rationale:**
    *   **Responsiveness:** `asyncio` allows the server to handle other requests while waiting for I/O.
    *   **Safety:** `asyncio.Lock` prevents concurrent modification issues.
    *   **Compatibility:** `asyncio.to_thread` allows using synchronous libraries within the async framework without blocking the main loop significantly.
*   **Impact:** Increased code complexity due to `async`/`await` syntax. Requires careful management of locks to avoid deadlocks. Background tasks need robust error handling and graceful shutdown mechanisms.

---

## ADL-004: Vector Index Persistence Strategy

*   **Date:** ~2025-03-28 (During core unification/bug fixing)
*   **Status:** Implemented
*   **Decision:** Persist the `MemoryVectorIndex` state as two separate files: `faiss_index.bin` (using `faiss.write_index`) and `faiss_index.bin.mapping.json` (JSON dump of the `id_to_index` dictionary mapping string IDs to 64-bit int IDs). Use atomic writes (write to temp file, then rename/move) for the mapping file.
*   **Context:** The state of the `IndexIDMap` (both the vectors and the ID mapping) needs to be saved to survive restarts. FAISS's `write_index` only saves the vector data and internal FAISS structure, not the string-to-int64 mapping.
*   **Rationale:**
    *   **Completeness:** Saves both essential pieces of information needed to restore the index state.
    *   **Atomicity (Mapping):** Atomic write for the mapping file prevents corruption if a crash occurs during saving. (FAISS `write_index` is assumed to be reasonably atomic).
    *   **Decoupling:** Separates the potentially large binary index data from the smaller, human-readable mapping.
*   **Impact:** Requires loading *both* files during initialization. Introduces the possibility of the two files becoming desynchronized, necessitating integrity checks (See ADL-005). Backup strategies must include both files.

---

## ADL-005: Vector Index Integrity Checks & Repair

*   **Date:** ~2025-03-30 -> 2025-04-04 (Phase 5.8 Stability)
*   **Status:** Implemented
*   **Decision:**
    1.  Implement `MemoryVectorIndex.check_index_integrity` for **diagnosis only**, reporting consistency status and counts (FAISS vs. mapping).
    2.  Implement `MemoryVectorIndex.repair_index_async` (and `SynthiansMemoryCore.detect_and_repair_index_drift`) for **explicit repair actions**, including strategies like rebuilding the index from persistence or recreating the mapping.
    3.  Trigger `detect_and_repair_index_drift` automatically during `SynthiansMemoryCore.initialize`.
    4.  Add optional periodic background checks (`_auto_repair_drift_loop`).
*   **Context:** Discovered inconsistencies between FAISS vector count and the ID mapping count, leading to errors. The original verification method attempted repairs, causing side effects.
*   **Rationale:**
    *   **Separation of Concerns:** Diagnosis should not have side effects. Repair should be an explicit action.
    *   **Robustness:** Automatic checks and repairs on startup improve self-healing capabilities.
    *   **Observability:** Diagnostic information helps identify the *type* of inconsistency.
    *   **Flexibility:** Different repair strategies (`rebuild_from_persistence`, `recreate_mapping`) handle different failure modes.
*   **Impact:** Adds complexity to initialization and background tasks. Repair (especially rebuilding from persistence) can be time-consuming. Requires `MemoryPersistence` and `GeometryManager` instances to be passed to the repair function.

---

## ADL-006: Centralized Embedding Validation & Alignment

*   **Date:** ~2025-03-29 -> 2025-04-03 (Phase 5.8 Stability / Bug Fixing)
*   **Status:** Implemented
*   **Decision:** Centralize core embedding validation (NaN/Inf checks), dimension alignment (padding/truncation based on config), and normalization logic primarily within `GeometryManager`. Provide utility functions (`embedding_validators.py`) for components needing standalone validation (like API layers). Ensure components like `MemoryVectorIndex` perform necessary final validation/alignment specific to their internal requirements (e.g., matching FAISS index dimension).
*   **Context:** Inconsistent handling of invalid embeddings (NaN/Inf) and dimension mismatches (e.g., 384 vs. 768) across different modules led to errors, especially during vector comparisons and indexing. Validation logic was duplicated.
*   **Rationale:**
    *   **Consistency:** Ensures embeddings are handled uniformly across the system.
    *   **Robustness:** Prevents errors caused by malformed or mismatched vectors.
    *   **DRY:** Reduces code duplication.
    *   **Configurability:** Allows setting the target dimension and alignment strategy centrally.
*   **Impact:** Components must now use `GeometryManager` or the validation utilities for embedding operations. Requires careful configuration of `embedding_dim` and `alignment_strategy`. `MemoryVectorIndex` still performs its own final alignment to strictly match the FAISS dimension.

---

## ADL-007: Atomic Writes for Critical Files

*   **Date:** ~2025-04-02 (Phase 5.8 Stability)
*   **Status:** Implemented
*   **Decision:** Implement atomic writes for critical JSON files (`memory_index.json`, `faiss_index.bin.mapping.json`, stats/log files) using a "write to temp file + `os.replace` (or `shutil.move`)" pattern. Encapsulate this logic in a `MemoryPersistence.safe_write_json` static utility method.
*   **Context:** System crashes or interruptions during file writes could lead to corrupted index, mapping, or state files, preventing successful restarts.
*   **Rationale:** `os.replace`/`shutil.move` are generally atomic operations on most filesystems. This pattern ensures the original file is only overwritten once the new data has been successfully written entirely to a temporary file, minimizing the window for corruption.
*   **Impact:** Slightly more complex file writing logic. Requires careful handling of temporary filenames and cleanup on failure. Increases reliability significantly.

---

## ADL-008: Assembly Synchronization via Timestamp (`vector_index_updated_at`)

*   **Date:** ~2025-04-03 (Phase 5.8 Stability)
*   **Status:** Implemented
*   **Decision:** Add a `vector_index_updated_at` timestamp (nullable `datetime`) field to `MemoryAssembly`. This timestamp is updated *only* upon successful insertion or update of the assembly's composite embedding into the `MemoryVectorIndex`. Retrieval boosting logic (`_activate_assemblies`) will only consider assemblies "synchronized" (and thus eligible for boosting) if this timestamp is non-null and within a configurable `max_allowed_drift_seconds`.
*   **Context:** The composite embedding of an assembly changes when members are added/removed. The update to the vector index might be delayed or fail (handled by the retry queue). Boosting retrieval based on a potentially stale embedding in the index is undesirable.
*   **Rationale:**
    *   **Consistency:** Ensures boosting only uses up-to-date assembly representations.
    *   **Graceful Degradation:** Allows retrieval to continue (without boost from affected assemblies) even if index updates are pending.
    *   **Observability:** Provides a clear indicator of an assembly's sync status via API (`/assemblies`).
*   **Impact:** Adds complexity to the assembly update workflow (must update timestamp on *successful* index op). Requires careful handling of timezone-aware datetimes for drift calculation. The `max_allowed_drift_seconds` config needs tuning.

---

## ADL-009: Internal Vector Update Retry Queue

*   **Date:** ~2025-04-04 (Phase 5.8 Stability / Refactor)
*   **Status:** Implemented
*   **Decision:** Implement an internal `asyncio.Queue` (`_pending_vector_updates`) and a background task (`_vector_update_retry_loop`) within `SynthiansMemoryCore` to handle failed vector index operations (add, update, remove for both memories and assemblies). Deprecate the external `AssemblySyncManager`.
*   **Context:** Vector index operations (especially involving FAISS/GPU) can fail transiently. We needed a robust, non-blocking way to ensure eventual consistency. The `AssemblySyncManager` was an external component adding complexity.
*   **Rationale:**
    *   **Resilience:** Automatically retries failed operations.
    *   **Decoupling:** Core processing logic (`process_new_memory`, `_update_assemblies`) doesn't block on index failures.
    *   **Centralization:** Integrates retry logic directly into the main core class, simplifying the overall architecture compared to the external manager.
    *   **Observability:** Pending queue size is exposed via `/stats`.
*   **Impact:** Increases complexity within `SynthiansMemoryCore`. Requires careful management of the background retry task lifecycle (startup, shutdown, cancellation). The queue size needs monitoring. Failed items might be requeued multiple times up to a limit.

---

## ADL-010: Append-Only Merge Log (`MergeTracker`)

*   **Date:** ~2025-04-05 (Phase 5.9 Implementation)
*   **Status:** Implemented
*   **Decision:** Implement assembly merge tracking using an append-only JSON Lines (`.jsonl`) file (`merge_log.jsonl`). Log distinct events for "merge creation" (with source/target IDs, similarity, threshold) and "cleanup status update" (referencing the creation event ID, indicating "completed" or "failed"). API endpoints needing the *current* status will reconcile these events on read.
*   **Context:** Needed a robust way to log the history of assembly merges for the Phase 5.9 explainability features (`/explain_merge`, `/diagnostics/merge_log`). Updating a single record for cleanup status could lead to corruption or race conditions.
*   **Rationale:**
    *   **Robustness:** Append-only writes are less prone to corruption than file updates.
    *   **Atomicity:** Each event log is a single atomic write.
    *   **History:** Preserves the full history of status changes if needed later.
    *   **Simplicity (Write):** Simplifies the logging code within the merge/cleanup logic.
*   **Impact:** Requires reconciliation logic within `MergeTracker` or the API route handler (`/diagnostics/merge_log`) to determine the *latest* status for a given merge event when queried. Log file requires size management (rotation).

---

## ADL-011: Strict Allow-List for Runtime Config Exposure

*   **Date:** ~2025-04-05 (Phase 5.9 Implementation)
*   **Status:** Implemented
*   **Decision:** Expose runtime configuration via the `GET /diagnostics/runtime/config/{service_name}` endpoint, but **only** return key-value pairs explicitly defined in a predefined `SAFE_CONFIG_KEYS_*` allow-list within the API route handler (`diagnostics_routes.py`).
*   **Context:** Providing diagnostic visibility into runtime configuration is useful, but exposing sensitive values (API keys, passwords, internal paths) is a security risk.
*   **Rationale:** **Security:** Prevents accidental leakage of sensitive configuration data through the API. Provides controlled transparency.
*   **Impact:** Requires developers to explicitly maintain the allow-lists for each service (`memory-core`, `neural-memory`, `cce`) as configuration evolves. New "safe" config options must be added to the list to be visible.

---

*This log will be updated as further significant architectural decisions are made.*