Okay, here is the **complete documentation set** reflecting the finalized implementation plan (v3) for **Phase 5.8: Memory Assembly Stabilization & Integration**.

This includes updates to existing files and reflects the focus on stability, observability, consistency, and graceful degradation.

---
---

# **`docs/README.md` (Top Level)**

```markdown
# Synthians Cognitive Architecture - Documentation

Welcome to the documentation for the Synthians Cognitive Architecture, a system designed to emulate aspects of human memory and cognition.

## Overview

This documentation provides comprehensive details on the system's architecture, its core components (Memory Core, Neural Memory, Context Cascade Engine), the underlying APIs, and usage guidelines.

**Key Concepts (Post Phase 5.8):**

*   **Bi-Hemispheric Model:** Interaction between episodic/declarative memory (Memory Core) and adaptive associative memory (Neural Memory).
*   **Memory Assemblies:** Stable, persistent groups of related memories with composite embeddings, enhancing contextual retrieval.
*   **QuickRecal:** Dynamic relevance score for memories, influenced by factors like recency, emotion, and surprise feedback.
*   **Surprise Feedback:** Neural Memory signals novelty (loss, grad_norm) to boost corresponding memory relevance in the Core.
*   **Performance-Aware Adaptation (Phase 5+):** System dynamically selects optimal processing variants (MAC, MAG, MAL) based on performance and context.
*   **Vector Index Reliability:** Robust FAISS (`IndexIDMap`) integration with diagnostics, consistency checks, and graceful handling of failures.
*   **Asynchronous Processing:** Built with `asyncio` for efficient I/O.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** High-level overview, principles, Bi-Hemispheric model, Assembly integration.
*   **[Component Guide](./COMPONENT_GUIDE.md):** Detailed breakdown of Memory Core, Neural Memory, CCE, Tools, Testing.
*   **[API Reference & Client Usage](./api/README.md):** HTTP APIs and Python client library.
    *   [API Reference](./api/API_REFERENCE.md)
    *   [Client Usage Guide](./api/client_usage.md)
*   **[Guides](./guides/README.md):** Setup, development, configuration, tooling.
*   **[Architecture Changes](./architechture-changes.md):** Log of significant architectural decisions.
*   **[Changelog](./CHANGELOG.md):** Chronological list of changes.

## Getting Started

1.  Review the **[Architecture](./ARCHITECTURE.md)**.
2.  Explore the **[Component Guide](./COMPONENT_GUIDE.md)**.
3.  Consult the **[API Reference & Client Usage](./api/README.md)**.
4.  See the **[Guides](./guides/README.md)** for setup/development.

*This documentation is actively maintained alongside the codebase.*
```

---

# **`docs/core/README.md`**

```markdown
# Synthians Memory Core Documentation

This directory contains detailed documentation specifically for the `synthians_memory_core` package, the heart of the Synthians memory system.

## Core Components & Concepts

*   [**Architecture**](./Architecture.md): Detailed internal architecture of the Memory Core, component interactions, and data flow, including Memory Assemblies.
*   [**Memory Structures**](./memory_structures.md): Definition of `MemoryEntry` and `MemoryAssembly` data classes. *(Implied content based on code)*
*   [**Persistence**](./persistence.md): How memories and assemblies are saved to and loaded from disk, including the `memory_index.json` structure.
*   [**Vector Index (FAISS)**](./vector_index.md): Implementation details of the FAISS `IndexIDMap` integration, including async operations, persistence, validation, and diagnostics.
*   [**Embedding Handling**](./embedding_handling.md): System-wide strategy for managing different embedding dimensions and ensuring vector validity.
*   [**Geometry Management**](./geometry.md): Role of the `GeometryManager` in handling vector math, normalization, alignment, and different geometric spaces.
*   [**QuickRecall Scoring**](./quickrecal.md): Explanation of the `UnifiedQuickRecallCalculator` and the factors influencing memory relevance.
*   [**Emotional Intelligence**](./emotion.md): Details on the `EmotionAnalyzer` and `EmotionalGatingService`.
*   [**Metadata Synthesis**](./metadata.md): How `MetadataSynthesizer` enriches memories.
*   [**API & Verification**](./API.md): *(Link to main API Ref)* | [API Verification](./API_Verification.md).
*   [**Development Guide**](./Development.md): Guidelines for contributing to the Memory Core.
*   [**Configuration**](../guides/CONFIGURATION_GUIDE.md): *(Link to main Config Guide)*
*   [**Stability & Repair**](./STABILITY_IMPROVEMENTS.md): Overview of recent stability fixes, especially for vector index and assemblies.

## Phase 5.8 Highlights

This version incorporates **Phase 5.8: Memory Assembly Stabilization & Integration**, introducing:

*   Stable creation, persistence, and indexing of `MemoryAssembly` objects.
*   Integration of assemblies into the retrieval pipeline for contextual relevance boosting.
*   Robust consistency mechanisms (`vector_index_updated_at` timestamp) between assemblies and the vector index.
*   Graceful handling of failed index updates via a pending queue.
*   Enhanced diagnostics for assemblies and index state via `/stats` and new `/assemblies` endpoints.
*   Optional, configurable assembly lifecycle management (pruning, merging).
*   Improved vector index validation and reliability.

Refer to the specific documents for detailed implementation insights.
```

---

# **`docs/core/Architecture.md` (Updated Sections)**

```markdown
# Synthians Memory Core - Architecture

*(Existing Introduction)* ...

## System Overview (Updated)

Synthians Memory Core is a modular system managing memory entries and assemblies. It integrates vector search (FAISS), relevance scoring (QuickRecall), emotional intelligence, **stable Memory Assemblies**, and robust persistence. Phase 5.8 stabilizes assemblies and their interaction with the vector index, adding consistency checks and graceful degradation for failed updates.

## Component Architecture (Updated Diagram - ASCII for simplicity)

```
+-------------------------------------------------------------+
|                   Synthians Memory Core                     |
| +---------------------------------------------------------+ |
| |                    Orchestration Layer                  | |
| | +-----------------------------------------------------+ | |
| | | SynthiansMemoryCore (Main Class)                    | | |
| | |  - Manages components, API calls, background tasks  | | |
| | |  - Handles Assembly creation/update/activation      | | |
| | |  - Integrates Retrieval Boosting                    | | |
| | |  - Manages Pending Vector Update Queue              | | |
| | +--------------------------+--------------------------+ | |
| +---------------------------------------------------------+ |
|      |           |           |           |           |      |
|      v           v           v           v           v      |
| +-----------+ +-----------+ +-----------+ +-----------+ +-----------+
| | Geometry  | | QuickRecall | | Emotional | | Persistence | | Vector    |
| | Manager   | | Calculator| | Intel.    | | Layer     | | Index     |
| | (Vectors) | | (Scoring) | | (Gating)  | | (Save/Load) | | (FAISS)   |
| +-----------+ +-----------+ +-----------+ +-----------+ +-----------+
|      |           |           |           |           |
|      +-----------+-----------+-----------+-----------+------> Filesystem
|                  |           |           |
|                  v           v           v
|            +-----------+ +-----------+ +-----------+
|            | Memory    | | Memory    | | Adaptive  |
|            | Structures| | Assemblies| | Components|
|            | (Entry)   | | (Groups)  | |(Threshold)|
|            +-----------+ +-----------+ +-----------+
+-------------------------------------------------------------+
```

### Core Components (Updated Descriptions)

*   **Memory Structures:** Defines `MemoryEntry` and **`MemoryAssembly` (now including `vector_index_updated_at`)**.
*   **Memory Assemblies:** Manages groups of related memories. **Crucially interacts with Persistence and Vector Index for storing composite embeddings and ensuring consistency.**
*   **Vector Index (FAISS):** **Simplified wrapper around `IndexIDMap`**. Handles async CRUD, persistence, validation, diagnostics. **Consistency with assemblies managed via timestamp (`vector_index_updated_at`) and pending queue.**
*   **Persistence Layer:** Handles async save/load for **both `MemoryEntry` and `MemoryAssembly`**. Manages `memory_index.json` which now includes assembly info.
*   **SynthiansMemoryCore (Main Class):** Orchestrates all internal flows. **Manages the `_pending_vector_updates` queue and retry logic for graceful degradation.** Integrates assembly activation into retrieval boosting.

## Data Flow (Updated for Assemblies & Consistency)

### Memory Processing Flow (Updated)

```
Input -> Validate/Gen Embedding (GeometryMgr) -> QuickRecall Score -> Emotion Analysis
   |
   v
Metadata Synthesis -> Create MemoryEntry -> Store in Cache & Mark Dirty (Core)
   |                                           |
   +-------------------------------------------+
   |                                           v
   +-----> Save Memory (Persistence) <------ Add to Dirty Set (Core)
   |                                           |
   +-----> Add to Vector Index (VectorIndex) <-+--(Success?)--> Update Status
   |         (If fails, add to Pending Queue) --------> Retry Loop (Core)
   v
Update/Create Assemblies (Core) -> Calculate Composite -> Mark Assembly Dirty
   |                                  (GeometryMgr)          (Core)
   |                                                         |
   +---------------------------------------------------------+
   |                                                         v
   +--------> Save Assembly (Persistence) <-----------------+
   |                                                         |
   +--------> Update/Add Assembly Vector (VectorIndex) <-----+--(Success?)--> Update Timestamp (Assembly)
             (If fails, add to Pending Queue) ------------------> Retry Loop (Core)

```

1.  *(Steps 1-5 as before)*
2.  **Store MemoryEntry:** Stored in cache (`_memories`), marked dirty. Async save via `Persistence`.
3.  **Index Memory:** Embedding added to `VectorIndex`. **Failure queues retry.**
4.  **Update Assemblies:** Relevant assemblies identified. `add_memory` called (recalculates composite, sets `vector_index_updated_at=None`). Assembly marked dirty.
5.  **Index Assembly:** Async update/add of assembly composite embedding to `VectorIndex`. **Failure queues retry.**
6.  **Timestamp Sync:** On *successful* index update for assembly, `vector_index_updated_at` timestamp is set on the assembly object (under lock), marking it synchronized and ready for boosting. Assembly marked dirty again to save timestamp.

### Memory Retrieval Flow (Updated)

```
Query -> Gen/Validate Query Embedding (GeometryMgr) -> Activate Assemblies (Core)
   |          (Uses VectorIndex Search for "asm:*", checks timestamp)
   |                                                        |
   v                                                        v
Direct Vector Search (VectorIndex for "mem:*") <--- Store Activation Scores (Core)
   |
   v
Combine & Load Candidates (Core + Persistence) -> Calculate Relevance Score (Core)
   | (Incl. Base Similarity + Assembly Boost)              (GeometryMgr)
   v
Threshold Filter -> Emotional Gating -> Metadata Filter -> Sort & Return Top K
(Adaptive)        (Emotional Intel.)       (Core)          (Core)
```

1.  *(Steps 1-2 as before: Query -> Embedding)*
2.  **Activate Assemblies:** `_activate_assemblies` searches `VectorIndex` for relevant `asm:*` IDs. **Crucially, it only considers assemblies where `vector_index_updated_at` is not None (i.e., synchronized).** Stores activation scores.
3.  **Direct Search:** `VectorIndex` searches for relevant `mem:*` IDs.
4.  **Combine & Load:** Candidate IDs combined. Full memory data loaded (as dicts) using `get_memory_by_id_async`. Base `similarity` from direct search added.
5.  **Calculate Relevance:** Base `similarity` is **boosted** based on `max_activation` score from associated, *activated* assemblies. Result is `relevance_score`.
6.  *(Steps 6-8 as before: Filter by threshold, emotion, metadata; Sort by `relevance_score`; Return Top K)*

### Consistency & Degradation Flow

1.  **Assembly Update:** `add_memory` updates composite, sets `vector_index_updated_at = None`.
2.  **Index Update Attempt:** `_update_assemblies` calls `vector_index.update_entry` / `add`.
3.  **Success:** `vector_index_updated_at` timestamp is set on `MemoryAssembly` object. Assembly usable for boosting.
4.  **Failure:** Timestamp remains `None`. Update is added to `_pending_vector_updates` queue. Assembly *not used* for boosting (`_activate_assemblies` skips it).
5.  **Retry Loop:** Background task (`_vector_update_retry_loop`) retries operations from the queue. On success, it updates the assembly timestamp.
6.  **Diagnostics:** `/stats` endpoint shows `vector_index_pending_updates` count. Dashboard visualizes this.

*(Existing sections on Implementation Details, Integration Points, Performance, Security, Deployment remain largely the same but should be reviewed for consistency with the assembly changes)*.
```

---

# **`docs/core/vector_index.md` (Rewritten)**

```markdown
# Vector Index (FAISS) - Phase 5.8

The `synthians_memory_core.vector_index.MemoryVectorIndex` class provides a robust and asynchronous interface to the FAISS library for efficient vector similarity search. It is designed for stability and integration within the Synthians Memory Core.

## Core Design Principles (Phase 5.8)

1.  **Focus:** Primarily responsible for CRUD-like operations (Add, Search, Update, Remove) on vectors identified by string IDs and persisting the index state.
2.  **`IndexIDMap`:** Uses `faiss.IndexIDMap` internally to map user-provided string IDs (e.g., `mem_xyz`, `asm_abc`) to the 64-bit integer IDs required by FAISS base indexes. This allows for stable, non-sequential identifiers.
3.  **CPU-Centric `IndexIDMap`:** While the *base* index wrapped by `IndexIDMap` (e.g., `IndexFlatIP`) can potentially use the GPU for *search*, the `IndexIDMap` operations themselves (`add_with_ids`, `remove_ids`) are executed on the CPU due to FAISS limitations. GPU acceleration is primarily beneficial for searching large base indexes *without* `IndexIDMap`. For reliability with string IDs, we prioritize `IndexIDMap`.
4.  **Asynchronous Operations:** All methods involving potential I/O or significant computation (initialization, save, load, add, remove, update) are `async` and use internal locking (`asyncio.Lock`) and `asyncio.to_thread` to avoid blocking the main event loop.
5.  **Simplified Persistence:** Saves two files atomically: the FAISS index (`faiss_index.bin`) and the string-ID-to-numeric-ID mapping (`faiss_index.bin.mapping.json`).
6.  **Robust Initialization & Validation:** Includes a post-initialization check (`_post_initialize_check`) to verify the loaded/created index is usable (correct dimension, basic search works).
7.  **Diagnostic Focus:** `verify_index_integrity` provides diagnostic information only (count mismatch, map presence) without attempting repairs. Complex repair logic is externalized (e.g., `utils/vector_index_repair.py`).
8.  **Graceful Failure:** Methods return `True`/`False` or specific data structures, logging errors internally rather than raising exceptions for common operational failures (like adding an invalid vector).

## Key Component: `MemoryVectorIndex`

*   **Initialization:**
    *   Takes a config dict (`embedding_dim`, `storage_path`, `index_type`, `use_gpu` - affects *base* index search if not IDMap).
    *   `async initialize()`: Loads index and mapping from `storage_path`. If files don't exist or fail load, creates a new, empty `IndexIDMap`. Performs `_post_initialize_check`.
*   **Core Async Methods:**
    *   `add(id: str, embedding: np.ndarray) -> bool`: Validates embedding, calculates numeric ID, adds to index and mapping. Returns `True` on success.
    *   `remove_vector(id: str) -> bool`: Removes vector by string ID from index and mapping. Returns `True` if removed from FAISS.
    *   `update_entry(id: str, embedding: np.ndarray) -> bool`: Updates vector using remove-then-add pattern. Returns `True` on successful add.
    *   `search(query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]`: Validates query, performs search, converts numeric IDs back to string IDs, returns sorted list of `(id, similarity_score)`. (Note: Underlying FAISS search might block briefly).
*   **Persistence:**
    *   `async save() -> bool`: Atomically saves index `.bin` and mapping `.json`.
    *   `async load() -> bool`: Loads index and mapping. Included in `initialize`.
*   **Utilities:**
    *   `count() -> int`: Returns number of vectors currently in the FAISS index (`index.ntotal`).
    *   `verify_index_integrity() -> Tuple[bool, Dict]`: Returns consistency status and diagnostics.
    *   `reset() -> bool`: Clears the index and mapping.
    *   `get_stats() -> Dict`: Returns basic index statistics.

## ID Management

*   A deterministic hash (`hashlib.md5`) converts string IDs (`mem_...`, `asm:...`) to positive 64-bit integers suitable for `IndexIDMap`.
    ```python
    def _get_numeric_id(self, string_id: str) -> int:
        h = hashlib.md5(string_id.encode()).digest()
        return abs(int.from_bytes(h[:8], byteorder='little'))
    ```
*   The `id_to_index: Dict[str, int]` map stores this mapping in memory and is persisted to `faiss_index.bin.mapping.json`.

## Embedding Validation

*   Uses an internal `_validate_embedding` method before `add`/`update`/`search`.
*   Checks for `None`, correct type (`np.ndarray`), 1D shape.
*   Replaces `NaN`/`Inf` values with zeros and logs a warning.
*   **Aligns** vectors (pad/truncate) to match `self.embedding_dim`.
*   Ensures `np.float32` dtype.

## Configuration

*   `embedding_dim`: **Must match** the actual dimension of embeddings being stored.
*   `storage_path`: Directory where `.bin` and `.json` files are saved.
*   `index_type`: Base index metric (`L2`, `IP`, `Cosine`). `IP` or `Cosine` recommended for similarity search.
*   `use_gpu`: If `True` AND `IndexIDMap` is *not* used, attempts GPU acceleration for the base index search.

## Importance

Provides the core capability for fast semantic similarity search, essential for memory retrieval and assembly operations. The `IndexIDMap` ensures stable identification, and the async/validation features promote system stability.
```

---

# **`docs/core/persistence.md` (Updated)**

```markdown
# Memory Persistence

The `synthians_memory_core.memory_persistence.MemoryPersistence` class handles the asynchronous saving and loading of memory structures (`MemoryEntry`, `MemoryAssembly`) to/from the filesystem.

## Purpose

Ensures the state of the memory core (memories, assemblies, metadata) survives restarts and shutdowns, providing durability.

## Key Component: `MemoryPersistence`

*   **Functionality:**
    *   Provides asynchronous methods (`save_memory`, `load_memory`, `delete_memory`, **`save_assembly`**, **`load_assembly`**, **`delete_assembly`**, **`list_assemblies`**) using `aiofiles` or `asyncio.to_thread`.
    *   Saves individual `MemoryEntry` objects as separate JSON files in `storage_path/memories/`.
    *   **Saves `MemoryAssembly` objects as separate JSON files in `storage_path/assemblies/`.**
    *   Manages a central index file (`storage_path/memory_index.json`) mapping item IDs (`mem_*`, `asm_*`) to file paths and lightweight metadata.
    *   Uses atomic writes (temp file + rename) for safety.
*   **Integration:** Used by `SynthiansMemoryCore` for all disk operations related to memories and assemblies. Coordinates with `MemoryVectorIndex` during initialization and deletion.

## Storage Structure (Example - Post Phase 5.8)

```
<storage_path>/
├── memory_index.json        # Maps item_id -> {path, timestamp, type, ...}
├── memories/
│   ├── mem_<uuid_1>.json    # Complete MemoryEntry object
│   └── ...
├── assemblies/              # <--- NEW Directory
│   ├── asm_<uuid_a>.json    # Complete MemoryAssembly object
│   └── ...
└── vector_index/            # Managed by MemoryVectorIndex (Unchanged)
    ├── faiss_index.bin
    └── faiss_index.bin.mapping.json
```

## Memory Index Structure (`memory_index.json` - Updated)

The index now includes entries for both memories and assemblies, distinguished by the `type` field.

```json
{
  "mem_1234abcd": {
      "path": "memories/mem_1234abcd.json",
      "timestamp": "<iso-string>",
      "quickrecal": 0.75,
      "type": "memory"
  },
  "asm_wxyz_1": {
      "path": "assemblies/asm_wxyz_1.json",
      "timestamp": "<iso-string>",
      "type": "assembly",
      "name": "Project Alpha Notes"
  },
  // ... more entries
}
```

## Implementation Details

*   **Asynchronous I/O:** All file operations use `aiofiles` (preferred) or `asyncio.to_thread` for non-blocking behavior.
*   **Atomicity:** Safe writes using temporary files prevent data corruption during saves.
*   **Error Handling:** Includes `try...except` blocks for file operations, logging errors.
*   **Index Management:** The `memory_index.json` is loaded on initialization and saved atomically after modifications.

## Configuration

*   `storage_path`: Root directory for persistence.
*   `index_filename`: Name of the index file (default: `memory_index.json`).
*   `max_backups`: Number of backups for the index file.
*   `safe_write`: Enable/disable atomic writes (default: `True`).

## Importance

Provides data durability for the Memory Core. The asynchronous design ensures persistence operations don't block the main API service. The central index allows for efficient loading and discovery of persisted items.
```

---

# **`docs/guides/CONFIGURATION_GUIDE.md` (Updated)**

```markdown
# Synthians Cognitive Architecture: Configuration Guide

**Version:** 1.3 (Post Phase 5.8)
**Date:** *03/04/2025*

## 1. Overview

*(Existing Overview)*

## 2. Synthians Memory Core Configuration (`synthians_memory_core`)

*(Existing Intro)*

### 2.1. Core Parameters (`SynthiansMemoryCore` config dict)

| Parameter                       | Type    | Default                        | Description                                                                                             |
| :------------------------------ | :------ | :----------------------------- | :------------------------------------------------------------------------------------------------------ |
| `embedding_dim`                 | int     | 768                            | **CRITICAL:** Dimension of embeddings used system-wide. Must match model.                             |
| `geometry`                      | str     | "hyperbolic"                   | Geometric space: "euclidean", "hyperbolic", "spherical", "mixed". Affects similarity/distance.          |
| `hyperbolic_curvature`          | float   | -1.0                           | Curvature for hyperbolic geometry (`< 0`).                                                            |
| `storage_path`                  | str     | "/app/memory/stored/synthians" | **CRITICAL:** Base path for all persistent data (memories, assemblies, indexes).                      |
| `vector_index_type`             | str     | "Cosine"                       | Base FAISS index metric: "L2", "IP", "Cosine". (`IndexIDMap` uses this internally).                   |
| `use_gpu`                       | bool    | False                          | **(Experimental)** Attempt GPU for FAISS base index search (Not `IndexIDMap` ops). Requires `gpu_setup.py`. |
| `persistence_interval`          | float   | 60.0                           | Seconds between background persistence saves. `0` disables loop.                                        |
| `decay_interval`                | float   | 3600.0                         | Seconds between QuickRecal decay checks. `0` disables.                                                |
| `prune_check_interval`          | float   | 600.0                          | Seconds between memory/assembly pruning checks. `0` disables.                                         |
| `persistence_batch_size`        | int     | 100                            | Max items to save in one persistence batch.                                                           |
| **`diagnostic_mode`**           | bool    | False                          | **(New 5.8)** Enable extended diagnostics in API responses (e.g., embedding snippets).                |
| **`check_index_on_retrieval`**  | bool    | False                          | Run quick vector index integrity check before each retrieval (can add latency).                     |
| **`index_check_interval`**      | float   | 3600.0                         | Seconds between periodic background vector index integrity checks.                                    |
| **`vector_index_retry_interval`** | float | 60.0                           | **(New 5.8)** Seconds between retries for failed vector index updates.                                |
| **`vector_index_max_pending`**  | int     | 1000                           | **(New 5.8)** Max failed updates to keep in the retry queue.                                          |

### 2.2. Memory Assembly Parameters

| Parameter                       | Type    | Default   | Description                                                                          |
| :------------------------------ | :------ | :-------- | :----------------------------------------------------------------------------------- |
| `assembly_threshold`            | float   | 0.75      | Min similarity for memory to join/seed an assembly (0-1).                            |
| `max_assemblies_per_memory`     | int     | 3         | Max assemblies a memory can join.                                                    |
| **`assembly_activation_threshold`** | float | 0.6       | **(New 5.8)** Min similarity for query to activate an assembly for boosting (0-1).   |
| **`assembly_boost_mode`**       | str     | "additive"| **(New 5.8)** How boost is applied: "additive", "multiplicative".                    |
| **`assembly_boost_factor`**     | float   | 0.2       | **(New 5.8)** Factor scaling activation score into relevance boost (e.g., 0.0-1.0). |

### 2.3. Assembly Lifecycle Management Parameters (Optional)

| Parameter                       | Type    | Default | Description                                                                       |
| :------------------------------ | :------ | :------ | :-------------------------------------------------------------------------------- |
| **`enable_assembly_pruning`**   | bool    | True    | **(New 5.8)** Enable automatic pruning of assemblies.                             |
| `assembly_prune_min_memories`   | int     | 2       | Min memories required to avoid pruning.                                           |
| `assembly_prune_max_idle_days`  | float   | 30.0    | Max days inactive before pruning.                                                 |
| `assembly_prune_max_age_days`   | float   | 90.0    | Max age before pruning (unless sufficiently activated).                           |
| `assembly_prune_min_activation_level` | int | 5       | Min activations required to avoid age-based pruning.                            |
| **`enable_assembly_merging`**   | bool    | False   | **(New 5.8)** Enable automatic merging of similar assemblies (Default OFF).         |
| `assembly_merge_threshold`      | float   | 0.85    | Min similarity between assembly composites to trigger merge (0-1).                |
| `assembly_max_merges_per_run`   | int     | 10      | Max merges per pruning cycle.                                                     |

### 2.4. Component-Specific Parameters

*(Existing sections on GeometryManager, QuickRecallCalculator, etc. remain valid. Add notes about `embedding_dim` inheritance.)*

### 2.5. API Server Environment Variables

| Variable            | Default                       | Description                                 |
| :------------------ | :---------------------------- | :------------------------------------------ |
| `HOST`              | "0.0.0.0"                     | Host address for the API server.            |
| `PORT`              | "5010"                        | Port for the API server.                    |
| `LOG_LEVEL`         | "INFO"                        | Logging level (DEBUG, INFO, WARNING, ERROR). |
| `EMBEDDING_MODEL`   | "all-mpnet-base-v2"           | Sentence Transformer model to use.        |
| `STORAGE_PATH`      | "/app/memory/stored/synthians"| Overrides core config `storage_path`.       |
| `DISABLE_BACKGROUND`| "false"                       | Set to "true" to disable bg loops entirely. |

*(Sections for Neural Memory Server & CCE Configuration remain separate)*

## 3. Recommended Configurations

*(Existing sections remain valid)*

## 4. Important Notes (Updated)

*   **Embedding Dimension (`embedding_dim`):** MUST be consistent across Memory Core config, `GeometryManager`, `VectorIndex`, and the embedding model specified by `EMBEDDING_MODEL`.
*   **Storage Path:** Ensure the `storage_path` is writable and correctly mapped if using Docker volumes. The structure (`memories/`, `assemblies/`, `vector_index/`) will be created automatically.
*   **GPU Usage:** `use_gpu=True` only attempts GPU for base FAISS index search *if* `IndexIDMap` is *not* used (which it is by default for stability). Requires `gpu_setup.py` to have run successfully.
*   **Assembly Lifecycle:** Merging is disabled by default due to performance implications. Enable cautiously and monitor `/stats`. Pruning is enabled by default.
```

---

# **`docs/api/API_REFERENCE.md` (Updated)**

```markdown
# Synthians Cognitive Architecture: API Reference

**Date:** *03/04/2025*
**Version:** 1.1.0 (Post Phase 5.8)

*(Existing Intro, Table of Contents)*

---

## 1. Synthians Memory Core API

**Base URL:** `http://localhost:5010` (Default)

*(Existing Description)*

---

*(Endpoints: Root, Health Check - Unchanged)*

---

### Get Statistics

*   **Method:** `GET`
*   **Path:** `/stats`
*   **Description:** Retrieves detailed statistics about the Memory Core system, including **assembly counts, pending vector updates**, and vector index status.
*   **Response (Success - Updated Example):**
    ```json
    {
      "success": true,
      "core_stats": { // Renamed from api_server for clarity
        "total_memories": 500,
        "total_assemblies": 50, // Added
        "dirty_memories": 15,   // Added
        "vector_index_pending_updates": 2, // Added
        "initialized": true,
        "uptime_seconds": 1234.56 // Moved from api_server
      },
      "persistence_stats": { // Example structure
          "total_indexed_items": 550, // Memories + Assemblies in index file
          // ... other persistence stats ...
      },
      "quick_recal_stats": { /* ... QuickRecal stats ... */ },
      "threshold_stats": { /* ... Adaptive Threshold stats ... */ },
      "vector_index_stats": { // Updated structure
        "count": 550, // Total items (mem + asm) in FAISS index
        "id_mappings": 550,
        "embedding_dim": 768,
        "index_type": "Cosine", // Base index type
        "is_gpu": false,
        "is_id_map": true
      },
      "assemblies": { // Added detailed assembly stats
          "count": 50,
          "avg_memory_count": 10.5,
          "total_activations": 1230,
          "avg_activation_level": 0.65
      }
      // Removed redundant memory.total_memories, etc.
    }
    ```
*   *(Error Response - Unchanged)*

---

*(Endpoints: Process Memory, Retrieve Memories (Note: Response now includes `relevance_score`, `assembly_activation`, `assembly_boost`), Generate Embedding, Calculate QuickRecal, Analyze Emotion, Provide Feedback, Detect Contradictions, Process Transcription, Get Memory by ID - Largely Unchanged, but ensure examples reflect `relevance_score`)*

---

### List Assemblies

*   **Method:** `GET`
*   **Path:** `/assemblies`
*   **Description:** Lists basic information about all known memory assemblies.
*   **Response (Success - Updated Example):**
    ```json
    {
      "success": true,
      "assemblies": [
        {
          "assembly_id": "asm_abc123",
          "name": "Project Alpha Notes",
          "memory_count": 15,
          "last_activation": "2025-04-02T10:30:00Z", // ISO Format
          "vector_index_status": "synchronized" // Added
        },
        {
          "assembly_id": "asm_def456",
          "name": "Quantum Physics Concepts",
          "memory_count": 8,
          "last_activation": "2025-04-01T18:00:00Z",
          "vector_index_status": "pending_update" // Added
        }
        // ... more assemblies
      ],
      "count": 50 // Total number of assemblies
    }
    ```
*   *(Error Response - Unchanged)*

---

### Get Assembly Details

*   **Method:** `GET`
*   **Path:** `/assemblies/{assembly_id}`
*   **Description:** Retrieves detailed information about a specific memory assembly, including a sample of its member memories and its synchronization status.
*   **Path Parameter:** `assembly_id` (string) - The unique ID of the assembly (e.g., `asm_abc123`).
*   **Response (Success - Updated Example):**
    ```json
    {
      "success": true,
      "assembly": {
          "assembly_id": "asm_abc123",
          "name": "Project Alpha Notes",
          "description": "Notes related to Project Alpha planning",
          "memory_count": 15,
          "creation_time": "2025-04-01T09:00:00Z",
          "last_access_time": "2025-04-02T11:00:00Z",
          "last_activation": "2025-04-02T10:30:00Z",
          "activation_count": 150,
          "activation_level": 0.85,
          "keywords": ["alpha", "planning", "roadmap", "budget", "..."], // Sample
          "memory_ids": ["mem_1...", "mem_2...", "..."], // Sample or all
          "composite_embedding_norm": 0.998, // Example detail
          "vector_index_status": "synchronized", // Added: synchronized | pending_update | error
          "vector_index_updated_at": "2025-04-02T09:45:10Z" // Added: ISO timestamp or null
          // Potentially add emotion profile summary
      },
      "error": null
    }
    ```
*   *(Error Response (Not Found) - Unchanged)*

---

### (Meta) Phase Feedback Endpoint (Stub)

*   **Method:** `POST`
*   **Path:** `/feedback/phase/{phase_id}`
*   **Description:** Endpoint stub for collecting feedback on development phases (primarily for internal tooling).
*   **Path Parameter:** `phase_id` (string) - Identifier for the phase (e.g., "5.8").
*   **Request Model:**
    ```json
    {
      "status": "success" | "failure" | "partial",
      "notes": "Optional string with feedback notes.",
      "metrics": { /* Optional structured metrics */ }
    }
    ```
*   **Response (Success):**
    ```json
    {
      "message": "Feedback received for phase {phase_id}",
      "timestamp": "<iso-string>"
    }
    ```

*(Existing sections on Trainer Integration Endpoints, Common Error Handling remain valid)*
```

---

# **`docs/api/client_usage.md` (Updated)**

```markdown
# Memory Core Python Client Usage Guide

*(Existing Intro, Installation)*

## 2. Basic Operations

*(Existing Storing Memory, Retrieving Memories - Update examples to show/mention `relevance_score` which includes boost)*

### Retrieving a Specific Memory by ID

*(Update method call example if API path changed slightly - likely unchanged)*

## 3. Utility Endpoints

*(Unchanged)*

## 4. Advanced Features

*(Existing Feedback, Contradiction Detection - Unchanged)*

### **(NEW)** Working with Assemblies

```python
async def assembly_example(client: SynthiansClient):
    # --- Listing Assemblies ---
    try:
        list_resp = await client.list_assemblies() # Assuming client method is added
        if list_resp.get("success"):
            print(f"\nFound {list_resp.get('count')} assemblies:")
            for asm in list_resp.get("assemblies", [])[:5]: # Print first 5
                print(f"  - ID: {asm.get('assembly_id')}, Name: {asm.get('name')}, "
                      f"Members: {asm.get('memory_count')}, Sync Status: {asm.get('vector_index_status')}")
        else:
            print(f"\nFailed to list assemblies: {list_resp.get('error')}")
    except AttributeError:
        print("\nSkipping list_assemblies example: client method not found.")
    except Exception as e:
        print(f"\nError listing assemblies: {e}")


    # --- Getting Assembly Details ---
    # Assuming we got an ID from the list call or know one
    known_assembly_id = "asm_abc123" # Replace with a real ID if possible, or skip
    if known_assembly_id:
        try:
            detail_resp = await client.get_assembly_details(known_assembly_id) # Assuming client method is added
            if detail_resp.get("success") and detail_resp.get("assembly"):
                print(f"\nDetails for Assembly {known_assembly_id}:")
                print(json.dumps(detail_resp["assembly"], indent=2, default=str))
            elif not detail_resp.get("success"):
                 print(f"\nFailed to get assembly details for {known_assembly_id}: {detail_resp.get('error')}")
            else: # Success false, but no error (e.g., not found)
                 print(f"\nAssembly {known_assembly_id} not found.")
        except AttributeError:
            print(f"\nSkipping get_assembly_details example: client method not found.")
        except Exception as e:
            print(f"\nError getting assembly details: {e}")
```

*(Existing Transcription Processing - Unchanged)*

## 5. Error Handling

*(Unchanged)*

## 6. Best Practices

*(Add a note about checking `vector_index_status` for assemblies if critical decisions depend on their indexed state.)*
```

---

# **`docs/guides/CONFIGURATION_GUIDE.md` (Already Updated Above)**

---

# **`docs/guides/implementation_guide.md` (Updated)**

```markdown
# Bi-Hemispheric Cognitive Architecture: Implementation Guide

*(Existing Intro, System Requirements)*

## Component Deployment

*(Existing Docker Compose/Manual Deployment info - Add note about running `gpu_setup.py` before manual server starts if GPU is intended)*

## Configuration (Updated)

Refer to the detailed [Configuration Guide](./CONFIGURATION_GUIDE.md) for all parameters. Key environment variables include:

*   `MEMORY_CORE_URL`, `NEURAL_MEMORY_URL` (for CCE)
*   `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `STORAGE_PATH` (for Memory Core)
*   `TITANS_VARIANT` (for CCE - controls attention variant)
*   **Assembly Configs (in Memory Core config dict):** `assembly_threshold`, `assembly_activation_threshold`, `assembly_boost_*`, `enable_assembly_pruning`, `enable_assembly_merging`, etc.

## Component Integration (Updated)

### GeometryManager
*(Unchanged)*

### Vector Index Management (Updated)
The `MemoryVectorIndex` now focuses on reliable `IndexIDMap` operations (CPU default) and persistence.
*   Initialization includes a post-check (`_post_initialize_check`).
*   Handles async operations with internal locking.
*   Provides diagnostics via `verify_index_integrity`. Complex repair is external.
*   **Consistency with assemblies** is managed via `vector_index_updated_at` timestamp and a **pending update queue** in `SynthiansMemoryCore`.

```python
# In Memory Core:
# Check consistency before retrieval (optional)
is_consistent, diagnostics = await memory_core.vector_index.verify_index_integrity()
if not is_consistent:
    logger.warning(f"Index inconsistency: {diagnostics}")
    # Optionally trigger repair: await memory_core.repair_index()

# Add memory embedding (failure adds to pending queue)
await memory_core.vector_index.add(mem_id, embedding)

# Update assembly embedding (failure adds to pending queue)
await memory_core.vector_index.update_entry(f"asm:{asm_id}", composite_embedding)
```

### Metadata Enrichment
*(Unchanged)*

## Robust Error Handling (Updated)

### Embedding Validation
Use `geometry_manager._validate_vector()` or `utils.embedding_validators.validate_embedding()` extensively on external inputs and before indexing/comparison.

### Dimension Mismatch Handling
`GeometryManager` handles alignment based on `alignment_strategy` (default 'truncate'). `MemoryVectorIndex` ensures internal FAISS dimension consistency.

### **(NEW)** Vector Index Failures (Graceful Degradation)
*   Failures during `vector_index.add` or `update_entry` (e.g., transient disk errors, FAISS issues) are caught internally.
*   The failed operation (`(item_id, operation_type, embedding_list)`) is added to `SynthiansMemoryCore._pending_vector_updates` queue.
*   A background task (`_vector_update_retry_loop`) periodically retries queued operations.
*   `/stats` endpoint shows the number of pending updates (`vector_index_pending_updates`).
*   Retrieval boosting via assemblies (`_activate_assemblies`) checks the `vector_index_updated_at` timestamp, skipping assemblies whose index update is pending/failed, ensuring only consistent assemblies contribute.

*(Existing Performance Optimization, Deployment Example - Unchanged)*

## GPU Acceleration Notes (Updated)
*   Run `python gpu_setup.py` before starting services to ensure correct FAISS package (CPU or GPU) is installed.
*   Memory Core config `use_gpu=True` *only* affects base FAISS index search *if* `IndexIDMap` is not used (default uses `IndexIDMap` on CPU for stability). Search performance gains might be limited when using `IndexIDMap`.
```

---

# **`docs/guides/tooling_guide.md` (Updated)**

```markdown
# System Tooling Guide

*(Existing Intro)*

## Available Tools & Utilities

### 1. Vector Index Verification (`MemoryVectorIndex.verify_index_integrity`)
*(Updated Description)*
*   **Functionality:** Performs **diagnostic checks** on the consistency between the FAISS index file (`.bin`) and the string ID-to-int64 ID mapping (`.json`). Checks counts and basic structure. **Does not perform repairs.** Detects mismatches.
*   **Usage:** Called internally on init/periodically. Can be exposed via admin endpoint (e.g., `/admin/verify_index`). Returns `(bool_consistent, dict_diagnostics)`.

### 2. **(NEW/Externalized)** Vector Index Repair (`utils/vector_index_repair.py`)
*   **Location:** Utility module `synthians_memory_core.utils.vector_index_repair` or potentially a standalone script `tools/repair_vector_index.py`.
*   **Functionality:** Provides functions like `repair_vector_index` that attempt to fix inconsistencies diagnosed by `verify_index_integrity`. Strategies might include:
    *   Rebuilding the `.json` mapping from memory persistence files (if index `.bin` seems ok but mapping is bad).
    *   Rebuilding the FAISS `.bin` index from memory persistence files (if index `.bin` is corrupt but mapping/memories are ok). Requires fetching embeddings.
*   **Usage:** Typically run offline as a maintenance task when `verify_index_integrity` reports significant issues. Requires access to memory files and potentially a way to fetch/regenerate embeddings.

*(Existing sections on Index ID Mapping Reconstruction, Memory Index Reconstruction, FAISS Index Migration - Update to reflect they might be part of the external repair tool)*

### 3. Diagnostic API Endpoints (Updated)
*   **Location:** Memory Core API (`api/server.py`).
*   **Functionality:**
    *   `/health`, `/stats` (now includes assembly stats and **pending vector update count**).
    *   **`/assemblies`**, **`/assemblies/{assembly_id}`** (New endpoints for assembly inspection, including `vector_index_status`).
    *   **(Optional)** `/admin/verify_index`: Trigger `verify_index_integrity`.
    *   **(Optional)** `/admin/trigger_retry_loop`: Manually trigger the pending vector update retry loop.
*   **Usage:** Monitoring, debugging, administration (secure appropriately).

### 4. Backup & Restore Scripts
*(Unchanged)*

### 5. **(NEW)** Dashboard (`tools/variant_diagnostics_dashboard.py` or similar)
*   **Functionality:** Monitors `/stats` (including assembly and pending queue stats) and potentially `/assemblies` endpoint to provide a real-time view of system health and assembly status.
*   **Usage:** Real-time monitoring and troubleshooting.

*(Existing Best Practices - Update to mention monitoring pending queue)*
```

---

# **`docs/CHANGELOG.md` (Add Entry)**

```markdown
## [Unreleased] - Phase 5.8

### Added
- **Memory Assemblies:** Stable implementation with persistence (`assemblies/` dir, `memory_index.json` update).
- **Assembly Indexing:** Composite embeddings stored and updated in `MemoryVectorIndex` using `asm:` prefix.
- **Assembly Retrieval Boosting:** Activated assemblies now boost relevance scores of member memories during retrieval.
- **Consistency Mechanism:** Introduced `MemoryAssembly.vector_index_updated_at` timestamp to track sync status with vector index. Retrieval boosting respects this status.
- **Graceful Degradation:** Implemented `_pending_vector_updates` queue and `_vector_update_retry_loop` in `SynthiansMemoryCore` to handle failed vector index operations without blocking.
- **Diagnostics:** Enhanced `/stats` endpoint with assembly details and pending update count. Added `/assemblies` and `/assemblies/{id}` endpoints. Updated dashboard tooling (`tools/variant_diagnostics_dashboard.py`) to reflect new stats.
- **Lifecycle Management (Optional):** Added configurable assembly pruning (based on size, age, inactivity) and merging (based on similarity). Disabled merging by default.
- **Core Stability:** Refactored `MemoryVectorIndex` initialization and async operations for improved reliability. Added `_post_initialize_check`. Moved complex repair logic externally.
- **GPU Setup:** Externalized FAISS installation logic to `gpu_setup.py`.

### Changed
- **`MemoryVectorIndex`:** Simplified structure, focusing on core `IndexIDMap` (CPU default) operations and persistence. Diagnostic `verify_index_integrity` no longer attempts repairs.
- **`SynthiansMemoryCore`:** Refactored initialization and shutdown sequences for robustness. Integrated assembly creation, update, activation, and pending queue logic.
- **`MemoryPersistence`:** Added methods for handling assembly persistence. Updated `memory_index.json` schema.
- **Retrieval Logic:** `retrieve_memories` now incorporates assembly boost based on synchronized assembly embeddings. Filters/sorts using final `relevance_score`.

### Fixed
- Addressed potential race conditions with improved locking in `MemoryVectorIndex` and `SynthiansMemoryCore`.
- Corrected `MemoryAssembly` serialization/deserialization issues.
- Ensured proper `await` usage for all async index/persistence operations.
- Fixed test fixtures causing server startup issues.
```

---

# **`docs/NEWEST-DOCUMENTATION.md` (Update Status)**

```markdown
## Development Roadmap & Status (*Current Date*)

**Project:** Synthians Cognitive Architecture (Lucidia)
**Focus:** Bi-Hemispheric Memory System
**Status:** Phase 5.8 Completed

---

### Phase 1: Memory Core Unification & Foundation (Completed)
*(No Change)*

---

### Phase 2: Neural Memory Module Implementation (Completed)
*(No Change)*

---

### Phase 3: Context Cascade Engine / Orchestration (Completed)
*(No Change)*

---

### Phase 4: Meta-Attentional Systems (Titans Variants) (Completed - Previously Phase 5 Adaptive Layers)
*   **Objective:** Implement adaptive intelligence through dynamic variant selection, LLM-guided memory operations, adaptive attention heuristics, and enhanced diagnostics.
*   **Status:** **DONE** (Functionality implemented, stabilization ongoing in separate efforts if needed)
*   **Key Outcomes:**
    *   `ContextCascadeEngine` manages variant switching (MAC, MAG, MAL).
    *   `VariantSelector` chooses variants based on context/performance/hints.
    *   `MemoryLLMRouter` integrates external LLM guidance (via LM Studio).
    *   Adaptive attention heuristics influence context length/focus.
    *   Diagnostics dashboard monitors variant performance.

---

### **Phase 5.8: Memory Assembly Stabilization & Integration (Completed)**
*   **Objective:** Fully integrate Memory Assemblies as a stable, persistent, and core feature, enhancing contextual retrieval and providing robust lifecycle management.
*   **Status:** **DONE**
*   **Key Outcomes:**
    *   Stable persistence and vector indexing (`IndexIDMap`) for `MemoryAssembly`.
    *   Assemblies reliably contribute to retrieval relevance via boosting.
    *   Robust consistency mechanism (`vector_index_updated_at` timestamp) implemented.
    *   Graceful degradation for failed index updates (pending queue + retry loop).
    *   Enhanced diagnostics for assemblies and index state (`/stats`, `/assemblies`).
    *   Optional assembly lifecycle management (pruning/merging).
    *   Improved `MemoryVectorIndex` stability and simplified design.
    *   Core startup reliability improved.

---

### Phase 5.9: Memory Insight & Interpretability (Next)
*   **Objective:** Develop tools and methods for understanding and visualizing memory system behavior.
*   **Status:** **TODO**
*   **Tasks:**
    *   Enhance dashboards with visualizations (e.g., assembly graphs, activation timelines).
    *   Implement `explain_assembly_activation` functionality.
    *   Develop methods for analyzing semantic drift in assemblies (using versioned embeddings - requires further design).
    *   Refine `RecoveryTimeline` concept for failed operations.
    *   Integrate more detailed metrics into `MetricsStore`.

---

*(Remaining sections on Protocol Seal, Reflective Summary - Unchanged)*
```

---
---

This comprehensive set should accurately reflect the state of the system and its documentation after successfully implementing the refined Phase 5.8 plan. Remember to replace placeholders like `*03/04/2025*` and review code snippets against the actual final code.