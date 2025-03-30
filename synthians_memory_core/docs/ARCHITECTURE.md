# Synthians Cognitive Architecture (March 2025)

## 1. Overview

This document describes the architecture of the Synthians cognitive system, implementing a **Bi-Hemispheric Cognitive Architecture**. This model separates persistent, indexed memory storage/retrieval (Memory Core) from dynamic, sequence-aware associative memory processing (Neural Memory Server), orchestrated by the Context Cascade Engine (CCE).

The system aims to enable adaptive memory recall, continuous learning from experience via test-time adaptation, contextual awareness through attention mechanisms, and robust handling of complex data like embeddings and emotional context.

**Core Principles:**

*   **Memory is weighted, not just chronological** (QuickRecal)
*   **Emotion shapes recall** (Emotional Gating)
*   **Surprise signals significance** (Neural Memory Loss/Grad → QuickRecal Boost)
*   **Ideas cluster and connect** (Assemblies & Attention)
*   **Presence emerges from adaptive memory** (Test-Time Learning & Variants)

## 2. System Components

The system comprises three main microservices:

1.  **Synthians Memory Core (`synthians_memory_core`):** The primary repository for structured memories (content, metadata, embeddings). Handles storage, indexing (FAISS with `IndexIDMap`), retrieval, relevance scoring (HPC-QuickRecal), metadata synthesis, emotional analysis/gating, and persistence. Analogous to a searchable, adaptive library.
2.  **Neural Memory Server (`synthians_trainer_server`):** Implements an adaptive, associative memory inspired by the Titans paper, capable of test-time learning on sequences of embeddings. Handles K/V/Q projections, associative retrieval, and test-time weight updates, providing surprise metrics. Analogous to learning temporal patterns and associations.
3.  **Context Cascade Engine (`orchestrator`):** Acts as the central orchestrator, managing the information flow between the Memory Core and the Neural Memory Server. Implements the core cognitive cycle and different attention-based variants (MAC, MAG, MAL).

**Diagram:**

```text
+--------------------------+        +--------------------------+        +-----------------------------+
|                          |        |                          |        |                             |
|  Synthians Memory Core   |<-(5)---|  Context Cascade Engine  |---(2,4,6)->|   Neural Memory Server      |
|  (Storage/Retrieval)     |-------|       (Orchestrator)     |-------|   (Associative/Predictive)  |
|  (FAISS, QuickRecal,     |  (1)   |   (Handles Variants &    |  (3,7) |   (Test-Time Learning,      |
|   Emotion, Persistence)  |        |    Sequence History)     |        |    Surprise Metrics)        |
+--------------------------+        +--------------------------+        +-----------------------------+
       |         ^                                                           |         ^
       |         | (Filesystem, JSON)                                        |         | (TensorFlow, State Files)
       v         |                                                           v         |
+-----------------+-------+                                           +-----------------+----------+
| Memory Persistence &    |                                           | Neural Memory Module (M) & |
|   Vector Index          |                                           |   Momentum State (S)       |
+-------------------------+                                           +----------------------------+

Key Steps (Refactored & Functional Flow):
1. Input -> CCE -> Memory Core (/process_memory) -> Get x_t, memory_id, initial_qr
2. CCE -> Neural Memory (/get_projections) -> Get k_t, v_t, q_t
3. CCE -> Variant Pre-Update (MAG: /calculate_gates; MAL: calc v'_t)
4. CCE -> Neural Memory (/update_memory w/ variant mods) -> Get loss, grad_norm
5. CCE -> Memory Core (/api/memories/update_quickrecal_score) -> Apply boost -> **FUNCTIONAL**
6. CCE -> Neural Memory (/retrieve) -> Get y_t_raw, q_t
7. CCE -> Variant Post-Retrieval (MAC: calc attended_y_t) -> Get y_t_final
8. CCE -> Update Sequence History (ts, id, x, k, v, q, y_final)
```

## 3. Key Components Deep Dive

### 3.1. Synthians Memory Core (`synthians_memory_core` package)

*   **Role:** Long-term, indexed, searchable memory.
*   **Key Classes:**
    *   `SynthiansMemoryCore`: Main orchestrating class.
    *   `MemoryEntry` / `MemoryAssembly`: Data structures.
    *   `MemoryVectorIndex`: FAISS `IndexIDMap` wrapper for fast, ID-keyed vector search. Handles GPU/CPU, persistence, migration, integrity checks.
    *   `MemoryPersistence`: Asynchronous JSON-based storage for `MemoryEntry` and `MemoryAssembly` objects. Manages `memory_index.json`.
    *   `UnifiedQuickRecallCalculator`: Calculates memory relevance (`quickrecal_score`) based on multiple factors (recency, emotion, similarity, importance, surprise feedback, etc.).
    *   `MetadataSynthesizer`: Enriches memories with derived metadata (temporal, emotional, cognitive, embedding stats).
    *   `GeometryManager`: Centralized handling of embedding validation (NaN/Inf), normalization, dimension alignment (padding/truncation), and geometric distance/similarity calculations (Euclidean, Hyperbolic).
    *   `EmotionAnalyzer` / `EmotionalGatingService`: Processes text for emotion; filters/re-ranks retrieved memories based on emotional context and cognitive load.
    *   `ThresholdCalibrator`: Dynamically adjusts retrieval similarity thresholds.
    *   `TrainerIntegrationManager`: Handles API calls related to the trainer feedback loop.
*   **Status:** Core functionality implemented and stabilized. Vector index uses robust `IndexIDMap`. Surprise feedback loop is functional.

### 3.2. Neural Memory Server (`synthians_trainer_server` package)

*   **Role:** Adaptive associative memory, learning temporal patterns.
*   **Key Classes:**
    *   `NeuralMemoryModule`: TensorFlow/Keras model implementing Titans-style test-time learning (inner loop updates `M`, outer loop trains projections/gates).
    *   `MemoryMLP`: The internal MLP (M) storing associations.
    *   `http_server.py`: FastAPI server exposing NM functionality.
    *   `MetricsStore`: Collects operational metrics (updates, boosts, retrievals) and generates diagnostic reports.
    *   *(Surprise Calculation)*: The `/update_memory` endpoint calculates associative error (`loss`) and gradient magnitude (`grad_norm`), returning them as surprise metrics.
*   **Status:** Implemented with test-time learning. API supports variant interactions (`/get_projections`, `/calculate_gates`, modified `/update_memory`). Auto-initializes. TF/NumPy compatibility issues resolved via lazy loading.

### 3.3. Context Cascade Engine (`orchestrator` package)

*   **Role:** Orchestrates the cognitive cycle, manages history, and applies variant logic.
*   **Key Classes:**
    *   `ContextCascadeEngine`: Main orchestrator implementing the refactored flow.
    *   `SequenceContextManager`: Manages deque-based history `(ts, id, x, k, v, q, y_final)` for attention.
    *   `titans_variants.py`: Base class and specific logic for MAC, MAG, MAL variants, interacting with NM server API and `SequenceContextManager`.
*   **Status:** Implements the corrected flow for all variants. Dynamically configures attention based on NM server response. Manages history. Initiates QuickRecal boost feedback.

## 4. Core Concepts & Strategies

### 4.1. Embedding Handling

*   **Centralized Management:** `GeometryManager` handles validation, alignment, normalization, and distance/similarity calculations based on configured `embedding_dim` and `geometry_type`.
*   **Validation:** Checks for `None`, `NaN`, `Inf`. Invalid vectors are typically replaced with zero vectors and warnings logged.
*   **Dimension Alignment:** Handles mismatches (e.g., 384 vs 768) using configured `alignment_strategy` ('truncate' or 'pad') via `align_vectors`. Alignment occurs at API boundaries and before vector index operations.
*   **Normalization:** L2 normalization is typically applied before storage and similarity calculations.
*   **See:** `docs/core/embedding_handling.md`

### 4.2. Vector Indexing (FAISS `IndexIDMap`)

*   **Implementation:** `MemoryVectorIndex` uses `faiss.IndexIDMap` wrapping a base `IndexFlatL2` or `IndexFlatIP`.
*   **ID Management:** String `memory_id`s are hashed to unique `int64` numeric IDs for use with `add_with_ids`. `id_to_index` maps `string_id -> numeric_id`. `search` uses reverse mapping.
*   **Persistence:** The `.faiss` index file now stores vectors and numeric IDs together. `.mapping.json` serves as a backup for the string->numeric map.
*   **Integrity & Repair:** `verify_index_integrity` checks consistency. `migrate_to_idmap` converts legacy indices. `recreate_mapping` recovers the string->numeric map from backup or filesystem scan.
*   **See:** `docs/core/vector_index.md`

### 4.3. Titans Variants & Attention

*   **Orchestration:** The CCE manages the variant-specific flow.
*   **MAC (Post-Retrieval):** Enhances retrieved `y_t` using attention over historical `(k_i, y_i)`.
*   **MAG (Pre-Update):** Calculates attention over historical `k_i` to determine external gates (`alpha, theta, eta`) passed to `/update_memory`.
*   **MAL (Pre-Update):** Calculates attention over historical `(k_i, v_i)` to create `v'_t`, which replaces `v_t` in the `/update_memory` call.
*   **History:** `SequenceContextManager` stores `(ts, id, x, k, v, q, y_final)` required for attention.
*   **See:** `docs/orchestrator/titans_variants.md`

### 4.4. TensorFlow Lazy Loading

*   **Problem:** TensorFlow importing NumPy early caused version conflicts with `fix_numpy.py`.
*   **Solution:** A `_get_tf()` helper function in `titans_variants.py` delays `import tensorflow` until it's first needed, allowing NumPy downgrade to occur first. Code using TF calls `_get_tf()` instead of direct import.

### 4.5. Surprise Feedback Loop

*   **Mechanism:** NM Server's `/update_memory` returns `loss` and `grad_norm`. CCE calculates a `boost` value. CCE calls Memory Core's `/api/memories/update_quickrecal_score` endpoint with the target `memory_id` and `boost` value. The Memory Core service then updates the `quickrecal_score` and associated metadata for that specific memory entry.
*   **Impact:** Connects the adaptive learning of the Neural Memory directly to the relevance ranking within the Memory Core, allowing surprising or hard-to-associate memories to gain significance.
*   **Status:** Fully functional and tested end-to-end.

## 5. Current Status & Known Gaps

*   **Status:** Core architecture implemented. Bi-hemispheric loop with surprise feedback is functional. Vector index is robust using `IndexIDMap`. Retrieval pipeline is stabilized. Basic variant flows are implemented in CCE.
*   **Known Gaps:**
    *   **Variant Testing:** Dedicated integration tests needed for MAC, MAG, MAL effects.
    *   **Performance:** NM test-time update lacks parallelization.
    *   **Outer Loop Training:** NM `/train_outer` needs significant development for effective use.
    *   **Component Deep Dives:** Documentation for QuickRecal factors, Metadata Synthesizer pipeline, etc., needs more detail.
    *   **Configuration:** Ensure all key parameters are exposed via `CONFIGURATION_GUIDE.md`.
    *   **Test Teardown:** Investigate remaining background task cancellation warnings during test shutdown.

---

*(This document reflects the state as of late March 2025 and should be updated alongside major architectural changes.)*
