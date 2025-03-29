
**1. Updated `ARCHITECTURE.md`**

```markdown
# Synthians Cognitive Architecture (Current State - March 2025)

## 1. Overview

This document describes the currently implemented architecture of the Synthians cognitive system. It follows a **Bi-Hemispheric Cognitive Architecture** model, separating persistent, indexed memory storage and retrieval from dynamic, sequence-aware associative memory processing.

The system is composed of three main services:

1.  **Synthians Memory Core:** The primary repository for structured memories, handling storage, indexing, metadata enrichment, emotional context, and relevance scoring (HPC-QuickRecal).
2.  **Neural Memory Server (`synthians_trainer_server`):** Implements an associative memory inspired by the Titans paper, capable of test-time learning based on sequences of embeddings.
3.  **Context Cascade Engine (`orchestrator`):** Acts as the central orchestrator, managing the flow of information between the Memory Core and the Neural Memory Server, implementing the core cognitive cycle and supporting different attention-based variants (MAC, MAG, MAL).

The architecture aims to enable adaptive memory recall, continuous learning from experience, and context-aware information processing. **The core cognitive cycle, including the surprise feedback loop from the Neural Memory Server to the Memory Core's QuickRecal score, is now functional and verified.**

```text
+-------------------------+        +--------------------------+        +--------------------------+
|                         |        |                          |        |                          |
|  Synthians Memory Core  |<-.(5).-|  Context Cascade Engine  |-.(2,4,6)-|  Neural Memory Server    |
|   (Storage/Retrieval)   |--------|       (Orchestrator)     |--------| (Associative/Predictive) |
| (Incl. QuickRecal Boost)|  (1)   |                          |  (3,7) | (Incl. Surprise Metrics) |
+-------------------------+        +--------------------------+        +--------------------------+
       |      ^                                                                 |      ^
       |      | (FAISS, Persistence,                                            |      | (TensorFlow,
       |      |  QuickRecal, Emotion)                                           |      |  Test-Time Learning)
       v      |                                                                 v      |
+--------------+----------+                                           +-----------------+-------+
|  Memory Persistence &   |                                           | Neural Memory Module (M)|
|     Vector Index        |                                           +-----------------+-------+

Key Steps (Refactored & Functional Flow):
1. Input -> CCE -> Memory Core (/process_memory) -> Get x_t, memory_id, initial_qr
2. CCE -> Neural Memory (/get_projections) -> Get k_t, v_t, q_t
3. CCE -> Variant Pre-Update (MAG: /calculate_gates; MAL: calc v'_t)
4. CCE -> Neural Memory (/update_memory w/ variant mods) -> Get loss, grad_norm
5. CCE -> Memory Core (/api/memories/update_quickrecal_score) -> Apply boost -> **FUNCTIONAL**
6. CCE -> Neural Memory (/retrieve) -> Get y_t_raw, q_t
7. CCE -> Variant Post-Retrieval (MAC: calc attended_y_t) -> Get y_t_final
```

## 2. Key Components

### 2.1. Synthians Memory Core (`synthians_memory_core` package)

*   **Role:** Long-term, indexed memory storage and retrieval. Analogous to a searchable library.
*   **Responsibilities:**
    *   Storing `MemoryEntry` objects (content, embedding, metadata, scores).
    *   Managing `MemoryAssembly` objects (groups of related memories).
    *   Providing fast similarity search via `MemoryVectorIndex` (FAISS, GPU-aware).
    *   Calculating memory importance using `UnifiedQuickRecallCalculator` (HPC-QR principles).
    *   **Accepting QuickRecal score updates** based on external feedback (e.g., surprise from Neural Memory) via its API (`/api/memories/update_quickrecal_score`).
    *   Enriching memories with contextual information via `MetadataSynthesizer`.
    *   Analyzing and applying emotional context using `EmotionAnalyzer` and `EmotionalGatingService`.
    *   Handling persistent storage and backups via `MemoryPersistence`.
    *   Dynamically adjusting retrieval thresholds via `ThresholdCalibrator`.
    *   Managing embedding geometry (Euclidean, Hyperbolic, etc.) via `GeometryManager`.
    *   Exposing functionality via a FastAPI server (`api/server.py`).
    *   Providing methods (`get_memory_by_id`, `update_memory`) used by `TrainerIntegrationManager` to implement the QuickRecal feedback mechanism.
*   **Current Status:** Core components are implemented and integrated. FAISS integration is robust. **The methods (`get_memory_by_id`, `update_memory`) required for the feedback loop are implemented and the API endpoint (`/api/memories/update_quickrecal_score`) is functional.**

### 2.2. Neural Memory Server (`synthians_trainer_server` package)

*   **Role:** Adaptive, associative memory performing test-time learning. Analogous to learning patterns and associations.
*   **Responsibilities:**
    *   Implementing the `NeuralMemoryModule` based on the Titans paper using TensorFlow.
    *   Calculating Key, Value, Query projections (`/get_projections`).
    *   Performing test-time updates to its internal memory weights (`M`) based on input embeddings and calculated gates (`/update_memory`). **Returns loss and gradient norm as surprise metrics.**
    *   Retrieving associated value embeddings based on a query projection (`/retrieve`).
    *   Calculating dynamic gate values (`alpha_t`, `theta_t`, `eta_t`) based on attention outputs (`/calculate_gates`) for the MAG variant.
    *   Providing configuration details (`/config`).
    *   Collecting and exposing diagnostic metrics via `MetricsStore`.
    *   Exposing functionality via a FastAPI server (`synthians_trainer_server/http_server.py`).
*   **Current Status:** Implemented, including API endpoints for variant support. Test-time learning mechanism is in place. Fixes for TF/Keras layer registration and save/load have been applied. Configuration includes handling for dimension mismatches. Auto-initialization on startup confirmed.

### 2.3. Context Cascade Engine (`orchestrator` package)

*   **Role:** Orchestrates the interaction between the Memory Core and Neural Memory Server, implementing the full cognitive cycle.
*   **Responsibilities:**
    *   Receiving new inputs (content/embedding/metadata).
    *   Coordinating the **Refactored Information Flow** (see Section 3).
    *   Selecting and activating the appropriate Titans Architecture Variant (MAC, MAG, MAL) based on configuration (`TITANS_VARIANT` env variable).
    *   Managing sequence history (`SequenceContextManager`) required for attention mechanisms in variants.
    *   **Initiating surprise-based feedback** (loss/grad_norm -> boost) to the Memory Core via its API (`/api/memories/update_quickrecal_score`). **This feedback mechanism is now functional.**
    *   Handling errors and communication between the two main memory components.
*   **Current Status:** Implements the corrected, functional flow for variant integration and the surprise feedback loop. Dynamically configures itself based on Neural Memory `/config`. Uses lazy loading for TensorFlow.

## 3. Refactored Information Flow (Cognitive Cycle)

The CCE orchestrates the following sequence for processing a new input (`content`, `embedding`, `metadata`):

1.  **Store Memory:** CCE sends input to Memory Core (`/process_memory`). Memory Core stores it, generates metadata, calculates initial QuickRecal, and returns the validated embedding (`x_t`), `memory_id`, and `quickrecal_score`.
2.  **Get Projections:** CCE sends `x_t` to Neural Memory Server (`/get_projections`). NM Server returns Key (`k_t`), Value (`v_t`), and Query (`q_t`) projections *without* updating its internal weights.
3.  **Variant Pre-Update (MAG/MAL):**
    *   If **MAG** is active: CCE calculates attention output (using `q_t`, historical keys `K_hist`) and calls NM Server (`/calculate_gates`) to get external gate values (`alpha_t`, `theta_t`, `eta_t`).
    *   If **MAL** is active: CCE calculates attention output (using `q_t`, historical keys `K_hist`, historical values `V_hist`), combines it with `v_t` to create a modified value projection (`v'_t`).
    *   If **NONE** or **MAC**: This step is skipped.
4.  **Update Neural Memory:** CCE calls NM Server (`/update_memory`) providing:
    *   Base: `input_embedding` (`x_t`).
    *   MAG: External gate values (`external_alpha_gate`, etc.).
    *   MAL: Explicit projections (`key_projection=k_t`, `value_projection=v'_t`).
    *   NM Server performs the test-time update using the provided parameters and returns `loss` and `grad_norm`.
5.  **Apply QuickRecal Boost:** CCE calculates a boost value based on `loss`/`grad_norm`. It **successfully calls** Memory Core (`/api/memories/update_quickrecal_score`) to apply this boost to the original memory's score.
6.  **Retrieve from Neural Memory:** CCE sends `x_t` to NM Server (`/retrieve`). NM Server calculates the query projection `q_t` (may differ slightly from step 2 if weights changed) and retrieves the associated raw embedding (`y_t_raw`) using its internal memory `M(q_t)`. It returns `y_t_raw` and the `query_projection` used.
7.  **Variant Post-Retrieval (MAC):**
    *   If **MAC** is active: CCE calculates attention output (using `q_t` from step 6, historical keys `K_hist`, historical outputs `Y_hist`), combines it with `y_t_raw` to create an attended output (`y_t_final`).
    *   Otherwise, `y_t_final` is set to `y_t_raw`.
8.  **Update History:** CCE adds the full context tuple `(timestamp, memory_id, x_t, k_t, v_t, q_t, y_t_final)` to the `SequenceContextManager`.
9.  **Finalize:** CCE constructs and returns a response containing the `memory_id`, processing status, surprise metrics, retrieval results (`y_t_final`), QuickRecal feedback status, and variant metrics.

## 4. Titans Architecture Variants (Implemented in CCE)

*(No changes needed in this section)*

## 5. Supporting Systems

*(No significant changes needed, but confirm descriptions are accurate)*

## 6. API Layer

*(No significant changes needed, but descriptions should align with API_REFERENCE.md)*

## 7. Current Status & Known Gaps

*   **Status:** The core architectural components (Memory Core, Neural Memory, CCE) are implemented. The CCE orchestrates the refactored flow, enabling the activation and correct *timing* for MAC, MAG, and MAL variants. Basic memory storage, retrieval, test-time learning, variant processing, and **the surprise->QuickRecal feedback loop are functional and verified.** Diagnostics and metrics collection are integrated.
*   **Known Gaps:**
    *   **Variant Testing:** While the CCE flow is correct, the MAC, MAG, and MAL variants require dedicated integration tests to verify their specific attention-based effects.
    *   **Performance:** The Neural Memory's test-time update (`/update_memory`) is computationally intensive and currently lacks parallelization optimizations mentioned in the Titans paper.
    *   **Outer Loop Training:** While an endpoint exists (`/train_outer`), effective outer loop training for the Neural Memory module requires significant further development.
    *   **Documentation:** Still requires ongoing consolidation and refinement to match the very latest code state across all documents.
    *   **Test Teardown:** Background task cancellation warnings during test shutdown need investigation.

