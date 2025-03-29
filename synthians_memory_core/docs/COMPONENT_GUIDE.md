Okay, here are the component-specific documentation guides based on the current state of the codebase, emphasizing integration points.

---

## Component Guide: Synthians Memory Core

**Version:** 1.0
**Date:** March 29, 2025
**Primary Files:** `synthians_memory_core/`, `api/server.py`, `api/client/client.py`

### 1. Overview

The Synthians Memory Core serves as the primary, persistent storage and retrieval system for the Synthians cognitive architecture. It is responsible for managing individual memory entries (`MemoryEntry`) and related groups (`MemoryAssembly`), calculating memory relevance (`quickrecal_score`), handling emotional context, and providing fast, indexed access to memories. It is analogous to a highly organized, searchable knowledge base or library.

### 2. Core Responsibilities

*   **Memory Storage:** Persists `MemoryEntry` objects, including content, embeddings (Euclidean and optionally Hyperbolic), and rich metadata.
*   **Memory Retrieval:** Retrieves memories based on semantic similarity (via vector search), QuickRecal scores, emotional resonance, and optional metadata filters.
*   **Relevance Scoring:** Calculates `quickrecal_score` using the `UnifiedQuickRecallCalculator` based on factors like recency, emotion, importance, surprise feedback (intended), etc.
*   **Metadata Synthesis:** Enriches memories with temporal, emotional, cognitive, and embedding-based metadata using `MetadataSynthesizer`.
*   **Vector Indexing:** Provides fast similarity search using `MemoryVectorIndex` (FAISS), supporting CPU/GPU and persistence.
*   **Emotional Processing:** Analyzes emotional content (`EmotionAnalyzer`) and applies emotional gating/filtering during retrieval (`EmotionalGatingService`).
*   **Memory Assemblies:** Manages groups of related memories, maintaining composite embeddings and activation levels.
*   **Persistence:** Handles asynchronous saving/loading of memories and assemblies to disk (`MemoryPersistence`).
*   **Adaptive Thresholding:** Optionally adjusts retrieval similarity thresholds based on user feedback (`ThresholdCalibrator`).
*   **Geometry Management:** Uses `GeometryManager` for consistent handling of embedding dimensions, normalization, and geometric calculations.
*   **API Exposure:** Provides a comprehensive FastAPI interface for external interaction.
*   **Trainer Integration:** Provides endpoints (`/api/memories/get_sequence_embeddings`, `/api/memories/update_quickrecal_score`) for interaction with the sequence trainer/orchestrator.

### 3. Key Classes/Modules

*   `synthians_memory_core.SynthiansMemoryCore`: Main orchestrating class.
*   `synthians_memory_core.memory_structures`: Defines `MemoryEntry`, `MemoryAssembly`.
*   `synthians_memory_core.hpc_quickrecal`: `UnifiedQuickRecallCalculator`.
*   `synthians_memory_core.geometry_manager`: `GeometryManager`.
*   `synthians_memory_core.emotional_intelligence`: `EmotionalAnalyzer`, `EmotionalGatingService`.
*   `synthians_memory_core.memory_persistence`: `MemoryPersistence`.
*   `synthians_memory_core.vector_index`: `MemoryVectorIndex`.
*   `synthians_memory_core.metadata_synthesizer`: `MetadataSynthesizer`.
*   `synthians_memory_core.adaptive_components`: `ThresholdCalibrator`.
*   `synthians_memory_core.api.server`: FastAPI application exposing the core.
*   `synthians_memory_core.memory_core.trainer_integration`: `TrainerIntegrationManager`.

### 4. Configuration

*   Primary configuration is passed as a dictionary to the `SynthiansMemoryCore` constructor.
*   Key parameters include `embedding_dim`, `geometry` type, `storage_path`, `vector_index_type`, `persistence_interval`, etc.
*   Environment variables (`HOST`, `PORT`, `LOG_LEVEL`, `EMBEDDING_MODEL`) control the API server runtime.
*   `gpu_setup.py` attempts to configure FAISS for GPU usage during startup.

### 5. API Endpoints (Purpose)

The API (`api/server.py`) exposes core functionalities, including:
*   Memory CRUD-like operations (Process/Store, Retrieve).
*   Supporting functions (Generate Embedding, Analyze Emotion, Calculate QuickRecal).
*   Feedback mechanisms (Provide Relevance Feedback).
*   Advanced features (Detect Contradictions, Process Transcriptions, Assembly Management).
*   Integration endpoints for the trainer/orchestrator.

*(See `API_REFERENCE.md` for detailed endpoint definitions)*

### 6. Internal Workflow Example (Memory Storage)

1.  `/process_memory` endpoint receives data.
2.  Validates/aligns/normalizes incoming embedding (or generates one).
3.  Calls `SynthiansMemoryCore.process_new_memory`.
4.  `process_new_memory` orchestrates:
    *   Calculate QuickRecal score (`UnifiedQuickRecallCalculator`).
    *   Analyze emotion (`EmotionAnalyzer`).
    *   Calculate hyperbolic embedding if needed (`GeometryManager`).
    *   Synthesize metadata (`MetadataSynthesizer`).
    *   Create `MemoryEntry`.
    *   Store entry in `self._memories`.
    *   Save to disk (`MemoryPersistence.save_memory`).
    *   Update relevant `MemoryAssembly` objects.
    *   Add embedding to `MemoryVectorIndex`.
5.  Returns details of the processed memory.

### 7. Integration Points

*   **Receives From Context Cascade Engine (CCE):**
    *   New memory data via `POST /process_memory`.
    *   Requests for sequence embeddings via `POST /api/memories/get_sequence_embeddings`.
    *   Requests to update QuickRecal scores via `POST /api/memories/update_quickrecal_score` (**NOTE: Requires `update_memory` implementation**).
*   **Sends To Context Cascade Engine (CCE):**
    *   Response from `/process_memory` (includes `memory_id`, `embedding`, `quickrecal_score`, `metadata`).
    *   Response from `/retrieve_memories` (list of memory dictionaries).
    *   Response from `/api/memories/get_sequence_embeddings` (list of sequence embeddings).
    *   Response from `/api/memories/update_quickrecal_score` (update status).
*   **Internal Dependencies:** Relies heavily on its internal components (`GeometryManager`, `MemoryPersistence`, `MemoryVectorIndex`, `UnifiedQuickRecallCalculator`, etc.).

### 8. Current Status & Known Gaps

*   **Status:** Core storage, retrieval, indexing, emotion, metadata, and persistence functionalities are largely implemented and robust. FAISS integration handles GPU and validation well. API provides broad coverage.
*   **Critical Gap:** The feedback loop is **broken**. The `SynthiansMemoryCore` class **lacks the implemented `update_memory` and `get_memory_by_id` methods**. This prevents the `TrainerIntegrationManager` (and thus the CCE) from applying QuickRecal score boosts based on surprise metrics received from the Neural Memory Server.
*   **Minor Gaps:** Contradiction detection is basic; assembly removal logic is simplified. Embedding generation could be further centralized.

---

## Component Guide: Neural Memory Server

**Version:** 1.0
**Date:** March 29, 2025
**Primary Files:** `synthians_trainer_server/`

### 1. Overview

The Neural Memory Server implements an adaptive, associative memory based on the principles outlined in the Titans paper. Its core component is the `NeuralMemoryModule`, a TensorFlow/Keras model capable of **test-time learning**. It learns associations between Key and Value projections derived from input embeddings and can retrieve associated Values based on Query projections. It runs as a separate service, providing low-level associative memory operations.

### 2. Core Responsibilities

*   **Key/Value/Query Projections:** Calculates distinct vector projections (K, V, Q) from input embeddings using learned weight matrices (outer parameters).
*   **Associative Retrieval:** Given a Query projection (`q_t`), predicts/retrieves the associated Value embedding (`y_t`) using its internal `MemoryMLP` (`M`).
*   **Test-Time Update:** Updates the weights of its internal `MemoryMLP` (`M`) based on the association between the current `k_t` and `v_t` (or `v'_t` for MAL). Calculates `loss` and `grad_norm` as surprise metrics during this update.
*   **Dynamic Gate Calculation (for MAG):** Calculates adaptive gate values (`alpha_t`, `theta_t`, `eta_t`) based on attention outputs provided by the CCE.
*   **State Management:** Manages internal memory weights (`M`) and momentum state.
*   **Persistence:** Provides mechanisms to save and load its complete state (config, weights, momentum).
*   **Diagnostics:** Collects metrics (`MetricsStore`) and exposes diagnostic endpoints.
*   **API Exposure:** Provides a FastAPI interface for the CCE and potentially other tools.

### 3. Key Classes/Modules

*   `neural_memory.NeuralMemoryModule`: The core TensorFlow/Keras model implementing the memory logic.
*   `neural_memory.MemoryMLP`: The internal MLP representing the associative memory `M`.
*   `neural_memory.NeuralMemoryConfig`: Configuration class.
*   `http_server.py`: FastAPI application exposing the Neural Memory API.
*   `metrics_store.py`: `MetricsStore` for collecting operational metrics.
*   `surprise_detector.py`: `SurpriseDetector` used by `/analyze_surprise` endpoint.

### 4. Configuration

*   Primary configuration via `NeuralMemoryConfig` (passed during initialization or loaded from state).
*   Key parameters: `input_dim`, `key_dim`, `value_dim`, `query_dim`, `memory_hidden_dims`, gate initial values, `outer_learning_rate`.
*   Initialized automatically on startup via `http_server.py`'s `startup_event`, but can be re-initialized via `POST /init`.
*   State persistence paths specified in `/save` and `/load` requests.

### 5. API Endpoints (Purpose)

The API (`http_server.py`) provides low-level operations for the CCE:
*   `POST /init`: Initialize/re-initialize the module.
*   `POST /get_projections`: Get K, V, Q projections without updating memory.
*   `POST /update_memory`: Perform the test-time learning update step, optionally using external gates (MAG) or projections (MAL). Returns surprise metrics.
*   `POST /retrieve`: Retrieve associated value embedding (`y_t`) given an input embedding.
*   `POST /calculate_gates`: Calculate dynamic gates based on attention output (for MAG).
*   `GET /config`, `POST /config`: Get/Set configuration details and capabilities.
*   `POST /save`, `POST /load`: Persist or load the module's state.
*   `GET /health`, `GET /status`: Check service health and initialization status.
*   `POST /analyze_surprise`: Analyze surprise between two embeddings.
*   `GET /diagnose_emoloop`: Get diagnostic metrics.

*(See `API_REFERENCE.md` for detailed endpoint definitions)*

### 6. Internal Workflow Example (Update Step)

1.  `/update_memory` receives data (input `x_t`, optional external K/V/Gates).
2.  `NeuralMemoryModule.update_step` is called.
3.  If K/V not provided externally, calculates `k_t, v_t` from `x_t` using `WK_layer`, `WV_layer`.
4.  Determines gate values (`alpha_t`, `theta_t`, `eta_t`) using external values (if MAG) or internal logits.
5.  Calculates predicted value `predicted_v = M(k_t)` using `MemoryMLP`.
6.  Calculates `loss = ||predicted_v - v_t||^2` (using original `v_t` or `v'_t` from MAL).
7.  Calculates gradients of `loss` w.r.t. `MemoryMLP` weights (`M`).
8.  Calculates `grad_norm`.
9.  Updates internal `momentum_state` using gradients and `theta_t`, `eta_t`.
10. Updates `MemoryMLP` weights (`M`) using `momentum_state` and `alpha_t`.
11. Returns `loss` and `grad_norm`.

### 7. Integration Points

*   **Receives From Context Cascade Engine (CCE):**
    *   Initialization requests via `POST /init`.
    *   Input embeddings (`x_t`) for projection via `POST /get_projections`.
    *   Input embeddings (`x_t`), optional external projections (`k_t`, `v'_t`), and optional external gates (`alpha_t`, `theta_t`, `eta_t`) for memory update via `POST /update_memory`.
    *   Input embeddings (`x_t`) for retrieval via `POST /retrieve`.
    *   Attention outputs for gate calculation via `POST /calculate_gates`.
    *   Requests for configuration/capabilities via `GET /config`.
    *   Requests for diagnostics via `GET /diagnose_emoloop`.
    *   Requests to save/load state via `POST /save`, `POST /load`.
*   **Sends To Context Cascade Engine (CCE):**
    *   Projections (`key_projection`, `value_projection`, `query_projection`) from `/get_projections`.
    *   Loss and gradient norm from `/update_memory`, along with projections/gates used.
    *   Retrieved embeddings (`retrieved_embedding`) and query projection from `/retrieve`.
    *   Calculated gate values (`alpha`, `theta`, `eta`) from `/calculate_gates`.
    *   Configuration details from `/config`.
    *   Diagnostic metrics from `/diagnose_emoloop`.
    *   Status/health information.
*   **Internal Dependencies:** TensorFlow, NumPy, `MetricsStore`, `SurpriseDetector`.

### 8. Current Status & Known Gaps

*   **Status:** Implemented and functional, including support for external projections (MAL) and gates (MAG) via the API. Test-time learning mechanism works. Persistence and diagnostics are integrated. Auto-initialization on startup implemented.
*   **Gaps:**
    *   **Performance:** Single-threaded `update_step` is slow; parallelization needed for high throughput.
    *   **Outer Loop:** `/train_outer` exists, but requires significant effort to use effectively for meta-learning optimal projection/gate parameters.
    *   **Complexity:** The internal dynamics are complex and require robust monitoring via the `MetricsStore` and diagnostic endpoints.

---

## Component Guide: Context Cascade Engine (CCE)

**Version:** 1.0
**Date:** March 29, 2025
**Primary Files:** `orchestrator/`

### 1. Overview

The Context Cascade Engine (CCE) serves as the central orchestrator for the Synthians cognitive architecture. It manages the bi-directional flow of information between the persistent Memory Core and the adaptive Neural Memory Server. The CCE implements the core cognitive cycle and dynamically adapts its processing based on the configured Titans Architecture Variant (NONE, MAC, MAG, MAL), integrating attention mechanisms where appropriate.

### 2. Core Responsibilities

*   **Orchestration:** Manages the step-by-step execution of the cognitive cycle for processing new inputs.
*   **Service Integration:** Communicates with the Memory Core and Neural Memory Server APIs.
*   **Variant Management:** Selects and executes the logic for the active Titans variant (MAC, MAG, MAL, or NONE).
*   **History Management:** Maintains a sequential history of embeddings and projections (`SequenceContextManager`) needed for attention calculations in variants.
*   **Surprise Feedback:** Receives surprise metrics (loss, grad_norm) from the Neural Memory Server and initiates QuickRecal score updates in the Memory Core (**Note: Currently failing due to missing MC methods**).
*   **Context Propagation:** Ensures relevant information (embeddings, projections, metadata) is passed between stages.
*   **Error Handling:** Manages communication errors with downstream services.

### 3. Key Classes/Modules

*   `context_cascade_engine.ContextCascadeEngine`: The main orchestrating class.
*   `history.SequenceContextManager`: Manages the deque of historical context tuples.
*   `titans_variants.py`: Defines `TitansVariantType` enum, `TitansVariantBase`, `MACVariant`, `MAGVariant`, `MALVariant` classes, and the `create_titans_variant` factory.
*   `server.py`: Basic FastAPI application exposing the CCE (primarily `/process_memory`).

### 4. Configuration

*   Reads Memory Core URL (`MEMORY_CORE_URL`) and Neural Memory Server URL (`NEURAL_MEMORY_URL`) from environment variables or defaults.
*   Reads the active Titans variant (`TITANS_VARIANT`) from environment variables (defaults to `NONE`).
*   Configures `SequenceContextManager` length.
*   Retrieves dynamic configuration (e.g., attention parameters) from the Neural Memory Server via `/config` on initialization.

### 5. API Endpoints (Purpose)

The API (`orchestrator/server.py`) primarily exposes:
*   `POST /process_memory`: The main entry point that triggers the entire orchestrated cognitive cycle for a given input.
*   Potentially other passthrough endpoints (like `/get_sequence_embeddings`, `/analyze_surprise`).

*(See `API_REFERENCE.md` for detailed endpoint definitions)*

### 6. Internal Workflow (Refactored Cognitive Cycle)

The CCE's `process_new_input` method executes the following orchestrated steps:
1.  Call Memory Core (`/process_memory`) to store input and get `x_t`, `memory_id`.
2.  Call Neural Memory (`/get_projections`) to get `k_t`, `v_t`, `q_t`.
3.  If MAG/MAL active, execute variant pre-update logic (calculating gates or `v'_t`).
4.  Call Neural Memory (`/update_memory`) with `x_t` and any variant modifications (gates/projections). Receive `loss`, `grad_norm`.
5.  Call Memory Core (`/api/memories/update_quickrecal_score`) with surprise metrics to boost QuickRecal (**Currently Failing**).
6.  Call Neural Memory (`/retrieve`) with `x_t` to get raw associated embedding `y_t_raw` and the `q_t` used.
7.  If MAC active, execute variant post-retrieval logic to get final `y_t_final`. Otherwise `y_t_final = y_t_raw`.
8.  Store the full context `(timestamp, memory_id, x_t, k_t, v_t, q_t, y_t_final)` in `SequenceContextManager`.
9.  Return a consolidated response.

*(See Architecture Diagram in main ARCHITECTURE.md)*

### 7. Integration Points

*   **Calls Synthians Memory Core API:**
    *   `POST /process_memory` (Input: content, embedding, metadata; Output: memory_id, embedding, score, metadata)
    *   `POST /api/memories/update_quickrecal_score` (Input: memory_id, delta, reason; Output: status) (**Intended Call - Currently Failing**)
    *   `POST /api/memories/get_sequence_embeddings` (Passthrough - Input: filters; Output: sequence)
*   **Calls Neural Memory Server API:**
    *   `POST /get_projections` (Input: input_embedding; Output: k, v, q projections)
    *   `POST /update_memory` (Input: input_embedding, optional external k/v/gates; Output: loss, grad_norm, projections used, gates applied)
    *   `POST /retrieve` (Input: input_embedding; Output: retrieved_embedding, query_projection)
    *   `POST /calculate_gates` (MAG only - Input: attention_output; Output: alpha, theta, eta)
    *   `GET /config` (Input: None; Output: Configuration details)
    *   `POST /analyze_surprise` (Passthrough - Input: pred_emb, actual_emb; Output: surprise metrics)
*   **Internal Dependencies:** `SequenceContextManager`, `TitansVariantBase` subclasses, shared `GeometryManager`, `MetricsStore`.

### 8. Current Status & Known Gaps

*   **Status:** Implements the refactored cognitive flow correctly, enabling proper timing for MAC, MAG, and MAL variants. Integrates with `SequenceContextManager` for history. Dynamically configures itself. Uses lazy loading for TensorFlow.
*   **Gaps:**
    *   **Dependent on Memory Core Fix:** The critical surprise feedback loop to update QuickRecal scores is non-functional until the Memory Core API is fixed.
    *   **Error Handling:** While basic error handling exists, more sophisticated strategies for handling failures in Memory Core or Neural Memory calls (e.g., retries, fallback logic) could be added.
    *   **State Management:** If the CCE were to become stateful (beyond the sequence history), careful management would be needed. Currently designed as mostly stateless per request cycle.

---

## Inter-Component Integration Summary

*   **New Input:** User/System -> **CCE (`/process_memory`)** -> **Memory Core (`/process_memory`)** -> Returns `x_t`, `mem_id` to CCE.
*   **Association Learning:** CCE -> **Neural Memory (`/get_projections`)** -> Returns `k_t, v_t, q_t` -> CCE -> **Variant Pre-Update (MAG/MAL)** -> CCE -> **Neural Memory (`/update_memory`)** -> Returns `loss`, `grad_norm`.
*   **Surprise Feedback:** CCE -> **Memory Core (`/update_quickrecal_score`)** with `loss`/`grad_norm` -> Memory Core updates score (**BROKEN**).
*   **Associative Retrieval:** CCE -> **Neural Memory (`/retrieve`)** -> Returns `y_t_raw`, `q_t` -> CCE -> **Variant Post-Update (MAC)** -> Generates `y_t_final`.
*   **History:** CCE updates `SequenceContextManager` with `(ts, mem_id, x_t, k_t, v_t, q_t, y_t_final)`.
*   **Configuration:** CCE -> **Neural Memory (`/config`)** -> Returns NM/Attention config.

This flow highlights the central role of the CCE in mediating all interactions and implementing the core logic of the bi-hemispheric model and its variants. The broken feedback link to the Memory Core is the most significant integration issue to resolve.
```