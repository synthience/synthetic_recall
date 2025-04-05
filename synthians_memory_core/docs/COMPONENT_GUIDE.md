# Synthians Cognitive Architecture: Component Guide

The Synthians Cognitive Architecture consists of several core components that work together to emulate human-like memory and reasoning. This guide provides details on each component, their roles, and integration points.

## Core Components (Current & Planned)

### 1. Synthians Memory Core (`synthians_memory_core`)

*   **Role:** Persistent, indexed, context-aware memory storage and retrieval. Manages `MemoryEntry` and `MemoryAssembly`.
*   **Key Modules:**
    *   `SynthiansMemoryCore`: Main orchestrator.
    *   `memory_structures`: `MemoryEntry`, `MemoryAssembly` (including `merged_from` field).
    *   `memory_persistence`: Async save/load for entries and assemblies.
    *   `vector_index`: FAISS `IndexIDMap` wrapper.
    *   `geometry_manager`: Vector math and alignment.
    *   `hpc_quickrecal`: Relevance scoring.
    *   `emotional_intelligence`: Emotion analysis and gating.
    *   `metadata_synthesizer`: Metadata enrichment.
    *   `adaptive_components`: Threshold calibration.
    *   **`explainability/` (Planned for 5.9): Core logic for explaining activations, merges, lineage.**
    *   **`metrics/` (Planned for 5.9): Tracking for merge events (`MergeTracker`), activation stats.**
*   **API:** Exposed via `api/server.py`. Includes core memory, assembly management. **Planned for Phase 5.9: explainability (`/explain_*`) and diagnostics (`/diagnostics/*`, `/config/*`) endpoints**.
*   **Current & Planned:** Currently supports basic memory operations and assembly management. Phase 5.9 will add backend logic and APIs for explaining assembly activation/merges, tracking merge history, exposing sanitized runtime config, and tracking basic activation stats with an `ENABLE_EXPLAINABILITY` flag.
*   **Integration:** Primary data store; receives boost updates from CCE; provides sequences/data to CCE/NM; **will provide diagnostic/explanation data to Dashboard via new APIs (planned)**.
*   **See:** `docs/core/README.md`

### 2. Neural Memory Server (`synthians_trainer_server`)

*   **Role:** Adaptive associative sequence memory (Titans-based). Test-time learning. Surprise calculation.
*   **Key Modules:** `neural_memory`, `http_server`, `metrics_store`, `surprise_detector`.
*   **API:** Provides endpoints for update, retrieval, projections, gate calculation, config, diagnostics (`/diagnose_emoloop`).
*   **Planned for Phase 5.9:** Runtime configuration may be exposed via MC API proxy (`/config/runtime/neural-memory`). No major internal changes expected.
*   **Integration:** Receives embeddings from CCE; returns loss/grad_norm (surprise) and retrieved embeddings to CCE. **Will provide diagnostic data (`/diagnose_emoloop`) to Dashboard via its API.**
*   **See:** `docs/trainer/README.md`

### 3. Context Cascade Engine (CCE) (`orchestrator`)

*   **Role:** Orchestrates MC <-> NM flow, implements cognitive cycle, dynamic variant selection, LLM guidance integration.
*   **Key Modules:** `context_cascade_engine`, `titans_variants`, `history`, `variant_selector`, `memory_logic_proxy`.
*   **API:** Exposes `/process_memory`; may proxy others. **Planned enhancement to metrics via `/metrics/recent_cce_responses`.** Runtime config may be exposed via MC API proxy (`/config/runtime/cce`).
*   **Planned for Phase 5.9:** Response structure for `/metrics/recent_cce_responses` will be enhanced to include detailed variant selection reasons and LLM usage info.
*   **Integration:** Calls MC and NM APIs; receives input, sends back final response. **Will provide CCE metrics (`/metrics/recent_cce_responses`) to Dashboard via its API.**
*   **See:** `docs/orchestrator/README.md`

### 4. Explainability & Diagnostics Backend (Planned for MC)

*   **Role:** Will provide the data and logic foundation for system introspection. Not a separate service, but distinct modules within the Memory Core.
*   **Key Modules (Planned):**
    *   `explainability/`: Will contain Python functions to generate explanations (activation, merge, lineage). Uses `GeometryManager`, `MemoryPersistence`.
    *   `metrics/`: Will contain `MergeTracker` (writes `merge_log.jsonl`) and logic for persisting activation stats.
*   **Integration (Planned):** Will be called by MC API endpoints (`/explain_*`, `/diagnostics/*`). Will read data from persistence and logs. Core MC logic (`_execute_merge`) will call `MergeTracker`.

### 5. Synthians Cognitive Dashboard (Planned for Phase 5.9.1 - Frontend)

*   **Role:** User interface for monitoring, inspecting, and understanding the Synthians cognitive architecture.
*   **Technology:** React/Vite, TypeScript, TailwindCSS/Shadcn UI, Recharts, TanStack Query.
*   **Functionality (Planned):** Display service statuses, core metrics, assembly lists/details, explanations, logs, runtime config, chat interface (placeholder), admin actions (placeholder).
*   **Integration (Planned):** Will consume APIs exposed by the **dashboard's own backend proxy server**, which in turn calls the MC, NM, and CCE APIs.
*   **See:** `docs/guides/DASHBOARD_SPECIFICATION.md`

## Explainability & Diagnostics Layer (Phase 5.9 - Part of Memory Core)

This layer provides introspection capabilities, enabled via the `ENABLE_EXPLAINABILITY` flag.

### Explainability Module (`synthians_memory_core/explainability/`)

*   **Role:** Provides functions to generate human-understandable explanations for specific Memory Core decisions.
*   **Key Modules:**
    *   `activation.py`: Contains `generate_activation_explanation` logic.
    *   `merge.py`: Contains `generate_merge_explanation` logic.
    *   `lineage.py`: Contains `trace_lineage` logic (incl. cycle/depth handling).
    *   `_explain_helpers.py`: Shared utilities (e.g., `safe_load_assembly`, `calculate_similarity`).
*   **Integration:** Accessed via `/explain_*` and `/lineage` API endpoints. Requires `MemoryPersistence`, `GeometryManager`, `MergeTracker`, and core `config` as inputs.
*   **See:** `docs/core/explainability.md`

### Metrics Module (`synthians_memory_core/metrics/`)

*   **Role:** Tracks and persists key system events and statistics for diagnostic purposes.
*   **Key Modules:**
    *   `merge_tracker.py`: Implements `MergeTracker` class. Manages the append-only `merge_log.jsonl`, handling event logging (creation, cleanup status) and log rotation. Provides methods for querying and reconciling log entries.
    *   **(Implicit):** Logic within `SynthiansMemoryCore` for tracking assembly activation counts (`_assembly_activation_counts`) and periodically persisting them to `stats/assembly_activation_stats.json` via `_persist_activation_stats`.
*   **Integration:** `MergeTracker` is called by `SynthiansMemoryCore` during merge operations. Activation stats are tracked internally. Data is exposed via `/diagnostics/merge_log` and `/stats` API endpoints.
*   **See:** `docs/core/diagnostics.md`

### Diagnostics & Explainability API Routes (`synthians_memory_core/api/`)

*   **Role:** Expose the underlying explainability and diagnostics functions via secure, well-defined REST endpoints.
*   **Key Modules:**
    *   `explainability_routes.py`: Defines `/assemblies/{id}/explain_*` and `/assemblies/{id}/lineage` routes. Handles request validation, dependency injection (getting core components), calling the core logic, and formatting responses according to Pydantic models. Includes API-level caching for `/lineage`.
    *   `diagnostics_routes.py`: Defines `/diagnostics/merge_log` and `/config/runtime/{service_name}` routes. Handles request validation, calls `MergeTracker` for reconciled logs, implements configuration allow-listing logic.
*   **Integration:** Included by the main `api/server.py` conditionally based on `ENABLE_EXPLAINABILITY`. Relies on FastAPI's dependency injection to get access to `request.app.state.memory_core`.

## Shared Utilities & Tools

### 1. Embedding Utilities (`geometry_manager.py`, `embedding_validators.py`)
*   Provide consistent embedding handling, validation, and transformation.
*   Support for different embedding models and dimensions.
*   Normalization and geometry-specific operations.

### 2. Diagnostic Tools
*   `variant_diagnostics_dashboard.py`: Real-time CLI dashboard for CCE variant metrics.
*   `lucidia_think_trace.py`: Tracing and visualizing cognitive process flow.
*   `rebuild_vector_index.py`: Rebuilding faulty FAISS indices.
*   `repair_index.py`: Diagnosing and fixing index inconsistencies.

### 3. Testing Framework
*   Comprehensive test suite for all system components.
*   Integration tests for end-to-end cognitive flows.
*   Performance and stress tests for components.
*   See `docs/testing/` for details.