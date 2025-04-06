# Synthians Cognitive Architecture: Component Guide

The Synthians Cognitive Architecture consists of several core components that work together to emulate human-like memory and reasoning. This guide provides details on each component, their roles, and integration points.

## Core Components (Post Phase 5.9 Backend)

### 1. Synthians Memory Core (`synthians_memory_core`)

*   **Role:** Persistent, indexed, context-aware memory storage and retrieval. Manages `MemoryEntry` and `MemoryAssembly`. **Includes backend logic for Explainability & Diagnostics (Phase 5.9).**
*   **Key Modules:**
    *   `SynthiansMemoryCore`: Main orchestrator. Manages components, API calls, background tasks (persistence, decay, **vector update retry**). Integrates assembly boosting and **Phase 5.9 features**.
    *   `memory_structures`: `MemoryEntry`, `MemoryAssembly` (including `merged_from`, `vector_index_updated_at`).
    *   `memory_persistence`: Async save/load for entries and assemblies to JSON files. Manages `memory_index.json`.
    *   `vector_index`: FAISS `IndexIDMap` wrapper with async operations, persistence, validation, diagnostics, and repair capabilities. **Note:** IDMap add/remove ops are CPU-bound.
    *   `geometry_manager`: Vector math and alignment (Euclidean, Hyperbolic).
    *   `hpc_quickrecal`: Relevance scoring (`UnifiedQuickRecallCalculator`).
    *   `emotional_intelligence`: `EmotionAnalyzer` and `EmotionalGatingService`.
    *   `metadata_synthesizer`: Enriches `MemoryEntry` metadata.
    *   `adaptive_components`: `ThresholdCalibrator` for retrieval feedback.
    *   **`explainability/` (Phase 5.9):** Contains `activation.py`, `merge.py`, `lineage.py` for generating explanations. Uses Persistence, GeometryManager, MergeTracker.
    *   **`metrics/` (Phase 5.9):** Contains `merge_tracker.py` (`MergeTracker` class managing `merge_log.jsonl`). Activation stats logic integrated into `SynthiansMemoryCore`.
*   **API:** Exposed via `api/server.py`. Includes core memory, assembly management, **explainability (`/explain_*`, `/lineage`)**, and **diagnostics (`/diagnostics/*`, `/config/*`)** endpoints. Controlled by `ENABLE_EXPLAINABILITY` flag.
*   **Integration:** Primary data store; receives boost updates from CCE; provides sequences/data to CCE/NM; provides diagnostic/explanation data to Dashboard Proxy.
*   **See:** `docs/core/README.md`

### 2. Neural Memory Server (`synthians_trainer_server`)

*   **Role:** Adaptive associative sequence memory (Titans-based). Test-time learning. Surprise calculation.
*   **Key Modules:** `neural_memory`, `http_server`, `metrics_store`, `surprise_detector`.
*   **API:** Provides endpoints for update, retrieval, projections, gate calculation, config, diagnostics (`/diagnose_emoloop`).
*   **Phase 5.9:** Runtime configuration exposed via MC API proxy (`/config/runtime/neural-memory`). Provides diagnostic data (`/diagnose_emoloop`) to Dashboard Proxy.
*   **Integration:** Receives embeddings from CCE; returns loss/grad_norm (surprise) and retrieved embeddings to CCE.
*   **See:** `docs/trainer/README.md`

### 3. Context Cascade Engine (CCE) (`orchestrator`)

*   **Role:** Orchestrates MC <-> NM flow, implements cognitive cycle, dynamic variant selection, LLM guidance integration.
*   **Key Modules:** `context_cascade_engine`, `titans_variants`, `history`, `variant_selector`, `memory_logic_proxy`.
*   **API:** Exposes `/process_memory`; reports enhanced metrics via `/metrics/recent_cce_responses`. Runtime config exposed via MC API proxy (`/config/runtime/cce`).
*   **Phase 5.9:** Response structure for `/metrics/recent_cce_responses` includes detailed variant selection reasons and LLM usage info.
*   **Integration:** Calls MC and NM APIs; receives input, sends back final response. Provides CCE metrics to Dashboard Proxy.
*   **See:** `docs/orchestrator/README.md`

### 4. Explainability & Diagnostics Backend (Integrated into Memory Core)

*   **Role:** Provides the data and logic foundation for system introspection.
*   **Key Modules:**
    *   `explainability/`: Python functions generating explanations.
    *   `metrics/`: `MergeTracker` (writes `merge_log.jsonl`). Activation stats logic in `SynthiansMemoryCore`.
*   **Integration:** Called by MC API endpoints (`/explain_*`, `/diagnostics/*`). Reads data from persistence and logs. Core MC logic (`_execute_merge`) calls `MergeTracker`.

### 5. Synthians Cognitive Dashboard (Planned for Phase 5.9.1 - Frontend)

*   **Role:** User interface for monitoring, inspecting, and understanding the Synthians cognitive architecture.
*   **Technology:** React/Vite, TypeScript, TailwindCSS/Shadcn UI, Recharts, TanStack Query.
*   **Functionality (Planned):** Display service statuses, core metrics, assembly lists/details, explanations, logs, runtime config, chat interface (placeholder), admin actions (placeholder).
*   **Integration (Planned):** Consumes APIs exposed by the **dashboard's own backend proxy server**, which in turn calls the MC, NM, and CCE APIs.
*   **See:** `docs/guides/DASHBOARD_SPECIFICATION.md`

## Shared Utilities & Tools

### 1. Embedding Utilities (`geometry_manager.py`, `utils/embedding_validators.py`)
*   Provide consistent embedding handling, validation (NaN/Inf, dimension), alignment, and transformation.

### 2. Diagnostic Tools
*   `tools/variant_diagnostics_dashboard.py`: Real-time CLI dashboard for CCE variant metrics.
*   `tools/lucidia_think_trace.py`: Tracing and visualizing cognitive process flow.
*   `tools/rebuild_vector_index.py`: Rebuilding FAISS indices from persistence.
*   `tools/repair_index.py`: (Likely deprecated/merged) Diagnosing and fixing index inconsistencies (now mostly handled by `vector_index.py` repair methods).

### 3. Testing Framework
*   Comprehensive test suite (`tests/`) using `pytest` and `pytest-asyncio`.
*   Includes unit, integration, API, and stress tests.
*   See `docs/testing/` for details.