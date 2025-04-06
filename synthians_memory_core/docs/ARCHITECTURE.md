# Synthians Cognitive Architecture Overview

*(Existing Introduction - Keep as is)*

## System Principles

*   Memory is weighted (QuickRecal).
*   Emotion shapes recall (Emotional Gating).
*   Surprise signals significance (NM → QR Boost).
*   Ideas cluster and connect (Assemblies).
*   **Assemblies have traceable history (`merged_from`).**
*   Presence emerges from adaptive memory (NM Learning + Variant Selection).
*   **Cognition should be explainable (Phase 5.9: Implemented Backend Layer).**

## High-Level Components (Post Phase 5.9 Backend)

1.  **Synthians Memory Core (MC):** The stable archive & introspection hub. Manages `MemoryEntry` and `MemoryAssembly` persistence, vector indexing (FAISS), QuickRecall scoring, emotional context. **Includes Explainability & Diagnostics backend logic and APIs (Phase 5.9).**
2.  **Neural Memory Server (NM):** The adaptive associator. Implements test-time learning (Titans-based) on embedding sequences, calculates surprise metrics. Configuration exposed via MC API (`/config/runtime/neural-memory`).
3.  **Context Cascade Engine (CCE):** The orchestrator. Manages MC<->NM flow, dynamic variant selection, LLM guidance integration. Reports detailed metrics. Configuration exposed via MC API (`/config/runtime/cce`).
4.  **Explainability & Diagnostics Backend (Phase 5.9 - Integrated into MC):** Internal logic (`explainability/`) and persistent logs (`metrics/`) providing data for API endpoints that reveal system behavior (merge events, activations, lineage, config, stats).
5.  **Synthians Cognitive Dashboard (Planned for Phase 5.9.1 - Frontend):** The visualization and interaction layer. Consumes APIs from MC, NM, CCE to display system state, metrics, and explanations. *(Specification in `docs/guides/DASHBOARD_SPECIFICATION.md`)*

## Architecture Diagram (Conceptual - Post Phase 5.9 Backend)

```mermaid
graph LR
    subgraph "User/External Interface"
        A[Dashboard (Phase 5.9.1+)]
        B[Other Client Apps]
    end

    subgraph "API Layer (Memory Core - FastAPI)"
        API[MC API Server]
        CoreAPI[Core Memory Endpoints\n(/process_memory, /retrieve)]
        AssemblyAPI[Assembly Endpoints\n(/assemblies)]
        ExplainAPI[Explainability Endpoints\n(/explain/*, /lineage)]
        DiagAPI[Diagnostics Endpoints\n(/diagnostics/*, /config/*)]
    end

    subgraph "Synthians Memory Core (Backend Logic & State)"
        MC[SynthiansMemoryCore Class]
         subgraph "Phase 5.9 Components"
            ExplainLogic[Explainability Module]
            MetricsLogic[Metrics Module\n(MergeTracker, ActivationStats)]
        end
        Persistence[Persistence Layer\n(Filesystem: JSON, FAISS .bin, .jsonl)]
        VectorIndex[Vector Index (FAISS)]
        RetryQueue[Vector Update Retry Queue]
        OtherCore[Other MC Components\n(GeoMgr, QuickRecal, Emo)]
    end

    subgraph "Orchestrator Service"
        CCE[Context Cascade Engine]
    end

    subgraph "Neural Memory Service"
        NM[Neural Memory Server]
    end

    subgraph "External Services"
        LLM[LLM Guidance (LM Studio)]
    end

    %% Connections
    A -- HTTP API --> API
    B -- HTTP API --> API

    API -- Calls --> CoreAPI
    API -- Calls --> AssemblyAPI
    API -- Calls --> ExplainAPI
    API -- Calls --> DiagAPI

    CoreAPI -- Uses --> MC
    AssemblyAPI -- Uses --> MC
    ExplainAPI -- Uses --> ExplainLogic
    DiagAPI -- Uses --> MetricsLogic & MC(Config)

    MC -- Uses --> Persistence
    MC -- Uses --> VectorIndex
    MC -- Uses --> RetryQueue
    MC -- Uses --> OtherCore
    MC -- Uses --> ExplainLogic
    MC -- Uses --> MetricsLogic

    MC -- Orchestrates --> CCE
    CCE -- Manages --> NM
    CCE -- Calls --> LLM

    ExplainLogic -- Reads --> Persistence
    ExplainLogic -- Reads --> MetricsLogic
    ExplainLogic -- Uses --> OtherCore(GeometryManager)

    MetricsLogic -- Writes --> Persistence(Logs/Stats Files)
    DiagAPI -- Reads --> Persistence

    %% Dashboard Backend Proxy (Implicit Layer between Dashboard and APIs)
    %% Dashboard Backend calls MC, NM, CCE APIs based on frontend requests
```

*(Existing sections on Bi-Hemispheric Model, Cognitive Cycle - Review for consistency, ensure cycle description mentions logging points for merge/activation)*

### Phase 5.9: Introspection Layer (Backend Implemented)

Phase 5.9 implemented the backend foundation for system introspection within the Memory Core:
*   **Event Capture:** `MergeTracker` logs assembly merge creation and cleanup status events to `merge_log.jsonl`. Activation counts are tracked in `_assembly_activation_counts` and persisted.
*   **Lineage Tracking:** `MemoryAssembly.merged_from` field is populated during merges and used by `explainability.lineage.trace_lineage`.
*   **Explanation Logic:** Functions within the `explainability/` module generate explanations for activation, merges, and lineage based on persisted data, logs, and runtime state.
*   **Diagnostic APIs:** Endpoints (`/diagnostics/*`, `/config/*`, updated `/stats`) expose merge history, sanitized configuration, and activation statistics. Controlled by `ENABLE_EXPLAINABILITY` flag.

*(Existing sections on Embedding Handling, Stateless Design - Keep as is)*

## Memory Assemblies (Post Phase 5.9 Backend)

Memory Assemblies (`MemoryAssembly`) represent dynamically formed groups of related `MemoryEntry` objects.
*   They possess a `composite_embedding` (semantic center).
*   Stability improved with `vector_index_updated_at` timestamp tracking synchronization with the vector index. Only synchronized assemblies contribute to retrieval boosting. Failed index updates are queued via `_pending_vector_updates` for retry.
*   **Phase 5.9:** The `merged_from: List[str]` field tracks merge ancestry. Merge events are logged persistently via `MergeTracker` to `merge_log.jsonl`. Explainability APIs allow inspection of merges and lineage.

*(Existing sections on Implementation Guidelines - Keep as is)*