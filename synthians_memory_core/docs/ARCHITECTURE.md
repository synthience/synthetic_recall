# Synthians Cognitive Architecture Overview

*(Existing Introduction - Keep as is)*

## System Principles

*   Memory is weighted (QuickRecal).
*   Emotion shapes recall (Emotional Gating).
*   Surprise signals significance (NM → QR Boost).
*   Ideas cluster and connect (Assemblies).
*   Presence emerges from adaptive memory (NM Learning + Variant Selection).
*   **Cognition should be explainable (Planned for Phase 5.9: Introspection Layer).**

## High-Level Components (Current & Planned)

1.  **Synthians Memory Core (MC):** The stable archive. Manages `MemoryEntry` and `MemoryAssembly` persistence, vector indexing (FAISS), QuickRecall scoring, emotional context. **Planned for Phase 5.9: Expose diagnostic and explainability data via new APIs.**
2.  **Neural Memory Server (NM):** The adaptive associator. Implements test-time learning (Titans-based) on embedding sequences, calculates surprise metrics. Configuration planned to be exposed via MC API (`/config/runtime/neural-memory`).
3.  **Context Cascade Engine (CCE):** The orchestrator. Manages MC<->NM flow, dynamic variant selection, LLM guidance integration. **Planned for Phase 5.9: Enhanced metrics reporting (`/metrics/recent_cce_responses`) for dashboard consumption.** Configuration may be exposed via MC API (`/config/runtime/cce`).
4.  **Explainability & Diagnostics Backend (Planned New Layer - Part of MC):** Internal logic (`explainability/`) and persistent logs (`metrics/`) providing data for API endpoints that reveal system behavior (merge events, activations, lineage).
5.  **Synthians Cognitive Dashboard (Planned for Phase 5.9.1 - Frontend):** The visualization and interaction layer. Will consume APIs from MC, NM, CCE to display system state, metrics, and explanations. *(Specification in `docs/guides/DASHBOARD_SPECIFICATION.md`)*

## Architecture Diagram (Conceptual - Including Planned Features)

```mermaid
graph LR
    subgraph "User/External Interface"
        Dashboard[Cognitive Dashboard (Phase 5.9.1+)]
        ClientApp[Other Client Apps]
    end

    subgraph "API Layer (Memory Core - FastAPI)"
        API[MC API Server]
        ExplainAPI[Explainability Endpoints\n(/explain/*) - Planned]
        DiagAPI[Diagnostics Endpoints\n(/diagnostics/*, /config/*) - Planned]
        CoreAPI[Core Memory Endpoints\n(/process_memory, /retrieve)]
        AssemblyAPI[Assembly Endpoints\n(/assemblies)]
    end

    subgraph "Synthians Memory Core (Backend Logic)"
        MC[SynthiansMemoryCore Class]
        ExplainLogic[Explainability Module - Planned]
        MetricsLogic[Metrics Module\n(MergeTracker, ActivationStats) - Planned]
        Persistence[Persistence Layer]
        VectorIndex[Vector Index (FAISS)]
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
    Dashboard -- HTTP API --> API
    ClientApp -- HTTP API --> API

    API -- Calls --> ExplainAPI
    API -- Calls --> DiagAPI
    API -- Calls --> CoreAPI
    API -- Calls --> AssemblyAPI

    ExplainAPI -- Uses --> ExplainLogic
    DiagAPI -- Uses --> MetricsLogic
    CoreAPI -- Uses --> MC
    AssemblyAPI -- Uses --> MC

    MC -- Uses --> ExplainLogic
    MC -- Uses --> MetricsLogic
    MC -- Uses --> Persistence
    MC -- Uses --> VectorIndex
    MC -- Uses --> OtherCore

    MC -- Orchestrates --> CCE
    CCE -- Manages --> NM
    CCE -- Calls --> LLM

    %% Data Flow for Explanations (Planned)
    ExplainLogic -- Reads --> Persistence
    ExplainLogic -- Reads --> MetricsLogic
    ExplainLogic -- Uses --> OtherCore(GeometryManager)

    %% Data Flow for Diagnostics (Planned)
    MetricsLogic -- Writes --> Filesystem(Logs/Stats Files)
    DiagAPI -- Reads --> Filesystem
    DiagAPI -- Reads --> MC(Config)

    %% Dashboard Backend Proxy (Implicit Layer between Dashboard and APIs)
    %% Dashboard Backend calls MC, NM, CCE APIs based on frontend requests
```

*(Existing sections on Bi-Hemispheric Model, Cognitive Cycle - Review for consistency, ensure cycle description mentions logging points for merge/activation)*

### Planned for Phase 5.9: Introspection Layer

Phase 5.9 will introduce a dedicated backend layer within the Memory Core focused on making the system's internal operations observable and explainable. This will involve:
*   **Capturing Key Events:** Logging significant events like assembly merges with relevant context (similarity scores, source IDs) via `MergeTracker`.
*   **Persisting Lineage:** Storing the `merged_from` history directly within `MemoryAssembly` objects (already implemented in the data structure).
*   **Calculating Explanations:** Implementing logic within the `explainability/` module to re-evaluate or retrieve data that explains *why* an activation occurred (similarity vs. threshold) or *how* a merge happened (source assemblies, similarity).
*   **Exposing Diagnostics:** Providing API endpoints to query merge history (`/diagnostics/merge_log`), runtime configuration (`/config/runtime/*`), and basic performance metrics (like assembly activation in `/stats`).
*   **Enabling Dashboard Integration:** Creating the necessary API surface for a dedicated monitoring and visualization dashboard.

*(Existing sections on Embedding Handling, Stateless Design - Keep as is)*

## Memory Assemblies (Current & Planned)

Memory Assemblies (`MemoryAssembly`) represent dynamically formed groups of related `MemoryEntry` objects.
*   They possess a `composite_embedding` (semantic center).
*   **Phase 5.8 (Current):** Stability improved with `vector_index_updated_at` timestamp tracking synchronization with the vector index. Only synchronized assemblies contribute to retrieval boosting. Failed index updates are queued for retry.
*   **Phase 5.9 (Planned):** Will add full utilization of the existing `merged_from: List[str]` field to track merge ancestry. Merge events will be logged persistently via `MergeTracker` to `merge_log.jsonl`.

*(Existing sections on Implementation Guidelines - Keep as is)*